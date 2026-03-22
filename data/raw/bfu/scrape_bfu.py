#!/usr/bin/env python3
"""
Skatteetaten BFU Scraper
========================
Scrapes bindende forhåndsuttalelser (binding advance rulings) from skatteetaten.no
and outputs structured JSONL files compatible with NOR-CASEHOLD.

Usage:
    pip install requests beautifulsoup4 lxml
    python scrape_bfu.py --output data/raw/bfu/

Phase 1: Discover all BFU URLs via the site's internal API
Phase 2: Fetch each page, extract summary + full text
Phase 3: Output as JSONL
"""

import requests
import json
import os
import re
import time
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.skatteetaten.no"
BFU_LIST_URL = "https://www.skatteetaten.no/rettskilder/type/uttalelser/bfu/"

# Skatteetaten uses an internal search/filter API for the listing page
# We'll try multiple discovery methods
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) NOR-CASEHOLD-Research/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "nb-NO,nb;q=0.9,no;q=0.8,en;q=0.7",
}

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests


def discover_bfu_urls_via_api():
    """Try to discover BFU URLs via Skatteetaten's internal API."""
    print("Attempting API-based URL discovery...")
    
    # Skatteetaten's rettskilder section typically uses an API endpoint
    # Try common patterns
    api_urls = [
        "https://www.skatteetaten.no/api/rettskilder/?type=bfu&pageSize=500",
        "https://www.skatteetaten.no/api/rettskilder/search?type=bfu&pageSize=500",
        "https://www.skatteetaten.no/rettskilder/api/search?type=bfu&take=500",
        "https://www.skatteetaten.no/api/search?contentType=bfu&take=500",
    ]
    
    for url in api_urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  Found API at: {url}")
                print(f"  Response keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                return data
        except Exception as e:
            continue
    
    print("  No API endpoint found.")
    return None


def discover_bfu_urls_via_sitemap():
    """Try to find BFU URLs in the sitemap."""
    print("Attempting sitemap-based URL discovery...")
    
    sitemap_urls = [
        "https://www.skatteetaten.no/sitemap.xml",
        "https://www.skatteetaten.no/sitemap_index.xml",
        "https://www.skatteetaten.no/robots.txt",
    ]
    
    for url in sitemap_urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                print(f"  Found: {url}")
                # Parse sitemap for BFU URLs
                if url.endswith('.xml'):
                    soup = BeautifulSoup(resp.text, 'lxml-xml')
                    locs = soup.find_all('loc')
                    bfu_urls = [
                        loc.text for loc in locs 
                        if '/rettskilder/type/uttalelser/bfu/' in loc.text
                        and loc.text.rstrip('/') != 'https://www.skatteetaten.no/rettskilder/type/uttalelser/bfu'
                    ]
                    if bfu_urls:
                        print(f"  Found {len(bfu_urls)} BFU URLs in sitemap")
                        return bfu_urls
                elif 'robots.txt' in url:
                    print(f"  robots.txt content (first 500 chars):")
                    print(f"  {resp.text[:500]}")
                    # Look for sitemap references
                    for line in resp.text.split('\n'):
                        if 'sitemap' in line.lower():
                            print(f"  Sitemap reference: {line}")
        except Exception as e:
            continue
    
    print("  No sitemap found.")
    return None


def discover_bfu_urls_via_crawl():
    """Crawl the BFU listing page and extract links."""
    print("Attempting crawl-based URL discovery...")
    
    # Try both Norwegian and English URL variants
    list_urls = [
        "https://www.skatteetaten.no/rettskilder/type/uttalelser/bfu/",
        "https://www.skatteetaten.no/en/rettskilder/type/uttalelser/bfu/",
    ]
    
    all_urls = set()
    
    for list_url in list_urls:
        try:
            resp = requests.get(list_url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Find all links that point to individual BFU pages
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    full_url = urljoin(BASE_URL, href)
                    if '/rettskilder/type/uttalelser/bfu/' in full_url:
                        # Skip the listing page itself and the search page
                        if full_url.rstrip('/').endswith('/bfu') or '/sok' in full_url:
                            continue
                        all_urls.add(full_url)
                
                print(f"  Found {len(all_urls)} URLs from {list_url}")
                
                # Also look for any JavaScript data or API calls in the page source
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'rettskilder' in (script.string or ''):
                        # Look for API endpoints or data URLs
                        api_matches = re.findall(r'["\'](/api/[^"\']+)["\']', script.string)
                        for match in api_matches:
                            print(f"  Found API reference in JS: {match}")
                
        except Exception as e:
            print(f"  Error crawling {list_url}: {e}")
    
    return list(all_urls) if all_urls else None


def discover_bfu_urls_via_google():
    """
    Use known BFU URL patterns and serial numbers to discover pages.
    BFUs are numbered: BFU 1/2024, BFU 2/2024, etc.
    """
    print("Attempting pattern-based URL discovery...")
    print("  Trying known BFU slug patterns from search results...")
    
    # Known BFU slugs from our earlier search results
    known_slugs = [
        "syntetiske-aksjer",
        "bindende-forhandsuttalelse-fra-skattedirektoratet",
        "bindende-forhandsuttalelse---aksjeutbytte",
        "klage-over-bindende-forhandsuttalelse",
        "delvis-bindende-forhandsuttalelse-og-delvis-veiledende-uttalelse--sporsmal-om-merverdiavgiftsloven--2-2-og--3-10",
        "bindende-forhandsuttalelse--merverdiavgiftsmessige-konsekvenser-ved-dropdown-fisjon",
        "sporsmal-om-en-kommunes-rett-til-kompensasjon-for-merverdiavgiftskostnader-som-padras-ved-etablering-av-en-golfbane-jf.-kompensasjonsloven--4-forste-ledd",
        "sporsmal-om-kommunens-og-statens-overtakelse-av-infrastruktur-medforer-en-justeringsforpliktelse-for-utbygger",
    ]
    
    urls = []
    for slug in known_slugs:
        # Try both Norwegian and English URL variants
        urls.append(f"https://www.skatteetaten.no/rettskilder/type/uttalelser/bfu/{slug}/")
    
    print(f"  Have {len(urls)} known URLs as seed. Need to discover more via crawling.")
    return urls


def fetch_page(url, retries=3):
    """Fetch a page with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                return None
            else:
                print(f"  HTTP {resp.status_code} for {url}, retry {attempt+1}")
        except Exception as e:
            print(f"  Error fetching {url}: {e}, retry {attempt+1}")
    return None


def extract_bfu_data(html, url):
    """
    Extract structured data from a BFU page.
    
    Structure:
    - Title: <h1> tag
    - Metadata: published date, serial number
    - Summary: paragraphs before "Innsenders fremstilling" or the main analysis
    - Full text: everything after the summary
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract title
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else ""
    
    if not title:
        return None
    
    # Find the main content area
    # Skatteetaten typically uses an article or main content div
    content = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|article|body'))
    
    if not content:
        # Fallback: use the whole body
        content = soup.find('body')
    
    if not content:
        return None
    
    # Extract metadata
    metadata = {}
    
    # Look for date
    date_patterns = [
        r'Avgitt\s+(\d{1,2}\.\s*\w+\s+\d{4})',
        r'Published:\s*(\d{1,2}\s+\w+\s+\d{4})',
        r'Avgitt\s+(\d{1,2}\s+\w+\s+\d{4})',
    ]
    
    page_text = content.get_text()
    for pattern in date_patterns:
        match = re.search(pattern, page_text)
        if match:
            metadata['date'] = match.group(1)
            break
    
    # Look for serial number (e.g., "BFU 9/2024" or "Whole serial number 9/2024")
    serial_match = re.search(r'(?:BFU|serial number)\s*(\d+/\d{4})', page_text, re.IGNORECASE)
    if serial_match:
        metadata['serial_number'] = f"BFU {serial_match.group(1)}"
    
    # Extract all text blocks in order
    all_elements = content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])
    
    # Find the split point between summary and full text
    # Common markers for where the full analysis begins:
    split_markers = [
        "innsenders fremstilling",
        "fra anmodningen",
        "anmodningen gjengis",
        "vi viser til",
        "sakens bakgrunn",
        "saken gjelder",
        "faktum",
    ]
    
    paragraphs = []
    for el in all_elements:
        text = el.get_text(strip=True)
        if text and len(text) > 5:
            # Skip navigation/header elements
            if el.name == 'h1':
                continue
            paragraphs.append({
                'text': text,
                'tag': el.name,
                'is_heading': el.name in ['h2', 'h3', 'h4', 'h5']
            })
    
    if not paragraphs:
        return None
    
    # Find split point
    split_idx = None
    for i, para in enumerate(paragraphs):
        text_lower = para['text'].lower()
        # Skip the first few paragraphs (they're likely summary/intro)
        if i < 2:
            continue
        for marker in split_markers:
            if marker in text_lower:
                # Check if this is a heading or the start of a new section
                if para['is_heading'] or text_lower.startswith(marker):
                    split_idx = i
                    break
        if split_idx is not None:
            break
    
    # If no clear split found, use a heuristic:
    # First 2-3 paragraphs are summary, rest is full text
    if split_idx is None:
        # Look for the first heading after initial paragraphs
        for i, para in enumerate(paragraphs):
            if i >= 2 and para['is_heading']:
                split_idx = i
                break
    
    if split_idx is None:
        # Default: first 3 paragraphs as summary
        split_idx = min(3, len(paragraphs))
    
    # Build summary and full text
    summary_parts = [p['text'] for p in paragraphs[:split_idx] if not p['is_heading']]
    full_text_parts = [p['text'] for p in paragraphs[split_idx:]]
    
    sammendrag = ' '.join(summary_parts).strip()
    full_text = ' '.join(full_text_parts).strip()
    
    # Quality checks
    if len(sammendrag) < 50:
        print(f"  Warning: Very short summary ({len(sammendrag)} chars) for {url}")
        return None
    
    if len(full_text) < 200:
        print(f"  Warning: Very short full text ({len(full_text)} chars) for {url}")
        return None
    
    # Build the slug-based ID
    slug = url.rstrip('/').split('/')[-1]
    bfu_id = metadata.get('serial_number', f"BFU-{slug}")
    
    return {
        'case_id': bfu_id,
        'title': title,
        'date': metadata.get('date', ''),
        'category': 'skatterett',
        'source': 'skatteetaten_bfu',
        'source_url': url,
        'sammendrag': sammendrag,
        'full_text': full_text,
        'metadata': metadata,
    }


def discover_more_urls_from_page(html, known_urls):
    """Extract additional BFU URLs from a fetched page."""
    soup = BeautifulSoup(html, 'html.parser')
    new_urls = set()
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(BASE_URL, href)
        if '/rettskilder/type/uttalelser/bfu/' in full_url:
            clean_url = full_url.split('?')[0].split('#')[0]
            if (clean_url.rstrip('/') != 'https://www.skatteetaten.no/rettskilder/type/uttalelser/bfu' 
                and '/sok' not in clean_url
                and clean_url not in known_urls):
                new_urls.add(clean_url)
    
    return new_urls


def main():
    parser = argparse.ArgumentParser(description="Scrape Skatteetaten BFU for NOR-CASEHOLD")
    parser.add_argument("--output", type=str, default="data/raw/bfu/")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to fetch")
    parser.add_argument("--urls-file", type=str, default=None, help="File with URLs to scrape (one per line)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Phase 1: Discover URLs
    print("=" * 60)
    print("Phase 1: Discovering BFU URLs")
    print("=" * 60)
    
    all_urls = set()
    
    if args.urls_file:
        # Load from file
        with open(args.urls_file) as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('http'):
                    all_urls.add(line)
        print(f"Loaded {len(all_urls)} URLs from {args.urls_file}")
    else:
        # Try discovery methods in order
        
        # Method 1: API
        api_result = discover_bfu_urls_via_api()
        if api_result:
            print(f"  API returned data, parsing...")
        
        # Method 2: Sitemap
        sitemap_urls = discover_bfu_urls_via_sitemap()
        if sitemap_urls:
            all_urls.update(sitemap_urls)
        
        # Method 3: Crawl listing page
        crawl_urls = discover_bfu_urls_via_crawl()
        if crawl_urls:
            all_urls.update(crawl_urls)
        
        # Method 4: Known patterns
        known_urls = discover_bfu_urls_via_google()
        if known_urls:
            all_urls.update(known_urls)
    
    print(f"\nTotal unique URLs discovered: {len(all_urls)}")
    
    if not all_urls:
        print("\nNo URLs found. Try these manual steps:")
        print("1. Open https://www.skatteetaten.no/rettskilder/type/uttalelser/bfu/")
        print("2. In browser DevTools (Network tab), look for API calls when the list loads")
        print("3. Or manually collect URLs and save to a file, then run:")
        print(f"   python {__file__} --urls-file bfu_urls.txt")
        return
    
    # Phase 2: Fetch and extract
    print("\n" + "=" * 60)
    print("Phase 2: Fetching and extracting BFU data")
    print("=" * 60)
    
    results = []
    failed = []
    urls_to_process = list(all_urls)
    processed_urls = set(all_urls)
    
    if args.max_pages:
        urls_to_process = urls_to_process[:args.max_pages]
    
    for i, url in enumerate(urls_to_process):
        print(f"\n[{i+1}/{len(urls_to_process)}] Fetching: {url}")
        
        html = fetch_page(url)
        if not html:
            print(f"  Failed to fetch")
            failed.append(url)
            continue
        
        # Try to discover more URLs from this page
        new_urls = discover_more_urls_from_page(html, processed_urls)
        if new_urls:
            print(f"  Discovered {len(new_urls)} new URLs from this page")
            for new_url in new_urls:
                if new_url not in processed_urls:
                    urls_to_process.append(new_url)
                    processed_urls.add(new_url)
        
        # Extract data
        data = extract_bfu_data(html, url)
        if data:
            results.append(data)
            print(f"  OK: {data['case_id']} — {data['title'][:60]}...")
            print(f"  Summary: {len(data['sammendrag'])} chars, Full text: {len(data['full_text'])} chars")
        else:
            print(f"  Could not extract structured data")
            failed.append(url)
    
    # Phase 3: Save results
    print("\n" + "=" * 60)
    print("Phase 3: Saving results")
    print("=" * 60)
    
    # Save as JSONL
    jsonl_path = os.path.join(args.output, "bfu_raw.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} records to {jsonl_path}")
    
    # Save URL list for future runs
    urls_path = os.path.join(args.output, "bfu_urls.txt")
    with open(urls_path, 'w') as f:
        for url in sorted(processed_urls):
            f.write(url + '\n')
    print(f"Saved {len(processed_urls)} URLs to {urls_path}")
    
    # Save failed URLs
    if failed:
        failed_path = os.path.join(args.output, "bfu_failed.txt")
        with open(failed_path, 'w') as f:
            for url in failed:
                f.write(url + '\n')
        print(f"Saved {len(failed)} failed URLs to {failed_path}")
    
    # Summary stats
    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"URLs discovered: {len(processed_urls)}")
    print(f"Pages fetched:   {len(urls_to_process)}")
    print(f"Records saved:   {len(results)}")
    print(f"Failed:          {len(failed)}")
    
    if results:
        avg_summary = sum(len(r['sammendrag']) for r in results) / len(results)
        avg_full = sum(len(r['full_text']) for r in results) / len(results)
        print(f"Avg summary:     {avg_summary:.0f} chars")
        print(f"Avg full text:   {avg_full:.0f} chars")
    
    print(f"\nNext steps:")
    print(f"1. Review {jsonl_path} for quality")
    print(f"2. If more URLs needed, check the listing page manually and add to {urls_path}")
    print(f"3. Re-run with: python {__file__} --urls-file {urls_path}")


if __name__ == "__main__":
    main()
