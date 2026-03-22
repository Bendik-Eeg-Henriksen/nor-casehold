#!/usr/bin/env python3
"""
NOR-CASEHOLD v2: BFU Conversion Pipeline
==========================================
Converts raw scraped BFU data into NOR-CASEHOLD format:
  - Strips boilerplate from sammendrag
  - Sentence-splits full_text (Norwegian legal-aware)
  - Filters to usable pairs (clean sammendrag >= 100 chars, full_text >= 500 chars)
  - Extracts dates from boilerplate
  - Outputs clean JSONL matching existing Høyesterett schema

Usage:
    python convert_bfu.py --input bfu_raw.jsonl --output bfu_clean.jsonl

Schema output (matches existing NOR-CASEHOLD):
    {
        "doc_id": "bfu-<slug>",
        "source": "skatteetaten_bfu",
        "case_name": "<title>",
        "date": "YYYY-MM-DD",
        "sammendrag": "<cleaned summary text>",
        "full_text": "<original full text>",
        "sentences": [{"text": "..."}, {"text": "..."}, ...],
        "url": "<source_url>"
    }
"""

import json
import re
import sys
import argparse
from collections import Counter


# ============================================================
# Sammendrag cleaning
# ============================================================

def clean_sammendrag(sam):
    """Strip navigation boilerplate and BFU reference headers."""
    original = sam
    
    # 1. Strip standard navigation prefix
    prefix = "Rettskilder Rettskilder per type Uttalelser Bindende forhåndsuttalelser Bindende forhåndsuttalelse"
    if sam.startswith(prefix):
        sam = sam[len(prefix):].strip()
    
    # 2. Strip date fields
    sam = re.sub(r'^Publisert:\d{2}\.\d{2}\.\d{4}\s*', '', sam).strip()
    sam = re.sub(r'^Avgitt\s*\d{2}\.\d{2}\.\d{4}\s*', '', sam).strip()
    
    # 3. Strip BFU reference header lines (various formats)
    # "Bindende forhåndsuttalelse fra Skattedirektoratet, avgitt mars 2004 (BFU 25/04)"
    # "Bindende forhåndsuttalelse fra Skattedirektoratet. BFU 26/09. Avgitt 22.09.2009"
    sam = re.sub(
        r'^Bindende forhåndsuttalelse fra Skattedirektoratet[^.]*\.\s*', 
        '', sam
    ).strip()
    sam = re.sub(
        r'^Bindende forhåndsuttalelse,?\s*avgitt[^.]*\.\s*', 
        '', sam
    ).strip()
    
    # 4. Strip standalone BFU number + date
    sam = re.sub(r'^BFU\s+\d+/\d+[^.]*\.\s*', '', sam).strip()
    sam = re.sub(r'^Avgitt\s+[\d.]+\s*', '', sam).strip()
    
    # 5. Strip leading law reference in parentheses if followed by real content
    #    e.g. "(Merverdiavgiftsloven § 5b føres ledd nr. 13)" at start
    sam = re.sub(r'^\([^)]{5,80}\)\s*', '', sam).strip()
    
    return sam


def extract_date(sammendrag_raw):
    """Extract date from boilerplate. Prefer 'Avgitt' (decision date) over 'Publisert'."""
    # Try Avgitt first (the actual decision date)
    avg = re.search(r'Avgitt\s*(\d{2})\.(\d{2})\.(\d{4})', sammendrag_raw)
    if avg:
        d, m, y = avg.group(1), avg.group(2), avg.group(3)
        return f"{y}-{m}-{d}"
    
    # Fall back to Publisert
    pub = re.search(r'Publisert:(\d{2})\.(\d{2})\.(\d{4})', sammendrag_raw)
    if pub:
        d, m, y = pub.group(1), pub.group(2), pub.group(3)
        return f"{y}-{m}-{d}"
    
    return ""


# ============================================================
# Norwegian legal sentence splitter
# ============================================================

# Pre-compile abbreviation patterns
_ABBREV_PATTERNS = [
    re.compile(p) for p in [
        r'jf\.', r'jfr\.', r'sktl\.', r'mval\.', r'nr\.', r'flg\.', r'mv\.',
        r'bl\.a\.', r'pkt\.', r'kap\.', r'fsfin\.', r'kr\.', r'ca\.', r'dvs\.',
        r'f\.eks\.', r'mfl\.', r'resp\.', r'evt\.', r'inkl\.', r'ekskl\.',
        r'hhv\.', r'ifm\.', r'iht\.', r'pga\.', r'osv\.', r'etc\.', r'ev\.',
        r'St\.meld\.', r'St\.prp\.', r'Ot\.prp\.', r'Innst\.O\.', r'Prop\.',
        r'Rt\.', r'Utv\.', r'op\.cit\.', r'ibid\.', r'vs\.',
        r'§\s*\d+[-\d]*\.', 
        r'\d+\.\s+(?:januar|februar|mars|april|mai|juni|juli|august|september|oktober|november|desember)',
        r'\d+\.\d+',
    ]
]


def split_sentences_no(text):
    """Norwegian-aware sentence splitter for legal text."""
    if not text or not text.strip():
        return []
    
    protected = text
    for pat in _ABBREV_PATTERNS:
        protected = pat.sub(lambda m: m.group().replace('.', '<DOT>'), protected)
    
    # Split on sentence-ending punctuation followed by whitespace + uppercase letter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÆØÅ\d«"])', protected)
    
    sentences = []
    for p in parts:
        restored = p.replace('<DOT>', '.').strip()
        if restored:
            sentences.append(restored)
    
    return sentences


# ============================================================
# Main conversion
# ============================================================

def convert_record(raw):
    """Convert a raw BFU record to NOR-CASEHOLD format. Returns None if unusable."""
    
    # Clean sammendrag
    sammendrag_raw = raw.get('sammendrag', '')
    sammendrag_clean = clean_sammendrag(sammendrag_raw)
    
    # Get full text
    full_text = raw.get('full_text', '').strip()
    
    # Filter: need substantial sammendrag and full_text
    if len(sammendrag_clean) < 100:
        return None
    if len(full_text) < 500:
        return None
    
    # Filter: sammendrag shouldn't be most of the full text
    if len(sammendrag_clean) / len(full_text) > 0.8:
        return None
    
    # Sentence-split full text
    sentences = split_sentences_no(full_text)
    
    # Filter: need enough sentences for extractive task
    valid_sentences = [s for s in sentences if len(s.strip()) >= 10]
    if len(valid_sentences) < 5:
        return None
    
    # Extract date
    date = extract_date(sammendrag_raw)
    
    # Build doc_id
    case_id = raw.get('case_id', '')
    if case_id:
        doc_id = case_id.lower()
        if not doc_id.startswith('bfu'):
            doc_id = f"bfu-{doc_id}"
    else:
        # Fallback: slug from URL
        url = raw.get('source_url', '')
        slug = url.rstrip('/').split('/')[-1] if url else 'unknown'
        doc_id = f"bfu-{slug}"
    
    return {
        "doc_id": doc_id,
        "source": "skatteetaten_bfu",
        "case_name": raw.get('title', ''),
        "date": date,
        "sammendrag": sammendrag_clean,
        "full_text": full_text,
        "sentences": [{"text": s} for s in sentences],
        "url": raw.get('source_url', ''),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert raw BFU data to NOR-CASEHOLD format")
    parser.add_argument("--input", type=str, required=True, help="Path to bfu_raw.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to output bfu_clean.jsonl")
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics")
    args = parser.parse_args()
    
    # Load raw data
    raw_records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    
    print(f"Loaded {len(raw_records)} raw BFU records")
    
    # Convert
    converted = []
    dropped_reasons = Counter()
    
    for raw in raw_records:
        sammendrag_clean = clean_sammendrag(raw.get('sammendrag', ''))
        full_text = raw.get('full_text', '').strip()
        
        if len(sammendrag_clean) < 100:
            dropped_reasons['short_sammendrag'] += 1
            continue
        if len(full_text) < 500:
            dropped_reasons['short_fulltext'] += 1
            continue
        if len(sammendrag_clean) / len(full_text) > 0.8:
            dropped_reasons['high_ratio'] += 1
            continue
        
        result = convert_record(raw)
        if result is None:
            dropped_reasons['few_sentences'] += 1
            continue
        
        converted.append(result)
    
    # Check for duplicate doc_ids
    doc_ids = [r['doc_id'] for r in converted]
    dupes = [did for did, cnt in Counter(doc_ids).items() if cnt > 1]
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate doc_ids found, deduplicating...")
        seen = set()
        deduped = []
        for r in converted:
            if r['doc_id'] not in seen:
                seen.add(r['doc_id'])
                deduped.append(r)
            else:
                dropped_reasons['duplicate'] += 1
        converted = deduped
    
    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Report
    print(f"\nConversion complete:")
    print(f"  Input:    {len(raw_records)} raw records")
    print(f"  Output:   {len(converted)} clean records")
    print(f"  Dropped:  {len(raw_records) - len(converted)}")
    for reason, count in sorted(dropped_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    
    if args.stats and converted:
        import statistics
        sam_lens = [len(r['sammendrag']) for r in converted]
        ft_lens = [len(r['full_text']) for r in converted]
        sent_counts = [len(r['sentences']) for r in converted]
        ratios = [len(r['sammendrag'])/len(r['full_text']) for r in converted]
        dates_present = sum(1 for r in converted if r['date'])
        
        print(f"\nStatistics:")
        print(f"  Sammendrag chars: min={min(sam_lens)}, median={statistics.median(sam_lens):.0f}, mean={statistics.mean(sam_lens):.0f}, max={max(sam_lens)}")
        print(f"  Full text chars:  min={min(ft_lens)}, median={statistics.median(ft_lens):.0f}, mean={statistics.mean(ft_lens):.0f}, max={max(ft_lens)}")
        print(f"  Sentences/doc:    min={min(sent_counts)}, median={statistics.median(sent_counts):.0f}, mean={statistics.mean(sent_counts):.0f}, max={max(sent_counts)}")
        print(f"  Sam/FT ratio:     median={statistics.median(ratios):.3f}, mean={statistics.mean(ratios):.3f}")
        print(f"  Date extracted:   {dates_present}/{len(converted)} ({dates_present/len(converted)*100:.0f}%)")


if __name__ == "__main__":
    main()
