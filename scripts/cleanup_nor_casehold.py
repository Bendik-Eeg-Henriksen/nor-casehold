#!/usr/bin/env python3
"""
NOR-CASEHOLD Dataset Cleanup Script v1.1
=========================================
Cleans the NOR-CASEHOLD benchmark dataset:

1. Drops -U (ankeutvalg) records with garbage sammendrag (judge names only)
2. Strips page-break header artifacts from sentences  
3. Flags junk sentences as is_short_fragment=True
4. Strips embedded newlines from sentence text
5. Recalculates num_scorable_sentences
6. Produces a cleanup report

Usage:
    python cleanup_nor_casehold.py --input-dir ./data --output-dir ./data_clean

Expects train.jsonl, val.jsonl, test.jsonl in input-dir.
"""

import json
import re
import os
import argparse
from collections import Counter
from pathlib import Path


# === DETECTION PATTERNS ===

# Page-break header pattern: "22-076582SIV-HRET)\n\n13\n" prepended to sentences
PAGE_HEADER_RE = re.compile(r'^\d{2}-\d+[A-Z]+-[A-Z]+\)\s*\n+\s*\d+\s*\n+')

# Case ID in parenthetical reference: "HR-2023-299-A, (sak nr."
CASE_ID_REF_RE = re.compile(r'^HR-\d{4}-\d+-[A-Z],?\s*\(sak\s+nr\.')

# Norwegian month names for date fragment detection
MONTHS = r'(januar|februar|mars|april|mai|juni|juli|august|september|oktober|november|desember)'

# Date fragment: sentence starts with month name from a split date
DATE_FRAG_RE = re.compile(rf'^{MONTHS}\s+\d{{4}}', re.IGNORECASE)

# Left side of date split: ends with "number." like "Tilsynet påla 12."
DATE_SPLIT_LEFT_RE = re.compile(r'\b\d{1,2}\.$')


def clean_sentence_text(text):
    """Strip embedded newlines, normalize whitespace."""
    # Replace newlines with spaces
    cleaned = text.replace('\n', ' ')
    # Collapse multiple spaces
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    return cleaned.strip()


def strip_page_header(text):
    """Remove page-break header artifacts from start of sentence."""
    return PAGE_HEADER_RE.sub('', text).strip()


def classify_sentence(text, case_id):
    """
    Classify a sentence as junk or normal.
    Returns (is_junk: bool, category: str, cleaned_text: str)
    """
    original = text
    txt = text.strip()
    
    # Step 1: Strip page header if present
    had_header = bool(PAGE_HEADER_RE.match(txt))
    if had_header:
        txt = strip_page_header(txt)
    
    words = len(txt.split())
    
    # Case ID reference as standalone sentence: "HR-2023-299-A, (sak nr."
    if CASE_ID_REF_RE.match(txt):
        return True, 'case_id_header', txt
    
    # Check if case_id appears in short fragment
    # Normalize both for comparison (some IDs have -00 padding)
    cid_norm = case_id.replace('-00', '-').replace('-0', '-')
    txt_norm = txt.replace('-00', '-').replace('-0', '-')
    if cid_norm in txt_norm and words <= 6:
        return True, 'case_id_header', txt
    
    # After header strip, if barely anything remains
    if had_header and words <= 2:
        return True, 'page_header_remnant', txt
    
    # Short date fragment: "desember 2019." (from split "13. desember 2019")
    if DATE_FRAG_RE.match(txt) and words <= 3:
        return True, 'date_fragment', txt
    
    # Left side of date split: "Tilsynet påla 12." or "139/2004 av 20."
    if words <= 4 and DATE_SPLIT_LEFT_RE.search(txt):
        return True, 'date_split_left', txt
    
    # Minimal fragments
    if words <= 2:
        lower = txt.lower().rstrip('.')
        if lower in ['', 'likeså', 'mars 1998']:
            return True, 'minimal_fragment', txt
        # Pure numbers/punctuation
        if re.match(r'^[\d\s\.\-]+$', txt):
            return True, 'page_number', txt
    
    # Not junk - return cleaned text (with header stripped if applicable)
    return False, 'normal', txt


def process_record(record):
    """Process a single record: clean sentences, flag junk, recalculate stats."""
    case_id = record['case_id']
    
    new_sentences = []
    num_scorable = 0
    
    for sent in record['sentences']:
        is_junk, category, cleaned_text = classify_sentence(sent['text'], case_id)
        
        # Clean the text (strip newlines etc)
        final_text = clean_sentence_text(cleaned_text)
        
        new_sent = {
            'sentence_idx': sent['sentence_idx'],
            'text': final_text,
            'rouge1_f': sent['rouge1_f'],
            'rouge2_f': sent['rouge2_f'],
            'hm_score': sent['hm_score'],
            'is_short_fragment': is_junk,
        }
        new_sentences.append(new_sent)
        
        if not is_junk:
            num_scorable += 1
    
    # Rebuild record
    cleaned = {
        'case_id': record['case_id'],
        'title': record['title'],
        'date': record.get('date', ''),
        'category': record.get('category', ''),
        'rettsomrade': record.get('rettsomrade', ''),
        'source_url': record['source_url'],
        'sammendrag': record['sammendrag'],
        'full_text': record['full_text'],
        'sentences': new_sentences,
        'num_sentences': len(new_sentences),
        'num_scorable_sentences': num_scorable,
        'num_words_full': record['num_words_full'],
        'num_words_sammendrag': record['num_words_sammendrag'],
        'compression_ratio': record['compression_ratio'],
        'top5_indices': record['top5_indices'],
        'top5_hm_scores': record['top5_hm_scores'],
    }
    
    return cleaned


def should_drop(record):
    """Check if record should be dropped entirely."""
    # Drop -U records with garbage sammendrag (just judge names)
    if record['case_id'].endswith('-U') and record['num_words_sammendrag'] <= 15:
        return True, 'ankeutvalg_no_real_sammendrag'
    return False, None


def process_split(input_path, output_path, split_name, report):
    """Process a single split file."""
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    
    original_count = len(records)
    dropped = []
    kept = []
    
    for r in records:
        drop, reason = should_drop(r)
        if drop:
            dropped.append((r['case_id'], reason))
        else:
            cleaned = process_record(r)
            kept.append(cleaned)
    
    # Write output
    with open(output_path, 'w') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Collect stats for report
    total_sents = sum(r['num_sentences'] for r in kept)
    total_scorable = sum(r['num_scorable_sentences'] for r in kept)
    total_flagged = total_sents - total_scorable
    
    split_report = {
        'split': split_name,
        'original_records': original_count,
        'dropped_records': len(dropped),
        'final_records': len(kept),
        'dropped_details': dropped,
        'total_sentences': total_sents,
        'scorable_sentences': total_scorable,
        'flagged_junk': total_flagged,
        'junk_pct': f"{total_flagged/total_sents*100:.1f}%" if total_sents > 0 else "0%",
    }
    report['splits'].append(split_report)
    
    return kept


def main():
    parser = argparse.ArgumentParser(description='Clean NOR-CASEHOLD dataset')
    parser.add_argument('--input-dir', required=True, help='Directory with train.jsonl, val.jsonl, test.jsonl')
    parser.add_argument('--output-dir', required=True, help='Output directory for cleaned files')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {'splits': [], 'summary': {}}
    
    split_files = [
        ('train', 'train.jsonl'),
        ('val', 'val.jsonl'),
        ('test', 'test.jsonl'),
    ]
    
    total_original = 0
    total_final = 0
    
    for split_name, filename in split_files:
        input_path = input_dir / filename
        output_path = output_dir / filename
        
        if not input_path.exists():
            print(f"  WARNING: {input_path} not found, skipping")
            continue
        
        print(f"Processing {split_name}...")
        kept = process_split(input_path, output_path, split_name, report)
        total_original += report['splits'][-1]['original_records']
        total_final += len(kept)
        
        sr = report['splits'][-1]
        print(f"  {sr['original_records']} -> {sr['final_records']} records ({sr['dropped_records']} dropped)")
        print(f"  {sr['total_sentences']} sentences, {sr['flagged_junk']} flagged as junk ({sr['junk_pct']})")
        if sr['dropped_details']:
            for cid, reason in sr['dropped_details']:
                print(f"    DROPPED: {cid} ({reason})")
    
    report['summary'] = {
        'total_original': total_original,
        'total_final': total_final,
        'total_dropped': total_original - total_final,
    }
    
    # Write report
    report_path = output_dir / 'cleanup_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"CLEANUP COMPLETE")
    print(f"{'='*60}")
    print(f"Records: {total_original} -> {total_final} ({total_original - total_final} dropped)")
    print(f"Report saved to: {report_path}")
    print(f"Cleaned files saved to: {output_dir}")


if __name__ == '__main__':
    main()

