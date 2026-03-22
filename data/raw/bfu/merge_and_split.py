#!/usr/bin/env python3
"""
NOR-CASEHOLD v2: Merge & Stratified Split
==========================================
Merges Høyesterett (v1) and BFU records into a single dataset,
then creates source-stratified train/val/test splits.

Usage:
    python merge_and_split.py \
        --hoyesterett data/splits/train.jsonl data/splits/val.jsonl data/splits/test.jsonl \
        --bfu bfu_clean.jsonl \
        --output-dir data/v2/splits \
        --train-ratio 0.7 --val-ratio 0.115 --test-ratio 0.185

Split ratios default to 70/11.5/18.5 to produce roughly:
    - Train: ~877 docs (614 HR + 432 BFU)
    - Val:   ~144 docs (72 HR + 71 BFU)
    - Test:  ~232 docs (116 HR + 114 BFU)

The test set is slightly larger than v1 to give more statistical power,
with balanced representation from both sources.
"""

import json
import os
import random
import argparse
from collections import Counter


SEED = 42


def load_jsonl(paths):
    """Load records from one or more JSONL files."""
    records = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def stratified_split(records, train_ratio, val_ratio, test_ratio, seed=SEED):
    """
    Split records into train/val/test with stratification by source.
    Each source gets split independently at the same ratios.
    """
    random.seed(seed)
    
    # Group by source
    by_source = {}
    for r in records:
        src = r.get('source', 'unknown')
        by_source.setdefault(src, []).append(r)
    
    train, val, test = [], [], []
    
    for source, source_records in sorted(by_source.items()):
        # Shuffle deterministically
        shuffled = list(source_records)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Test gets the remainder
        
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
    
    # Shuffle each split (so sources are interleaved)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def print_split_stats(name, records):
    sources = Counter(r.get('source', 'unknown') for r in records)
    sam_lens = [len(r.get('sammendrag', '')) for r in records]
    ft_lens = [len(r.get('full_text', '')) for r in records]
    sent_counts = [len(r.get('sentences', [])) for r in records]
    
    print(f"\n  {name}: {len(records)} docs")
    for src, cnt in sorted(sources.items()):
        print(f"    {src}: {cnt}")
    if sam_lens:
        import statistics
        print(f"    Sammendrag: median={statistics.median(sam_lens):.0f}, mean={statistics.mean(sam_lens):.0f}")
        print(f"    Full text:  median={statistics.median(ft_lens):.0f}, mean={statistics.mean(ft_lens):.0f}")
        print(f"    Sentences:  median={statistics.median(sent_counts):.0f}, mean={statistics.mean(sent_counts):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Merge and split NOR-CASEHOLD v2")
    parser.add_argument("--hoyesterett", nargs='+', required=True,
                       help="Path(s) to existing Høyesterett JSONL files")
    parser.add_argument("--bfu", type=str, required=True,
                       help="Path to cleaned BFU JSONL")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.115)
    parser.add_argument("--test-ratio", type=float, default=0.185)
    args = parser.parse_args()
    
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 0.01, \
        f"Ratios must sum to 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}"
    
    # Load data
    hr_records = load_jsonl(args.hoyesterett)
    bfu_records = load_jsonl(args.bfu)
    
    # Ensure source field is set
    for r in hr_records:
        if 'source' not in r or not r['source']:
            r['source'] = 'hoyesterett'
    for r in bfu_records:
        if 'source' not in r or not r['source']:
            r['source'] = 'skatteetaten_bfu'
    
    print(f"Loaded:")
    print(f"  Høyesterett: {len(hr_records)} records")
    print(f"  BFU:         {len(bfu_records)} records")
    
    # Merge
    all_records = hr_records + bfu_records
    print(f"  Combined:    {len(all_records)} records")
    
    # Check for doc_id collisions
    doc_ids = [r.get('doc_id', '') for r in all_records]
    dupes = [did for did, cnt in Counter(doc_ids).items() if cnt > 1 and did]
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate doc_ids across sources")
        for d in dupes[:5]:
            print(f"  {d}")
    
    # Split
    train, val, test = stratified_split(
        all_records, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # Save
    save_jsonl(train, os.path.join(args.output_dir, 'train.jsonl'))
    save_jsonl(val, os.path.join(args.output_dir, 'val.jsonl'))
    save_jsonl(test, os.path.join(args.output_dir, 'test.jsonl'))
    
    # Also save the full merged dataset
    save_jsonl(all_records, os.path.join(args.output_dir, 'all.jsonl'))
    
    # Report
    print(f"\nSplits saved to {args.output_dir}/")
    print(f"  Ratios: {args.train_ratio:.0%} / {args.val_ratio:.1%} / {args.test_ratio:.1%}")
    print_split_stats("Train", train)
    print_split_stats("Val", val)
    print_split_stats("Test", test)
    
    # Source balance in test set
    test_sources = Counter(r.get('source', '?') for r in test)
    print(f"\n  Test set source balance:")
    for src, cnt in sorted(test_sources.items()):
        print(f"    {src}: {cnt} ({cnt/len(test)*100:.0f}%)")


if __name__ == "__main__":
    main()
