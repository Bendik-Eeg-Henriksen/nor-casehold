#!/usr/bin/env python3
"""
NOR-CASEHOLD: Hybrid BM25 + Dense Retrieval Baseline
=====================================================
Combines BM25 lexical scores with dense encoder cosine similarity
using a weighted combination: score = (1-α)·BM25 + α·Dense

Tunes α on the validation set, evaluates on the test set.

Usage (GPU recommended):
    python hybrid_baseline.py \
        --val-path data/v2/splits/val.jsonl \
        --test-path data/v2/splits/test.jsonl \
        --model bendik-eeg-henriksen/norwegian-legal-bert \
        --output results/hybrid_results.json

Usage (custom alpha, skip tuning):
    python hybrid_baseline.py \
        --test-path data/v2/splits/test.jsonl \
        --model bendik-eeg-henriksen/norwegian-legal-bert \
        --alpha 0.3 \
        --output results/hybrid_results.json
"""

import json
import os
import re
import math
import argparse
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer

# Reproducibility
SEED = 42
np.random.seed(SEED)

N_BOOTSTRAP = 10000
CI_LEVEL = 95

# ============================================================
# Shared utilities
# ============================================================

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_valid_sentences(sentences):
    valid = []
    for i, sent in enumerate(sentences):
        try:
            text = sent["text"].strip() if isinstance(sent, dict) else str(sent).strip()
        except (KeyError, TypeError, AttributeError):
            continue
        if len(text) >= 10:
            valid.append((i, text))
    return valid


def select_top_n_in_order(scored_items, n):
    top_n = sorted(scored_items, key=lambda x: x[1], reverse=True)[:n]
    top_n.sort(key=lambda x: x[0])
    return " ".join(t[2] for t in top_n)


# ============================================================
# BM25 scoring (per-document, returns per-sentence scores)
# ============================================================

def bm25_score_fn(query_tokens, doc_tokens, doc_freqs, n_docs, avgdl, k1=1.5, b=0.75):
    score = 0.0
    dl = len(doc_tokens)
    for term in query_tokens:
        if term not in doc_tokens:
            continue
        tf = doc_tokens.count(term)
        df = doc_freqs.get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        score += idf * tf_norm
    return score


def get_bm25_scores(record):
    """Returns dict mapping sentence_idx -> BM25 score."""
    sammendrag = record["sammendrag"]
    valid = get_valid_sentences(record["sentences"])
    tokenize = lambda t: re.findall(r'\w+', t.lower())
    query_tokens = tokenize(sammendrag)
    all_doc_tokens = [tokenize(text) for _, text in valid]
    
    doc_freqs = Counter()
    for dtoks in all_doc_tokens:
        for term in set(dtoks):
            doc_freqs[term] += 1
    
    n_docs = len(valid)
    avgdl = sum(len(d) for d in all_doc_tokens) / max(n_docs, 1)
    
    scores = {}
    for i, (idx, text) in enumerate(valid):
        s = bm25_score_fn(query_tokens, all_doc_tokens[i], doc_freqs, n_docs, avgdl)
        scores[idx] = s
    
    return scores


# ============================================================
# Dense scoring (batch per-document)
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts(texts, tokenizer, model, device, batch_size=16):
    all_embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        embeddings = mean_pool(output, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def get_dense_scores(record, tokenizer, model, device):
    """Returns dict mapping sentence_idx -> cosine similarity."""
    sammendrag = record["sammendrag"]
    sentences = record["sentences"]
    valid = get_valid_sentences(sentences)
    
    if not valid:
        return {}
    
    sent_texts = [text for _, text in valid]
    all_texts = [sammendrag] + sent_texts
    embeddings = encode_texts(all_texts, tokenizer, model, device)
    
    sammendrag_emb = embeddings[0]
    sent_embs = embeddings[1:]
    
    scores = {}
    for i, (orig_idx, text) in enumerate(valid):
        sim = float(np.dot(sammendrag_emb, sent_embs[i]))
        scores[orig_idx] = sim
    
    return scores


# ============================================================
# Hybrid scoring
# ============================================================

def normalize_scores(scores_dict):
    """Min-max normalize scores to [0, 1]."""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng < 1e-9:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - mn) / rng for k, v in scores_dict.items()}


def hybrid_score_and_extract(record, bm25_scores, dense_scores, alpha, n, scorer):
    """
    Combine BM25 and dense scores with weight alpha.
    score = (1 - alpha) * BM25_norm + alpha * dense_norm
    """
    sammendrag = record["sammendrag"]
    valid = get_valid_sentences(record["sentences"])
    
    # Normalize both score distributions to [0, 1]
    bm25_norm = normalize_scores(bm25_scores)
    dense_norm = normalize_scores(dense_scores)
    
    # Combine
    scored = []
    for idx, text in valid:
        b = bm25_norm.get(idx, 0.0)
        d = dense_norm.get(idx, 0.0)
        combined = (1 - alpha) * b + alpha * d
        scored.append((idx, combined, text))
    
    extracted = select_top_n_in_order(scored, n)
    if not extracted:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scores = scorer.score(sammendrag, extracted)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_ci(scores, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL):
    scores = np.array(scores)
    n = len(scores)
    boot_means = np.array([
        np.mean(scores[np.random.randint(0, n, size=n)])
        for _ in range(n_bootstrap)
    ])
    alpha_pct = (100 - ci_level) / 2
    lower = np.percentile(boot_means, alpha_pct)
    upper = np.percentile(boot_means, 100 - alpha_pct)
    return float(lower), float(upper)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid BM25 + Dense Baseline")
    parser.add_argument("--val-path", type=str, default=None,
                       help="Path to val JSONL (for alpha tuning)")
    parser.add_argument("--test-path", type=str, required=True,
                       help="Path to test JSONL")
    parser.add_argument("--model", type=str,
                       default="bendik-eeg-henriksen/norwegian-legal-bert",
                       help="HuggingFace model ID for dense encoder")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Fixed alpha (skip tuning). 0=pure BM25, 1=pure dense")
    parser.add_argument("--n-sentences", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    n = args.n_sentences
    
    # ---- Alpha tuning on val set ----
    if args.alpha is not None:
        best_alpha = args.alpha
        print(f"Using fixed alpha: {best_alpha}")
    else:
        if args.val_path is None:
            print("ERROR: Need --val-path for alpha tuning, or specify --alpha directly")
            return
        
        val_data = load_jsonl(args.val_path)
        print(f"Loaded {len(val_data)} val records for alpha tuning")
        
        # Pre-compute all BM25 and dense scores for val
        print("Pre-computing BM25 scores (val)...")
        val_bm25 = [get_bm25_scores(r) for r in val_data]
        
        print("Pre-computing dense scores (val)...")
        val_dense = []
        for i, r in enumerate(val_data):
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(val_data)}...")
            val_dense.append(get_dense_scores(r, tokenizer, model, device))
        
        # Sweep alpha
        alphas = np.arange(0.0, 1.01, 0.05)
        best_alpha = 0.0
        best_r1 = 0.0
        
        print(f"\nSweeping alpha from 0.0 to 1.0 (step 0.05)...")
        for alpha in alphas:
            r1_scores = []
            for i, r in enumerate(val_data):
                result = hybrid_score_and_extract(
                    r, val_bm25[i], val_dense[i], alpha, n, scorer
                )
                r1_scores.append(result["rouge1"])
            
            mean_r1 = np.mean(r1_scores) * 100
            if mean_r1 > best_r1:
                best_r1 = mean_r1
                best_alpha = alpha
            
            print(f"  α={alpha:.2f}  ROUGE-1={mean_r1:.2f}")
        
        print(f"\nBest alpha: {best_alpha:.2f} (val ROUGE-1: {best_r1:.2f})")
        
        # Clean up val data
        del val_data, val_bm25, val_dense
        torch.cuda.empty_cache()
    
    # ---- Evaluate on test set ----
    test_data = load_jsonl(args.test_path)
    print(f"\nLoaded {len(test_data)} test records")
    
    # Pre-compute scores
    print("Pre-computing BM25 scores (test)...")
    test_bm25 = [get_bm25_scores(r) for r in test_data]
    
    print("Pre-computing dense scores (test)...")
    test_dense = []
    for i, r in enumerate(test_data):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_data)}...")
        test_dense.append(get_dense_scores(r, tokenizer, model, device))
    
    # Score with best alpha
    print(f"\nScoring test set with α={best_alpha:.2f}...")
    test_results = []
    for i, r in enumerate(test_data):
        result = hybrid_score_and_extract(
            r, test_bm25[i], test_dense[i], best_alpha, n, scorer
        )
        test_results.append(result)
    
    r1_scores = [s["rouge1"] for s in test_results]
    r2_scores = [s["rouge2"] for s in test_results]
    rl_scores = [s["rougeL"] for s in test_results]
    
    r1_mean = np.mean(r1_scores) * 100
    r2_mean = np.mean(r2_scores) * 100
    rl_mean = np.mean(rl_scores) * 100
    
    r1_ci = bootstrap_ci(r1_scores, args.n_bootstrap)
    r2_ci = bootstrap_ci(r2_scores, args.n_bootstrap)
    rl_ci = bootstrap_ci(rl_scores, args.n_bootstrap)
    
    # Also compute per-source if source field exists
    per_source = {}
    for i, r in enumerate(test_data):
        src = r.get("source", "unknown")
        if src not in per_source:
            per_source[src] = {"rouge1": [], "rouge2": [], "rougeL": []}
        per_source[src]["rouge1"].append(test_results[i]["rouge1"])
        per_source[src]["rouge2"].append(test_results[i]["rouge2"])
        per_source[src]["rougeL"].append(test_results[i]["rougeL"])
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  Hybrid BM25 + {args.model.split('/')[-1]}")
    print(f"  α = {best_alpha:.2f}  |  n = {len(test_data)}  |  top-{n}")
    print(f"{'='*70}")
    print(f"  ROUGE-1: {r1_mean:.2f} [{r1_ci[0]*100:.2f}, {r1_ci[1]*100:.2f}]")
    print(f"  ROUGE-2: {r2_mean:.2f} [{r2_ci[0]*100:.2f}, {r2_ci[1]*100:.2f}]")
    print(f"  ROUGE-L: {rl_mean:.2f} [{rl_ci[0]*100:.2f}, {rl_ci[1]*100:.2f}]")
    
    if len(per_source) > 1:
        print(f"\n  Per-source breakdown:")
        for src in sorted(per_source):
            src_r1 = np.mean(per_source[src]["rouge1"]) * 100
            src_n = len(per_source[src]["rouge1"])
            print(f"    {src} (n={src_n}): ROUGE-1 = {src_r1:.2f}")
    
    print(f"{'='*70}")
    
    # Save
    if args.output:
        output_data = {
            "method": f"Hybrid BM25 + {args.model.split('/')[-1]}",
            "alpha": best_alpha,
            "model": args.model,
            "n_documents": len(test_data),
            "n_sentences": n,
            "n_bootstrap": args.n_bootstrap,
            "results": {
                "rouge1": {"mean": round(r1_mean, 2),
                          "ci_lower": round(r1_ci[0]*100, 2),
                          "ci_upper": round(r1_ci[1]*100, 2)},
                "rouge2": {"mean": round(r2_mean, 2),
                          "ci_lower": round(r2_ci[0]*100, 2),
                          "ci_upper": round(r2_ci[1]*100, 2)},
                "rougeL": {"mean": round(rl_mean, 2),
                          "ci_lower": round(rl_ci[0]*100, 2),
                          "ci_upper": round(rl_ci[1]*100, 2)},
            },
            "per_source": {
                src: {
                    "n": len(scores["rouge1"]),
                    "rouge1_mean": round(np.mean(scores["rouge1"]) * 100, 2),
                    "rouge2_mean": round(np.mean(scores["rouge2"]) * 100, 2),
                    "rougeL_mean": round(np.mean(scores["rougeL"]) * 100, 2),
                }
                for src, scores in per_source.items()
            },
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
