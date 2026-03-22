#!/usr/bin/env python3
"""
NOR-CASEHOLD Full Significance Testing (Sparse + Dense + Sequential Oracle)
============================================================================
Bootstrap confidence intervals and pairwise significance tests
for all 8 baselines plus a greedy sequential oracle.

Usage (GPU required for dense models):
    python significance_full.py --output results/significance_full.json

Usage (sparse only, no GPU):
    python significance_full.py --no-dense --output results/significance_sparse.json
"""

import json
import os
import sys
import math
import re
import argparse
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Optional imports
HAS_TORCH = False
HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

SPLITS_DIR = "./data/splits"
N_BOOTSTRAP = 10000
CI_LEVEL = 95


# ============================================================
# Shared utilities
# ============================================================

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
# Sparse baselines (per-document scoring)
# ============================================================

def score_lead(record, n, scorer):
    sammendrag = record["sammendrag"]
    valid = get_valid_sentences(record["sentences"])
    extracted = " ".join(text for _, text in valid[:n])
    if not extracted:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(sammendrag, extracted)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


def score_oracle_greedy(record, n, scorer):
    """Original greedy oracle — picks top N sentences by individual ROUGE-1."""
    sammendrag = record["sammendrag"]
    valid = get_valid_sentences(record["sentences"])
    scored = []
    for idx, text in valid:
        s = scorer.score(sammendrag, text)
        scored.append((idx, s["rouge1"].fmeasure, text))
    extracted = select_top_n_in_order(scored, n)
    scores = scorer.score(sammendrag, extracted)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


def score_oracle_sequential(record, n, scorer):
    """Greedy sequential oracle — picks sentences that maximize cumulative ROUGE-1."""
    sammendrag = record["sammendrag"]
    valid = get_valid_sentences(record["sentences"])
    
    selected = []
    remaining = list(valid)
    
    for _ in range(min(n, len(remaining))):
        best_score = -1
        best_idx = -1
        best_item = None
        
        for i, (idx, text) in enumerate(remaining):
            # Try adding this sentence to current selection
            candidate_texts = [t for _, t in selected] + [text]
            candidate_extract = " ".join(candidate_texts)
            s = scorer.score(sammendrag, candidate_extract)
            r1 = s["rouge1"].fmeasure
            
            if r1 > best_score:
                best_score = r1
                best_idx = i
                best_item = (idx, text)
        
        if best_item is not None:
            selected.append(best_item)
            remaining.pop(best_idx)
    
    # Restore document order
    selected.sort(key=lambda x: x[0])
    extracted = " ".join(text for _, text in selected)
    
    if not extracted:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(sammendrag, extracted)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


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


def score_bm25(record, n, scorer):
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
    scored = []
    for i, (idx, text) in enumerate(valid):
        s = bm25_score_fn(query_tokens, all_doc_tokens[i], doc_freqs, n_docs, avgdl)
        scored.append((idx, s, text))
    extracted = select_top_n_in_order(scored, n)
    scores = scorer.score(sammendrag, extracted)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


def score_tfidf(record, n, scorer):
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
    vocab = sorted(set(query_tokens) | set(t for d in all_doc_tokens for t in d))
    term_to_idx = {t: i for i, t in enumerate(vocab)}
    idf = {t: math.log((n_docs + 1) / (doc_freqs.get(t, 0) + 1)) + 1 for t in vocab}

    def tfidf_vec(tokens):
        vec = np.zeros(len(vocab))
        counts = Counter(tokens)
        for t, c in counts.items():
            if t in term_to_idx:
                tf = c / max(len(tokens), 1)
                vec[term_to_idx[t]] = tf * idf.get(t, 1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    query_vec = tfidf_vec(query_tokens)
    scored = []
    for i, (idx, text) in enumerate(valid):
        doc_vec = tfidf_vec(all_doc_tokens[i])
        sim = np.dot(query_vec, doc_vec)
        scored.append((idx, sim, text))
    extracted = select_top_n_in_order(scored, n)
    scores = scorer.score(sammendrag, extracted)
    return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}


# ============================================================
# Dense model scoring (per-document)
# ============================================================

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


def make_encoder_scorer(model_id):
    """Returns a per-document scoring function for a BERT encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    def score_fn(record, n, scorer):
        sammendrag = record["sammendrag"]
        sentences = record["sentences"]
        sent_texts = [s["text"] for s in sentences]
        all_texts = [sammendrag] + sent_texts
        embeddings = encode_texts(all_texts, tokenizer, model, device)
        sammendrag_emb = embeddings[0]
        sentence_embs = embeddings[1:]
        valid = get_valid_sentences(sentences)
        scored = []
        for orig_idx, text in valid:
            sim = np.dot(sammendrag_emb, sentence_embs[orig_idx]) / (
                np.linalg.norm(sammendrag_emb) * np.linalg.norm(sentence_embs[orig_idx]) + 1e-9)
            scored.append((orig_idx, sim, text))
        extracted = select_top_n_in_order(scored, n)
        scores = scorer.score(sammendrag, extracted)
        return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}

    return score_fn, model, tokenizer


def make_st_scorer(model_id):
    """Returns a per-document scoring function for a sentence-transformer."""
    st_model = SentenceTransformer(model_id)

    def score_fn(record, n, scorer):
        sammendrag = record["sammendrag"]
        sentences = record["sentences"]
        valid = get_valid_sentences(sentences)
        sent_texts = [text for _, text in valid]
        all_texts = [sammendrag] + sent_texts
        embeddings = st_model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        sammendrag_emb = embeddings[0]
        sent_embs = embeddings[1:]
        scored = []
        for i, (orig_idx, text) in enumerate(valid):
            sim = np.dot(sammendrag_emb, sent_embs[i])
            scored.append((orig_idx, sim, text))
        extracted = select_top_n_in_order(scored, n)
        scores = scorer.score(sammendrag, extracted)
        return {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}

    return score_fn, st_model


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
    alpha = (100 - ci_level) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return float(lower), float(upper)


def pairwise_bootstrap_test(scores_a, scores_b, n_bootstrap=N_BOOTSTRAP):
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)
    a_wins = 0
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        if np.mean(scores_a[idx]) > np.mean(scores_b[idx]):
            a_wins += 1
    return 1.0 - (a_wins / n_bootstrap)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NOR-CASEHOLD Full Significance Testing")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_sentences", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--no-dense", action="store_true", help="Skip dense models")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load data
    split_path = os.path.join(SPLITS_DIR, f"{args.split}.jsonl")
    test_data = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    print(f"Loaded {len(test_data)} records from {split_path}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Confidence level: {CI_LEVEL}%")
    print()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    n = args.n_sentences

    # Build method list
    methods = {}

    # Sparse + structural
    methods["BM25"] = lambda r: score_bm25(r, n, scorer)
    methods["TF-IDF"] = lambda r: score_tfidf(r, n, scorer)
    methods["Oracle (greedy R-1)"] = lambda r: score_oracle_greedy(r, n, scorer)
    methods["Oracle (sequential R-1)"] = lambda r: score_oracle_sequential(r, n, scorer)
    methods["Lead-5"] = lambda r: score_lead(r, n, scorer)

    # Dense models
    if not args.no_dense:
        dense_models = [
            ("Norwegian Legal BERT", "bendik-eeg-henriksen/norwegian-legal-bert", "encoder"),
            ("NB-BERT-base", "NbAiLab/nb-bert-base", "encoder"),
            ("mBERT", "bert-base-multilingual-cased", "encoder"),
            ("MiniLM (multilingual ST)", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "st"),
        ]

        for name, model_id, model_type in dense_models:
            print(f"Loading: {name}...")
            if model_type == "encoder" and HAS_TORCH:
                score_fn, model, tokenizer = make_encoder_scorer(model_id)
                methods[name] = lambda r, fn=score_fn: fn(r, n, scorer)
            elif model_type == "st" and HAS_SENTENCE_TRANSFORMERS:
                score_fn, st_model = make_st_scorer(model_id)
                methods[name] = lambda r, fn=score_fn: fn(r, n, scorer)
            else:
                print(f"  Skipping {name} (missing dependencies)")

    # Score all documents for each method
    all_scores = {}
    for name, fn in methods.items():
        print(f"Scoring: {name}...")
        doc_scores = []
        for i, record in enumerate(test_data):
            if i % 20 == 0 and i > 0:
                print(f"  {i}/{len(test_data)}...")
            doc_scores.append(fn(record))
        all_scores[name] = {
            "rouge1": [s["rouge1"] for s in doc_scores],
            "rouge2": [s["rouge2"] for s in doc_scores],
            "rougeL": [s["rougeL"] for s in doc_scores],
        }

        # Free GPU memory after each dense model
        if not args.no_dense and HAS_TORCH:
            torch.cuda.empty_cache()

    # Print results with CIs
    col_w = 28
    print(f"\n{'='*96}")
    print(f"  NOR-CASEHOLD Results with {CI_LEVEL}% Bootstrap CIs")
    print(f"  (test split, n={len(test_data)}, top-{n}, {args.n_bootstrap} bootstrap samples)")
    print(f"{'='*96}")
    print(f"  {'METHOD':<{col_w}} {'ROUGE-1':>22} {'ROUGE-2':>22} {'ROUGE-L':>22}")
    print(f"  {'-'*(col_w + 68)}")

    results_output = {}
    for name in methods:
        r1 = all_scores[name]["rouge1"]
        r2 = all_scores[name]["rouge2"]
        rl = all_scores[name]["rougeL"]

        r1_mean = np.mean(r1) * 100
        r2_mean = np.mean(r2) * 100
        rl_mean = np.mean(rl) * 100

        r1_ci = bootstrap_ci(r1, args.n_bootstrap)
        r2_ci = bootstrap_ci(r2, args.n_bootstrap)
        rl_ci = bootstrap_ci(rl, args.n_bootstrap)

        r1_str = f"{r1_mean:5.2f} [{r1_ci[0]*100:5.2f}, {r1_ci[1]*100:5.2f}]"
        r2_str = f"{r2_mean:5.2f} [{r2_ci[0]*100:5.2f}, {r2_ci[1]*100:5.2f}]"
        rl_str = f"{rl_mean:5.2f} [{rl_ci[0]*100:5.2f}, {rl_ci[1]*100:5.2f}]"

        print(f"  {name:<{col_w}} {r1_str:>22} {r2_str:>22} {rl_str:>22}")

        results_output[name] = {
            "rouge1": {"mean": round(r1_mean, 2), "ci_lower": round(r1_ci[0]*100, 2), "ci_upper": round(r1_ci[1]*100, 2)},
            "rouge2": {"mean": round(r2_mean, 2), "ci_lower": round(r2_ci[0]*100, 2), "ci_upper": round(r2_ci[1]*100, 2)},
            "rougeL": {"mean": round(rl_mean, 2), "ci_lower": round(rl_ci[0]*100, 2), "ci_upper": round(rl_ci[1]*100, 2)},
        }

    print(f"  {'='*(col_w + 68)}")

    # Pairwise tests
    comparisons = [
        ("BM25", "TF-IDF"),
        ("BM25", "Oracle (greedy R-1)"),
        ("BM25", "Oracle (sequential R-1)"),
        ("Oracle (sequential R-1)", "Oracle (greedy R-1)"),
    ]

    # Add dense comparisons if available
    if "Norwegian Legal BERT" in all_scores:
        comparisons.extend([
            ("BM25", "Norwegian Legal BERT"),
            ("Norwegian Legal BERT", "NB-BERT-base"),
            ("Norwegian Legal BERT", "mBERT"),
            ("Norwegian Legal BERT", "MiniLM (multilingual ST)"),
            ("NB-BERT-base", "mBERT"),
        ])

    print(f"\n{'='*70}")
    print(f"  Pairwise Bootstrap Significance Tests (ROUGE-1)")
    print(f"{'='*70}")

    pairwise_output = []
    for a, b in comparisons:
        if a not in all_scores or b not in all_scores:
            continue
        p_val = pairwise_bootstrap_test(
            all_scores[a]["rouge1"],
            all_scores[b]["rouge1"],
            args.n_bootstrap
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        mean_a = np.mean(all_scores[a]["rouge1"]) * 100
        mean_b = np.mean(all_scores[b]["rouge1"]) * 100
        diff = mean_a - mean_b

        print(f"  {a} vs {b}")
        print(f"    Δ R-1 = {diff:+.2f}  p = {p_val:.4f}  {sig}")
        print()

        pairwise_output.append({
            "method_a": a, "method_b": b,
            "delta_r1": round(diff, 2), "p_value": round(p_val, 4),
            "significant": sig
        })

    print(f"  *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print(f"{'='*70}")

    # Save
    if args.output:
        output_data = {
            "config": {
                "n_documents": len(test_data),
                "n_sentences": n,
                "n_bootstrap": args.n_bootstrap,
                "ci_level": CI_LEVEL,
                "seed": SEED,
            },
            "confidence_intervals": results_output,
            "pairwise_tests": pairwise_output,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
