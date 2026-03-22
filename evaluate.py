#!/usr/bin/env python3
"""
NOR-CASEHOLD Benchmark Evaluation Harness
==========================================
The standard evaluation tool for the NOR-CASEHOLD benchmark.

Evaluate any model against the NOR-CASEHOLD benchmark with one command.
Test data is auto-downloaded from HuggingFace. Your results are printed
alongside published baselines for direct comparison.

Usage:
    # Evaluate a BERT-style encoder
    python evaluate.py --model your-org/your-legal-bert

    # Evaluate a sentence-transformer
    python evaluate.py --sentence-transformer your-org/your-st-model

    # Run all published baselines (requires GPU for dense encoder models)
    python evaluate.py --all

    # Run only sparse/structural baselines (no GPU needed)
    python evaluate.py --all --no-dense

    # Evaluate on a different split
    python evaluate.py --model your-org/your-model --split val

Baselines included:
    Sparse:     BM25, TF-IDF
    Structural: Lead-5 (first 5 sentences)
    Oracle:     Greedy ROUGE-1 sentence selector
    Dense:      Norwegian Legal BERT, NB-BERT-base, mBERT, MiniLM (ST)

All dense encoder embeddings use mean pooling with max_length=256.

Repository: https://github.com/Bendik-Eeg-Henriksen/nor-casehold
Dataset:    https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold
Paper:      [forthcoming]

License: Apache 2.0 (code), CC-BY-4.0 (dataset)
"""

import json
import os
import sys
import argparse
import math
import re
import random
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Optional imports
HAS_TORCH = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_DATASETS = False

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

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    pass


# ============================================================
# Published results (NOR-CASEHOLD v2, test split n=233, top-5)
# ============================================================

PUBLISHED_RESULTS = [
    {"encoder": "Oracle (sequential R-1)", "model_id": "Sequential greedy ROUGE-1 sentence selection",              "rouge1": 55.87, "rouge2": 30.59, "rougeL": 34.68},
    {"encoder": "TF-IDF",                  "model_id": "TF-IDF cosine similarity",                                 "rouge1": 47.85, "rouge2": 26.46, "rougeL": 30.99},
    {"encoder": "BM25",                    "model_id": "BM25 (k1=1.5, b=0.75)",                                    "rouge1": 47.49, "rouge2": 26.21, "rougeL": 30.64},
    {"encoder": "Oracle (greedy R-1)",      "model_id": "Greedy ROUGE-1 sentence selection",                        "rouge1": 42.68, "rouge2": 20.65, "rougeL": 25.30},
    {"encoder": "Norwegian Legal BERT",     "model_id": "bendik-eeg-henriksen/norwegian-legal-bert",                "rouge1": 38.40, "rouge2": 15.97, "rougeL": 20.93},
    {"encoder": "MiniLM (multilingual ST)", "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "rouge1": 37.47, "rouge2": 16.03, "rougeL": 21.54},
    {"encoder": "mBERT",                    "model_id": "bert-base-multilingual-cased",                             "rouge1": 37.34, "rouge2": 15.14, "rougeL": 20.49},
    {"encoder": "NB-BERT-base",             "model_id": "NbAiLab/nb-bert-base",                                    "rouge1": 37.28, "rouge2": 15.51, "rougeL": 20.64},
    {"encoder": "Lead-5",                   "model_id": "First 5 sentences",                                        "rouge1": 25.07, "rouge2":  7.54, "rougeL": 13.44},
]

# Default models for --all mode
DEFAULT_ENCODERS = [
    ("norwegian-legal-bert", "bendik-eeg-henriksen/norwegian-legal-bert"),
    ("nb-bert-base", "NbAiLab/nb-bert-base"),
    ("mbert", "bert-base-multilingual-cased"),
]

DEFAULT_ST_MODELS = [
    ("multilingual-MiniLM", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
]

VALID_SPLITS = {"train", "val", "validation", "test"}
SPLIT_ALIASES = {"val": "validation", "validation": "validation"}

SPLITS_DIR = "./data/splits"
HF_DATASET = "bendik-eeg-henriksen/nor-casehold"


# ============================================================
# Data loading (auto-download from HuggingFace)
# ============================================================

def load_test_data(split="test"):
    """Load data from local files or auto-download from HuggingFace."""
    # Normalize split name
    hf_split = SPLIT_ALIASES.get(split, split)
    local_path = os.path.join(SPLITS_DIR, f"{split}.jsonl")

    # Also try aliased name locally
    alias_path = os.path.join(SPLITS_DIR, f"{hf_split}.jsonl") if hf_split != split else None

    # Try local first
    for path in [local_path, alias_path]:
        if path and os.path.exists(path):
            test_data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        test_data.append(json.loads(line))
            print(f"Loaded {len(test_data)} records from {path}")
            return test_data

    # Auto-download from HuggingFace
    if HAS_DATASETS:
        print(f"Local data not found. Downloading from HuggingFace ({HF_DATASET})...")
        ds = load_dataset(HF_DATASET, split=hf_split)
        test_data = [record for record in ds]

        # Cache locally
        os.makedirs(SPLITS_DIR, exist_ok=True)
        cache_path = os.path.join(SPLITS_DIR, f"{hf_split}.jsonl")
        with open(cache_path, "w", encoding="utf-8") as f:
            for record in test_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Downloaded and cached {len(test_data)} records to {cache_path}")
        return test_data

    print(f"ERROR: Test data not found at {local_path}")
    print("Install 'datasets' for auto-download: pip install datasets")
    print(f"Or manually download from: https://huggingface.co/datasets/{HF_DATASET}")
    sys.exit(1)


# ============================================================
# Utility functions
# ============================================================

def get_valid_sentences(sentences):
    """Filter out short fragments, return list of (index, text) tuples."""
    valid = []
    for i, sent in enumerate(sentences):
        try:
            text = sent["text"].strip() if isinstance(sent, dict) else str(sent).strip()
        except (KeyError, TypeError, AttributeError):
            continue
        if len(text) >= 10:
            valid.append((i, text))
    return valid


def score_rouge(sammendrag, extracted, scorer):
    """Compute ROUGE scores between extracted text and gold sammendrag."""
    if not extracted or not sammendrag:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(sammendrag, extracted)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def select_top_n_in_order(scored_items, n):
    """Take top-N items by score, return them in original document order."""
    top_n = sorted(scored_items, key=lambda x: x[1], reverse=True)[:n]
    top_n.sort(key=lambda x: x[0])
    return " ".join(t[2] for t in top_n)


def aggregate_results(encoder_name, model_id, all_r1, all_r2, all_rl, n_sentences, n_docs):
    """Compute mean and std for ROUGE scores."""
    return {
        "encoder": encoder_name,
        "model_id": model_id,
        "n_sentences": n_sentences,
        "n_documents": n_docs,
        "rouge1": round(np.mean(all_r1) * 100, 2),
        "rouge2": round(np.mean(all_r2) * 100, 2),
        "rougeL": round(np.mean(all_rl) * 100, 2),
        "rouge1_std": round(np.std(all_r1) * 100, 2),
        "rouge2_std": round(np.std(all_r2) * 100, 2),
        "rougeL_std": round(np.std(all_rl) * 100, 2),
    }


def print_single_result(name, results):
    """Print ROUGE results for one model."""
    print(f"\nResults for {name}:")
    print(f"  ROUGE-1: {results['rouge1']:.2f} (±{results['rouge1_std']:.2f})")
    print(f"  ROUGE-2: {results['rouge2']:.2f} (±{results['rouge2_std']:.2f})")
    print(f"  ROUGE-L: {results['rougeL']:.2f} (±{results['rougeL_std']:.2f})")


# ============================================================
# Baseline: Lead-5 (first N valid sentences)
# ============================================================

def run_lead(test_data, n_sentences):
    """Lead-N baseline — selects the first N valid sentences from the document."""
    print(f"\n{'='*60}")
    print(f"Evaluating: Lead-{n_sentences}")
    print(f"{'='*60}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []

    for record in test_data:
        sammendrag = record["sammendrag"]
        valid = get_valid_sentences(record["sentences"])

        # Take first N valid sentences (already in document order)
        lead = valid[:n_sentences]
        extracted = " ".join(text for _, text in lead)

        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    results = aggregate_results(f"Lead-{n_sentences}", f"First {n_sentences} sentences",
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result(f"Lead-{n_sentences}", results)
    return results


# ============================================================
# Baseline: Oracle (greedy ROUGE-1 sentence selection)
# ============================================================

def run_oracle(test_data, n_sentences):
    """Oracle baseline — greedily selects sentences by ROUGE-1 overlap with sammendrag."""
    print(f"\n{'='*60}")
    print(f"Evaluating: Oracle (greedy ROUGE-1 sentence selection)")
    print(f"{'='*60}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []

    for record in test_data:
        sammendrag = record["sammendrag"]
        valid = get_valid_sentences(record["sentences"])

        scored = []
        for idx, text in valid:
            s = scorer.score(sammendrag, text)
            scored.append((idx, s["rouge1"].fmeasure, text))

        extracted = select_top_n_in_order(scored, n_sentences)
        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    results = aggregate_results("Oracle (greedy R-1)", "Greedy ROUGE-1 sentence selection",
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result("Oracle (greedy R-1)", results)
    return results


# ============================================================
# Baseline: Oracle (sequential ROUGE-1 — cumulative coverage)
# ============================================================

def run_oracle_sequential(test_data, n_sentences):
    """Sequential oracle — greedily selects sentences to maximize cumulative ROUGE-1."""
    print(f"\n{'='*60}")
    print(f"Evaluating: Oracle (sequential ROUGE-1 sentence selection)")
    print(f"{'='*60}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []

    for record in test_data:
        sammendrag = record["sammendrag"]
        valid = get_valid_sentences(record["sentences"])

        selected = []
        remaining = list(valid)

        for _ in range(min(n_sentences, len(remaining))):
            best_score = -1
            best_idx = -1
            best_item = None

            for i, (idx, text) in enumerate(remaining):
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

        selected.sort(key=lambda x: x[0])
        extracted = " ".join(text for _, text in selected)

        if not extracted:
            all_r1.append(0.0); all_r2.append(0.0); all_rl.append(0.0)
            continue

        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    results = aggregate_results("Oracle (sequential R-1)", "Sequential greedy ROUGE-1 sentence selection",
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result("Oracle (sequential R-1)", results)
    return results


# ============================================================
# Baseline: BM25
# ============================================================

def bm25_score(query_tokens, doc_tokens, doc_freqs, n_docs, avgdl, k1=1.5, b=0.75):
    """Compute BM25 score for a single document against a query."""
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


def run_bm25(test_data, n_sentences):
    """BM25 sparse retrieval baseline."""
    print(f"\n{'='*60}")
    print(f"Evaluating: BM25")
    print(f"{'='*60}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []
    tokenize = lambda t: re.findall(r'\w+', t.lower())

    for record in test_data:
        sammendrag = record["sammendrag"]
        valid = get_valid_sentences(record["sentences"])
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
            s = bm25_score(query_tokens, all_doc_tokens[i], doc_freqs, n_docs, avgdl)
            scored.append((idx, s, text))

        extracted = select_top_n_in_order(scored, n_sentences)
        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    results = aggregate_results("BM25", "BM25 (k1=1.5, b=0.75)",
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result("BM25", results)
    return results


# ============================================================
# Baseline: TF-IDF
# ============================================================

def run_tfidf(test_data, n_sentences):
    """TF-IDF cosine similarity baseline."""
    print(f"\n{'='*60}")
    print(f"Evaluating: TF-IDF")
    print(f"{'='*60}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []
    tokenize = lambda t: re.findall(r'\w+', t.lower())

    for record in test_data:
        sammendrag = record["sammendrag"]
        valid = get_valid_sentences(record["sentences"])
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

        extracted = select_top_n_in_order(scored, n_sentences)
        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    results = aggregate_results("TF-IDF", "TF-IDF cosine similarity",
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result("TF-IDF", results)
    return results


# ============================================================
# BERT encoder evaluation (mean pooling, max_length=256)
# ============================================================

def mean_pool(model_output, attention_mask):
    """Mean pooling over token embeddings."""
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts(texts, tokenizer, model, device, batch_size=16):
    """Encode texts into normalized embeddings using mean pooling (max_length=256)."""
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


def evaluate_encoder(encoder_name, model_id, test_data, n_sentences):
    """Evaluate a BERT-style encoder model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {encoder_name} ({model_id})")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []

    for idx, record in enumerate(test_data):
        if idx % 20 == 0:
            print(f"  Processing {idx + 1}/{len(test_data)}...")

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

        extracted = select_top_n_in_order(scored, n_sentences)
        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = aggregate_results(encoder_name, model_id,
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result(encoder_name, results)
    return results


# ============================================================
# Sentence-transformer evaluation
# ============================================================

def evaluate_sentence_transformer(st_name, model_id, test_data, n_sentences):
    """Evaluate a sentence-transformers model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {st_name} ({model_id})")
    print(f"{'='*60}")

    model = SentenceTransformer(model_id)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    all_r1, all_r2, all_rl = [], [], []

    for idx, record in enumerate(test_data):
        if idx % 20 == 0:
            print(f"  Processing {idx + 1}/{len(test_data)}...")

        sammendrag = record["sammendrag"]
        sentences = record["sentences"]
        valid = get_valid_sentences(sentences)
        sent_texts = [text for _, text in valid]

        all_texts = [sammendrag] + sent_texts
        embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        sammendrag_emb = embeddings[0]
        sent_embs = embeddings[1:]

        scored = []
        for i, (orig_idx, text) in enumerate(valid):
            sim = np.dot(sammendrag_emb, sent_embs[i])
            scored.append((orig_idx, sim, text))

        extracted = select_top_n_in_order(scored, n_sentences)
        s = score_rouge(sammendrag, extracted, scorer)
        all_r1.append(s["rouge1"]); all_r2.append(s["rouge2"]); all_rl.append(s["rougeL"])

    del model

    results = aggregate_results(st_name, model_id,
                                all_r1, all_r2, all_rl, n_sentences, len(test_data))
    print_single_result(st_name, results)
    return results


# ============================================================
# Results table with published baselines
# ============================================================

def print_comparison_table(new_results):
    """Print new results alongside published baselines."""
    col_w = 36
    print(f"\n{'='*70}")
    print(f"  Published benchmark results (NOR-CASEHOLD v2, test n=233, top-5)")
    print(f"{'='*70}")
    print(f"  {'METHOD':<{col_w}} {'R-1':>8} {'R-2':>8} {'R-L':>8}")
    print(f"  {'-'*(col_w + 26)}")

    for b in PUBLISHED_RESULTS:
        if b['rouge1'] > 0:  # skip placeholder entries
            print(f"  {b['encoder']:<{col_w}} {b['rouge1']:>8.2f} {b['rouge2']:>8.2f} {b['rougeL']:>8.2f}")

    if new_results:
        print(f"  {'-'*(col_w + 26)}")
        print(f"  {'YOUR MODEL':<{col_w}}")
        print(f"  {'-'*(col_w + 26)}")
        for r in new_results:
            if "error" not in r:
                label = f"► {r['encoder']}"
                print(f"  {label:<{col_w}} {r['rouge1']:>8.2f} {r['rouge2']:>8.2f} {r['rougeL']:>8.2f}")
            else:
                label = f"► {r['encoder']}"
                print(f"  {label:<{col_w}} {'ERROR':>8} {'':>8} {'':>8}")

    print(f"  {'='*(col_w + 26)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOR-CASEHOLD Benchmark Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model your-org/your-legal-bert
  python evaluate.py --sentence-transformer your-org/your-st-model
  python evaluate.py --all                              # Run all published baselines
  python evaluate.py --all --no-dense                   # Sparse + structural baselines only

Published baselines: https://github.com/Bendik-Eeg-Henriksen/nor-casehold
        """)
    parser.add_argument("--model", type=str, default=None,
                       help="HuggingFace BERT model to evaluate (e.g. your-org/your-model)")
    parser.add_argument("--sentence-transformer", type=str, default=None,
                       help="Sentence-transformers model to evaluate")
    parser.add_argument("--all", action="store_true",
                       help="Run all published baselines (Lead-5, Oracle, BM25, TF-IDF, + dense models)")
    parser.add_argument("--no-dense", action="store_true",
                       help="Skip dense encoder models (only run Lead-5, BM25, TF-IDF, Oracle)")
    parser.add_argument("--n_sentences", type=int, default=5,
                       help="Number of sentences to extract (default: 5)")
    parser.add_argument("--split", type=str, default="test",
                       help="Which split to evaluate on: train, val, test (default: test)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    args = parser.parse_args()

    # Validate split
    if args.split not in VALID_SPLITS:
        print(f"ERROR: Invalid split '{args.split}'. Must be one of: {', '.join(sorted(VALID_SPLITS))}")
        sys.exit(1)

    # Validate arguments
    if not args.model and not args.sentence_transformer and not args.all:
        parser.print_help()
        print("\nError: Specify --model, --sentence-transformer, or --all")
        sys.exit(1)

    # Load data
    test_data = load_test_data(args.split)
    print(f"Extracting top {args.n_sentences} sentences per document")

    new_results = []

    # --all mode: run published baselines
    if args.all:
        new_results.append(run_lead(test_data, args.n_sentences))
        new_results.append(run_oracle(test_data, args.n_sentences))
        new_results.append(run_oracle_sequential(test_data, args.n_sentences))
        new_results.append(run_bm25(test_data, args.n_sentences))
        new_results.append(run_tfidf(test_data, args.n_sentences))

        if not args.no_dense:
            if HAS_TORCH:
                for enc_name, model_id in DEFAULT_ENCODERS:
                    try:
                        result = evaluate_encoder(enc_name, model_id, test_data, args.n_sentences)
                        new_results.append(result)
                    except Exception as e:
                        print(f"ERROR evaluating {enc_name}: {e}")
                        new_results.append({"encoder": enc_name, "model_id": model_id, "error": str(e)})
            else:
                print("\nSkipping dense encoder models (torch not available)")
                print("Install with: pip install torch transformers")

            if HAS_SENTENCE_TRANSFORMERS:
                for st_name, model_id in DEFAULT_ST_MODELS:
                    try:
                        result = evaluate_sentence_transformer(st_name, model_id, test_data, args.n_sentences)
                        new_results.append(result)
                    except Exception as e:
                        print(f"ERROR evaluating {st_name}: {e}")
                        new_results.append({"encoder": st_name, "model_id": model_id, "error": str(e)})
            else:
                print("\nSkipping sentence-transformers (not installed)")
                print("Install with: pip install sentence-transformers")

    # --model mode: evaluate custom BERT encoder
    if args.model:
        if not HAS_TORCH:
            print("ERROR: --model requires torch and transformers.")
            print("Install with: pip install torch transformers")
            sys.exit(1)
        name = args.model.split("/")[-1]
        result = evaluate_encoder(name, args.model, test_data, args.n_sentences)
        new_results.append(result)

    # --sentence-transformer mode: evaluate custom ST model
    if args.sentence_transformer:
        if not HAS_SENTENCE_TRANSFORMERS:
            print("ERROR: --sentence-transformer requires sentence-transformers.")
            print("Install with: pip install sentence-transformers")
            sys.exit(1)
        name = args.sentence_transformer.split("/")[-1]
        result = evaluate_sentence_transformer(
            name, args.sentence_transformer, test_data, args.n_sentences)
        new_results.append(result)

    # Print results
    if args.all:
        col_w = 36
        print(f"\n{'='*70}")
        print(f"  NOR-CASEHOLD Results (current run, {args.split} split, top-{args.n_sentences})")
        print(f"{'='*70}")
        print(f"  {'METHOD':<{col_w}} {'R-1':>8} {'R-2':>8} {'R-L':>8}")
        print(f"  {'-'*(col_w + 26)}")
        for r in new_results:
            if "error" not in r:
                print(f"  {r['encoder']:<{col_w}} {r['rouge1']:>8.2f} {r['rouge2']:>8.2f} {r['rougeL']:>8.2f}")
            else:
                print(f"  {r['encoder']:<{col_w}} {'ERROR':>8} {'':>8} {'':>8}")
        print(f"  {'='*(col_w + 26)}")
    else:
        print_comparison_table(new_results)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(new_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
