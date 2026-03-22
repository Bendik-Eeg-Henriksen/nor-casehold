#!/usr/bin/env python3
"""
NOR-CASEHOLD: BM25 + Cross-Encoder Reranker
=============================================
Fine-tunes Norwegian Legal BERT as a cross-encoder reranker.

Pipeline:
  1. BM25 retrieves top-K candidate sentences per document
  2. Cross-encoder scores each (sammendrag, sentence) pair
  3. Top-5 reranked sentences are selected
  4. Evaluate with ROUGE

Training:
  - Positives: oracle top-5 sentences (highest individual ROUGE-1)
  - Hard negatives: BM25 top-20 sentences that are NOT in oracle top-5
  - Random negatives: 5 random sentences not in BM25 top-20 or oracle
  - Binary classification: is this sentence extractive-summary-worthy?

Usage:
    # Train + evaluate
    python cross_encoder_reranker.py \
        --train-path data/v2/splits/train.jsonl \
        --val-path data/v2/splits/val.jsonl \
        --test-path data/v2/splits/test.jsonl \
        --output results/reranker_results.json

    # Evaluate only (with saved model)
    python cross_encoder_reranker.py \
        --test-path data/v2/splits/test.jsonl \
        --model-dir reranker_model/ \
        --eval-only \
        --output results/reranker_results.json
"""

import json
import os
import re
import math
import random
import argparse
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

N_BOOTSTRAP = 10000
CI_LEVEL = 95


# ============================================================
# Utilities
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
    """scored_items: list of (orig_idx, score, text)"""
    top_n = sorted(scored_items, key=lambda x: x[1], reverse=True)[:n]
    top_n.sort(key=lambda x: x[0])
    return " ".join(t[2] for t in top_n)


# ============================================================
# BM25
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


def get_bm25_top_k(record, k=20):
    """Returns list of (sentence_idx, bm25_score, text) sorted by score desc."""
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

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def get_oracle_top_n(record, n, rouge_scorer_obj):
    """Returns set of sentence indices for oracle top-N (greedy individual ROUGE-1)."""
    sammendrag = record["sammendrag"]
    valid = get_valid_sentences(record["sentences"])
    scored = []
    for idx, text in valid:
        s = rouge_scorer_obj.score(sammendrag, text)
        scored.append((idx, s["rouge1"].fmeasure))
    scored.sort(key=lambda x: x[1], reverse=True)
    return set(idx for idx, _ in scored[:n])


# ============================================================
# Training data construction
# ============================================================

def build_training_pairs(records, scorer, bm25_k=20, oracle_n=5, n_random_neg=5):
    """
    For each document, build (sammendrag, sentence, label) pairs.
    
    Positives: oracle top-5 sentences
    Hard negatives: BM25 top-K that are NOT in oracle top-5
    Random negatives: sentences not in BM25 top-K or oracle
    """
    all_pairs = []
    
    for doc_idx, record in enumerate(records):
        if (doc_idx + 1) % 100 == 0:
            print(f"  Building pairs: {doc_idx+1}/{len(records)}...")
        
        sammendrag = record["sammendrag"]
        valid = get_valid_sentences(record["sentences"])
        
        if len(valid) < 10:
            continue
        
        # Get oracle positives
        oracle_indices = get_oracle_top_n(record, oracle_n, scorer)
        
        # Get BM25 candidates
        bm25_candidates = get_bm25_top_k(record, bm25_k)
        bm25_indices = set(idx for idx, _, _ in bm25_candidates)
        
        # Build index -> text mapping
        idx_to_text = {idx: text for idx, text in valid}
        all_indices = set(idx for idx, _ in valid)
        
        # Positives: oracle sentences
        for idx in oracle_indices:
            if idx in idx_to_text:
                all_pairs.append((sammendrag, idx_to_text[idx], 1))
        
        # Hard negatives: BM25 top-K minus oracle
        hard_neg_indices = bm25_indices - oracle_indices
        for idx in hard_neg_indices:
            if idx in idx_to_text:
                all_pairs.append((sammendrag, idx_to_text[idx], 0))
        
        # Random negatives: not in BM25 top-K and not in oracle
        remaining = list(all_indices - bm25_indices - oracle_indices)
        if remaining:
            n_rand = min(n_random_neg, len(remaining))
            for idx in random.sample(remaining, n_rand):
                all_pairs.append((sammendrag, idx_to_text[idx], 0))
    
    return all_pairs


class RerankerDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sammendrag, sentence, label = self.pairs[idx]
        encoding = self.tokenizer(
            sammendrag, sentence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Training
# ============================================================

def train_reranker(train_pairs, val_pairs, model_id, output_dir,
                   epochs=3, batch_size=16, lr=2e-5, max_length=512):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2
    ).to(device)
    
    # Datasets
    train_dataset = RerankerDataset(train_pairs, tokenizer, max_length)
    val_dataset = RerankerDataset(val_pairs, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Class weights (positives are rarer)
    n_pos = sum(1 for _, _, l in train_pairs if l == 1)
    n_neg = sum(1 for _, _, l in train_pairs if l == 0)
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, min(pos_weight, 5.0)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\nTraining: {len(train_pairs)} pairs ({n_pos} pos, {n_neg} neg)")
    print(f"Validation: {len(val_pairs)} pairs")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Class weight for positives: {class_weights[1]:.2f}")
    print()
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={total_loss/(batch_idx+1):.4f} acc={correct/total:.3f}")
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")
    
    print(f"\nBest epoch: {best_epoch} (val_acc={best_val_acc:.3f})")
    
    # Clean up
    del model, optimizer
    torch.cuda.empty_cache()
    
    return output_dir


# ============================================================
# Evaluation: BM25 + Cross-encoder reranking
# ============================================================

def evaluate_reranker(test_data, model_dir, bm25_k=20, top_n=5,
                      n_bootstrap=N_BOOTSTRAP, max_length=512):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    
    all_results = []
    per_source = {}
    
    # Also compute pure BM25 for direct comparison
    bm25_results = []
    
    for doc_idx, record in enumerate(test_data):
        if (doc_idx + 1) % 20 == 0:
            print(f"  Evaluating: {doc_idx+1}/{len(test_data)}...")
        
        sammendrag = record["sammendrag"]
        
        # Step 1: BM25 retrieves top-K candidates
        bm25_candidates = get_bm25_top_k(record, bm25_k)
        
        # Pure BM25 baseline (top-5 from BM25 directly)
        bm25_top5 = bm25_candidates[:top_n]
        bm25_top5.sort(key=lambda x: x[0])
        bm25_extracted = " ".join(text for _, _, text in bm25_top5)
        bm25_scores = scorer.score(sammendrag, bm25_extracted)
        bm25_results.append({k: bm25_scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]})
        
        # Step 2: Cross-encoder reranks the BM25 candidates
        rerank_scores = []
        
        for sent_idx, bm25_score, text in bm25_candidates:
            encoding = tokenizer(
                sammendrag, text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                output = model(**encoding)
                # Use softmax probability of positive class as score
                probs = torch.softmax(output.logits, dim=-1)
                pos_prob = probs[0, 1].item()
            
            rerank_scores.append((sent_idx, pos_prob, text))
        
        # Step 3: Select top-5 after reranking, restore document order
        extracted = select_top_n_in_order(rerank_scores, top_n)
        
        if not extracted:
            result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        else:
            scores = scorer.score(sammendrag, extracted)
            result = {k: scores[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}
        
        all_results.append(result)
        
        # Per-source tracking
        src = record.get("source", "unknown")
        if src not in per_source:
            per_source[src] = {"rouge1": [], "rouge2": [], "rougeL": []}
        per_source[src]["rouge1"].append(result["rouge1"])
        per_source[src]["rouge2"].append(result["rouge2"])
        per_source[src]["rougeL"].append(result["rougeL"])
    
    return all_results, bm25_results, per_source


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
    parser = argparse.ArgumentParser(description="BM25 + Cross-Encoder Reranker")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--model-id", type=str,
                       default="bendik-eeg-henriksen/norwegian-legal-bert")
    parser.add_argument("--model-dir", type=str, default="reranker_model")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--oracle-n", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    
    # ---- Training ----
    if not args.eval_only:
        if args.train_path is None or args.val_path is None:
            print("ERROR: Need --train-path and --val-path for training")
            return
        
        train_data = load_jsonl(args.train_path)
        val_data = load_jsonl(args.val_path)
        print(f"Train: {len(train_data)} docs, Val: {len(val_data)} docs")
        
        # Build training pairs
        print("\nBuilding training pairs (with BM25 hard negatives)...")
        train_pairs = build_training_pairs(
            train_data, scorer, bm25_k=args.bm25_k,
            oracle_n=args.oracle_n, n_random_neg=5
        )
        
        print(f"\nBuilding validation pairs...")
        val_pairs = build_training_pairs(
            val_data, scorer, bm25_k=args.bm25_k,
            oracle_n=args.oracle_n, n_random_neg=5
        )
        
        # Train
        print(f"\n{'='*60}")
        print(f"  Training cross-encoder reranker")
        print(f"{'='*60}")
        
        train_reranker(
            train_pairs, val_pairs, args.model_id, args.model_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, max_length=args.max_length
        )
        
        # Free memory
        del train_data, val_data, train_pairs, val_pairs
        torch.cuda.empty_cache()
    
    # ---- Evaluation ----
    test_data = load_jsonl(args.test_path)
    print(f"\n{'='*60}")
    print(f"  Evaluating: BM25 top-{args.bm25_k} → Cross-encoder rerank → top-5")
    print(f"  Test docs: {len(test_data)}")
    print(f"{'='*60}")
    
    reranker_results, bm25_results, per_source = evaluate_reranker(
        test_data, args.model_dir, bm25_k=args.bm25_k,
        max_length=args.max_length, n_bootstrap=args.n_bootstrap
    )
    
    # Compute stats
    rr_r1 = [s["rouge1"] for s in reranker_results]
    rr_r2 = [s["rouge2"] for s in reranker_results]
    rr_rl = [s["rougeL"] for s in reranker_results]
    
    bm_r1 = [s["rouge1"] for s in bm25_results]
    
    rr_r1_mean = np.mean(rr_r1) * 100
    rr_r2_mean = np.mean(rr_r2) * 100
    rr_rl_mean = np.mean(rr_rl) * 100
    bm_r1_mean = np.mean(bm_r1) * 100
    
    rr_r1_ci = bootstrap_ci(rr_r1, args.n_bootstrap)
    rr_r2_ci = bootstrap_ci(rr_r2, args.n_bootstrap)
    rr_rl_ci = bootstrap_ci(rr_rl, args.n_bootstrap)
    
    # Pairwise test: reranker vs BM25
    p_val = pairwise_bootstrap_test(rr_r1, bm_r1, args.n_bootstrap)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    delta = rr_r1_mean - bm_r1_mean
    
    # Print
    print(f"\n{'='*70}")
    print(f"  BM25 + Legal BERT Cross-Encoder Reranker")
    print(f"  BM25 top-{args.bm25_k} → rerank → top-5  |  n = {len(test_data)}")
    print(f"{'='*70}")
    print(f"  Reranker  ROUGE-1: {rr_r1_mean:.2f} [{rr_r1_ci[0]*100:.2f}, {rr_r1_ci[1]*100:.2f}]")
    print(f"  Reranker  ROUGE-2: {rr_r2_mean:.2f} [{rr_r2_ci[0]*100:.2f}, {rr_r2_ci[1]*100:.2f}]")
    print(f"  Reranker  ROUGE-L: {rr_rl_mean:.2f} [{rr_rl_ci[0]*100:.2f}, {rr_rl_ci[1]*100:.2f}]")
    print(f"")
    print(f"  BM25      ROUGE-1: {bm_r1_mean:.2f}")
    print(f"  Δ ROUGE-1: {delta:+.2f}  p = {p_val:.4f}  {sig}")
    
    if len(per_source) > 1:
        print(f"\n  Per-source breakdown (reranker):")
        for src in sorted(per_source):
            src_r1 = np.mean(per_source[src]["rouge1"]) * 100
            src_n = len(per_source[src]["rouge1"])
            print(f"    {src} (n={src_n}): ROUGE-1 = {src_r1:.2f}")
    
    print(f"{'='*70}")
    
    # Save
    if args.output:
        output_data = {
            "method": "BM25 + Legal BERT Cross-Encoder Reranker",
            "model": args.model_id,
            "bm25_k": args.bm25_k,
            "top_n": 5,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_documents": len(test_data),
            "n_bootstrap": args.n_bootstrap,
            "reranker": {
                "rouge1": {"mean": round(rr_r1_mean, 2),
                          "ci_lower": round(rr_r1_ci[0]*100, 2),
                          "ci_upper": round(rr_r1_ci[1]*100, 2)},
                "rouge2": {"mean": round(rr_r2_mean, 2),
                          "ci_lower": round(rr_r2_ci[0]*100, 2),
                          "ci_upper": round(rr_r2_ci[1]*100, 2)},
                "rougeL": {"mean": round(rr_rl_mean, 2),
                          "ci_lower": round(rr_rl_ci[0]*100, 2),
                          "ci_upper": round(rr_rl_ci[1]*100, 2)},
            },
            "bm25_baseline": {
                "rouge1_mean": round(bm_r1_mean, 2),
            },
            "comparison": {
                "delta_r1": round(delta, 2),
                "p_value": round(p_val, 4),
                "significant": sig,
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
