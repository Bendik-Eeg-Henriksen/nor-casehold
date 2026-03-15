#!/usr/bin/env python3
"""
NOR-CASEHOLD Benchmark Evaluation
===================================
Reproduces the benchmark results from the NOR-CASEHOLD dataset card.

Given an encoder model, this script:
1. Loads the NOR-CASEHOLD test split from HuggingFace
2. Encodes all sentences and the gold sammendrag using mean-pooled [CLS] embeddings
3. Ranks sentences by cosine similarity to the sammendrag embedding
4. Selects the top-k sentences (excluding short fragments)
5. Concatenates them in original document order
6. Scores the extracted summary against the gold sammendrag using ROUGE

Usage:
    # Evaluate Norwegian Legal BERT (default)
    python benchmark.py

    # Evaluate a specific model
    python benchmark.py --model NbAiLab/nb-bert-base

    # Evaluate all three baselines + oracle
    python benchmark.py --all

    # Change top-k
    python benchmark.py --top-k 3

Requirements:
    pip install datasets transformers torch rouge-score numpy
"""

import argparse
import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer


# === Embedding ===

def mean_pool(model_output, attention_mask):
    """Mean pooling over token embeddings, respecting attention mask."""
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counted = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counted


def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=512):
    """Encode a list of texts into embeddings using mean pooling."""
    all_embeddings = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        embeddings = mean_pool(output, encoded["attention_mask"])
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# === Sentence selection ===

def select_top_k_sentences(sentences, sammendrag_emb, sentence_embs, k=5):
    """
    Select top-k sentences by cosine similarity to the sammendrag embedding.
    Excludes sentences flagged as short fragments.
    Returns selected sentences in original document order.
    """
    # Filter to scorable sentences
    scorable = [
        (i, s)
        for i, s in enumerate(sentences)
        if not s.get("is_short_fragment", False)
    ]

    if not scorable:
        return []

    indices, sents = zip(*scorable)
    indices = list(indices)

    # Cosine similarity (embeddings are already L2-normalized)
    embs = sentence_embs[indices]
    similarities = embs @ sammendrag_emb.T
    similarities = similarities.flatten()

    # Top-k by similarity
    top_k_positions = np.argsort(similarities)[::-1][:k]
    selected_indices = sorted([indices[p] for p in top_k_positions])

    return [sentences[i]["text"] for i in selected_indices]


def oracle_top_k_sentences(sentences, k=5):
    """
    Oracle baseline: select top-k sentences by pre-computed ROUGE hm_score.
    Returns selected sentences in original document order.
    """
    scorable = [
        s for s in sentences if not s.get("is_short_fragment", False)
    ]

    ranked = sorted(scorable, key=lambda s: s["hm_score"], reverse=True)[:k]
    # Restore document order
    ranked_by_idx = sorted(ranked, key=lambda s: s["sentence_idx"])

    return [s["text"] for s in ranked_by_idx]


# === ROUGE evaluation ===

def evaluate_rouge(predictions, references):
    """
    Compute mean ROUGE-1, ROUGE-2, ROUGE-L F-measure across document pairs.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {key: np.mean(vals) * 100 for key, vals in scores.items()}


# === Main ===

def run_model_evaluation(model_name, dataset, top_k, device):
    """Run full evaluation pipeline for a single model."""
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    predictions = []
    references = []

    for idx, example in enumerate(dataset):
        sentences = example["sentences"]
        sammendrag = example["sammendrag"]

        # Collect all sentence texts
        sentence_texts = [s["text"] for s in sentences]

        # Encode sammendrag + all sentences
        all_texts = [sammendrag] + sentence_texts
        all_embs = encode_texts(all_texts, tokenizer, model, device)

        sammendrag_emb = all_embs[0:1]
        sentence_embs = all_embs[1:]

        # Select top-k
        selected = select_top_k_sentences(
            sentences, sammendrag_emb, sentence_embs, k=top_k
        )

        predictions.append(" ".join(selected))
        references.append(sammendrag)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} documents")

    scores = evaluate_rouge(predictions, references)
    return scores


def run_oracle_evaluation(dataset, top_k):
    """Run oracle baseline evaluation."""
    print("\nRunning oracle baseline...")

    predictions = []
    references = []

    for example in dataset:
        selected = oracle_top_k_sentences(example["sentences"], k=top_k)
        predictions.append(" ".join(selected))
        references.append(example["sammendrag"])

    return evaluate_rouge(predictions, references)


def main():
    parser = argparse.ArgumentParser(
        description="NOR-CASEHOLD Benchmark Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bendik-eeg-henriksen/norwegian-legal-bert",
        help="HuggingFace model ID to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all baselines (Norwegian Legal BERT, nb-bert-base, mBERT) + oracle",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run oracle baseline only",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sentences to extract (default: 5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading NOR-CASEHOLD dataset...")
    dataset = load_dataset("bendik-eeg-henriksen/nor-casehold", split=args.split)
    print(f"Loaded {len(dataset)} documents from '{args.split}' split")

    results = {}

    if args.oracle:
        scores = run_oracle_evaluation(dataset, args.top_k)
        results["Oracle"] = scores
    elif args.all:
        models = [
            ("Norwegian Legal BERT", "bendik-eeg-henriksen/norwegian-legal-bert"),
            ("NB-BERT-base", "NbAiLab/nb-bert-base"),
            ("mBERT", "bert-base-multilingual-cased"),
        ]
        for display_name, model_id in models:
            scores = run_model_evaluation(model_id, dataset, args.top_k, device)
            results[display_name] = scores

        oracle_scores = run_oracle_evaluation(dataset, args.top_k)
        results["Oracle"] = oracle_scores
    else:
        scores = run_model_evaluation(args.model, dataset, args.top_k, device)
        results[args.model] = scores

    # Print results
    print(f"\n{'=' * 62}")
    print(f"  NOR-CASEHOLD Results (top-{args.top_k}, {args.split} split, n={len(dataset)})")
    print(f"{'=' * 62}")
    print(f"  {'Encoder':<35} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
    print(f"  {'-' * 35} {'-' * 7} {'-' * 7} {'-' * 7}")

    for name, scores in results.items():
        print(
            f"  {name:<35} {scores['rouge1']:>7.2f} {scores['rouge2']:>7.2f} {scores['rougeL']:>7.2f}"
        )

    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
