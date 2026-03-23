# Nor-CaseHOLD

**A Retrieval Benchmark for Norwegian Legal AI**

NOR-CASEHOLD is an extractive legal retrieval benchmark built from 1,244 Norwegian legal documents — 627 Supreme Court (Høyesterett) decisions and 617 Skatteetaten bindende forhåndsuttalelser (BFU) — each paired with its official summary. To the author's knowledge, this is the first open-source legal retrieval benchmark for Norwegian.

The benchmark evaluates how well retrieval models can identify the sentences in a legal document that best match the official summary. Nine methods are evaluated with bootstrap confidence intervals and pairwise significance testing.

**Companion model:** [Norwegian Legal BERT](https://huggingface.co/bendik-eeg-henriksen/norwegian-legal-bert) — the first open-source domain-adapted Norwegian legal language model.

📄 **Technical Report:** [NOR-CASEHOLD: A Benchmark for Evaluating Norwegian Legal AI Systems](link-to-ssrn)

## Benchmark Results

**Test set: n=233 (117 Høyesterett, 116 BFU), top-5 sentence extraction, 95% bootstrap CIs from 10,000 samples.**

| Method | Type | ROUGE-1 | 95% CI | ROUGE-2 | ROUGE-L |
|--------|------|---------|--------|---------|---------|
| Oracle (sequential R-1) | Oracle | **55.87** | [54.67, 57.12] | 30.59 | 34.68 |
| TF-IDF | Sparse | 47.85 | [46.50, 49.23] | 26.46 | 30.99 |
| BM25 (k1=1.5, b=0.75) | Sparse | 47.49 | [46.16, 48.82] | 26.21 | 30.64 |
| Oracle (greedy R-1) | Oracle | 42.68 | [41.52, 43.85] | 20.65 | 25.30 |
| Norwegian Legal BERT | Dense | **38.40** | [37.19, 39.64] | 15.97 | 20.93 |
| MiniLM (multilingual ST) | Dense (ST) | 37.47 | [36.28, 38.73] | 16.03 | 21.54 |
| mBERT | Dense | 37.34 | [36.28, 38.41] | 15.14 | 20.49 |
| NB-BERT-base | Dense | 37.28 | [36.18, 38.35] | 15.51 | 20.64 |
| Lead-5 | Structural | 25.07 | [23.93, 26.21] | 7.54 | 13.44 |

### Pairwise Significance Tests (ROUGE-1)

| Comparison | Δ ROUGE-1 | p-value | Sig. |
|------------|-----------|---------|------|
| BM25 vs TF-IDF | -0.36 | 0.971 | n.s. |
| BM25 vs Oracle (greedy) | +4.81 | < 0.001 | *** |
| Oracle (sequential) vs BM25 | +8.38 | < 0.001 | *** |
| Norwegian Legal BERT vs NB-BERT-base | +1.12 | < 0.001 | *** |
| Norwegian Legal BERT vs mBERT | +1.06 | 0.001 | ** |
| Norwegian Legal BERT vs MiniLM | +0.92 | 0.055 | n.s. |
| NB-BERT-base vs mBERT | -0.05 | 0.563 | n.s. |

### Per-Source Breakdown (BM25 ROUGE-1)

| Source | n | ROUGE-1 |
|--------|---|---------|
| Høyesterett | 117 | 48.40 |
| Skatteetaten BFU | 116 | 46.57 |
| Combined | 233 | 47.49 |

## Key Findings

- **Sparse retrieval dominates.** BM25 and TF-IDF outperform all dense encoders, consistent with BM25's known strength on legal retrieval tasks (Rosa et al., 2021).
- **Domain pretraining helps.** Norwegian Legal BERT significantly outperforms NB-BERT-base (p < 0.001) and mBERT (p = 0.001) among dense encoders.
- **Rankings hold across sources.** Method rankings are consistent across both Høyesterett and BFU data.
- **Hybrid and reranking don't beat BM25.** Neither hybrid BM25+Legal BERT scoring nor cross-encoder reranking improved over pure BM25.

## Dataset

| Split | Høyesterett | BFU | Total |
|-------|-------------|-----|-------|
| Train | 438 | 431 | 869 |
| Validation | 72 | 70 | 142 |
| Test | 117 | 116 | 233 |
| **Total** | **627** | **617** | **1,244** |

**HuggingFace dataset:** [bendik-eeg-henriksen/nor-casehold](https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold)

## Quick Start

### Evaluate any HuggingFace encoder

```bash
pip install rouge-score transformers torch sentence-transformers
python evaluate.py --model your-model-name --test-path data/splits/test.jsonl
```

### Run full significance testing

```bash
python significance_full.py --output results/significance.json
```

## Repository Structure

```
nor-casehold/
├── data/
│   └── splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── results/
│   ├── significance_v2.json
│   ├── hybrid_results.json
│   └── reranker_results.json
├── evaluate.py              # Single-model evaluation
├── significance_full.py     # Full benchmark with CIs
├── hybrid_baseline.py       # Hybrid BM25+dense scoring
├── cross_encoder_reranker.py # Cross-encoder reranking pipeline
└── README.md
```

## How to Add Your Model

1. Upload your model to HuggingFace
2. Run: `python evaluate.py --model your-org/your-model --test-path data/splits/test.jsonl`
3. Compare against the baselines in the table above

### Sample Output

```
Loading model: bendik-eeg-henriksen/norwegian-legal-bert
Evaluating 233 documents...
  20/233...
  40/233...
  ...
  220/233...

Results (top-5, n=233):
  ROUGE-1: 38.40
  ROUGE-2: 15.97
  ROUGE-L: 20.93
```

## Reproducibility

- **Random seed:** 42
- **Bootstrap samples:** 10,000
- **Hardware:** NVIDIA T4 GPU (Google Colab)
- **Dense encoder pooling:** Mean pooling, max_length=256
- **Framework:** HuggingFace Transformers

## Version History

- **v2 (March 2026):** Expanded from 627 to 1,244 documents with Skatteetaten BFU data. Stratified splits by source. Re-ran all baselines on 233-doc test set. Added hybrid and cross-encoder experiments.
- **v1 (March 2026):** Initial release with 627 Høyesterett decisions and 8 baselines.

## Citation

```
Eeg-Henriksen, B. (2026). NOR-CASEHOLD: A Benchmark for Evaluating
Norwegian Legal AI Systems. Technical Report.
```

## License

- **Dataset:** CC-BY-4.0
- **Code:** Apache 2.0
- **Model (Norwegian Legal BERT):** Apache 2.0

## Links

- **Dataset:** [huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold](https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold)
- **Model:** [huggingface.co/bendik-eeg-henriksen/norwegian-legal-bert](https://huggingface.co/bendik-eeg-henriksen/norwegian-legal-bert)
- **Author:** [huggingface.co/bendik-eeg-henriksen](https://huggingface.co/bendik-eeg-henriksen)
