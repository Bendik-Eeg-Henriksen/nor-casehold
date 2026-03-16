# NOR-CASEHOLD

**A benchmark dataset for evaluating Norwegian legal language models.**

## Overview

NOR-CASEHOLD is a benchmark dataset for evaluating legal language models on Norwegian court decisions. It contains **627 Norwegian Supreme Court (*Høyesterett*) decisions** paired with their official ***sammendrag*** — expert-written summaries published by the court that capture the legal holding and key reasoning of each case.

Unlike the original multiple-choice [CaseHOLD](https://github.com/reglab/casehold) formulation designed for US common law citation patterns, NOR-CASEHOLD uses **extractive sentence selection** evaluated with ROUGE, following the [ITA-CASEHOLD](https://github.com/dlicari/ITA-CASEHOLD) methodology adapted for civil law jurisdictions.

Also available on [HuggingFace](https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold).

## Why this exists

There was previously **no benchmark dataset for evaluating Norwegian legal NLP models**. English has [CaseHOLD](https://github.com/reglab/casehold) (Zheng et al., ICAIL 2021) and [LexGLUE](https://huggingface.co/datasets/lex_glue). Italian has [ITA-CASEHOLD](https://github.com/dlicari/ITA-CASEHOLD) (Licari et al., ICAIL 2023). Norwegian had nothing.

NOR-CASEHOLD fills this gap, enabling researchers and practitioners to measure whether domain-adapted models actually outperform general models on Norwegian legal text.

## Task

Given a full-text Høyesterett decision, select the sentences that best capture the substance of the official *sammendrag*.

This is an **extractive legal summarization** task. Models encode all sentences and the sammendrag into embeddings, rank sentences by cosine similarity to the sammendrag embedding, and select the top-N sentences. The extracted summary is then scored against the gold-standard *sammendrag* using ROUGE (R-1, R-2, R-L).

The oracle baseline uses direct ROUGE-1 overlap for sentence selection and serves as an approximate upper bound for extractive retrieval quality.

## Benchmark Results

Results on the test set (n=96), extracting top 5 sentences per document:

| Encoder | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| **[Norwegian Legal BERT](https://huggingface.co/bendik-eeg-henriksen/norwegian-legal-bert)** | **41.51** | **13.46** | **19.47** |
| [NbAiLab/nb-bert-base](https://huggingface.co/NbAiLab/nb-bert-base) | 38.98 | 12.12 | 18.21 |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) | 39.01 | 11.71 | 18.21 |
| Oracle baseline | 47.93 | 25.89 | 28.78 |

**Key finding:** [Norwegian Legal BERT](https://huggingface.co/bendik-eeg-henriksen/norwegian-legal-bert) outperforms both general-purpose baselines consistently across all three ROUGE metrics, demonstrating that legal domain pretraining improves extractive holding extraction for Norwegian court decisions.

## Dataset Structure

### Splits

| Split | Documents |
|---|---|
| Train | 440 |
| Validation | 91 |
| Test | 96 |
| **Total** | **627** |

### Fields

| Field | Description |
|---|---|
| `case_id` | Høyesterett case identifier (e.g., HR-2023-299-A) |
| `title` | Case title |
| `date` | Decision date |
| `category` | Case category (civil/criminal) |
| `rettsomrade` | Legal area |
| `source_url` | URL to original decision on domstol.no |
| `sammendrag` | Official summary/holding (gold standard) |
| `full_text` | Full text of the court decision |
| `sentences` | Individual sentences with per-sentence ROUGE scores against the sammendrag |
| `compression_ratio` | Ratio of sammendrag length to full text length |

### Pre-computed sentence scores

Each sentence in the `sentences` array includes:

| Field | Description |
|---|---|
| `rouge1_f` | ROUGE-1 F-measure against the sammendrag |
| `rouge2_f` | ROUGE-2 F-measure against the sammendrag |
| `hm_score` | Harmonic mean of R-1 and R-2 (training target, following ITA-CASEHOLD) |
| `is_short_fragment` | Boolean flag for sentence-splitting artifacts to exclude during selection |

## Statistics

| Metric | Value |
|---|---|
| Total decision-sammendrag pairs | 627 |
| Mean full text length | 3,887 words |
| Mean sammendrag length | 246 words |
| Mean compression ratio | 8.5% |
| Mean sentences per document | 218.2 |
| Total sentences | 136,805 |
| Scorable sentences | 128,843 |
| Years covered | Primarily 2004–2026 |

## Lineage

| Dataset | Language | Legal system | Task | Size | Published |
|---|---|---|---|---|---|
| [CaseHOLD](https://github.com/reglab/casehold) | English | Common law | Multiple-choice QA | 53,000 | ICAIL 2021 |
| [ITA-CASEHOLD](https://github.com/dlicari/ITA-CASEHOLD) | Italian | Civil law | Extractive summarization | 1,101 | ICAIL 2023 |
| **NOR-CASEHOLD** | **Norwegian** | **Civil law** | **Extractive summarization** | **627** | **2026** |

## Usage

```python
# Load from HuggingFace (recommended)
from datasets import load_dataset

dataset = load_dataset("bendik-eeg-henriksen/nor-casehold")

train = dataset["train"]
test = dataset["test"]

example = test[0]
print(example["case_id"])
print(example["sammendrag"][:200])

# Get top-scoring sentences for a document
sentences = example["sentences"]
top = sorted(
    [s for s in sentences if not s.get("is_short_fragment")],
    key=lambda x: x["hm_score"],
    reverse=True
)[:5]
for s in top:
    print(f"[{s['hm_score']:.4f}] {s['text'][:100]}...")
```

Or clone this repo and load directly:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl",
    "test": "data/test.jsonl"
})
```

## Running the Benchmark

```bash
pip install -r requirements.txt

# Evaluate Norwegian Legal BERT (default)
python benchmark.py

# Evaluate all three baselines + oracle
python benchmark.py --all

# Evaluate a specific model
python benchmark.py --model NbAiLab/nb-bert-base

# Change top-k
python benchmark.py --top-k 3
```

## Repository Structure

```
nor-casehold/
├── README.md
├── LICENSE
├── requirements.txt
├── benchmark.py                  # Benchmark evaluation script
├── benchmark_results_final.txt   # Full benchmark output (all models + oracle)
├── data/
│   ├── train.jsonl               # 440 documents (Git LFS)
│   ├── val.jsonl                 # 91 documents (Git LFS)
│   └── test.jsonl                # 96 documents (Git LFS)
└── scripts/
    ├── cleanup_nor_casehold.py   # Data cleaning pipeline
    └── cleanup_report.json       # Cleanup run report
```

## Methodology

Following ITA-CASEHOLD (Licari et al., ICAIL 2023):

1. Full-text decisions are split into sentences
2. Per-sentence ROUGE-1 and ROUGE-2 F-measure scores are computed against the gold *sammendrag*
3. The harmonic mean of R-1 and R-2 serves as the target relevance score
4. Models are evaluated by their ability to select the most relevant sentences using embedding cosine similarity
5. Extracted summaries (top-N sentences in original document order) are scored against the gold *sammendrag* using ROUGE

A sample of test examples was manually validated by a legal professional to confirm that high-scoring sentences capture substantive legal reasoning rather than procedural boilerplate.

### Data cleaning (v1.1)

- **15 *ankeutvalg* decisions removed** whose *sammendrag* contained only judge names or party listings
- **Sentence-splitting artifacts flagged** via `is_short_fragment` (~5.8% of sentences)
- **Embedded newlines stripped** from sentence text

## Data Source

All data sourced from publicly accessible Norwegian government websites:

- **[domstol.no](https://www.domstol.no/no/hoyesterett/avgjorelser/)** — Høyesterett decisions and *sammendrag*

The underlying court decisions are public domain under Norwegian law. No Lovdata content was used.

## Limitations

- **Size:** 627 examples from Høyesterett only. The current size reflects the total available pool of decisions with published *sammendrag* from public sources.
- **Extractive only:** Does not capture abstractive aspects of legal summarization.
- **Task format:** Uses extractive summarization rather than the original CaseHOLD multiple-choice format, reflecting the different data available in Norwegian civil law.

## Citation

```bibtex
@misc{eeg-henriksen2026norcasehold,
  title={NOR-CASEHOLD: A Norwegian Legal Benchmark for Extractive Holding Extraction},
  author={Eeg-Henriksen, Jan Bendik},
  year={2026},
  url={https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold}
}
```

## License

Dataset: [CC BY 4.0](LICENSE)  
Code: MIT

## Related Resources

- **[Norwegian Legal BERT](https://huggingface.co/bendik-eeg-henriksen/norwegian-legal-bert)** — Domain-adapted BERT for Norwegian legal text
- **[NbAiLab/nb-bert-base](https://huggingface.co/NbAiLab/nb-bert-base)** — Base Norwegian BERT model
- **[CaseHOLD](https://github.com/reglab/casehold)** — English legal benchmark (Zheng et al., ICAIL 2021)
- **[ITA-CASEHOLD](https://github.com/dlicari/ITA-CASEHOLD)** — Italian legal benchmark (Licari et al., ICAIL 2023)

## Acknowledgments

This work builds on the methodological foundations established by Zheng et al. (2021) with CaseHOLD and Licari et al. (2023) with ITA-CASEHOLD. The base Norwegian language model was developed by the [National Library of Norway AI Lab](https://huggingface.co/NbAiLab).
