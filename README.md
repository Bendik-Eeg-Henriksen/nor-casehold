# NOR-CASEHOLD

*A Norwegian legal AI benchmark dataset.*

## Overview

NOR-CASEHOLD is a benchmark dataset for evaluating legal language models on Norwegian court decisions. It contains **627 Norwegian Supreme Court (Høyesterett) decisions** paired with their official **sammendrag** — court-published summaries that capture the core outcome and key legal reasoning of each case.

Unlike the original multiple-choice **CaseHOLD** formulation designed for U.S. common-law citation patterns, NOR-CASEHOLD uses **extractive sentence selection** evaluated with **ROUGE**, following the general methodological direction of **ITA-CASEHOLD** for civil-law jurisdictions where official summaries are available.

## Why this exists

There was previously no widely available benchmark for evaluating Norwegian legal AI systems on court decisions. English has **CaseHOLD** (Zheng et al., ICAIL 2021) and **LexGLUE**. Italian has **ITA-CASEHOLD** (Licari et al., ICAIL 2023). NOR-CASEHOLD is intended to help fill that gap for Norwegian legal text.

It enables researchers and practitioners to test whether domain-adapted models actually outperform general-purpose models on Norwegian judicial language.

## Task

Given a full Høyesterett decision, select the sentences that best capture the substance of the official **sammendrag**.

This is an **extractive legal summarization** task. Models rank document sentences by relevance to the **sammendrag** and select the top-N. The extracted summary is then scored against the gold-standard **sammendrag** using **ROUGE-1, ROUGE-2, and ROUGE-L**.

## Benchmark your model in one command

NOR-CASEHOLD includes a lightweight evaluation harness so you can benchmark any compatible Hugging Face encoder directly on the released test set.

```bash
# Install
pip install -r requirements.txt

# Evaluate a BERT-style encoder
python evaluate.py --model your-org/your-legal-bert

# Evaluate a sentence-transformer
python evaluate.py --sentence-transformer your-org/your-st-model

# Reproduce all published baselines
python evaluate.py --all
```

Test data is downloaded automatically from Hugging Face. Results are printed alongside the published baselines.

## Benchmark Results

Results on the **test set (n = 96)**, extracting the top 5 sentences per document:

| Method | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| BM25 (k1=1.5, b=0.75) | Sparse | 48.27 | 23.65 | 27.90 |
| TF-IDF cosine similarity | Sparse | 47.93 | 23.00 | 27.15 |
| Oracle (greedy ROUGE-1) | Oracle | 44.64 | 16.91 | 22.36 |
| Norwegian Legal BERT | Dense | 41.19 | 13.44 | 19.47 |
| NbAiLab/nb-bert-base | Dense | 39.21 | 12.63 | 18.69 |
| bert-base-multilingual-cased | Dense | 39.26 | 12.40 | 18.33 |
| paraphrase-multilingual-MiniLM-L12-v2 | Dense (ST) | 36.07 | 11.90 | 18.56 |
| Lead-5 | Structural | 20.53 | 6.86 | 11.03 |

### Key findings

- **Sparse methods perform best on this benchmark.** BM25 and TF-IDF outperform all tested dense encoders.
- **Domain pretraining helps among dense baselines.** Norwegian Legal BERT is the strongest dense encoder baseline, outperforming NB-BERT-base, mBERT, and multilingual MiniLM.
- **General-purpose sentence transformers lag behind.** Multilingual MiniLM performs below the BERT-based encoders in this setting.
- **Document position alone is not enough.** Lead-5 achieves only **20.53 ROUGE-1**, indicating that salient sentences are often distributed throughout the decision rather than concentrated at the beginning.
- **Task-specific fine-tuning is a promising next step.** The gap between sparse and dense methods suggests room for stronger supervised approaches.

## Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/Bendik-Eeg-Henriksen/nor-casehold.git
cd nor-casehold
pip install -r requirements.txt
```


## Dataset Structure

### Splits

| Split | Documents |
|---|---:|
| Train | 440 |
| Validation | 91 |
| Test | 96 |
| Total | 627 |

### Fields

Each record contains:

| Field | Description |
|---|---|
| `case_id` | Høyesterett case identifier (e.g. `HR-2023-299-A`) |
| `title` | Case title |
| `date` | Decision date |
| `category` | Case category (civil/criminal) |
| `rettsomrade` | Legal area |
| `source_url` | URL to original decision on `domstol.no` |
| `sammendrag` | Official summary (gold standard) |
| `full_text` | Full text of the court decision |
| `sentences` | Individual sentences with per-sentence ROUGE-derived relevance scores |
| `compression_ratio` | Ratio of sammendrag length to full text length |

### Pre-computed sentence scores

Each sentence in the `sentences` array includes:

- `rouge1_f` — ROUGE-1 F-measure against the **sammendrag**
- `rouge2_f` — ROUGE-2 F-measure against the **sammendrag**
- `hm_score` — harmonic mean of ROUGE-1 and ROUGE-2, used as the sentence relevance target

## Methodology

Following the general extractive setup used in ITA-CASEHOLD:

1. Full-text decisions are split into sentences.
2. Per-sentence **ROUGE-1** and **ROUGE-2 F-measure** scores are computed against the gold **sammendrag**.
3. The harmonic mean of ROUGE-1 and ROUGE-2 is used as a sentence relevance score.
4. Models rank document sentences by relevance.
5. The top-N sentences, restored to original document order, are treated as the extracted summary and evaluated against the gold **sammendrag** using **ROUGE-1, ROUGE-2, and ROUGE-L**.

All dense encoder embeddings use **mean pooling** with **max_length=256**.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("bendik-eeg-henriksen/nor-casehold")

train = dataset["train"]
test = dataset["test"]

example = test[0]
print(example["case_id"])
print(example["sammendrag"][:200])

sentences = example["sentences"]
top = sorted(
    [s for s in sentences if not s.get("is_short_fragment")],
    key=lambda x: x["hm_score"],
    reverse=True
)[:5]

for s in top:
    print(f"[{s['hm_score']:.4f}] {s['text'][:100]}...")
```

## Data Source

All data is sourced from publicly accessible Norwegian government websites:

- **domstol.no** — Høyesterett decisions and published summaries

Only decisions with officially published **sammendrag** are included. During quality control, **15 ankeutvalg decisions** were excluded because their published **sammendrag** consisted of procedural stub text rather than substantive summaries.

## Lineage

| Dataset | Language | Legal system | Task | Size | Published |
|---|---|---|---|---:|---|
| CaseHOLD | English | Common law | Multiple-choice QA | 53,000 | ICAIL 2021 |
| ITA-CASEHOLD | Italian | Civil law | Extractive summarization | 1,101 | ICAIL 2023 |
| NOR-CASEHOLD | Norwegian | Civil law | Extractive summarization | 627 | 2026 |

## Limitations

- **Size:** 627 examples from Høyesterett only. Expansion to lower courts and additional legal sources would strengthen the benchmark.
- **Extractive only:** The task evaluates extractive sentence selection, not abstractive legal summarization.
- **Variation in summaries:** Official **sammendrag** vary in length and style across decisions and time periods.
- **Civil-law adaptation:** Unlike the original CaseHOLD multiple-choice format, NOR-CASEHOLD uses extractive summarization to better match the structure of Norwegian legal materials.

## Citation

If you find this useful, please consider citing:

```bibtex
@misc{eeg-henriksen2026norcasehold,
  title={NOR-CASEHOLD: A Norwegian Legal Benchmark for Extractive Legal Summarization},
  author={Eeg-Henriksen, Jan Bendik},
  year={2026},
  url={https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold}
}
```

## License

- **Dataset:** CC BY 4.0
- **Code:** MIT

## Author

**Bendik Eeg-Henriksen**

## Related Resources

- **Norwegian Legal BERT** — `bendik-eeg-henriksen/norwegian-legal-bert`
- **NbAiLab/nb-bert-base**
- **CaseHOLD** — Zheng et al. (ICAIL 2021)
- **ITA-CASEHOLD** — Licari et al. (ICAIL 2023)

## Acknowledgments

This work builds on the methodological foundations established by **Zheng et al. (2021)** with **CaseHOLD** and **Licari et al. (2023)** with **ITA-CASEHOLD**. The base Norwegian language model used for domain adaptation was developed by the **National Library of Norway AI Lab**.
