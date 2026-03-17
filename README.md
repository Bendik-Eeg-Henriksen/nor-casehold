# NOR-CASEHOLD

**The first Norwegian legal NLP benchmark dataset.**

Norwegian adaptation of [CaseHOLD](https://github.com/reglab/casehold) (Zheng et al., 2021) and [ITA-CASEHOLD](https://github.com/dlicari/ITA-CASEHOLD) (Licari et al., ICAIL 2023).

## What is this?

NOR-CASEHOLD is a benchmark dataset for extractive legal holding extraction from Norwegian Supreme Court (*Høyesterett*) decisions. It pairs 627 full-text decisions with their official *sammendrag* (holdings/summaries), written by legal experts at domstol.no.

The task: given a full court decision, select the sentences that best capture the legal holding — as measured against the gold-standard sammendrag.

## Benchmark Results

Test set (n=96), top-5 sentence extraction:

| Method | Type | R-1 | R-2 | R-L |
|---|---|---|---|---|
| BM25 (k1=1.5, b=0.75) | Sparse | **48.27** | **23.65** | **27.90** |
| TF-IDF cosine similarity | Sparse | 47.93 | 23.00 | 27.15 |
| Oracle (greedy ROUGE-1) | Oracle | 44.64 | 16.91 | 22.36 |
| Norwegian Legal BERT | Dense | **41.19** | **13.44** | **19.47** |
| NB-BERT-base | Dense | 39.21 | 12.63 | 18.69 |
| mBERT | Dense | 39.26 | 12.40 | 18.33 |
| Multilingual MiniLM (ST) | Dense | 36.07 | 11.90 | 18.56 |
| Lead-5 | Structural | 20.53 | 6.86 | 11.03 |

Sparse methods dominate on formulaic legal text. Norwegian Legal BERT is the strongest dense encoder. Lead-5 scores less than half of BM25, confirming that key sentences are distributed throughout the decision. Fine-tuning (HM-BERT regression head) is the next step to close the sparse-dense gap.

## Evaluate Your Model

```bash
# Install dependencies
pip install rouge-score datasets transformers torch sentence-transformers

# Evaluate a BERT-style encoder
python evaluate.py --model your-org/your-legal-bert

# Evaluate a sentence-transformer
python evaluate.py --sentence-transformer your-org/your-st-model

# Reproduce all published baselines
python evaluate.py --all

# Sparse + structural baselines only (no GPU needed)
python evaluate.py --all --no-dense
```

Test data is auto-downloaded from [HuggingFace](https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold). Your results are printed alongside published baselines.

## Lineage

- **CaseHOLD** (English, common law) — 53K multiple-choice questions from US case law
- **ITA-CASEHOLD** (Italian, civil law) — 1,101 extractive summarization pairs from Italian administrative courts
- **NOR-CASEHOLD** (Norwegian, civil law) — 627 extractive summarization pairs from Høyesterett

## Dataset

**627** decision-sammendrag pairs · **Split:** 440 train / 91 val / 96 test · **Years:** 2000–2025

Available on [HuggingFace](https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold):

```python
from datasets import load_dataset
dataset = load_dataset("bendik-eeg-henriksen/nor-casehold")
```

## Data Sources

All data is sourced from publicly accessible Norwegian government websites:
- **domstol.no** — Høyesterett decisions and sammendrag (public domain)
- No Lovdata content is used

## Citation
if you find this useful, please consider citing
```bibtex
@misc{eeg-henriksen2026norcasehold,
  title={NOR-CASEHOLD: A Norwegian Legal Benchmark for Extractive Holding Extraction},
  author={Eeg-Henriksen, Jan Bendik},
  year={2026},
  url={https://huggingface.co/datasets/bendik-eeg-henriksen/nor-casehold}
}
```

## License

Dataset: CC-BY-4.0
Code: MIT

## Author

**Bendik Eeg-Henriksen**

