# Amazon ML Challenge 2026 — Business Entity Resolution

> **Score: 0.968235 (F₀.₅) · Rank: 1584**

## Problem Statement

Given business records from **3 independent data sources** with noisy and inconsistent fields (names, addresses, countries), determine which records across sources refer to the **same real-world business entity**. Source 1 is the deduplicated reference — the task is to find all matching records from Source 2 and Source 3 for each Source 1 entity.

**Key challenges:**
- Multilingual data (English, Hindi, Tamil, Kannada, Bengali scripts + French in test)
- Name variations — abbreviations, typos, leet-speak, DBA/trade names, domain names as business names
- Address noise — missing components, transliterations, `null` values, landmark-based references
- Massive scale — ~12M records per split (train & test)

**Metric:** F₀.₅ (precision-heavy — false merges are penalised more than missed links)

## Pipeline Architecture

```
raw TSVs ─► normalize (learned translit + abbreviation maps) ─► fine-tune multilingual bi-encoder (contrastive)
                                                                         │
     ┌───────────────────────────────────────────────────────────────────┘
     ▼
 BLOCKING per country: dense top-8 (FAISS IVF) ∪ TF-IDF name-char / name-word / address top-4
     ▼
 STAGE 1  XGBoost(GPU) on cheap scores + ranks + margins  ─► keep top-3 candidates  ═► candidate_pairs.tsv
     ▼
 STAGE 2  XGBoost(GPU) on 40+ fuzzy/number/context features  (+ cross-encoder on the ambiguous band)
     ▼
 ASSIGNMENT (each S2/S3 → at most one S1) ─► per-S1 expected-F0.5 subset selection  ═► matching_results.tsv
```

### Stage Details

| Stage | Description |
|---|---|
| **Normalisation** | Learned transliteration maps (Indic→Latin) from training pairs, abbreviation expansion, leet-speak correction, ordinal/number canonicalization, legal suffix removal. 5 name views (clean, core, alias, compact, skeleton) + structured address tokens |
| **Bi-Encoder** | `intfloat/multilingual-e5-small` (MIT, 118M params) fine-tuned with symmetric InfoNCE loss on world-A pairs. Same-region batching for hard in-batch negatives |
| **Blocking** | Per-country: FAISS IVF dense top-8 ∪ TF-IDF sparse top-4 (name-char, name-word, address). Union of all candidate sources |
| **Stage 1 (Pruning)** | XGBoost on dense cosine + sparse scores + rank/margin features → keeps top-3 candidates per S2/S3 record |
| **Stage 2 (Matching)** | XGBoost on 40+ features (fuzzy ratios, Jaro-Winkler, token set/sort ratios, number overlap, address component matching) with 3-seed ensemble. Optional cross-encoder (`xlm-roberta-base`, MIT, 278M) on the ambiguous confidence band |
| **Assignment** | Each S2/S3 record → at most one S1. Per-S1 expected-F₀.₅ subset selection tuned on held-out validation |

### Key Design Decisions

- **Disjoint validation worlds** — S1 entities split into World A (fit) and World B (held out) by hashing the first name token. Unmatched "sibling" records follow their name group's world, preventing validation score inflation
- **Test-matched distractor density** — Both worlds are topped up to the test's 5.76 others-per-S1 ratio with orphan records, keeping validation precision honest
- **Country-isolated blocking** — Matches never cross countries (verified: 0/693K pairs), so blocking runs within each country. No country feature in the model, so unseen countries (France) work out of the box

## Repository Structure

```
student_resource/
├── amazon_ml_challenge_problem_statement.txt   # Official problem statement
├── src/
│   ├── amz-ml-challenge-kaggle-notebook.ipynb  # Complete end-to-end pipeline
│   └── requirements.txt                        # Pinned dependencies
├── utils/
│   └── validate_submission.py                  # Official submission validator
├── .gitignore
└── README.md
```

## 🔧 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.10.0+cu128 | Bi-encoder & cross-encoder training/inference |
| Transformers | 5.0.0 | Pretrained multilingual models (E5-small, XLM-RoBERTa) |
| XGBoost | 3.2.0 | Stage-1 pruning & Stage-2 matching (GPU-accelerated) |
| FAISS | (cpu) | Dense nearest-neighbour blocking (IVF index) |
| RapidFuzz | 3.14.6 | Fast fuzzy string similarity (Levenshtein, Jaro-Winkler) |
| sparse_dot_topn | 1.2.0 | Multi-threaded sparse TF-IDF top-k |
| scikit-learn | 1.6.1 | TF-IDF vectorization |
| Unidecode | 1.4.0 | Unicode → ASCII transliteration fallback |
| pandas | 2.3.3 | Data wrangling |
| NumPy | 2.0.2 | Numerical operations |
| SciPy | 1.16.3 | Sparse matrix operations |

## Reproduction Steps

### Data Setup

1. Download the dataset from the Amazon ML Challenge 2026 portal

### Running on Kaggle (Recommended)

1. **Upload dataset:** Zip `dataset/` (train + test) and `utils/validate_submission.py`, upload as a private Kaggle Dataset
2. **Create notebook:** Upload `src/amz-ml-challenge-kaggle-notebook.ipynb`
3. **Settings:** Accelerator **GPU T4 ×2**, **Internet ON**, Persistence OFF
4. **Dry run:** Set `SMOKE = True` in the config cell → Run All (~10–15 min)
5. **Full run:** Set `SMOKE = False` → **Save Version → Save & Run All (Commit)**. Expect **3–5 hours**
6. Download `output/*.tsv` from the run artifacts

### Running Locally

```bash
# Install dependencies
pip install -r src/requirements.txt
pip install faiss-cpu

# Set environment variables
export BER_DATA=./dataset
export BER_WORK=./output

# Run the notebook
jupyter nbconvert --to notebook --execute src/amz-ml-challenge-kaggle-notebook.ipynb
```

> ⚠️ Requires a CUDA GPU with ≥16 GB VRAM. Two T4s (Kaggle default) are ideal.

### Validate Output

```bash
python utils/validate_submission.py \
    --matching src/kaggle-outputs/matching_results.tsv \
    --candidate src/kaggle-outputs/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Results

| Metric | Score |
|---|---|
| **F₀.₅ (Public Leaderboard)** | **0.968235** |
| **Rank** | **1584** |

## Team

- [Abhinav Shukla](https://github.com/abhishukla0204)
- [Krishna Kumar Gupta](https://github.com/krishnagupta7171)
- [Amit Kumar](https://github.com/Amitkumar-21)
