# Pharma Sales AI Analyst

An interactive, insight-driven AI application built on a pharmaceutical sales dataset. Ask plain-English questions about reps, territories, brands, HCPs, payor mix, and market share — and get accurate, data-grounded answers instantly.

---

## Project Structure

```
├── data/                   # All 9 raw CSV files (star-schema dataset)
├── data_analysis.ipynb     # Step 1 — Exploratory Data Analysis
├── data_engine.py          # Step 2 — Core analytics & context engine
├── ai_application.ipynb    # Step 3 — Groq integration test notebook
├── app.py                  # Step 4 — Streamlit chat interface
├── test_suite.py           # Step 7 — 136 pytest test cases
├── requirements.txt        # Python dependencies
└── .env                    # API key (not committed)
```

---

## How It Was Built

### Step 1 — Data Analysis (`data_analysis.ipynb`)
Explored all 9 tables of a pharma sales star-schema dataset:
- 5 dimension tables: HCPs, reps, territories, accounts, dates
- 4 fact tables: prescriptions (Rx), rep activity, payor mix, line metrics
- Analysed Rx trends, HCP tier breakdowns, rep performance, territory scorecards, payor mix shifts, and market share

### Step 2 — Data Engine (`data_engine.py`)
Built a pure Python analytics engine (no Streamlit dependency) that:
- Loads and validates all 9 CSVs with strict schema checks
- Pre-computes all key aggregations: brand TRx/NRx, QoQ growth, rep scorecards, territory efficiency, HCP coverage gaps, payor mix, and market share
- Runs an **insights engine** that flags ALERT / WARN / CRITICAL / POSITIVE signals automatically
- Converts all computed results into a structured context string fed to the LLM

### Step 3 — Groq Integration (`ai_application.ipynb`)
- Connected to Groq API using `llama-3.3-70b-versatile`
- Tested the full pipeline: load data → build context → send to LLM → get answer
- Validated multi-turn conversation history

### Step 4 — Streamlit Interface (`app.py`)
- Streaming chat UI (responses appear word by word)
- Sidebar with live dataset metrics
- 8 suggested questions on first load
- Source citation expander below every answer showing which data sections were used
- Graceful error handling for API failures and out-of-scope questions

### Step 5 — Why Answers Are Accurate (Prompt Engineering)
Most AI chat tools search for keywords and guess. This system works differently:

- **No guessing from raw data** — Pandas pre-computes every aggregation (totals, averages, rankings, QoQ growth rates) before the LLM sees anything. The LLM reasons over real numbers, not raw rows.
- **Insights are pre-flagged** — the data engine automatically labels ALERT / WARN / CRITICAL / POSITIVE signals before the question is even asked, so the LLM has business context baked in.
- **Strict system prompt** — the LLM is instructed to: cite the exact data section it is drawing from, show numbers side by side when comparing, and explicitly say "I don't have enough data" when a question is out of scope. It cannot make up figures.
- **Source citations** — every answer in the UI shows which data sections were used (e.g. "REP SCORECARD", "TERRITORY SCORECARD"), so the user can verify the reasoning.
- **Scoped knowledge** — the LLM only knows what the dataset contains. It will not hallucinate competitor data or external market figures.

### Step 6 — Verification Layer (`data_engine.py`)
Every answer is silently checked against the pre-computed facts before being shown to the user:
- Numbers are extracted from the AI answer and matched against known dataset values
- **Verified** — answer contains numbers that match raw data → shown cleanly, nothing extra
- **Flagged** — numbers present but don't match any known fact → a single quiet line is appended: *"For critical decisions, please verify against source data."*
- **No numbers** (e.g. greetings, definitions) → no verification triggered, nothing shown
- The user never sees technical details — only the outcome when relevant

### Step 7 — Testing (`test_suite.py`)
- **136 tests across 17 test classes — all passing ✓**
- Covers every function in `data_engine.py` including the full verification layer
- Scenarios tested: happy paths, empty DataFrames, missing files, corrupt CSVs, missing columns, division by zero, invalid API keys, empty questions, API failures, null responses, unicode input, greeting detection, made-up numbers, and full end-to-end pipeline

---

## Setup

**1. Clone the repo and install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**3. Run the app**
```bash
streamlit run app.py
```

**4. Run the tests**
```bash
pytest test_suite.py -v
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python + Pandas | Data loading, validation, aggregations |
| Groq API (`llama-3.3-70b-versatile`) | LLM inference — fast, accurate responses |
| Streamlit | Chat UI |
| python-dotenv | API key management via `.env` |
| pytest | 136-test suite covering all edge cases |

---

## Dataset

9 CSV files covering a pharmaceutical sales operation:
- **3 territories**, **9 reps**, **90 HCPs**, **24 accounts**
- **1,530** prescription records | **2,962** rep activity records
- Date range: August 2024 onwards
