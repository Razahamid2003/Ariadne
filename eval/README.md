# Ariadne evaluation harness

A self-contained way to stress-test how well Ariadne answers, and to capture deep
debug traces for finding bugs and weaknesses.

## What's here

| File | Purpose |
|---|---|
| `test_corpus/` | A small, fact-rich synthetic corpus (a fictional company, two products, policies, a report, a resume, a staff directory) with **known** answers. |
| `question_bank.py` | 37 question intents, ~109 phrasings — paraphrases, abbreviations, typos, cross-document comparisons, list/aggregation, unanswerable questions, and adversarial false-premise traps. Each carries the expected answer so responses can be graded automatically. |
| `ariadne_probe.py` | The probe: runs every variant with retries, records the full chat + raw-retrieval debug per call, grades it, flags weaknesses, checks determinism, and writes a report. |
| `runs/` | Generated output (one JSONL deep log + one Markdown report per run). |

## How to run it

**1. Index the test corpus.** Copy the corpus into your input folder and rebuild:

```bat
copy eval\test_corpus\* data\input\
ingest.bat
```

**2. Start Ariadne** (in another window):

```bat
start.bat
```

**3. Run the probe:**

```bash
python eval/ariadne_probe.py
```

That runs every variant twice against `http://127.0.0.1:8080`. Useful options:

```bash
python eval/ariadne_probe.py --retries 3                 # 3 calls per variant
python eval/ariadne_probe.py --answer-mode detailed      # test a different answer style
python eval/ariadne_probe.py --only-category unanswerable # just the honesty tests
python eval/ariadne_probe.py --base-url http://192.168.1.20:8080  # a LAN instance
python eval/ariadne_probe.py --keep-chats                # keep probe chats in history
```

## What you get

- **`runs/ariadne_probe_<timestamp>.jsonl`** — one line per call with the *complete*
  technical record: the request, the full `/api/chat` response (answer, status,
  confidence, citations, citation-validation result, retrieval diagnostics,
  evidence with scores and match reasons, source documents, latencies), the raw
  `/api/search` retrieval for the same query, and the automatic grade. This is
  what you open when you want to see exactly why something failed.
- **`runs/ariadne_probe_<timestamp>.md`** — a readable report: pass rate by
  category, a weakness-flag summary, a determinism check across retries, latency
  percentiles, and every failure spelled out with its answer and top retrieval.

## How to read the weakness flags

| Flag | What it tells you |
|---|---|
| `false_no_answer` | The answer **is** in the corpus, but Ariadne declined — confidence gate too aggressive, or retrieval missed it. |
| `missing_terms` | It answered, but the expected fact wasn't there — generation or retrieval problem. Check the raw retrieval in the JSONL to tell which. |
| `no_citations` | Claimed a supported answer with no citations — a grounding bug. |
| `unexpected_source` | Answer didn't cite the document that actually holds the fact — retrieval ranking issue. |
| `hallucination` | Answered an unanswerable or false-premise question — the honesty guard failed. |
| `leaked_forbidden_term` | Produced a term it should never have (e.g. agreed with a false premise). |
| `raw_slice_artifact` | Raw retrieval text leaked into the answer — a context/formatting bug. |
| `citation_validation_failed` | The backend itself flagged the citations as invalid. |
| `non-determinism` | The same question gave different answers across retries — a stability concern. |

**Tip for diagnosing:** when a question fails with `missing_terms` or
`unexpected_source`, look at `search_retrieval` in the JSONL line. If the right
document is in the top results, the problem is in context-building or generation;
if it isn't, the problem is in retrieval.
