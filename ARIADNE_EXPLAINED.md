# Ariadne — Complete Technical Briefing (Explain Everything)

An exhaustive, ground-up explanation of the system: every component, the exact
mechanism behind it, why it exists, and how to defend it under questioning. Two
topics that a technical client always probes — **outside network calls** and
**multi-hop reasoning** — get their own deep sections (6 and 7).

> How to use this: read sections 2–5 for the full flow, then study 6, 7, 8 (the
> hard questions), then the Q&A bank (14). The principle that saves you in the
> room: **know the journey of a document in and a question out.** Everything else
> is detail hanging off that spine.

---

## 1. Contents

1. Contents
2. Architecture overview + technology stack
3. Stage A — Ingestion (deep)
4. Stage B — Answering (deep)
5. The data model and where everything lives
6. **Does it make any outside calls?** (the precise answer)
7. **Multi-hop: the complete, honest story**
8. Anti-hallucination: how it avoids making things up
9. Security and data handling
10. The API surface
11. The user interface
12. Configuration and tunability
13. Performance, scale, determinism
14. Deep Q&A bank (including the hard ones)
15. Glossary
16. Meeting tactics

---

## 2. Architecture overview + technology stack

Ariadne is a **Retrieval-Augmented Generation (RAG)** system: it *retrieves*
relevant passages from your documents, then a language model *generates* an answer
restricted to those passages.

Two stages:
- **Ingestion** (once per document): parse → OCR → chunk → classify → embed →
  index. Produces a searchable index.
- **Answering** (per question): retrieve → fuse → re-rank → build context →
  prompt the model → validate citations → return a cited answer or an honest
  "no answer."

**Everything runs locally.** The complete stack:

| Role | Component | Notes |
|---|---|---|
| Web framework | **FastAPI** + Uvicorn (Python 3.11+) | Serves the API and the browser UI. |
| Language model (the "writer") | **llama3.1:8b** via **Ollama** | 8-billion-parameter open model; served locally over an OpenAI-compatible HTTP API at `localhost:11434`. |
| Embedding model (the "meaning fingerprint") | **sentence-transformers/all-MiniLM-L6-v2** | Turns text into a 384-number vector. Runs locally on CPU/GPU. |
| Re-ranker (the "second opinion") | **cross-encoder/ms-marco-MiniLM-L-6-v2** | Scores query+passage together for precise relevance. |
| Vision model (optional) | **qwen2.5vl:7b** via Ollama | Captions images. |
| OCR (optional) | **Tesseract** | Reads scanned/image-only text. |
| Metadata + chunks | **SQLite** (`metadata.db`) | One file, no server. |
| Keyword index | **SQLite FTS5** | Full-text search. |
| Vector index | **NumPy** array (`embeddings.npy` + `metadata.jsonl`) | No external vector DB. |

The deliberate theme: **no external services.** No cloud, no managed vector
database, no API keys. That is what makes a single-machine, air-gapped deployment
realistic.

---

## 3. Stage A — Ingestion (deep)

What happens to each file when you run ingestion.

### 3.1 Archive extraction
ZIP/RAR archives are unpacked first into a working folder; the extracted files
then enter the normal pipeline. (So a zipped folder of mixed documents "just works.")

### 3.2 Loaders (per format)
Each format has a dedicated reader that extracts clean text and structure:
- **PDF** — a PDF library (PyMuPDF / pypdf) pulls text page by page.
- **Word (.docx)** — python-docx reads paragraphs and tables.
- **PowerPoint (.pptx)** — python-pptx reads slide text and notes.
- **Excel (.xlsx) / CSV** — openpyxl / the CSV reader read **row by row**.
- **Text / Markdown / JSON** — read directly.
- **Images** — passed to OCR and/or the vision model.

Why this matters: the system parses *structure*, so a spreadsheet stays tabular
and a heading stays a heading rather than becoming a wall of text.

### 3.3 OCR and vision
- **OCR (Tesseract):** when a PDF page or image has no selectable text (a scan),
  OCR converts the picture of text into real, searchable text.
- **Vision captioning (qwen2.5vl):** can produce a text description of an image so
  it becomes searchable too.
Both are optional and toggled in config. Both run locally.

### 3.4 Chunking (the unit of retrieval)
A whole document is too coarse to retrieve, so it is split into **chunks**:
- **Prose** is split at headings and sentence boundaries, with a small **overlap**
  between adjacent chunks so context isn't severed mid-thought.
- **Spreadsheet/CSV rows become individual "row" records** — each row is its own
  chunk, tagged with a record type ending in `_row`. (This is what later makes
  table counting possible.)
- There is light cleanup ("dirty-span repair") so a chunk doesn't begin or end on a
  broken fragment.

The chunk is the atom of everything downstream: you retrieve chunks, cite chunks,
and build context out of chunks.

### 3.5 Document-type classification (auto-metadata)
The local model reads each document once and assigns a **type** label (report,
spec, roster, contract, resume, etc.). This is stored as metadata and gives
retrieval a gentle signal later ("this question looks like it wants a spec").

### 3.6 Embedding (turning meaning into numbers)
Each chunk is run through the embedding model to produce a **384-dimensional
vector** — a numerical fingerprint of its meaning. Vectors are **normalized to unit
length**, which is what lets similarity be computed as a simple dot product later
(a normalized dot product *is* cosine similarity). Similar meaning → vectors that
point in nearly the same direction.

### 3.7 The two indexes
From the SQLite database, two complementary indexes are built:
- **Keyword index (FTS5):** an inverted full-text index for exact words, codes,
  part numbers, IDs.
- **Vector index (NumPy):** all chunk vectors stacked in one array
  (`embeddings.npy`), with a parallel `metadata.jsonl` mapping each row back to its
  chunk. Search is a single matrix multiply.

### 3.8 Incremental ingestion
Re-running ingestion only re-processes files whose contents changed (tracked by a
hash/working set), and it cleans up records for files you removed. You don't
rebuild the world to add one document.

**End state:** every document is parsed, OCR'd if needed, chunked, labelled,
embedded, and indexed two ways.

---

## 4. Stage B — Answering (deep)

The full journey from a question to a cited answer.

### 4.1 Retrieval leg 1 — vector search (by meaning)
The question is embedded with the **same** model used at ingestion. The query
vector is compared against every chunk vector with one operation —
`scores = vectors @ query` — i.e. a dot product across the whole matrix. Because
everything is unit-normalized, that score **is cosine similarity**. The top matches
by score are taken (configurable, e.g. top 25).

### 4.2 Retrieval leg 2 — keyword search (by exact term)
The query also goes to **FTS5**, which returns chunks containing the exact terms.
This catches things vectors are weak at: specific codes, IDs, rare proper nouns.

### 4.3 Retrieval leg 3 — document-type signal
A **soft nudge** based on the document-type metadata, so the right *kind* of source
is mildly favoured. It is intentionally gentle (a small weight) so it never
overrides genuine relevance — it only breaks ties.

### 4.4 Fusion — Reciprocal Rank Fusion (RRF)
The vector and keyword lists rank chunks differently and on **incompatible score
scales**, so you can't just add the scores. RRF solves this by scoring each chunk
on its **rank** in each list: a chunk's fused score is the sum of `1 / (k + rank)`
across the lists it appears in (with `k` a small constant, 60 by default). The
effect: chunks that rank well in *both* lists rise to the top. It's simple,
parameter-light, and well established.

### 4.5 Re-ranking — the cross-encoder
The fused shortlist is re-scored by a **cross-encoder**. Difference that matters:
the embedding model encodes the query and a chunk **separately** (fast, approximate);
the cross-encoder reads the query and chunk **together in one pass** (slower, much
more accurate at judging "does this passage actually answer this question?"). We run
it only on the shortlist — speed first, then precision.

### 4.6 Filtering and the confidence gate
Two trims:
- A **minimum-score cut-off** (`min_score`) drops weak tail matches.
- A rule that drops chunks that only matched on the vector leg below a threshold
  (`drop_vector_only_below_score`) — i.e. a vague semantic match with no keyword
  support is treated with suspicion.
Then a **confidence** level (high / medium / low) is computed from the top scores.
If confidence is too low, the system declines (honest "no answer") rather than
forcing an answer. This is the first anti-hallucination layer.

### 4.7 Table-aware completion (aggregation)
If the question is a **counting / totalling / "which is highest"** type *and* the
retrieved set already contains a table row, the system loads **all** rows of that
table from storage so the model can reason over the complete table instead of a
partial slice. Important honesty points you should know:
- It only completes a table the retriever **already surfaced** — it doesn't drag in
  arbitrary tables.
- The added rows get a **neutral score** (the table's own lowest retrieved score),
  so nothing is artificially boosted.
- The model still does the counting; this only makes the evidence complete.
- It's recorded in diagnostics (`table_completion_applied`, `rows_added`) so it's
  auditable.

### 4.8 Context building — the evidence packet
The surviving chunks are assembled into a compact **context** the model will be
restricted to:
- Each chunk gets a short **citation label** the model is told to cite.
- There's a **character budget** and a **chunk cap** so the packet fits the model
  comfortably; table rows are allowed past the chunk cap (bounded by a larger char
  budget) so a full table isn't cut off.
- **Neighbour expansion:** for a matched prose chunk, the adjacent chunk(s) may be
  pulled in for continuity.

### 4.9 The prompt — the model's rulebook
The model receives a **system prompt** (rules) + the evidence packet + the question.
The rules are your anti-hallucination story; memorize them:
1. Use **only** the supplied evidence; no outside/world knowledge.
2. **Cite every factual claim** with the provided labels, exactly.
3. If the evidence doesn't support an answer, return the **honest no-answer** message.
4. Don't invent names, dates, figures, or procedures.
5. **False-premise guard:** if a question assumes something untrue ("why did X
   fail?" when X passed), say the premise isn't supported and state what the
   evidence actually shows — don't manufacture a reason.
6. **Complete-table rule:** when the full table is supplied, count/total/compare
   across all of it rather than declining.
There are also formatting and structured-row preservation rules.

### 4.10 Generation — the local model
The prompt goes to **llama3.1:8b** via Ollama over a local HTTP call
(`localhost:11434/v1/chat/completions`, the OpenAI chat format, low temperature for
determinism). The model writes the answer, inserting citation markers like
`[DOCS: doc-…-chunk-0007]` as it goes. "8b" = 8 billion parameters: the
capability-vs-footprint trade that lets it run on a single GPU.

### 4.11 Citation validation — checking the receipts
Before anything is shown, the **validator** extracts every citation marker in the
answer and checks each one against the set of labels that were actually in the
evidence packet. It also recognises a clean "no answer" as valid and flags raw
retrieval text that leaked into the answer. If there are invalid citations, that's
an error that triggers a single **retry** with stricter instructions.

### 4.12 Citation salvage — saving a good answer
Sometimes the model writes a correct answer but forgets to attach markers. Rather
than throw it away, **salvage** compares the answer's sentences against the evidence
chunks by **content-token overlap**; if a sentence clearly came from a specific
chunk (overlap above a threshold, 0.5), it attaches that chunk's correct label. It
**never invents** a citation — it can only attach a label for a chunk whose text
actually overlaps. If salvage can't ground the answer, the system falls back to the
honest no-answer.

### 4.13 The response — what comes back
The API returns a rich object: the **answer**; a **status**
(`supported` / `no_answer` / `error`); a **confidence** level; the list of
**citations**; **source documents** (clean names); a **citation source map**
(turning `[DOCS: …]` markers into "Source 1 — filename" cards for the UI);
**retrieval diagnostics** (scores, what fired, table completion); the **evidence**
with per-chunk scores and match reasons; and **latencies**. This is why the UI can
show not just an answer but *why* it was produced.

---

## 5. The data model and where everything lives

- `data/input/` — your source documents (and `_extracted/` for unpacked archives).
- `storage/metadata.db` — SQLite: the **documents** table and the **chunks** table
  (chunk_id, document_id, text, source_file, source_system, record_type, title,
  chunk_index, sensitivity, citation_label, metadata).
- `storage/vector/embeddings.npy` + `metadata.jsonl` — the vector index.
- `storage/logs/` — query log and chat history (local).
- `config/client.yaml` — the baseline configuration; `config/ui_overrides.yaml` —
  settings changed from the UI, layered on top.

Nothing here is remote. It's all files under the project folder.

---

## 6. Does it make any outside calls? (the precise answer)

This is the question to get exactly right. The honest, defensible answer has two
halves.

### 6.1 At runtime: no outside calls
While answering questions, the **only** network traffic the system makes is to
**`localhost` (the same machine)** — specifically the Ollama model server on
`127.0.0.1:11434`. That is a *loopback* connection; the packets never reach a
network interface that leaves the box. The embedding model and re-ranker run
**in-process** (no network at all).

To guarantee this even against a misbehaving third-party library, the system
installs an **egress guard** at startup (`airgap.py`). Mechanically, it
monkey-patches Python's networking primitives — `socket.connect`,
`socket.connect_ex`, and `socket.getaddrinfo` — and applies a policy:
- **Allowed:** loopback (`127.0.0.1`/`::1`), private and link-local IP ranges,
  `localhost`, the machine's own hostname, `.local` names, and any explicitly
  approved LAN hosts (for the optional network-sharing mode).
- **Blocked:** everything else — and not just connections. It also blocks **DNS
  resolution** of external hostnames, which means a library can't even *look up* an
  external address, closing the DNS-exfiltration path. A blocked attempt raises an
  error rather than failing silently.

It also sets **offline environment flags** (`HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, telemetry-off, etc.) very early, so the ML libraries never
even attempt to "check for updates" or download anything.

This posture is controlled by `security.allow_external_calls` (default **false**).
You can prove the posture is active via the health endpoint.

### 6.2 At setup, once: yes, by necessity
The **only** time Ariadne touches the internet is during initial setup, to download
the things it then runs locally forever:
- Python packages (from PyPI),
- the language model (from the Ollama registry, via `ollama pull`),
- the embedding and re-ranker models (from Hugging Face, on first run).

After that one-time download, the machine can be disconnected permanently. For a
truly air-gapped target you do the download on a connected machine and copy the
models across (covered in the README). Once `offline_mode` is on and the guard is
installed, the system actively refuses to talk to anything but the local machine.

### 6.3 The nuance a sharp client will raise
*"But it makes HTTP calls to the model — that's a network call."* Correct, and the
honest answer is: **yes, to `localhost` — the same physical machine — over the
loopback interface.** No data crosses to any other host. If you want, the model
server can even be bound so it's not reachable from the network at all. The egress
guard would block any attempt to reach a real external address.

**One-line summary for the room:** "At runtime it makes no external calls — the only
network traffic is loopback to the local model on the same machine, and a guard
blocks every non-local connection, including DNS. The single internet touch is the
one-time model download during installation."

---

## 7. Multi-hop: the complete, honest story

You flagged this specifically, and it has subtlety. Here is the whole truth, plus
how to frame it.

### 7.1 What "multi-hop" means
A **multi-hop** question needs information chained across multiple places: answer
part A from document 1, use it to find part B in document 2, etc. The textbook way
to do it algorithmically is an **iterative retrieve-read loop**: retrieve, read out
the "bridge" entities, retrieve again on those, then synthesize. (RAGFlow and
similar systems implement variants of this via query decomposition or knowledge-graph
traversal.)

### 7.2 What Ariadne actually does: single-pass retrieval
Ariadne performs **one retrieval pass per question.** There is no iterative loop and
no query decomposition in the current build. In fact the codebase is explicit about
it: a diagnostics helper stamps every answer with `single_pass_rag = true` and
`automatic_multihop_disabled = true`, and strips any latent multi-hop diagnostic
fields. So if asked directly: **the system uses single-pass RAG; automatic
multi-step retrieval is not enabled in this build.**

### 7.3 Why multi-document questions still work
Here's the part that matters for the demo: **single-pass retrieval still answers
many multi-document questions**, because one retrieval pass — drawing from a top-k of
several chunks and from both the vector and keyword legs — frequently surfaces the
needed pieces from *different* documents into the same evidence packet, and the
language model then synthesizes across them. In our stress testing, questions that
required joining two documents (e.g. "which requirement failed, who owns that
subsystem") were answered correctly the large majority of the time, citing **both**
source documents. So in practice it does cross-document reasoning; it just doesn't
do it with an explicit multi-step search algorithm.

The honest boundary: where the needed evidence is **not** co-retrievable in one pass
— i.e. you genuinely have to find A first to even know what B is — single-pass can
miss it, and that's exactly the case an iterative loop is for.

### 7.4 The one targeted exception: table completion
The table-aware step (4.7) is a *narrow, deterministic* second fetch: when an
aggregation question hits a table, the system fetches the rest of that table's rows.
It is not general multi-hop — it's a specific completeness fix for tabular data — but
it's worth mentioning as "the one place we do a targeted second retrieval."

### 7.5 How to describe it to the client (accurate, not alarming)
> "It uses single-pass retrieval — one search per question. It answers
> cross-document questions by pulling evidence from several documents in a single
> pass and reasoning across them, which in testing handled multi-document questions
> well. It does **not** currently run an iterative, multi-step retrieval loop; that
> kind of automatic multi-hop is an available enhancement we can enable if your
> workload needs deeper reasoning chains."

That is fully truthful, doesn't pretend a feature exists that doesn't, and frames the
gap as a scoped enhancement rather than a defect. **Do not claim it "does multi-hop"**
— claim accurate cross-document synthesis via single-pass retrieval, with iterative
multi-hop as a roadmap item.

---

## 8. Anti-hallucination: how it avoids making things up

Five independent layers, worth enumerating because it's the trust question:
1. **Grounding instruction** — the model may use only the supplied evidence.
2. **Confidence gate** — if retrieval quality is too low, it declines before
   generating.
3. **Mandatory citations** — every claim must carry a label.
4. **Citation validation** — every label is checked against what was actually
   retrieved; invalid ⇒ retry, then no-answer. It **cannot cite a source it didn't
   retrieve.**
5. **False-premise guard** — it pushes back on untrue assumptions instead of
   inventing causes.
Plus salvage, which recovers a good answer's citations *only* by genuine content
overlap, never by invention.

The product stance: **a confident wrong answer is worse than "I don't know,"** so the
system is tuned to decline when unsure.

---

## 9. Security and data handling

- **Air-gapped by default** (section 6): no external calls at runtime; guard blocks
  egress and DNS.
- **Data locality:** documents, database, and indexes live under the project folder;
  nothing is uploaded.
- **Sensitivity field:** chunks carry a sensitivity tag in the data model, and the
  system is built to keep internal identifiers (chunk IDs) behind the UI rather than
  surfacing them as the answer.
- **Auditability:** local query logs; per-answer diagnostics; configurable log
  retention.
- **LAN mode:** if you choose to share over the network, only approved local hosts
  are permitted, and the egress policy still blocks the public internet.

---

## 10. The API surface

A small, clear FastAPI service:
- `POST /api/chat` — ask a question; returns the full answer object (section 4.13).
- `POST /api/search` — raw retrieval only (ranked chunks + scores), no generation —
  useful for diagnosing *why* an answer came out a certain way.
- `GET /api/chats`, `DELETE /api/chats/{id}`, bulk delete — local conversation
  history.
- Admin endpoints — ingestion, rebuilds, index/status, model checks.
- `GET /health` — reports the air-gap/offline posture.

---

## 11. The user interface

- **Home** — ask a question; get a grounded answer with **source cards**; choose an
  answer style (brief/balanced/detailed) and scope.
- **Threads** — conversations saved locally, each in its own thread.
- **Sources** — search the evidence directly without generating an answer.
- **Admin ("The Loom")** — run ingestion, watch index readiness, manage local models.
The UI maps the raw `[DOCS: …]` markers to friendly "Source N — filename" chips, so
users never see internal IDs.

---

## 12. Configuration and tunability

One file, `config/client.yaml`, controls behaviour without code changes:
- **Model:** `llm.model` (swap the model in one line — and because embeddings are
  separate, you do **not** re-index documents to change the answering model).
- **Retrieval weights:** vector vs keyword weight, fusion method, RRF `k`, top-k
  sizes.
- **Thresholds:** `min_score`, `drop_vector_only_below_score`, confidence cut-offs.
- **Features:** OCR on/off, vision on/off, table completion on/off, re-ranking on/off.
- **Security:** `allow_external_calls` (default false), approved LAN hosts.
The UI writes a small set of safe tuning values to `ui_overrides.yaml`, layered on
top of the baseline.

---

## 13. Performance, scale, determinism

- **Latency:** dominated by the model's generation time; retrieval is milliseconds.
  Typical answers in a few seconds on a capable GPU.
- **Determinism:** temperature is low, so repeated identical questions are stable
  (not bit-for-bit guaranteed, but consistent).
- **Scale:** single workstation. The vector search is one matrix multiply, fast into
  the tens/hundreds of thousands of chunks; beyond that you'd add an approximate
  index. Larger corpora need more RAM and ingestion time. Throughput is bounded by
  the local GPU.
- **GPU/CPU:** auto-detected at setup; the right PyTorch build is installed
  (CUDA-enabled for NVIDIA, including a specific build for newer cards; CPU fallback
  otherwise).

---

## 14. Deep Q&A bank (including the hard ones)

**Does it phone home / send our data anywhere?**
No. At runtime the only network traffic is loopback to the local model on the same
machine; a guard blocks every non-local connection and even external DNS lookups. The
sole internet use is the one-time model download at install.

**It calls an HTTP endpoint though — isn't that a network call?**
To `localhost` — the same machine, over the loopback interface. Nothing crosses to
another host. The model server can be bound so it isn't reachable from the network at
all.

**Does it do multi-hop reasoning?**
It uses single-pass retrieval. It answers cross-document questions by retrieving from
several documents in one pass and synthesizing across them, which tested well. It does
not run an iterative multi-step retrieval loop today — that's an available enhancement.

**How is it stopped from inventing answers?**
Five layers: evidence-only instruction, a confidence gate that declines on weak
matches, mandatory citations, citation validation against retrieved evidence (it
can't cite what it didn't retrieve), and a false-premise guard. It prefers "no
answer" to a guess.

**Why two search methods?**
Vector search finds meaning/paraphrase but misses exact codes; keyword search nails
exact terms but misses paraphrase. Combined and re-ranked, they're robust.

**What's RRF / why not just add scores?**
The two searches produce scores on different scales, so they're not addable. RRF
combines them by **rank** instead, rewarding chunks that rank well in both.

**Why a re-ranker on top of search?**
The first search scores query and passage separately (fast, approximate). The
cross-encoder reads them together (slower, accurate). Speed first, precision second.

**What model, and can we change it?**
llama3.1 8B via Ollama; embeddings via a separate small model. Swap the answering
model in one config line, no re-indexing.

**What about accuracy?**
Strong on factual lookups, numeric details, dates, cross-document and
requirement-tracing questions, and recognising false premises. The area still being
tuned is heavy arithmetic over very large tables — bounded by the 8B model — fixable
with a larger model or a structured-data path. Real accuracy on your corpus is what
the review session establishes.

**How do we add documents?**
Drop them in the input folder, re-run ingestion; only changed files are reprocessed.

**Scanned documents?**
OCR converts them to searchable text at ingestion.

**Where does our data physically sit?**
Under the project folder: the source files, the SQLite database, and the index files.
Nowhere else.

**What are the failure modes?**
Large-table arithmetic; phrasing/synonym sensitivity; genuine A-then-B multi-hop
chains; and single-machine scale limits. All listed honestly in section 13/limits.

**Why local instead of a cloud RAG?**
Security and data sovereignty: a cloud service would send documents off-site; this
never does, and it runs with no internet.

---

## 15. Glossary

- **RAG** — Retrieval-Augmented Generation: retrieve passages, then generate from them.
- **Chunk** — a small retrievable piece of a document; the unit of everything.
- **Embedding** — a 384-number vector representing text meaning.
- **Cosine similarity** — closeness of two vectors' directions; here, a normalized
  dot product.
- **Vector search** — finding chunks by meaning via embeddings.
- **FTS5 / keyword search** — SQLite full-text search for exact terms.
- **Hybrid retrieval** — vector + keyword + a document-type nudge, combined.
- **RRF** — Reciprocal Rank Fusion; merges ranked lists by rank, not score.
- **Cross-encoder / re-ranker** — reads query+passage together for precise relevance.
- **Confidence gate** — declines to answer when match quality is too low.
- **Context / grounding** — the evidence packet the model is restricted to.
- **Citation validation** — checking each cited label was actually retrieved.
- **Citation salvage** — re-attaching forgotten labels by content overlap (never
  invents).
- **Single-pass RAG** — one retrieval per question (what Ariadne does).
- **Multi-hop** — chaining retrieval across steps (not enabled here).
- **Egress guard** — the startup mechanism that blocks all non-local network traffic.
- **Loopback / localhost** — the same machine; traffic that never leaves the box.
- **Ollama** — the local server that runs the language model.
- **LLM** — Large Language Model (here llama3.1:8b).

---

## 16. Meeting tactics

- **Lead with the flow** (document in → question out). It pre-answers most questions.
- **Demo the three hard claims in 30 seconds:** ask a question (grounding +
  citations), then ask something the documents don't cover (honest refusal). That
  single demo proves the trust story better than any slide.
- **Own the limits first** — multi-hop is single-pass, large-table arithmetic is
  bounded by the model. Saying it yourself reads as mastery.
- **Never improvise a number.** "I'll confirm that exact figure and send it" is a
  professional, safe answer in a defence meeting; a wrong number is the only real way
  this goes badly.
- **On security, be precise, not vague:** "no external calls at runtime; loopback to
  the local model only; egress and DNS blocked; one-time download at install." Precise
  beats reassuring.
- **On multi-hop, be precise, not flattering:** "single-pass retrieval, accurate
  cross-document synthesis, iterative multi-hop available as an enhancement."

*Best preparation: run it a few times and watch the flow — every term here maps to
something you can see happen on screen, and watching beats re-reading.*
