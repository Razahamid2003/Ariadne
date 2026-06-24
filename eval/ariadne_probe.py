"""Ariadne evaluation probe.

Runs every question variant in the question bank against a running Ariadne
instance, with retries, and records a full technical debug trace for each call
so bugs and weaknesses can be identified.

For every call it captures:
  * the exact request,
  * the complete /api/chat response (answer, status, confidence, citations,
    citation-validation result, retrieval diagnostics, evidence with scores and
    match reasons, source documents, latencies),
  * the raw /api/search retrieval for the same query (ranked candidates +
    scores), so you can tell whether a wrong answer was a *retrieval* failure or
    a *generation* failure,
  * an automatic grade against the expected answer, with weakness flags.

Outputs (under --outdir, default eval/runs):
  * <run>.jsonl  — one line per call, the deep debug record.
  * <run>.md     — a readable report: pass rates by category, weakness summary,
                   determinism across retries, latency percentiles, and a
                   detailed list of every failure.

Usage:
    python eval/ariadne_probe.py
    python eval/ariadne_probe.py --base-url http://127.0.0.1:8080 --retries 3
    python eval/ariadne_probe.py --only-category unanswerable --answer-mode detailed
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ariadne emits citations inline as [PREFIX: doc-...-chunk-NNNN], where PREFIX is
# the source system (DOCS, STRUCTURED, HEADS, or a folder-derived token). These are
# intended citation markers, NOT leaked raw text, so we strip them before scanning
# for genuine retrieval leakage.
CITATION_MARKER = re.compile(r"\[[A-Za-z0-9_]+:\s*[^\]]*\]")
ARTIFACT_PATTERNS = [
    re.compile(r"chunk[_-]?\d{3,}", re.I),
    re.compile(r"\bdoc[_-]?\d{3,}", re.I),
    re.compile(r"evidence_id", re.I),
    re.compile(r"text_preview", re.I),
]
# Phrases that signal an honest "no evidence" answer.
NO_EVIDENCE_HINTS = [
    "no answer", "not found", "no relevant", "couldn't find", "could not find",
    "don't have", "do not have", "no information", "insufficient", "not contain",
    "isn't in", "is not in", "no evidence", "unable to find", "cannot find",
    "does not contain", "not explicitly mention", "not explicitly stated",
    "not specified", "not available in", "not mentioned in",
]


def http_post(url: str, payload: dict, timeout: float) -> tuple[int, dict | None, str | None, float]:
    """POST JSON. Returns (status_code, json_or_None, error_or_None, elapsed_ms)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, json.loads(body), None, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - start) * 1000
        detail = exc.read().decode("utf-8", "replace")[:2000]
        return exc.code, None, f"HTTP {exc.code}: {detail}", elapsed
    except Exception as exc:  # connection refused, timeout, etc.
        elapsed = (time.perf_counter() - start) * 1000
        return 0, None, f"{type(exc).__name__}: {exc}", elapsed


def http_delete(url: str, timeout: float) -> None:
    req = urllib.request.Request(url, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception:
        pass  # cleanup is best-effort


def contains_any(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t.lower() in low]


def find_artifacts(text: str) -> list[str]:
    # Remove intended citation markers first so they aren't mistaken for leakage.
    cleaned = CITATION_MARKER.sub("", text or "")
    hits = []
    for pat in ARTIFACT_PATTERNS:
        if pat.search(cleaned):
            hits.append(pat.pattern)
    return hits


def cited_sources(response: dict) -> list[str]:
    return [d.get("display_name", "") for d in (response.get("source_documents") or [])]


def grade(expect: dict, status_code: int, resp: dict | None, err: str | None) -> dict:
    """Grade one chat response. Returns {passed, flags, notes}."""
    flags: list[str] = []
    notes: list[str] = []

    if err or resp is None:
        return {"passed": False, "flags": ["request_failed"], "notes": [err or "no response"]}
    if status_code != 200:
        return {"passed": False, "flags": ["http_error"], "notes": [f"HTTP {status_code}"]}

    status = resp.get("status")
    answer = resp.get("answer") or ""
    citations = resp.get("citations") or []
    validation = resp.get("validation") or {}
    sources = cited_sources(resp)

    if status == "error":
        return {"passed": False, "flags": ["server_error"], "notes": ["answer status=error"]}

    # Raw-retrieval leakage into the answer text.
    artifacts = find_artifacts(answer)
    if artifacts:
        flags.append("raw_slice_artifact")
        notes.append(f"artifact patterns: {artifacts}")

    # Citation validation result, if the backend reported one.
    if isinstance(validation, dict) and validation.get("valid") is False:
        flags.append("citation_validation_failed")
        notes.append(f"validation errors: {validation.get('errors')}")

    forbid = expect.get("forbid_terms") or []
    leaked = contains_any(answer, forbid)
    if leaked:
        flags.append("leaked_forbidden_term")
        notes.append(f"forbidden terms present: {leaked}")

    if expect.get("answerable", True):
        # Should have produced a grounded answer.
        if status == "no_answer":
            flags.append("false_no_answer")
            notes.append("declined to answer an answerable question")
        any_terms = expect.get("any_terms") or []
        all_terms = expect.get("all_terms") or []
        if any_terms and not contains_any(answer, any_terms):
            flags.append("missing_terms")
            notes.append(f"none of expected any_terms present: {any_terms}")
        missing_all = [t for t in all_terms if t.lower() not in answer.lower()]
        # all_terms is treated as "at least the set of facts" — if ANY variant
        # spelling matches we count it; here we require each listed term, but
        # callers list interchangeable spellings carefully.
        if all_terms and len(missing_all) == len(all_terms):
            flags.append("missing_terms")
            notes.append(f"expected all_terms absent: {all_terms}")
        # all_required is STRICT: every listed term must appear (multi-hop / multi-fact).
        all_required = expect.get("all_required") or []
        missing_req = [t for t in all_required if t.lower() not in answer.lower()]
        if missing_req:
            flags.append("missing_terms")
            notes.append(f"missing required terms (strict): {missing_req}")
        if status == "supported" and not citations:
            flags.append("no_citations")
            notes.append("supported answer carried no citations")
        exp_src = expect.get("expect_source")
        if exp_src:
            wanted = [exp_src] if isinstance(exp_src, str) else list(exp_src)
            if not any(any(w.lower() in s.lower() for s in sources) for w in wanted):
                flags.append("unexpected_source")
                notes.append(f"expected source {wanted} not among cited {sources}")
        # Pass = answered with right facts and grounded; source/artifact issues
        # are weaknesses but not automatic correctness failures.
        hard_fail = {"false_no_answer", "missing_terms", "no_citations",
                     "leaked_forbidden_term", "raw_slice_artifact"}
        passed = not (set(flags) & hard_fail)
    else:
        # Unanswerable / adversarial: must decline, OR explicitly reject the false
        # premise (e.g. "it actually passed", "the premise is not supported"), and
        # must not fabricate. A correct rejection counts even when it cites the
        # evidence that disproves the premise.
        rejection_hints = [
            "not a failure", "did not fail", "actually pass", "was passed",
            "listed as pass", "is a pass", "marked pass", "premise", "not supported",
            "no such", "did not occur", "there was no", "was not waived",
            "not waived", "incorrect", "in fact", "contrary", "no evidence of",
            "not explicitly", "rather than", "passed acceptance", "passed its",
        ]
        rejected = bool(contains_any(answer, rejection_hints))
        declined = (status == "no_answer") or (not citations and contains_any(answer, NO_EVIDENCE_HINTS))
        if not (declined or rejected):
            flags.append("hallucination")
            notes.append("answered an unanswerable/false-premise question without declining or rejecting the premise")
        passed = (declined or rejected) and "leaked_forbidden_term" not in flags and "hallucination" not in flags

    return {"passed": passed, "flags": flags, "notes": notes}


def run(args) -> int:
    import importlib
    try:
        bank = importlib.import_module(args.bank)
        QUESTIONS = bank.QUESTIONS
    except Exception as exc:
        print(f"[FATAL] Could not load question bank '{args.bank}': {exc}")
        return 2

    chat_url = f"{args.base_url.rstrip('/')}/api/chat"
    search_url = f"{args.base_url.rstrip('/')}/api/search"

    intents = QUESTIONS
    if args.only_category:
        intents = [q for q in intents if q["category"] == args.only_category]
    if args.max_intents:
        intents = intents[: args.max_intents]

    # Quick connectivity check.
    code, resp, err, _ = http_post(chat_url, {"query": "connectivity check", "show_evidence": False}, args.timeout)
    if err and code == 0:
        print(f"[FATAL] Could not reach Ariadne at {args.base_url}\n        {err}")
        print("        Is the server running? Start it with start.bat (or start_lan.bat).")
        return 2
    if resp and resp.get("chat_id"):
        http_delete(f"{args.base_url.rstrip('/')}/api/chats/{resp['chat_id']}", args.timeout)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"ariadne_probe_{ts}.jsonl"
    md_path = outdir / f"ariadne_probe_{ts}.md"

    records = []
    total_calls = sum(len(q["variants"]) for q in intents) * args.retries
    done = 0
    print(f"Running {len(intents)} intents x variants x {args.retries} retries = {total_calls} calls\n")

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for q in intents:
            for variant in q["variants"]:
                for attempt in range(1, args.retries + 1):
                    chat_payload = {
                        "query": variant, "chat_id": None, "top_k": args.top_k,
                        "show_evidence": True, "answer_mode": args.answer_mode,
                        "preview_chars": 1500,
                    }
                    code, resp, err, client_ms = http_post(chat_url, chat_payload, args.timeout)

                    search_resp = None
                    if not args.no_search and err is None:
                        s_payload = {"query": variant, "top_k": args.top_k, "preview_chars": 500}
                        _, search_resp, _, _ = http_post(search_url, s_payload, args.timeout)

                    g = grade(q["expect"], code, resp, err)

                    # cleanup the probe chat to avoid cluttering history
                    if resp and resp.get("chat_id") and not args.keep_chats:
                        http_delete(f"{args.base_url.rstrip('/')}/api/chats/{resp['chat_id']}", args.timeout)

                    record = {
                        "intent_id": q["id"], "category": q["category"],
                        "capability": q.get("capability", q["category"]), "attempt": attempt,
                        "query": variant, "expect": q["expect"],
                        "http_status": code, "client_latency_ms": round(client_ms, 1),
                        "error": err,
                        "grade": g,
                        # full debug payload:
                        "chat_response": resp,
                        "search_retrieval": (search_resp or {}).get("results") if search_resp else None,
                    }
                    jf.write(json.dumps(record) + "\n")
                    records.append(record)

                    done += 1
                    mark = "PASS" if g["passed"] else "FAIL"
                    flagstr = (" " + ",".join(g["flags"])) if g["flags"] else ""
                    print(f"[{done:>3}/{total_calls}] {mark:4} {q['id']:<20} a{attempt} \"{variant[:46]}\"{flagstr}")

    report = build_report(records, intents, args)
    md_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 64)
    print(summary_block(records))
    print(f"\nDeep debug log : {jsonl_path}")
    print(f"Report         : {md_path}")
    return 0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (s[f + 1] - s[f]) * (k - f)


def summary_block(records: list[dict]) -> str:
    total = len(records)
    passed = sum(1 for r in records if r["grade"]["passed"])
    lat = [r["client_latency_ms"] for r in records if r["error"] is None]
    flags: dict[str, int] = {}
    for r in records:
        for fl in r["grade"]["flags"]:
            flags[fl] = flags.get(fl, 0) + 1
    lines = [f"RESULT: {passed}/{total} calls passed ({100*passed/max(total,1):.1f}%)"]
    if lat:
        lines.append(f"Latency ms  p50={percentile(lat,0.5):.0f}  p95={percentile(lat,0.95):.0f}  max={max(lat):.0f}")
    if flags:
        lines.append("Weakness flags: " + ", ".join(f"{k}={v}" for k, v in sorted(flags.items(), key=lambda x: -x[1])))
    return "\n".join(lines)


def build_report(records: list[dict], intents: list[dict], args) -> str:
    total = len(records)
    passed = sum(1 for r in records if r["grade"]["passed"])
    lat = [r["client_latency_ms"] for r in records if r["error"] is None]

    # by category
    cats: dict[str, list[dict]] = {}
    for r in records:
        cats.setdefault(r["category"], []).append(r)

    # weakness flag tally
    flags: dict[str, int] = {}
    for r in records:
        for fl in r["grade"]["flags"]:
            flags[fl] = flags.get(fl, 0) + 1

    # determinism: group by (intent, query), check status + citation set stability
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["intent_id"], r["query"]), []).append(r)
    nondeterministic = []
    for (iid, query), rs in groups.items():
        if len(rs) < 2:
            continue
        statuses = {(r["chat_response"] or {}).get("status") for r in rs}
        citesets = {tuple(sorted((r["chat_response"] or {}).get("citations") or [])) for r in rs}
        passes = {r["grade"]["passed"] for r in rs}
        if len(statuses) > 1 or len(citesets) > 1 or len(passes) > 1:
            nondeterministic.append((iid, query, statuses, passes))

    out = []
    out.append(f"# Ariadne probe report\n")
    out.append(f"- Run: {datetime.now(timezone.utc).isoformat()}")
    out.append(f"- Target: `{args.base_url}`  ·  answer_mode=`{args.answer_mode}`  ·  top_k={args.top_k}  ·  retries={args.retries}")
    out.append(f"- **Overall: {passed}/{total} calls passed ({100*passed/max(total,1):.1f}%)**")
    if lat:
        out.append(f"- Latency (ms): p50 {percentile(lat,0.5):.0f} · p95 {percentile(lat,0.95):.0f} · max {max(lat):.0f}")
    out.append("")

    out.append("## Pass rate by category\n")
    out.append("| Category | Passed | Total | Rate |")
    out.append("|---|---|---|---|")
    for cat in sorted(cats):
        rs = cats[cat]
        p = sum(1 for r in rs if r["grade"]["passed"])
        out.append(f"| {cat} | {p} | {len(rs)} | {100*p/len(rs):.0f}% |")
    out.append("")

    # by capability (maps to acceptance criteria / quotation requirements)
    caps: dict[str, list[dict]] = {}
    for r in records:
        caps.setdefault(r.get("capability", r["category"]), []).append(r)
    if set(caps) != set(cats):
        out.append("## Pass rate by capability\n")
        out.append("| Capability | Passed | Total | Rate |")
        out.append("|---|---|---|---|")
        for cap in sorted(caps):
            rs = caps[cap]
            p = sum(1 for r in rs if r["grade"]["passed"])
            out.append(f"| {cap} | {p} | {len(rs)} | {100*p/len(rs):.0f}% |")
        out.append("")

    out.append("## Weakness flags\n")
    if flags:
        out.append("| Flag | Count | Meaning |")
        out.append("|---|---|---|")
        meaning = {
            "false_no_answer": "declined a question that IS answerable from the corpus",
            "missing_terms": "answered, but the expected fact was absent",
            "no_citations": "claimed a supported answer with no citations",
            "unexpected_source": "did not cite the document that holds the answer",
            "hallucination": "answered an unanswerable / false-premise question",
            "leaked_forbidden_term": "answer contained a term it should never produce",
            "raw_slice_artifact": "raw retrieval text leaked into the answer",
            "citation_validation_failed": "backend flagged the citations as invalid",
            "server_error": "answer status was 'error'",
            "http_error": "non-200 HTTP response",
            "request_failed": "no response (timeout / connection)",
        }
        for k, v in sorted(flags.items(), key=lambda x: -x[1]):
            out.append(f"| `{k}` | {v} | {meaning.get(k,'')} |")
    else:
        out.append("None. ✅")
    out.append("")

    out.append("## Determinism across retries\n")
    if nondeterministic:
        out.append("These questions gave **different** status / citations / pass across retries — a stability concern:\n")
        for iid, query, statuses, passes in nondeterministic:
            out.append(f"- `{iid}` — \"{query}\" → statuses={statuses}, pass={passes}")
    else:
        out.append("All repeated questions were stable across retries. ✅")
    out.append("")

    out.append("## Failures in detail\n")
    fails = [r for r in records if not r["grade"]["passed"]]
    if not fails:
        out.append("No failures. ✅")
    else:
        for r in fails:
            resp = r["chat_response"] or {}
            ans = (resp.get("answer") or "").replace("\n", " ")
            out.append(f"### `{r['intent_id']}` · {r['category']} · attempt {r['attempt']}")
            out.append(f"- **Query:** {r['query']}")
            out.append(f"- **Flags:** {', '.join(r['grade']['flags']) or '—'}")
            out.append(f"- **Notes:** {'; '.join(r['grade']['notes']) or '—'}")
            out.append(f"- **Status:** {resp.get('status')} · confidence: {resp.get('confidence')} · citations: {resp.get('citations')}")
            out.append(f"- **Answer:** {ans[:400]}")
            # show the top retrieval so you can see if it was retrieval or generation
            retr = r.get("search_retrieval") or []
            if retr:
                top = retr[0]
                out.append(f"- **Top retrieved:** {top.get('source_file','?')} (score {top.get('combined_score','?')})")
            out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ariadne evaluation probe")
    ap.add_argument("--bank", default="question_bank",
                    help="question-bank module to load (e.g. question_bank, stress_question_bank)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--retries", type=int, default=2, help="calls per question variant")
    ap.add_argument("--answer-mode", default="balanced", choices=["brief", "balanced", "detailed"])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--only-category", default=None)
    ap.add_argument("--max-intents", type=int, default=None)
    ap.add_argument("--no-search", action="store_true", help="skip the raw retrieval probe")
    ap.add_argument("--keep-chats", action="store_true", help="do not delete probe chats afterwards")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
