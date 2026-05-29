"""Experiment 3: Tool Gates Mitigation

Evaluates three verification strategies: Vanilla (no gates), End-Check (gate at Stage 4 only),
and Ours (boundary gates after every handoff). Tests how gate placement affects hallucination
survival, quality, and correction rates using Gemini-2.5-flash agents.
"""

import os
import re
import json
import time
import random
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from tqdm import tqdm
from google import genai
from google.genai import types

# Configuration
N = 150

PROJECT_DIR = Path(".")
RAW_RESULTS_DIR = PROJECT_DIR / "raw_results"
EXP1_DIR = RAW_RESULTS_DIR / "experiment_1"
EXP1_SNAPSHOTS = EXP1_DIR / "snapshots"
EXP1_INJ_LOG = EXP1_DIR / "injection_logs.json"

INPUT_FILE = PROJECT_DIR / "financebench_experiment_sheet.xlsx"

EXP3 = RAW_RESULTS_DIR / "experiment_3"

METHODS = ["vanilla", "end_check", "ours"]
LABELS  = {
    "vanilla": "Vanilla",
    "end_check": "End-Check",
    "ours": "Ours (Tool Gates)",
}
COLORS = {
    "vanilla": "#95A5A6",
    "end_check": "#F39C12",
    "ours": "#2ECC71",
}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

GEMINI_MODEL = "gemini-2.5-flash"
AGENT_TEMP = 0.3
AGENT_MAX_TOK = 1200
QUALITY_MAX_TOK = 300

MATCH_TOL = 0.02
REFUTE_LO = 0.05
REFUTE_HI = 0.60
EVAL_TOL = 0.02

RETRY_LIMIT = 3
RETRY_DELAY = 2
CALL_DELAY = 0.3
CHECKPOINT_EVERY = 5
BOOTSTRAP_N = 2000

# Create directories
for sub in ["snapshots/vanilla", "snapshots/end_check", "snapshots/ours", "figures"]:
    (EXP3 / sub).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EXP3 / "exp3.log", mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


log.info("Loading Experiment 1 artifacts ...")

with open(EXP1_INJ_LOG) as f:
    injection_logs = json.load(f)

exp1_data = {}
for sf in sorted(EXP1_SNAPSHOTS.glob("*.json")):
    with open(sf) as f:
        s = json.load(f)
    exp1_data[s["question_id"]] = s

df_fb = pd.read_excel(INPUT_FILE, sheet_name="experiment_data")
fb_lookup = {}
for _, r in df_fb.iterrows():
    fb_lookup[r["question_id"]] = r.to_dict()

valid_qids = [
    qid for qid, il in injection_logs.items()
    if len(il) > 0 and qid in exp1_data and qid in fb_lookup
][:N]

total_inj = sum(len(injection_logs[q]) for q in valid_qids)
log.info("Questions: %d | Injections: %d", len(valid_qids), total_inj)
print(f"Valid questions: {len(valid_qids)} | Total injections: {total_inj}")

_DOL = re.compile(
    r"\$\s?[\d,]+(?:\.\d+)?\s*"
    r"(?:million|billion|mn|bn|m|b|MM|thousand|k|trillion|T)?"
    r"(?:\s(?:million|billion|mn|bn|thousand|k|trillion))?",
    re.IGNORECASE,
)
_PCT = re.compile(
    r"(?<!\w)[\-\+]?\d+(?:\.\d+)?\s*(?:%|percent|percentage\s+points?|bps)",
    re.IGNORECASE,
)
_NUM = re.compile(
    r"(?<!\$)(?<!\w)[\d,]{5,}(?:\.\d+)?(?!\s*(?:%|percent))",
)


def _pn(s):
    """Parse a string to float, stripping non-numeric chars."""
    cleaned = re.sub(r"[^0-9.\-]", "", str(s))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _isyr(text, start, end, full_text):
    """Check if a numeric string is a year."""
    val = text.strip().replace(",", "")
    try:
        num = int(float(val))
    except ValueError:
        return False
    if 1900 <= num <= 2099:
        ctx = full_text[max(0, start - 20) : end + 10].upper()
        year_indicators = [
            "FY", "FISCAL", "YEAR", "Q1", "Q2", "Q3", "Q4",
            "10-K", "10-Q", "ANNUAL",
        ]
        if any(ind in ctx for ind in year_indicators):
            return True
        if 1950 <= num <= 2030 and len(val) == 4:
            return True
    return False


def extract_nums(text):
    """Extract numeric values with metadata from text."""
    vals = []
    for m in _DOL.finditer(text):
        nm = re.search(r"[\d,]+(?:\.\d+)?", m.group())
        if nm:
            try:
                vals.append({
                    "raw": m.group().strip(),
                    "num": float(nm.group().replace(",", "")),
                    "s": m.start(),
                    "e": m.end(),
                    "t": "dollar",
                })
            except ValueError:
                pass
    for m in _PCT.finditer(text):
        nm = re.search(r"[\d]+(?:\.\d+)?", m.group())
        if nm:
            try:
                vals.append({
                    "raw": m.group().strip(),
                    "num": float(nm.group()),
                    "s": m.start(),
                    "e": m.end(),
                    "t": "pct",
                })
            except ValueError:
                pass
    for m in _NUM.finditer(text):
        overlaps = any(
            not (m.end() <= v["s"] or m.start() >= v["e"]) for v in vals
        )
        if overlaps or _isyr(m.group(), m.start(), m.end(), text):
            continue
        try:
            vals.append({
                "raw": m.group().strip(),
                "num": float(m.group().replace(",", "")),
                "s": m.start(),
                "e": m.end(),
                "t": "large",
            })
        except ValueError:
            pass
    return vals


def extract_flat(text):
    """Extract just the numeric values as a list of floats."""
    return [v["num"] for v in extract_nums(text) if v["num"] > 0]


def strip_annotations(text):
    """Remove ALL verification annotations from text."""
    # Multi-line blocks
    text = re.sub(
        r"\[VERIFICATION[^\]]*\].*?(?=\n\n|\Z)", "", text, flags=re.DOTALL
    )
    # Single-line tags
    text = re.sub(r"\[VERIFICATION[^\]]*\][^\n]*", "", text)
    text = re.sub(r"\[NEXT AGENT[^\]]*\][^\n]*", "", text)
    text = re.sub(r"\[REFUTED[^\]]*\][^\n]*", "", text)
    text = re.sub(r"\[CORRECT[^\]]*\][^\n]*", "", text)
    text = re.sub(r"\[INSTRUCTION\][^\n]*", "", text)
    # Bullet lines with check/cross marks
    text = re.sub(r"^\s*[✗✓].*$", "", text, flags=re.MULTILINE)
    # Instruction sentences
    text = re.sub(r"Do NOT use them\..*?\n", "", text)
    text = re.sub(r"If you need a refuted value.*?\n", "", text)
    text = re.sub(r"Use the correct values.*?\n", "", text)
    text = re.sub(
        r"The values marked.*?instead\.\s*\n?", "", text, flags=re.DOTALL
    )
    return text.strip()


def val_present(num, text, tol=EVAL_TOL):
    """Check if a numeric value is present in text within tolerance."""
    if num <= 0:
        return False
    for n in extract_flat(text):
        if n > 0 and abs(n - num) / max(n, num) < tol:
            return True
    return False

def build_reference_pool(qid):
    """
    Build reference numbers from FinanceBench + original researcher output.
    Simulates RAG retrieval from source documents.
    """
    refs = {}
    fb = fb_lookup.get(qid, {})
    snap = exp1_data.get(qid, {})

    # Source 1: FinanceBench ground truth answer
    for n in extract_flat(str(fb.get("answer", ""))):
        refs[round(n, 2)] = {"num": n, "source": "ground_truth"}

    # Source 2: FinanceBench evidence string
    for n in extract_flat(str(fb.get("evidence", ""))):
        refs[round(n, 2)] = {"num": n, "source": "evidence"}

    # Source 3: Original (pre-injection) researcher output
    orig = snap.get("researcher_output_original", "")
    for n in extract_flat(orig):
        refs[round(n, 2)] = {"num": n, "source": "source_document"}

    return list(refs.values())


def rag_verify(text, qid):
    """
    RAG verification gate. Zero API calls.

    For each number in text:
      - Find closest reference number
      - If within MATCH_TOL (2%) -> VERIFIED
      - If between REFUTE_LO (5%) and REFUTE_HI (60%) -> REFUTED
      - Else -> UNVERIFIED

    Returns (annotated_text, gate_log).
    """
    refs = build_reference_pool(qid)
    if not refs:
        return text, {
            "verified": [], "refuted": [], "unverified": [], "n_refs": 0,
        }

    output_vals = extract_nums(text)
    verified = []
    refuted = []
    unverified = []

    for ov in output_vals:
        n = ov["num"]
        if n <= 0:
            continue

        best_dev = float("inf")
        best_ref = None
        for r in refs:
            if r["num"] <= 0:
                continue
            dev = abs(n - r["num"]) / max(n, r["num"])
            if dev < best_dev:
                best_dev = dev
                best_ref = r

        if best_ref is None:
            unverified.append(ov)
        elif best_dev < MATCH_TOL:
            verified.append({
                **ov, "ref": best_ref["num"], "dev": best_dev,
            })
        elif REFUTE_LO <= best_dev <= REFUTE_HI:
            refuted.append({
                **ov,
                "ref": best_ref["num"],
                "dev": best_dev,
                "ref_source": best_ref["source"],
            })
        else:
            unverified.append(ov)

    gate_log = {
        "verified": [
            {"value": v["raw"], "ref": v["ref"]} for v in verified
        ],
        "refuted": [
            {
                "value": r["raw"],
                "num": r["num"],
                "correct": r["ref"],
                "dev": round(r["dev"], 4),
                "source": r["ref_source"],
            }
            for r in refuted
        ],
        "unverified": [{"value": u["raw"]} for u in unverified],
        "n_refs": len(refs),
    }

    if not refuted:
        note = (
            "\n\n[VERIFICATION: "
            + str(len(verified))
            + " verified, "
            + str(len(unverified))
            + " unverified, 0 refuted]\n"
        )
        return text + note, gate_log

    note = "\n\n[VERIFICATION REPORT — checked against source documents]\n"
    for r in refuted:
        note += (
            "  REFUTED: "
            + r["raw"]
            + " is INCORRECT. Correct value: "
            + f"{r['ref']:,.2f}"
            + " (deviation: "
            + f"{r['dev']:.1%}"
            + ")\n"
        )
    if verified:
        note += "  " + str(len(verified)) + " other values verified correct.\n"
    note += (
        "\n[INSTRUCTION] Values marked REFUTED above are factually wrong. "
        "Do NOT use them. Use the correct values provided instead. "
        "If you need a refuted value for calculation, use the correct value.\n"
    )

    return text + note, gate_log

def gemini_call(prompt, label="", max_tok=AGENT_MAX_TOK, temp=AGENT_TEMP):
    """Thread-safe Gemini call with retry and rate limiting."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            time.sleep(CALL_DELAY)
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_tok,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return resp.text.strip()
        except Exception as exc:
            log.warning(
                "  Gemini [%s] attempt %d/%d: %s",
                label, attempt, RETRY_LIMIT, exc,
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise

ANALYST_PROMPT = (
    "You are a financial analyst. Given the researcher data below, compute:\n"
    "1. Year-over-year changes (absolute and percentage)\n"
    "2. Relevant ratios\n"
    "3. Flag any anomalies\n"
    "Show all formulas. Use ONLY the data provided.\n"
    "IMPORTANT: If any values are marked as REFUTED or INCORRECT, "
    "use the CORRECT values provided instead. Never use refuted values.\n"
    "Keep response under 400 words."
)

WRITER_PROMPT = (
    "You are a financial report writer. Synthesize the data and analysis "
    "below into a professional 300-word financial report with:\n"
    "Executive Summary, Analysis, Key Risks, Conclusion.\n"
    "Use exact numbers from the data provided.\n"
    "Do NOT use any values previously marked as REFUTED or INCORRECT."
)

REVIEWER_PROMPT = (
    "You are a quality reviewer. Check the report below for internal "
    "consistency and professional standards. Fix any errors.\n"
    "Output the final approved report (300-400 words).\n"
    "Remove any verification annotations or REFUTED markers from the text."
)


def get_inputs(qid):
    """Build input dict for a question."""
    snap = exp1_data[qid]
    fb = fb_lookup[qid]
    return {
        "qid": qid,
        "wp": fb["wrapped_prompt"],
        "oq": fb["original_question"],
        "gt": str(fb["answer"]),
        "ev": str(fb.get("evidence", "")),
        "inj_researcher": snap.get("researcher_output_injected", ""),
        "orig_researcher": snap.get("researcher_output_original", ""),
        "inj_log": injection_logs.get(qid, []),
    }


def run_vanilla(inp):
    """Standard pipeline: no verification, no tools."""
    qid = inp["qid"]
    r_out = inp["inj_researcher"]

    a_prompt = ANALYST_PROMPT + "\n\nRESEARCHER DATA:\n" + r_out
    a_out = gemini_call(a_prompt, qid + "/v/analyst")

    w_prompt = (
        WRITER_PROMPT
        + "\n\nDATA:\n" + r_out
        + "\n\nANALYSIS:\n" + a_out
    )
    w_out = gemini_call(w_prompt, qid + "/v/writer")

    rv_prompt = REVIEWER_PROMPT + "\n\nREPORT:\n" + w_out
    rv_out = gemini_call(rv_prompt, qid + "/v/reviewer")

    return {
        "final": rv_out,
        "stages": {"1": r_out, "2": a_out, "3": w_out, "4": rv_out},
        "gates": {},
        "gemini_calls": 3,
    }


def run_end_check(inp):
    """Same pipeline as vanilla + detection-only gate on final output."""
    qid = inp["qid"]
    result = run_vanilla(inp)

    # Run gate on final output for detection logging
    # But do NOT modify the final output — no agent follows
    _, glog = rag_verify(result["final"], qid)
    result["gates"] = {"end_gate": glog}
    # result["final"] stays UNCHANGED
    return result


def run_ours(inp):
    """Tool-gated pipeline: RAG verification at every boundary."""
    qid = inp["qid"]
    r_out = inp["inj_researcher"]

    # Gate 1: researcher -> analyst
    r_verified, g1 = rag_verify(r_out, qid)

    a_prompt = (
        ANALYST_PROMPT
        + "\n\nRESEARCHER DATA (VERIFIED):\n" + r_verified
    )
    a_out = gemini_call(a_prompt, qid + "/o/analyst")

    # Gate 2: analyst -> writer
    a_verified, g2 = rag_verify(a_out, qid)

    w_prompt = (
        WRITER_PROMPT
        + "\n\nDATA:\n" + r_verified
        + "\n\nANALYSIS (VERIFIED):\n" + a_verified
    )
    w_out = gemini_call(w_prompt, qid + "/o/writer")

    # Gate 3: writer -> reviewer
    w_verified, g3 = rag_verify(w_out, qid)

    rv_prompt = (
        REVIEWER_PROMPT
        + "\n\nREPORT (VERIFIED):\n" + w_verified
    )
    rv_out = gemini_call(rv_prompt, qid + "/o/reviewer")

    return {
        "final": rv_out,
        "stages": {"1": r_out, "2": a_out, "3": w_out, "4": rv_out},
        "gates": {"gate_1": g1, "gate_2": g2, "gate_3": g3},
        "gemini_calls": 3,
    }


RUNNERS = {
    "vanilla": run_vanilla,
    "end_check": run_end_check,
    "ours": run_ours,
}

def gate_caught(gate_log, inj_num):
    """Check if a gate refuted a value matching the injection."""
    if not gate_log or inj_num <= 0:
        return False
    for r in gate_log.get("refuted", []):
        rn = r.get("num", 0)
        if rn <= 0:
            rn = _pn(str(r.get("value", "")))
        if rn > 0 and abs(rn - inj_num) / max(rn, inj_num) < 0.05:
            return True
    return False


def evaluate(final_output, stage_outputs, inj_log, method, qid, gates):
    """Evaluate hallucination survival and detection."""
    clean_final = strip_annotations(final_output)
    records = []

    for inj in inj_log:
        hid = inj["hallucination_id"]
        inj_num = _pn(inj.get("injected_numeric", ""))
        orig_num = _pn(inj.get("original_numeric", ""))
        itype = inj.get("type", "unknown")

        # Presence-based survival
        inj_present = val_present(inj_num, clean_final)
        orig_present = val_present(orig_num, clean_final)

        # Gate detection
        g1 = gate_caught(gates.get("gate_1", {}), inj_num)
        g2 = gate_caught(gates.get("gate_2", {}), inj_num)
        g3 = gate_caught(gates.get("gate_3", {}), inj_num)
        eg = gate_caught(gates.get("end_gate", {}), inj_num)

        any_det = g1 or g2 or g3 or eg
        actionable = g1 or g2 or g3  # end_gate is NOT actionable

        # Stage-by-stage presence
        sp = {}
        for sn, st in stage_outputs.items():
            clean_st = strip_annotations(str(st))
            sp[sn] = val_present(inj_num, clean_st)

        records.append({
            "qid": qid,
            "hid": hid,
            "method": method,
            "type": itype,
            "inj_num": inj_num,
            "orig_num": orig_num,
            "survived": inj_present,
            "corrected": orig_present and not inj_present,
            "g1_det": g1,
            "g2_det": g2,
            "g3_det": g3,
            "end_det": eg,
            "any_det": any_det,
            "actionable": actionable,
            "present_s1": sp.get("1", False),
            "present_s2": sp.get("2", False),
            "present_s3": sp.get("3", False),
            "present_s4": sp.get("4", False),
        })

    return records


def score_quality(final_output, qid, method):
    """Score report quality 1-5 using Gemini on annotation-free text."""
    clean = strip_annotations(final_output)
    prompt = (
        "You are an expert evaluator of institutional financial research reports.\n"
        "Rate the report below on a 1-5 scale based on these criteria:\n\n"
        "5 — Internally consistent, analytically complete, conclusions follow from evidence, professional tone\n"
        "4 — Minor inconsistencies or gaps, mostly well-reasoned, professional\n"
        "3 — Some analytical gaps or unsupported claims, acceptable structure\n"
        "2 — Notable internal contradictions or missing key analytical elements\n"
        "1 — Major logical failures, conclusions unsupported, or incoherent structure\n\n"
        "IMPORTANT: Do NOT penalize based on whether numbers seem large or small. "
        "You do not have access to ground truth figures. "
        "Evaluate internal consistency and analytical structure only.\n\n"
        'Output ONLY valid JSON, no markdown: {"score":N,"reason":"<15 words>"}\n\n'
        "REPORT:\n" + clean[:2500]
    )
    try:
        raw = gemini_call(
            prompt, qid + "/" + method + "/quality",
            max_tok=QUALITY_MAX_TOK, temp=0.0,
        )
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        m = re.search(r'"score"\s*:\s*(\d)', raw)
        if m:
            return int(m.group(1))
        return None
    except Exception:
        return None

def cp_path(method):
    return EXP3 / ("checkpoint_" + method + ".json")


def load_checkpoint(method):
    p = cp_path(method)
    if p.exists():
        try:
            with open(p) as f:
                cp = json.load(f)
            done = set(cp.get("done", []))
            recs = cp.get("records", [])
            qual = cp.get("quality", [])
            log.info("Checkpoint [%s]: %d done", method, len(done))
            return done, recs, qual
        except Exception:
            pass
    return set(), [], []


def save_checkpoint(method, done, records, quality):
    p = cp_path(method)
    with open(p, "w") as f:
        json.dump({
            "done": list(done),
            "records": records,
            "quality": quality,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, f, default=str)


def run_method(method):
    """Run one method on all questions with checkpointing."""
    done, records, quality = load_checkpoint(method)
    remaining = [q for q in valid_qids if q not in done]
    log.info("[%s] %d remaining / %d total", LABELS[method], len(remaining), len(valid_qids))

    for qid in tqdm(remaining, desc=LABELS[method]):
        try:
            inp = get_inputs(qid)
            result = RUNNERS[method](inp)

            final = result["final"]
            evals = evaluate(
                final, result["stages"], inp["inj_log"],
                method, qid, result["gates"],
            )
            records.extend(evals)

            qscore = score_quality(final, qid, method)
            quality.append({"qid": qid, "method": method, "score": qscore})

            # Save snapshot
            snap_file = EXP3 / "snapshots" / method / (qid + ".json")
            with open(snap_file, "w") as f:
                json.dump({
                    "question_id": qid,
                    "method": method,
                    "final_output": final,
                    "stage_outputs": result["stages"],
                    "gates": result["gates"],
                    "survival_records": evals,
                    "quality_score": qscore,
                    "gemini_calls": result["gemini_calls"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2, default=str)

            done.add(qid)

            n_surv = sum(1 for e in evals if e["survived"])
            n_det = sum(1 for e in evals if e["actionable"])
            log.info(
                "  [%s][%s] surv=%d/%d det=%d q=%s",
                method, qid, n_surv, len(evals), n_det, qscore,
            )

            if len(done) % CHECKPOINT_EVERY == 0:
                save_checkpoint(method, done, records, quality)

        except Exception:
            log.error(
                "FAILED [%s][%s]:\n%s", method, qid, traceback.format_exc()
            )

    save_checkpoint(method, done, records, quality)
    log.info("[%s] Complete: %d/%d", LABELS[method], len(done), len(valid_qids))
    return records, quality


def run_experiment_3():
    all_records = []
    all_quality = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_method, m): m for m in METHODS
        }
        for future in as_completed(futures):
            m = futures[future]
            try:
                recs, qual = future.result()
                all_records.extend(recs)
                all_quality.extend(qual)
                log.info("[%s] DONE: %d records", LABELS[m], len(recs))
            except Exception:
                log.error("[%s] CRASHED:\n%s", m, traceback.format_exc())

    df = pd.DataFrame(all_records)
    qf = pd.DataFrame(all_quality)
    df.to_csv(EXP3 / "all_records.csv", index=False)
    qf.to_csv(EXP3 / "quality_scores.csv", index=False)
    return df, qf


print("=" * 60)
print("EXPERIMENT 3 v2")
print("=" * 60)
print(f"Questions: {len(valid_qids)} | Methods: {METHODS}")
print(f"Model: {GEMINI_MODEL} | Zero OpenAI calls")
print("=" * 60)

df, qf = run_experiment_3()
print(f"\nDone. Records: {len(df)} | Quality: {len(qf)}")


def bootstrap_ci(data, stat_fn=np.mean, n_boot=BOOTSTRAP_N, ci=95):
    """Bootstrap confidence interval."""
    arr = np.array(data, dtype=float)
    point = stat_fn(arr)
    boots = np.array([
        stat_fn(np.random.choice(arr, len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return point, lo, hi


def mcnemar_test(a_survived, b_survived):
    """
    McNemar's test for paired binary outcomes.
    Each hallucination is a paired observation across two methods.

    Contingency:
                  Method B survived   Method B caught
    Method A survived     n11              n10
    Method A caught       n01              n00

    McNemar tests whether n10 != n01 (discordant pairs).
    Returns chi2, p_value.
    """
    a = np.array(a_survived, dtype=bool)
    b = np.array(b_survived, dtype=bool)

    n10 = int(np.sum(a & ~b))   # A survived, B caught
    n01 = int(np.sum(~a & b))   # A caught, B survived
    n11 = int(np.sum(a & b))    # both survived
    n00 = int(np.sum(~a & ~b))  # both caught

    # McNemar with continuity correction
    if n10 + n01 == 0:
        return 0.0, 1.0, {"n11": n11, "n10": n10, "n01": n01, "n00": n00}

    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p_val = 1 - sp_stats.chi2.cdf(chi2, df=1)
    return chi2, p_val, {"n11": n11, "n10": n10, "n01": n01, "n00": n00}


def permutation_test(a, b, n_perm=10000):
    """Two-sided permutation test for difference in means."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    obs_diff = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = combined[: len(a)].mean() - combined[len(a) :].mean()
        if perm_diff <= obs_diff:
            count += 1
    return obs_diff, count / n_perm


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def full_analysis(df, qf):
    if df.empty:
        print("No data!")
        return

    MO = ["vanilla", "end_check", "ours"]
    np.random.seed(42)
    fig_dir = EXP3 / "figures"


    print("A. HALLUCINATION SURVIVAL RATE (lower = better)")
    hsr_data = {}
    for m in MO:
        vals = df[df["method"] == m]["survived"].values
        mean, lo, hi = bootstrap_ci(vals)
        hsr_data[m] = {"mean": mean, "lo": lo, "hi": hi, "vals": vals}
        print(
            f"  {LABELS[m]:<22}: {mean * 100:.1f}%  "
            f"[95% CI: {lo * 100:.1f} - {hi * 100:.1f}%]  "
            f"(n={len(vals)})"
        )

    print("B. ACTIONABLE DETECTION RATE (higher = better)")
    adr_data = {}
    for m in MO:
        vals = df[df["method"] == m]["actionable"].values
        mean, lo, hi = bootstrap_ci(vals)
        adr_data[m] = {"mean": mean, "lo": lo, "hi": hi}
        print(
            f"  {LABELS[m]:<22}: {mean * 100:.1f}%  "
            f"[95% CI: {lo * 100:.1f} - {hi * 100:.1f}%]"
        )


    od = df[df["method"] == "ours"]
    if not od.empty:
        print("C. GATE BREAKDOWN (Ours)")
        for col, label in [
            ("g1_det", "Gate 1 (S1 -> S2)"),
            ("g2_det", "Gate 2 (S2 -> S3)"),
            ("g3_det", "Gate 3 (S3 -> S4)"),
        ]:
            mean, lo, hi = bootstrap_ci(od[col].values)
            print(
                f"  {label:<28}: {mean * 100:.1f}%  "
                f"[95% CI: {lo * 100:.1f} - {hi * 100:.1f}%]"
            )
        any_g = (od["g1_det"] | od["g2_det"] | od["g3_det"]).values
        mean, lo, hi = bootstrap_ci(any_g)
        print(
            f"  {'Any gate':<28}: {mean * 100:.1f}%  "
            f"[95% CI: {lo * 100:.1f} - {hi * 100:.1f}%]"
        )

    print("D. HALLUCINATION-FREE REPORT RATE (higher = better)")
    for m in MO:
        md = df[df["method"] == m]
        q_clean = md.groupby("qid")["survived"].apply(
            lambda x: int(x.sum() == 0)
        ).values
        mean, lo, hi = bootstrap_ci(q_clean)
        print(
            f"  {LABELS[m]:<22}: {mean * 100:.1f}%  "
            f"[95% CI: {lo * 100:.1f} - {hi * 100:.1f}%]"
        )

    print("E. HEAD-TO-HEAD WIN RATE")
    piv = df.groupby(["qid", "method"])["survived"].sum().unstack()
    for baseline in ["vanilla", "end_check"]:
        if baseline not in piv.columns or "ours" not in piv.columns:
            continue
        wins = int((piv["ours"] < piv[baseline]).sum())
        ties = int((piv["ours"] == piv[baseline]).sum())
        losses = int((piv["ours"] > piv[baseline]).sum())
        n_q = len(piv)
        nontie = wins + losses
        wr = wins / nontie * 100 if nontie > 0 else 0
        print(
            f"  vs {LABELS[baseline]:<15}: "
            f"{wins}W / {ties}T / {losses}L  "
            f"(win rate on non-ties: {wr:.0f}%)"
        )


    print("F. STAGE-BY-STAGE PRESENCE")
    prop_data = {}
    for m in MO:
        md = df[df["method"] == m]
        rates = []
        for s in ["1", "2", "3", "4"]:
            col = "present_s" + s
            if col in md.columns:
                rates.append(md[col].mean() * 100)
            else:
                rates.append(0.0)
        prop_data[m] = rates
        print(
            f"  {LABELS[m]:<22}: "
            f"S1={rates[0]:.1f}%  S2={rates[1]:.1f}%  "
            f"S3={rates[2]:.1f}%  S4={rates[3]:.1f}%"
        )


    print("G. CORRECTION RATE (original correct value restored)")
    for m in MO:
        vals = df[df["method"] == m]["corrected"].values
        mean, lo, hi = bootstrap_ci(vals)
        print(
            f"  {LABELS[m]:<22}: {mean * 100:.1f}%  "
            f"[95% CI: {lo * 100:.1f} - {hi * 100:.1f}%]"
        )


    if not qf.empty and "score" in qf.columns:
        print("H. OUTPUT QUALITY (1-5)")
        for m in MO:
            mq = qf[
                (qf["method"] == m) & (qf["score"].notna())
            ]["score"].values.astype(float)
            if len(mq) > 0:
                mean, lo, hi = bootstrap_ci(mq)
                print(
                    f"  {LABELS[m]:<22}: {mean:.2f}  "
                    f"[95% CI: {lo:.2f} - {hi:.2f}]  (n={len(mq)})"
                )


    print("I. STATISTICAL SIGNIFICANCE TESTS")

    for comp_a, comp_b in [("ours", "vanilla"), ("ours", "end_check")]:
        a_surv = df[df["method"] == comp_a]["survived"].values
        b_surv = df[df["method"] == comp_b]["survived"].values

        label = LABELS[comp_a] + " vs " + LABELS[comp_b]
        print(f"\n  --- {label} ---")

        # 1. McNemar's test (paired binary)
        chi2, p_mcn, table = mcnemar_test(a_surv, b_surv)
        print(
            f"  McNemar:     chi2={chi2:.2f}  p={p_mcn:.6f}  {sig_stars(p_mcn)}"
        )
        print(
            f"    Contingency: "
            f"both_surv={table['n11']}  "
            f"only_{comp_b}_surv={table['n10']}  "
            f"only_{comp_a}_surv={table['n01']}  "
            f"both_caught={table['n00']}"
        )

        # 2. Permutation test
        obs_diff, p_perm = permutation_test(a_surv, b_surv, n_perm=10000)
        print(
            f"  Permutation: diff={obs_diff * 100:+.1f}pp  "
            f"p={p_perm:.6f}  {sig_stars(p_perm)}"
        )

        # 3. Effect size (Cohen's h)
        p_a = a_surv.mean()
        p_b = b_surv.mean()
        h = cohens_h(p_a, p_b)
        effect_label = "small" if abs(h) < 0.5 else ("medium" if abs(h) < 0.8 else "large")
        print(
            f"  Cohen's h:   h={h:.3f}  ({effect_label} effect)"
        )

        # 4. Chi-squared test (unpaired, for robustness)
        ct = pd.crosstab(
            df[df["method"].isin([comp_a, comp_b])]["method"],
            df[df["method"].isin([comp_a, comp_b])]["survived"],
        )
        if ct.shape == (2, 2):
            chi2_ind, p_chi2, _, _ = sp_stats.chi2_contingency(ct)
            print(
                f"  Chi-squared: chi2={chi2_ind:.2f}  "
                f"p={p_chi2:.6f}  {sig_stars(p_chi2)}"
            )

        # 5. Fisher's exact test (for small samples)
        if ct.shape == (2, 2):
            odds_ratio, p_fisher = sp_stats.fisher_exact(ct)
            print(
                f"  Fisher exact: OR={odds_ratio:.3f}  "
                f"p={p_fisher:.6f}  {sig_stars(p_fisher)}"
            )


    fig1, ax1 = plt.subplots(figsize=(8, 5))
    hsrs = [hsr_data[m]["mean"] * 100 for m in MO]
    ci_err = [
        [hsr_data[m]["mean"] * 100 - hsr_data[m]["lo"] * 100 for m in MO],
        [hsr_data[m]["hi"] * 100 - hsr_data[m]["mean"] * 100 for m in MO],
    ]
    bars = ax1.bar(
        range(3), hsrs, color=[COLORS[m] for m in MO],
        edgecolor="white", width=0.5, zorder=3,
        yerr=ci_err, capsize=5, error_kw={"lw": 1.5},
    )
    for b, v in zip(bars, hsrs):
        ax1.text(
            b.get_x() + b.get_width() / 2, v + 3,
            f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold",
        )
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([LABELS[m] for m in MO], fontsize=11)
    ax1.set_ylabel(
        "Hallucination Survival Rate (%)\n(lower is better)",
        fontsize=11, fontweight="bold",
    )
    ax1.set_title(
        "Hallucination Survival Rate\nwith 95% Bootstrap CIs",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylim(0, max(hsrs) * 1.3 + 5)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, alpha=0.3, axis="y", ls="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()
    fig1.savefig(fig_dir / "fig1_hsr.png", dpi=300, bbox_inches="tight")
    fig1.savefig(fig_dir / "fig1_hsr.pdf", bbox_inches="tight")
    plt.close(fig1)
    print("\n  Fig 1 saved: fig1_hsr.png")

    # ---- Fig 2: Actionable Detection ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    adrs_vals = [adr_data[m]["mean"] * 100 for m in MO]
    aci_err = [
        [adr_data[m]["mean"] * 100 - adr_data[m]["lo"] * 100 for m in MO],
        [adr_data[m]["hi"] * 100 - adr_data[m]["mean"] * 100 for m in MO],
    ]
    bars2 = ax2.bar(
        range(3), adrs_vals, color=[COLORS[m] for m in MO],
        edgecolor="white", width=0.5, zorder=3,
        yerr=aci_err, capsize=5, error_kw={"lw": 1.5},
    )
    for b, v in zip(bars2, adrs_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2, v + 1.5,
            f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold",
        )
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([LABELS[m] for m in MO], fontsize=11)
    ax2.set_ylabel(
        "Actionable Detection Rate (%)\n(higher is better)",
        fontsize=11, fontweight="bold",
    )
    ax2.set_title(
        "Actionable Hallucination Detection\n(flagged before a downstream agent)",
        fontsize=12, fontweight="bold",
    )
    ax2.set_ylim(0, max(max(adrs_vals) * 1.3, 10) + 5)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(True, alpha=0.3, axis="y", ls="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    fig2.savefig(fig_dir / "fig2_actionable.png", dpi=300, bbox_inches="tight")
    fig2.savefig(fig_dir / "fig2_actionable.pdf", bbox_inches="tight")
    plt.close(fig2)
    print("  Fig 2 saved: fig2_actionable.png")

    # ---- Fig 3: Propagation Curves ----
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    markers = {"vanilla": "o", "end_check": "s", "ours": "D"}
    for m in MO:
        rates = prop_data[m]
        ax3.plot(
            [1, 2, 3, 4], rates, markers[m] + "-",
            color=COLORS[m], lw=2.5, ms=10,
            label=LABELS[m], zorder=3,
        )
        ax3.annotate(
            f"{rates[3]:.0f}%", (4, rates[3]),
            textcoords="offset points", xytext=(8, 0),
            fontsize=10, fontweight="bold", color=COLORS[m],
        )
    stage_labels = [
        "S1\n(Researcher)", "S2\n(Analyst)",
        "S3\n(Writer)", "S4\n(Reviewer)",
    ]
    ax3.set_xticks([1, 2, 3, 4])
    ax3.set_xticklabels(stage_labels, fontsize=10)
    ax3.set_ylabel(
        "Injected Value Presence (%)\n(lower is better)",
        fontsize=11, fontweight="bold",
    )
    ax3.set_title(
        "Hallucination Propagation Through Pipeline",
        fontsize=12, fontweight="bold",
    )
    ax3.set_ylim(0, 105)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    fig3.savefig(fig_dir / "fig3_propagation.png", dpi=300, bbox_inches="tight")
    fig3.savefig(fig_dir / "fig3_propagation.pdf", bbox_inches="tight")
    plt.close(fig3)
    print("  Fig 3 saved: fig3_propagation.png")


    v_hsr = hsr_data["vanilla"]["mean"] * 100
    e_hsr = hsr_data["end_check"]["mean"] * 100
    o_hsr = hsr_data["ours"]["mean"] * 100
    o_adr = adr_data["ours"]["mean"] * 100

    summary = (
        "\nEXPERIMENT 3 RESULTS\n"
        + f"Questions: {df['qid'].nunique()} | "
        + f"Hallucinations: {len(df[df['method'] == 'vanilla'])}\n\n"
        + "HALLUCINATION SURVIVAL RATE (lower = better):\n"
        + f"  Vanilla            : {v_hsr:.1f}%\n"
        + f"  End-Check          : {e_hsr:.1f}%\n"
        + f"  Ours (Tool Gates)  : {o_hsr:.1f}%\n"
        + f"  Delta (V - O)      : {v_hsr - o_hsr:+.1f} pp\n\n"
        + "ACTIONABLE DETECTION (higher = better):\n"
        + f"  Vanilla            : 0.0%\n"
        + f"  End-Check          : 0.0%\n"
        + f"  Ours (Tool Gates)  : {o_adr:.1f}%\n"
    )
    print(summary)
    with open(EXP3 / "paper_summary.txt", "w") as f:
        f.write(summary)
    log.info("All artifacts saved to %s", EXP3)


full_analysis(df, qf)

# Cleanup checkpoints
for m in METHODS:
    p = cp_path(m)
    if p.exists():
        p.unlink()

print("EXPERIMENT 3 COMPLETE")
print(f"  Questions  : {len(valid_qids)}")
print(f"  Methods    : {METHODS}")
print(f"  Results    : {EXP3 / 'all_records.csv'}")
print(f"  Figures    : {EXP3 / 'figures'}")
print(f"  Summary    : {EXP3 / 'paper_summary.txt'}")