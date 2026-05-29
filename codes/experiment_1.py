"""Experiment 1: Hallucination Snowball Effect - Detectability Decay

Injects hallucinations after Stage 1 and traces detectability decay across all 4 pipeline stages.
"""

import os
import re
import json
import time
import random
import hashlib
import logging
import traceback
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Configuration
PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "raw_results"
RESULTS_DIR = DATA_DIR / "experiment_1"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"
FIGURES_DIR = DATA_DIR / "figures"

INPUT_FILE = PROJECT_DIR / "financebench_experiment_sheet.xlsx"
DETECTION_FILE = RESULTS_DIR / "detection_results.csv"
INJECTION_LOG_FILE = RESULTS_DIR / "injection_logs.json"
CHECKPOINT_FILE = RESULTS_DIR / "experiment_checkpoint.json"

for d in [RESULTS_DIR, SNAPSHOTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

AGENT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.0

N = 150
RANDOM_SEED = 42
INJECTION_MIN = 2
INJECTION_MAX = 3
PERTURBATION_RANGE = (0.15, 0.40)
PCT_SHIFT_RANGE = (3, 12)
RETRY_LIMIT = 3
RETRY_DELAY = 2

random.seed(RANDOM_SEED)

LOG_FILE = RESULTS_DIR / "experiment.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

agent_llm = ChatOpenAI(model=AGENT_MODEL, temperature=AGENT_TEMPERATURE, max_tokens=1500)
judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE, max_tokens=2000)

logger.info("Config loaded. N=%d", N)



AGENT_PROMPTS = {
    "researcher": """You are a Senior Financial Researcher at a top-tier investment research firm.

ROLE & MANDATE:
You are the first analyst in a sequential research pipeline. Your output will be consumed by downstream analysts who have NO access to original source documents — they rely ENTIRELY on your work. This makes factual precision your highest obligation.

TASK:
Given a financial research directive, produce a comprehensive data extraction report.

REQUIREMENTS:
1. EXACT FIGURES: Report all financial figures with full precision as they appear in SEC filings (10-K, 10-Q). Include units (USD millions, billions, etc.), time periods, and line-item labels exactly as reported.
2. SOURCE ATTRIBUTION: For every figure you report, state the specific source (e.g., "per the FY2022 10-K, Consolidated Statements of Operations").
3. MULTI-PERIOD DATA: Where relevant, extract data for at least 2-3 fiscal years/quarters to enable trend analysis downstream.
4. CONTEXTUAL DATA: Include related metrics that provide context (e.g., if asked about COGS, also pull total revenue, gross profit, and prior-year COGS).
5. RAW DATA ONLY: Do NOT interpret, analyze, or editorialize. Present facts and figures. Leave analysis to the next stage.

OUTPUT FORMAT:
Structure your response as:
- COMPANY & PERIOD: [Identification]
- PRIMARY METRIC: [The exact figure requested with full attribution]
- SUPPORTING DATA: [Related metrics, multi-period data]
- DATA SOURCES: [Specific filings and statement locations referenced]

CRITICAL: Accuracy is paramount. If you are uncertain about a figure, explicitly state your uncertainty rather than presenting an approximation as fact.""",

    "analyst": """You are a Senior Financial Analyst at a leading institutional investment firm.

ROLE & MANDATE:
You are the second analyst in a sequential research pipeline. You receive a data extraction report from a Financial Researcher. You must treat the provided data as your working dataset, but maintain analytical rigor in your computations.

TASK:
Perform a thorough financial analysis using the data provided in the researcher's report.

REQUIREMENTS:
1. TREND ANALYSIS: Compute year-over-year or sequential changes for all key metrics. Express changes in both absolute and percentage terms.
2. RATIO COMPUTATION: Calculate relevant financial ratios as appropriate to the research directive.
3. BENCHMARKING: Where possible, contextualize performance against known industry averages or sector norms.
4. ANOMALY IDENTIFICATION: Flag any unusual movements, outliers, or inconsistencies in the data.
5. SHOW YOUR WORK: For every computed figure, show the exact formula and input values used.

OUTPUT FORMAT:
- KEY METRICS SUMMARY: [Primary figures from researcher, restated]
- COMPUTED ANALYSIS: [All derived metrics with formulas shown]
- TREND ASSESSMENT: [Direction and magnitude of changes]
- CONTEXTUAL BENCHMARKS: [Industry comparisons]
- NOTABLE FLAGS: [Any anomalies or points requiring attention]

CRITICAL: Every number you produce must be traceable to the input data or a clearly shown computation. Do not introduce figures not derived from the researcher's report.""",

    "writer": """You are a Senior Financial Report Writer at an elite investment advisory firm.

ROLE & MANDATE:
You are the third professional in a sequential research pipeline. You receive a data report and an analytical assessment. Your task is to synthesize these into a publication-ready financial analysis report.

TASK:
Produce a professional, narrative-driven financial analysis report suitable for institutional investors.

REQUIREMENTS:
1. EXECUTIVE FRAMING: Open with a concise executive summary stating the key finding upfront.
2. NARRATIVE INTEGRATION: Weave quantitative findings into a coherent narrative. Write in professional prose.
3. PRECISE LANGUAGE: Use exact numbers from the provided analysis. Do not round or approximate.
4. CONTEXTUAL DEPTH: Connect financial metrics to operational drivers where the data supports it.
5. BALANCED TONE: Present findings objectively.
6. STRUCTURE: Executive Summary, Detailed Analysis, Key Risks & Considerations, Conclusion.

OUTPUT FORMAT:
A 300-500 word professional financial analysis report.

CRITICAL: Every figure and claim must originate from the researcher's or analyst's data. Do not introduce new data points or fabricate contextual claims.""",

    "reviewer": """You are a Managing Director and Head of Quality Assurance at a leading financial research institution.

ROLE & MANDATE:
You are the final reviewer in a sequential research pipeline. You receive a draft financial analysis report. Perform a rigorous quality review and produce the final approved version.

REQUIREMENTS:
1. NUMERICAL CONSISTENCY: Verify all figures are internally consistent. Check derived figures match base numbers.
2. LOGICAL COHERENCE: Ensure conclusions follow from evidence. Flag logical leaps.
3. COMPLETENESS: Verify the report addresses all aspects of the directive.
4. PROFESSIONAL STANDARDS: Ensure tone and formatting meet publication standards.
5. EXPLICIT CORRECTIONS: Document any changes made.

OUTPUT FORMAT:
- REVIEW FINDINGS: [Issues found]
- CORRECTIONS APPLIED: [Changes with justification]
- FINAL APPROVED REPORT: [Complete corrected report]
- QUALITY VERDICT: [APPROVED / APPROVED WITH REVISIONS / REQUIRES MAJOR REVISION]

CRITICAL: Your review is based solely on internal consistency. You do NOT have access to original source documents. You cannot verify base figures — only internal consistency.""",
}

logger.info("Agent prompts loaded — %d agents", len(AGENT_PROMPTS))



_YEAR_RE_EXCLUDE = re.compile(r'\b(?:FY\s?)?(?:19|20)\d{2}\b', re.IGNORECASE)

_DOLLAR_RE = re.compile(
    r'\$\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|mn|bn|m|b|MM|'
    r'thousand|k|trillion|T)?(?:\s(?:million|billion|mn|bn|'
    r'thousand|k|trillion))?',
    re.IGNORECASE,
)

_PCT_RE = re.compile(
    r'(?<!\w)[\-\+]?\d+(?:\.\d+)?\s*(?:%|percent|percentage\s+points?|bps)',
    re.IGNORECASE,
)

_NUM_RE = re.compile(
    r'(?<!\$)(?<!\w)[\d,]{5,}(?:\.\d+)?(?!\s*(?:%|percent))',
)


def _is_year(text: str, start: int, end: int, full_text: str) -> bool:
    value = text.strip().replace(',', '')
    try:
        num = int(float(value))
    except ValueError:
        return False
    if 1900 <= num <= 2099:
        ctx_start = max(0, start - 20)
        ctx = full_text[ctx_start:end + 10].upper()
        indicators = ['FY', 'FISCAL', 'YEAR', 'Q1', 'Q2', 'Q3', 'Q4',
                      '10-K', '10-Q', '20F', 'ANNUAL', 'QUARTER']
        if any(ind in ctx for ind in indicators):
            return True
        if 1950 <= num <= 2030 and len(value) == 4:
            return True
    return False


def _find_number_span(s: str):
    m = re.search(r'[\d,]+(?:\.\d+)?', s)
    if m:
        return m.start(), m.end(), m.group()
    return -1, -1, ""


def _perturb_dollar_or_number(value_str: str, seed: int):
    rng = random.Random(seed)
    start, end, num_str = _find_number_span(value_str)
    if start == -1:
        return value_str, "", 0.0

    clean = num_str.replace(',', '')
    try:
        original_num = float(clean)
    except ValueError:
        return value_str, "", 0.0
    if original_num == 0:
        return value_str, "", 0.0

    pmin, pmax = PERTURBATION_RANGE
    factor = rng.uniform(pmin, pmax)
    direction = rng.choice([-1, 1])
    new_num = original_num * (1 + direction * factor)

    has_commas = ',' in num_str
    has_dec = '.' in num_str

    if has_dec:
        dp = len(num_str.split('.')[1])
        new_str = f"{new_num:,.{dp}f}" if has_commas else f"{new_num:.{dp}f}"
    else:
        r = int(round(new_num))
        new_str = f"{r:,}" if has_commas else str(r)

    new_value = value_str[:start] + new_str + value_str[end:]
    return new_value, new_str, direction * factor


def _perturb_percentage(value_str: str, seed: int):
    rng = random.Random(seed)
    m = re.search(r'[\-\+]?\d+(?:\.\d+)?', value_str)
    if not m:
        return value_str, "", 0.0

    orig = float(m.group())
    smin, smax = PCT_SHIFT_RANGE
    shift = rng.uniform(smin, smax) * rng.choice([-1, 1])
    new_val = round(max(0.01, orig + shift), 2)

    if '.' in m.group():
        dp = len(m.group().split('.')[1])
        new_str = f"{new_val:.{dp}f}"
    else:
        new_str = f"{new_val:.1f}"

    new_value = value_str[:m.start()] + new_str + value_str[m.end():]
    return new_value, new_str, shift


def inject_hallucinations(text, question_id, seed=RANDOM_SEED):
    rng = random.Random(seed)
    candidates = []

    for m in _DOLLAR_RE.finditer(text):
        if _is_year(m.group(), m.start(), m.end(), text):
            continue
        candidates.append({"type": "dollar_amount", "original": m.group(),
                           "start": m.start(), "end": m.end()})

    for m in _PCT_RE.finditer(text):
        candidates.append({"type": "percentage", "original": m.group(),
                           "start": m.start(), "end": m.end()})

    for m in _NUM_RE.finditer(text):
        overlaps = any(not (m.end() <= c["start"] or m.start() >= c["end"]) for c in candidates)
        if overlaps or _is_year(m.group(), m.start(), m.end(), text):
            continue
        candidates.append({"type": "large_number", "original": m.group(),
                           "start": m.start(), "end": m.end()})

    if not candidates:
        logger.warning("  [%s] No injectable candidates!", question_id)
        return text, []

    n_inj = min(rng.randint(INJECTION_MIN, INJECTION_MAX), len(candidates))
    selected = rng.sample(candidates, n_inj)
    selected.sort(key=lambda c: c["start"], reverse=True)

    injection_log = []
    injected_text = text

    for i, cand in enumerate(selected):
        sub_seed = seed + abs(hash(question_id)) % 100000 + i * 7919

        if cand["type"] == "percentage":
            new_val, new_num, delta = _perturb_percentage(cand["original"], sub_seed)
        else:
            new_val, new_num, delta = _perturb_dollar_or_number(cand["original"], sub_seed)

        if new_val.strip() == cand["original"].strip():
            new_val2, new_num2, delta2 = (
                _perturb_percentage(cand["original"], sub_seed + 999983)
                if cand["type"] == "percentage"
                else _perturb_dollar_or_number(cand["original"], sub_seed + 999983)
            )
            if new_val2.strip() != cand["original"].strip():
                new_val, new_num, delta = new_val2, new_num2, delta2

        if new_val.strip() == cand["original"].strip():
            continue

        injected_text = injected_text[:cand["start"]] + new_val + injected_text[cand["end"]:]

        prior = text[:cand["start"]]
        sent_idx = prior.count('.') + prior.count('!') + prior.count('?')

        # Extract core numeric for matching
        orig_core = re.sub(r'[^0-9.\-]', '', cand["original"])
        inj_core = re.sub(r'[^0-9.\-]', '', new_val)

        injection_log.append({
            "hallucination_id": f"{question_id}_H{i+1}",
            "type": cand["type"],
            "original_value": cand["original"],
            "injected_value": new_val,
            "original_numeric": orig_core,
            "injected_numeric": inj_core,
            "position_char": cand["start"],
            "sentence_approx": sent_idx + 1,
            "perturbation_delta": round(delta, 4),
        })

    injection_log.reverse()

    if injected_text == text and injection_log:
        logger.error("  [%s] TEXT UNCHANGED!", question_id)
        injection_log = []

    return injected_text, injection_log


# --- Injection Test ---
def test_injection():
    sample = (
        "- COMPANY & PERIOD: 3M Company, Fiscal Year 2018\n"
        "- PRIMARY METRIC: CapEx amounted to $1,517 million per FY2018 10-K.\n"
        "- SUPPORTING DATA:\n"
        "  - FY2017 CapEx: $1,445 million\n"
        "  - FY2016 CapEx: $1,452 million\n"
        "  - Revenue FY2018: $32,765 million\n"
        "  - Gross Margin: 47.2%\n"
    )
    injected, log = inject_hallucinations(sample, "TEST_001", seed=42)
    print("=" * 70)
    print("INJECTION TEST")
    print("=" * 70)
    print(f"\nChanges ({len(log)}):")
    for e in log:
        print(f"  [{e['type']}] '{e['original_value']}' → '{e['injected_value']}'")
    assert injected != sample, "FAIL: Text unchanged!"
    for e in log:
        assert e['original_value'] != e['injected_value'], f"FAIL: No change: {e['original_value']}"
    print("\nAll injection assertions passed.\n")

test_injection()

class PipelineState(TypedDict):
    question_id: str
    wrapped_prompt: str
    original_question: str
    ground_truth_answer: str
    evidence: str
    researcher_output: str
    analyst_output: str
    writer_output: str
    reviewer_output: str
    researcher_output_original: str
    injection_log: list[dict]
    injection_applied: bool
    snapshots: list[dict]
    timestamp: str
    error: str


def _call_agent(agent_key: str, input_text: str) -> str:
    sys_prompt = AGENT_PROMPTS[agent_key]
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = agent_llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=input_text),
            ])
            return resp.content.strip()
        except Exception as e:
            logger.warning("  Agent '%s' attempt %d: %s", agent_key, attempt, e)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def _save_snapshot(state, stage, agent_key, output):
    state["snapshots"].append({
        "stage": stage, "agent": agent_key, "output": output,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "char_count": len(output), "word_count": len(output.split()),
    })


def node_researcher(state: PipelineState) -> dict:
    prompt = (f"RESEARCH DIRECTIVE:\n{state['wrapped_prompt']}\n\n"
              f"ORIGINAL QUESTION:\n{state['original_question']}")
    output = _call_agent("researcher", prompt)
    _save_snapshot(state, 1, "researcher", output)
    return {"researcher_output_original": output, "researcher_output": output,
            "snapshots": state["snapshots"]}


def node_inject(state: PipelineState) -> dict:
    original = state["researcher_output_original"]
    injected, log = inject_hallucinations(
        original, state["question_id"],
        seed=RANDOM_SEED + abs(hash(state["question_id"])) % 10000,
    )
    logger.info("  [%s] Injected %d hallucinations:", state["question_id"], len(log))
    for e in log:
        logger.info("    '%s' → '%s'", e["original_value"], e["injected_value"])

    for snap in state["snapshots"]:
        if snap["stage"] == 1:
            snap["output_injected"] = injected
            snap["injection_count"] = len(log)

    return {"researcher_output": injected, "injection_log": log,
            "injection_applied": True, "snapshots": state["snapshots"]}


def node_analyst(state: PipelineState) -> dict:
    prompt = (f"RESEARCH DIRECTIVE:\n{state['wrapped_prompt']}\n\n"
              f"RESEARCHER'S DATA REPORT:\n{state['researcher_output']}\n\n"
              f"Perform your financial analysis based on the data above.")
    output = _call_agent("analyst", prompt)
    _save_snapshot(state, 2, "analyst", output)
    return {"analyst_output": output, "snapshots": state["snapshots"]}


def node_writer(state: PipelineState) -> dict:
    prompt = (f"RESEARCH DIRECTIVE:\n{state['wrapped_prompt']}\n\n"
              f"RESEARCHER'S DATA:\n{state['researcher_output']}\n\n"
              f"ANALYST'S ASSESSMENT:\n{state['analyst_output']}\n\n"
              f"Synthesize into a professional financial analysis report.")
    output = _call_agent("writer", prompt)
    _save_snapshot(state, 3, "writer", output)
    return {"writer_output": output, "snapshots": state["snapshots"]}


def node_reviewer(state: PipelineState) -> dict:
    prompt = (f"RESEARCH DIRECTIVE:\n{state['wrapped_prompt']}\n\n"
              f"DRAFT REPORT:\n{state['writer_output']}\n\n"
              f"Perform quality review and produce the final approved report.")
    output = _call_agent("reviewer", prompt)
    _save_snapshot(state, 4, "reviewer", output)
    return {"reviewer_output": output, "snapshots": state["snapshots"]}


graph = StateGraph(PipelineState)
graph.add_node("researcher", node_researcher)
graph.add_node("inject", node_inject)
graph.add_node("analyst", node_analyst)
graph.add_node("writer", node_writer)
graph.add_node("reviewer", node_reviewer)
graph.set_entry_point("researcher")
graph.add_edge("researcher", "inject")
graph.add_edge("inject", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)
pipeline = graph.compile()
logger.info("Pipeline compiled.")



JUDGE_SYSTEM_PROMPT = """You are an expert financial forensic auditor performing a line-by-line fact check.

TASK:
Analyze the financial text below. For EVERY specific financial figure (dollar amounts, percentages, ratios, growth rates, counts), assess whether it appears accurate or suspicious.

CHECK FOR:
1. INTERNAL INCONSISTENCY: Do percentages match the base numbers they're derived from? Does "$X increase from $Y to $Z" actually add up?
2. IMPLAUSIBILITY: Are figures within a reasonable range for this company and industry? A $70B COGS for a $30B revenue company is impossible.
3. TREND ANOMALIES: Do multi-year trends make sense? A 40% swing in CapEx year-over-year is unusual for a stable industrial company.
4. PRECISION MISMATCHES: Are some figures suspiciously round or precise compared to others?
5. CROSS-REFERENCE: If multiple figures are given for different years, check whether the implied trends are plausible.

Be AGGRESSIVE in flagging. It is better to flag a correct number as suspicious than to miss a fabricated one.

OUTPUT FORMAT (strict JSON only, no markdown):
{
  "flagged_claims": [
    {
      "claim_text": "<the sentence or phrase containing the suspicious figure>",
      "claim_value": "<the specific number/dollar amount/percentage>",
      "reason": "<detailed explanation of why this is suspicious>",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "total_claims_reviewed": <int>,
  "assessment": "<one sentence summary>"
}

Flag ALL suspicious figures. Output ONLY valid JSON."""


def run_gpt4o_judge(text: str, stage: int, question_id: str) -> dict:
    user_msg = f"FINANCIAL TEXT TO FACT-CHECK (Stage {stage}):\n\n{text}"
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = judge_llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            raw = resp.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            parsed = json.loads(raw)
            parsed["stage"] = stage
            parsed["question_id"] = question_id
            return parsed
        except json.JSONDecodeError:
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                return {"flagged_claims": [], "total_claims_reviewed": 0,
                        "assessment": "PARSE_ERROR", "stage": stage,
                        "question_id": question_id}
        except Exception as e:
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {"flagged_claims": [], "total_claims_reviewed": 0,
                        "assessment": "API_ERROR", "stage": stage,
                        "question_id": question_id}


def _extract_all_numbers(text: str) -> list[dict]:
    """Extract all numeric values from text."""
    values = []
    for m in _DOLLAR_RE.finditer(text):
        nm = re.search(r'[\d,]+(?:\.\d+)?', m.group())
        if nm:
            try:
                values.append({"raw": m.group().strip(),
                    "numeric": float(nm.group().replace(',', '')),
                    "type": "dollar_amount", "start": m.start(), "end": m.end()})
            except ValueError: pass

    for m in _PCT_RE.finditer(text):
        nm = re.search(r'[\d]+(?:\.\d+)?', m.group())
        if nm:
            try:
                values.append({"raw": m.group().strip(),
                    "numeric": float(nm.group()),
                    "type": "percentage", "start": m.start(), "end": m.end()})
            except ValueError: pass

    for m in _NUM_RE.finditer(text):
        overlaps = any(not (m.end() <= v["start"] or m.start() >= v["end"]) for v in values)
        if overlaps or _is_year(m.group(), m.start(), m.end(), text):
            continue
        try:
            values.append({"raw": m.group().strip(),
                "numeric": float(m.group().replace(',', '')),
                "type": "large_number", "start": m.start(), "end": m.end()})
        except ValueError: pass

    return values


def run_retrieval_checker(
    text: str,
    ground_truth: str,
    evidence: str,
    original_researcher_output: str,
    stage: int,
    question_id: str,
) -> dict:
    """
    Compare values in text against THREE reference sources:
      1. FinanceBench ground truth answer
      2. FinanceBench evidence string
      3. Original (pre-injection) researcher output  ← KEY ADDITION

    The original researcher output gives us the "clean" reference for ALL
    values in the pipeline, not just the primary metric.
    """
    text_values = _extract_all_numbers(text)

    # Build reference pool from all sources
    gt_values = _extract_all_numbers(ground_truth)
    ev_values = _extract_all_numbers(evidence)
    orig_values = _extract_all_numbers(original_researcher_output)

    # All reference numbers (deduplicated by numeric value)
    ref_nums = {}
    for rv in gt_values + ev_values + orig_values:
        key = round(rv["numeric"], 2)
        if key not in ref_nums:
            ref_nums[key] = rv

    reference_values = list(ref_nums.values())

    results = {
        "question_id": question_id, "stage": stage,
        "claims_checked": len(text_values),
        "matches": [], "mismatches": [], "unchecked": [],
    }

    # Tight tolerance: 1% for matching (was 5%)
    MATCH_TOL = 0.01

    for tv in text_values:
        matched = False
        for rv in reference_values:
            if tv["numeric"] == 0 and rv["numeric"] == 0:
                matched = True; break
            if rv["numeric"] != 0:
                dev = abs(tv["numeric"] - rv["numeric"]) / abs(rv["numeric"])
                if dev < MATCH_TOL:
                    results["matches"].append({
                        "text_value": tv["raw"],
                        "reference_value": rv["raw"],
                        "type": tv["type"],
                    })
                    matched = True
                    break

        if not matched:
            # Find nearest reference for context
            best_dev = float('inf')
            best_ref = None
            for rv in reference_values:
                if rv["numeric"] != 0:
                    dev = abs(tv["numeric"] - rv["numeric"]) / abs(rv["numeric"])
                    if dev < best_dev:
                        best_dev = dev
                        best_ref = rv

            if best_ref and best_dev < 1.0:
                results["mismatches"].append({
                    "text_value": tv["raw"],
                    "text_numeric": tv["numeric"],
                    "nearest_reference": best_ref["raw"],
                    "nearest_ref_numeric": best_ref["numeric"],
                    "type": tv["type"],
                    "deviation": round(best_dev, 4),
                })
            else:
                results["unchecked"].append({
                    "text_value": tv["raw"],
                    "text_numeric": tv["numeric"],
                    "type": tv["type"],
                })

    return results


def _parse_num(s: str) -> float:
    cleaned = re.sub(r'[^0-9.\-]', '', str(s))
    try: return float(cleaned)
    except ValueError: return 0.0


def match_detections_to_injections(
    injection_log: list[dict],
    judge_result: dict,
    retrieval_result: dict,
) -> list[dict]:
    """
    For each injected hallucination, check if either detector caught it.

    Judge matching: check if any flagged claim's value is within 5% of the
    injected value, OR if the injected numeric appears as a substring.

    Retrieval matching: check if any mismatch's text_numeric is within 5%
    of the injected numeric.
    """
    records = []
    MATCH_TOL = 0.05  # 5% tolerance for matching detections to injections

    for inj in injection_log:
        inj_num = _parse_num(inj["injected_numeric"])
        orig_num = _parse_num(inj["original_numeric"])

        # === GPT-4o Judge Detection ===
        judge_caught = False
        judge_reason = ""

        for flagged in judge_result.get("flagged_claims", []):
            # Strategy 1: Numeric proximity
            flagged_num = _parse_num(flagged.get("claim_value", ""))
            if flagged_num > 0 and inj_num > 0:
                dev = abs(flagged_num - inj_num) / max(flagged_num, inj_num)
                if dev < MATCH_TOL:
                    judge_caught = True
                    judge_reason = flagged.get("reason", "")
                    break

            # Strategy 2: Check if judge flagged the ORIGINAL value
            # (meaning it noticed something was wrong in that area)
            if orig_num > 0 and flagged_num > 0:
                dev = abs(flagged_num - orig_num) / max(flagged_num, orig_num)
                if dev < MATCH_TOL:
                    judge_caught = True
                    judge_reason = flagged.get("reason", "")
                    break

            # Strategy 3: Substring match in claim_text
            claim_text = flagged.get("claim_text", "") + flagged.get("claim_value", "")
            inj_clean = inj["injected_numeric"].replace(',', '')
            if len(inj_clean) >= 3 and inj_clean in claim_text.replace(',', ''):
                judge_caught = True
                judge_reason = flagged.get("reason", "")
                break

        # === Retrieval Checker Detection ===
        retrieval_caught = False

        for mm in retrieval_result.get("mismatches", []):
            mm_num = mm.get("text_numeric", _parse_num(mm.get("text_value", "")))
            if isinstance(mm_num, str):
                mm_num = _parse_num(mm_num)

            if mm_num > 0 and inj_num > 0:
                dev = abs(mm_num - inj_num) / max(mm_num, inj_num)
                if dev < MATCH_TOL:
                    retrieval_caught = True
                    break

            # Also check if the mismatch corresponds to original value being replaced
            if mm_num > 0 and orig_num > 0:
                dev = abs(mm_num - orig_num) / max(mm_num, orig_num)
                if dev < MATCH_TOL:
                    retrieval_caught = True
                    break

        records.append({
            "question_id": judge_result["question_id"],
            "hallucination_id": inj["hallucination_id"],
            "stage": judge_result["stage"],
            "injection_type": inj["type"],
            "original_value": inj["original_value"],
            "injected_value": inj["injected_value"],
            "detected_by_judge": judge_caught,
            "judge_reason": judge_reason,
            "detected_by_retrieval": retrieval_caught,
            "detected_by_either": judge_caught or retrieval_caught,
        })

    return records

def run_single_case(row: pd.Series) -> dict:
    qid = row["question_id"]
    logger.info("Processing %s ...", qid)

    initial_state = {
        "question_id": qid,
        "wrapped_prompt": row["wrapped_prompt"],
        "original_question": row["original_question"],
        "ground_truth_answer": row["answer"],
        "evidence": str(row.get("evidence", "")),
        "researcher_output": "", "analyst_output": "",
        "writer_output": "", "reviewer_output": "",
        "researcher_output_original": "",
        "injection_log": [], "injection_applied": False,
        "snapshots": [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error": "",
    }

    final_state = pipeline.invoke(initial_state)

    # Collect stage outputs
    stage_outputs = {}
    for snap in final_state["snapshots"]:
        s = snap["stage"]
        stage_outputs[s] = snap.get("output_injected", snap["output"]) if s == 1 else snap["output"]

    # Run detectors at each stage
    all_detections = []
    original_output = final_state["researcher_output_original"]

    for stage_num in [1, 2, 3, 4]:
        stage_text = stage_outputs.get(stage_num, "")
        if not stage_text:
            continue

        judge_result = run_gpt4o_judge(stage_text, stage_num, qid)

        retrieval_result = run_retrieval_checker(
            text=stage_text,
            ground_truth=row["answer"],
            evidence=str(row.get("evidence", "")),
            original_researcher_output=original_output,  # ← KEY
            stage=stage_num,
            question_id=qid,
        )

        # Log retrieval findings for debugging
        if retrieval_result["mismatches"]:
            logger.info("  Stage %d retrieval mismatches: %s", stage_num,
                        [(m["text_value"], "≠", m["nearest_reference"]) for m in retrieval_result["mismatches"]])

        detections = match_detections_to_injections(
            final_state["injection_log"], judge_result, retrieval_result,
        )
        all_detections.extend(detections)

        # Log detection summary
        caught = sum(1 for d in detections if d["detected_by_either"])
        logger.info("  Stage %d: %d/%d hallucinations detected (J:%d R:%d)",
                     stage_num, caught, len(detections),
                     sum(1 for d in detections if d["detected_by_judge"]),
                     sum(1 for d in detections if d["detected_by_retrieval"]))

    # Save snapshot
    snapshot_path = SNAPSHOTS_DIR / f"{qid}.json"
    with open(snapshot_path, "w") as f:
        json.dump({
            "question_id": qid,
            "original_question": row["original_question"],
            "ground_truth_answer": row["answer"],
            "researcher_output_original": original_output,
            "researcher_output_injected": stage_outputs.get(1, ""),
            "analyst_output": final_state["analyst_output"],
            "writer_output": final_state["writer_output"],
            "reviewer_output": final_state["reviewer_output"],
            "injection_log": final_state["injection_log"],
            "snapshots_meta": final_state["snapshots"],
            "timestamp": final_state["timestamp"],
        }, f, indent=2, default=str)

    return {
        "question_id": qid,
        "injection_log": final_state["injection_log"],
        "detections": all_detections,
        "num_injections": len(final_state["injection_log"]),
    }


def _load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            cp = json.load(f)
        return (set(cp.get("completed_ids", [])),
                cp.get("all_detections", []),
                cp.get("all_injection_logs", {}))
    return set(), [], {}


def _save_checkpoint(completed, detections, inj_logs):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed_ids": list(completed),
                    "all_detections": detections,
                    "all_injection_logs": inj_logs,
                    "saved_at": datetime.utcnow().isoformat() + "Z"}, f, default=str)


def run_snowball_experiment(df: pd.DataFrame, n: int = None) -> pd.DataFrame:
    if n is not None:
        df = df.head(n).copy()
    logger.info("Running on %d questions", len(df))

    completed, all_detections, all_inj_logs = _load_checkpoint()
    failed = []

    remaining = [(i, r) for i, r in df.iterrows() if r["question_id"] not in completed]
    logger.info("%d remaining (of %d)", len(remaining), len(df))

    for idx, row in tqdm(remaining, desc="Snowball Experiment"):
        qid = row["question_id"]
        try:
            result = run_single_case(row)
            all_detections.extend(result["detections"])
            all_inj_logs[qid] = result["injection_log"]
            completed.add(qid)

            if len(completed) % 5 == 0:
                _save_checkpoint(completed, all_detections, all_inj_logs)

        except Exception as e:
            logger.error("FAILED %s:\n%s", qid, traceback.format_exc())
            failed.append({"question_id": qid, "error": str(e)})

    _save_checkpoint(completed, all_detections, all_inj_logs)

    with open(INJECTION_LOG_FILE, "w") as f:
        json.dump(all_inj_logs, f, indent=2)

    det_df = pd.DataFrame(all_detections)
    if not det_df.empty:
        det_df.to_csv(DETECTION_FILE, index=False)
    logger.info("Done. %d detection records, %d failed", len(det_df), len(failed))

    if failed:
        pd.DataFrame(failed).to_csv(RESULTS_DIR / "failed_cases.csv", index=False)

    return det_df


df = pd.read_excel(INPUT_FILE, sheet_name="experiment_data")
logger.info("Loaded %d questions", len(df))

det_df = run_snowball_experiment(df, n=N)

print(f"\nDone. Detection records: {len(det_df)}")


def analyze_and_plot(det_df=None, injection_logs_path=None):
    """
    Full analysis suite:
      A) Stage detection rates + snowball decay curve
      B) Detection by hallucination type
      C) Hallucination survival rate (new)
      D) Per-hallucination trajectory distribution (new)
      E) Judge vs Retrieval gap chart (new)
      F) Injection statistics summary (new)
      G) Paper-ready summary (new)
    """
    if det_df is None or det_df.empty:
        det_df = pd.read_csv(DETECTION_FILE)
    if det_df.empty:
        print("No data!"); return

    # Load injection logs (needed for analyses F and G)
    if injection_logs_path is None:
        injection_logs_path = INJECTION_LOG_FILE
    if Path(injection_logs_path).exists():
        with open(injection_logs_path) as f:
            injection_logs = json.load(f)
    else:
        injection_logs = {}
        logger.warning("injection_logs.json not found; injection statistics will be skipped.")

    logger.info("Analyzing %d records ...", len(det_df))


    stage_rates = det_df.groupby("stage").agg(
        judge_rate=("detected_by_judge", "mean"),
        retrieval_rate=("detected_by_retrieval", "mean"),
        either_rate=("detected_by_either", "mean"),
        n_hallucinations=("hallucination_id", "count"),
    ).reset_index()
    stage_rates["judge_pct"] = (stage_rates["judge_rate"] * 100).round(1)
    stage_rates["retrieval_pct"] = (stage_rates["retrieval_rate"] * 100).round(1)
    stage_rates["either_pct"] = (stage_rates["either_rate"] * 100).round(1)
    stage_rates["gap"] = (stage_rates["retrieval_pct"] - stage_rates["judge_pct"]).round(1)

    print("DETECTION RATES BY PIPELINE STAGE")
    print(stage_rates[["stage", "judge_pct", "retrieval_pct",
                        "either_pct", "n_hallucinations"]].to_string(index=False))

    stage_labels = ["Stage 1\n(Researcher)", "Stage 2\n(Analyst)",
                    "Stage 3\n(Writer)", "Stage 4\n(Reviewer)"]
    stages = stage_rates["stage"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(stages, stage_rates["judge_pct"], 'o-',
            color='#E74C3C', lw=2.5, ms=10, label='GPT-4o Judge', zorder=3)
    ax.plot(stages, stage_rates["retrieval_pct"], 's-',
            color='#3498DB', lw=2.5, ms=10, label='Retrieval Checker', zorder=3)
    ax.plot(stages, stage_rates["either_pct"], 'D--',
            color='#2ECC71', lw=2, ms=8, label='Either (Union)', alpha=0.8, zorder=3)

    ax.set_xlabel("Pipeline Stage", fontsize=13, fontweight='bold')
    ax.set_ylabel("Detection Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title("The Hallucination Snowball Effect\nDetection Rate Decay Across Stages",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(stages)
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, s in enumerate(stages):
        ax.annotate(f"{stage_rates['either_pct'].iloc[i]:.0f}%",
                    (s, stage_rates['either_pct'].iloc[i]),
                    textcoords="offset points", xytext=(0, 14),
                    ha='center', fontsize=10, fontweight='bold', color='#2ECC71')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "snowball_decay_curve.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "snowball_decay_curve.pdf", bbox_inches='tight')
    plt.show()

    if "injection_type" in det_df.columns and det_df["injection_type"].nunique() > 1:
        type_stage = det_df.groupby(["stage", "injection_type"]).agg(
            det_rate=("detected_by_either", "mean"), count=("hallucination_id", "count"),
        ).reset_index()
        type_stage["det_pct"] = (type_stage["det_rate"] * 100).round(1)

        print("\nDETECTION BY TYPE × STAGE:")
        print(type_stage.to_string(index=False))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        types = type_stage["injection_type"].unique()
        colors = {'dollar_amount': '#E74C3C', 'percentage': '#3498DB', 'large_number': '#F39C12'}
        for j, t in enumerate(types):
            sub = type_stage[type_stage["injection_type"] == t]
            offset = (j - len(types) / 2 + 0.5) * 0.25
            ax2.bar(sub["stage"] + offset, sub["det_pct"], width=0.25,
                    label=t.replace('_', ' ').title(),
                    color=colors.get(t, '#95A5A6'), edgecolor='white')
        ax2.set_xlabel("Stage", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Detection Rate (%)", fontsize=12, fontweight='bold')
        ax2.set_title("Detection by Hallucination Type", fontsize=13, fontweight='bold')
        ax2.set_xticks([1, 2, 3, 4])
        ax2.set_xticklabels(stage_labels)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y', ls='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        fig2.savefig(FIGURES_DIR / "detection_by_type.png", dpi=300, bbox_inches='tight')
        plt.show()


    print("ANALYSIS: HALLUCINATION SURVIVAL RATE")

    stage4 = det_df[det_df["stage"] == 4].copy()
    total_h = len(stage4)

    survived_either = stage4[~stage4["detected_by_either"]]
    survived_judge = stage4[~stage4["detected_by_judge"]]
    survived_retrieval = stage4[~stage4["detected_by_retrieval"]]

    print(f"\nAt Stage 4 (Final Output):")
    print(f"  Total hallucinations tracked:     {total_h}")
    print(f"  Undetected by GPT-4o Judge:       {len(survived_judge)} ({len(survived_judge)/total_h*100:.1f}%)")
    print(f"  Undetected by Retrieval Checker:  {len(survived_retrieval)} ({len(survived_retrieval)/total_h*100:.1f}%)")
    print(f"  Undetected by EITHER (survived):  {len(survived_either)} ({len(survived_either)/total_h*100:.1f}%)")

    print(f"\nSurvival Rate by Injection Type:")
    for itype in stage4["injection_type"].unique():
        subset = stage4[stage4["injection_type"] == itype]
        surv = subset[~subset["detected_by_either"]]
        print(f"  {itype:20s}: {len(surv)}/{len(subset)} survived ({len(surv)/len(subset)*100:.1f}%)")

    headline = {
        "total_hallucinations_at_stage4": total_h,
        "survived_both_detectors": len(survived_either),
        "survival_rate_either": f"{len(survived_either)/total_h*100:.1f}%",
        "survived_judge_only": len(survived_judge),
        "survival_rate_judge": f"{len(survived_judge)/total_h*100:.1f}%",
        "survived_retrieval_only": len(survived_retrieval),
        "survival_rate_retrieval": f"{len(survived_retrieval)/total_h*100:.1f}%",
    }
    pd.DataFrame([headline]).to_csv(RESULTS_DIR / "survival_rates.csv", index=False)
    print(f"\nSaved: survival_rates.csv")


    print("ANALYSIS: PER-HALLUCINATION TRAJECTORY DISTRIBUTION")

    trajectories = []
    for h_id in det_df["hallucination_id"].unique():
        h_data = det_df[det_df["hallucination_id"] == h_id].sort_values("stage")
        if len(h_data) != 4:
            continue  # skip incomplete trajectories

        traj = tuple(h_data["detected_by_either"].astype(int).values)
        judge_traj = tuple(h_data["detected_by_judge"].astype(int).values)

        trajectories.append({
            "hallucination_id": h_id,
            "question_id": h_data["question_id"].iloc[0],
            "injection_type": h_data["injection_type"].iloc[0],
            "trajectory_either": traj,
            "trajectory_judge": judge_traj,
            "stage1_detected": traj[0],
            "stage4_detected": traj[3],
        })

    traj_df = pd.DataFrame(trajectories)

    def classify_trajectory(traj):
        if traj == (1, 1, 1, 1):
            return "Always Detected"
        elif traj == (0, 0, 0, 0):
            return "Never Detected"
        elif traj[0] == 1 and traj[3] == 0:
            return "Full Decay (caught→missed)"
        elif traj[0] == 0 and traj[3] == 1:
            return "Late Detection"
        elif sum(traj) > 0 and traj[3] == 1:
            return "Partially Detected (caught at end)"
        else:
            return "Partial Decay"

    traj_df["category"] = traj_df["trajectory_either"].apply(classify_trajectory)
    traj_df["category_judge"] = traj_df["trajectory_judge"].apply(classify_trajectory)

    print("\n--- EITHER Detector Trajectories ---")
    cat_counts = traj_df["category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat:35s}: {count:4d} ({count/len(traj_df)*100:.1f}%)")

    print(f"\n--- GPT-4o Judge Trajectories ---")
    cat_counts_j = traj_df["category_judge"].value_counts()
    for cat, count in cat_counts_j.items():
        print(f"  {cat:35s}: {count:4d} ({count/len(traj_df)*100:.1f}%)")

    print(f"\n--- Top 10 Exact Trajectory Patterns (Either) ---")
    pattern_counts = traj_df["trajectory_either"].value_counts().head(10)
    for pattern, count in pattern_counts.items():
        label = "→".join(["✓" if s else "✗" for s in pattern])
        print(f"  {label:20s}: {count:4d} ({count/len(traj_df)*100:.1f}%)")

    # FIGURE 3: Trajectory Distribution Bar Chart
    traj_order = [
        "Always Detected", "Full Decay (caught→missed)", "Partial Decay",
        "Partially Detected (caught at end)", "Late Detection", "Never Detected",
    ]
    colors_map = {
        "Always Detected": "#2ECC71",
        "Full Decay (caught→missed)": "#E74C3C",
        "Partial Decay": "#F39C12",
        "Partially Detected (caught at end)": "#3498DB",
        "Late Detection": "#9B59B6",
        "Never Detected": "#7F8C8D",
    }

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in [(axes3[0], "category", "Either Detector"),
                           (axes3[1], "category_judge", "GPT-4o Judge Only")]:
        counts = traj_df[col].value_counts()
        cats = [c for c in traj_order if c in counts.index]
        vals = [counts[c] for c in cats]
        pcts = [v / len(traj_df) * 100 for v in vals]
        bar_colors = [colors_map.get(c, "#95A5A6") for c in cats]

        bars = ax.barh(range(len(cats)), pcts, color=bar_colors, edgecolor='white', height=0.6)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=10)
        ax.set_xlabel("Percentage of Hallucinations (%)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, pct, val in zip(bars, pcts, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}% (n={val})", va='center', fontsize=9)

    plt.suptitle("Hallucination Detection Trajectory Distribution",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "trajectory_distribution.png", dpi=300, bbox_inches='tight')
    fig3.savefig(FIGURES_DIR / "trajectory_distribution.pdf", bbox_inches='tight')
    plt.show()

    traj_df.to_csv(RESULTS_DIR / "hallucination_trajectories.csv", index=False)
    print(f"\nSaved: trajectory_distribution.png/pdf, hallucination_trajectories.csv")

    print("ANALYSIS: JUDGE vs RETRIEVAL DETECTION GAP")

    print("\nStage | Judge (%) | Retrieval (%) | Gap (pp)")
    for _, row in stage_rates.iterrows():
        print(f"  {int(row['stage']):d}   |   {row['judge_pct']:5.1f}   |     {row['retrieval_pct']:5.1f}     | {row['gap']:+5.1f}")

    fig4, ax4 = plt.subplots(figsize=(9, 5.5))

    ax4.plot(stages, stage_rates["retrieval_pct"], 's-',
             color='#3498DB', lw=2.5, ms=11, label='Retrieval Checker', zorder=3)
    ax4.plot(stages, stage_rates["judge_pct"], 'o-',
             color='#E74C3C', lw=2.5, ms=11, label='GPT-4o Judge', zorder=3)

    ax4.fill_between(stages,
                     stage_rates["judge_pct"],
                     stage_rates["retrieval_pct"],
                     alpha=0.15, color='#8E44AD',
                     label='Detection Gap\n(values present but LLM-invisible)')

    for i, s in enumerate(stages):
        gap = stage_rates["gap"].iloc[i]
        mid = (stage_rates["judge_pct"].iloc[i] + stage_rates["retrieval_pct"].iloc[i]) / 2
        ax4.annotate(f"{gap:.0f} pp",
                     (s, mid), fontsize=10, fontweight='bold',
                     color='#8E44AD', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#8E44AD', alpha=0.8))

    for i, s in enumerate(stages):
        ax4.annotate(f"{stage_rates['retrieval_pct'].iloc[i]:.0f}%",
                     (s, stage_rates['retrieval_pct'].iloc[i]),
                     textcoords="offset points", xytext=(12, 5),
                     fontsize=9, color='#3498DB', fontweight='bold')
        ax4.annotate(f"{stage_rates['judge_pct'].iloc[i]:.0f}%",
                     (s, stage_rates['judge_pct'].iloc[i]),
                     textcoords="offset points", xytext=(12, -12),
                     fontsize=9, color='#E74C3C', fontweight='bold')

    ax4.set_xlabel("Pipeline Stage", fontsize=13, fontweight='bold')
    ax4.set_ylabel("Detection Rate (%)", fontsize=13, fontweight='bold')
    ax4.set_title(
        "The Hallucination Snowball Effect\n"
        "Hallucinated values persist in text but become invisible to LLM judges",
        fontsize=13, fontweight='bold', pad=15
    )
    ax4.set_xticks(stages)
    ax4.set_xticklabels(stage_labels, fontsize=10)
    ax4.set_ylim(0, 105)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax4.legend(fontsize=10, loc='lower left')
    ax4.grid(True, alpha=0.3, ls='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "judge_vs_retrieval_gap.png", dpi=300, bbox_inches='tight')
    fig4.savefig(FIGURES_DIR / "judge_vs_retrieval_gap.pdf", bbox_inches='tight')
    plt.show()
    print(f"\nSaved: judge_vs_retrieval_gap.png/pdf")

    print("INJECTION STATISTICS")

    all_injections = []
    for qid, logs in injection_logs.items():
        for inj in logs:
            inj_copy = dict(inj)
            inj_copy["question_id"] = qid
            all_injections.append(inj_copy)

    inj_df = pd.DataFrame(all_injections)

    if not inj_df.empty:
        no_injection_qs = [qid for qid, logs in injection_logs.items() if len(logs) == 0]
        has_injection_qs = [qid for qid, logs in injection_logs.items() if len(logs) > 0]

        print(f"\n  Total questions processed:        {len(injection_logs)}")
        print(f"  Questions WITH injections:        {len(has_injection_qs)}")
        print(f"  Questions WITHOUT injections:     {len(no_injection_qs)}")
        print(f"  Total hallucinations injected:    {len(inj_df)}")
        print(f"  Avg injections per question:      {len(inj_df)/len(has_injection_qs):.1f}")

        print(f"\n  By Type:")
        for t, count in inj_df["type"].value_counts().items():
            print(f"    {t:20s}: {count:4d} ({count/len(inj_df)*100:.1f}%)")

        print(f"\n  Perturbation Magnitude:")
        print(f"    Mean |delta|:  {inj_df['perturbation_delta'].abs().mean():.3f} ({inj_df['perturbation_delta'].abs().mean()*100:.1f}%)")
        print(f"    Min  |delta|:  {inj_df['perturbation_delta'].abs().min():.3f}")
        print(f"    Max  |delta|:  {inj_df['perturbation_delta'].abs().max():.3f}")

        if no_injection_qs:
            print(f"\n  No-injection question IDs: {no_injection_qs}")

        inj_df.to_csv(RESULTS_DIR / "injection_statistics.csv", index=False)
        print(f"\nSaved: injection_statistics.csv")
    else:
        inj_df = pd.DataFrame()
        has_injection_qs = []
        no_injection_qs = []
        logger.warning("  No injection data available for statistics.")

    print("EXPERIMENT 1 — COMPLETE PAPER-READY SUMMARY")

    s1_j = stage_rates[stage_rates['stage'] == 1]['judge_pct'].values[0]
    s4_j = stage_rates[stage_rates['stage'] == 4]['judge_pct'].values[0]
    s1_r = stage_rates[stage_rates['stage'] == 1]['retrieval_pct'].values[0]
    s4_r = stage_rates[stage_rates['stage'] == 4]['retrieval_pct'].values[0]
    s1_e = stage_rates[stage_rates['stage'] == 1]['either_rate'].values[0] * 100
    s4_e = stage_rates[stage_rates['stage'] == 4]['either_rate'].values[0] * 100

    survival = len(survived_either) / total_h * 100
    survival_judge = len(survived_judge) / total_h * 100

    full_decay_pct = len(traj_df[traj_df["category"] == "Full Decay (caught→missed)"]) / len(traj_df) * 100
    always_pct = len(traj_df[traj_df["category"] == "Always Detected"]) / len(traj_df) * 100
    never_pct = len(traj_df[traj_df["category"] == "Never Detected"]) / len(traj_df) * 100

    gap_s1 = stage_rates[stage_rates['stage'] == 1]['gap'].values[0]
    gap_s4 = stage_rates[stage_rates['stage'] == 4]['gap'].values[0]

    inj_stats_str = ""
    if not inj_df.empty:
        inj_stats_str = (
            f"  - {len(has_injection_qs)} of {len(injection_logs)} FinanceBench questions produced "
            f"quantitative researcher outputs; {len(no_injection_qs)} qualitative questions excluded\n"
            f"  - {total_h} hallucinations tracked (~{len(inj_df)/max(len(has_injection_qs),1):.1f} per case)\n"
            f"  - Perturbation range: "
            f"{inj_df['perturbation_delta'].abs().min()*100:.0f}%-"
            f"{inj_df['perturbation_delta'].abs().max()*100:.0f}%"
        )
    else:
        inj_stats_str = f"  - {total_h} hallucinations tracked\n  - Injection log unavailable"

    summary_text = f"""
FOR THE ABSTRACT:
  "We track {total_h} automatically injected hallucinations across {len(has_injection_qs)}
   FinanceBench test cases. Hallucination detection by GPT-4o drops from
   {s1_j:.0f}% at the research stage to {s4_j:.0f}% at the final review
   ({s1_j - s4_j:.0f} pp decline), while {survival:.1f}% of hallucinations
   survive to the final output completely undetected."

FOR THE INTRODUCTION:
{inj_stats_str}

KEY NUMBERS:
  GPT-4o Judge:       {s1_j:.1f}% → {s4_j:.1f}% (drop: {s1_j - s4_j:.1f} pp)
  Retrieval Checker:  {s1_r:.1f}% → {s4_r:.1f}% (drop: {s1_r - s4_r:.1f} pp)
  Either (union):     {s1_e:.1f}% → {s4_e:.1f}% (drop: {s1_e - s4_e:.1f} pp)
  Survival rate:      {survival:.1f}% (undetected by both at Stage 4)
  Judge survival:     {survival_judge:.1f}% (undetected by GPT-4o at Stage 4)

TRAJECTORY DISTRIBUTION:
  Full Decay:     {full_decay_pct:.1f}%
  Always Caught:  {always_pct:.1f}%
  Never Caught:   {never_pct:.1f}%

DETECTION GAP (Retrieval − Judge):
  Stage 1: {gap_s1:.1f} pp
  Stage 4: {gap_s4:.1f} pp
  → Gap WIDENS by {gap_s4 - gap_s1:.1f} pp across the pipeline
"""

    print(summary_text)

    with open(RESULTS_DIR / "paper_ready_summary.txt", "w") as f:
        f.write(summary_text)

    stage_rates.to_csv(RESULTS_DIR / "stage_detection_rates.csv", index=False)

    legacy_summary = {
        "total_cases": det_df["question_id"].nunique(),
        "total_hallucinations": det_df[det_df["stage"] == 1]["hallucination_id"].nunique(),
        "stage_1_detection": f"{stage_rates[stage_rates['stage']==1]['either_pct'].values[0]:.1f}%",
        "stage_4_detection": f"{stage_rates[stage_rates['stage']==4]['either_pct'].values[0]:.1f}%",
        "absolute_drop": f"{stage_rates[stage_rates['stage']==1]['either_pct'].values[0] - stage_rates[stage_rates['stage']==4]['either_pct'].values[0]:.1f} pp",
    }
    print("EXPERIMENT 1 SUMMARY")
    for k, v in legacy_summary.items():
        print(f"  {k:30s}: {v}")

    pd.DataFrame([legacy_summary]).to_csv(RESULTS_DIR / "experiment1_summary.csv", index=False)

    print(f"\nAll files saved to: {RESULTS_DIR}")
    print(f"Figures saved to:   {FIGURES_DIR}")
    print("\nFiles generated:")
    print("  - stage_detection_rates.csv")
    print("  - experiment1_summary.csv")
    print("  - survival_rates.csv")
    print("  - hallucination_trajectories.csv")
    print("  - injection_statistics.csv")
    print("  - paper_ready_summary.txt")
    print("  - snowball_decay_curve.png/pdf")
    print("  - detection_by_type.png")
    print("  - trajectory_distribution.png/pdf")
    print("  - judge_vs_retrieval_gap.png/pdf")


analyze_and_plot(det_df)

if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()