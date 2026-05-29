"""Experiment 2: Quad Skeptic Agent Failure

Tests whether state-of-the-art LLMs can catch injected hallucinations across 4 diverse models
(Gemini, DeepSeek, Qwen, Llama) placed at the best-case position (Stage 1 output).
Compares detection rates to show the failure is structural, not model-specific.
"""

import os
import re
import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm

from google import genai
from google.genai import types
from openai import OpenAI
# Set N = 1   to run a single question (quick test / debug)
# Set N = 150 to run all questions (full experiment)
# NOTE: this N applies to ALL models — total API calls = N x 4
N = 150

PROJECT_DIR    = Path("/content/hallucination_snowball")
DATA_DIR       = PROJECT_DIR / "data"

EXP1_DIR       = PROJECT_DIR / "experiment_1"
EXP1_SNAPSHOTS = EXP1_DIR / "snapshots"
EXP1_INJ_LOG   = EXP1_DIR / "injection_logs.json"
EXP1_DET_FILE  = EXP1_DIR / "detection_results.csv"

EXP2_DIR        = PROJECT_DIR / "results" / "experiment_2"
EXP2_SNAPSHOTS  = EXP2_DIR / "snapshots"
EXP2_FIGURES    = EXP2_DIR / "figures"

for d in [EXP2_DIR, EXP2_SNAPSHOTS, EXP2_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

SKEPTIC_MODELS = [
    {
        "name"        : "Gemini-2.5-Flash",
        "model_id"    : "gemini-2.5-flash",
        "slug"        : "gemini_2_5_flash",
        "color"       : "#3498DB",
        "client_type" : "gemini",
        "max_tokens"  : 8192,   
        "det_file"    : EXP2_DIR / "skeptic_detection_results_gemini_2_5_flash.csv",
        "checkpoint"  : EXP2_DIR / "checkpoint_gemini_2_5_flash.json",
    },
    {
        "name"        : "DeepSeek-V3.2",
        "model_id"    : "deepseek-ai/DeepSeek-V3-0324:novita",
        "slug"        : "deepseek_v3",
        "color"       : "#E67E22",
        "client_type" : "hf",
        "max_tokens"  : 2000,
        "det_file"    : EXP2_DIR / "skeptic_detection_results_deepseek_v3.csv",
        "checkpoint"  : EXP2_DIR / "checkpoint_deepseek_v3.json",
    },
    {
        "name"        : "Qwen3.5-397B",
        "model_id"    : "Qwen/Qwen3.5-397B-A17B:novita",
        "slug"        : "qwen35",
        "color"       : "#27AE60",
        "client_type" : "hf",
        "max_tokens"  : 2000,
        "det_file"    : EXP2_DIR / "skeptic_detection_results_qwen35.csv",
        "checkpoint"  : EXP2_DIR / "checkpoint_qwen35.json",
    },
    {
        "name"        : "Meta-Llama-3-70B-Instruct",
        "model_id"    : "meta-llama/Meta-Llama-3-70B-Instruct:novita",
        "slug"        : "llama3_70b",
        "color"       : "#8E44AD",
        "client_type" : "hf",
        "max_tokens"  : 4096,   
        "det_file"    : EXP2_DIR / "skeptic_detection_results_llama3_70b.csv",
        "checkpoint"  : EXP2_DIR / "checkpoint_llama3_70b.json",
    },
]

SKEPTIC_TEMPERATURE = 0.0
MATCH_TOL           = 0.05   
RETRY_LIMIT         = 3
RETRY_DELAY         = 2     

LOG_FILE = EXP2_DIR / "experiment2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info("Experiment 2 (Quad Skeptic) configuration loaded.")
for m in SKEPTIC_MODELS:
    logger.info("  Model: %s  |  ID: %s  |  Client: %s", m["name"], m["model_id"], m["client_type"])
logger.info(
    "N (questions per model): %d  |  Total API calls: %d",
    N, N * len(SKEPTIC_MODELS),
)


with open(EXP1_INJ_LOG) as f:
    injection_logs: dict = json.load(f)

exp1_data: dict = {}
snapshot_files = sorted(EXP1_SNAPSHOTS.glob("*.json"))

for sf in tqdm(snapshot_files, desc="Loading Exp1 snapshots"):
    with open(sf) as f:
        snap = json.load(f)
    exp1_data[snap["question_id"]] = snap

valid_qids: list = [
    qid for qid, log in injection_logs.items() if len(log) > 0
][:N]

total_injections = sum(len(injection_logs[qid]) for qid in valid_qids)

logger.info(
    "Loaded %d Exp1 snapshots | %d questions selected (N=%d) | %d hallucinations",
    len(exp1_data), len(valid_qids), N, total_injections,
)
print(f"N (questions per model)           : {N}")
print(f"Total Exp1 snapshots loaded       : {len(exp1_data)}")
print(f"Questions selected for this run   : {len(valid_qids)}")
print(f"Total injected hallucinations     : {total_injections}")
print(f"Total API calls (N x 4 models)    : {N * len(SKEPTIC_MODELS)}")


SKEPTIC_SYSTEM_PROMPT = """You are a skeptical financial fact-checker embedded within a multi-agent research pipeline.

ROLE:
You have been placed AFTER a Financial Researcher agent to review its output BEFORE it reaches downstream analysts. Your sole purpose is to identify any claims that might be fabricated, inaccurate, or hallucinated.

CRITICAL CONTEXT:
- LLMs frequently hallucinate financial figures -- inventing plausible-looking numbers that are entirely wrong.
- You do NOT have access to any source documents, SEC filings, or databases.
- You must rely SOLELY on your general financial knowledge and internal reasoning to flag suspicious claims.

WHAT TO CHECK:
1. Are the reported financial figures plausible for this company and time period?
2. Are there internal inconsistencies (e.g., numbers that do not add up, contradictory trends)?
3. Do year-over-year changes seem reasonable (e.g., a 40% swing in revenue for a stable industrial company is unusual)?
4. Are any figures suspiciously round or suspiciously precise compared with others?
5. Do the claimed data sources (filing names, periods) seem legitimate?

INSTRUCTIONS:
- Review EVERY specific financial claim (dollar amounts, percentages, ratios, growth rates, counts).
- For each claim, decide: PLAUSIBLE or SUSPICIOUS.
- Be thorough but calibrated -- flag only claims where you have genuine reason for concern.
- For each suspicious claim, explain specifically WHY it seems wrong.

OUTPUT FORMAT — you MUST return strict JSON only. No markdown fences. No preamble. No explanation outside the JSON object. Start your response with { and end with }.

{
  "flagged_claims": [
    {
      "claim_text"  : "<sentence or phrase containing the suspicious figure>",
      "claim_value" : "<the specific number / dollar amount / percentage>",
      "reason"      : "<specific explanation of why this seems suspicious>",
      "confidence"  : "high" | "medium" | "low"
    }
  ],
  "total_claims_reviewed": <integer count of ALL claims you reviewed>,
  "overall_assessment"   : "<one sentence summary of your review>"
}

If nothing appears suspicious, return flagged_claims as an empty list [].
Your entire response must be valid JSON. Do not write anything before { or after }."""

SKEPTIC_USER_TEMPLATE = """FINANCIAL RESEARCHER OUTPUT TO REVIEW:

COMPANY / CONTEXT: {context}

--- BEGIN RESEARCHER OUTPUT ---
{researcher_output}
--- END RESEARCHER OUTPUT ---

Review every financial claim above for potential fabrication or inaccuracy. Be thorough. Respond with JSON only."""

_DOLLAR_RE = re.compile(
    r'\$\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|mn|bn|m|b|MM|'
    r'thousand|k|trillion|T)?(?:\s(?:million|billion|mn|bn|'
    r'thousand|k|trillion))?',
    re.IGNORECASE,
)


def _parse_num(s: str) -> float:
    """Strip non-numeric characters and parse to float. Returns 0.0 on failure."""
    cleaned = re.sub(r'[^0-9.\-]', '', str(s))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _recover_partial_json(raw: str, model_name: str) -> dict:
    """
    Partial JSON recovery for truncated or malformed model responses.

    When max_tokens cuts the response mid-array, or when the model emits
    a small amount of preamble text, the JSON may be partially valid.
    This function:
      1. Tries to extract just the JSON object using brace-matching
      2. Falls back to regex-extracting complete claim objects individually
      3. Returns a best-effort result rather than total data loss

    Returns an empty-result dict if all recovery attempts fail.
    """
    # Recovery attempt 1: find the outermost { ... } block
    try:
        first_brace = raw.index('{')
        last_brace  = raw.rindex('}')
        candidate   = raw[first_brace:last_brace + 1]
        parsed      = json.loads(candidate)
        logger.warning("  [%s] Brace-extraction recovery succeeded.", model_name)
        parsed["model"]        = model_name
        parsed["raw_response"] = raw
        return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        claim_pattern = re.compile(
            r'\{[^{}]*"claim_text"[^{}]*"claim_value"[^{}]*"reason"[^{}]*"confidence"[^{}]*\}',
            re.DOTALL,
        )
        claims = []
        for m in claim_pattern.finditer(raw):
            try:
                obj = json.loads(m.group())
                claims.append(obj)
            except json.JSONDecodeError:
                continue

        tcr_match    = re.search(r'"total_claims_reviewed"\s*:\s*(\d+)', raw)
        total_claims = int(tcr_match.group(1)) if tcr_match else 0

        if claims:
            logger.warning(
                "  [%s] Regex recovery succeeded: %d complete claims extracted.",
                model_name, len(claims),
            )
            return {
                "flagged_claims"       : claims,
                "total_claims_reviewed": total_claims,
                "overall_assessment"   : "PARTIAL_RECOVERY",
                "model"                : model_name,
                "raw_response"         : raw,
            }
    except Exception as e:
        logger.warning("  [%s] Regex recovery also failed: %s", model_name, e)

    logger.error("  [%s] All recovery attempts failed -- returning empty result.", model_name)
    return {
        "flagged_claims"       : [],
        "total_claims_reviewed": 0,
        "overall_assessment"   : "PARSE_ERROR",
        "model"                : model_name,
        "raw_response"         : raw,
    }


def _call_skeptic_gemini(
    researcher_output: str,
    context: str,
    model_cfg: dict,
) -> dict:
    """
    Submit a researcher output to Gemini 2.5 Flash for skeptical fact-checking
    via the Google GenAI SDK.

    Fixes applied (from original Experiment 2 Gemini code):
        - SKEPTIC_MAX_TOKENS raised to 8192 (eliminates truncation)
        - Partial JSON recovery on parse failure (salvages complete claims)

    Returns a dict with keys:
        flagged_claims, total_claims_reviewed, overall_assessment,
        model, raw_response
    """
    full_prompt = (
        SKEPTIC_SYSTEM_PROMPT
        + "\n\n"
        + SKEPTIC_USER_TEMPLATE.format(
            context=context,
            researcher_output=researcher_output,
        )
    )

    raw = ""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = gemini_client.models.generate_content(
                model=model_cfg["model_id"],
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=SKEPTIC_TEMPERATURE,
                    max_output_tokens=model_cfg["max_tokens"],
                ),
            )
            raw = resp.text.strip()

            # Strip accidental markdown code fences
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            parsed = json.loads(raw)
            parsed["model"]        = model_cfg["name"]
            parsed["raw_response"] = raw
            return parsed

        except json.JSONDecodeError:
            logger.warning(
                "  [%s][%s] JSON parse error (attempt %d/%d). "
                "Raw length: %d chars. Preview: %.80s",
                model_cfg["name"], context[:40], attempt, RETRY_LIMIT, len(raw), raw,
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                logger.warning(
                    "  [%s] All retries exhausted -- attempting partial JSON recovery.",
                    model_cfg["name"],
                )
                return _recover_partial_json(raw, model_cfg["name"])

        except Exception as exc:
            logger.warning(
                "  [%s] Gemini API error (attempt %d/%d): %s",
                model_cfg["name"], attempt, RETRY_LIMIT, exc,
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.error(
                    "  [%s] All retries exhausted (API error) -- returning empty result.",
                    model_cfg["name"],
                )
                return {
                    "flagged_claims"       : [],
                    "total_claims_reviewed": 0,
                    "overall_assessment"   : "API_ERROR",
                    "model"                : model_cfg["name"],
                    "raw_response"         : str(exc),
                }


def _call_skeptic_hf(
    researcher_output: str,
    context: str,
    model_cfg: dict,
) -> dict:
    """
    Submit a researcher output to a HuggingFace Router model for skeptical
    fact-checking via the OpenAI-compatible chat completions interface.

    Covers: DeepSeek-V3.2, Qwen3.5-397B, Meta-Llama-3-70B-Instruct.

    Returns a dict with keys:
        flagged_claims, total_claims_reviewed, overall_assessment,
        model, raw_response
    """
    user_message = SKEPTIC_USER_TEMPLATE.format(
        context=context,
        researcher_output=researcher_output,
    )

    raw = ""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = hf_client.chat.completions.create(
                model=model_cfg["model_id"],
                messages=[
                    {"role": "system", "content": SKEPTIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=SKEPTIC_TEMPERATURE,
                max_tokens=model_cfg["max_tokens"],
            )

            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown code fences
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```\s*$',       '', raw, flags=re.MULTILINE)
            raw = raw.strip()

            parsed = json.loads(raw)
            parsed["model"]        = model_cfg["name"]
            parsed["raw_response"] = raw
            return parsed

        except json.JSONDecodeError:
            logger.warning(
                "  [%s][%s] JSON parse error (attempt %d/%d). "
                "Raw length: %d chars. Preview: %.120s",
                model_cfg["name"], context[:40], attempt, RETRY_LIMIT, len(raw), raw,
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                logger.warning(
                    "  [%s] All retries exhausted -- attempting JSON recovery.",
                    model_cfg["name"],
                )
                return _recover_partial_json(raw, model_cfg["name"])

        except Exception as exc:
            logger.warning(
                "  [%s] API error (attempt %d/%d): %s",
                model_cfg["name"], attempt, RETRY_LIMIT, exc,
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.error(
                    "  [%s] All retries exhausted (API error) -- returning empty result.",
                    model_cfg["name"],
                )
                return {
                    "flagged_claims"       : [],
                    "total_claims_reviewed": 0,
                    "overall_assessment"   : "API_ERROR",
                    "model"                : model_cfg["name"],
                    "raw_response"         : str(exc),
                }


def call_skeptic(
    researcher_output: str,
    context: str,
    model_cfg: dict,
) -> dict:
    """
    Dispatch to the appropriate backend caller based on model_cfg["client_type"].

    "gemini" -> _call_skeptic_gemini (Google GenAI SDK)
    "hf"     -> _call_skeptic_hf     (HuggingFace Router, OpenAI-compatible)
    """
    if model_cfg["client_type"] == "gemini":
        return _call_skeptic_gemini(researcher_output, context, model_cfg)
    elif model_cfg["client_type"] == "hf":
        return _call_skeptic_hf(researcher_output, context, model_cfg)
    else:
        raise ValueError(f"Unknown client_type: {model_cfg['client_type']}")


def match_skeptic_to_injections(
    injection_log: list,
    skeptic_result: dict,
    question_id: str,
    model_name: str,
) -> list:
    """
    For each injected hallucination, determine whether the skeptic flagged it.
    Three-strategy matching mirrors Experiment 1 exactly for comparability.

    Returns one record per injection with full provenance for downstream
    analysis (type, confidence, reason, flag counts).
    """
    records = []

    for inj in injection_log:
        inj_num  = _parse_num(inj["injected_numeric"])
        orig_num = _parse_num(inj["original_numeric"])

        caught             = False
        matched_reason     = ""
        matched_confidence = ""

        for flagged in skeptic_result.get("flagged_claims", []):
            flagged_num = _parse_num(flagged.get("claim_value", ""))

            # Strategy 1: Proximity to injected value
            if flagged_num > 0 and inj_num > 0:
                dev = abs(flagged_num - inj_num) / max(flagged_num, inj_num)
                if dev < MATCH_TOL:
                    caught             = True
                    matched_reason     = flagged.get("reason", "")
                    matched_confidence = flagged.get("confidence", "")
                    break

            # Strategy 2: Proximity to original (pre-injection) value
            if orig_num > 0 and flagged_num > 0:
                dev = abs(flagged_num - orig_num) / max(flagged_num, orig_num)
                if dev < MATCH_TOL:
                    caught             = True
                    matched_reason     = flagged.get("reason", "")
                    matched_confidence = flagged.get("confidence", "")
                    break

            # Strategy 3: Substring match in claim text + claim value fields
            claim_text = (
                flagged.get("claim_text", "") + " " + flagged.get("claim_value", "")
            )
            inj_clean = inj["injected_numeric"].replace(",", "")
            if len(inj_clean) >= 3 and inj_clean in claim_text.replace(",", ""):
                caught             = True
                matched_reason     = flagged.get("reason", "")
                matched_confidence = flagged.get("confidence", "")
                break

        records.append({
            "question_id"           : question_id,
            "hallucination_id"      : inj["hallucination_id"],
            "model"                 : model_name,
            "injection_type"        : inj["type"],
            "original_value"        : inj["original_value"],
            "injected_value"        : inj["injected_value"],
            "detected"              : caught,
            "reason"                : matched_reason,
            "confidence"            : matched_confidence,
            "total_flags_by_skeptic": len(skeptic_result.get("flagged_claims", [])),
            "total_claims_reviewed" : skeptic_result.get("total_claims_reviewed", 0),
        })

    return records


def run_skeptic_on_question(
    qid: str,
    snap: dict,
    inj_log: list,
    model_cfg: dict,
) -> dict:
    """
    Run one skeptic model on one question's injected researcher output.

    Retrieves the injected Stage 1 output from the Experiment 1 snapshot,
    calls the skeptic, matches flags to known injections, and returns
    a fully populated result dict.

    Returns:
        {
            question_id      : str,
            detections       : list[dict],
            skeptic_response : dict,
            skipped          : bool,
            skip_reason      : str,
        }
    """
    # Retrieve injected researcher output (Stage 1, post-injection)
    injected_output = snap.get("researcher_output_injected", "")

    if not injected_output:
        # Fallback: retrieve from snapshots_meta list if present
        for sm in snap.get("snapshots_meta", []):
            if sm.get("stage") == 1 and "output_injected" in sm:
                injected_output = sm["output_injected"]
                break

    if not injected_output:
        logger.warning(
            "  [%s][%s] No injected researcher output -- skipping.",
            model_cfg["name"], qid,
        )
        return {
            "question_id"     : qid,
            "detections"      : [],
            "skeptic_response": {},
            "skipped"         : True,
            "skip_reason"     : "missing_injected_output",
        }

    context        = snap.get("original_question", "Financial research analysis")
    skeptic_result = call_skeptic(injected_output, context, model_cfg)
    detections     = match_skeptic_to_injections(
        inj_log, skeptic_result, qid, model_cfg["name"]
    )

    n_caught = sum(1 for d in detections if d["detected"])
    logger.info(
        "  [%s][%s] %d/%d caught | flags=%d | claims_reviewed=%d",
        model_cfg["name"], qid, n_caught, len(detections),
        len(skeptic_result.get("flagged_claims", [])),
        skeptic_result.get("total_claims_reviewed", 0),
    )

    return {
        "question_id"     : qid,
        "detections"      : detections,
        "skeptic_response": skeptic_result,
        "skipped"         : False,
        "skip_reason"     : "",
    }


def _load_checkpoint(checkpoint_path: Path) -> tuple:
    """Load per-model checkpoint; return empty state if none exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            cp = json.load(f)
        completed  = set(cp.get("completed_ids", []))
        detections = cp.get("all_detections", [])
        logger.info(
            "Checkpoint loaded (%s) -- %d questions already processed.",
            checkpoint_path.name, len(completed),
        )
        return completed, detections
    return set(), []


def _save_checkpoint(
    checkpoint_path: Path,
    completed: set,
    detections: list,
) -> None:
    """Persist per-model checkpoint to disk."""
    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "completed_ids" : list(completed),
                "all_detections": detections,
                "saved_at"      : datetime.utcnow().isoformat() + "Z",
            },
            f,
            default=str,
        )


def run_experiment_for_model(model_cfg: dict) -> pd.DataFrame:
    """
    Run the full skeptic experiment for ONE model.
    Called sequentially for each model in SKEPTIC_MODELS.

    Checkpoints every 10 completions for Colab session resilience.
    Saves per-question JSON snapshots for qualitative paper analysis.

    Returns det_df: one row per (question x injection) for this model.
    """
    print(f"\n{'=' * 60}")
    print(f"  Running skeptic: {model_cfg['name']}")
    print(f"  Model ID       : {model_cfg['model_id']}")
    print(f"  Client type    : {model_cfg['client_type']}")
    print(f"{'=' * 60}")

    completed, all_detections = _load_checkpoint(model_cfg["checkpoint"])
    remaining = [qid for qid in valid_qids if qid not in completed]

    logger.info(
        "[%s] %d questions remaining (of %d, N=%d).",
        model_cfg["name"], len(remaining), len(valid_qids), N,
    )

    failed = []

    for qid in tqdm(remaining, desc=f"Skeptic: {model_cfg['name']}"):

        if qid not in exp1_data:
            logger.warning(
                "  [%s][%s] Not in Exp1 snapshots -- skipping.",
                model_cfg["name"], qid,
            )
            continue

        inj_log = injection_logs.get(qid, [])
        if not inj_log:
            logger.warning(
                "  [%s][%s] No injection log -- skipping.",
                model_cfg["name"], qid,
            )
            continue

        try:
            result = run_skeptic_on_question(
                qid, exp1_data[qid], inj_log, model_cfg
            )

            if not result["skipped"]:
                all_detections.extend(result["detections"])

                # Persist per-question snapshot for qualitative review in paper
                snap_path = EXP2_SNAPSHOTS / f"{qid}_{model_cfg['slug']}.json"
                with open(snap_path, "w") as f:
                    json.dump(
                        {
                            "question_id"     : qid,
                            "model"           : model_cfg["name"],
                            "model_id"        : model_cfg["model_id"],
                            "injected_output" : exp1_data[qid].get(
                                "researcher_output_injected", ""
                            ),
                            "injection_log"   : inj_log,
                            "skeptic_response": result["skeptic_response"],
                            "detections"      : result["detections"],
                            "timestamp"       : datetime.utcnow().isoformat() + "Z",
                        },
                        f,
                        indent=2,
                        default=str,
                    )

            completed.add(qid)

            # Checkpoint every 10 questions
            if len(completed) % 10 == 0:
                _save_checkpoint(model_cfg["checkpoint"], completed, all_detections)

        except Exception:
            logger.error(
                "FAILED [%s][%s]:\n%s",
                model_cfg["name"], qid, traceback.format_exc(),
            )
            failed.append(qid)

    # Final save
    _save_checkpoint(model_cfg["checkpoint"], completed, all_detections)

    det_df = pd.DataFrame(all_detections)
    if not det_df.empty:
        det_df.to_csv(model_cfg["det_file"], index=False)
        logger.info(
            "[%s] Saved %d detection records -> %s",
            model_cfg["name"], len(det_df), model_cfg["det_file"],
        )
    else:
        logger.warning("[%s] No detection records produced.", model_cfg["name"])

    if failed:
        pd.DataFrame({"question_id": failed}).to_csv(
            EXP2_DIR / f"failed_cases_{model_cfg['slug']}.csv", index=False
        )
        logger.warning("[%s] %d questions failed.", model_cfg["name"], len(failed))

    return det_df


all_model_results: dict = {}   # slug -> DataFrame

for model_cfg in SKEPTIC_MODELS:
    df = run_experiment_for_model(model_cfg)
    all_model_results[model_cfg["slug"]] = df

# Combine into a single CSV for cross-model analysis
combined_df   = pd.concat(list(all_model_results.values()), ignore_index=True)
combined_path = EXP2_DIR / "skeptic_detection_results_combined.csv"
combined_df.to_csv(combined_path, index=False)

print(f"\nAll models complete.")
for model_cfg in SKEPTIC_MODELS:
    df   = all_model_results[model_cfg["slug"]]
    rate = df["detected"].mean() * 100 if not df.empty else 0.0
    print(f"  {model_cfg['name']:30s}: {len(df):4d} records | {rate:.1f}% detection rate")
print(f"\nCombined CSV: {combined_path}")


def compute_model_stats(det_df: pd.DataFrame, model_cfg: dict, exp1_judge_pct: float) -> dict:
    """
    Compute all scalar statistics for one model's detection DataFrame.
    Returns a flat dict used by both the print sections and CSV saving.
    """
    total_hallucinations = len(det_df)
    total_caught         = int(det_df["detected"].sum())
    total_missed         = total_hallucinations - total_caught
    detection_rate       = det_df["detected"].mean() * 100
    missed_rate          = 100.0 - detection_rate

    per_q = (
        det_df.groupby("question_id")
        .agg(
            true_positives   =("detected", "sum"),
            total_flags      =("total_flags_by_skeptic", "first"),
            total_claims     =("total_claims_reviewed", "first"),
            total_injections =("hallucination_id", "count"),
        )
        .reset_index()
    )
    per_q["false_flags"]   = (per_q["total_flags"] - per_q["true_positives"]).clip(lower=0)
    per_q["precision"]     = per_q.apply(
        lambda r: r["true_positives"] / r["total_flags"] if r["total_flags"] > 0 else 0.0,
        axis=1,
    )
    per_q["recall"]        = per_q.apply(
        lambda r: r["true_positives"] / r["total_injections"] if r["total_injections"] > 0 else 0.0,
        axis=1,
    )
    per_q["detection_pct"] = per_q["recall"] * 100

    type_rates = (
        det_df.groupby("injection_type")
        .agg(
            detection_rate =("detected", "mean"),
            count          =("hallucination_id", "count"),
            caught         =("detected", "sum"),
        )
        .reset_index()
    )
    type_rates["detection_pct"] = (type_rates["detection_rate"] * 100).round(1)
    type_rates["missed"]        = type_rates["count"] - type_rates["caught"]
    type_rates["survival_pct"]  = (100.0 - type_rates["detection_pct"]).round(1)

    return {
        "model_name"           : model_cfg["name"],
        "model_id"             : model_cfg["model_id"],
        "total_hallucinations" : total_hallucinations,
        "total_caught"         : total_caught,
        "total_missed"         : total_missed,
        "detection_rate"       : round(det_df["detected"].mean(), 4),
        "detection_pct"        : round(detection_rate, 1),
        "survival_pct"         : round(missed_rate, 1),
        "avg_flags"            : round(per_q["total_flags"].mean(), 1),
        "avg_false_flags"      : round(per_q["false_flags"].mean(), 1),
        "avg_claims"           : round(per_q["total_claims"].mean(), 1),
        "avg_precision_pct"    : round(per_q["precision"].mean() * 100, 1),
        "avg_recall_pct"       : round(per_q["recall"].mean() * 100, 1),
        "pct_zero_det_q"       : round((per_q["true_positives"] == 0).mean() * 100, 1),
        "exp1_judge_pct"       : round(exp1_judge_pct, 1),
        "delta_vs_judge_pp"    : round(detection_rate - exp1_judge_pct, 1),
        "per_q_df"             : per_q,
        "type_rates_df"        : type_rates,
    }


def analyze_experiment_2(
    all_model_results: Optional[dict] = None,
) -> None:
    """
    Full analysis and publication-quality figures for the quad-skeptic experiment.

    Per-model sections (A–F) run for each model independently.
    Cross-model section G produces the key comparison figures for the paper.

    Figures produced:
        Per model  : detection_rate, by_type, vs_pipeline, distribution
        Cross-model: skeptic_model_comparison (grouped bar, key paper figure)
                     skeptic_by_type_comparison (grouped bar by type)
    """
    # --- Load from disk if needed ---
    if all_model_results is None:
        all_model_results = {}
        for model_cfg in SKEPTIC_MODELS:
            if model_cfg["det_file"].exists():
                all_model_results[model_cfg["slug"]] = pd.read_csv(model_cfg["det_file"])
            else:
                logger.error("Missing results file: %s", model_cfg["det_file"])
                return

    # Check all models have data
    for model_cfg in SKEPTIC_MODELS:
        df = all_model_results.get(model_cfg["slug"])
        if df is None or df.empty:
            logger.error("[%s] No data to analyse.", model_cfg["name"])
            return

    # Load Exp1 baseline
    exp1_det       = pd.read_csv(EXP1_DET_FILE)
    exp1_s1        = exp1_det[exp1_det["stage"] == 1]
    exp1_judge_pct = exp1_s1["detected_by_judge"].mean() * 100

    # Compute stats for all models
    stats_by_slug = {}
    for model_cfg in SKEPTIC_MODELS:
        df    = all_model_results[model_cfg["slug"]]
        stats = compute_model_stats(df, model_cfg, exp1_judge_pct)
        stats_by_slug[model_cfg["slug"]] = stats


    for model_cfg in SKEPTIC_MODELS:
        s      = stats_by_slug[model_cfg["slug"]]
        det_df = all_model_results[model_cfg["slug"]]

        print(f"MODEL: {s['model_name']}")

        # A. Overall Detection Rate
        print("\nA. OVERALL DETECTION RATE (RECALL)")
        print(f"  Model                : {s['model_name']}")
        print(f"  N (questions run)    : {N}")
        print(f"  Total hallucinations : {s['total_hallucinations']}")
        print(f"  Detected (caught)    : {s['total_caught']}  ({s['detection_pct']:.1f}%)")
        print(f"  Missed               : {s['total_missed']}  ({s['survival_pct']:.1f}%)")

        # B. Flagging Behaviour
        print("\nB. FALSE POSITIVE RATE & FLAGGING BEHAVIOUR (per question)")
        print(f"  Avg flags issued per question      : {s['avg_flags']:.1f}")
        print(f"  Avg FALSE flags per question       : {s['avg_false_flags']:.1f}")
        print(f"  Avg claims reviewed per question   : {s['avg_claims']:.1f}")
        print(f"  Avg precision of flags             : {s['avg_precision_pct']:.1f}%")
        print(f"  Avg recall per question            : {s['avg_recall_pct']:.1f}%")
        print(f"  Questions with 0 detections        : {s['pct_zero_det_q']:.1f}%")

        # C. By Type
        print("\nC. DETECTION BY HALLUCINATION TYPE")
        for _, row in s["type_rates_df"].iterrows():
            print(
                f"  {row['injection_type']:20s}: {row['detection_pct']:5.1f}% detected "
                f"({int(row['caught'])}/{int(row['count'])}) | "
                f"{row['survival_pct']:.1f}% survived"
            )

        # D. Confidence Distribution
        caught_df = det_df[det_df["detected"]]
        if not caught_df.empty and "confidence" in caught_df.columns:
            conf_dist = caught_df.groupby("confidence").size().reset_index(name="count")
            conf_dist["pct"] = (conf_dist["count"] / len(caught_df) * 100).round(1)
            print("\nD. CONFIDENCE DISTRIBUTION -- CORRECT DETECTIONS")
            for _, row in conf_dist.iterrows():
                print(
                    f"  {row['confidence']:10s}: {int(row['count'])} detections  "
                    f"({row['pct']:.1f}%)"
                )

        # E. Survival Rate
        print("\nE. SURVIVAL RATE (undetected by skeptic)")
        print(
            f"  Overall survival rate  : {s['survival_pct']:.1f}%  "
            f"({s['total_missed']}/{s['total_hallucinations']})"
        )
        for _, row in s["type_rates_df"].iterrows():
            print(f"  {row['injection_type']:20s}: {row['survival_pct']:.1f}% survived")

        # F. vs Exp1 Judge
        print("\nF. COMPARISON: SKEPTIC vs EXP1 GPT-4o JUDGE (Stage 1, same position)")
        print(f"  Exp1 GPT-4o Judge @ Stage 1  : {exp1_judge_pct:.1f}%")
        print(
            f"  Skeptic {s['model_name']:20s}: {s['detection_pct']:.1f}%  "
            f"({s['delta_vs_judge_pp']:+.1f} pp)"
        )


        fig, ax = plt.subplots(figsize=(7, 5))
        bar = ax.bar(
            [s["model_name"]], [s["detection_pct"]],
            color=model_cfg["color"], edgecolor="white", width=0.35, zorder=3,
            label=f"{s['model_name']}  ({s['detection_pct']:.1f}%)",
        )
        ax.axhline(
            y=exp1_judge_pct, color="#E74C3C", ls="--", lw=2, zorder=2,
            label=f"Exp1 GPT-4o Judge @ Stage 1  ({exp1_judge_pct:.0f}%)",
        )
        ax.text(
            bar[0].get_x() + bar[0].get_width() / 2,
            s["detection_pct"] + 1.8,
            f"{s['detection_pct']:.1f}%",
            ha="center", fontsize=13, fontweight="bold",
        )
        ax.set_ylabel("Detection Rate (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Skeptic Agent Failure\n{s['model_name']} Cannot Reliably Detect Injected Hallucinations",
            fontsize=13, fontweight="bold", pad=15,
        )
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y", ls="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(
            EXP2_FIGURES / f"skeptic_detection_rate_{model_cfg['slug']}.png",
            dpi=300, bbox_inches="tight",
        )
        fig.savefig(
            EXP2_FIGURES / f"skeptic_detection_rate_{model_cfg['slug']}.pdf",
            bbox_inches="tight",
        )
        plt.show()

        TYPE_COLORS = {
            "dollar_amount": "#E74C3C",
            "percentage"   : "#3498DB",
            "large_number" : "#F39C12",
        }
        tr     = s["type_rates_df"]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        x_pos  = list(range(len(tr)))
        bars2  = ax2.bar(
            x_pos, tr["detection_pct"].values,
            color=[TYPE_COLORS.get(t, "#95A5A6") for t in tr["injection_type"].values],
            edgecolor="white", width=0.45, zorder=3,
        )
        for b, rate in zip(bars2, tr["detection_pct"].values):
            ax2.text(
                b.get_x() + b.get_width() / 2, rate + 1.5,
                f"{rate:.1f}%", ha="center", fontsize=11, fontweight="bold",
            )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(
            [t.replace("_", " ").title() for t in tr["injection_type"].values], fontsize=11
        )
        ax2.set_ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Skeptic Detection by Hallucination Type\n({s['model_name']})",
            fontsize=13, fontweight="bold",
        )
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.grid(True, alpha=0.3, axis="y", ls="--")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        fig2.savefig(
            EXP2_FIGURES / f"skeptic_by_type_{model_cfg['slug']}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.show()

        exp1_stages = (
            exp1_det.groupby("stage")
            .agg(judge_rate=("detected_by_judge", "mean"))
            .reset_index()
        )
        exp1_stages["judge_pct"] = exp1_stages["judge_rate"] * 100
        stage_labels = [
            "Stage 1\n(Researcher)", "Stage 2\n(Analyst)",
            "Stage 3\n(Writer)",     "Stage 4\n(Reviewer)",
        ]
        stages = exp1_stages["stage"].values

        fig3, ax3 = plt.subplots(figsize=(9, 5.5))
        ax3.plot(
            stages, exp1_stages["judge_pct"].values,
            "o-", color="#E74C3C", lw=2.5, ms=10, zorder=3,
            label="Exp1: GPT-4o Judge (in-pipeline)",
        )
        ax3.axhline(
            y=s["detection_pct"], color=model_cfg["color"], ls="--", lw=2,
            alpha=0.9, zorder=2,
            label=f"Skeptic: {s['model_name']}  ({s['detection_pct']:.0f}%)",
        )
        ax3.fill_between(
            stages, s["detection_pct"], exp1_stages["judge_pct"].values,
            where=exp1_stages["judge_pct"].values >= s["detection_pct"],
            alpha=0.08, color="#E74C3C", label="Detection gap",
        )
        ax3.set_xlabel("Pipeline Stage", fontsize=13, fontweight="bold")
        ax3.set_ylabel("Detection Rate (%)", fontsize=13, fontweight="bold")
        ax3.set_title(
            f"Skeptic Agent vs In-Pipeline Stage Detection\n"
            f"{s['model_name']} placed at best-case position (Stage 1) still underperforms",
            fontsize=13, fontweight="bold", pad=15,
        )
        ax3.set_xticks(stages)
        ax3.set_xticklabels(stage_labels, fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax3.legend(fontsize=10, loc="lower left")
        ax3.grid(True, alpha=0.3, ls="--")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        fig3.savefig(
            EXP2_FIGURES / f"skeptic_vs_pipeline_{model_cfg['slug']}.png",
            dpi=300, bbox_inches="tight",
        )
        fig3.savefig(
            EXP2_FIGURES / f"skeptic_vs_pipeline_{model_cfg['slug']}.pdf",
            bbox_inches="tight",
        )
        plt.show()

        per_q = s["per_q_df"]
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.hist(
            per_q["detection_pct"],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            color=model_cfg["color"], edgecolor="white", alpha=0.85, zorder=3,
        )
        ax4.axvline(
            x=s["detection_pct"], color="#E74C3C", ls="--", lw=2,
            label=f"Mean detection rate ({s['detection_pct']:.1f}%)",
        )
        ax4.set_xlabel("Per-Question Detection Rate (%)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Number of Questions", fontsize=12, fontweight="bold")
        ax4.set_title(
            f"Distribution of Per-Question Detection Rates\n"
            f"({s['model_name']} Skeptic, n={len(per_q)} questions)",
            fontsize=13, fontweight="bold",
        )
        ax4.set_xlim(0, 100)
        ax4.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis="y", ls="--")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        plt.tight_layout()
        fig4.savefig(
            EXP2_FIGURES / f"skeptic_detection_distribution_{model_cfg['slug']}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.show()

        logger.info("Per-model figures saved for %s", s["model_name"])


    print("G. CROSS-MODEL COMPARISON (all 4 skeptics)")
    for model_cfg in SKEPTIC_MODELS:
        s = stats_by_slug[model_cfg["slug"]]
        print(
            f"  {s['model_name']:30s}: {s['detection_pct']:5.1f}% detected | "
            f"{s['survival_pct']:.1f}% survived | "
            f"delta vs judge: {s['delta_vs_judge_pp']:+.1f} pp"
        )
    print(f"  {'Exp1 GPT-4o Judge (ref)':30s}: {exp1_judge_pct:5.1f}% detected")


    model_names  = [stats_by_slug[m["slug"]]["model_name"] for m in SKEPTIC_MODELS]
    model_rates  = [stats_by_slug[m["slug"]]["detection_pct"] for m in SKEPTIC_MODELS]
    model_colors = [m["color"] for m in SKEPTIC_MODELS]

    fig_c1, ax_c1 = plt.subplots(figsize=(11, 5.5))
    x_pos = list(range(len(SKEPTIC_MODELS)))
    bars  = ax_c1.bar(
        x_pos, model_rates, color=model_colors,
        edgecolor="white", width=0.5, zorder=3,
    )
    for bar, rate in zip(bars, model_rates):
        ax_c1.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.5,
            f"{rate:.1f}%",
            ha="center", fontsize=11, fontweight="bold",
        )
    ax_c1.axhline(
        y=exp1_judge_pct, color="#E74C3C", ls="--", lw=2, zorder=2,
        label=f"Exp1 GPT-4o Judge @ Stage 1  ({exp1_judge_pct:.0f}%)",
    )
    ax_c1.set_xticks(x_pos)
    ax_c1.set_xticklabels(model_names, fontsize=11)
    ax_c1.set_ylabel("Detection Rate (%)", fontsize=13, fontweight="bold")
    ax_c1.set_title(
        "Quad Skeptic Agent Failure\n"
        "All Four Models Fail to Reliably Detect Injected Hallucinations",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax_c1.set_ylim(0, 100)
    ax_c1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_c1.legend(fontsize=10, loc="upper right")
    ax_c1.grid(True, alpha=0.3, axis="y", ls="--")
    ax_c1.spines["top"].set_visible(False)
    ax_c1.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_c1.savefig(
        EXP2_FIGURES / "skeptic_model_comparison.png", dpi=300, bbox_inches="tight"
    )
    fig_c1.savefig(
        EXP2_FIGURES / "skeptic_model_comparison.pdf", bbox_inches="tight"
    )
    plt.show()
    logger.info("Cross-model comparison figure saved.")

    all_types = sorted(set(
        t for m in SKEPTIC_MODELS
        for t in stats_by_slug[m["slug"]]["type_rates_df"]["injection_type"].values
    ))
    n_types   = len(all_types)
    n_models  = len(SKEPTIC_MODELS)
    bar_width = 0.18
    x         = list(range(n_types))

    fig_c2, ax_c2 = plt.subplots(figsize=(12, 6))
    for i, model_cfg in enumerate(SKEPTIC_MODELS):
        tr    = stats_by_slug[model_cfg["slug"]]["type_rates_df"]
        rates = []
        for t in all_types:
            row = tr[tr["injection_type"] == t]
            rates.append(float(row["detection_pct"].values[0]) if len(row) > 0 else 0.0)

        offset  = (i - (n_models - 1) / 2) * bar_width
        bars_t  = ax_c2.bar(
            [xi + offset for xi in x],
            rates,
            width=bar_width,
            color=model_cfg["color"],
            edgecolor="white",
            zorder=3,
            label=model_cfg["name"],
        )
        for bar, rate in zip(bars_t, rates):
            ax_c2.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 1.0,
                f"{rate:.0f}%",
                ha="center", fontsize=8, fontweight="bold",
            )

    ax_c2.set_xticks(x)
    ax_c2.set_xticklabels(
        [t.replace("_", " ").title() for t in all_types], fontsize=11
    )
    ax_c2.set_ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
    ax_c2.set_title(
        "Detection by Hallucination Type — All Four Skeptic Models\n"
        "Failure is consistent across injection types and models",
        fontsize=13, fontweight="bold",
    )
    ax_c2.set_ylim(0, 100)
    ax_c2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_c2.legend(fontsize=9)
    ax_c2.grid(True, alpha=0.3, axis="y", ls="--")
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_c2.savefig(
        EXP2_FIGURES / "skeptic_by_type_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    logger.info("Cross-model by-type figure saved.")

    lines = []
    lines.append(f"""
EXPERIMENT 2 -- QUAD SKEPTIC PAPER-READY SUMMARY
Generated : {datetime.utcnow().isoformat()}Z
Models    : {', '.join(m['name'] for m in SKEPTIC_MODELS)}
N per model (questions run) : {N}
===========================================================================

HEADLINE NUMBERS:""")

    for model_cfg in SKEPTIC_MODELS:
        s = stats_by_slug[model_cfg["slug"]]
        lines.append(
            f"  {s['model_name']:30s}: {s['detection_pct']:.1f}% detected | "
            f"{s['survival_pct']:.1f}% survived | delta vs judge: {s['delta_vs_judge_pp']:+.1f} pp"
        )
    lines.append(f"  {'Exp1 GPT-4o Judge (ref)':30s}: {exp1_judge_pct:.1f}% detected\n")

    for model_cfg in SKEPTIC_MODELS:
        s  = stats_by_slug[model_cfg["slug"]]
        tr = s["type_rates_df"]
        lines.append(f"""
--- {s['model_name']} ---
  Total hallucinations   : {s['total_hallucinations']}
  Detected               : {s['total_caught']}  ({s['detection_pct']:.1f}%)
  Missed                 : {s['total_missed']}  ({s['survival_pct']:.1f}%)
  Avg precision          : {s['avg_precision_pct']:.1f}%
  Avg recall             : {s['avg_recall_pct']:.1f}%
  Avg flags/question     : {s['avg_flags']:.1f}
  Avg false flags        : {s['avg_false_flags']:.1f}
  Questions 0 detections : {s['pct_zero_det_q']:.1f}%

  By Type:
{tr[['injection_type', 'detection_pct', 'survival_pct', 'count', 'caught']].to_string(index=False)}""")

    model_list_str = ", ".join(m["name"] for m in SKEPTIC_MODELS)
    rate_strs      = " and ".join(
        f"{stats_by_slug[m['slug']]['model_name']} detected only "
        f"{stats_by_slug[m['slug']]['detection_pct']:.1f}%"
        for m in SKEPTIC_MODELS
    )

    lines.append(f"""
FOR SECTION 3 (Why Skepticism Fails):
  "We evaluate four state-of-the-art skeptic agents -- {model_list_str} --
   each placed at the optimal position in the pipeline (immediately after
   the Researcher, Stage 1). Despite this best-case placement, {rate_strs}
   of the injected hallucinations. The consistency of this failure across
   four architecturally distinct models confirms that the limitation is
   structural: without access to ground truth or external verification tools,
   no LLM can reliably distinguish 15-40% perturbations of plausible
   financial figures."

COMPARISON TABLE (for paper):
  | Method                                    | Detection | Survival | Has Tools? |
  |-------------------------------------------|-----------|----------|------------|
  | Exp1 GPT-4o Judge (Stage 1)               | {exp1_judge_pct:5.1f}%   | {100 - exp1_judge_pct:5.1f}%  | No         |""")

    for model_cfg in SKEPTIC_MODELS:
        s = stats_by_slug[model_cfg["slug"]]
        lines.append(
            f"  | Skeptic: {s['model_name']:<32s} | {s['detection_pct']:5.1f}%   "
            f"| {s['survival_pct']:5.1f}%  | No         |"
        )

    paper_text = "\n".join(lines)
    print(paper_text)
    with open(EXP2_DIR / "paper_ready_summary.txt", "w") as f:
        f.write(paper_text)


    # skeptic_model_rates.csv — one row per model, consumed by Code 22
    model_rates_rows = []
    for model_cfg in SKEPTIC_MODELS:
        s = stats_by_slug[model_cfg["slug"]]
        model_rates_rows.append({
            "model"                  : s["model_name"],
            "model_id"               : s["model_id"],
            "n_questions"            : N,
            "total_hallucinations"   : s["total_hallucinations"],
            "total_caught"           : s["total_caught"],
            "detection_rate"         : s["detection_rate"],
            "detection_pct"          : s["detection_pct"],
            "survival_pct"           : s["survival_pct"],
            "avg_precision_pct"      : s["avg_precision_pct"],
            "avg_recall_pct"         : s["avg_recall_pct"],
            "avg_flags_per_question" : s["avg_flags"],
            "avg_false_flags"        : s["avg_false_flags"],
            "pct_questions_zero_det" : s["pct_zero_det_q"],
            "exp1_judge_stage1_pct"  : s["exp1_judge_pct"],
            "delta_vs_judge_pp"      : s["delta_vs_judge_pp"],
        })
    pd.DataFrame(model_rates_rows).to_csv(EXP2_DIR / "skeptic_model_rates.csv", index=False)

    # skeptic_type_rates.csv — all models, all types
    type_rates_all = pd.concat(
        [
            stats_by_slug[m["slug"]]["type_rates_df"].assign(model=m["name"])
            for m in SKEPTIC_MODELS
        ],
        ignore_index=True,
    )
    type_rates_all.to_csv(EXP2_DIR / "skeptic_type_rates.csv", index=False)

    # per_question_stats.csv — all models
    per_q_all = pd.concat(
        [
            stats_by_slug[m["slug"]]["per_q_df"].assign(model=m["name"])
            for m in SKEPTIC_MODELS
        ],
        ignore_index=True,
    )
    per_q_all.to_csv(EXP2_DIR / "per_question_stats.csv", index=False)

    pd.DataFrame(model_rates_rows).to_csv(EXP2_DIR / "experiment2_summary.csv", index=False)

    logger.info("All Experiment 2 artefacts saved to %s", EXP2_DIR)
    print(f"\nAll output saved to: {EXP2_DIR}")


analyze_experiment_2(all_model_results)


for model_cfg in SKEPTIC_MODELS:
    if model_cfg["checkpoint"].exists():
        model_cfg["checkpoint"].unlink()
        logger.info("Checkpoint removed: %s", model_cfg["checkpoint"].name)

print("EXPERIMENT 2 (QUAD SKEPTIC) COMPLETE")
for model_cfg in SKEPTIC_MODELS:
    df   = all_model_results[model_cfg["slug"]]
    rate = df["detected"].mean() * 100 if not df.empty else 0.0
    print(f"  {model_cfg['name']:30s}: {rate:.1f}% detection rate")
print(f"\n  N per model    : {N}")
print(f"  Figures        : {EXP2_FIGURES}")
print(f"  Summary        : {EXP2_DIR / 'paper_ready_summary.txt'}")
print(f"  Model rates    : {EXP2_DIR / 'skeptic_model_rates.csv'}")
print(f"  Combined CSV   : {EXP2_DIR / 'skeptic_detection_results_combined.csv'}")