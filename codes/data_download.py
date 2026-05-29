"""FinanceBench Dataset Loader & Analytical Prompt Wrapper

Downloads FinanceBench from HuggingFace and wraps each question into a complex
analytical prompt using OpenAI GPT-4o-mini for the 4-agent pipeline.
"""
import os
import re
import hashlib
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

# Configuration
PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "raw_results"
OUTPUT_FILE = PROJECT_DIR / "financebench_experiment_sheet.xlsx"
LOG_FILE = DATA_DIR / "data_download.log"

DATASET_ID = "PatronusAI/financebench"
RANDOM_SEED = 42
WRAP_MODEL = "gpt-4o-mini"
WRAP_TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SAVE_EVERY = 5

# Load API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
def _safe_str(value) -> str:
    """Coerce any field value to a clean string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if v).strip()
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value).strip()


def _generate_question_id(question: str, doc_name: str, idx: int) -> str:
    """Deterministic, human-readable question ID."""
    raw = f"{question}||{doc_name}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:8].upper()
    return f"FB_{idx + 1:03d}_{h}"


logger.info("Downloading FinanceBench dataset: %s", DATASET_ID)
ds = load_dataset(DATASET_ID, split="train")
logger.info("Loaded %d records", len(ds))

# Inspect schema
logger.info("Dataset columns: %s", ds.column_names)
print("\n--- Raw sample record (first entry) ---")
print(json.dumps(ds[0], indent=2, default=str)[:3000])

rows = []
for idx, record in enumerate(tqdm(ds, desc="Parsing records")):

    question = _safe_str(record.get("question") or record.get("query"))
    answer = _safe_str(record.get("answer") or record.get("ground_truth") or record.get("label"))
    evidence = _safe_str(record.get("evidence") or record.get("evidence_text") or record.get("context"))
    doc_name = _safe_str(record.get("doc_name") or record.get("document_name") or record.get("source_doc"))
    doc_link = _safe_str(record.get("doc_link") or record.get("document_link") or record.get("source_url"))
    question_type = _safe_str(record.get("question_type") or record.get("type"))
    domain = _safe_str(record.get("domain") or record.get("sector"))

    qid = _generate_question_id(question, doc_name, idx)

    rows.append({
        "question_id": qid,
        "original_question": question,
        "answer": answer,
        "evidence": evidence,
        "doc_name": doc_name,
        "doc_link": doc_link,
        "question_type": question_type,
        "domain": domain,
    })

df = pd.DataFrame(rows)
logger.info("DataFrame built — %d rows × %d cols", len(df), len(df.columns))
print(f"\nShape: {df.shape}")
df.head(3)

WRAPPER_SYSTEM_PROMPT = """You are an expert financial research director at a top-tier investment bank.

YOUR TASK:
You will receive a simple financial lookup question (e.g., "What is Boeing's FY2022 COGS?").
Convert it into a rich, multi-part analytical research directive that would require a full team of:
  1. A Financial Researcher — to find and report exact figures from SEC filings
  2. A Financial Analyst — to analyze trends, compute ratios, and benchmark
  3. A Report Writer — to produce a professional narrative report
  4. A Quality Reviewer — to verify coherence and accuracy

CRITICAL RULES:
- The wrapped prompt MUST preserve the core financial fact being asked about. The same company, same metric, same time period must be central to the analysis.
- Do NOT mention the 4 agents or pipeline — write it as a single research directive that naturally demands all four capabilities.
- The prompt should be 3-5 sentences long. Be specific and professional. No fluff.
- Include explicit instructions to: (a) retrieve exact figures from official filings, (b) analyze trends or compute ratios, (c) write a professional report, and (d) ensure accuracy.
- Vary your phrasing naturally — do not repeat the same sentence structure across different questions.
- Output ONLY the wrapped prompt. No preamble, no explanation, no quotes around it."""

WRAPPER_USER_TEMPLATE = """Convert this simple financial question into a complex analytical research directive:

QUESTION: {question}
COMPANY FILING: {doc_name}

Wrapped analytical prompt:"""

def wrap_question_openai(question: str, doc_name: str) -> str:
    """Convert a FinanceBench question into an analytical prompt."""
    user_msg = WRAPPER_USER_TEMPLATE.format(
        question=question,
        doc_name=doc_name,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=WRAP_MODEL,
                temperature=WRAP_TEMPERATURE,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": WRAPPER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            wrapped = response.choices[0].message.content.strip()
            # Strip surrounding quotes if model adds them
            if wrapped.startswith('"') and wrapped.endswith('"'):
                wrapped = wrapped[1:-1].strip()
            return wrapped

        except Exception as e:
            logger.warning(
                "  Attempt %d/%d failed for question: %s | Error: %s",
                attempt, MAX_RETRIES, question[:50], str(e)
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.error("  All retries exhausted. Using fallback.")
                return (
                    f"Conduct a comprehensive financial analysis related to "
                    f"the following question about {doc_name}: {question} "
                    f"Retrieve exact figures from official SEC filings, analyze "
                    f"trends and ratios, and produce a professional report."
                )


logger.info("Wrapping %d questions via %s", len(df), WRAP_MODEL)

# Load or initialize checkpoint
CHECKPOINT_FILE = DATA_DIR / "wrap_checkpoint.json"

if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
    logger.info("Resumed from checkpoint — %d already wrapped", len(checkpoint))
else:
    checkpoint = {}

wrapped_prompts = []
api_cost_input_tokens = 0
api_cost_output_tokens = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Wrapping via OpenAI"):
    qid = row["question_id"]

    if qid in checkpoint:
        wrapped_prompts.append(checkpoint[qid])
        continue

    wrapped = wrap_question_openai(
        question=row["original_question"],
        doc_name=row["doc_name"],
    )
    wrapped_prompts.append(wrapped)
    checkpoint[qid] = wrapped
    if (idx + 1) % BATCH_SAVE_EVERY == 0:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f)
        logger.info("  Checkpoint saved at %d/%d", idx + 1, len(df))

# Final checkpoint save
with open(CHECKPOINT_FILE, "w") as f:
    json.dump(checkpoint, f)

df["wrapped_prompt"] = wrapped_prompts

# Add experiment tracking columns
df["injection_done"] = False
df["pipeline_run"] = False
df["notes"] = ""

# Reorder columns
col_order = [
    "question_id",
    "original_question",
    "wrapped_prompt",
    "answer",
    "evidence",
    "doc_name",
    "doc_link",
    "question_type",
    "domain",
    "injection_done",
    "pipeline_run",
    "notes",
]
df = df[[c for c in col_order if c in df.columns]]

print(f"\nFinal DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
df[["question_id", "original_question", "wrapped_prompt"]].head(5)

logger.info("Writing Excel workbook: %s", OUTPUT_FILE)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="experiment_data", index=False)

    ws = writer.sheets["experiment_data"]
    for col_cells in ws.columns:
        col_letter = col_cells[0].column_letter
        max_len = max(len(str(c.value or "")) for c in col_cells)
        ws.column_dimensions[col_letter].width = min(max_len + 4, 60)
    ws.freeze_panes = "A2"
    meta = pd.DataFrame([
        ("dataset_source", DATASET_ID),
        ("huggingface_url", f"https://huggingface.co/datasets/{DATASET_ID}"),
        ("github_url", "https://github.com/patronus-ai/financebench"),
        ("num_questions", str(len(df))),
        ("wrapping_model", WRAP_MODEL),
        ("wrapping_temperature", str(WRAP_TEMPERATURE)),
        ("generated_at", datetime.utcnow().isoformat() + "Z"),
        ("random_seed", str(RANDOM_SEED)),
    ], columns=["key", "value"])
    meta.to_excel(writer, sheet_name="metadata", index=False)

logger.info("Excel export complete: %d rows", len(df))

df_check = pd.read_excel(OUTPUT_FILE, sheet_name="experiment_data")
print(f"\nVerification — reloaded {df_check.shape[0]} rows × {df_check.shape[1]} cols")
print(f"Columns: {list(df_check.columns)}\n")

for i in [0, 37, 74, 112, 149]:
    if i < len(df_check):
        row = df_check.iloc[i]
        print(f"{'='*80}")
        print(f"  ID       : {row['question_id']}")
        print(f"  Original : {row['original_question']}")
        print(f"  Wrapped  : {row['wrapped_prompt']}")
        print(f"  Answer   : {str(row['answer'])[:80]}")
        print(f"  Doc      : {row['doc_name']}")
        print()

print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
print("Done — ready for pipeline experiments.")

# Clean up checkpoint after successful export
if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()
    logger.info("Checkpoint file cleaned up.")
