# ❄️ The Hallucination Snowball

Official repository for the ICML 2026 Failure modes in Agentic AI (FAGEN) Workshop paper: **"The Hallucination Snowball: Modeling Error Propagation as State Transitions in Multi-Agent LLM Pipelines"**.

**Authors:** Prabhjot Singh and Bhushan Pawar

## 📖 Overview
Sequential multi-agent LLM pipelines chain specialized agents without verification at handoffs, creating a structural flaw with measurable and severe consequences. We show that hallucinations injected early in the pipeline do not merely persist; they transform:
* **Raw numerical facts** $\to$ **Derived computations** $\to$ **Narrative prose** $\to$ **Invisible conclusions**

At each transformation, detectability degrades near-irreversibly. We formalize this as a first-order Markov process with empirically measured per-boundary escape probabilities. We show that *when* you verify matters more than *whether* you verify, and we demonstrate how boundary gates at early stages can intercept this cascading failure.

## 📊 Key Findings

Based on an evaluation of **346 automatically injected quantitative hallucinations** across **140 FinanceBench test cases** in a 4-agent sequential pipeline (Researcher $\to$ Analyst $\to$ Writer $\to$ Reviewer):

* **Experiment 1 (Detectability Decay)**
  * `gpt-4o` detection drops from **72.0%** at Stage 1 to **50.9%** at Stage 4 (a 21.1 pp decline).
  * **23.7%** of hallucinations survive completely undetected in the final output.
* **Experiment 2 (Capability Ceilings of LLM Skepticism)**
  * Evaluated 4 state-of-the-art models as skeptic agents at Stage 1: Meta-Llama-3-70B-Instruct (51.4%), gemini-2.5-flash (72.5%), DeepSeek-V3.2 (75.7%), and Qwen3.5-397B-A17B (87.0%).
  * No model approaches 100%. Even projecting the strongest model (87.0%) through measured decay rates leaves ~35–40% undetected in the final report.
* **Experiment 3 (Timing Dominates Method)**
  * **Vanilla (No Verification):** 60.7% survival rate.
  * **End-Check:** Deterministic numeric gate after Stage 4 only drops survival to 58.4% (a statistically negligible 2.3 pp improvement).
  * **Ours (Boundary Gates):** Deterministic gates after every handoff plummet the survival rate to **16.2%** (a 42.2 pp reduction over End-Check, Cohen's $h = -0.911$). Hallucination-free reports jump from 18.6% (Vanilla) to 68.6%.
* **Markov State-Transition Model**
  * Escape probabilities skyrocket as the hallucination is laundered: $S_1{\to}S_2$ (**24.6%**), $S_2{\to}S_3$ (**48.3%**), $S_3{\to}S_4$ (**89.3%**).
  * By the final stage ($S_3{\to}S_4$), nearly 90% of narrative-embedded hallucinations are structurally unrecoverable by any downstream gate.

## 📁 Repository Structure
```text
.
├── codes/
│   ├── data_download.py    # Downloads/formats FinanceBench & generates analytical wrappers
│   ├── experiment_1.py     # End-to-end pipeline tracing hallucination detectability decay
│   ├── experiment_2.py     # Evaluation of Quad Skeptic Agents (Llama, Gemini, DeepSeek, Qwen)
│   └── experiment_3.py     # Mitigation methods: Vanilla vs. End-Check vs. Boundary Gates
├── figures/                # Visualizations (Snowball decay, gap charts, trajectories)
├── financebench_experiment_sheet.xlsx # Generated data used across experiments
└── raw_results/            # Cached experimental data, checkpoints, and snapshots
    ├── experiment_1/       # Trajectories, stage detection rates, injection logs
    ├── experiment_2/       # Skeptic model detections and unified bootstrap CI stats
    └── experiment_3/       # Verification gate performance and metrics
```

## ⚙️ Setup & Installation
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/prabhjotschugh/hallucination-snowball.git
   cd hallucination-snowball
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install pandas matplotlib seaborn tqdm openai langchain-openai langchain-core langgraph datasets openpyxl google-genai
   ```
4. Configure your API keys (Add these to your environment to run all scripts):
   ```bash
   export OPENAI_API_KEY="sk-..."
   # Additional keys might be needed depending on your router for HF models
   ```

## 🚀 Running the Experiments
All code is designed to be run sequentially. You can execute them in your local Python environment or adjust them for Google Colab.
* **Data Preparation:** `python codes/data_download.py` (Generates `financebench_experiment_sheet.xlsx`)
* **Experiment 1 (Tracing the Snowball):** `python codes/experiment_1.py` (Uses the generated sheet)
* **Experiment 2 (Skeptic Evaluation):** `python codes/experiment_2.py`
* **Experiment 3 (Mitigation Pipeline):** `python codes/experiment_3.py`