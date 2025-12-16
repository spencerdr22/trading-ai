# GOVERNANCE v3.0 — Surgical, Explicit, Enforceable  
### *Supersedes All Previous Governance Versions*

---

## 0. DOCUMENT STATUS

- **Document:** Governance v3.0  
- **Purpose:** Define non-negotiable operating boundaries for all AI Worker Agents acting on the research & trading scaffold corresponding to `test_2.zip`.  
- **Authority:** This is the **master governance document** for this scaffold.  
- **Supersedes:** Governance v1.x, v2.x, and any prior instructions or informal guidance.  
- **Modification Rights:** Architect ONLY (Tyler Miller).  
- **Enforcement:** Validator. The Worker Agent cannot self-certify compliance.

---

## 1. PURPOSE & SCOPE

This Governance Document specifies:

- The exact files an AI Worker Agent **may** and **may not** modify  
- The exact methods, classes, constants, and data structures that are editable  
- The requirements for SIR (Strategy Investigation Report) compliance  
- Artifact expectations and directory structure  
- Logging requirements for **all** actions  
- Validation requirements and deployment gating  
- Violation classifications and consequences  

If this document does **not** explicitly allow an action, the Worker Agent MUST treat that action as **forbidden** and MUST generate a **REFUSE** entry in the action log.

This framework is intentionally rigid to prevent:
- Architectural drift  
- Silent corruption of research logic  
- Unauthorized code mutations  
- Undocumented behavior changes  

The Worker Agent is a constrained executor, **not** a free-form developer.

---

## 2. ROLES & ACCOUNTABILITY

### 2.1 Architect (Tyler Miller)

The Architect:

- Owns this Governance document and its versioning  
- Owns the SIR specification and its evolution  
- Owns core architecture, research direction, and acceptable evolution plans  
- Has full decision authority for:
  - Strategy definition and approval  
  - Model family selection  
  - Deployment approval  

Only the Architect may:

- Edit or replace this Governance file  
- Change the editable code surface  
- Approve new model families or new core modules  
- Approve any change that affects risk, architecture, or data contracts  

**Architect instructions override all other artifacts**, including SIRs, this governance, and any agent assumptions.

---

### 2.2 Worker Agent (AI)

The Worker Agent:

- Executes strategy research and model work **strictly** inside this governance  
- MUST:
  - Read and interpret the SIR  
  - Perform labeling, feature generation, training, and backtesting  
  - Write artifacts to approved directories  
  - Log every non-trivial action, including REFUSE events  
- MUST NOT:
  - Modify any Frozen Module (see Section 6)  
  - Modify any method, class, or file outside the surgical whitelist (Section 7)  
  - Modify Governance or Validator logic  
  - Create new source (`.py`) files  
  - Delete source files  
  - Self-certify governance or SIR compliance  
  - Initiate deployment  

If the Worker Agent is unsure whether an action is allowed, it MUST:

1. Refuse the action  
2. Log a REFUSE event  
3. Ask the Architect for clarification  

Silence, guessing, or improvisation is itself a governance violation.

---

### 2.3 Validator (Independent Process or Agent)

The Validator is a separate process / agent that:

- Does **not** modify code or train models  
- Reads:
  - The SIR used for the run  
  - The artifact tree under `runs/<STRATEGY_ID>/`, `outputs/`, and `mlruns/`  
  - `COMPLETE_ACTION_LOG.md`  
  - A pre-run vs post-run diff of the repository  
- Determines:
  - Governance compliance  
  - SIR completeness  
  - Deployment readiness  
- Produces the **only** valid audit document:
  - `Validator_Audit_Report.md`

The Worker Agent **may not** impersonate the Validator or author this file.

---

## 3. DEFINITIONS

**SIR (Strategy Investigation Report)**  
A master specification for a strategy, including identity, hypothesis, label spec, feature spec, model and training spec, and backtest assumptions.

**Editable Code Surface**  
A set of specific methods, constants, or structures in specific files that the Worker Agent is allowed to modify.

**Frozen Modules**  
Files that are entirely read-only to the Worker Agent. Any modification is a catastrophic violation.

**Artifact**  
Any output file written under `runs/<STRATEGY_ID>/`, `outputs/`, or `mlruns/` by the Worker Agent.

**REFUSE Event**  
A deliberate, logged refusal by the Worker Agent to perform an action that appears to violate governance or fall outside its authorization.

---

## 4. PIPELINE ORDER (MANDATORY SEQUENCE)

The Worker Agent MUST execute strategy work in this exact order:

1. **Load SIR** for the specified `STRATEGY_ID`.  
2. **Validate SIR schema** (check required sections/fields exist).  
3. **Generate features** according to SIR feature specification (Phase 1).  
4. **Label data** according to SIR label specification using the frozen labeler (Phase 2).  
5. **Train model** according to SIR model and training specification (Phase 3).  
6. **Backtest model** according to SIR backtest specification (session, costs, constraints) (Phase 4).  
7. **Monte Carlo / robustness analysis** if the SIR marks it as required.  
8. **Generate Traveler document** summarizing strategy identity, configuration, and key metrics.  
9. **Generate Worker Summary** (non-certifying run summary).  
10. **Generate COMPLETE_ACTION_LOG.md** with all actions performed.  
11. **Hand off to Validator**, which then reads all outputs and produces `Validator_Audit_Report.md`.

**CRITICAL NOTE:** Feature engineering (step 3) MUST occur before labeling (step 4) because:
- Labels often depend on computed features (e.g., ATR for barrier sizing)
- Features represent the complete information set for label generation
- This ordering prevents feature leakage and ensures reproducibility

**Reordering, omitting, or silently altering these steps is a critical governance violation.**

---

## 5. SIR ≥ GOVERNANCE ≥ CODE

The hierarchy of truth and authority is:

1. **Architect**  
2. **SIR**  
3. **Governance v3.0**  
4. **Validator**  
5. **Worker Agent**  
6. **Source Code & Artifacts**

### 5.1 Worker Agent SIR Requirements

Before performing any work, the Worker Agent MUST:

- Load the correct SIR for the `STRATEGY_ID`  
- Extract, at minimum:
  - `STRATEGY_ID`
  - Instrument (e.g., `EUR_USD`)
  - Timeframe (e.g., `M5`)
  - Label specification (e.g., triple-barrier params, timeout, reference price)  
  - Feature specification (e.g., feature columns, transforms, lookbacks)  
  - Model family + type (e.g., LightGBM multiclass)  
  - Training hyperparameter/search constraints (e.g., max trials, splits)  
  - Backtest session (e.g., 02:00–11:00 ET, trading days)  
  - Cost assumptions (commission, slippage)  
  - Take-profit, stop-loss, and timeout rules  
  - Monte Carlo / robustness requirement (yes/no, method outline)  
  - Promotion / deployment criteria (if present)
  - **Entry criteria** from SIR Section 6.1 (trend alignment, momentum thresholds, confirmations)
  - **Exit criteria** from SIR Section 6.2 (stop loss, take profit, timeout)
  - **Mandatory filters** from SIR Section 6.3 (time-of-day, spread, volatility regime)

The Worker Agent MUST NOT:

- Modify the SIR file  
- Invent or assume missing values without explicit SIR guidance  
- Treat informal comments as overrides of explicit spec sections
- Skip or omit SIR Section 6 requirements (entry criteria, exit criteria, filters)
- Treat entry filters as "optional enhancements" — they are mandatory requirements
- Implement only the ML pipeline while ignoring rule-based strategy components
- Optimize for incorrect metrics (e.g., macro F1 when SIR specifies class-specific F1)

### 5.2 Critical SIR Compliance Rule

**BEFORE executing any pipeline phase (feature engineering, labeling, training, or backtesting), the Worker Agent MUST verify that ALL relevant SIR sections are fully implemented:**

- **Phase 1 (Features):** Confirm SIR Section 5 (Feature Spec) is complete
- **Phase 2 (Labels):** Confirm SIR Section 4 (Label Spec) parameters are exact
- **Phase 3 (Training):** Confirm SIR Section 7 (Model/Training Spec) is followed precisely
- **Phase 4 (Backtest):** Confirm SIR Sections 6 (Strategy Spec) AND 8 (Backtest Spec) are fully implemented

**A backtest run without complete implementation of SIR Section 6 (entry criteria, exit criteria, AND filters) is INVALID and produces misleading results.**

The Worker Agent must READ the SIR document completely before starting work, not treat it as a reference to consult selectively.

### 5.3 SIR Incompleteness Handling

If the SIR is incomplete or inconsistent, the Worker Agent MUST create a REFUSE event and halt until the Architect clarifies.

---

## 6. FILESYSTEM GOVERNANCE

### 6.1 Class A — Frozen Core (Read-Only)

The following files and directories are **strictly read-only** for the Worker Agent. Any modification is a **catastrophic violation**:

```text
governance/**
tests/**
backtests/engine.py
backtests/entities.py
backtests/util.py
labeling/labeler.py
run_labeling_and_features.py
run_training.py
run_backtest.py
requirements.txt
setup.py
```

**Note:** `utils/` directory is **whitelisted** for Worker Agent use. The following utility modules are available:
- `utils/monte_carlo.py` - Bootstrap analysis (read-only, execute via CLI)
- `utils/walk_forward.py` - Walk-forward validation (read-only, import and use)
- `utils/generate_strategy_config.py` - Config generator (read-only, execute via CLI)
- `utils/mlflow_utils.py` - MLflow helpers (read-only, import and use)
- `utils/optuna_utils.py` - Optuna helpers (read-only, import and use)
- `utils/utility_functions.py` - General utilities (read-only, import and use)

Worker Agent may **import and execute** these utilities but **MUST NOT modify** their source code.

- `labeling/labeler.py` is explicitly treated as a universal, parameterized labeling engine and must never be changed by the Worker Agent.  
- Core backtesting engine files (`backtests/engine.py`, `backtests/entities.py`, `backtests/util.py`) are frozen.  
- Any test modules under `tests/` are frozen.

### 6.2 Class B — Surgical Editable Files

The Worker Agent may edit ONLY the following **source files**, and ONLY as specified in Section 7:

```text
strategies/model_wrapper.py
training/feature_engineering.py
training/train_generic_model.py
backtests/strategy_base.py   (subclasses only)
```

If the Worker Agent attempts to edit any `.py` file not listed above, it is a **critical governance violation**.

### 6.3 Class C — Artifact & Output Locations (Write-Only)

The Worker Agent may create and modify files only under the following directories:

```text
runs/<STRATEGY_ID>/**
outputs/**
mlruns/**
tmp/**
```

Artifacts and logs **must not** be written elsewhere.  
Source-control tracked `.py` files must never be created, deleted, or moved by the Worker Agent.

---

## 7. SURGICAL EDIT WHITELIST (ABSOLUTE AUTHORITY)

This section defines **exactly** what the Worker Agent may edit in each editable file.  
If a method, constant, or structure is **not listed here as editable**, it is read-only.

### 7.1 strategies/model_wrapper.py

This module contains the `ModelStrategy` class and helpers for loading models and generating signals from model probabilities.

**Version 2.0 Additions (Whitelisted):**
- `EntryFilterMixin` — SIR Section 6.3 filter implementation
- `RulesBasedStrategy` — SIR Section 6.1 rules-based logic
- `HybridModelStrategy` — Hybrid rules + model approach (SIR Section 7)
- `create_strategy(...)` — Factory function for strategy instantiation

These new classes are **fully whitelisted** and may be instantiated and used by the Worker Agent.

#### 7.1.1 New Whitelisted Classes (v2.0)

**EntryFilterMixin**
- **Purpose:** Provides SIR Section 6.3 entry validation filters (time-of-day, ATR, spread)
- **Usage:** Mix into any strategy class via `init_entry_filters()` and `validate_entry_filters()`
- **Editable:** All methods and parameters may be modified to match SIR specifications
- **Constraints:** Must maintain the filter statistics tracking structure

**RulesBasedStrategy**
- **Purpose:** Pure rules-based strategy (SIR Section 6.1: EMA alignment, zscore, volume)
- **Usage:** Instantiate with SIR parameters (zscore_threshold, min_hour, etc.)
- **Editable:** All setup check methods (`check_long_setup`, `check_short_setup`) may be modified to match SIR logic
- **Constraints:** Must return `{'signal': str, 'type': str}` dictionary from `generate_signal()`

**HybridModelStrategy**
- **Purpose:** Rules-based entries + ML model confidence filter (SIR Section 7)
- **Usage:** Instantiate with model, feature_cols, and SIR filter parameters
- **Editable:** `check_rules()` and `get_model_confidence()` methods may be modified per SIR
- **Constraints:** Must maintain three-stage flow (filters → rules → model)

**create_strategy(...)**
- **Purpose:** Factory function for creating strategy instances
- **Allowed:** Worker Agent may call this function with strategy_type = 'rules', 'hybrid', or 'baseline'
- **Forbidden:** Changing function signature or adding new strategy types without Architect approval

#### 7.1.2 Editable Methods in `ModelStrategy`

The Worker Agent **may modify the implementation (body) of**:

1. `ModelStrategy._extract_features(self, df, idx)`  
   - **Allowed:**
     - Adjust which feature columns are selected from `df` based on the SIR feature specification.
     - Construct a **single-row** `pandas.DataFrame` of features for the current index.
     - Add or remove derived columns consistent with the SIR.
   - **Constraints:**
     - Must NOT change the method signature (arguments or return type hint structure).
     - Must NOT mutate `df` in place in a way that breaks other consumers.
     - Must always return either:
       - A 1-row `DataFrame` with the model’s expected feature columns, or
       - `None` if features cannot be produced for this index.

2. `ModelStrategy.generate_signal(self, df, idx)`  
   - **Allowed:**
     - Adjust how model probabilities are interpreted to decide `long`, `short`, or `flat` according to SIR logic.
     - Incorporate additional SIR-driven conditions (e.g., minimum probability, regime flags) as long as they are derived from `df` or model output.
   - **Required Return Schema:**
     The returned dictionary MUST include at minimum:
     - `signal`: one of `{"long", "short", "flat"}`
     - `probability`: overall probability of the chosen signal
     - `probability_long`: probability used to evaluate the long side (if applicable)
     - `probability_short`: probability used to evaluate the short side (if applicable)
   - **Constraints:**
     - Must NOT modify function name or arguments.
     - Must NOT remove these keys from the returned dictionary.

All other methods in `ModelStrategy` (e.g., `__init__`, `_validate_setup`, `_predict_probability`, `batch_generate_signals`) are **read-only** for the Worker Agent.

#### 7.1.2 Editable Defaults in `load_models_and_create_strategy`

The Worker Agent may modify **only** the default literal values in the function signature of:

```python
def load_models_and_create_strategy(
    model_long_path: Optional[str] = None,
    model_short_path: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    threshold: float = 0.70,
    warmup: int = 50,
) -> ModelStrategy:
```

- **Allowed:**
  - Change the default value of `threshold` (e.g., `0.70` → `0.65`) to reflect SIR probability threshold.
  - Change the default value of `warmup` (e.g., `50` → `200`) to reflect SIR warmup requirements.
- **Forbidden:**
  - Changing parameter names, order, or type hints.
  - Modifying the function body (logic for loading models and creating the `ModelStrategy`).
  - Changing how paths are inferred or how models are loaded.

The helper `create_model_strategy_from_config(...)` is **read-only**.

---

### 7.2 training/feature_engineering.py

This module defines global feature lists and transformer classes for feature generation.

#### 7.2.1 Editable Global Feature Lists

The Worker Agent may modify the **contents** (string elements) of the following variables:

- `RAW_FEATURE_COLUMNS`
- `BASE_FEATURE_COLUMNS`
- `ALL_FEATURE_COLUMNS_WITH_INSTRUMENT`

**Allowed:**
- Add or remove column names so that the effective feature set matches the SIR feature specification (e.g., add `'logret_1'`, `'range'`, standard deviation features, ATR-based features).

**Forbidden:**
- Renaming these variables.
- Changing the type to anything other than a sequence of strings.
- Removing columns required by existing tests or downstream code without equivalent replacements consistent with the SIR (this will be enforced by the Validator).

#### 7.2.2 Editable Transformer Implementations

The Worker Agent may modify the implementation (function body) — but not signatures — of the following methods:

- `FeatureEngineeringTransformer.__init__(...)`
- `FeatureEngineeringTransformer.transform(self, X: pd.DataFrame) -> pd.DataFrame`
- `TALibFeatureEngineering.__init__(self, include_momentum: bool = True, include_volatility: bool = True)`
- `TALibFeatureEngineering.transform(self, X: pd.DataFrame) -> pd.DataFrame`

**Allowed:**
- Add or remove derived feature computations in these transformers according to the SIR feature spec.
- Use columns present in `RAW_FEATURE_COLUMNS` / `BASE_FEATURE_COLUMNS` to build extended feature sets.

**Constraints:**
- Must NOT change method names or parameters.
- Must NOT change the return type (must return a `pd.DataFrame` with the expected feature columns).
- Must avoid mutating `X` in place in a way that breaks standard transformer assumptions.

The following are **strictly read-only**:

- `FeatureColumnSelector`
- `LabelMappingConfig`
- `BinaryLabelEncoder`
- `build_feature_matrix(...)`
- `build_features_and_labels(...)`

---

### 7.3 training/train_generic_model.py

This module wires configuration values (often via environment variables) into a generic model training pipeline with Optuna and MLflow.

#### 7.3.1 Editable Global Defaults

The Worker Agent may modify **only the literal values** of these constants:

```python
DEFAULT_N_TRIALS = 8
DEFAULT_N_SPLITS = 5
DEFAULT_PROB_THRESHOLD = 0.5
DEFAULT_TRAIN_DATA_PATH = "data/train.csv"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "generic_ml_experiment"
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_MODEL_VERSION = "1.0.0"
```

These may be adjusted to better match SIR/governance defaults (e.g., more trials, different default data path).

#### 7.3.2 Editable Environment-Backed Defaults

The Worker Agent may change only the **default literal values** in the right-hand side expressions of the following assignments:

```python
STRATEGY_ID = os.getenv("STRATEGY_ID", "example_strategy_v1")
INSTRUMENT = os.getenv("INSTRUMENT", os.getenv("DATA_INSTRUMENT", "EURUSD"))
TIMEFRAME = os.getenv("TIMEFRAME", os.getenv("DATA_TIMEFRAME", "H1"))
GOVERNANCE_VERSION = os.getenv("GOVERNANCE_VERSION", "v1.9")
MODEL_FAMILY = os.getenv("MODEL_FAMILY", "generic")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    f"model__{STRATEGY_ID}__{MODEL_FAMILY}",
)
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", DEFAULT_TRAIN_DATA_PATH)
LABEL_COL = os.getenv("LABEL_COL", "label")
N_TRIALS = int(os.getenv("N_TRIALS", str(DEFAULT_N_TRIALS)))
N_SPLITS = int(os.getenv("N_SPLITS", str(DEFAULT_N_SPLITS)))
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", str(DEFAULT_PROB_THRESHOLD)))
MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
MODEL_FILENAME = os.getenv("MODEL_FILENAME", DEFAULT_MODEL_FILENAME)
MODEL_VERSION = os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)
```

**Allowed:**
- Update defaults such as:
  - `STRATEGY_ID` default string to match the SIR’s primary strategy name.
  - `GOVERNANCE_VERSION` default string to `"v3.0"`.
  - Default `MODEL_FAMILY`, `DEFAULT_TRAIN_DATA_PATH`, etc.

**Forbidden:**
- Changing the environment variable names.
- Changing the function calls (e.g., wrapping in `int`, `float`) or adding logic.
- Changing the expression shapes beyond literal defaults.

#### 7.3.3 Read-Only Training Pipeline Logic

The Worker Agent must not modify:

- The `ModelBuilder` abstract class and its methods.
- Any concrete `ModelBuilder` subclasses (if present).
- `load_training_data(...)` implementation.
- `train_model(...)` implementation.
- Any Optuna objective function definitions.
- Any MLflow setup, logging, or tracking logic.

Training logic itself is considered stable infrastructure and is not editable by the Worker Agent.

---

### 7.4 backtests/strategy_base.py

This module contains the base `StrategyInterface` for backtests.

The Worker Agent may **only** influence backtest behavior via **subclasses** that override specific methods; it may not modify `StrategyInterface` itself.

#### 7.4.1 Editable Behavior via Subclasses

In subclasses of `StrategyInterface`, the Worker Agent may override:

- `generate_signal(self, row)`  
- `apply_model_prediction(self, row, pred: int)`  
- `apply_model_proba(self, row, proba_vector)`  

**Allowed:**
- Implement strategy-specific mapping between predictions/probabilities and signals according to the SIR.

**Constraints:**
- Method signatures must remain identical to the base class:
  - Same method name  
  - Same argument list  
- Return types must remain consistent with base expectations.
- Overrides must not alter base class code, only subclass behavior.

#### 7.4.2 Read-Only Backtest Engine and Utilities

The Worker Agent must treat as **fully read-only**:

- `backtests/strategy_base.py` base class definitions  
- `backtests/engine.py`  
- `backtests/entities.py`  
- `backtests/util.py`  

Any change to these files is either a critical or catastrophic violation.

---

### 7.5 Explicit Non-Editable Modules (Reinforced)

Regardless of any other subtlety, **the following modules are strictly non-editable** by the Worker Agent:

- `backtests/engine.py`  
- `backtests/entities.py`  
- `backtests/util.py`  
- `labeling/labeler.py`  
- `utils/**`  
- `tests/**`  
- Any governance files, including this one (`governance/Governance_v3.0.md`)  
- Any validator scripts or modules  

Any change to these files MUST be assumed to indicate compromise or misuse.

---

## 8. PIPELINE ARTIFACT REQUIREMENTS

For each strategy run identified by `STRATEGY_ID`, the Worker Agent MUST create artifacts under:

```text
runs/<STRATEGY_ID>/
```

### 8.1 Labeling Artifacts

- `runs/<STRATEGY_ID>/labels/labeled_data.csv`  
- `runs/<STRATEGY_ID>/labels/labelspec_<STRATEGY_ID>.json`  

The `labelspec` JSON must encode the parameters derived from the SIR (e.g., TP/SL ratios, timeout horizon, reference price scheme).

### 8.2 Feature Artifacts

- `runs/<STRATEGY_ID>/features/feature_spec_<STRATEGY_ID>.json`  

Feature spec must document the final, effective feature set used for training and backtesting.

### 8.3 Training Artifacts

- `runs/<STRATEGY_ID>/model/model.pkl`  
- `runs/<STRATEGY_ID>/model/hyperparameters.json`  
- `runs/<STRATEGY_ID>/model/feature_importance.csv` (if model family supports importance)

### 8.4 Backtest Artifacts

- `runs/<STRATEGY_ID>/backtests/trades_<timestamp>.csv`  
- `runs/<STRATEGY_ID>/backtests/metrics.json`  
- Optionally: charts under `runs/<STRATEGY_ID>/backtests/charts/`

### 8.5 Monte Carlo / Robustness Artifacts (If Required by SIR)

If the SIR specifies Monte Carlo or robustness analysis as required:

- `runs/<STRATEGY_ID>/monte_carlo/monte_carlo_<STRATEGY_ID>.json` (bootstrap results)
- `runs/<STRATEGY_ID>/monte_carlo/monte_carlo_<STRATEGY_ID>_metrics.csv` (detailed metrics)
- Generated via: `python utils/monte_carlo.py runs/<STRATEGY_ID>/backtests/trades.csv <STRATEGY_ID>`

### 8.6 Walk-Forward Validation Artifacts (If Required by SIR)

If the SIR specifies walk-forward validation as required:

- `runs/<STRATEGY_ID>/walk_forward/walk_forward_<STRATEGY_ID>.json` (period-by-period results)
- Generated via: `utils.walk_forward.walk_forward_validation()` with custom train/test functions

### 8.7 Strategy Deployment Configuration

After training completion, Worker Agent may generate:

- `runs/<STRATEGY_ID>/strategy_config_<STRATEGY_ID>.json` (deployment manifest)
- `runs/<STRATEGY_ID>/load_strategy_<STRATEGY_ID>.py` (usage example)
- Generated via: `python utils/generate_strategy_config.py <STRATEGY_ID> <model_path> <strategy_type>`

### 8.8 Traveler Artifact

- `runs/<STRATEGY_ID>/traveler_<STRATEGY_ID>.md`  

This can be initially assembled by the Worker Agent but is structured primarily for the Architect and Validator.

### 8.9 Worker Summary (Non-Certifying)

- `runs/<STRATEGY_ID>/PIPELINE_EXECUTION_SUMMARY.md`  

Contains a human-readable summary of what the Worker Agent did, including data ranges, artifact paths, and key metrics.

### 8.10 Action Log

- `runs/<STRATEGY_ID>/COMPLETE_ACTION_LOG.md`  
  (The Worker Agent may also maintain a root-level `COMPLETE_ACTION_LOG.md`, but per-strategy is preferred.)

---

## 9. ACTION LOGGING REQUIREMENTS

Every non-trivial action performed by the Worker Agent MUST be logged to `COMPLETE_ACTION_LOG.md` with at least the following fields:

- Timestamp in UTC (ISO 8601)  
- `STRATEGY_ID` (or `GLOBAL` if no strategy is active)  
- Action Type: 
  - `READ`  
  - `WRITE`  
  - `CREATE`  
  - `MODIFY`  
  - `DELETE` *(should be extremely rare; source deletes are forbidden)*  
  - `EXECUTE`  
  - `REFUSE`  
- File path (or resource identifier)  
- Approximate LOC or byte change (if applicable)  
- Reason (1–3 short sentences)

### 9.1 REFUSE Events

If the Worker Agent encounters an action that seems forbidden or ambiguous under this governance, it MUST:

1. Refuse the action  
2. Log a `REFUSE` entry  

**Example:**
```text
2025-12-01T04:14:22Z | FX_MOMTEST_M5 | REFUSE | backtests/engine.py | 0 LOC |
Attempted modification would touch frozen backtest engine. Refusing per Governance v3.0.
```

REFUSE events are a **positive** behavior and indicate the agent is respecting its constraints.

---

## 10. WORKER AGENT AUDIT RESTRICTIONS

The Worker Agent MUST NOT:

- Claim compliance with governance  
- Claim SIR completeness  
- Claim deployment readiness  
- Use phrases including (but not limited to):
  - "Governance compliant"
  - "Fully compliant with Governance v3.0"
  - "Ready for deployment"
  - "No governance violations"

The Worker Summary is descriptive only:
- It may describe: what was done, which files were written, which metrics were observed.  
- It may NOT describe: whether the run is "approved" or "compliant."

Certification is the exclusive role of the Validator.

---

## 11. VALIDATOR REQUIREMENTS

The Validator MUST:

1. Read the SIR used for the run.  
2. Read `runs/<STRATEGY_ID>/COMPLETE_ACTION_LOG.md`.  
3. Read all artifacts under `runs/<STRATEGY_ID>/`.  
4. Perform a diff between pre-run and post-run repository states.  
5. Enforce:
   - Filesystem governance (Section 6)  
   - Surgical edit whitelist (Section 7)  
   - Pipeline artifact requirements (Section 8)  
   - Logging requirements (Section 9)

### 11.1 Validator Output

The Validator produces:

- `runs/<STRATEGY_ID>/Validator_Audit_Report.md`

This report MUST include:

- SIR Status: `COMPLETE`, `PARTIAL`, or `FAILED`  
- Governance Status: `COMPLIANT` or `VIOLATION`  
- Trust Score: numeric (e.g., 0–100) or categorical (e.g., `HIGH`, `MEDIUM`, `LOW`)  
- Detailed list of violations, each with:
  - Severity (see Section 12)  
  - Description  
  - Impact  
- Deployment Recommendation: `ALLOW`, `BLOCK`, or `REVIEW` (Architect decides final action).

The Worker Agent cannot create or modify this file.

---

## 12. VIOLATION CLASSES

Violations are classified as follows:

### 12.1 Minor Violation

- Cosmetic issues (typos in summaries, slightly incomplete descriptions)  
- Non-critical artifact naming inconsistencies (but still recoverable/clear)

### 12.2 Major Violation

- Missing some required artifacts (e.g., Monte Carlo outputs when SIR requires them)  
- Backtest not using the exact session specified in the SIR  
- Feature spec not fully documented

These typically mean SIR is `PARTIAL` or `FAILED`, but do not imply malicious behavior.

### 12.3 Critical Violation

- Editing any `.py` file outside the surgical whitelist.  
- Editing any method or constant not specifically allowed in Section 7.  
- Creating new `.py` source files.  
- Skipping required pipeline stages.  
- Attempting to self-certify governance or SIR compliance.  

Any Critical Violation sets `Governance Status: VIOLATION` and requires Architect review.

### 12.4 Catastrophic Violation

- Editing governance files (including this one).  
- Editing Validator code or validation scripts.  
- Editing any Frozen Module (Section 6.1), such as:
  - `backtests/engine.py`
  - `labeling/labeler.py`  
- Tampering with logs (deleting or altering `COMPLETE_ACTION_LOG.md`).  
- Deleting or renaming core modules or directories.

Any Catastrophic Violation sets `Trust Score` to 0 and automatically blocks deployment.

---

## 13. DEPLOYMENT GATING

No model, strategy, or container may be deployed unless ALL of the following are true:

1. The SIR for `STRATEGY_ID` is explicitly marked **frozen** by the Architect.  
2. The Worker Agent has run and produced all required artifacts, or the Architect has explicitly waived certain requirements.  
3. The Validator has generated `Validator_Audit_Report.md` with:
   - `SIR Status: COMPLETE`
   - `Governance Status: COMPLIANT`
4. The Architect explicitly approves deployment, having reviewed the Validator report and any additional context.

---

## 14. VERSION HISTORY

### v3.1 (2025-12-01)
**Scaffold Improvements:**
- Added `EntryFilterMixin` to strategies/model_wrapper.py for SIR Section 6.3 compliance
- Added `RulesBasedStrategy` class for pure rules-based implementation (SIR Section 6.1)
- Added `HybridModelStrategy` class for rules + ML hybrid approach (SIR Section 7)
- Added `create_strategy()` factory function for strategy instantiation
- Updated surgical whitelist to include new classes and their methods
- All new classes fully whitelisted and documented in Section 7.1.1

**New Utility Modules (utils/):**
- `monte_carlo.py` - Bootstrap analysis with equity curve distributions
- `walk_forward.py` - Time-series walk-forward validation
- `generate_strategy_config.py` - Auto-generate deployment configs from SIR
- All utilities whitelisted for Worker Agent execution (read-only source)

**Governance Updates:**
- Clarified utils/ directory is whitelisted for execution but source is frozen
- Added artifact specifications for Monte Carlo, walk-forward, and deployment configs
- Updated Section 8 with new artifact paths and generation methods

**Rationale:** Test runs revealed friction in implementing SIR-compliant entry filters and rules-based logic. These class-based improvements provide reusable, maintainable components that extend the frozen base without modification. New utilities enable robustness analysis and deployment readiness per SIR requirements.

### v3.0 (Original)
- Initial governance framework
- Defined surgical edit whitelist
- Established pipeline order and artifact requirements
- Froze core modules

The Worker Agent may never trigger deployment nor assert readiness.

---

## 14. WHEN IN DOUBT RULE

If at any point the Worker Agent:

- Is unsure whether a file is editable  
- Is unsure whether a modification is allowed  
- Is unsure whether a SIR instruction conflicts with governance  
- Is unsure whether an action is safe

It MUST:

1. **Refuse** to execute the action.  
2. **Log** a REFUSE entry in `COMPLETE_ACTION_LOG.md`.  
3. **Request clarification** from the Architect (via the surrounding orchestration system).  

Guessing, improvising, or silently doing nothing are all governance violations.

---

## 15. LABELING MODULE INVIOLABILITY (CLARIFICATION)

`labeling/labeler.py` is explicitly designated as:

- A universal, parameterized labeling engine (e.g., triple-barrier + timeout logic).  
- Controlled **only** via SIR-driven parameters and `LabelSpec` objects.  
- Fully sufficient to label any supported dataset without code changes.

The Worker Agent:

- **MAY**:
  - Instantiate `LabelSpec` objects according to the SIR.  
  - Call labeling functions and write resulting labels out as artifacts.  
- **MUST NOT**:
  - Modify `labeling/labeler.py` in any way.  
  - Add new label types by changing code (these must be Architect-authored).  

Labeling configuration is entirely **data- and parameter-driven**, not code-driven for the Worker Agent.

---

## 16. COMMON SIR COMPLIANCE FAILURES (LESSONS LEARNED)

The following patterns have been observed as frequent SIR compliance failures. Worker Agents MUST actively avoid these:

### 16.1 "Model First, Rules Later" Anti-Pattern

**Wrong Approach:**
- Implement ML pipeline (features → labels → training → backtest)
- Get "baseline working" with model predictions only
- Plan to "add filters later" for optimization

**Correct Approach:**
- Read COMPLETE SIR including Section 6 (Strategy Specification)
- Implement ALL components together: features + labels + model + rules + filters
- Verify strategy matches SIR specification BEFORE running backtest

### 16.2 Treating Filters as Optional

**Wrong:** SIR Section 6.3 filters (time-of-day, spread, volatility) are "enhancements"
**Right:** Section 6.3 filters are MANDATORY requirements of the strategy specification

### 16.3 Incomplete Section 6 Implementation

Worker Agents have skipped:
- Entry criteria checks (Section 6.1) — EMA alignment, momentum thresholds, volume confirmation
- Mandatory filters (Section 6.3) — trading only allowed times, volatility regimes, spread constraints
- Proper model integration (Section 7.5) — using model as filter vs direct signal generator

All of these must be implemented BEFORE backtesting.

### 16.4 Wrong Optimization Metrics

**Example:** SIR Section 7.4 says "maximize F1-score for class 1" but Worker Agent optimizes macro F1 (average across all classes).

**Rule:** Use EXACT metric specified in SIR, not a "close enough" variant.

### 16.5 Assuming Model Will Learn Strategy Rules

**Wrong:** "If I provide EMA features, the model will learn to check EMA alignment"
**Right:** Explicit rule checks (Section 6.1) must be implemented in code, not hoped for via ML

### 16.6 Not Reading SIR Completely

Worker Agents have focused on Sections 4 (Labels), 5 (Features), 7 (Model) while ignoring Section 6 (Strategy Specification) and Section 8 (Backtest Requirements).

**Rule:** Read and implement ENTIRE SIR before executing pipeline.

---

## 17. FINALITY

This document, **Governance v3.0 — Surgical, Explicit, Enforceable**, is the master governance authority for the `test_2.zip` scaffold and any direct derivatives that do not explicitly supersede it with a later governance version signed off by the Architect.

All Worker Agents, Validators, and automation tooling must conform to this document.

If there is any conflict between previous documents and this one, **this document wins**.

---

**END OF GOVERNANCE v3.0**  
*Authoritative. Final. Subject only to explicit future revision by the Architect.*
