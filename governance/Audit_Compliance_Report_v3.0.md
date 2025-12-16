# 🛡️ AI AGENT EXECUTION AUDIT & COMPLIANCE REPORT
### **Template v3.0 — Implementation-Locked, No EDA**
### Aligned to SIR v1.9 (Frozen)
### Purpose: Verify the agent’s execution *against* the SIR and Governance documents, with strict protection of the codebase.

---

# 0. META

- **Strategy Name:**  
- **Strategy ID:**  
- **SIR Path:**  
- **Governance Path:**  
- **Agent Name / Version:**  
- **Execution Timestamp (UTC):**  
- **Run Type(s):**  
  - [ ] Labeling  
  - [ ] Feature Engineering  
  - [ ] Model Training  
  - [ ] Backtest  
  - [ ] Report Generation  
  - [ ] Promotion Prep  

- **Git Commit Before Run:**  
- **Git Commit After Run:**  
- **Approved Artifacts Directory:**  
- **Repository Root:**  

---

# 1. HARD RESTRICTION COMPLIANCE  
**These are absolute rules. Any violation is an automatic FAIL.**

## 1.1 File Creation / Deletion Rules

### 1.1.1 Unauthorized File Creation  
The agent **may NOT** create any new source files, modules, scripts, or directories.

Allowed new outputs:  
- CSV files  
- Charts (PNG, SVG)  
- JSON logs  
- Markdown reports  
- Pickles / model artifacts (if defined)  
- MLflow artifacts  

**Did the agent create any unapproved files?**  
- [ ] No  
- [ ] Yes → list:
```
path/to/file (why?)
```

### 1.1.2 Unauthorized File Deletion  
The agent **may NOT delete or rename any existing file.**

**Did the agent delete or rename any files?**  
- [ ] No  
- [ ] Yes → list:
```
path/to/file (details)
```

### 1.1.3 Unauthorized Modification of Protected Files  
Protected files include:  
- Strategy code  
- Labeler classes  
- Feature engineering scripts  
- Training scripts  
- Backtesting scripts  
- Infrastructure/config files  
- Any governance or SIR documents  

The agent may only modify files **explicitly stated as editable in the task**.

**Did the agent modify any protected files?**  
- [ ] No  
- [ ] Yes → list modifications and justification:

**Compliance Status:** `PASS / FAIL / NEEDS REVIEW`

---

# 2. NAMING CONVENTION COMPLIANCE

## 2.1 MLflow Run Names  
Expected patterns:

- **Models:** `model__<Strategy_ID>__<Model_Family>`  
- **Backtests:** `backtest__<Strategy_ID>__<RunType>`

Fill observed:

| Run Type | Observed Name | Compliant? | Notes |
|----------|----------------|------------|--------|
| Model    |                | Y/N        |        |
| Backtest |                | Y/N        |        |

## 2.2 Model Registry Names  
Expected pattern:  
`<Strategy_ID>__<Model_Family>__v<Major.Minor>`

Observed:
```
<List registry names and compliance>
```

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 3. CONFIGURATION & ENVIRONMENT COMPLIANCE

## 3.1 .env Usage  
- Did the agent read config from `.env`?  
  - [ ] Yes  
  - [ ] No → list hardcoded values:

## 3.2 Introduction of New Env Vars  
- [ ] No  
- [ ] Yes → list:

## 3.3 Use of Approved Configuration Patterns  
- All configuration handled via `.env` or governance-approved config files  
- No hardcoded risk parameters  
- No secret keys embedded  

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 4. LABELING COMPLIANCE (SIR Section 4)

## 4.1 Triple-Barrier Implementation
From SIR:
- TP:  
- SL:  
- Timeout:  

### Code Location:
```
path/to/labeler.py::class/method
```

### Deviations
- [ ] None  
- [ ] Yes → list:

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

## 4.2 Label Semantics  
Expected from SIR:
```
{ -1, 0, 1 }
```

### Actual Values Produced:
```
List observed classes
```

### Extra Labels?
- [ ] No  
- [ ] Yes → list:

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 5. FEATURE SPECIFICATION COMPLIANCE (SIR Section 5)

## 5.1 Feature Implementation Map

| SIR Feature | Implemented? | Code Location | Transformations | Leakage-Free? |
|-------------|--------------|----------------|------------------|----------------|
|             | Y/N          |                |                  | Y/N            |

## 5.2 Extra Features Introduced  
- [ ] None  
- [ ] Yes → list & justify:

## 5.3 Leakage Checks  
- Rolling windows correct?  
- No forward leakage?  
- Correct use of past-dependent features?  

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 6. FORMAL STRATEGY SPEC COMPLIANCE (SIR Section 6)

## 6.1 Entry Criteria  
- SIR Summary:  
- Code Location:  

**Match?**  
- [ ] Yes  
- [ ] Partially  
- [ ] No

## 6.2 Exit Criteria  
- SIR Summary:  
- Code Location:  

**Match?**  
- [ ] Yes  
- [ ] Partially  
- [ ] No

## 6.3 Filters  
**Match SIR exactly?**  
- [ ] Yes  
- [ ] No → list unauthorized filters:

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 7. MODEL TRAINING COMPLIANCE (SIR Section 7)

## 7.1 Required Training Outputs  
- [ ] Training metrics logged  
- [ ] Validation metrics logged  
- [ ] Model artifact saved  
- [ ] Correct loss function  
- [ ] Correct target  
- [ ] Hyperparameter plan followed  
- [ ] MLflow tags added  
- [ ] Feature importance / SHAP (if required)  
- [ ] No new training scripts created  

### MLflow Run IDs:
```
List runs
```

### Deviations:
List mismatches.

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 8. BACKTEST COMPLIANCE (SIR Section 8–9)

## 8.1 Backtest Engine  
Must use governance-approved backtest engine.  
- [ ] Yes  
- [ ] No → details:

## 8.2 Required Metrics Logged  
- [ ] R-multiples distribution  
- [ ] Points/pips  
- [ ] Equity curve  
- [ ] Drawdowns  
- [ ] Win rate  
- [ ] Expectancy  
- [ ] Regime breakdown (if required)  
- [ ] Monte Carlo results  

## 8.3 Trade Logs  
- [ ] Trades logged to DB  
- [ ] Each trade tagged with `strategy_id`  
- [ ] Each trade tagged with `model_name`  
- [ ] Position lifecycle intact  

**Compliance:** `PASS / FAIL / NEEDS REVIEW`

---

# 9. PROHIBITED BEHAVIOR CHECKS  
Agent MUST answer these truthfully:

- **Created new source files?**  
  - [ ] No  
  - [ ] Yes → list violations

- **Deleted or renamed files?**  
  - [ ] No  
  - [ ] Yes → list

- **Introduced new dependencies?**  
  - [ ] No  
  - [ ] Yes → list

- **Modified infrastructure or unrelated code?**  
  - [ ] No  
  - [ ] Yes → list

- **Changed risk parameters without instruction?**  
  - [ ] No  
  - [ ] Yes → details

- **Hallucinated non-existent files/modules?**  
  - [ ] No  
  - [ ] Yes → list

**OVERALL PROHIBITED BEHAVIOR STATUS:**  
`PASS / FAIL / NEEDS REVIEW`

---

# 10. AGENT SELF-VERIFICATION

```
1. I loaded the SIR and Governance before execution.        (Yes/No)
2. I only modified files permitted by the task.             (Yes/No)
3. I created no scripts or source files.                    (Yes/No)
4. I deleted or renamed no files.                           (Yes/No)
5. All logic matches the SIR.                               (Yes/No)
6. All outputs are reproducible.                            (Yes/No)
7. All deviations are documented.                           (Yes/No)
```

---

# 11. HUMAN REVIEW VERDICT

- **Execution Result:** `PASS / CONDITIONAL PASS / FAIL`  
- **Approved to proceed to next pipeline step?** `YES / NO`  
- **Reviewer:**  
- **Date:**  
- **Required Remediation:**

---

# END OF AI AGENT EXECUTION AUDIT & COMPLIANCE REPORT v3.0