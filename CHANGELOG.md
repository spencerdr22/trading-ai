# Trading-AI System — Adaptive RL Upgrade
**Version:** 2025-11  
**Maintainer:** Spencer Druckenbroad  
**AI Collaborator:** GPT-5 Adaptive Systems Suite  

---

# 🚀 Overview

This update introduces a **full adaptive reinforcement-learning system**, a modernized supervised training pipeline, hybrid strategy logic, and automated code deployment tooling.

These changes transition the system from a traditional classifier-based engine into a **continuously self-improving, dual-model trading intelligence platform**.

---

# 🧩 What Was Added

## 1. Adaptive Reinforcement Learning Framework
New modules under `app/adaptive/`:

- **model_hub.py**  
  Centralized ML model registry with SQLite persistence.

- **reward.py**  
  Risk-adjusted reward engine (Sharpe, Sortino, Max Drawdown, leverage penalties).

- **learner.py**  
  Offline RL engine (PyTorch REINFORCE, medium policy network).

- **optimizer.py**  
  Optuna Bayesian hyperparameter tuning (adaptive nightly tuning).

- **run_offline_rl.py**  
  CLI runner with Plotly HTML charts.

---

## 2. Supervised System Enhancements
Updated: `app/ml/trainer.py`

- RandomForest  
- LSTM (PyTorch)  
- Hybrid LSTM + RandomForest (stacked model)  
- Metadata persistence  
- Clean architecture & optimized preprocessing  

---

## 3. Strategy Engine Upgrade
Updated: `app/strategy/engine.py`

- Dynamic blending of supervised & RL policy outputs  
- RL-confidence weighted decision system  
- Full LSTM/Hybrid model compatibility  

This module now acts as the **true trading brain**, intelligently merging both learning paradigms.

---

## 4. DevOps & Tooling
Added: `generate_code_changes.py`

This script:

- Writes all updated files  
- Creates `.bak` backups  
- Generates `.patch` files  
- Logs changes  
- Supports dry-run safety mode  

---

# 🧠 Architecture Diagrams

## A. ASCII Architecture Diagram

app/
├── adaptive/
│ ├── model_hub.py
│ ├── reward.py
│ ├── learner.py
│ ├── optimizer.py
│ └── run_offline_rl.py
│
├── ml/
│ ├── features.py
│ └── trainer.py
│
├── strategy/
│ └── engine.py
│
├── data/
│ ├── loader.py
│ ├── simulator.py
│ └── live_feed.py
│
├── execution/
│ ├── paper_executor.py
│ └── tradovate_client.py
│
├── db/
│ ├── init.py
│ └── schema.py
│
├── monitor/
│ └── logger.py
│
└── utils/
└── helpers.py



---

## B. Mermaid Architecture Diagram

```mermaid
flowchart TD

subgraph DATA
  loader[data.loader]
  sim[data.simulator]
  live[data.live_feed]
end

subgraph DB
  schema[db.schema]
  init[db.init]
end

subgraph ML
  features[ml.features]
  trainer[ml.trainer]
end

subgraph ADAPTIVE
  hub[adaptive.model_hub]
  reward[adaptive.reward]
  learner[adaptive.learner]
  optuna[adaptive.optimizer]
  cli[adaptive.run_offline_rl]
end

subgraph STRATEGY
  engine[strategy.engine]
end

subgraph EXEC
  paper[execution.paper_executor]
  liveexec[execution.tradovate_client]
end

loader --> features
sim --> features

features --> trainer
trainer --> hub

hub --> engine

engine --> paper
engine --> liveexec

engine --> learner
learner --> hub
reward --> learner
optuna --> learner
cli --> learner
