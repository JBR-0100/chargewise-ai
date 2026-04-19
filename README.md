<img width="1476" height="842" alt="image" src="https://github.com/user-attachments/assets/8d818dbf-329a-4ea7-a7da-38a9b5c46fa1" />

# ChargeWise AI 

**Intelligent EV Charging Demand Prediction & Agentic Infrastructure Planning.**

ChargeWise AI transforms raw charging telemetry into actionable urban strategy. By combining high-fidelity Machine Learning forecasting with a reasoning-based Agentic AI workflow, the system helps city planners navigate the transition to electric mobility with mathematical precision.

---

## Problem Statement
The global electric vehicle (EV) transition faces a multi-billion dollar bottleneck: **Reactive Infrastructure Planning.** 

Urban charging demand is highly volatile, fluctuating based on commuter cycles, residential habits, and seasonal shifts. Without accurate foresight, municipalities suffer from:
- **Grid Instability**: Localized failures due to unmanaged peak-load spikes.
- **Wasted CAPEX**: Over-building infrastructure in low-demand zones while high-density areas remain underserved.
- **User Attrition**: Long wait times and "range anxiety" that directly slow down EV adoption rates.

## Expected Outcomes
ChargeWise AI is designed to move city planning from "best guesses" to "data-to-strategy" reality:
- **80%+ Forecasting Accuracy**: Capturing neighborhood-level demand patterns across 300+ urban zones.
- **Strategic Proactivity**: Anticipating hardware needs (DCFC vs Level 2) *before* congestion reaches critical levels.
- **Optimized Grid Integration**: Implementing algorithmic load-balancing to reduce grid stress during evening peak windows by up to 30%.
- **Professional Deliverables**: Automated, regulatory-compliant site reports that can be used for municipal budget approvals.

---

## Key Features

### 1. Agentic Planning Core (RAG)
Beyond simple numbers, our system uses **Retrieval-Augmented Generation (RAG)** to "read" official infrastructure guidelines and apply them to local data. The agent provides site-specific advice backed by industry standards (e.g., NEVI).

### 2. Multi-Model ML Pipeline
A robust predictive engine using **Ridge Regression** and **Random Forests**, validated through 5-fold **TimeSeriesSplit cross-validation** to ensure reliability on true "future" data.

### 3. Comprehensive Planner Dashboard
- **Future Forecasting**: Predictive tools with 95% Confidence Intervals.
- **Strategic Heatmaps**: City-scale visualization of demand intensity by hour and zone.
- **Model Transparency**: Deep-dives into the performance (MAE, R2) of every single zone-level model.
- **Automated PDF Export**: Professional report generation for immediate stakeholder delivery.

---

## Project Structure

```
chargewise-ai/
├── agent/                  # Core Logic Modules
│   ├── data_agent.py       # Feature engineering & anomaly detection
│   ├── ml_pipeline.py      # Cross-validated model training & forecasting
│   ├── rag.py              # TF-IDF Retrieval over guidelines
│   ├── planning_agent.py   # State-managed AI reasoning assistant
│   └── pdf_export.py       # Professional report generation
│
├── knowledge/              # Knowledge Base for RAG
│   └── ev_planning_guidelines.txt
│
├── processed/              # High-fidelity Data Artifacts
│   ├── zone_hourly_volume_long.csv
│   └── zone_model_results.csv
│
├── app.py                  # Streamlit Multi-tab Dashboard
├── preprocess_run.py       # Data Pipeline Execution Script
├── .env                    # Environment Secrets (Ignored by Git)
├── requirements.txt
└── README.md
```

---


## Setup & Usage

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Secrets
Create a `.env` file in the project root and add your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_key_here
```

### 3. Run the System
To launch the interactive planning dashboard:
```bash
streamlit run app.py
```

*Note: For first-time setups, ensure you have the processed CSVs in the `processed/` directory, or run `python preprocess_run.py` to regenerate the baseline records.*

---

## Feedback Resolution
This version of ChargeWise AI incorporates full resolution of Milestone 1 feedback, including:
- **Full Training Pipeline**: Implemented with MAE/RMSE/R2 metrics.
- **Model Robustness**: Verified via 5-fold cross-validation.
- **Advanced Visuals**: Added system-wide heatmaps and peak analysis grids.
- **Interactive UI**: Added future-time selection and interactive site report generation.
