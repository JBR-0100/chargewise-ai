# ChargeWise AI

Intelligent EV Charging Demand Prediction and Infrastructure Planning System.

## Project Status

This repository currently contains the initial project structure and development environment setup. Core functionality will be implemented in later stages.

---

## Setup Guide

Follow these steps to set up the project locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/neel3o115/chargewise-ai.git
cd chargewise-ai
```

---

### 2️⃣ Create a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

After activation, your terminal prompt should display:

```bash
(venv)
```

---

### 3️⃣ Install Dependencies

```bash
pip3 install -r requirements.txt
```

---

## Verify Setup

Run the welcome script to confirm that everything is working correctly:

```bash
python3 welcome.py
```

If successful, a console animation and welcome message will appear.

---

## Project Structure

```
chargewise-ai/
│
├── processed/              # Preprocessed output CSVs
│   ├── zone_hourly_volume_long.csv
│   └── zone_model_results.csv
│
├── app.py                  # Streamlit dashboard / main UI
├── preprocess_run.py       # Data preprocessing pipeline script
├── preprocessing.ipynb     # Exploratory preprocessing notebook
│
├── requirements.txt
├── .gitignore
├── welcome.py              # Environment verification script
└── README.md
```