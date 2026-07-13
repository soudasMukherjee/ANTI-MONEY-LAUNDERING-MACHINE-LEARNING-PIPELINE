# Anti-Money Laundering Machine Learning Pipeline

An interactive Streamlit dashboard for detecting suspicious financial transactions using machine learning, anomaly detection, risk scoring, and forensic visual analytics.

## Project Overview

This project helps analyze transaction records and identify possible money laundering or fraud patterns. It combines a trained classification pipeline with engineered transaction features, anomaly detection signals, and visual investigation tools so analysts can review high-risk activity quickly.

The application supports both real CSV uploads and synthetic simulation mode, making it useful for testing AML workflows, demonstrating fraud detection logic, and exploring how transaction risk changes under different conditions.

## Key Features

- Upload transaction CSV files and generate fraud probability scores.
- Choose between XGBoost and Random Forest classifier models.
- Classify transactions into LOW, MID, and HIGH risk bands.
- Detect anomalies using Isolation Forest and One-Class SVM signals.
- Generate synthetic transaction data for simulation and testing.
- Visualize risk distribution, probability density, feature importance, and alert volume.
- Explore forensic charts such as network links, Sankey flows, 3D anomaly clusters, and velocity heatmaps.
- Tune risk thresholds dynamically to understand operational impact.

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- Joblib

## Repository Structure

```text
.
|-- aml_dashboard.py
|-- aml_pipeline_production_artifacts.pkl
|-- requirements.txt
`-- README.md
```

## How It Works

1. The dashboard loads serialized ML pipeline artifacts from `aml_pipeline_production_artifacts.pkl`.
2. If the artifact file is missing, the app builds fallback models from generated sample data.
3. Raw transactions are converted into engineered AML features such as:
   - log amount
   - amount risk bins
   - currency mismatch
   - cross-location transfer flag
   - night or weekend transaction flag
   - sender fan-out count
   - Benford first digit
   - statistical outlier signals
   - anomaly model scores
4. The selected classifier predicts a fraud probability for each transaction.
5. Transactions are assigned a risk level and displayed through metrics, tables, and charts.

## Expected CSV Input

Uploaded transaction files should include the following columns:

```text
Time
Date
Sender_account
Receiver_account
Amount
Payment_currency
Received_currency
Sender_bank_location
Receiver_bank_location
Payment_type
```

Optional columns such as `Is_laundering` and `Laundering_type` may exist in training or simulation datasets, but they are not required for dashboard scoring.

## Installation

Clone the repository:

```bash
git clone https://github.com/soudasMukherjee/ANTI-MONEY-LAUNDERING-MACHINE-LEARNING-PIPELINE.git
cd ANTI-MONEY-LAUNDERING-MACHINE-LEARNING-PIPELINE
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Start the Streamlit app:

```bash
streamlit run aml_dashboard.py
```

Then open the local Streamlit URL shown in the terminal.

## Live Demo

The dashboard is deployed here:

https://aml-fraud-detection-pipeline-interface.streamlit.app/



## Usage

### Upload and Analyze

Use this mode to upload a CSV transaction log. The app processes the file, applies feature engineering, scores each transaction, and highlights suspicious activity with probability and risk labels.

### Simulation Mode

Use this mode to generate synthetic AML transaction data. You can control dataset size and fraud percentage, then run the same scoring and visualization pipeline without needing an external CSV file.

## Risk Bands

The project uses probability thresholds to classify risk:

- HIGH: fraud probability greater than or equal to 0.65
- MID: fraud probability from 0.35 to 0.64
- LOW: fraud probability below 0.35

The dashboard also includes sliders for tuning risk thresholds and reviewing how alert volume changes.

## Model Artifacts

The file `aml_pipeline_production_artifacts.pkl` stores the trained pipeline components used by the dashboard, including encoders, scalers, anomaly detectors, and classifier models.

If this file is not available, the app automatically creates fallback models at runtime. For best results, keep the artifact file in the project root.

## Disclaimer

This project is intended for educational, demonstration, and prototype AML analytics use. It should not be used as the only control for real compliance decisions without validation, monitoring, audit review, and approval from qualified risk and compliance teams.
