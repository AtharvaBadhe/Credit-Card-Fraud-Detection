# Credit Card Fraud Detection – End-to-End System

A complete, production-ready fraud detection solution built on the benchmark Kaggle Credit Card Fraud Detection dataset (284,807 transactions, 492 fraudulent). The system delivers high detection rates on severely imbalanced data while keeping alert volume low enough for practical operational use.

## Business Impact

- Captures approximately 84% of all fraudulent transactions on the validation set  
- Generates alerts on only approximately 0.28% of total volume (approximately 160 alerts from 57,000 transactions)  
- Under conservative assumptions ($500 average fraud loss, $10 per manual review), the model prevents far more loss than the cost of investigation, yielding an estimated net monthly savings of over $35,000  
- Provides explicit action recommendations (Auto-Block, Manual Review, Allow/Monitor) that map directly to existing risk and payment workflows  
- Includes an interactive business impact calculator enabling stakeholders to adjust loss and review-cost assumptions and immediately see ROI changes

## Solution Overview

- Automated data ingestion and preprocessing  
- Feature engineering on PCA-transformed variables: log amount, hour-of-day, amount z-score, rolling velocity statistics  
- LightGBM classifier achieving >0.95 ROC-AUC and >0.90 PR-AUC  
- Precision-recall threshold optimization tailored to operational review capacity  
- Executive Streamlit dashboard presenting:  
  – One-line performance summary  
  – Key business KPIs  
  – Score distribution and action buckets  
  – Plain-language per-transaction explanations  
  – Top model drivers  
  – Editable savings calculator and one-click report export

## Live Demo

https://credit-card-fraud-detection-fbew7dzyppi8tqyjdjh5eh.streamlit.app/

## Quick Start

```bash
git clone https://github.com/AtharvaBadhe/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
python data_download.py        # downloads the dataset
streamlit run app/app.py       # launches the dashboard
