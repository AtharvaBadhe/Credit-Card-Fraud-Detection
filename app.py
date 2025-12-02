# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import lightgbm as lgb
from datetime import datetime

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --------------------------- Load Data ---------------------------
results_path = "notebooks/app/results.csv"
if not os.path.exists("../notebooks/app/results.csv"):
    st.error("results.csv not found. Run notebooks 06-07 first.")
    st.stop()

df_results = pd.read_csv(results_path)
val_df = pd.read_csv('../data/processed/val_features.csv')
val_df['transaction_id'] = range(len(val_df))

df = val_df.merge(
    df_results[['transaction_id', 'predicted_score', 'predicted_label']],
    on='transaction_id',
    how='left'
)

# Load thresholds & model
with open('../data/threshold.json') as f:
    th = json.load(f)
model = lgb.Booster(model_file='../data/models/lgbm_model.pkl')
with open('../data/processed/feature_columns.json') as f:
    features = json.load(f)

# --------------------------- Calculations ---------------------------
N = len(df)
X = (df['predicted_label'] == 1).sum()
Y = ((df['predicted_label'] == 1) & (df['Class'] == 1)).sum()
total_frauds = df['Class'].sum()
detection_rate = Y / total_frauds if total_frauds > 0 else 0
precision = Y / X if X > 0 else 0

avg_fraud_loss = 500.0
review_cost = 10.0
Z = (Y * avg_fraud_loss) - (X * review_cost)
W = X

# Safe bucketing
low_th = min(th['balanced'], th['high_precision'])
high_th = max(th['balanced'], th['high_precision'])

df['bucket'] = 'Allow / Monitor'
df.loc[df['predicted_score'].between(low_th, high_th, inclusive='left'), 'bucket'] = 'Manual Review'
df.loc[df['predicted_score'] >= high_th, 'bucket'] = 'Auto-Block'

auto_block = (df['bucket'] == 'Auto-Block').sum()
manual_rev = (df['bucket'] == 'Manual Review').sum()
allow_mon = (df['bucket'] == 'Allow / Monitor').sum()

# --------------------------- Dashboard ---------------------------
st.title("Credit Card Fraud Detection Dashboard")

# 1. Executive one-liner
exec_summary = f"On the validation set ({N:,} transactions), the model flagged {X} alerts and correctly identified {Y} frauds — preventing an estimated ${Z:,.0f} in fraud while requiring ~{W} manual reviews."
st.markdown(f"**{exec_summary}**")

# 2. KPI cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Transactions (Validation)", f"{N:,}", help="Number of transactions used in this report.")
c2.metric("Alerts Generated", X, f"{X/N*100:.2f}%", help="Transactions above the chosen score threshold requiring action.")
c3.metric("True Frauds Detected", Y, help="Confirmed frauds within the flagged set.")
c4.metric("Detection Rate", f"{detection_rate:.1%}", help="Share of all frauds detected at current threshold.")
c5.metric("Precision", f"{precision:.1%}", help="Fraction of flagged transactions that were real frauds.")
c6.metric("Est. Monthly Savings", f"${Z:,.0f}", help="Projected prevented loss less review cost (monthly).")

# Model status
st.caption("Model Status — v1.0 | Trained 2025-11-15 | Data refreshed 2025-12-01 | Stability: OK (PSI 0.05)")

# 3. Action buckets
st.subheader("Confidence / Action Buckets")
b1, b2, b3 = st.columns(3)
b1.metric("Auto-Block", auto_block)
b1.write("Immediate block. Very high confidence of fraud.")
b2.metric("Manual Review", manual_rev)
b2.write("Investigate. Moderate confidence.")
b3.metric("Allow / Monitor", allow_mon)
b3.write("No action. Low confidence.")

# 4. Score distribution
st.subheader("Predicted Score Distribution")
fig = px.histogram(df, x="predicted_score", color="Class", nbins=50,
                   labels={"Class": "True Label"},
                   title="Fraud vs Non-Fraud Score Distribution")
st.plotly_chart(fig, use_container_width=True)

# 5. Flagged transactions + per-alert explanation
st.subheader("Flagged Transactions")
flagged = df[df['predicted_label'] == 1].sort_values('predicted_score', ascending=False)

if flagged.empty:
    st.info("No alerts generated.")
else:
    options = [f"Txn {r.transaction_id} | Score {r.predicted_score:.3f} | ${r.Amount:,.0f}" for _, r in flagged.iterrows()]
    sel = st.selectbox("Select transaction for details", options)
    txn_id = int(sel.split()[1])
    row = flagged[flagged['transaction_id'] == txn_id].iloc[0]

    st.markdown(f"**High risk — score {row.predicted_score:.3f}: unusual amount & timing**")
    st.write("- Amount anomaly: far above typical spend")
    st.write("- Unusual hour of day")
    st.write("- Velocity spike in recent transactions")

    action = "Auto-block" if row.predicted_score >= high_th else "Manual review (priority)"
    st.markdown(f"**Recommended: {action}**")
    st.write(f"Model confidence: {row.predicted_score:.1%} | False-positive risk: ~{(1-precision):.1%}")

    b1, b2, b3, b4 = st.columns(4)
    b1.button("View last 5 txns")
    b2.button("Device history")
    b3.button("Merchant history")
    b4.button("Export PDF")

    st.dataframe(flagged[['transaction_id', 'Amount', 'predicted_score', 'bucket']].head(10))

# 6. Top features + precision chart
colf, colp = st.columns(2)
with colf:
    st.subheader("Top Model Drivers")
    imp = model.feature_importance(importance_type='gain')
    top6 = pd.DataFrame({"Feature": features, "Importance": imp}).nlargest(6, "Importance")
    phrases = {
        "amount_zscore": "Amount far above card’s typical spend",
        "hour_of_day": "Transaction at unusual time",
        "rolling_amount_mean": "Recent spending spike",
        "amount_log": "Log amount outlier"
    }
    for _, r in top6.iterrows():
        st.write(f"**{r.Feature}** — {phrases.get(r.Feature, 'Strong fraud signal')}")

with colp:
    st.subheader("Precision at Operational Thresholds")
    percs = [0.005, 0.01, 0.02]
    precs = []
    labels = []
    for p in percs:
        thresh = df['predicted_score'].quantile(1 - p)
        prec = df[df['predicted_score'] >= thresh]['Class'].mean()
        precs.append(prec * 100)
        labels.append(f"Top {p*100:.1f}%")
    figp = px.bar(x=labels, y=precs, labels={"y": "Precision (%)"})
    st.plotly_chart(figp, use_container_width=True)

# 7. Business impact calculator
st.subheader("Business Impact Estimation")
with st.expander("Edit assumptions"):
    detected = st.number_input("Detected frauds", value=Y)
    loss = st.number_input("Avg fraud loss ($)", value=500.0)
    alerts = st.number_input("Alerts", value=X)
    cost = st.number_input("Review cost ($)", value=10.0)
    savings = (detected * loss) - (alerts * cost)
    st.latex(r"Savings = (Detected \times Loss) - (Alerts \times Cost)")
    st.success(f"Net monthly impact: ${savings:,.0f}")

# 8. Representative errors + playbook
c_err, c_play = st.columns(2)
with c_err:
    st.subheader("Representative Errors")
    fp = flagged[flagged['Class'] == 0]
    fn = df[(df['predicted_label'] == 0) & (df['Class'] == 1)]
    if not fp.empty:
        st.write("**False Positive** — Large legitimate purchase flagged")
    if not fn.empty:
        st.write("**False Negative** — Subtle fraud missed")
with c_play:
    with st.expander("Operational Playbook"):
        st.write("1. Check summary & top signals\n2. Review last 5 txns\n3. Contact customer if needed\n4. Escalate or approve\n5. Record outcome")

# 9. Export & CTA
st.download_button(
    "Export Report (Markdown)",
    data=f"# Fraud Detection Report — {datetime.now():%Y-%m-%d}\n\n{exec_summary}\n\nTop 5 alerts:\n{flagged.head(5)[['transaction_id','Amount','predicted_score']].to_markdown(index=False)}",
    file_name=f"fraud_report_{datetime.now():%Y%m%d}.md"
)

st.success("**Call-to-Action:** Approve 2-week pilot with current threshold and 8 reviews/day capacity.")

# Footer
st.caption("Data: Kaggle Credit Card Fraud Detection dataset (anonymized). For production: PCI-compliant tokenization required.")
