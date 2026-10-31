"""Streamlit app — Next Month Electricity Bill Prediction (Warangal).
Run:  streamlit run app.py      (after running bill_prediction.ipynb to create bill_model.joblib)"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from bill_utils import add_lag_features, INPUT_COLS, TARGET  # needed to unpickle FeatureEngineer

st.set_page_config(page_title="Next Bill Predictor", page_icon="⚡", layout="wide")


@st.cache_resource
def load_artifact():
    return joblib.load("bill_model.joblib")


@st.cache_data
def load_data():
    return add_lag_features(pd.read_csv("electricity_bills_simple.csv"))


def next_month_label(month: str) -> str:
    return (pd.Period(month, freq="M") + 1).strftime("%b %Y")


art = load_artifact()
model = art["model"]
df = load_data()

st.title("⚡ Next Month Electricity Bill Prediction")
st.caption(f"Location: Warangal · Model: {art['model_name']} · Trained on {art['trained_on']}")

page = st.sidebar.radio("Choose a page", ["Existing customer", "New customer / what-if",
                                          "Batch prediction", "Model performance"])
m = art["test_metrics"]
st.sidebar.markdown("### Test performance")
st.sidebar.metric("MAE (avg error)", f"₹{m['MAE']:.0f}")
st.sidebar.metric("R²", f"{m['R2']:.3f}")
st.sidebar.metric("MAPE", f"{m['MAPE_%']:.1f}%")

# ---------------------------------------------------------------- 1. existing customer
if page == "Existing customer":
    cid = st.selectbox("Customer ID", sorted(df.customer_id.unique()))
    hist = df[df.customer_id == cid].sort_values("month")
    latest = hist.tail(1)
    pred = float(model.predict(latest[INPUT_COLS])[0])
    cur = float(latest.bill_amount.iloc[0])
    month = latest.month.iloc[0]

    info = latest.iloc[0]
    st.write(f"**Area:** {info.area} · **Connection:** {info.connection_type} · "
             f"**House:** {info.house_type} · **Family size:** {info.family_size} · "
             f"**ACs:** {info.num_acs} · **Solar:** {info.has_solar}")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Current bill ({pd.Period(month, freq='M').strftime('%b %Y')})", f"₹{cur:,.0f}")
    c2.metric(f"Predicted bill ({next_month_label(month)})", f"₹{pred:,.0f}",
              delta=f"{pred - cur:+,.0f} ₹", delta_color="inverse")
    c3.metric("Expected range (± test MAE)", f"₹{max(0, pred - m['MAE']):,.0f} – ₹{pred + m['MAE']:,.0f}")

    if pred > 1.25 * cur:
        st.warning("High bill expected next month — consider reducing AC usage.")

    chart = hist.set_index("month")[["bill_amount"]].rename(columns={"bill_amount": "Actual bill"})
    nxt = (pd.Period(month, freq="M") + 1).strftime("%Y-%m")
    chart.loc[nxt, "Predicted bill"] = pred
    chart.loc[month, "Predicted bill"] = cur  # connect the lines
    st.subheader("Bill history and prediction")
    st.line_chart(chart)
    with st.expander("Raw history"):
        st.dataframe(hist.drop(columns=[TARGET]), use_container_width=True)

# ---------------------------------------------------------------- 2. what-if
elif page == "New customer / what-if":
    st.subheader("Enter customer details")
    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        area = c1.selectbox("Area", sorted(df.area.dropna().unique()))
        conn = c1.selectbox("Connection type", ["Domestic", "Commercial"])
        house = c1.selectbox("House type", ["1BHK", "2BHK", "3BHK", "Independent"])
        solar = c1.selectbox("Rooftop solar", ["No", "Yes"])
        family = c2.number_input("Family size", 1, 15, 4)
        acs = c2.number_input("Number of ACs", 0, 6, 1)
        month = c2.selectbox("Current month", pd.period_range("2025-01", "2027-12", freq="M").strftime("%Y-%m"),
                             index=15)
        temp = c2.slider("Avg temperature this month (°C)", 15.0, 42.0, 30.0)
        hum = c3.slider("Humidity (%)", 20.0, 95.0, 55.0)
        units = c3.number_input("Units consumed this month (kWh)", 0, 5000, 250)
        bill = c3.number_input("This month's bill (₹)", 0.0, 50000.0, 1200.0)
        prev1 = c3.number_input("Last month's bill (₹)", 0.0, 50000.0, 1000.0)
        prev2 = c3.number_input("Bill 2 months ago (₹)", 0.0, 50000.0, 900.0)
        go = st.form_submit_button("Predict next month bill")

    if go:
        row = pd.DataFrame([{
            "area": area, "connection_type": conn, "house_type": house, "has_solar": solar,
            "family_size": family, "num_acs": acs, "month": month, "avg_temp_c": temp,
            "humidity_pct": hum, "units_consumed": units, "bill_amount": bill,
            "prev_bill": prev1, "rolling_3m_bill": np.mean([bill, prev1, prev2]),
            "bill_change": bill - prev1}])
        pred = float(model.predict(row[INPUT_COLS])[0])
        c1, c2 = st.columns(2)
        c1.metric(f"Predicted bill for {next_month_label(month)}", f"₹{pred:,.0f}",
                  delta=f"{pred - bill:+,.0f} ₹ vs this month", delta_color="inverse")
        c2.metric("Expected range", f"₹{max(0, pred - m['MAE']):,.0f} – ₹{pred + m['MAE']:,.0f}")

# ---------------------------------------------------------------- 3. batch
elif page == "Batch prediction":
    st.subheader("Upload a CSV of monthly records")
    st.write("Columns needed: `customer_id, month, " + ", ".join(c for c in INPUT_COLS
             if c not in ("month", "prev_bill", "rolling_3m_bill", "bill_change")) +
             "`. History features are computed automatically. Missing values are allowed.")
    st.download_button("Download sample file", df.drop(columns=["prev_bill", "rolling_3m_bill", "bill_change"])
                       .head(50).to_csv(index=False), "sample_input.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = add_lag_features(pd.read_csv(up))
        missing = [c for c in INPUT_COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["predicted_next_bill"] = model.predict(data[INPUT_COLS]).round(2)
            if TARGET in data.columns:
                mae = (data[TARGET] - data.predicted_next_bill).abs().mean()
                st.info(f"File contains actual next-month bills → MAE on this file: ₹{mae:,.0f}")
            st.dataframe(data, use_container_width=True)
            st.download_button("Download predictions", data.to_csv(index=False), "predictions.csv")

# ---------------------------------------------------------------- 4. performance
else:
    st.subheader("Model comparison (validation months Apr–Jun 2026)")
    st.dataframe(art["comparison"], use_container_width=True, hide_index=True)
    st.subheader("Time-series cross-validation (training months)")
    st.dataframe(art["cv_results"], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Actual vs predicted (test, Jul–Sep 2026)")
        st.scatter_chart(art["test_results"], x="actual", y="predicted")
    with c2:
        st.subheader("Feature importance")
        st.caption("How much the error (₹) grows when a feature is shuffled")
        st.bar_chart(art["importance"].sort_values(ascending=False))

    res = art["test_results"].assign(error=lambda d: (d.actual - d.predicted).abs())
    st.subheader("Monitoring: MAE by month")
    st.bar_chart(res.groupby("month").error.mean())