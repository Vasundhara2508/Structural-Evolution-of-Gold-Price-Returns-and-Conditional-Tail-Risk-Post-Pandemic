from pathlib import Path

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "gold_results"
TABLE_DIR = RESULT_DIR / "tables"
FIG_DIR = RESULT_DIR / "figures"

st.set_page_config(page_title="Gold Tail Risk Dashboard", layout="wide")

@st.cache_data
def load_data():
    timeseries = pd.read_csv(TABLE_DIR / "timeseries_outputs.csv", parse_dates=["Date"])
    regimes = pd.read_csv(TABLE_DIR / "regime_table.csv", parse_dates=["start_date", "end_date"])
    regime_tail = pd.read_csv(TABLE_DIR / "regime_tail_table.csv")
    regime_tail_label = pd.read_csv(TABLE_DIR / "regime_tail_label_summary.csv")
    model_comp = pd.read_csv(TABLE_DIR / "model_comparison.csv")
    regime_model = pd.read_csv(TABLE_DIR / "regime_model_comparison.csv", parse_dates=["start_date", "end_date"])
    counter = pd.read_csv(TABLE_DIR / "counterfactual_prices.csv", parse_dates=["Date"])
    tail_asym = pd.read_csv(TABLE_DIR / "tail_asymmetry_table.csv")
    with open(TABLE_DIR / "analysis_summary.json", "r") as f:
        summary = json.load(f)
    return timeseries, regimes, regime_tail, regime_tail_label, model_comp, regime_model, counter, tail_asym, summary

def bootstrap_counterfactual(counter_df, event_start, event_end, donor_start, donor_end, drift_shift=0.0, vol_scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    df = counter_df[["Date", "Price", "log_return"]].copy()
    donor = df[(df["Date"] >= donor_start) & (df["Date"] <= donor_end)]["log_return"].dropna().to_numpy()
    mask = (df["Date"] >= event_start) & (df["Date"] <= event_end)
    if len(donor) == 0 or mask.sum() == 0:
        df["sim_return"] = df["log_return"]
    else:
        sampled = rng.choice(donor, size=mask.sum(), replace=True)
        donor_mean = donor.mean()
        sampled = donor_mean + (sampled - donor_mean) * vol_scale + drift_shift
        df["sim_return"] = df["log_return"]
        df.loc[mask, "sim_return"] = sampled
    df["sim_price"] = np.nan
    df.loc[df.index[0], "sim_price"] = df.loc[df.index[0], "Price"]
    for i in range(1, len(df)):
        df.loc[df.index[i], "sim_price"] = df.loc[df.index[i-1], "sim_price"] * np.exp(df.loc[df.index[i], "sim_return"])
    return df

timeseries, regimes, regime_tail, regime_tail_label, model_comp, regime_model, counter, tail_asym, summary = load_data()

st.title("Gold Tail Risk Comparison Dashboard")
st.caption("Baseline model: EWMA + Historical VaR | Regime-aware model: Markov Switching + EVT")

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected breaks", len(summary.get("break_dates", [])))
    c2.metric("Best quantile-loss model", summary.get("best_model_by_quantile_loss", "N/A"))
    c3.metric("Pandemic window start", summary.get("pandemic_window_start", "N/A"))
    c4.metric("Pandemic window end", summary.get("pandemic_window_end", "N/A"))

st.markdown("### What this dashboard is saying")
st.write(
    "The dashboard compares a simple baseline against a regime-aware tail model. "
    "The main questions are: when did volatility structure change, which model tracks risk more cleanly, "
    "and how different would the price path look if the pandemic shock window had been replaced by calmer pre-pandemic behavior?"
)

st.markdown("---")
st.markdown("## 1) Regimes and structural changes")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=timeseries["Date"], y=timeseries["Price"], mode="lines", name="Price"))
for _, row in regimes.iterrows():
    fig_price.add_vrect(x0=row["start_date"], x1=row["end_date"], annotation_text=row["label"], annotation_position="top left", opacity=0.08)
fig_price.update_layout(title="Gold price with detected variance regimes", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_price, use_container_width=True)

st.dataframe(regimes, use_container_width=True)
st.info("Interpretation: each shaded block is a period where return variance behaves differently. This becomes the basis for the later model and tail comparisons.")

st.markdown("---")
st.markdown("## 2) Competing models")

st.write("The table below uses easy-to-read metrics instead of dense test statistics.")
st.dataframe(model_comp, use_container_width=True)

st.info(
    "How to read it: breach rate should sit near the target alpha, average breach severity shows how painful misses were, "
    "clustering ratio shows whether misses come in bunches, and quantile loss rewards cleaner VaR estimation."
)

fig_var = go.Figure()
fig_var.add_trace(go.Scatter(x=timeseries["Date"], y=timeseries["log_return"], mode="lines", name="Actual return"))
fig_var.add_trace(go.Scatter(x=timeseries["Date"], y=timeseries["ewma_hs_var"], mode="lines", name="EWMA + Historical VaR"))
fig_var.add_trace(go.Scatter(x=timeseries["Date"], y=timeseries["ms_evt_var"], mode="lines", name="Markov Switching + EVT"))
fig_var.update_layout(title="Actual returns against competing VaR estimates", xaxis_title="Date", yaxis_title="Return / VaR")
st.plotly_chart(fig_var, use_container_width=True)

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=timeseries["Date"], y=timeseries["ewma_vol"], mode="lines", name="EWMA volatility"))
fig_vol.add_trace(go.Scatter(x=timeseries["Date"], y=timeseries["ms_sigma"], mode="lines", name="Markov switching sigma"))
fig_vol.update_layout(title="Volatility signal comparison", xaxis_title="Date", yaxis_title="Sigma")
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("### Regime-level model comparison")
st.dataframe(regime_model, use_container_width=True)

st.markdown("---")
st.markdown("## 3) Tail evolution and asymmetry")

fig_tail = px.bar(regime_tail, x="label", y="shape_xi", color="tail", barmode="group", hover_data=["n_exceedances", "shape_ci_low", "shape_ci_high"], title="Tail index by regime and tail")
st.plotly_chart(fig_tail, use_container_width=True)

st.dataframe(tail_asym, use_container_width=True)
st.info(
    "Interpretation: a larger left-tail shape means crashes are more extreme. A larger right-tail shape means upside spikes are more extreme. "
    "The asymmetry column simply shows right minus left."
)

st.markdown("---")
st.markdown("## 4) What-if simulator: remove a break event")

st.write(
    "This simulator replaces returns inside the selected event window with bootstrap draws from a calmer donor regime. "
    "It does not claim a true causal world; it provides an intuitive counterfactual path for discussion and presentation."
)

event_options = {f"{r['label']} | {r['start_date'].date()} to {r['end_date'].date()}": (r['start_date'], r['end_date']) for _, r in regimes.iterrows()}
donor_options = {f"{r['label']} | {r['start_date'].date()} to {r['end_date'].date()}": (r['start_date'], r['end_date']) for _, r in regimes.iterrows()}

col1, col2, col3, col4 = st.columns(4)
selected_event = col1.selectbox("Event window to neutralize", list(event_options.keys()), index=min(1, len(event_options)-1))
selected_donor = col2.selectbox("Donor regime", list(donor_options.keys()), index=0)
drift_shift = col3.slider("Drift shift applied to donor returns", min_value=-0.01, max_value=0.01, value=0.0, step=0.0005, format="%.4f")
vol_scale = col4.slider("Volatility scale", min_value=0.5, max_value=1.5, value=1.0, step=0.05)

(event_start, event_end) = event_options[selected_event]
(donor_start, donor_end) = donor_options[selected_donor]
sim = bootstrap_counterfactual(counter, event_start, event_end, donor_start, donor_end, drift_shift=drift_shift, vol_scale=vol_scale, seed=42)

fig_cf = go.Figure()
fig_cf.add_trace(go.Scatter(x=sim["Date"], y=sim["Price"], mode="lines", name="Actual price"))
fig_cf.add_trace(go.Scatter(x=sim["Date"], y=sim["sim_price"], mode="lines", name="Counterfactual price"))
fig_cf.add_vrect(x0=event_start, x1=event_end, opacity=0.1, annotation_text="Event window", annotation_position="top left")
fig_cf.update_layout(title="Counterfactual price path after neutralizing the selected event window", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_cf, use_container_width=True)

last_actual = sim["Price"].iloc[-1]
last_cf = sim["sim_price"].iloc[-1]
delta_pct = ((last_cf / last_actual) - 1) * 100
st.metric("End-of-sample counterfactual gap", f"{delta_pct:.2f}%")

preview = sim[["Date", "Price", "sim_price"]].tail(20).copy()
st.dataframe(preview, use_container_width=True)

csv = sim.to_csv(index=False).encode("utf-8")
st.download_button("Download simulated path CSV", data=csv, file_name="simulated_counterfactual_path.csv", mime="text/csv")

st.markdown("---")
st.markdown("## 5) Result files already generated")
file_rows = []
for p in sorted(TABLE_DIR.glob("*.csv")):
    file_rows.append({"type": "table", "name": p.name, "path": str(p)})
for p in sorted(FIG_DIR.glob("*.png")):
    file_rows.append({"type": "figure", "name": p.name, "path": str(p)})
st.dataframe(pd.DataFrame(file_rows), use_container_width=True)
