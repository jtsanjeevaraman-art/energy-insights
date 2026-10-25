import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Campus Energy Insights", layout="wide")

TARIFF = 8.0      # ₹ per kWh (change to your campus rate)
EMISSION = 0.82   # kg CO2 per kWh (approx. Indian grid)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------- 1. DATA ----------
@st.cache_data
def sample_data():
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-08-01", periods=24 * 60, freq="h")
    base = {"Library": 40, "Hostel A": 60, "Admin Block": 35, "Lab Block": 80}
    frames = []
    for name, b in base.items():
        day = ((ts.hour >= 8) & (ts.hour <= 18)).astype(float)
        weekday = (ts.dayofweek < 5).astype(float)
        kwh = b * (0.35 + 0.65 * day * (0.5 + 0.5 * weekday))
        kwh = kwh + rng.normal(0, b * 0.04, len(ts))
        if name == "Lab Block":  # simulate equipment left on at night
            night = (ts.hour >= 22) | (ts.hour <= 4)
            kwh = kwh + night * b * 0.5 * (rng.random(len(ts)) > 0.4)
        spikes = rng.choice(len(ts), 8, replace=False)
        kwh[spikes] *= 2.5
        frames.append(pd.DataFrame({"timestamp": ts, "building": name,
                                    "kwh": kwh.clip(min=0)}))
    return pd.concat(frames, ignore_index=True)


st.sidebar.header("Data")
file = st.sidebar.file_uploader("Upload your CSV", type="csv")
if file:
    raw = pd.read_csv(file)
    cols = list(raw.columns)
    c_time = st.sidebar.selectbox("Time column", cols, index=0)
    c_bld = st.sidebar.selectbox("Building column", cols, index=min(1, len(cols) - 1))
    c_kwh = st.sidebar.selectbox("Energy (kWh) column", cols, index=min(2, len(cols) - 1))
    df = raw[[c_time, c_bld, c_kwh]].copy()
    df.columns = ["timestamp", "building", "kwh"]
else:
    df = sample_data()
    st.sidebar.info("Showing sample data. Upload the hackathon CSV to use real data.")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
df = df.dropna(subset=["timestamp", "building", "kwh"])
df["hour"] = df.timestamp.dt.hour
df["dow"] = df.timestamp.dt.dayofweek
df["date"] = df.timestamp.dt.date

# ---------- 2. FILTERS ----------
buildings = sorted(df["building"].unique())
sel = st.sidebar.multiselect("Buildings", buildings, default=buildings)
d0, d1 = df.date.min(), df.date.max()
picked = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
if len(picked) == 2:
    df = df[(df.date >= picked[0]) & (df.date <= picked[1])]
df = df[df.building.isin(sel)]
if df.empty:
    st.warning("No data for this selection.")
    st.stop()

# ---------- 3. ANOMALY SCORE (z-score vs normal for that building + hour) ----------
thr = st.sidebar.slider("Anomaly sensitivity (z-score)", 2.0, 5.0, 3.0, 0.5)
grp = df.groupby(["building", "hour"])["kwh"]
df["normal"] = grp.transform("mean")
df["z"] = (df["kwh"] - df["normal"]) / grp.transform("std")
df["anomaly"] = df["z"] > thr

# ---------- 4. HEADER + KPIs ----------
st.title("⚡ Campus Energy Insights")
total = df.kwh.sum()
n_days = max(df.date.nunique(), 1)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total energy", f"{total:,.0f} kWh")
k2.metric("Estimated cost", f"₹{total * TARIFF:,.0f}")
k3.metric("CO₂ emitted", f"{total * EMISSION / 1000:,.1f} t")
k4.metric("Anomalies found", int(df.anomaly.sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Anomalies", "Forecast", "Recommendations"])

# ---------- OVERVIEW ----------
with tab1:
    daily = df.groupby(["date", "building"], as_index=False).kwh.sum()
    st.plotly_chart(px.line(daily, x="date", y="kwh", color="building",
                            title="Daily consumption (kWh)"))
    left, right = st.columns(2)
    by_b = df.groupby("building", as_index=False).kwh.sum()
    left.plotly_chart(px.bar(by_b, x="building", y="kwh", color="building",
                             title="Total by building"))
    heat = (df.groupby(["dow", "hour"], as_index=False).kwh.mean()
              .pivot(index="dow", columns="hour", values="kwh").reindex(range(7)))
    heat.index = DAYS
    right.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                                 labels=dict(x="Hour of day", y="Day", color="Avg kWh"),
                                 title="When is energy used? (hour × weekday)"))

# ---------- ANOMALIES ----------
with tab2:
    b = st.selectbox("Building", sel, key="anom_b")
    g = df[df.building == b].sort_values("timestamp")
    fig = px.line(g, x="timestamp", y="kwh", title=f"{b}: usage with anomalies marked")
    bad = g[g.anomaly]
    fig.add_trace(go.Scatter(x=bad.timestamp, y=bad.kwh, mode="markers",
                             marker=dict(color="red", size=9), name="Anomaly"))
    st.plotly_chart(fig)
    out = df[df.anomaly].copy()
    out["why flagged"] = out.apply(
        lambda r: f"{r.kwh:.0f} kWh at {r.hour}:00 vs normal {r.normal:.0f} kWh", axis=1)
    out["extra kWh"] = (out.kwh - out.normal).round(1)
    st.subheader("Top anomalies")
    st.dataframe(out.sort_values("extra kWh", ascending=False)
                    [["timestamp", "building", "why flagged", "extra kWh"]].head(15))

# ---------- FORECAST ----------
def forecast(g, horizon=168):
    s = g.set_index("timestamp")["kwh"].sort_index().resample("h").sum()
    train, test = s[:-horizon], s[-horizon:]
    prof = train.groupby([train.index.dayofweek, train.index.hour]).mean()
    pred = pd.Series([prof.get((t.dayofweek, t.hour), np.nan) for t in test.index],
                     index=test.index)
    ok = test > 0
    mape = (abs(test[ok] - pred[ok]) / test[ok]).mean() * 100
    full = s.groupby([s.index.dayofweek, s.index.hour]).mean()
    fidx = pd.date_range(s.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    fut = pd.Series([full.get((t.dayofweek, t.hour), np.nan) for t in fidx], index=fidx)
    return s, fut, mape

with tab3:
    fb = st.selectbox("Building", sel, key="fore_b")
    hist, fut, mape = forecast(df[df.building == fb])
    if len(hist) < 24 * 14:
        st.info("Need at least 14 days of data to forecast.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=hist.index[-336:], y=hist.values[-336:], name="Actual")
        fig.add_scatter(x=fut.index, y=fut.values, name="Forecast (next 7 days)",
                        line=dict(dash="dash"))
        fig.update_layout(title=f"{fb}: next 7 days demand (kWh per hour)")
        st.plotly_chart(fig)
        c1, c2 = st.columns(2)
        c1.metric("Forecast next 7 days", f"{fut.sum():,.0f} kWh")
        c2.metric("Backtest error (MAPE)", f"{mape:.1f}%")
        st.caption("Method: average usage for each weekday and hour. "
                   "Error is measured by predicting the last 7 days of known data.")

# ---------- RECOMMENDATIONS ----------
with tab4:
    st.subheader("What to do to cut wastage")
    month_saving_kwh = 0
    for b, gb in df.groupby("building"):
        night = gb[(gb.hour >= 22) | (gb.hour <= 5)]
        day = gb[(gb.hour >= 9) & (gb.hour <= 17)]
        if len(night) and len(day):
            ratio = night.kwh.mean() / day.kwh.mean()
            if ratio > 0.6:
                excess = max(night.kwh.mean() - 0.4 * day.kwh.mean(), 0) * len(night)
                month_saving_kwh += excess * 30 / n_days
                st.warning(f"**{b}**: night usage is {ratio:.0%} of daytime usage. "
                           f"Install auto shut-off for 10 PM to 6 AM. "
                           f"Possible saving: {excess:,.0f} kWh in this period.")
        wk = gb[gb.dow >= 5].kwh.mean()
        wd = gb[gb.dow < 5].kwh.mean()
        if wd > 0 and wk / wd > 0.85:
            st.warning(f"**{b}**: weekend usage is {wk / wd:.0%} of weekday usage. "
                       f"Switch off HVAC and lights on weekends.")
        n_bad = int(gb.anomaly.sum())
        if n_bad > 5:
            st.info(f"**{b}**: {n_bad} abnormal spikes detected. "
                    f"Check equipment on the Anomalies tab.")
    st.success(f"Estimated monthly saving if night shut-offs are applied: "
               f"{month_saving_kwh:,.0f} kWh, ₹{month_saving_kwh * TARIFF:,.0f}, "
               f"{month_saving_kwh * EMISSION:,.0f} kg CO₂")
               