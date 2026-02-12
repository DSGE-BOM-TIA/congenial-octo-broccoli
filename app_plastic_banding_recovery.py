# Save as: app_plastic_banding_recovery.py
# Optional image in repo root: hero_infographic.png (lowercase) for background.
# Requires: streamlit, pandas, numpy, matplotlib

from __future__ import annotations

import base64
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def money(x: float) -> str:
    try:
        return "${:,.0f}".format(float(x))
    except Exception:
        return str(x)

def money2(x: float) -> str:
    try:
        return "${:,.2f}".format(float(x))
    except Exception:
        return str(x)

def pct(x: float) -> str:
    try:
        return "{:.2%}".format(float(x))
    except Exception:
        return str(x)

def load_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def traffic_light(value: float, target: float, warn_band: float = 0.03) -> str:
    # Green if meeting target, yellow if close, red otherwise
    if value >= target:
        return "🟢"
    if value >= target - warn_band:
        return "🟡"
    return "🔴"

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)

# -------------------------
# Page + Theme (bold / readable)
# -------------------------
st.set_page_config(page_title="Plastic Banding Recovery Hero", page_icon="♻️", layout="wide")

hero_b64 = ""
try:
    hero_b64 = load_image_as_base64("hero_infographic.png")
except Exception:
    hero_b64 = ""

st.markdown(f"""
<style>
.stApp {{
  background:
    linear-gradient(rgba(6,10,18,0.97), rgba(6,10,18,0.98)),
    url('data:image/png;base64,{hero_b64}') no-repeat top center fixed;
  background-size: cover;
}}
html, body, [class*="css"] {{
  color: #FFFFFF !important;
  font-size: 18px !important;
}}
h1 {{ font-size: 46px !important; font-weight: 900 !important; }}
h2 {{ font-size: 32px !important; font-weight: 850 !important; }}
h3 {{ font-size: 24px !important; font-weight: 800 !important; }}

.hero-card {{
  padding: 22px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.22);
  background: rgba(0,0,0,0.55);
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.40);
}}
.pill {{
  display:inline-block; padding: 7px 12px; border-radius: 999px;
  background: rgba(0,229,255,0.14);
  border: 1px solid rgba(0,229,255,0.35);
  font-weight: 900; font-size: 13px;
}}
[data-testid="stMetricValue"] {{
  font-size: 36px !important;
  font-weight: 900 !important;
  color: #00E5FF !important;
}}
[data-testid="stMetricLabel"] {{
  font-size: 18px !important;
  font-weight: 850 !important;
}}
section[data-testid="stSidebar"] {{
  background-color: rgba(10,16,30,0.98) !important;
  border-right: 1px solid rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] * {{
  font-size: 17px !important;
  color: #FFFFFF !important;
}}
button[data-baseweb="tab"] {{
  font-size: 18px !important;
  font-weight: 850 !important;
  color: #FFFFFF !important;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Scenario model
# -------------------------
SCENARIOS = ["BEST", "LIKELY", "WORST"]

def scenario_values(best: float, likely: float, worst: float) -> Dict[str, float]:
    # enforce sensible ordering
    vals = sorted([float(best), float(likely), float(worst)])
    return {"BEST": vals[0], "LIKELY": vals[1], "WORST": vals[2]}

def compute_one_scenario(
    W_lbs: float,
    recovery_rate: float,
    yield_rate: float,
    downtime_rate: float,
    contamination_scrap_rate: float,
    price_sell_per_lb: float,
    internal_value_per_lb: float,
    mix_sell_share: float,
    disposal_avoid_per_lb: float,
    carbon_value_per_lb: float,
    op_cost_monthly: float,
) -> Dict[str, float]:
    """
    W_lbs: incoming strap waste lbs/month
    recovery_rate: sorting capture rate (0-1)
    yield_rate: conversion yield after process (0-1)
    downtime_rate: lost production fraction (0-1)
    contamination_scrap_rate: additional scrap fraction after recovery (0-1)
    mix_sell_share: share of finished output that is sold (0-1); remainder is internal reuse value
    """

    rr = clamp(recovery_rate, 0.0, 1.0)
    yr = clamp(yield_rate, 0.0, 1.0)
    dr = clamp(downtime_rate, 0.0, 1.0)
    sr = clamp(contamination_scrap_rate, 0.0, 1.0)
    sell_share = clamp(mix_sell_share, 0.0, 1.0)

    # Recovered material after collection + sorting
    recovered_lbs = W_lbs * rr

    # Scrap loss due to contamination after recovered
    usable_feed_lbs = recovered_lbs * (1.0 - sr)

    # Finished filament/pellet output after yield and downtime impact
    finished_lbs = usable_feed_lbs * yr * (1.0 - dr)

    # Value streams
    sold_lbs = finished_lbs * sell_share
    internal_lbs = finished_lbs * (1.0 - sell_share)

    revenue_sell = sold_lbs * price_sell_per_lb
    value_internal = internal_lbs * internal_value_per_lb

    # Disposal avoidance: treat as avoided cost per recovered pound (or per incoming pound if you prefer)
    disposal_avoid = recovered_lbs * disposal_avoid_per_lb

    # Carbon/ESG value per recovered pound (simple proxy)
    carbon_value = recovered_lbs * carbon_value_per_lb

    gross_value = revenue_sell + value_internal + disposal_avoid + carbon_value
    net_profit = gross_value - op_cost_monthly

    return {
        "Incoming waste (lbs/mo)": W_lbs,
        "Recovered (lbs/mo)": recovered_lbs,
        "Finished output (lbs/mo)": finished_lbs,
        "Sold output (lbs/mo)": sold_lbs,
        "Internal reuse (lbs/mo)": internal_lbs,
        "Sell revenue ($/mo)": revenue_sell,
        "Internal value ($/mo)": value_internal,
        "Disposal avoided ($/mo)": disposal_avoid,
        "Carbon/ESG value ($/mo)": carbon_value,
        "Gross value ($/mo)": gross_value,
        "Operating cost ($/mo)": op_cost_monthly,
        "Net profit ($/mo)": net_profit,
    }

def compute_all_scenarios(
    sites: float,
    W_site: Dict[str, float],
    recovery: Dict[str, float],
    yield_rate: Dict[str, float],
    downtime: Dict[str, float],
    scrap: Dict[str, float],
    sell_price: Dict[str, float],
    internal_value: Dict[str, float],
    sell_share: Dict[str, float],
    disposal_avoid: Dict[str, float],
    carbon_value: Dict[str, float],
    op_cost: Dict[str, float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for s in SCENARIOS:
        result = compute_one_scenario(
            W_lbs=W_site[s] * sites,
            recovery_rate=recovery[s],
            yield_rate=yield_rate[s],
            downtime_rate=downtime[s],
            contamination_scrap_rate=scrap[s],
            price_sell_per_lb=sell_price[s],
            internal_value_per_lb=internal_value[s],
            mix_sell_share=sell_share[s],
            disposal_avoid_per_lb=disposal_avoid[s],
            carbon_value_per_lb=carbon_value[s],
            op_cost_monthly=op_cost[s] * sites,   # assume op cost scales with sites
        )
        result["Scenario"] = s
        rows.append(result)
    df = pd.DataFrame(rows)
    # arrange columns
    cols = ["Scenario"] + [c for c in df.columns if c != "Scenario"]
    return df[cols]

# -------------------------
# Sidebar Inputs (ALL FLOATS)
# -------------------------
st.sidebar.markdown("## Inputs")

st.sidebar.markdown("### 🏢 Scale")
sites = st.sidebar.number_input("Number of fulfillment sites", min_value=1.0, max_value=5000.0, value=1.0, step=1.0)
st.sidebar.caption("All results scale by site count. Start with 1 site for a pilot, then increase to model rollout.")

st.sidebar.markdown("### ♻️ Waste Inflow (lbs/month per site)")
W_vals = scenario_values(
    st.sidebar.number_input("Best: strap waste lbs/mo per site", 0.0, 10_000_000.0, 2000.0, 100.0),
    st.sidebar.number_input("Likely: strap waste lbs/mo per site", 0.0, 10_000_000.0, 3000.0, 100.0),
    st.sidebar.number_input("Worst: strap waste lbs/mo per site", 0.0, 10_000_000.0, 4500.0, 100.0),
)

st.sidebar.markdown("### 🧪 Process Performance (rates)")
recovery_vals = scenario_values(
    st.sidebar.number_input("Recovery rate Best (0-1)", 0.0, 1.0, 0.95, 0.01),
    st.sidebar.number_input("Recovery rate Likely (0-1)", 0.0, 1.0, 0.90, 0.01),
    st.sidebar.number_input("Recovery rate Worst (0-1)", 0.0, 1.0, 0.80, 0.01),
)

yield_vals = scenario_values(
    st.sidebar.number_input("Yield Best (0-1)", 0.0, 1.0, 0.97, 0.01),
    st.sidebar.number_input("Yield Likely (0-1)", 0.0, 1.0, 0.93, 0.01),
    st.sidebar.number_input("Yield Worst (0-1)", 0.0, 1.0, 0.85, 0.01),
)

downtime_vals = scenario_values(
    st.sidebar.number_input("Downtime Best (0-1)", 0.0, 1.0, 0.03, 0.01),
    st.sidebar.number_input("Downtime Likely (0-1)", 0.0, 1.0, 0.07, 0.01),
    st.sidebar.number_input("Downtime Worst (0-1)", 0.0, 1.0, 0.15, 0.01),
)

scrap_vals = scenario_values(
    st.sidebar.number_input("Contamination scrap Best (0-1)", 0.0, 1.0, 0.02, 0.01),
    st.sidebar.number_input("Contamination scrap Likely (0-1)", 0.0, 1.0, 0.05, 0.01),
    st.sidebar.number_input("Contamination scrap Worst (0-1)", 0.0, 1.0, 0.12, 0.01),
)

st.sidebar.markdown("### 💵 Value per lb (Sell + Internal Reuse)")
sell_price_vals = scenario_values(
    st.sidebar.number_input("Sell price Best ($/lb)", 0.0, 1000.0, 14.0, 0.5),
    st.sidebar.number_input("Sell price Likely ($/lb)", 0.0, 1000.0, 10.0, 0.5),
    st.sidebar.number_input("Sell price Worst ($/lb)", 0.0, 1000.0, 7.0, 0.5),
)

internal_value_vals = scenario_values(
    st.sidebar.number_input("Internal reuse value Best ($/lb)", 0.0, 1000.0, 18.0, 0.5),
    st.sidebar.number_input("Internal reuse value Likely ($/lb)", 0.0, 1000.0, 12.0, 0.5),
    st.sidebar.number_input("Internal reuse value Worst ($/lb)", 0.0, 1000.0, 8.0, 0.5),
)
st.sidebar.caption("Internal reuse value = avoided purchase cost of filament/pellets you’d otherwise buy.")

st.sidebar.markdown("### 🔀 Sell vs Internal Mix")
sell_share_vals = scenario_values(
    st.sidebar.number_input("Sell share Best (0-1)", 0.0, 1.0, 0.70, 0.05),
    st.sidebar.number_input("Sell share Likely (0-1)", 0.0, 1.0, 0.50, 0.05),
    st.sidebar.number_input("Sell share Worst (0-1)", 0.0, 1.0, 0.30, 0.05),
)

st.sidebar.markdown("### 🗑 Disposal Avoidance + 🌍 Carbon/ESG value")
disposal_avoid_vals = scenario_values(
    st.sidebar.number_input("Disposal avoided Best ($/lb recovered)", 0.0, 100.0, 0.20, 0.05),
    st.sidebar.number_input("Disposal avoided Likely ($/lb recovered)", 0.0, 100.0, 0.15, 0.05),
    st.sidebar.number_input("Disposal avoided Worst ($/lb recovered)", 0.0, 100.0, 0.10, 0.05),
)
carbon_value_vals = scenario_values(
    st.sidebar.number_input("Carbon/ESG value Best ($/lb recovered)", 0.0, 100.0, 0.12, 0.02),
    st.sidebar.number_input("Carbon/ESG value Likely ($/lb recovered)", 0.0, 100.0, 0.06, 0.02),
    st.sidebar.number_input("Carbon/ESG value Worst ($/lb recovered)", 0.0, 100.0, 0.02, 0.01),
)
st.sidebar.caption("Carbon/ESG value can represent carbon credit, PR value, compliance, or internal ESG scorecard valuation.")

st.sidebar.markdown("### 🧰 Operating Cost (per site / month)")
op_cost_vals = scenario_values(
    st.sidebar.number_input("Op cost Best ($/mo per site)", 0.0, 10_000_000.0, 4500.0, 250.0),
    st.sidebar.number_input("Op cost Likely ($/mo per site)", 0.0, 10_000_000.0, 6000.0, 250.0),
    st.sidebar.number_input("Op cost Worst ($/mo per site)", 0.0, 10_000_000.0, 8000.0, 250.0),
)

st.sidebar.markdown("## Capital + Payback")
capex = st.sidebar.number_input("One-time CapEx (pilot or per-site) $", min_value=0.0, max_value=500_000_000.0, value=35_000.0, step=1000.0)
capex_scope = st.sidebar.radio("CapEx is:", ["Pilot total (not per-site)", "Per-site"], horizontal=False)

# -------------------------
# Compute
# -------------------------
df = compute_all_scenarios(
    sites=sites,
    W_site=W_vals,
    recovery=recovery_vals,
    yield_rate=yield_vals,
    downtime=downtime_vals,
    scrap=scrap_vals,
    sell_price=sell_price_vals,
    internal_value=internal_value_vals,
    sell_share=sell_share_vals,
    disposal_avoid=disposal_avoid_vals,
    carbon_value=carbon_value_vals,
    op_cost=op_cost_vals,
)

# Derive totals + payback
def get(df: pd.DataFrame, scenario: str, col: str) -> float:
    return float(df.loc[df["Scenario"] == scenario, col].values[0])

best_profit = get(df, "BEST", "Net profit ($/mo)")
likely_profit = get(df, "LIKELY", "Net profit ($/mo)")
worst_profit = get(df, "WORST", "Net profit ($/mo)")

best_gross = get(df, "BEST", "Gross value ($/mo)")
likely_gross = get(df, "LIKELY", "Gross value ($/mo)")
worst_gross = get(df, "WORST", "Gross value ($/mo)")

# Capex scaling
if capex_scope == "Per-site":
    total_capex = capex * sites
else:
    total_capex = capex

def payback_months(capex_total: float, monthly_profit: float) -> float:
    if monthly_profit <= 0:
        return float("inf")
    return capex_total / monthly_profit

pb_best = payback_months(total_capex, best_profit)
pb_likely = payback_months(total_capex, likely_profit)
pb_worst = payback_months(total_capex, worst_profit)

# Quality-style targets (simple executive targets)
target_recovery = 0.90
target_yield = 0.92
target_downtime_max = 0.08
target_scrap_max = 0.07

# Likely indicators
rec_l = recovery_vals["LIKELY"]
yld_l = yield_vals["LIKELY"]
dwn_l = downtime_vals["LIKELY"]
scp_l = scrap_vals["LIKELY"]

# -------------------------
# HERO HEADER
# -------------------------
st.markdown("""
<div class="hero-card">
  <div class="pill">Circular Economy • Fulfillment Centers • Boardroom-ready</div>
  <h1 style="margin:10px 0 6px 0;">Plastic Banding Recovery Hero Dashboard</h1>
  <div style="opacity:0.9;">
    Convert <b>plastic strap waste</b> into <b>filament/pellets</b> + <b>internal reuse</b>,
    quantify <b>best/likely/worst value</b>, and build a <b>ROI/payback case</b> stakeholders can approve fast.
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Math + Definitions (easy)
# -------------------------
with st.expander("📘 Definitions + Math (simple, executive-friendly)", expanded=True):
    st.markdown("### What this dashboard does")
    st.write("It turns monthly plastic strap waste into a financial and ESG value case under Best / Likely / Worst assumptions.")
    st.markdown("### Core variables (plain language)")
    st.markdown("""
- **W** = strap waste in pounds per month  
- **R** = recovery rate (how much you successfully capture and sort)  
- **S** = contamination scrap rate (how much recovered material you must throw away)  
- **Y** = processing yield (how efficiently you turn clean material into filament/pellets)  
- **D** = downtime rate (lost production time)  
- **P** = sell price per lb  
- **V** = internal reuse value per lb (avoided purchase cost)  
- **Mix** = % sold vs % used internally  
- **OpCost** = monthly operating cost  
- **CapEx** = one-time equipment setup cost  
""")
    st.markdown("### Core formulas")
    st.latex(r"W_r = W \cdot R")
    st.latex(r"W_{usable} = W_r \cdot (1 - S)")
    st.latex(r"F = W_{usable} \cdot Y \cdot (1 - D)")
    st.latex(r"Revenue = (F \cdot Mix) \cdot P")
    st.latex(r"InternalValue = (F \cdot (1-Mix)) \cdot V")
    st.latex(r"GrossValue = Revenue + InternalValue + DisposalAvoid + CarbonValue")
    st.latex(r"Profit = GrossValue - OpCost")
    st.latex(r"PaybackMonths = CapEx / Profit")
    st.caption("This is a simple decision model. It’s designed for fast stakeholder alignment and clear ROI storytelling.")

# -------------------------
# KPI TOP ROW (Best/Likely/Worst)
# -------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Net Profit (Best / mo)", money(best_profit))
k2.metric("Net Profit (Likely / mo)", money(likely_profit))
k3.metric("Net Profit (Worst / mo)", money(worst_profit))
k4.metric("CapEx (total)", money(total_capex))

k5, k6, k7, k8 = st.columns(4)
k5.metric("Payback (Best)", "∞" if not math.isfinite(pb_best) else f"{pb_best:.1f} mo")
k6.metric("Payback (Likely)", "∞" if not math.isfinite(pb_likely) else f"{pb_likely:.1f} mo")
k7.metric("Payback (Worst)", "∞" if not math.isfinite(pb_worst) else f"{pb_worst:.1f} mo")
k8.metric("Sites modeled", f"{sites:,.0f}")

st.divider()

# -------------------------
# TRAFFIC LIGHTS (process health)
# -------------------------
st.markdown("## Process Health (Likely scenario indicators)")

t1 = traffic_light(rec_l, target_recovery, warn_band=0.05)
t2 = traffic_light(yld_l, target_yield, warn_band=0.05)
# For downtime/scrap, lower is better: invert logic by comparing (1 - rate) to (1 - target_max)
t3 = traffic_light(1.0 - dwn_l, 1.0 - target_downtime_max, warn_band=0.05)
t4 = traffic_light(1.0 - scp_l, 1.0 - target_scrap_max, warn_band=0.05)

p1, p2, p3, p4 = st.columns(4)
p1.metric(f"{t1} Recovery rate", pct(rec_l))
p2.metric(f"{t2} Yield rate", pct(yld_l))
p3.metric(f"{t3} Downtime (lower is better)", pct(dwn_l))
p4.metric(f"{t4} Contamination scrap (lower is better)", pct(scp_l))

st.caption("Targets are simple defaults. You can tune them to your organization’s standards anytime.")

st.divider()

# -------------------------
# NARRATED MODE (CFO / Ops / Sustainability)
# -------------------------
st.markdown("## Narrated Mode (easy to explain in a meeting)")
aud = st.radio("Choose your audience lens:", ["CFO", "Operations", "Sustainability / ESG"], horizontal=True)

likely_row = df[df["Scenario"] == "LIKELY"].iloc[0].to_dict()

if aud == "CFO":
    st.markdown(f"""
**What this means financially (Likely scenario)**  
- Gross value: **{money(likely_gross)}/month**  
- Net profit: **{money(likely_profit)}/month**  
- Payback: **{"∞" if not math.isfinite(pb_likely) else f"{pb_likely:.1f} months"}**  
- Biggest levers: **sell price**, **recovery rate**, **yield**, and **downtime**.

**Decision question**  
> Do we approve CapEx to convert waste into margin + internal supply security?
""")

elif aud == "Operations":
    st.markdown(f"""
**How operations wins (Likely scenario)**  
- Incoming waste processed: **{likely_row['Incoming waste (lbs/mo)']:,.0f} lbs/mo**  
- Recovered: **{likely_row['Recovered (lbs/mo)']:,.0f} lbs/mo**  
- Finished output: **{likely_row['Finished output (lbs/mo)']:,.0f} lbs/mo**

**What to control first**  
1) Sorting discipline (improves Recovery)  
2) Moisture + contamination controls (reduces Scrap)  
3) Stable extrusion settings + QA checks (improves Yield)  
4) Preventive maintenance (reduces Downtime)

**Execution question**  
> What single standard work change lifts recovery + yield within 30–60 days?
""")

else:
    st.markdown(f"""
**What ESG gets (Likely scenario)**  
- Waste diverted (recovered): **{likely_row['Recovered (lbs/mo)']:,.0f} lbs/month**  
- Carbon/ESG value proxy: **{money(likely_row['Carbon/ESG value ($/mo)'])}/month**  
- Story: **turn a universal waste stream into a circular product.**

**Narrative stakeholders understand**  
> “We convert fulfillment center plastic waste into useful manufacturing feedstock and reduce disposal.”
""")

st.divider()

# -------------------------
# Tabs: Breakdown / Visuals / Sensitivity / Export
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["1) Breakdown", "2) Visuals", "3) Sensitivity (Likely)", "4) Export"])

with tab1:
    st.subheader("Best / Likely / Worst Scenario Table (scaled by site count)")
    st.dataframe(df, use_container_width=True)

    st.subheader("Where the value comes from (Likely)")
    cols = ["Sell revenue ($/mo)", "Internal value ($/mo)", "Disposal avoided ($/mo)", "Carbon/ESG value ($/mo)"]
    value_break = pd.DataFrame([{"Component": c, "Value ($/mo)": float(likely_row[c])} for c in cols]).sort_values("Value ($/mo)", ascending=False)
    st.dataframe(value_break, use_container_width=True)

with tab2:
    st.subheader("Net Profit band (Best → Likely → Worst)")
    fig = plt.figure()
    plt.bar(["Best", "Likely", "Worst"], [best_profit, likely_profit, worst_profit])
    plt.ylabel("Net Profit ($ / month)")
    st.pyplot(fig)

    st.subheader("Payback band (months)")
    fig2 = plt.figure()
    y = [
        pb_best if math.isfinite(pb_best) else 0.0,
        pb_likely if math.isfinite(pb_likely) else 0.0,
        pb_worst if math.isfinite(pb_worst) else 0.0
    ]
    plt.bar(["Best", "Likely", "Worst"], y)
    plt.ylabel("Payback (months) — 0 shown if infinite")
    st.pyplot(fig2)

    st.subheader("Likely: value components")
    fig3 = plt.figure()
    plt.bar(value_break["Component"], value_break["Value ($/mo)"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("$ / month")
    st.pyplot(fig3)

with tab3:
    st.subheader("Sensitivity (Likely) — move the levers live")
    st.caption("These sliders apply quick +/- changes to the Likely scenario to show which lever matters most.")

    # Base likely values
    base = df[df["Scenario"] == "LIKELY"].iloc[0].to_dict()
    base_profit = float(base["Net profit ($/mo)"])
    base_gross = float(base["Gross value ($/mo)"])

    # Sensitivity knobs (as multipliers or deltas)
    price_mult = st.slider("Sell price multiplier", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
    internal_mult = st.slider("Internal reuse value multiplier", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
    recovery_delta = st.slider("Recovery rate delta (+/-)", min_value=-0.20, max_value=0.20, value=0.0, step=0.01)
    yield_delta = st.slider("Yield delta (+/-)", min_value=-0.20, max_value=0.20, value=0.0, step=0.01)
    downtime_delta = st.slider("Downtime delta (+/-)", min_value=-0.20, max_value=0.20, value=0.0, step=0.01)
    scrap_delta = st.slider("Scrap delta (+/-)", min_value=-0.20, max_value=0.20, value=0.0, step=0.01)

    # Recompute with adjusted likely parameters
    adj = compute_one_scenario(
        W_lbs=float(W_vals["LIKELY"] * sites),
        recovery_rate=clamp(recovery_vals["LIKELY"] + recovery_delta, 0.0, 1.0),
        yield_rate=clamp(yield_vals["LIKELY"] + yield_delta, 0.0, 1.0),
        downtime_rate=clamp(downtime_vals["LIKELY"] + downtime_delta, 0.0, 1.0),
        contamination_scrap_rate=clamp(scrap_vals["LIKELY"] + scrap_delta, 0.0, 1.0),
        price_sell_per_lb=float(sell_price_vals["LIKELY"] * price_mult),
        internal_value_per_lb=float(internal_value_vals["LIKELY"] * internal_mult),
        mix_sell_share=float(sell_share_vals["LIKELY"]),
        disposal_avoid_per_lb=float(disposal_avoid_vals["LIKELY"]),
        carbon_value_per_lb=float(carbon_value_vals["LIKELY"]),
        op_cost_monthly=float(op_cost_vals["LIKELY"] * sites),
    )

    adj_profit = float(adj["Net profit ($/mo)"])
    delta_profit = adj_profit - base_profit

    s1, s2, s3 = st.columns(3)
    s1.metric("Base Likely Profit", money(base_profit))
    s2.metric("Adjusted Likely Profit", money(adj_profit))
    s3.metric("Change", money(delta_profit))

    if adj_profit > base_profit:
        st.success("This sensitivity set increases profitability. Use it to justify the exact improvement plan.")
    else:
        st.warning("This sensitivity set reduces profitability. Useful for worst-case risk explanation.")

with tab4:
    st.subheader("Export (client-ready)")
    st.download_button("Download scenario table (CSV)", data=df.to_csv(index=False), file_name="banding_recovery_scenarios.csv", mime="text/csv")

    likely_components = value_break.copy()
    st.download_button("Download likely value components (CSV)", data=likely_components.to_csv(index=False), file_name="banding_recovery_value_components_likely.csv", mime="text/csv")

    # Executive summary row
    summary = pd.DataFrame([{
        "Sites": sites,
        "CapEx total": total_capex,
        "Best profit/mo": best_profit,
        "Likely profit/mo": likely_profit,
        "Worst profit/mo": worst_profit,
        "Best payback (mo)": pb_best if math.isfinite(pb_best) else np.nan,
        "Likely payback (mo)": pb_likely if math.isfinite(pb_likely) else np.nan,
        "Worst payback (mo)": pb_worst if math.isfinite(pb_worst) else np.nan,
        "Likely recovered lbs/mo": float(likely_row["Recovered (lbs/mo)"]),
        "Likely finished lbs/mo": float(likely_row["Finished output (lbs/mo)"]),
    }])
    st.download_button("Download executive summary (CSV)", data=summary.to_csv(index=False), file_name="banding_recovery_executive_summary.csv", mime="text/csv")
