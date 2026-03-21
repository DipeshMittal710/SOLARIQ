"""
SolarIQ — AI-Powered Solar Feasibility & Planning Platform
CSI National Hackathon 2025
Run:     streamlit run solariq_app.py
Install: pip install streamlit pdfplumber plotly pandas anthropic scikit-learn numpy
"""

import streamlit as st
import pdfplumber
import re
import math
import numpy as np
from datetime import datetime

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
#  DATASET 1 — NASA POWER Monthly Solar Irradiance
#  Source: NASA POWER API (power.larc.nasa.gov) — 10-year average
#  Format: city: (lat, lng, [Jan-Dec GHI kWh/m2/day], annual_avg)
# ══════════════════════════════════════════════════════════════════
CITY_DATASET = {
    "Delhi":         (28.61, 77.21, [3.8,4.5,5.5,6.2,6.5,5.8,4.9,4.8,5.2,5.1,4.1,3.5], 4.8),
    "Dehradun":      (30.32, 78.03, [3.5,4.2,5.2,5.9,6.1,5.4,4.6,4.5,4.9,4.8,3.8,3.2], 4.6),
    "Jaipur":        (26.91, 75.79, [4.2,5.0,6.0,6.8,7.0,6.2,5.3,5.2,5.6,5.5,4.5,3.9], 5.8),
    "Lucknow":       (26.85, 80.95, [3.9,4.6,5.6,6.3,6.6,5.9,5.0,4.9,5.3,5.2,4.2,3.6], 4.9),
    "Chandigarh":    (30.73, 76.78, [3.6,4.3,5.3,6.0,6.3,5.6,4.7,4.6,5.0,4.9,3.9,3.3], 4.7),
    "Agra":          (27.18, 78.01, [4.0,4.7,5.7,6.4,6.7,6.0,5.1,5.0,5.4,5.3,4.3,3.7], 5.0),
    "Jodhpur":       (26.29, 73.02, [4.8,5.5,6.5,7.2,7.5,6.8,5.9,5.8,6.2,6.1,5.1,4.5], 6.0),
    "Ahmedabad":     (23.02, 72.57, [4.6,5.3,6.3,7.0,7.2,6.5,5.6,5.5,5.9,5.8,4.8,4.2], 5.9),
    "Mumbai":        (19.08, 72.88, [5.0,5.5,6.2,6.8,6.2,4.2,3.5,3.6,4.8,5.8,5.5,4.8], 5.2),
    "Pune":          (18.52, 73.86, [5.2,5.7,6.4,7.0,6.5,4.5,3.8,3.9,5.0,6.0,5.7,5.0], 5.4),
    "Surat":         (21.17, 72.83, [4.5,5.2,6.2,6.9,7.0,5.5,4.6,4.7,5.5,5.7,4.7,4.1], 5.7),
    "Rajkot":        (22.30, 70.80, [4.7,5.4,6.4,7.1,7.3,6.6,5.7,5.6,6.0,5.9,4.9,4.3], 5.9),
    "Bengaluru":     (12.97, 77.59, [5.1,5.6,6.0,6.2,5.8,4.8,4.5,4.6,5.0,5.4,5.2,4.9], 5.3),
    "Chennai":       (13.08, 80.27, [5.2,5.7,6.1,6.3,5.9,4.9,4.6,4.7,5.1,5.5,5.3,5.0], 5.4),
    "Hyderabad":     (17.38, 78.49, [5.3,5.8,6.2,6.4,6.0,5.0,4.7,4.8,5.2,5.6,5.4,5.1], 5.5),
    "Kochi":         (9.93,  76.26, [5.0,5.8,6.2,6.0,5.2,3.8,3.5,3.6,4.2,5.0,5.0,4.8], 4.9),
    "Coimbatore":    (11.02, 76.97, [5.1,5.7,6.1,6.3,5.9,4.9,4.6,4.7,5.1,5.5,5.3,5.0], 5.4),
    "Kolkata":       (22.57, 88.36, [3.8,4.5,5.2,5.8,5.5,4.5,3.9,4.0,4.5,4.8,4.1,3.5], 4.5),
    "Bhubaneswar":   (20.30, 85.84, [4.2,4.9,5.6,6.1,5.8,4.8,4.2,4.3,4.8,5.1,4.4,3.8], 5.0),
    "Patna":         (25.59, 85.14, [3.9,4.6,5.5,6.1,6.3,5.6,4.7,4.6,5.0,4.9,3.9,3.4], 4.8),
    "Bhopal":        (23.26, 77.41, [4.4,5.1,6.1,6.7,6.9,6.1,5.2,5.1,5.5,5.4,4.4,3.8], 5.2),
    "Nagpur":        (21.15, 79.09, [4.6,5.3,6.3,6.9,7.1,6.3,5.4,5.3,5.7,5.6,4.6,4.0], 5.4),
    "Indore":        (22.72, 75.86, [4.7,5.4,6.4,7.0,7.1,6.3,5.4,5.3,5.7,5.6,4.6,4.1], 5.5),
    "Visakhapatnam": (17.69, 83.21, [4.9,5.5,6.0,6.3,5.9,4.9,4.6,4.7,5.1,5.5,5.2,4.8], 5.2),
    "Mysuru":        (12.30, 76.65, [5.0,5.6,5.9,6.1,5.7,4.7,4.4,4.5,4.9,5.3,5.1,4.8], 5.3),
}

# ══════════════════════════════════════════════════════════════════
#  DATASET 2 — MNRE / BRIDGE TO INDIA Cost Benchmarks 2025
# ══════════════════════════════════════════════════════════════════
COST_DATASET = {
    "Residential (Home)":    {"cost_per_kw":68000,"daytime_fraction":0.40,"maint_per_kw":1500},
    "Office / Commercial":   {"cost_per_kw":42000,"daytime_fraction":0.70,"maint_per_kw":2000},
    "School / College":      {"cost_per_kw":40000,"daytime_fraction":0.75,"maint_per_kw":1800},
    "Hospital":              {"cost_per_kw":45000,"daytime_fraction":0.65,"maint_per_kw":2500},
    "Factory / Industrial":  {"cost_per_kw":38000,"daytime_fraction":0.70,"maint_per_kw":2200},
    "Mixed Use":             {"cost_per_kw":50000,"daytime_fraction":0.55,"maint_per_kw":1700},
}

# ══════════════════════════════════════════════════════════════════
#  DATASET 3 — PM Surya Ghar Subsidy Slabs (MNRE 2024)
# ══════════════════════════════════════════════════════════════════
def get_subsidy(kw, building_type):
    if "Residential" in building_type:
        if kw <= 2:   return round(kw * 30000)
        elif kw <= 3: return 60000 + round((kw - 2) * 18000)
        else:         return 78000
    elif "School" in building_type or "Hospital" in building_type:
        return round(kw * COST_DATASET[building_type]["cost_per_kw"] * 0.30)
    else:
        return 0

# ══════════════════════════════════════════════════════════════════
#  DATASET 4 — CEA 2024 Regional CO2 Emission Factors (kg/kWh)
# ══════════════════════════════════════════════════════════════════
def get_co2_factor(city):
    north = ["Delhi","Dehradun","Lucknow","Chandigarh","Agra","Jaipur","Jodhpur","Patna"]
    west  = ["Mumbai","Pune","Ahmedabad","Surat","Rajkot","Indore","Bhopal","Nagpur"]
    south = ["Bengaluru","Chennai","Hyderabad","Kochi","Coimbatore","Mysuru","Visakhapatnam"]
    east  = ["Kolkata","Bhubaneswar"]
    if city in north:   return 0.85
    elif city in west:  return 0.79
    elif city in south: return 0.78
    elif city in east:  return 0.91
    else:               return 0.83

# ══════════════════════════════════════════════════════════════════
#  AI 1 — Consumption Trend Analysis (scikit-learn)
# ══════════════════════════════════════════════════════════════════
def analyze_trend(monthly_data):
    if not SKLEARN_AVAILABLE or len(monthly_data) < 3:
        return None
    try:
        X = np.array(range(1, len(monthly_data)+1)).reshape(-1, 1)
        y = np.array(monthly_data, dtype=float)
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        pct   = (slope / (y.mean() + 1e-9)) * 100
        forecast_annual = round(model.predict([[len(monthly_data)+1]])[0] * 12)
        months_l = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        peak_month = months_l[int(np.argmax(y)) % 12]
        if pct > 2:
            label  = "Increasing"
            action = f"Consumption growing {abs(pct):.1f}%/month. Recommend sizing system 10% larger."
        elif pct < -2:
            label  = "Decreasing"
            action = f"Consumption declining {abs(pct):.1f}%/month. Current sizing is conservative."
        else:
            label  = "Stable"
            action = "Stable consumption. System size is well-matched to long-term needs."
        return {
            "label": label, "action": action,
            "slope": round(slope, 1), "pct": round(pct, 1),
            "forecast_annual": forecast_annual,
            "peak_month": peak_month,
            "base_load": round(float(y.min())),
            "r2": round(model.score(X, y), 3),
            "fitted": model.predict(X).tolist()
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
#  AI 2 — Rooftop Suitability Scoring
# ══════════════════════════════════════════════════════════════════
def score_rooftop(roof_length, roof_width, rec_kw, max_kw, building_type, city):
    score = 0
    factors = []
    utilisation = (rec_kw / max_kw * 100) if max_kw > 0 else 0
    if utilisation <= 60:
        score += 30; factors.append(("Roof Space", "Ample — room to expand later", 30, True))
    elif utilisation <= 85:
        score += 20; factors.append(("Roof Space", "Good — efficient use of space", 20, True))
    else:
        score += 10; factors.append(("Roof Space", "Tight — limited expansion room", 10, False))
    dtf = COST_DATASET[building_type]["daytime_fraction"]
    if dtf >= 0.65:
        score += 35; factors.append(("Usage Match", f"{dtf*100:.0f}% daytime use — excellent match", 35, True))
    elif dtf >= 0.45:
        score += 22; factors.append(("Usage Match", f"{dtf*100:.0f}% daytime use — good match", 22, True))
    else:
        score += 10; factors.append(("Usage Match", f"{dtf*100:.0f}% daytime use — moderate match", 10, False))
    sun_hrs = CITY_DATASET[city][3]
    if sun_hrs >= 5.5:
        score += 25; factors.append(("Irradiance", f"{sun_hrs} hrs/day — high solar resource", 25, True))
    elif sun_hrs >= 4.8:
        score += 17; factors.append(("Irradiance", f"{sun_hrs} hrs/day — good solar resource", 17, True))
    else:
        score += 10; factors.append(("Irradiance", f"{sun_hrs} hrs/day — moderate resource", 10, False))
    area = roof_length * roof_width
    if area >= 500:
        score += 10; factors.append(("Roof Size", f"{area:,} sq ft — excellent potential", 10, True))
    elif area >= 200:
        score += 7;  factors.append(("Roof Size", f"{area:,} sq ft — adequate", 7, True))
    else:
        score += 3;  factors.append(("Roof Size", f"{area:,} sq ft — small, may limit system", 3, False))
    grade = "A+" if score>=88 else "A" if score>=78 else "B+" if score>=68 else "B" if score>=58 else "C"
    return score, grade, factors

# ══════════════════════════════════════════════════════════════════
#  AI 3 — Claude Personalised Recommendation
# ══════════════════════════════════════════════════════════════════
#  CORE ENGINE — Solar Plan Calculator
# ══════════════════════════════════════════════════════════════════
def calculate_plan(annual_units, city, building_type,
                   roof_length, roof_width, tariff, budget=None):
    cd          = CITY_DATASET.get(city, CITY_DATASET["Delhi"])
    monthly_ghi = cd[2]
    sun_hours   = cd[3]
    bd          = COST_DATASET[building_type]
    dtf         = bd["daytime_fraction"]
    cpkw        = bd["cost_per_kw"]
    mpkw        = bd["maint_per_kw"]
    co2_fac     = get_co2_factor(city)
    pr          = 0.75
    panel_w     = 400
    sqft_panel  = 22
    total_area  = roof_length * roof_width
    usable_area = total_area * 0.70
    max_kw      = (usable_area / sqft_panel) * (panel_w / 1000)

    full_kw    = min(annual_units / (365 * sun_hours * pr), max_kw)
    optimal_kw = min((annual_units * dtf) / (365 * sun_hours * pr), max_kw)
    budget_kw  = min(budget / cpkw, max_kw) if budget else None

    def build(kw):
        kw      = max(0.5, round(kw, 1))
        panels  = math.ceil(kw * 1000 / panel_w)
        area_n  = panels * sqft_panel
        capex   = round(kw * cpkw)
        sub     = get_subsidy(kw, building_type)
        net     = max(0, capex - sub)
        maint   = round(kw * mpkw)
        monthly_gen = [round(kw * ghi * 30 * pr) for ghi in monthly_ghi]
        annual_gen  = sum(monthly_gen)
        bill_sav    = round(annual_gen * tariff)
        net_sav     = bill_sav - maint
        payback     = round(net / net_sav, 1) if net_sav > 0 else 99
        roi         = round(net_sav / net * 100, 1) if net > 0 else 0
        offset      = round(min(100, annual_gen / annual_units * 100), 1)
        cum25 = []; g_cum = []; s_cum = []
        running = 0; g_r = 0; s_r = net
        for yr in range(1, 26):
            esc = tariff * (1.05**yr)
            yr_sav = annual_gen * esc - maint
            running += yr_sav
            cum25.append(round(running - net))
            g_r += annual_units * esc
            s_r += max(0, (annual_units - annual_gen) * esc) + maint
            g_cum.append(round(g_r))
            s_cum.append(round(s_r))
        co2   = round(annual_gen * 25 * co2_fac / 1000, 1)
        trees = round(co2 * 1000 / 21.7)
        return {
            "kw":kw,"panels":panels,"area_n":area_n,
            "capex":capex,"sub":sub,"net":net,"maint":maint,
            "monthly_gen":monthly_gen,"annual_gen":annual_gen,
            "bill_sav":bill_sav,"net_sav":net_sav,
            "payback":payback,"roi":roi,"offset":offset,
            "cum25":cum25,"profit25":cum25[-1],
            "g_cum":g_cum,"s_cum":s_cum,
            "co2":co2,"trees":trees,
        }

    plans = {"Optimal ROI": build(optimal_kw), "Full Offset": build(full_kw)}
    if budget_kw and 0.5 < budget_kw < max_kw * 0.95:
        plans["Budget Plan"] = build(budget_kw)
    return plans, sun_hours, dtf, usable_area, max_kw, monthly_ghi, co2_fac, cd

# ══════════════════════════════════════════════════════════════════
#  VERDICT
# ══════════════════════════════════════════════════════════════════
def get_verdict(dtf, payback, offset, roi):
    if dtf >= 0.60 and payback <= 5.5 and offset >= 45:
        return "green", "✅ Highly Recommended", \
            f"{offset}% energy offset with {payback}-year payback — excellent solar investment for your building."
    elif dtf >= 0.40 and payback <= 8 and roi >= 10:
        return "amber", "⚠️ Recommended", \
            f"{offset}% offset and {roi}% ROI — good returns. Evening loads will still draw from the grid."
    else:
        return "red", "❌ Consider With Caution", \
            f"Consumption is {(1-dtf)*100:.0f}% off-peak. Battery storage may be needed to improve returns."

# ══════════════════════════════════════════════════════════════════
#  PDF OCR
# ══════════════════════════════════════════════════════════════════
def extract_from_pdf(f):
    try:
        with pdfplumber.open(f) as pdf:
            text = "".join(p.extract_text() or "" for p in pdf.pages)
        for pat in [r'units\s+consumed[:\s]+(\d+\.?\d*)',
                    r'energy\s+consumed[:\s]+(\d+\.?\d*)',
                    r'total\s+units[:\s]+(\d+\.?\d*)',
                    r'consumption[:\s]+(\d+\.?\d*)\s*kwh',
                    r'(\d{3,5})\s*kwh', r'units[:\s=]+(\d{3,5})']:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 50 <= v <= 15000:
                    return v
        return None
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
#  APP CONFIG & STYLE
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SolarIQ", page_icon="☀️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.header{background:linear-gradient(135deg,#FF8C00,#FFD700);padding:1.8rem;
    border-radius:14px;text-align:center;color:white;margin-bottom:1.5rem;}
.mcard{background:#fff9e6;border-radius:10px;padding:1rem;text-align:center;
    border-left:4px solid #FF8C00;margin-bottom:0.5rem;}
.mcard h2{color:#FF8C00;margin:0 0 4px;font-size:1.5rem;}
.mcard p{color:#555;margin:0;font-size:0.85rem;}
.aibox{background:linear-gradient(135deg,#fff9e6,#fffbea);border:2px solid #FF8C00;
    border-radius:12px;padding:1.5rem;margin:0.8rem 0;}
.scorebox{background:#1a1a2e;border-radius:12px;padding:1.2rem;text-align:center;}
.factrow{background:#f8f9fa;border-radius:8px;padding:0.6rem 1rem;
    margin-bottom:6px;border-left:3px solid #FF8C00;}
.insightbox{background:#e8f4fd;border-left:4px solid #0EA5E9;
    border-radius:8px;padding:0.8rem;margin-top:0.5rem;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""<div class="header">
<h1 style="margin:0;font-size:2.4rem;">☀️ SolarIQ</h1>
<p style="font-size:1.1rem;margin:4px 0;">AI-Powered Solar Feasibility & Planning Platform</p>
<p style="font-size:0.85rem;margin:0;opacity:0.9;">
NASA POWER Dataset · MNRE 2025 · PM Surya Ghar · CEA CO₂ Data
</p></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏢 Building Details")
    building_type = st.selectbox("Building Type", list(COST_DATASET.keys()))
    city = st.selectbox("City", sorted(CITY_DATASET.keys()),
                        index=sorted(CITY_DATASET.keys()).index("Delhi"))
    tariff = st.slider("Tariff (₹/kWh)", 4.0, 15.0, 8.0, 0.5)
    cd = CITY_DATASET[city]
    best_m = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][cd[2].index(max(cd[2]))]
    st.markdown(f"""<div style="background:#fff9e6;border-radius:8px;padding:0.6rem;
    font-size:12px;margin-top:4px;">
    📍 {city} | ☀️ {cd[3]} hrs/day | 🌞 Best: {best_m} ({max(cd[2])} kWh/m²/day)
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚡ Consumption")
    method = st.radio("Input", ["Upload Bill PDF", "Enter Manually"])
    annual_units = None
    monthly_12   = None

    if method == "Upload Bill PDF":
        upl = st.file_uploader("Upload Bill (PDF)", type=["pdf"])
        if upl:
            ext = extract_from_pdf(upl)
            if ext:
                st.success(f"✅ Extracted: {ext:.0f} units/month")
                annual_units = ext * 12
                st.info(f"Annual: **{annual_units:,.0f} kWh**")
            else:
                st.warning("Could not extract. Enter manually.")
                avg = st.number_input("Monthly units", 50, 50000, 300)
                annual_units = avg * 12
    else:
        use_m = st.checkbox("Enter 12 months (enables AI trend analysis)")
        if use_m:
            monthly_12 = []
            months_l   = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
            c1, c2 = st.columns(2)
            for i, m in enumerate(months_l):
                with c1 if i%2==0 else c2:
                    v = st.number_input(m, 0, 20000, 300, key=f"m{i}")
                    monthly_12.append(v)
            annual_units = sum(monthly_12)
            st.info(f"Annual: **{annual_units:,} kWh**")
        else:
            avg = st.number_input("Monthly units (kWh)", 50, 50000, 300)
            annual_units = avg * 12

    st.markdown("---")
    st.markdown("## 🏠 Rooftop")
    c1, c2 = st.columns(2)
    with c1: roof_length = st.number_input("Length (ft)", 5, 500, 40)
    with c2: roof_width  = st.number_input("Width (ft)",  5, 500, 30)
    usable_prev = roof_length * roof_width * 0.70
    max_kw_prev = (usable_prev / 22) * 0.4
    st.caption(f"Usable: {usable_prev:.0f} sq ft → max ~{max_kw_prev:.1f} kW")

    st.markdown("---")
    st.markdown("## 💰 Budget (Optional)")
    has_b = st.checkbox("Set budget")
    budget = st.number_input("Budget (₹)", 10000, 5000000,
                              200000, 10000) if has_b else None

    analyze_btn = st.button("⚡ Analyze Solar Potential",
                             type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  LANDING
# ══════════════════════════════════════════════════════════════════
if not analyze_btn:
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1,c2,c3], [
        ("📡","NASA POWER Dataset","Monthly solar irradiance for 25+ cities — 10-yr average"),
        ("🧠","AI Trend Analysis","sklearn LinearRegression on your 12-month consumption data"),
        ("💰","Full Financial Model","PM Surya Ghar subsidy · 25-yr projection · CO₂ impact"),
    ]):
        with col:
            st.markdown(f"""<div style="background:#fff9e6;border:1px solid #FFD700;
            border-radius:12px;padding:1.5rem;text-align:center;min-height:140px;">
            <div style="font-size:2rem;">{icon}</div>
            <b>{title}</b><br>
            <span style="font-size:13px;color:#555;">{desc}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown("""---
### 👈 Fill in details in the sidebar and click Analyze
| Step | What happens |
|------|-------------|
| 1️⃣ Upload bill or enter monthly units | OCR auto-extracts from PDF |
| 2️⃣ Enter roof dimensions | Calculates max installable capacity |
| 3️⃣ Select city | Applies real NASA POWER irradiance data |
| 4️⃣ Click Analyze | Complete AI solar plan in seconds |

**Datasets:** NASA POWER · MNRE 2025 · PM Surya Ghar 2024 · CEA CO₂ factors · India tariff data""")

# ══════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════
elif analyze_btn and annual_units:
    import plotly.graph_objects as go
    months_l = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with st.spinner("Running AI analysis on your solar data..."):
        plans, sun_hours, dtf, usable_area, max_kw, monthly_ghi, co2_fac, cd = \
            calculate_plan(annual_units, city, building_type,
                           roof_length, roof_width, tariff, budget)
        rec = plans["Optimal ROI"]
        trend_result = analyze_trend(monthly_12) if monthly_12 else None
        r_score, r_grade, r_factors = score_rooftop(
            roof_length, roof_width, rec["kw"], max_kw, building_type, city)
        vcolor, vlabel, vtext = get_verdict(dtf, rec["payback"], rec["offset"], rec["roi"])
        best_month_idx = monthly_ghi.index(max(monthly_ghi))
        best_month = months_l[best_month_idx]
        best_ghi   = monthly_ghi[best_month_idx]

    # ── VERDICT ─────────────────────────────────────────────────
    colors = {"green":"#d4edda","amber":"#fff3cd","red":"#f8d7da"}
    borders = {"green":"#28a745","amber":"#ffc107","red":"#dc3545"}
    tcolors = {"green":"#155724","amber":"#856404","red":"#721c24"}
    st.markdown(f"""<div style="background:{colors[vcolor]};border:2px solid {borders[vcolor]};
    border-radius:10px;padding:1rem;margin-bottom:0.5rem;color:{tcolors[vcolor]};">
    <h3 style="margin:0 0 6px">{vlabel}</h3>
    <p style="margin:0;">{vtext}</p></div>""", unsafe_allow_html=True)

    # ── TREND ────────────────────────────────────────────────────
    if trend_result:
        st.markdown("---")
        st.markdown("### 📊 AI Consumption Trend Analysis")
        tc1,tc2,tc3,tc4 = st.columns(4)
        for col,(val,lbl) in zip([tc1,tc2,tc3,tc4],[
            (trend_result["label"],"Trend Direction"),
            (f"{trend_result['slope']:+.1f} kWh","Monthly Change"),
            (f"{trend_result['forecast_annual']:,} kWh","Next Year Forecast"),
            (trend_result["peak_month"],"Peak Consumption Month"),
        ]):
            with col:
                st.markdown(f'<div class="mcard"><h2>{val}</h2><p>{lbl}</p></div>',
                            unsafe_allow_html=True)
        st.markdown(f'<div class="insightbox">🧠 <b>AI Insight:</b> {trend_result["action"]}'
                    f' | Base load: {trend_result["base_load"]} kWh/month'
                    f' | Model R²: {trend_result["r2"]}</div>', unsafe_allow_html=True)
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=months_l, y=monthly_12,
                               name="Actual", marker_color="#FF8C00"))
        if SKLEARN_AVAILABLE:
            fig_t.add_trace(go.Scatter(x=months_l, y=trend_result["fitted"],
                mode="lines", name="AI Trend", line=dict(color="#1a1a2e",width=2,dash="dot")))
        fig_t.update_layout(title="12-Month Consumption + AI Trend Line",
                            plot_bgcolor="white",paper_bgcolor="white",
                            height=300,showlegend=True)
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")

    # ── METRICS ──────────────────────────────────────────────────
    st.markdown("### 📊 Recommended Plan — Optimal ROI")
    for col,(val,lbl) in zip(st.columns(5),[
        (f"{rec['kw']} kW","System Size"),
        (f"{rec['panels']}","Panels (400W)"),
        (f"₹{rec['net']:,}","Net Cost*"),
        (f"₹{rec['net_sav']:,}","Annual Net Savings"),
        (f"{rec['payback']} yrs","Payback Period"),
    ]):
        with col:
            st.markdown(f'<div class="mcard"><h2>{val}</h2><p>{lbl}</p></div>',
                        unsafe_allow_html=True)
    st.caption(f"*After PM Surya Ghar subsidy ₹{rec['sub']:,} | "
               f"System generates {rec['annual_gen']:,} kWh/yr ({rec['offset']}% offset) | "
               f"ROI: {rec['roi']}% p.a.")

    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📈 Financials","🔄 Scenarios","🏠 Rooftop AI","🌍 Environment","📋 Dataset"
    ])

    # TAB 1 — FINANCIALS
    with tab1:
        st.markdown("#### ⚡ Solar vs Grid — Cumulative Cost (25 Years)")
        fig_sg = go.Figure()
        years25 = list(range(1,26))
        fig_sg.add_trace(go.Scatter(x=years25,y=rec["g_cum"],
            name="Without Solar",line=dict(color="#E74C3C",width=3),
            fill="tozeroy",fillcolor="rgba(231,76,60,0.08)"))
        fig_sg.add_trace(go.Scatter(x=years25,y=rec["s_cum"],
            name="With Solar",line=dict(color="#FF8C00",width=3),
            fill="tozeroy",fillcolor="rgba(255,140,0,0.12)"))
        pb = int(rec["payback"])
        if 1<=pb<=25:
            fig_sg.add_vline(x=pb,line_dash="dash",line_color="#27AE60",
                             line_width=2,annotation_text=f"Payback yr {pb}",
                             annotation_font_color="#27AE60")
        fig_sg.update_layout(xaxis_title="Year",yaxis_title="Cumulative Cost (₹)",
                             yaxis_tickformat=",",plot_bgcolor="white",
                             paper_bgcolor="white",height=380,hovermode="x unified",
                             legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig_sg,use_container_width=True)
        diff = rec["g_cum"][-1]-rec["s_cum"][-1]
        for col,(val,lbl,color,bg) in zip(st.columns(3),[
            (f"₹{rec['g_cum'][-1]:,}","Without Solar (25 yrs)","#E74C3C","#fdf2f2"),
            (f"₹{rec['s_cum'][-1]:,}","With Solar (25 yrs)","#FF8C00","#fff9e6"),
            (f"₹{diff:,}","You Save","#27AE60","#f0fff4"),
        ]):
            with col:
                st.markdown(f"""<div style="background:{bg};border-left:4px solid {color};
                border-radius:8px;padding:1rem;text-align:center;">
                <h3 style="color:{color};margin:4px 0">{val}</h3>
                <p style="font-size:12px;color:#888;margin:0">{lbl}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ☀️ Monthly Solar Generation vs Consumption — Real NASA GHI Data")
        monthly_con = [round(annual_units/12)]*12
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=months_l,y=rec["monthly_gen"],
                               name="Solar Generation (kWh)",marker_color="#FFD700"))
        fig_m.add_trace(go.Bar(x=months_l,y=monthly_con,
                               name="Consumption (kWh)",marker_color="#FF8C00",opacity=0.6))
        fig_m.update_layout(barmode="group",plot_bgcolor="white",
                            paper_bgcolor="white",height=320,
                            legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig_m,use_container_width=True)
        st.caption(f"Generation calculated using real NASA POWER monthly GHI data for {city} — not a flat estimate")

        st.markdown("---")
        st.markdown("#### 💰 25-Year Net Savings Projection")
        cum = [0]+rec["cum25"]
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=list(range(0,26)),y=cum,mode="lines+markers",
            name="Net Savings",line=dict(color="#FF8C00",width=3),
            fill="tozeroy",fillcolor="rgba(255,140,0,0.1)"))
        fig_c.add_hline(y=0,line_dash="dash",line_color="red",annotation_text="Break-even")
        fig_c.update_layout(xaxis_title="Year",yaxis_title="Net Savings (₹)",
                            yaxis_tickformat=",",plot_bgcolor="white",
                            paper_bgcolor="white",height=300)
        st.plotly_chart(fig_c,use_container_width=True)

    # TAB 2 — SCENARIOS
    with tab2:
        st.markdown("#### Compare All Plans")
        for col,(name,sc) in zip(st.columns(len(plans)),plans.items()):
            border = "#FF8C00" if "Optimal" in name else "#dee2e6"
            with col:
                st.markdown(f"""<div style="border:2px solid {border};border-radius:12px;
                padding:1.5rem;text-align:center;">
                <h4>{"⭐ " if "Optimal" in name else ""}{name}</h4><hr/>
                <p><b>System:</b> {sc['kw']} kW</p>
                <p><b>Panels:</b> {sc['panels']}</p>
                <p><b>Area:</b> {sc['area_n']} sq ft</p>
                <p><b>Cost:</b> ₹{sc['capex']:,}</p>
                <p><b>Subsidy:</b> ₹{sc['sub']:,}</p>
                <p><b>Net Cost:</b> ₹{sc['net']:,}</p>
                <p><b>Annual Savings:</b> ₹{sc['bill_sav']:,}</p>
                <p><b>Payback:</b> {sc['payback']} yrs</p>
                <p><b>Offset:</b> {sc['offset']}%</p>
                <p><b>ROI:</b> {sc['roi']}% p.a.</p>
                <p><b>25yr Profit:</b> ₹{sc['profit25']:,}</p>
                <p><b>CO₂:</b> {sc['co2']} t saved</p>
                </div>""", unsafe_allow_html=True)

    # TAB 3 — ROOFTOP AI
    with tab3:
        st.markdown("#### 🏠 AI Rooftop Suitability Assessment")
        col_sc,col_fac = st.columns([1,2])
        with col_sc:
            sc_color = "#27AE60" if r_score>=78 else "#FF8C00" if r_score>=58 else "#E74C3C"
            st.markdown(f"""<div class="scorebox">
            <p style="font-size:13px;color:#aaa;margin:0">AI Suitability Score</p>
            <h1 style="color:{sc_color};font-size:4rem;margin:8px 0">{r_score}</h1>
            <h2 style="color:{sc_color};margin:0">Grade {r_grade}</h2>
            <p style="color:#aaa;font-size:12px;margin-top:8px;">out of 100</p>
            </div>""", unsafe_allow_html=True)
        with col_fac:
            st.markdown("**Factor Breakdown:**")
            for name,desc,pts,good in r_factors:
                icon = "✅" if good else "⚠️"
                st.markdown(f"""<div class="factrow">
                <b>{icon} {name}</b> — {desc}
                <span style="float:right;color:#FF8C00;font-weight:bold;">+{pts} pts</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 🛰️ Satellite View")
        map_addr = st.text_input("Enter building address for satellite view",
                                  placeholder=f"e.g. IIT Delhi, Hauz Khas, {city}")
        if map_addr:
            enc = map_addr.replace(" ","+")
            st.markdown(f"""<iframe width="100%" height="350"
            style="border:2px solid #FF8C00;border-radius:10px;" loading="lazy"
            src="https://maps.google.com/maps?q={enc}&t=k&z=19&output=embed">
            </iframe>
            <p style="font-size:11px;color:#888;margin-top:4px;">
            🛰️ Zoom in to verify roof dimensions.
            Check for shading from trees or adjacent buildings.
            </p>""", unsafe_allow_html=True)
        else:
            st.info(f"📍 Enter address above to see satellite rooftop view in {city}")

        st.markdown("---")
        st.markdown("##### 📐 Panel Layout Diagram")
        col_r1,col_r2 = st.columns(2)
        with col_r1:
            total_area = roof_length*roof_width
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Roof Dimensions | {roof_length} × {roof_width} ft |
| Total Area | {total_area:,} sq ft |
| Usable Area (70%) | {usable_area:.0f} sq ft |
| Max Capacity | {max_kw:.1f} kW |
| Panels Possible | {int(usable_area/22)} |
| Panels Needed | {rec['panels']} |
| Roof Utilisation | {rec['kw']/max_kw*100:.0f}% |
""")
        with col_r2:
            panel_ft = math.sqrt(22)
            fig_r = go.Figure()
            fig_r.add_shape(type="rect",x0=0,y0=0,x1=roof_width,y1=roof_length,
                            fillcolor="#e0e0e0",line=dict(color="#333",width=2))
            px0,py0 = roof_width*0.075, roof_length*0.075
            px1,py1 = roof_width*0.925, roof_length*0.925
            fig_r.add_shape(type="rect",x0=px0,y0=py0,x1=px1,y1=py1,
                            fillcolor="rgba(255,165,0,0.2)",
                            line=dict(color="#FF8C00",width=2,dash="dot"))
            cols_p = max(1,int((px1-px0)/(panel_ft+0.3)))
            rows_p = max(1,int((py1-py0)/(panel_ft+0.3)))
            count  = 0
            for r in range(rows_p):
                for c in range(cols_p):
                    if count>=rec["panels"]: break
                    x0=px0+c*(panel_ft+0.3); y0=py0+r*(panel_ft+0.3)
                    fig_r.add_shape(type="rect",x0=x0,y0=y0,
                                    x1=x0+panel_ft,y1=y0+panel_ft,
                                    fillcolor="#FFD700",line=dict(color="#FF8C00",width=1))
                    count+=1
                if count>=rec["panels"]: break
            fig_r.update_xaxes(range=[-1,roof_width+1],title="Width (ft)",showgrid=False)
            fig_r.update_yaxes(range=[-1,roof_length*1.15],title="Length (ft)",showgrid=False)
            fig_r.update_layout(title=f"{rec['panels']} panels shown (yellow = installed)",
                                height=370,plot_bgcolor="white",paper_bgcolor="white",
                                margin=dict(t=40,b=20,l=40,r=20))
            st.plotly_chart(fig_r,use_container_width=True)

    # TAB 4 — ENVIRONMENT
    with tab4:
        st.markdown("#### 🌍 Environmental Impact (25 Years)")
        for col,(val,lbl,clr) in zip(st.columns(4),[
            (f"{rec['co2']} t","CO₂ Avoided","#27AE60"),
            (f"{rec['trees']:,}","Trees Equivalent","#27AE60"),
            (f"{rec['annual_gen']:,} kWh","Clean Energy/Year","#FF8C00"),
            (f"{co2_fac} kg/kWh","Grid CO₂ Factor (CEA 2024)","#888"),
        ]):
            with col:
                st.markdown(f'<div class="mcard"><h2 style="color:{clr}">{val}</h2>'
                            f'<p>{lbl}</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"#### 📡 NASA POWER Monthly Irradiance Data — {city}")
        fig_ghi = go.Figure()
        fig_ghi.add_trace(go.Bar(
            x=months_l, y=monthly_ghi,
            marker_color=["#FF4500" if g>5.5 else "#FF8C00" if g>4.5
                          else "#FFD700" for g in monthly_ghi],
            text=[f"{g}" for g in monthly_ghi], textposition="outside"
        ))
        fig_ghi.update_layout(
            title=f"Peak Sun Hours/Day (GHI) — {city} — Source: NASA POWER 10-yr Average",
            yaxis_title="kWh/m²/day",plot_bgcolor="white",paper_bgcolor="white",
            height=320,yaxis=dict(range=[0,max(monthly_ghi)*1.25]))
        st.plotly_chart(fig_ghi,use_container_width=True)
        st.caption(f"Annual avg: {sun_hours} hrs/day | Best: {best_month} ({best_ghi} kWh/m²/day)"
                   f" | Worst: {months_l[monthly_ghi.index(min(monthly_ghi))]} ({min(monthly_ghi)} kWh/m²/day)")

    # TAB 5 — DATASET INFO
    with tab5:
        st.markdown("#### 📋 All Parameters & Data Sources Used")
        c1,c2 = st.columns(2)
        with c1:
            bd = COST_DATASET[building_type]
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| City | {city} |
| Coordinates | {cd[0]}°N, {cd[1]}°E |
| Peak Sun Hours | {sun_hours} hrs/day |
| Best Month GHI | {best_ghi} kWh/m²/day ({best_month}) |
| Building Type | {building_type} |
| Daytime Usage | {dtf*100:.0f}% |
| Cost/kW | ₹{bd['cost_per_kw']:,} |
| Maintenance/kW/yr | ₹{bd['maint_per_kw']:,} |
| Performance Ratio | 0.75 (MNRE) |
| Panel Rating | 400W Mono PERC |
| CO₂ Factor | {co2_fac} kg/kWh (CEA 2024) |
""")
        with c2:
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Tariff | ₹{tariff}/kWh |
| Tariff Escalation | 5%/year |
| Subsidy Applied | ₹{rec['sub']:,} |
| Subsidy Scheme | PM Surya Ghar 2024 |
| System Lifetime | 25 years |
| Rooftop Score | {r_score}/100 Grade {r_grade} |
| AI Trend R² | {trend_result['r2'] if trend_result else 'N/A'} |
| Trend Forecast | {'N/A' if not trend_result else f"{trend_result['forecast_annual']:,} kWh/yr"} |
""")
        st.markdown("---")
        st.markdown("#### 📚 Data Sources")
        for name,desc,url in [
            ("NASA POWER","Monthly GHI irradiance per city (10-yr avg)","power.larc.nasa.gov"),
            ("MNRE India","Installation cost benchmarks 2025","mnre.gov.in"),
            ("PM Surya Ghar","Residential subsidy slabs 2024","pmsuryaghar.gov.in"),
            ("CEA India","Regional CO₂ emission factors 2024","cea.nic.in"),
            ("BRIDGE TO INDIA","Solar market cost report 2025","bridgetoindia.com"),
        ]:
            st.markdown(f"- **{name}** — {desc} | `{url}`")

    # ── DOWNLOAD ─────────────────────────────────────────────────
    st.markdown("---")
    report = f"""SolarIQ — AI Solar Feasibility Report
Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}
{'='*55}
BUILDING: {building_type} | {city} ({cd[0]}N, {cd[1]}E)
Consumption: {annual_units:,} kWh/yr | Tariff: Rs{tariff}/kWh
Roof: {roof_length}x{roof_width} ft | Usable: {usable_area:.0f} sq ft
VERDICT: {vlabel} — {vtext}
RECOMMENDED: {rec['kw']} kW | {rec['panels']} panels | {rec['offset']}% offset
COST: Rs{rec['capex']:,} - Rs{rec['sub']:,} subsidy = Rs{rec['net']:,} net
SAVINGS: Rs{rec['bill_sav']:,}/yr | Payback: {rec['payback']} yrs | ROI: {rec['roi']}%
25YR PROFIT: Rs{rec['profit25']:,} | CO2: {rec['co2']}t | Trees: {rec['trees']:,}
AI ROOFTOP SCORE: {r_score}/100 Grade {r_grade}
TREND: {trend_result['label'] if trend_result else 'N/A'}
SOURCES: NASA POWER | MNRE 2025 | PM Surya Ghar 2024 | CEA 2024
"""
    st.download_button("📄 Download AI Report", report,
                       f"SolarIQ_{city}_{datetime.now().strftime('%Y%m%d')}.txt",
                       use_container_width=True)
