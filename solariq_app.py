"""
SolarIQ — Smart Solar Feasibility & Planning Tool
CSI National Hackathon 2025
Run: streamlit run solariq_app.py
Install: pip install streamlit pdfplumber plotly pandas reportlab
"""

import streamlit as st
import pdfplumber
import re
import json
import math
import io
from datetime import datetime

# ─────────────────────────────────────────────
#  DATA: City → Peak Sun Hours (NASA POWER sourced)
# ─────────────────────────────────────────────
CITY_SUN_HOURS = {
    # North India
    "Delhi": 4.8, "Dehradun": 4.6, "Lucknow": 4.9, "Jaipur": 5.8,
    "Chandigarh": 4.7, "Agra": 5.0, "Varanasi": 5.0,
    # West India
    "Mumbai": 5.2, "Pune": 5.4, "Ahmedabad": 5.9, "Surat": 5.7,
    "Jodhpur": 6.0, "Udaipur": 5.8, "Rajkot": 5.9,
    # South India
    "Bengaluru": 5.3, "Chennai": 5.4, "Hyderabad": 5.5, "Kochi": 4.9,
    "Mysuru": 5.3, "Coimbatore": 5.4, "Visakhapatnam": 5.2,
    # East India
    "Kolkata": 4.5, "Bhubaneswar": 5.0, "Patna": 4.8, "Guwahati": 4.2,
    # Central India
    "Bhopal": 5.2, "Nagpur": 5.4, "Indore": 5.5, "Raipur": 5.3,
}

BUILDING_DAYTIME_FRACTION = {
    "Residential (Home)": 0.40,
    "Office / Commercial": 0.70,
    "School / College": 0.75,
    "Hospital": 0.65,
    "Factory / Industrial": 0.70,
    "Mixed Use": 0.55,
}

COST_PER_KW = {
    "Residential (Home)": 68000,
    "Office / Commercial": 42000,
    "School / College": 40000,
    "Hospital": 45000,
    "Factory / Industrial": 38000,
    "Mixed Use": 50000,
}

MAINTENANCE_PER_KW_ANNUAL = 1500  # ₹/kW/year

# ─────────────────────────────────────────────
#  HELPER: PDF Bill Parser
# ─────────────────────────────────────────────
def extract_units_from_pdf(uploaded_file):
    """Extract monthly units consumed from electricity bill PDF."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        # Common patterns across Indian DISCOM bills
        patterns = [
            r'units\s+consumed[:\s]+(\d+\.?\d*)',
            r'energy\s+consumed[:\s]+(\d+\.?\d*)',
            r'total\s+units[:\s]+(\d+\.?\d*)',
            r'net\s+units[:\s]+(\d+\.?\d*)',
            r'kwh\s+consumed[:\s]+(\d+\.?\d*)',
            r'consumption[:\s]+(\d+\.?\d*)\s*kwh',
            r'(\d{3,5})\s*kwh',
            r'units[:\s=]+(\d{3,5})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 50 <= val <= 10000:  # sanity check
                    return val, text[:500]  # return value + preview

        return None, text[:500]
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
#  CORE ENGINE: Solar Sizing
# ─────────────────────────────────────────────
def calculate_solar_plan(annual_units, city, building_type,
                          roof_length, roof_width, tariff, budget=None):
    """Main calculation engine — returns full solar plan."""

    sun_hours = CITY_SUN_HOURS.get(city, 5.0)
    daytime_fraction = BUILDING_DAYTIME_FRACTION.get(building_type, 0.55)
    cost_per_kw = COST_PER_KW.get(building_type, 55000)

    # Usable roof area
    total_roof_area = roof_length * roof_width          # sq ft
    usable_roof_area = total_roof_area * 0.70           # 70% usable
    sqft_per_kw = 100                                   # ~100 sq ft per kW
    max_kw_by_roof = usable_roof_area / sqft_per_kw

    # Sizing: solar-matchable consumption
    effective_units = annual_units * daytime_fraction
    pr = 0.75  # Performance Ratio (industry standard)

    # Three scenarios
    full_offset_kw = annual_units / (365 * sun_hours * pr)
    optimal_kw = effective_units / (365 * sun_hours * pr)
    budget_kw = (budget / cost_per_kw) if budget else None

    # Cap by roof
    full_offset_kw = min(full_offset_kw, max_kw_by_roof)
    optimal_kw = min(optimal_kw, max_kw_by_roof)

    def build_scenario(kw):
        kw = round(kw, 1)
        panel_count = math.ceil(kw * 1000 / 400)           # 400W panels
        area_needed = panel_count * 22                      # 22 sq ft per panel

        capex = kw * cost_per_kw

        # PM Surya Ghar subsidy (residential only)
        if "Residential" in building_type:
            if kw <= 2:
                subsidy = kw * 30000
            elif kw <= 3:
                subsidy = 60000 + (kw - 2) * 18000
            else:
                subsidy = 78000
        elif "School" in building_type or "Hospital" in building_type:
            # Accelerated depreciation benefit approx
            subsidy = capex * 0.12
        else:
            subsidy = 0

        net_cost = max(0, capex - subsidy)
        annual_maintenance = kw * MAINTENANCE_PER_KW_ANNUAL

        annual_gen = kw * sun_hours * 365 * pr             # kWh/year
        annual_savings_base = annual_gen * tariff
        # Net of maintenance
        annual_net_savings = annual_savings_base - annual_maintenance

        # 25-year projection with 5% annual tariff escalation
        total_savings_25yr = 0
        cumulative = []
        for yr in range(1, 26):
            yr_savings = annual_gen * tariff * (1.05 ** yr) - annual_maintenance
            total_savings_25yr += yr_savings
            cumulative.append(round(total_savings_25yr - net_cost, 0))

        payback = net_cost / annual_net_savings if annual_net_savings > 0 else 99
        roi_pct = (annual_net_savings / net_cost * 100) if net_cost > 0 else 0
        offset_pct = min(100, round(annual_gen / annual_units * 100, 1))
        co2_offset = round(annual_gen * 25 * 0.82 / 1000, 1)  # tonnes over 25 yrs

        return {
            "kw": kw,
            "panel_count": panel_count,
            "area_needed": area_needed,
            "capex": round(capex),
            "subsidy": round(subsidy),
            "net_cost": round(net_cost),
            "annual_gen": round(annual_gen),
            "annual_savings": round(annual_savings_base),
            "annual_maintenance": round(annual_maintenance),
            "annual_net_savings": round(annual_net_savings),
            "payback": round(payback, 1),
            "roi_pct": round(roi_pct, 1),
            "offset_pct": offset_pct,
            "co2_offset": co2_offset,
            "cumulative_25yr": cumulative,
            "total_savings_25yr": round(total_savings_25yr - net_cost),
        }

    scenarios = {
        "Optimal ROI": build_scenario(optimal_kw),
        "Full Offset": build_scenario(full_offset_kw),
    }
    if budget_kw and 0.5 < budget_kw < max_kw_by_roof:
        scenarios["Budget Plan"] = build_scenario(budget_kw)

    return scenarios, sun_hours, daytime_fraction, usable_roof_area, max_kw_by_roof


# ─────────────────────────────────────────────
#  RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def get_recommendation(daytime_fraction, payback_years, offset_pct, roi_pct):
    if daytime_fraction >= 0.60 and payback_years <= 5.5 and offset_pct >= 50:
        return "highly_recommended", "Highly Recommended", (
            "Your building's daytime usage pattern aligns excellently with solar generation "
            f"hours. With {offset_pct}% energy offset and a payback in {payback_years} years, "
            "this is a strong investment with long-term returns."
        )
    elif daytime_fraction >= 0.40 and payback_years <= 8 and roi_pct >= 10:
        return "recommended", "Recommended", (
            "Solar will significantly reduce your electricity bills. "
            f"Your system offsets {offset_pct}% of consumption with a {payback_years}-year "
            "payback. Evening/night loads will still draw from the grid."
        )
    else:
        return "low_benefit", "Consider With Caution", (
            "Your consumption pattern is heavily evening or night-weighted, which reduces "
            "solar self-consumption. Battery storage could improve returns — "
            "consult an installer for a detailed site assessment."
        )


# ─────────────────────────────────────────────
#  UI: PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SolarIQ — Solar Feasibility Tool",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF8C00, #FFD700);
        padding: 2rem; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1.2rem; text-align: center;
        border-left: 4px solid #FF8C00;
    }
    .rec-green {
        background: #d4edda; border: 2px solid #28a745;
        border-radius: 10px; padding: 1rem; color: #155724;
    }
    .rec-amber {
        background: #fff3cd; border: 2px solid #ffc107;
        border-radius: 10px; padding: 1rem; color: #856404;
    }
    .rec-red {
        background: #f8d7da; border: 2px solid #dc3545;
        border-radius: 10px; padding: 1rem; color: #721c24;
    }
    .scenario-card {
        background: white; border: 1px solid #dee2e6;
        border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background: #fff9e6; border: 1px solid #FFD700;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>☀️ SolarIQ</h1>
    <p style="font-size: 1.2rem; margin: 0;">
        Smart Solar Feasibility & Planning Tool for Indian Buildings
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
        Data-driven recommendations using real solar irradiance data
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR: INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Building Details")

    building_type = st.selectbox(
        "Building Type",
        list(BUILDING_DAYTIME_FRACTION.keys()),
        help="This determines your daytime usage pattern"
    )

    city = st.selectbox(
        "City",
        sorted(CITY_SUN_HOURS.keys()),
        index=sorted(CITY_SUN_HOURS.keys()).index("Delhi")
    )

    tariff = st.slider(
        "Electricity Tariff (₹/kWh)",
        min_value=4.0, max_value=15.0, value=8.0, step=0.5,
        help="Check your electricity bill for your tariff rate"
    )

    st.markdown("---")
    st.markdown("## 📄 Electricity Consumption")

    input_method = st.radio(
        "How to enter consumption?",
        ["Upload Bill PDF", "Enter Manually"],
        help="Upload your bill for auto-extraction"
    )

    monthly_units = None
    annual_units = None

    if input_method == "Upload Bill PDF":
        uploaded_bill = st.file_uploader(
            "Upload Electricity Bill (PDF)",
            type=["pdf"],
            help="We'll automatically extract your monthly units"
        )
        if uploaded_bill:
            with st.spinner("Reading your bill..."):
                extracted_units, preview = extract_units_from_pdf(uploaded_bill)
            if extracted_units:
                st.success(f"✅ Extracted: {extracted_units} units this month")
                monthly_units = extracted_units
                annual_units = monthly_units * 12
                st.info(f"Estimated annual consumption: **{annual_units:,.0f} kWh**")
            else:
                st.warning("⚠️ Could not auto-extract. Please enter manually below.")
                monthly_units = st.number_input(
                    "Monthly Units (kWh)", min_value=50, max_value=50000, value=300
                )
                annual_units = monthly_units * 12
    else:
        col1, col2 = st.columns(2)
        with col1:
            monthly_units = st.number_input(
                "Monthly Units (kWh)",
                min_value=50, max_value=50000, value=300,
                help="Average monthly electricity consumption"
            )
        with col2:
            annual_units = monthly_units * 12
        st.info(f"Annual estimate: **{annual_units:,.0f} kWh/year**")

    st.markdown("---")
    st.markdown("## 🏠 Rooftop Dimensions")
    st.caption("Used to calculate installable panel capacity")

    col1, col2 = st.columns(2)
    with col1:
        roof_length = st.number_input("Length (ft)", min_value=5, max_value=500, value=40)
    with col2:
        roof_width = st.number_input("Width (ft)", min_value=5, max_value=500, value=30)

    usable_area_preview = roof_length * roof_width * 0.70
    max_kw_preview = usable_area_preview / 100
    st.caption(f"Usable area: ~{usable_area_preview:.0f} sq ft → max ~{max_kw_preview:.1f} kW installable")

    st.markdown("---")
    st.markdown("## 💰 Optional: Budget")
    has_budget = st.checkbox("Set a budget limit")
    budget = None
    if has_budget:
        budget = st.number_input(
            "Budget (₹)", min_value=10000, max_value=5000000,
            value=200000, step=10000,
            format="%d"
        )

    analyze_btn = st.button("⚡ Analyze Solar Potential", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN RESULTS
# ─────────────────────────────────────────────
if analyze_btn and annual_units:
    with st.spinner("Calculating your solar plan..."):
        scenarios, sun_hours, daytime_fraction, usable_area, max_kw = calculate_solar_plan(
            annual_units=annual_units,
            city=city,
            building_type=building_type,
            roof_length=roof_length,
            roof_width=roof_width,
            tariff=tariff,
            budget=budget
        )

    # Recommended scenario = Optimal ROI
    rec = scenarios["Optimal ROI"]

    # Recommendation verdict
    verdict_key, verdict_label, verdict_text = get_recommendation(
        daytime_fraction=daytime_fraction,
        payback_years=rec["payback"],
        offset_pct=rec["offset_pct"],
        roi_pct=rec["roi_pct"]
    )

    verdict_class = {"highly_recommended": "rec-green", "recommended": "rec-amber",
                     "low_benefit": "rec-red"}[verdict_key]
    verdict_icon = {"highly_recommended": "✅", "recommended": "⚠️", "low_benefit": "❌"}[verdict_key]

    st.markdown(f"""
    <div class="{verdict_class}">
        <h3>{verdict_icon} {verdict_label}</h3>
        <p style="margin:0;">{verdict_text}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics
    st.markdown("### 📊 Your Recommended Plan (Optimal ROI)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h2 style="color:#FF8C00;">{rec['kw']} kW</h2>
            <p>System Size</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h2 style="color:#FF8C00;">{rec['panel_count']}</h2>
            <p>Solar Panels</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <h2 style="color:#FF8C00;">₹{rec['net_cost']:,}</h2>
            <p>Net Cost (after subsidy)</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <h2 style="color:#FF8C00;">₹{rec['annual_net_savings']:,}</h2>
            <p>Annual Savings</p></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <h2 style="color:#FF8C00;">{rec['payback']} yrs</h2>
            <p>Payback Period</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Financial Projection", "🔄 All Scenarios", "🏠 Rooftop Analysis", "ℹ️ System Details"
    ])

    with tab1:
        st.markdown("#### 25-Year Savings vs Investment")
        import plotly.graph_objects as go

        years = list(range(0, 26))
        cumulative = [0] + rec["cumulative_25yr"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=cumulative,
            mode='lines+markers',
            name='Net Cumulative Savings (₹)',
            line=dict(color='#FF8C00', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,140,0,0.1)'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      annotation_text="Break-even point")
        fig.update_layout(
            title="25-Year Financial Projection",
            xaxis_title="Years",
            yaxis_title="Net Savings (₹)",
            yaxis_tickformat=",",
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Subsidy", f"₹{rec['subsidy']:,}")
        with col2:
            st.metric("25-Year Net Profit", f"₹{rec['total_savings_25yr']:,}")
        with col3:
            st.metric("CO₂ Offset (25 yrs)", f"{rec['co2_offset']} tonnes")

        # Annual savings bar chart
        st.markdown("#### Monthly vs Solar Generation Estimate")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        seasonal_factor = [0.85, 0.9, 1.05, 1.1, 1.1, 0.95,
                           0.85, 0.85, 0.95, 1.0, 0.95, 0.85]
        monthly_gen = [round(rec['annual_gen'] / 12 * f) for f in seasonal_factor]
        monthly_consumption = [round(annual_units / 12)] * 12

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Solar Generation (kWh)', x=months, y=monthly_gen,
            marker_color='#FFD700'
        ))
        fig2.add_trace(go.Bar(
            name='Consumption (kWh)', x=months, y=monthly_consumption,
            marker_color='#FF8C00', opacity=0.6
        ))
        fig2.update_layout(
            barmode='group', title="Monthly Solar Generation vs Consumption",
            plot_bgcolor='white', paper_bgcolor='white', height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("#### Compare All Scenarios")
        cols = st.columns(len(scenarios))
        for idx, (name, sc) in enumerate(scenarios.items()):
            with cols[idx]:
                border_color = "#FF8C00" if name == "Optimal ROI" else "#dee2e6"
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 10px;
                            padding: 1.5rem; text-align: center;">
                    <h4>{"⭐ " if name=="Optimal ROI" else ""}{name}</h4>
                    <hr/>
                    <p><b>System Size:</b> {sc['kw']} kW</p>
                    <p><b>Panels:</b> {sc['panel_count']}</p>
                    <p><b>Area Needed:</b> {sc['area_needed']} sq ft</p>
                    <p><b>Total Cost:</b> ₹{sc['capex']:,}</p>
                    <p><b>Subsidy:</b> ₹{sc['subsidy']:,}</p>
                    <p><b>Net Cost:</b> ₹{sc['net_cost']:,}</p>
                    <p><b>Annual Savings:</b> ₹{sc['annual_savings']:,}</p>
                    <p><b>Payback:</b> {sc['payback']} years</p>
                    <p><b>Energy Offset:</b> {sc['offset_pct']}%</p>
                    <p><b>ROI:</b> {sc['roi_pct']}% p.a.</p>
                    <p><b>25-yr Profit:</b> ₹{sc['total_savings_25yr']:,}</p>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 🛰️ Rooftop Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Roof Dimensions:** {roof_length} ft × {roof_width} ft
            **Total Roof Area:** {roof_length * roof_width:,} sq ft
            **Usable Area (70% rule):** {usable_area:.0f} sq ft
            **Max Installable Capacity:** {max_kw:.1f} kW
            **Panels that fit:** {int(usable_area / 22)} panels
            """)
            st.info("""
            **Why 70% usable?**
            30% is typically lost to:
            - Water tanks & overhead structures
            - AC units & vents
            - Shading from parapets
            - Safety walkways
            - Structural restrictions
            """)
        with col2:
            # Visual representation of roof
            fig3 = go.Figure()
            fig3.add_shape(type="rect",
                           x0=0, y0=0, x1=roof_width, y1=roof_length,
                           fillcolor="lightgray", line=dict(color="black", width=2))
            # Usable zone
            usable_w = roof_width * 0.85
            usable_l = roof_length * 0.85
            fig3.add_shape(type="rect",
                           x0=roof_width * 0.075, y0=roof_length * 0.075,
                           x1=roof_width * 0.925, y1=roof_length * 0.925,
                           fillcolor="rgba(255,165,0,0.4)",
                           line=dict(color="#FF8C00", width=2, dash="dot"))
            fig3.add_annotation(
                x=roof_width / 2, y=roof_length / 2,
                text=f"Usable Zone\n{usable_area:.0f} sq ft",
                showarrow=False, font=dict(size=14, color="#FF8C00")
            )
            fig3.update_layout(
                title="Rooftop Layout (Top View)",
                xaxis_title="Width (ft)", yaxis_title="Length (ft)",
                height=350, plot_bgcolor='white', paper_bgcolor='white'
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("#### System Parameters Used")
        details = {
            "City": city,
            "Peak Sun Hours": f"{sun_hours} hrs/day",
            "Building Type": building_type,
            "Daytime Usage Fraction": f"{daytime_fraction * 100:.0f}%",
            "Performance Ratio": "0.75 (industry standard)",
            "Panel Rating": "400W (Mono PERC)",
            "Area per Panel": "~22 sq ft",
            "Electricity Tariff": f"₹{tariff}/kWh",
            "Annual Tariff Escalation": "5% (assumed)",
            "Maintenance Cost": f"₹{MAINTENANCE_PER_KW_ANNUAL}/kW/year",
            "CO₂ Factor": "0.82 kg CO₂/kWh (India grid)",
        }
        for k, v in details.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{k}**")
            with col2:
                st.markdown(v)

        st.markdown("---")
        st.markdown("#### 📋 Data Sources")
        st.markdown("""
        - **Solar irradiance data:** NASA POWER (power.larc.nasa.gov)
        - **Cost benchmarks:** MNRE & BRIDGE TO INDIA (2025)
        - **Subsidy slabs:** PM Surya Ghar Muft Bijli Yojana (2024)
        - **CO₂ factor:** Central Electricity Authority, India (2024)
        - **Performance ratio:** MNRE recommended standard
        """)

    # ─── Download Button ───
    st.markdown("---")
    st.markdown("### 📥 Download Your Report")

    report_text = f"""
SolarIQ — Solar Feasibility Report
Generated: {datetime.now().strftime('%d %B %Y')}
{'=' * 50}

BUILDING DETAILS
City: {city}
Building Type: {building_type}
Annual Consumption: {annual_units:,} kWh
Electricity Tariff: ₹{tariff}/kWh
Roof Area: {roof_length} ft × {roof_width} ft

SOLAR RECOMMENDATION: {verdict_label}
{verdict_text}

RECOMMENDED PLAN (Optimal ROI)
System Size: {rec['kw']} kW
Panel Count: {rec['panel_count']} panels (400W each)
Roof Area Required: {rec['area_needed']} sq ft
Energy Offset: {rec['offset_pct']}%
Annual Generation: {rec['annual_gen']:,} kWh

FINANCIAL SUMMARY
Total System Cost: ₹{rec['capex']:,}
Subsidy Amount: ₹{rec['subsidy']:,}
Net Cost (after subsidy): ₹{rec['net_cost']:,}
Annual Maintenance: ₹{rec['annual_maintenance']:,}
Annual Bill Savings: ₹{rec['annual_savings']:,}
Annual Net Savings: ₹{rec['annual_net_savings']:,}
Payback Period: {rec['payback']} years
ROI: {rec['roi_pct']}% per annum
25-Year Net Profit: ₹{rec['total_savings_25yr']:,}
CO₂ Offset (25 yrs): {rec['co2_offset']} tonnes

TECHNICAL PARAMETERS
Peak Sun Hours: {sun_hours} hrs/day
Performance Ratio: 0.75
Daytime Usage Match: {daytime_fraction * 100:.0f}%

DATA SOURCES
- NASA POWER (solar irradiance)
- MNRE India (cost benchmarks)
- PM Surya Ghar Yojana (subsidy slabs)

Disclaimer: This is an indicative estimate. Actual values may vary
based on site conditions, shading, equipment quality, and local
regulations. Consult an MNRE-empanelled installer for detailed assessment.
"""

    st.download_button(
        label="📄 Download Report (TXT)",
        data=report_text,
        file_name=f"SolarIQ_Report_{city}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )

elif not analyze_btn:
    # Landing state
    st.markdown("""
    ### 👈 Fill in your details in the sidebar and click Analyze

    **What SolarIQ does for you:**

    | Step | What happens |
    |------|-------------|
    | 1️⃣ Upload or enter bill | We extract your consumption data |
    | 2️⃣ Enter roof dimensions | We calculate installable capacity |
    | 3️⃣ Select city & building type | We apply real irradiance data |
    | 4️⃣ Click Analyze | Get your complete solar plan in seconds |

    ---

    **Built for CSI National Hackathon 2025**
    Data sources: NASA POWER · MNRE India · PM Surya Ghar Yojana
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#fff9e6;border:1px solid #FFD700;border-radius:10px;padding:1rem;text-align:center;">
        <h3>☀️</h3><b>Location-Aware</b><br/>
        Real sun-hour data for 30+ Indian cities
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#fff9e6;border:1px solid #FFD700;border-radius:10px;padding:1rem;text-align:center;">
        <h3>💰</h3><b>India-Specific Finance</b><br/>
        PM Surya Ghar subsidy auto-applied
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff9e6;border:1px solid #FFD700;border-radius:10px;padding:1rem;text-align:center;">
        <h3>🧠</h3><b>Smart Recommendation</b><br/>
        Go / No-Go verdict with clear reasoning
        </div>""", unsafe_allow_html=True)
