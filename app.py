import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from structure.prediction import LoanPrediction
from database.prediction_history import PredictionHistory

st.set_page_config(
    page_title="LoanPredict - AI Risk Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

predictor = LoanPrediction()
history = PredictionHistory()

# ── Session state defaults ─────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "new_prediction"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

# ── Welcome popup (shown once on first load) ───────────────────────────────
@st.dialog(" ")
def show_welcome():
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem 0.5rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🏦</div>
        <div style="font-size: 1.7rem; font-weight: 800; margin-bottom: 0.3rem;">LoanPredict</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #4361ee; margin-bottom: 1.4rem;">
            Welcome to the Loan Risk Assessment System
        </div>
        <div style="font-size: 0.92rem; line-height: 1.7; color: #94a3b8; margin-bottom: 1.8rem; text-align: left; padding: 0 0.5rem;">
            This intelligent application helps financial institutions evaluate loan applications
            by predicting the likelihood of loan approval and the applicant's risk of default.
            The prediction is powered by machine learning models trained on historical loan data.
        </div>
        <hr style="border-color: #1e2d45; margin-bottom: 1.2rem;">
        <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 0.4rem;">
            © 2026 LoanPredict
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.3rem;">
            👨‍💻 Founded &amp; Developed by
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #c7d2fe; margin-bottom: 0.6rem;">
            Sagnik Ghosh &amp; Anantika Ghosh
        </div>
        <div style="font-size: 0.75rem; color: #4b5563;">Version 1.0.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Get Started →", use_container_width=True, type="primary"):
        st.session_state.welcome_shown = True
        st.rerun()

if not st.session_state.welcome_shown:
    show_welcome()

# ── Theme colours ──────────────────────────────────────────────────────────
dark = {
    "app_bg":        "#0d1117",
    "navbar_bg":     "#101827",
    "navbar_border": "#1e2d45",
    "card_bg":       "#111827",
    "card_border":   "#1e2d45",
    "input_bg":      "#161b2e",
    "input_border":  "#263354",
    "text":          "#ffffff",
    "subtext":       "#94a3b8",
    "muted":         "#64748b",
    "accent":        "#4361ee",
    "footer_text":   "#374151",
    "header_sub":    "#60a5fa",
    "section_sub":   "#64748b",
    "hr":            "#1e2d45",
}
light = {
    "app_bg":        "#eef2ff",
    "navbar_bg":     "#dce8fd",
    "navbar_border": "#b8cef7",
    "card_bg":       "#f5f8ff",
    "card_border":   "#c5d8fb",
    "input_bg":      "#e4eeff",
    "input_border":  "#a8c3f5",
    "text":          "#0f172a",
    "subtext":       "#334155",
    "muted":         "#475569",
    "accent":        "#4361ee",
    "footer_text":   "#64748b",
    "header_sub":    "#2563eb",
    "section_sub":   "#475569",
    "hr":            "#b8cef7",
}

T = dark if st.session_state.dark_mode else light

# ── Inject global CSS (theme-aware) ───────────────────────────────────────
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.stApp {{
    background-color: {T['app_bg']};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}

/* ── Navbar ── */
.navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: {T['navbar_bg']};
    border-bottom: 1px solid {T['navbar_border']};
    padding: 0 2rem;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 999;
    margin: -4rem -4rem 0 -4rem;
}}
.navbar-brand {{ display: flex; align-items: center; gap: 0.6rem; }}
.brand-icon {{
    background: linear-gradient(135deg, #4361ee, #3a0ca3);
    border-radius: 10px; width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}}
.brand-text .name {{ color: {T['text']}; font-weight: 700; font-size: 1.1rem; line-height: 1.1; }}
.brand-text .sub  {{ color: #4361ee; font-size: 0.7rem; font-weight: 500; }}
.navbar-links {{ display: flex; align-items: center; gap: 0.25rem; }}
.nav-link {{
    color: {T['subtext']}; padding: 0.45rem 1rem; border-radius: 8px;
    font-size: 0.88rem; font-weight: 500; cursor: pointer;
    text-decoration: none; transition: all 0.15s;
    display: inline-flex; align-items: center; gap: 0.4rem; white-space: nowrap;
}}
.nav-link:hover {{ color: {T['text']}; background: {T['card_border']}; }}
.nav-link.active {{ background: #4361ee; color: #ffffff; }}
.nav-user {{
    display: flex; align-items: center; gap: 0.5rem;
    background: {T['card_border']}; border-radius: 30px;
    padding: 0.35rem 0.9rem 0.35rem 0.4rem; cursor: pointer;
}}
.avatar {{
    background: linear-gradient(135deg, #4361ee, #7209b7);
    border-radius: 50%; width: 30px; height: 30px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: #fff;
}}
.user-name {{ color: {T['text']}; font-size: 0.85rem; font-weight: 500; }}

/* ── Nav tab row (real page-switch buttons, docked under the brand bar) ── */
.st-key-navbar_buttons {{
    background-color: {T['navbar_bg']};
    border-bottom: 1px solid {T['navbar_border']};
    margin: 0 -4rem 2rem -4rem;
    padding: 0.5rem 2rem 0.6rem;
    position: sticky;
    top: 64px;
    z-index: 998;
}}
.st-key-navbar_buttons [data-testid="stHorizontalBlock"] {{
    gap: 0.25rem;
    max-width: 620px;
}}
.st-key-navbar_buttons .stButton {{ margin: 0 !important; }}
.st-key-navbar_buttons .stButton > button {{
    background: transparent !important;
    color: {T['subtext']} !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 8px !important;
    height: auto !important;
    width: auto !important;
    padding: 0.5rem 1.1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: normal !important;
    transform: none !important;
}}
.st-key-navbar_buttons .stButton > button:hover {{
    background: {T['card_border']} !important;
    color: {T['text']} !important;
    transform: none !important;
    box-shadow: none !important;
}}
.st-key-navbar_buttons .stButton > button[kind="primary"] {{
    background: #4361ee !important;
    color: #ffffff !important;
}}
.st-key-navbar_buttons .stButton > button[kind="primary"]:hover {{
    background: linear-gradient(90deg, #3451d1, #2a46c4) !important;
    color: #ffffff !important;
}}

/* ── Page header ── */
.page-header {{
    text-align: center; padding: 2.5rem 0 2rem;
    border-bottom: 1px solid {T['hr']}; margin-bottom: 2rem;
}}
.page-header h1 {{ color: {T['text']}; font-size: 2.4rem; font-weight: 800; margin: 0 0 0.4rem; }}
.page-header p  {{ color: {T['header_sub']}; font-size: 1rem; margin: 0; }}

/* ── Section text ── */
.section-title {{ color: {T['text']}; font-size: 1.35rem; font-weight: 700; margin-bottom: 0.15rem; }}
.section-sub   {{ color: {T['section_sub']}; font-size: 0.88rem; margin-bottom: 1.2rem; }}

/* ── Cards ── */
.card {{
    background: {T['card_bg']}; border: 1px solid {T['card_border']};
    border-radius: 14px; padding: 1.6rem 1.8rem; margin-bottom: 1.5rem;
}}
.card-title {{ color: {T['text']}; font-size: 1.05rem; font-weight: 700; margin-bottom: 1.2rem; }}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {{
    background-color: {T['input_bg']} !important;
    border: 1px solid {T['input_border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-size: 0.93rem !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: #4361ee !important;
    box-shadow: 0 0 0 2px rgba(67,97,238,0.2) !important;
}}
.stTextInput label, .stNumberInput label, .stSelectbox label {{
    color: {T['text']} !important;
    font-size: 0.88rem !important; font-weight: 500 !important;
}}

/* ── Run button ── */
.stButton > button {{
    background: linear-gradient(90deg, #4361ee, #3a56d4) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    padding: 0.85rem 2rem !important; width: 100% !important;
    height: 60px !important; cursor: pointer !important;
    transition: all 0.2s !important; letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background: linear-gradient(90deg, #3451d1, #2a46c4) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(67,97,238,0.35) !important;
}}

/* ── Result cards ── */
.result-approved {{
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981; border-radius: 14px;
    padding: 2rem; text-align: center; color: #fff;
}}
.result-denied {{
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444; border-radius: 14px;
    padding: 2rem; text-align: center; color: #fff;
}}
.result-icon  {{ font-size: 3rem; margin-bottom: 0.5rem; }}
.result-title {{ font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }}
.result-prob  {{ font-size: 1rem; opacity: 0.85; }}

/* ── Metric cards ── */
.metric-card {{
    background: {T['card_bg']}; border: 1px solid {T['card_border']};
    border-radius: 10px; padding: 1.1rem 1.4rem; text-align: center;
}}
.metric-val {{ font-size: 1.8rem; font-weight: 800; color: #4361ee; }}
.metric-lbl {{ font-size: 0.78rem; color: {T['muted']}; margin-top: 0.2rem; }}

/* ── Analytics tables ── */
.analytics-table {{
    background: {T['card_bg']}; border: 1px solid {T['card_border']};
    border-radius: 12px; overflow: hidden; margin-bottom: 1.2rem;
}}
.analytics-table table {{ width: 100%; border-collapse: collapse; }}
.analytics-table th {{
    background: {T['input_bg']}; color: {T['subtext']};
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; padding: 0.75rem 1rem; text-align: left;
    border-bottom: 1px solid {T['card_border']};
}}
.analytics-table td {{
    color: {T['text']}; font-size: 0.88rem;
    padding: 0.7rem 1rem; border-bottom: 1px solid {T['card_border']};
}}
.analytics-table tr:last-child td {{ border-bottom: none; }}
.analytics-table tr:hover td {{ background: {T['input_bg']}; }}

/* ── Footer ── */
.app-footer {{
    text-align: center; color: {T['footer_text']}; font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem; border-top: 1px solid {T['hr']}; margin-top: 2rem;
}}
.version-badge {{
    display: inline-block; background: {T['input_bg']};
    border: 1px solid {T['card_border']}; border-radius: 20px;
    padding: 0.2rem 0.75rem; font-size: 0.72rem; color: {T['muted']};
    margin-top: 0.4rem;
}}

input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {{ -webkit-appearance: none; }}
</style>
""", unsafe_allow_html=True)

# ── Navbar HTML ────────────────────────────────────────────────────────────
active = st.session_state.page
st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">
        <div class="brand-icon">🛡️</div>
        <div class="brand-text">
            <div class="name">LoanPredict</div>
            <div class="sub">AI Risk Analysis</div>
        </div>
    </div>
    <div class="navbar-links"></div>
    <div class="nav-user">
        <div class="avatar">CA</div>
        <span class="user-name">Credit Analyst</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons are wrapped in a keyed container so the CSS above (.st-key-navbar_buttons)
# can dock them visually to the brand bar and restyle them as nav pills, without
# affecting any other button (Run Prediction, Get Started, etc.) on the page.
with st.container(key="navbar_buttons"):
    col1, col2, col3 = st.columns([3, 3, 3])

    with col1:
        if st.button(
            "📄 New Prediction",
            use_container_width=True,
            type="primary" if st.session_state.page == "new_prediction" else "secondary",
        ):
            st.session_state.page = "new_prediction"
            st.rerun()

    with col2:
        if st.button(
            "🕐 Application History",
            use_container_width=True,
            type="primary" if st.session_state.page == "history" else "secondary",
        ):
            st.session_state.page = "history"
            st.rerun()

    with col3:
        if st.button(
            "📊 Analytics Dashboard",
            use_container_width=True,
            type="primary" if st.session_state.page == "analytics" else "secondary",
        ):
            st.session_state.page = "analytics"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: NEW PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.page == "new_prediction":

    st.markdown("""
    <div class="page-header">
        <h1>Loan Approval Predictor</h1>
        <p>Professional AI-powered loan analysis dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">New Prediction</div>', unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">Enter the applicant's financial parameters to evaluate loan risk and probability of approval.</div>", unsafe_allow_html=True)

    # ── Applicant Details ──────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Applicant Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", placeholder="Jane Doe", key="full_name")
    with col2:
        email = st.text_input("Email Address", placeholder="jane.doe@example.com", key="email")
    col3, col4 = st.columns(2)
    with col3:
        phone = st.text_input("Phone Number", placeholder="+1 (555) 000-0000", key="phone")
    with col4:
        address = st.text_input("Address", placeholder="123 Market St...", key="address")
    col5, col6 = st.columns(2)
    with col5:
        purpose = st.selectbox("Purpose", [
            "Debt Consolidation", "Credit Card", "Home Improvement",
            "Other", "Major Purchase", "Small Business",
            "Car", "Medical", "Moving", "Vacation", "Wedding", "Educational"
        ], key="purpose")
    with col6:
        credit_policy = st.selectbox("Credit Policy", [
            "Meets Policy (1)", "Does Not Meet Policy (0)"
        ], key="credit_policy")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Loan Parameters ────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Loan Parameters</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        int_rate   = st.number_input("Interest Rate (e.g. 0.15)", min_value=0.0, max_value=1.0, value=0.11, step=0.001, format="%.3f", key="int_rate")
        log_income = st.number_input("Log Annual Income", min_value=0.0, max_value=20.0, value=11.0, step=0.1, format="%.2f", key="log_income")
    with col2:
        installment = st.number_input("Monthly Installment", min_value=0.0, value=300.0, step=10.0, format="%.2f", key="installment")
        dti         = st.number_input("DTI Ratio (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1, format="%.2f", key="dti")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Credit Profile ─────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Credit Profile</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fico    = st.number_input("FICO Score", min_value=300, max_value=850, value=700, step=1, key="fico")
        rev_bal = st.number_input("Revolving Balance", min_value=0.0, value=10000.0, step=100.0, format="%.2f", key="rev_bal")
    with col2:
        days_cr  = st.number_input("Days with Credit Line", min_value=0.0, value=3500.0, step=10.0, format="%.1f", key="days_cr")
        rev_util = st.number_input("Revolving Util (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1, format="%.1f", key="rev_util")
    col1, col2, col3 = st.columns(3)
    with col1:
        inq_6m  = st.number_input("Inquiries (6m)", min_value=0, value=1, step=1, key="inq_6m")
    with col2:
        delinq  = st.number_input("Delinq (2y)", min_value=0, value=0, step=1, key="delinq")
    with col3:
        pub_rec = st.number_input("Public Records", min_value=0, value=0, step=1, key="pub_rec")
    not_paid = st.selectbox("Not Fully Paid (Previous)", ["No (0)", "Yes (1)"], key="not_paid")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run button ─────────────────────────────────────────────────────────
    run_clicked = st.button("Run Prediction Model  →", key="run_btn")

    if run_clicked:
        applicant = {
            "name": full_name,
            "address": address,
            "phone": phone,
            "email": email
        }

        loan_data = {
            "credit.policy": 1 if "1" in credit_policy else 0,
            "purpose": purpose.lower().replace(" ", "_"),
            "int.rate": int_rate,
            "installment": installment,
            "log.annual.inc": log_income,
            "dti": dti,
            "fico": fico,
            "days.with.cr.line": days_cr,
            "revol.bal": rev_bal,
            "revol.util": rev_util,
            "inq.last.6mths": inq_6m,
            "delinq.2yrs": delinq,
            "pub.rec": pub_rec
        }

        result = predictor.predict(loan_data)

        history.save_prediction(
            applicant,
            loan_data,
            result
        )

        prediction = result["Prediction"]
        probability = result["Probability"] * 100

        approved = prediction == 0

        st.markdown("---")
        if approved:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <div class="result-title">Low Risk</div>
                <div class="result-prob">
                    Probability: <strong>{probability:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div class="result-denied">
                    <div class="result-icon">❌</div>
                    <div class="result-title">High Risk</div>
                    <div class="result-prob">
                        Probability: <strong>{probability:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl, color in [
            (c1, fico, "FICO Score", "#4361ee"),
            (c2, f"{dti:.1f}%", "DTI Ratio", "#4361ee"),
            (c3, f"{int_rate:.3f}", "Interest Rate", "#4361ee"),
            (
                c4,
                f"{probability:.2f}%",
                "Prediction Probability",
                "#10b981" if approved else "#ef4444",
            ),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: APPLICATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "history":

    st.markdown("""
    <div class="page-header">
        <h1>Application History</h1>
        <p>Review all past loan applications and their outcomes</p>
    </div>
    """, unsafe_allow_html=True)

    records = history.get_history()

    if not records:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">📋</div>
            <div style="color:{T['subtext']}; font-size:1rem; margin-top:1rem;">
                No applications yet. Run a prediction first.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(records)
        df["result"] = df["prediction"].map({
            0: "Approved",
            1: "Denied"
        })
        df["result_display"] = df["result"].apply(lambda r: f"✅ {r}" if r == "Approved" else f"❌ {r}")
        display_cols = {
            "timestamp": "Date",
            "name": "Applicant",
            "email": "Email",
            "phone": "Phone",
            "purpose": "Purpose",
            "fico": "FICO",
            "dti": "DTI %",
            "int.rate": "Rate",
            "probability": "Probability",
            "result_display": "Result"
        }
        st.dataframe(
            df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols),
            use_container_width=True, hide_index=True,
        )
        approved_count = len(df[df["prediction"] == 0])
        total = len(df)
        c1, c2, c3 = st.columns(3)
        for col, val, lbl, color in [
            (c1, total,          "Total Applications", "#4361ee"),
            (c2, approved_count, "Approved",           "#10b981"),
            (c3, f"{approved_count/total*100:.1f}%" if total else "—", "Approval Rate", "#4361ee"),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD  (text + tables only, real input data)
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "analytics":

    st.markdown("""
    <div class="page-header">
        <h1>Analytics Dashboard</h1>
        <p>Statistical summary of submitted loan applications</p>
    </div>
    """, unsafe_allow_html=True)

    records = history.get_history()

    if not records:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">📊</div>
            <div style="color:{T['subtext']}; font-size:1rem; margin-top:1rem;">
                No data yet. Submit at least one prediction to see analytics.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(records)
        df["result"] = df["prediction"].map({
            0: "Approved",
            1: "Denied"
        })

        df["int_rate"] = df["int.rate"]
        df["log_income"] = df["log.annual.inc"]
        df["days_cr"] = df["days.with.cr.line"]
        df["rev_bal"] = df["revol.bal"]
        df["rev_util"] = df["revol.util"]
        df["inq_6m"] = df["inq.last.6mths"]
        df["delinq"] = df["delinq.2yrs"]
        df["pub_rec"] = df["pub.rec"]
        total = len(df)
        approved = len(df[df["prediction"] == 0])
        denied = len(df[df["prediction"] == 1])

        # KPI row
        c1,c2,c3,c4,c5 = st.columns(5)
        for col,(val,lbl,clr) in zip([c1,c2,c3,c4,c5],[
            (total,    "Total Submitted",  "#4361ee"),
            (approved, "Approved",         "#10b981"),
            (denied,   "Denied",           "#ef4444"),
            (f"{approved/total*100:.1f}%","Approval Rate","#4361ee"),
            (f"{df['probability'].mean()*100:.2f}%","Avg Risk Score","#4361ee"),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{clr}">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Financial averages table
        st.markdown('<div class="card"><div class="card-title">📈 Financial Averages</div>', unsafe_allow_html=True)
        rows = [
            ("Average FICO Score",          f"{df['fico'].mean():.0f}"),
            ("Average DTI Ratio",           f"{df['dti'].mean():.1f}%"),
            ("Average Interest Rate",       f"{df['int_rate'].mean():.3f}"),
            ("Average Log Annual Income",   f"{df['log_income'].mean():.2f}"),
            ("Average Monthly Installment", f"${df['installment'].mean():.2f}"),
            ("Average Revolving Balance",   f"${df['rev_bal'].mean():.2f}"),
            ("Average Revolving Util",      f"{df['rev_util'].mean():.1f}%"),
        ]
        st.markdown(f"""
        <div class="analytics-table"><table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{"".join(f"<tr><td>{l}</td><td><strong>{v}</strong></td></tr>" for l,v in rows)}</tbody>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Purpose breakdown
        st.markdown('<div class="card"><div class="card-title">🎯 Applications by Purpose</div>', unsafe_allow_html=True)
        pc = df["purpose"].value_counts().reset_index()
        pc.columns = ["Purpose","Count"]
        p_rows = "".join(
            f"<tr><td>{r['Purpose']}</td><td>{r['Count']}</td>"
            f"<td>{r['Count']/total*100:.1f}%</td>"
            f"<td>{len(df[(df['purpose']==r['Purpose'])&(df['result']=='Approved')])} approved / "
            f"{len(df[(df['purpose']==r['Purpose'])&(df['result']=='Denied')])} denied</td></tr>"
            for _,r in pc.iterrows()
        )
        st.markdown(f"""
        <div class="analytics-table"><table>
            <thead><tr><th>Purpose</th><th>Count</th><th>Share</th><th>Outcome</th></tr></thead>
            <tbody>{p_rows}</tbody>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Credit risk indicators
        st.markdown('<div class="card"><div class="card-title">⚠️ Credit Risk Indicators</div>', unsafe_allow_html=True)
        risk_rows = [
            ("High Inquiries (≥3 in 6m)", int((df["inq_6m"] >= 3).sum())),
            ("Delinquencies (2y)", int((df["delinq"] > 0).sum())),
            ("Public Records", int((df["pub_rec"] > 0).sum())),
        ]
        st.markdown(f"""
        <div class="analytics-table"><table>
            <thead><tr><th>Risk Factor</th><th>Count</th><th>% of Applications</th></tr></thead>
            <tbody>{"".join(f"<tr><td>{l}</td><td>{c}</td><td>{c/total*100:.1f}%</td></tr>" for l,c in risk_rows)}</tbody>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent 10
        st.markdown('<div class="card"><div class="card-title">🕐 Recent Applications (Last 10)</div>', unsafe_allow_html=True)
        rec_rows = "".join(
            f"<tr><td>{r['timestamp']}</td><td>{r['name']}</td><td>{r['purpose']}</td>"
            f"<td>{r['fico']}</td><td>{r['probability']*100:.2f}%</td>"
            f"<td style='color:{'#10b981' if r['result']=='Approved' else '#ef4444'};font-weight:600'>"
            f"{'✅' if r['result']=='Approved' else '❌'} {r['result']}</td></tr>"
            for _,r in df.head(10).iterrows()
        )
        st.markdown(f"""
        <div class="analytics-table"><table>
            <thead><tr><th>Date</th><th>Applicant</th><th>Purpose</th><th>FICO</th><th>Score</th><th>Result</th></tr></thead>
            <tbody>{rec_rows}</tbody>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
    © 2026 LoanPredict · AI-Powered Loan Risk Analysis Platform<br>
    👨‍💻 Founded &amp; Developed by <strong>Sagnik Ghosh &amp; Anantika Ghosh</strong>
    <div class="version-badge">Version 1.0.0</div>
</div>
""", unsafe_allow_html=True)