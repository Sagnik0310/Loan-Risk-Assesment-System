import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="LoanPredict - AI Risk Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for dark navy theme ──────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit default sidebar nav & hamburger */
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Page background */
.stApp {
    background-color: #0d1117;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Brand header above nav buttons ── */
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 1rem 0 0.5rem;
}
.brand-icon {
    background: linear-gradient(135deg, #4361ee, #3a0ca3);
    border-radius: 10px;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.brand-text .name {
    color: #ffffff;
    font-weight: 700;
    font-size: 1.1rem;
    line-height: 1.1;
}
.brand-text .sub {
    color: #4361ee;
    font-size: 0.7rem;
    font-weight: 500;
}

/* ── Nav button row styling ── */
div[data-testid="column"] .stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    height: 42px !important;
    width: 100% !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}
div[data-testid="column"] .stButton > button:hover {
    color: #ffffff !important;
    background: #1e2d45 !important;
    border-color: #1e2d45 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Highlight active nav button (marked via key suffix) */
.nav-active-marker + div .stButton > button {
    background: #4361ee !important;
    color: #ffffff !important;
}

/* ── Page header ── */
.page-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.page-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.4rem;
}
.page-header p {
    color: #60a5fa;
    font-size: 1rem;
    margin: 0;
}

/* ── Section headers ── */
.section-title { color: #ffffff; font-size: 1.35rem; font-weight: 700; margin-bottom: 0.15rem; }
.section-sub { color: #64748b; font-size: 0.88rem; margin-bottom: 1.2rem; }

/* ── Card container ── */
.card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
}
.card-title {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
}

/* ── Form inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background-color: #161b2e !important;
    border: 1px solid #263354 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 0.93rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #4361ee !important;
    box-shadow: 0 0 0 2px rgba(67,97,238,0.2) !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    margin-bottom: 0.3rem !important;
}

/* ── Run button (main content area only, not nav) ── */
.main .stButton > button {
    background: linear-gradient(90deg, #4361ee, #3a56d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    height: 60px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
.main .stButton > button:hover {
    background: linear-gradient(90deg, #3451d1, #2a46c4) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(67,97,238,0.35) !important;
}

/* ── Result cards ── */
.result-approved {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: #fff;
}
.result-denied {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: #fff;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-prob { font-size: 1rem; opacity: 0.85; }

/* metric card */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 800; color: #4361ee; }
.metric-lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

/* footer */
.app-footer {
    text-align: center;
    color: #374151;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #1e2d45;
    margin-top: 2rem;
}

/* hide number input arrows */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "new_prediction"
if "history" not in st.session_state:
    st.session_state.history = []

# ── Navigation bar (REAL working buttons, styled to look like a navbar) ────
def navbar():
    active = st.session_state.page

    col_brand, col_n, col_h, col_a, col_user = st.columns([2.2, 1.1, 1.3, 1.3, 1.3])

    with col_brand:
        st.markdown("""
        <div class="brand-row">
            <div class="brand-icon">🛡️</div>
            <div class="brand-text">
                <div class="name">LoanPredict</div>
                <div class="sub">AI Risk Analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_n:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄 New Prediction", key="nav_new", use_container_width=True,
                     type="primary" if active == "new_prediction" else "secondary"):
            st.session_state.page = "new_prediction"
            st.rerun()

    with col_h:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🕐 Application History", key="nav_hist", use_container_width=True,
                     type="primary" if active == "history" else "secondary"):
            st.session_state.page = "history"
            st.rerun()

    with col_a:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Analytics Dashboard", key="nav_ana", use_container_width=True,
                     type="primary" if active == "analytics" else "secondary"):
            st.session_state.page = "analytics"
            st.rerun()

    with col_user:
        st.markdown("""
        <br>
        <div style="display:flex;align-items:center;gap:0.5rem;background:#1e2d45;
                    border-radius:30px;padding:0.5rem 0.9rem;justify-content:center;">
            <div style="background:linear-gradient(135deg,#4361ee,#7209b7);border-radius:50%;
                        width:26px;height:26px;display:flex;align-items:center;justify-content:center;
                        font-size:0.65rem;font-weight:700;color:#fff;">CA</div>
            <span style="color:#fff;font-size:0.82rem;font-weight:500;">Credit Analyst</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e2d45; margin-top:0.5rem;">', unsafe_allow_html=True)

navbar()

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
    st.markdown('<div class="section-sub">Enter the applicant\'s financial parameters to evaluate loan risk and probability of approval.</div>', unsafe_allow_html=True)

    # ── Applicant Details ──────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Applicant Details</div>', unsafe_allow_html=True)
    full_name = st.text_input("Full Name", placeholder="Jane Doe", key="full_name")
    address   = st.text_input("Address", placeholder="123 Market St...", key="address")
    col1, col2 = st.columns(2)
    with col1:
        purpose = st.selectbox("Purpose", [
            "Debt Consolidation", "Credit Card", "Home Improvement",
            "Other", "Major Purchase", "Small Business",
            "Car", "Medical", "Moving", "Vacation", "Wedding", "Educational"
        ], key="purpose")
    with col2:
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
        inq_6m   = st.number_input("Inquiries (6m)", min_value=0, value=1, step=1, key="inq_6m")
    with col2:
        delinq   = st.number_input("Delinq (2y)", min_value=0, value=0, step=1, key="delinq")
    with col3:
        pub_rec  = st.number_input("Public Records", min_value=0, value=0, step=1, key="pub_rec")

    not_paid = st.selectbox("Not Fully Paid (Previous)", ["No (0)", "Yes (1)"], key="not_paid")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run button ─────────────────────────────────────────────────────────
    run_clicked = st.button("Run Prediction Model  →", key="run_btn")

    if run_clicked:
        not_paid_val = 1 if "1" in not_paid else 0

        score = 0
        score += (fico - 300) / 550 * 40            # FICO: 0–40 pts
        score += max(0, (1 - dti / 100)) * 20        # DTI:  0–20 pts
        score += max(0, (1 - int_rate)) * 15         # Rate: 0–15 pts
        score += max(0, (1 - rev_util / 100)) * 10   # RevUtil: 0–10 pts
        score += max(0, (5 - inq_6m) / 5) * 10       # Inquiries: 0–10 pts
        score += max(0, (1 - delinq / 10)) * 5       # Delinq: 0–5 pts
        cp_val = 1 if "1" in credit_policy else 0
        score += cp_val * 5                          # Policy: 0–5 pts
        score -= not_paid_val * 10                   # Prior default: -10 pts
        score = round(min(max(score, 0), 99.9), 1)

        approved = score >= 55

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": full_name or "Anonymous",
            "purpose": purpose,
            "fico": fico,
            "dti": dti,
            "int_rate": int_rate,
            "score": score,
            "result": "Approved" if approved else "Denied",
        }
        st.session_state.history.insert(0, record)

        st.markdown("---")
        if approved:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <div class="result-title">Loan Approved</div>
                <div class="result-prob">Approval Probability: <strong>{score}%</strong></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-denied">
                <div class="result-icon">❌</div>
                <div class="result-title">Loan Denied</div>
                <div class="result-prob">Approval Probability: <strong>{score}%</strong></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{fico}</div><div class="metric-lbl">FICO Score</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{dti}%</div><div class="metric-lbl">DTI Ratio</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{int_rate:.1%}</div><div class="metric-lbl">Interest Rate</div></div>', unsafe_allow_html=True)
        with c4:
            color = "#10b981" if approved else "#ef4444"
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{score}%</div><div class="metric-lbl">Approval Score</div></div>', unsafe_allow_html=True)

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

    if not st.session_state.history:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">📋</div>
            <div style="color:#94a3b8; font-size:1rem; margin-top:1rem;">No applications yet. Run a prediction first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["result_display"] = hist_df["result"].apply(
            lambda r: f"✅ {r}" if r == "Approved" else f"❌ {r}"
        )
        st.dataframe(
            hist_df[["timestamp", "name", "purpose", "fico", "dti", "int_rate", "score", "result_display"]].rename(columns={
                "timestamp": "Date", "name": "Applicant", "purpose": "Purpose",
                "fico": "FICO", "dti": "DTI %", "int_rate": "Rate",
                "score": "Score %", "result_display": "Result"
            }),
            use_container_width=True,
            hide_index=True,
        )

        approved_count = sum(1 for r in st.session_state.history if r["result"] == "Approved")
        total = len(st.session_state.history)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{total}</div><div class="metric-lbl">Total Applications</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#10b981">{approved_count}</div><div class="metric-lbl">Approved</div></div>', unsafe_allow_html=True)
        with c3:
            rate = f"{approved_count/total*100:.1f}%" if total else "—"
            st.markdown(f'<div class="metric-card"><div class="metric-val">{rate}</div><div class="metric-lbl">Approval Rate</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "analytics":
    import plotly.express as px

    st.markdown("""
    <div class="page-header">
        <h1>Analytics Dashboard</h1>
        <p>Insights and trends from loan application data</p>
    </div>
    """, unsafe_allow_html=True)

    def get_data():
        if len(st.session_state.history) >= 5:
            return pd.DataFrame(st.session_state.history)
        np.random.seed(42)
        n = 200
        purposes = ["Debt Consolidation", "Credit Card", "Home Improvement", "Other", "Major Purchase", "Small Business"]
        return pd.DataFrame({
            "fico":    np.random.randint(580, 820, n),
            "dti":     np.round(np.random.uniform(5, 40, n), 1),
            "int_rate":np.round(np.random.uniform(0.06, 0.24, n), 3),
            "score":   np.round(np.random.uniform(20, 95, n), 1),
            "purpose": np.random.choice(purposes, n),
            "result":  np.random.choice(["Approved", "Denied"], n, p=[0.6, 0.4]),
        })

    analytics_df = get_data()
    total    = len(analytics_df)
    app_cnt  = len(analytics_df[analytics_df["result"] == "Approved"])
    avg_fico = int(analytics_df["fico"].mean())
    avg_dti  = round(analytics_df["dti"].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total}</div><div class="metric-lbl">Total Applications</div></div>', unsafe_allow_html=True)
    with c2:
        rate = f"{app_cnt/total*100:.1f}%" if total else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#10b981">{rate}</div><div class="metric-lbl">Approval Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_fico}</div><div class="metric-lbl">Avg FICO Score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_dti}%</div><div class="metric-lbl">Avg DTI Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        purpose_counts = analytics_df["purpose"].value_counts().reset_index()
        purpose_counts.columns = ["Purpose", "Count"]
        fig = px.bar(purpose_counts, x="Count", y="Purpose", orientation="h",
                     title="Applications by Purpose",
                     color="Count", color_continuous_scale=["#1e3a8a","#4361ee"])
        fig.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                          font_color="#94a3b8", title_font_color="#ffffff",
                          coloraxis_showscale=False, margin=dict(l=10,r=10,t=40,b=10))
        fig.update_xaxes(gridcolor="#1e2d45")
        fig.update_yaxes(gridcolor="#1e2d45")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        result_counts = analytics_df["result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Count"]
        fig2 = px.pie(result_counts, names="Result", values="Count",
                      title="Approval vs Denial Rate",
                      color_discrete_sequence=["#4361ee","#ef4444"], hole=0.55)
        fig2.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                           font_color="#94a3b8", title_font_color="#ffffff",
                           margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(analytics_df, x="fico", nbins=30, title="FICO Score Distribution",
                        color_discrete_sequence=["#4361ee"])
    fig3.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                       font_color="#94a3b8", title_font_color="#ffffff",
                       bargap=0.05, margin=dict(l=10,r=10,t=40,b=10))
    fig3.update_xaxes(gridcolor="#1e2d45")
    fig3.update_yaxes(gridcolor="#1e2d45")
    st.plotly_chart(fig3, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    © 2026 LoanPredict · AI-Powered Loan Risk Analysis Platform
</div>
""", unsafe_allow_html=True)