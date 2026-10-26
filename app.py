"""
LoanPredict — Enterprise Credit Risk Assessment Platform
Main Streamlit application entry point.
"""

import streamlit as st

from ui.components import render_footer, render_sidebar
from ui.pages import (
    render_analytics_page,
    render_history_page,
    render_model_insights_page,
    render_new_assessment_page,
    render_overview_page,
)
from ui.services import fetch_history_records, get_predictor, prepare_history_dataframe
from ui.styles import inject_global_styles

st.set_page_config(
    page_title="LoanPredict · Credit Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ───────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "overview"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

# Pre-warm model cache once without blocking navigation switches
_ = get_predictor()

# Inject global design system CSS
T = inject_global_styles()


# ── Welcome Modal (retained & refined) ─────────────────────────────────────
@st.dialog("LoanPredict Platform Overview")
def show_welcome() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0.4rem;">
            <div style="width:52px; height:52px; margin:0 auto 0.75rem; border-radius:12px;
                        background:linear-gradient(135deg,#2563EB,#0EA5E9); display:flex;
                        align-items:center; justify-content:center; color:#FFFFFF;
                        font-weight:800; font-size:1.3rem; box-shadow:0 4px 14px rgba(37,99,235,0.35);">
                LP
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; margin-bottom: 0.2rem;">
                LoanPredict
            </div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #38BDF8; margin-bottom: 1.1rem;">
                Enterprise Loan Risk Assessment System
            </div>
            <div style="font-size: 0.88rem; line-height: 1.65; color: #94A3B8; margin-bottom: 1.4rem; text-align: left; padding: 0 0.25rem;">
                This platform assists financial institutions and credit analysts in evaluating loan applications
                by estimating default risk through a two-level <strong>Stacking Ensemble</strong> trained on
                historical credit and underwriting data.
            </div>
            <hr style="border-color: #1E293B; margin-bottom: 1rem;">
            <div style="font-size: 0.76rem; color: #64748B; margin-bottom: 0.25rem;">
                © 2026 LoanPredict · Credit Risk Intelligence
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.2rem;">
                Founded &amp; Developed by
            </div>
            <div style="font-size: 0.9rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.4rem;">
                Sagnik Ghosh &amp; Anantika Ghosh
            </div>
            <div style="font-size: 0.72rem; color: #64748B;">Version 1.0.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("Enter Workspace →", use_container_width=True, type="primary"):
        st.session_state.welcome_shown = True
        st.rerun()


if not st.session_state.welcome_shown:
    show_welcome()

# ── Load Cached History Records ────────────────────────────────────────────
raw_records, db_error = fetch_history_records()
history_df = prepare_history_dataframe(raw_records)

# ── Sidebar Navigation Shell ───────────────────────────────────────────────
render_sidebar(total_records=len(history_df), db_connected=(db_error is None))

# ── Page Routing ───────────────────────────────────────────────────────────
active_page = st.session_state.get("page", "overview")

if active_page == "overview":
    render_overview_page(history_df, db_error)
elif active_page == "new_prediction":
    render_new_assessment_page()
elif active_page == "history":
    render_history_page(history_df, db_error)
elif active_page == "analytics":
    render_analytics_page(history_df, db_error)
elif active_page == "model_insights":
    render_model_insights_page()
else:
    st.session_state.page = "overview"
    render_overview_page(history_df, db_error)

# ── Global Footer ──────────────────────────────────────────────────────────
render_footer()
