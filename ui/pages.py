"""
Page view implementations for the LoanPredict Fintech Dashboard:
1. Overview Dashboard
2. New Loan Assessment
3. Application History
4. Analytics Dashboard
5. Model Insights
"""

from datetime import datetime
import hashlib
import json
import math
import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    apply_chart_theme,
    mask_email,
    mask_phone,
    render_empty_state,
    render_kpi_card,
    render_page_header,
)
from ui.services import (
    PURPOSE_LABELS,
    PURPOSE_OPTIONS,
    delete_application_record,
    get_predictor,
    load_classification_reports,
    load_cross_validation_results,
    load_hyperparameter_results,
    load_model_comparison,
    prepare_history_dataframe,
    save_application_prediction,
    search_applications_exact,
)
from ui.styles import get_theme


# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def render_overview_page(df: pd.DataFrame, db_error: Optional[str]) -> None:
    """Render the executive overview landing dashboard."""
    T = get_theme()
    render_page_header(
        eyebrow="Executive Overview",
        title="Credit Risk Intelligence & Portfolio Command Center",
        subtitle="Real-time summary of loan risk assessments powered by the 7-model Stacking Ensemble",
        badge_text="● Stacking Classifier Active",
    )

    # Welcome & Quick-Action Banner
    st.markdown(
        f"""
        <div class="lp-card" style="background: linear-gradient(135deg, {T['surface']} 0%, {T['surface_alt']} 100%);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
                <div style="max-width:760px;">
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:{T['accent_teal']}; margin-bottom:0.3rem;">
                        Institutional Decision Support
                    </div>
                    <div style="font-size:1.18rem; font-weight:750; color:{T['text_primary']}; margin-bottom:0.4rem;">
                        Automated Loan Default Risk Assessment
                    </div>
                    <div style="font-size:0.88rem; color:{T['text_secondary']}; line-height:1.6;">
                        LoanPredict evaluates 13 underwriting, income, and credit-bureau attributes through a two-level
                        <strong>Stacking Ensemble</strong> combining Logistic Regression, SVM, KNN, Decision Tree,
                        Random Forest, Gaussian Naive Bayes, and XGBoost with a Logistic Regression meta-learner.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("✚  New Loan Assessment", key="ov_btn_new", type="primary", use_container_width=True):
            st.session_state.page = "new_prediction"
            st.rerun()
    with qa2:
        if st.button("▤  Application History", key="ov_btn_hist", type="secondary", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
    with qa3:
        if st.button("◔  Portfolio Analytics", key="ov_btn_ana", type="secondary", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
    with qa4:
        if st.button("❖  Model Benchmarks", key="ov_btn_mod", type="secondary", use_container_width=True):
            st.session_state.page = "model_insights"
            st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if db_error:
        st.warning(f"Database Notice: {db_error}")

    if df.empty:
        render_empty_state(
            title="No Loan Assessments Recorded Yet",
            message="Run your first loan risk assessment to populate portfolio KPIs, risk distributions, and recent activity logs.",
            icon="◈",
        )
        return

    total = len(df)
    low_risk_count = int((df["prediction"] == 0).sum())
    high_risk_count = int((df["prediction"] == 1).sum())
    low_risk_pct = (low_risk_count / total * 100.0) if total else 0.0
    high_risk_pct = (high_risk_count / total * 100.0) if total else 0.0
    avg_prob_pct = float(df["probability"].mean() * 100.0) if total else 0.0

    # Primary KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            label="Total Assessments",
            value=f"{total:,}",
            subtitle="Stored in MongoDB history",
            icon="▤",
        )
    with k2:
        render_kpi_card(
            label="Low-Risk (Class 0)",
            value=f"{low_risk_count:,}",
            subtitle=f"{low_risk_pct:.1f}% of evaluated applications",
            icon="✔",
            accent_color="#10B981",
        )
    with k3:
        render_kpi_card(
            label="High-Risk (Class 1)",
            value=f"{high_risk_count:,}",
            subtitle=f"{high_risk_pct:.1f}% of evaluated applications",
            icon="▲",
            accent_color="#EF4444",
        )
    with k4:
        render_kpi_card(
            label="Mean Class-1 Probability",
            value=f"{avg_prob_pct:.2f}%",
            subtitle="Average model risk score",
            icon="◔",
            accent_color=T["accent_teal"],
        )

    st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

    # Secondary Financial Summary Strip
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        render_kpi_card(
            label="Average FICO Score",
            value=f"{df['fico'].mean():.0f}",
            subtitle=f"Range: {int(df['fico'].min())} – {int(df['fico'].max())}",
            icon="◈",
        )
    with f2:
        render_kpi_card(
            label="Average DTI Ratio",
            value=f"{df['dti'].mean():.1f}%",
            subtitle=f"Median: {df['dti'].median():.1f}%",
            icon="◈",
        )
    with f3:
        render_kpi_card(
            label="Average Interest Rate",
            value=f"{df['int.rate'].mean() * 100:.2f}%",
            subtitle=f"Decimal mean: {df['int.rate'].mean():.3f}",
            icon="◈",
        )
    with f4:
        render_kpi_card(
            label="Avg Monthly Installment",
            value=f"${df['installment'].mean():,.2f}",
            subtitle=f"Avg Revol. Util: {df['revol.util'].mean():.1f}%",
            icon="◈",
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Charts Row: Risk Distribution + Loan Purpose Breakdown
    c_left, c_right = st.columns([5, 7])

    with c_left:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Predicted Risk Distribution</div>
                        <div class="lp-card-subtitle">Share of Low-Risk (Class 0) vs. High-Risk (Class 1) assessments</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        risk_counts = pd.DataFrame(
            {
                "Risk Category": ["Low Risk (Class 0)", "High Risk (Class 1)"],
                "Count": [low_risk_count, high_risk_count],
            }
        )
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts["Risk Category"],
                    values=risk_counts["Count"],
                    hole=0.62,
                    marker=dict(colors=["#10B981", "#EF4444"]),
                    textinfo="percent+label",
                    textposition="outside",
                    hovertemplate="<b>%{label}</b><br>Applications: %{value}<br>Share: %{percent}<extra></extra>",
                )
            ]
        )
        fig_donut = apply_chart_theme(fig_donut, height=300)
        fig_donut.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12, x=0.1))
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Assessments by Loan Purpose</div>
                        <div class="lp-card-subtitle">Volume and predicted risk outcome across loan categories</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        purpose_grp = (
            df.groupby(["purpose_label", "risk_label"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig_purpose = px.bar(
            purpose_grp,
            x="Count",
            y="purpose_label",
            color="risk_label",
            orientation="h",
            barmode="stack",
            color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
            labels={"purpose_label": "Loan Purpose", "Count": "Applications", "risk_label": "Predicted Risk"},
        )
        fig_purpose = apply_chart_theme(fig_purpose, height=300)
        fig_purpose.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_purpose, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Recent Application Activity Table
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">Recent Assessment Activity</div>
                    <div class="lp-card-subtitle">Latest applications evaluated by the model (contact details masked for privacy)</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    recent_df = df.head(8)
    rows_html = []
    for _, row in recent_df.iterrows():
        is_low = int(row["prediction"]) == 0
        badge = (
            '<span class="lp-badge-low">● Low Risk (0)</span>'
            if is_low
            else '<span class="lp-badge-high">▲ High Risk (1)</span>'
        )
        rows_html.append(
            f"<tr>"
            f"<td class='lp-mono'>{row['timestamp_str']}</td>"
            f"<td><strong>{row['display_name']}</strong></td>"
            f"<td class='lp-mono'>{mask_email(row['email'])}</td>"
            f"<td>{row['purpose_label']}</td>"
            f"<td class='lp-mono'>{int(row['fico'])}</td>"
            f"<td class='lp-mono'>{row['dti']:.1f}%</td>"
            f"<td class='lp-mono'>{row['prob_pct']:.2f}%</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <div class="lp-table-wrap">
            <table class="lp-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Applicant</th>
                        <th>Email (Masked)</th>
                        <th>Purpose</th>
                        <th>FICO</th>
                        <th>DTI</th>
                        <th>Class-1 Prob.</th>
                        <th>Risk Assessment</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. NEW LOAN ASSESSMENT PAGE
# ═══════════════════════════════════════════════════════════════════════════
def _apply_form_preset(preset_type: str) -> None:
    """Populate session state form defaults for quick testing without altering model logic."""
    if preset_type == "low_risk":
        st.session_state.update(
            {
                "f_full_name": "Jane Doe",
                "f_email": "jane.doe@example.com",
                "f_phone": "+1 (555) 234-5678",
                "f_address": "123 Financial District Blvd, Suite 400",
                "f_purpose_label": "Debt Consolidation",
                "f_credit_policy": "Meets Policy (1)",
                "f_int_rate": 0.110,
                "f_installment": 300.0,
                "f_log_income": 11.00,
                "f_dti": 15.00,
                "f_fico": 700,
                "f_days_cr": 3500.0,
                "f_rev_bal": 10000.0,
                "f_rev_util": 45.0,
                "f_inq_6m": 1,
                "f_delinq": 0,
                "f_pub_rec": 0,
            }
        )
    elif preset_type == "high_risk":
        st.session_state.update(
            {
                "f_full_name": "Alex Mercer",
                "f_email": "a.mercer@venturemail.org",
                "f_phone": "+1 (555) 876-9012",
                "f_address": "742 Evergreen Terrace, Apt 9B",
                "f_purpose_label": "Small Business",
                "f_credit_policy": "Does Not Meet Policy (0)",
                "f_int_rate": 0.210,
                "f_installment": 850.0,
                "f_log_income": 9.50,
                "f_dti": 28.50,
                "f_fico": 615,
                "f_days_cr": 1100.0,
                "f_rev_bal": 45000.0,
                "f_rev_util": 95.0,
                "f_inq_6m": 12,
                "f_delinq": 3,
                "f_pub_rec": 2,
            }
        )
    elif preset_type == "reset":
        st.session_state.update(
            {
                "f_full_name": "",
                "f_email": "",
                "f_phone": "",
                "f_address": "",
                "f_purpose_label": "Debt Consolidation",
                "f_credit_policy": "Meets Policy (1)",
                "f_int_rate": 0.110,
                "f_installment": 300.0,
                "f_log_income": 11.00,
                "f_dti": 15.00,
                "f_fico": 700,
                "f_days_cr": 3500.0,
                "f_rev_bal": 10000.0,
                "f_rev_util": 45.0,
                "f_inq_6m": 1,
                "f_delinq": 0,
                "f_pub_rec": 0,
            }
        )
        st.session_state.pop("last_assessment_output", None)


def render_new_assessment_page() -> None:
    """Render the guided loan risk assessment form and model results panel."""
    T = get_theme()
    render_page_header(
        eyebrow="Underwriting Workstation",
        title="New Loan Risk Assessment",
        subtitle="Enter applicant profile and credit attributes to run the Stacking Classifier inference pipeline",
        badge_text="13 Model Features · StandardScaler + LabelEncoder",
    )

    # Quick profile loader bar for convenient evaluation
    p_col1, p_col2, p_col3, p_col4 = st.columns([3, 3, 2, 4])
    with p_col1:
        if st.button("Load Standard Profile (Sample)", key="preset_low", type="secondary", use_container_width=True):
            _apply_form_preset("low_risk")
            st.rerun()
    with p_col2:
        if st.button("Load Elevated-Risk Profile (Sample)", key="preset_high", type="secondary", use_container_width=True):
            _apply_form_preset("high_risk")
            st.rerun()
    with p_col3:
        if st.button("Reset Form", key="preset_reset", type="secondary", use_container_width=True):
            _apply_form_preset("reset")
            st.rerun()
    with p_col4:
        st.markdown(
            f"<div style='font-size:0.78rem; color:{T['text_muted']}; padding-top:0.55rem; text-align:right;'>"
            "All 13 financial features map directly to the trained pipeline.</div>",
            unsafe_allow_html=True,
        )

    # ── Section 1: Applicant Information ───────────────────────────────────
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">1. Applicant Identification &amp; Contact Profile</div>
                    <div class="lp-card-subtitle">Recorded alongside the assessment outcome in MongoDB application history</div>
                </div>
                <span class="lp-badge-neutral">Applicant Metadata</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    a1, a2 = st.columns(2)
    with a1:
        full_name = st.text_input(
            "Full Name",
            value=st.session_state.get("f_full_name", ""),
            placeholder="e.g., Jane Doe",
            help="Applicant's full legal name for application tracking and history search.",
            key="f_full_name",
        )
        phone = st.text_input(
            "Phone Number",
            value=st.session_state.get("f_phone", ""),
            placeholder="e.g., +1 (555) 000-0000",
            help="Primary contact number used for record identification in MongoDB.",
            key="f_phone",
        )
    with a2:
        email = st.text_input(
            "Email Address",
            value=st.session_state.get("f_email", ""),
            placeholder="e.g., jane.doe@example.com",
            help="Applicant email address used for search and audit logging.",
            key="f_email",
        )
        address = st.text_input(
            "Residential Address",
            value=st.session_state.get("f_address", ""),
            placeholder="e.g., 123 Market St, Suite 400",
            help="Current street address of the applicant.",
            key="f_address",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: Loan Details & Underwriting Policy ──────────────────────
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">2. Loan Structure &amp; Underwriting Policy</div>
                    <div class="lp-card-subtitle">Core loan terms and credit policy qualification (features: credit.policy, purpose, int.rate, installment)</div>
                </div>
                <span class="lp-badge-neutral">4 Model Features</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    l1, l2 = st.columns(2)
    purpose_display_list = list(PURPOSE_OPTIONS.keys())
    default_purpose_idx = 0
    if st.session_state.get("f_purpose_label") in purpose_display_list:
        default_purpose_idx = purpose_display_list.index(st.session_state["f_purpose_label"])

    with l1:
        purpose_label = st.selectbox(
            "Loan Purpose (purpose)",
            options=purpose_display_list,
            index=default_purpose_idx,
            help="Mapped to the exact LabelEncoder classes trained in models/encoder.pkl.",
            key="f_purpose_label",
        )
        int_rate = st.number_input(
            "Interest Rate — Decimal (int.rate)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("f_int_rate", 0.11)),
            step=0.001,
            format="%.3f",
            help="Annual interest rate expressed as a decimal (e.g., 0.110 = 11.00%).",
            key="f_int_rate",
        )
    with l2:
        credit_policy = st.selectbox(
            "Credit Underwriting Policy (credit.policy)",
            options=["Meets Policy (1)", "Does Not Meet Policy (0)"],
            index=0 if "1" in st.session_state.get("f_credit_policy", "Meets Policy (1)") else 1,
            help="1 if the customer meets the credit underwriting criteria; 0 otherwise.",
            key="f_credit_policy",
        )
        installment = st.number_input(
            "Monthly Installment ($) (installment)",
            min_value=0.0,
            max_value=50000.0,
            value=float(st.session_state.get("f_installment", 300.0)),
            step=10.0,
            format="%.2f",
            help="Monthly installment amount owed by the borrower if the loan is funded.",
            key="f_installment",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Financial Profile & Debt Capacity ───────────────────────
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">3. Income &amp; Debt-to-Income Profile</div>
                    <div class="lp-card-subtitle">Borrower income scale and debt servicing ratio (features: log.annual.inc, dti)</div>
                </div>
                <span class="lp-badge-neutral">2 Model Features</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    inc1, inc2 = st.columns(2)
    with inc1:
        log_income = st.number_input(
            "Natural Log of Annual Income (log.annual.inc)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.get("f_log_income", 11.0)),
            step=0.1,
            format="%.2f",
            help="Natural logarithm of self-reported annual income (e.g., 11.00 ≈ $59,874/yr).",
            key="f_log_income",
        )
        approx_annual_inc = math.exp(log_income) if log_income <= 20 else 0.0
        st.caption(f"Implied Annual Income: **${approx_annual_inc:,.0f}** (`exp({log_income:.2f})`)")
    with inc2:
        dti = st.number_input(
            "Debt-to-Income Ratio % (dti)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get("f_dti", 15.0)),
            step=0.1,
            format="%.2f",
            help="Ratio of borrower's total monthly debt payments divided by monthly income.",
            key="f_dti",
        )
        st.caption(f"Effective Interest Rate Display: **{int_rate * 100:.2f}% APR**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Credit Bureau History ───────────────────────────────────
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">4. Credit Bureau &amp; Utilization Indicators</div>
                    <div class="lp-card-subtitle">FICO score, credit line tenure, revolving utilization, and derogatory events (7 features)</div>
                </div>
                <span class="lp-badge-neutral">7 Model Features</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    cb1, cb2 = st.columns(2)
    with cb1:
        fico = st.number_input(
            "FICO Credit Score (fico)",
            min_value=300,
            max_value=850,
            value=int(st.session_state.get("f_fico", 700)),
            step=1,
            help="Standard FICO credit score range: 300 to 850.",
            key="f_fico",
        )
        rev_bal = st.number_input(
            "Revolving Balance ($) (revol.bal)",
            min_value=0.0,
            max_value=10000000.0,
            value=float(st.session_state.get("f_rev_bal", 10000.0)),
            step=100.0,
            format="%.2f",
            help="Borrower's unpaid balance at the end of the credit card billing cycle.",
            key="f_rev_bal",
        )
    with cb2:
        days_cr = st.number_input(
            "Days with Credit Line (days.with.cr.line)",
            min_value=0.0,
            max_value=50000.0,
            value=float(st.session_state.get("f_days_cr", 3500.0)),
            step=10.0,
            format="%.1f",
            help="Number of days the borrower has had an active credit line.",
            key="f_days_cr",
        )
        rev_util = st.number_input(
            "Revolving Line Utilization % (revol.util)",
            min_value=0.0,
            max_value=150.0,
            value=float(st.session_state.get("f_rev_util", 45.0)),
            step=0.1,
            format="%.1f",
            help="Percentage of available revolving credit currently utilized.",
            key="f_rev_util",
        )

    cb3, cb4, cb5 = st.columns(3)
    with cb3:
        inq_6m = st.number_input(
            "Inquiries in Last 6 Months (inq.last.6mths)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("f_inq_6m", 1)),
            step=1,
            help="Number of credit inquiries by creditors in the past 6 months.",
            key="f_inq_6m",
        )
    with cb4:
        delinq = st.number_input(
            "Delinquencies in Last 2 Years (delinq.2yrs)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("f_delinq", 0)),
            step=1,
            help="Number of times borrower was 30+ days past due on a payment in the past 2 years.",
            key="f_delinq",
        )
    with cb5:
        pub_rec = st.number_input(
            "Public Derogatory Records (pub.rec)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("f_pub_rec", 0)),
            step=1,
            help="Number of derogatory public records (bankruptcy filings, tax liens, or judgments).",
            key="f_pub_rec",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Submit Controls
    sub_col1, sub_col2 = st.columns([8, 4])
    with sub_col1:
        run_clicked = st.button(
            "Run Stacking Ensemble Risk Assessment  →",
            key="run_prediction_btn",
            type="primary",
            use_container_width=True,
        )
    with sub_col2:
        allow_duplicate = st.checkbox(
            "Allow duplicate submission of identical inputs",
            value=False,
            help="By default, accidental consecutive submissions of the exact same payload are blocked from creating duplicate database records.",
        )

    if run_clicked:
        # Input validation notices
        if email.strip() and "@" not in email.strip():
            st.error("Please enter a valid email address (containing '@') or leave the field blank.")
            return

        applicant = {
            "name": full_name.strip(),
            "address": address.strip(),
            "phone": phone.strip(),
            "email": email.strip(),
        }

        # Exact feature dictionary and key order expected by LoanPrediction.predict()
        loan_data = {
            "credit.policy": 1 if "1" in credit_policy else 0,
            "purpose": PURPOSE_OPTIONS[purpose_label],
            "int.rate": float(int_rate),
            "installment": float(installment),
            "log.annual.inc": float(log_income),
            "dti": float(dti),
            "fico": int(fico),
            "days.with.cr.line": float(days_cr),
            "revol.bal": float(rev_bal),
            "revol.util": float(rev_util),
            "inq.last.6mths": int(inq_6m),
            "delinq.2yrs": int(delinq),
            "pub.rec": int(pub_rec),
        }

        payload_signature = hashlib.sha256(
            json.dumps({"applicant": applicant, "loan_data": loan_data}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if (
            not allow_duplicate
            and st.session_state.get("last_submission_hash") == payload_signature
            and "last_assessment_output" in st.session_state
        ):
            st.info(
                "Duplicate submission prevented: This exact applicant and loan profile was just evaluated and saved. "
                "Displaying the existing assessment result below (enable 'Allow duplicate submission' to record again)."
            )
        else:
            with st.spinner("Running preprocessing and Stacking Classifier inference..."):
                predictor = get_predictor()
                result = predictor.predict(loan_data)
                assessment_ts = datetime.now()
                saved_ok, save_err = save_application_prediction(applicant, loan_data, result)

                st.session_state.last_submission_hash = payload_signature
                st.session_state.last_assessment_output = {
                    "applicant": applicant,
                    "loan_data": loan_data,
                    "purpose_label": purpose_label,
                    "result": result,
                    "timestamp": assessment_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "saved_ok": saved_ok,
                    "save_err": save_err,
                }

    # Render Assessment Results Panel if available
    if "last_assessment_output" in st.session_state:
        _render_prediction_result_panel(st.session_state.last_assessment_output)


def _render_prediction_result_panel(output: Dict[str, Any]) -> None:
    """Render the professional assessment results panel for a completed prediction."""
    T = get_theme()
    result = output["result"]
    loan_data = output["loan_data"]
    applicant = output["applicant"]
    purpose_label = output["purpose_label"]
    ts_str = output["timestamp"]
    saved_ok = output["saved_ok"]
    save_err = output["save_err"]

    prediction_class = int(result["Prediction"])
    prob_class_1 = float(result["Probability"]) * 100.0
    prob_class_0 = 100.0 - prob_class_1
    is_low_risk = prediction_class == 0

    st.markdown("<hr style='border-color:" + T["border"] + "; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    panel_class = "lp-result-panel-low" if is_low_risk else "lp-result-panel-high"
    status_badge = (
        '<span class="lp-badge-low" style="font-size:0.85rem; padding:0.35rem 0.85rem;">✔ LOW RISK ASSESSMENT (Predicted Class 0)</span>'
        if is_low_risk
        else '<span class="lp-badge-high" style="font-size:0.85rem; padding:0.35rem 0.85rem;">▲ HIGH RISK ASSESSMENT (Predicted Class 1)</span>'
    )
    db_badge = (
        '<span class="lp-badge-low">✓ Saved to MongoDB History</span>'
        if saved_ok
        else '<span class="lp-badge-high">⚠ Not Saved to MongoDB</span>'
    )

    st.markdown(
        f"""
        <div class="{panel_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem; margin-bottom:0.85rem;">
                <div>{status_badge}</div>
                <div style="display:flex; align-items:center; gap:0.6rem;">
                    <span class="lp-badge-neutral lp-mono">Evaluated: {ts_str}</span>
                    {db_badge}
                </div>
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{T['text_primary']}; margin-bottom:0.35rem;">
                {"Model Assessment: Favorable Risk Profile (Class 0)" if is_low_risk else "Model Assessment: Elevated Default Risk Profile (Class 1)"}
            </div>
            <div style="font-size:0.88rem; color:{T['text_secondary']}; line-height:1.55;">
                The Stacking Classifier estimated a <strong>{prob_class_1:.2f}%</strong> probability for
                <code>Class 1 (not.fully.paid = 1)</code> (and <strong>{prob_class_0:.2f}%</strong> for
                <code>Class 0</code>). Using the model's standard 50.0% decision threshold, this application is classified as
                <strong>{"Low Risk (Class 0)" if is_low_risk else "High Risk (Class 1)"}</strong>.
                <em>Note: This output is a model-generated risk assessment to support credit underwriting, not a guaranteed lending decision.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not saved_ok and save_err:
        st.warning(save_err)

    # KPI summary cards for the prediction
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card(
            label="Predicted Risk Class",
            value="Low Risk (0)" if is_low_risk else "High Risk (1)",
            subtitle="Decision threshold: 50.0%",
            icon="✔" if is_low_risk else "▲",
            accent_color="#10B981" if is_low_risk else "#EF4444",
        )
    with r2:
        render_kpi_card(
            label="Class-1 Probability",
            value=f"{prob_class_1:.2f}%",
            subtitle=f"Class-0 Complement: {prob_class_0:.2f}%",
            icon="◔",
            accent_color="#10B981" if is_low_risk else "#EF4444",
        )
    with r3:
        render_kpi_card(
            label="Applicant FICO Score",
            value=str(loan_data["fico"]),
            subtitle=f"Credit Policy: {'Meets (1)' if loan_data['credit.policy'] == 1 else 'Does Not Meet (0)'}",
            icon="◈",
        )
    with r4:
        render_kpi_card(
            label="DTI & Interest Rate",
            value=f"{loan_data['dti']:.1f}% DTI",
            subtitle=f"Rate: {loan_data['int.rate']:.3f} ({loan_data['int.rate']*100:.2f}%)",
            icon="◈",
        )

    st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

    # Visual Risk Gauge + Submitted Application Summary
    g_col, s_col = st.columns([5, 7])
    with g_col:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Model Risk Probability Indicator</div>
                        <div class="lp-card-subtitle">Stacking Classifier Class-1 probability vs. 50% threshold</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_class_1,
                number={"suffix": "%", "valueformat": ".2f", "font": {"size": 34, "color": T["text_primary"]}},
                title={"text": "Predicted Class-1 Probability (not.fully.paid = 1)", "font": {"size": 12, "color": T["text_muted"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": T["text_muted"]},
                    "bar": {"color": "#10B981" if is_low_risk else "#EF4444", "thickness": 0.35},
                    "bgcolor": T["surface_alt"],
                    "borderwidth": 1,
                    "bordercolor": T["border"],
                    "steps": [
                        {"range": [0, 50], "color": "rgba(16, 185, 129, 0.14)"},
                        {"range": [50, 100], "color": "rgba(239, 68, 68, 0.14)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F59E0B", "width": 3},
                        "thickness": 0.8,
                        "value": 50.0,
                    },
                },
            )
        )
        fig_gauge = apply_chart_theme(fig_gauge, height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with s_col:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Submitted Assessment Summary</div>
                        <div class="lp-card-subtitle">Exact parameter snapshot evaluated by LoanPrediction.predict()</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        summary_pairs = [
            ("Applicant Name", applicant["name"] or "—"),
            ("Contact Email / Phone", f"{applicant['email'] or '—'} · {applicant['phone'] or '—'}"),
            ("Loan Purpose (`purpose`)", f"{purpose_label} (`{loan_data['purpose']}`)"),
            ("Credit Policy (`credit.policy`)", str(loan_data["credit.policy"])),
            ("Interest Rate (`int.rate`)", f"{loan_data['int.rate']:.3f} ({loan_data['int.rate']*100:.2f}%)"),
            ("Monthly Installment (`installment`)", f"${loan_data['installment']:,.2f}"),
            ("Log Annual Income (`log.annual.inc`)", f"{loan_data['log.annual.inc']:.2f}"),
            ("DTI Ratio (`dti`)", f"{loan_data['dti']:.2f}%"),
            ("FICO Score (`fico`)", str(loan_data["fico"])),
            ("Credit Line Tenure (`days.with.cr.line`)", f"{loan_data['days.with.cr.line']:,.1f} days"),
            ("Revolving Balance / Util", f"${loan_data['revol.bal']:,.2f} ({loan_data['revol.util']:.1f}%)"),
            (
                "Inquiries / Delinq / Public Rec",
                f"{loan_data['inq.last.6mths']} inq · {loan_data['delinq.2yrs']} delinq · {loan_data['pub.rec']} pub.rec",
            ),
        ]
        rows_html = "".join(
            f"<tr><td style='color:{T['text_muted']}; width:46%;'>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in summary_pairs
        )
        st.markdown(
            f"""
            <div class="lp-table-wrap" style="margin-bottom:0;">
                <table class="lp-table">
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. APPLICATION HISTORY PAGE
# ═══════════════════════════════════════════════════════════════════════════
def render_history_page(df: pd.DataFrame, db_error: Optional[str]) -> None:
    """Render the searchable, filterable application history & record management view."""
    T = get_theme()
    render_page_header(
        eyebrow="Audit & Record Management",
        title="Application History",
        subtitle="Search, filter, inspect, and manage historical loan risk assessments stored in MongoDB",
        badge_text=f"{len(df):,} Total Records",
    )

    if db_error:
        st.warning(f"Database Notice: {db_error}")

    # Backend Exact Search Expander (preserves PredictionHistory.search_application)
    with st.expander("🔍 Exact Applicant Lookup (Backend MongoDB Query: Name + Email + Phone)", expanded=False):
        st.caption(
            "Executes `PredictionHistory.search_application(name, email, phone)` directly against the MongoDB collection."
        )
        s1, s2, s3, s4 = st.columns([3, 3, 3, 2])
        with s1:
            q_name = st.text_input("Applicant Name", key="exact_q_name", placeholder="Jane Doe")
        with s2:
            q_email = st.text_input("Email Address", key="exact_q_email", placeholder="jane.doe@example.com")
        with s3:
            q_phone = st.text_input("Phone Number", key="exact_q_phone", placeholder="+1 (555) 000-0000")
        with s4:
            st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
            run_exact = st.button("Run Exact Lookup", key="btn_exact_search", type="primary", use_container_width=True)

        if run_exact:
            if not (q_name.strip() and q_email.strip() and q_phone.strip()):
                st.warning("Please provide Applicant Name, Email Address, and Phone Number for an exact backend lookup.")
            else:
                matches, err = search_applications_exact(q_name, q_email, q_phone)
                if err:
                    st.error(err)
                elif not matches:
                    st.info("No matching application found for the specified Name, Email, and Phone combination.")
                else:
                    st.success(f"Found {len(matches)} matching record(s) in MongoDB.")
                    match_df = prepare_history_dataframe(matches)
                    st.dataframe(
                        match_df[
                            [
                                "timestamp_str",
                                "display_name",
                                "email",
                                "phone",
                                "purpose_label",
                                "fico",
                                "dti",
                                "prob_pct",
                                "risk_label",
                            ]
                        ].rename(
                            columns={
                                "timestamp_str": "Timestamp",
                                "display_name": "Applicant",
                                "email": "Email",
                                "phone": "Phone",
                                "purpose_label": "Purpose",
                                "fico": "FICO",
                                "dti": "DTI %",
                                "prob_pct": "Class-1 Prob %",
                                "risk_label": "Risk Assessment",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    if df.empty:
        render_empty_state(
            title="No Historical Applications Found",
            message="Submit a loan risk assessment on the New Loan Assessment page to view and manage records here.",
            icon="▤",
        )
        return

    # ── Interactive Filter & Sort Controls ─────────────────────────────────
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">Filter &amp; Sort Applications</div>
                    <div class="lp-card-subtitle">Refine historical records by applicant, risk category, purpose, date, or credit metrics</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3, f_col4 = st.columns([4, 3, 3, 3])
    with f_col1:
        search_text = st.text_input(
            "Quick Search (Name, Email, Phone, Purpose)",
            value="",
            placeholder="Filter by name, email, phone, or purpose...",
            key="hist_search_text",
        )
    with f_col2:
        risk_filter = st.selectbox(
            "Risk Category",
            options=["All Assessments", "Low Risk (Class 0)", "High Risk (Class 1)"],
            key="hist_risk_filter",
        )
    with f_col3:
        available_purposes = ["All Purposes"] + sorted(df["purpose_label"].unique().tolist())
        purpose_filter = st.selectbox("Loan Purpose", options=available_purposes, key="hist_purpose_filter")
    with f_col4:
        sort_option = st.selectbox(
            "Sort Records By",
            options=[
                "Timestamp (Newest First)",
                "Timestamp (Oldest First)",
                "Class-1 Probability (High → Low)",
                "Class-1 Probability (Low → High)",
                "FICO Score (High → Low)",
                "FICO Score (Low → High)",
                "DTI Ratio (High → Low)",
            ],
            key="hist_sort_option",
        )

    # Optional Date Range Filter + Privacy Mask Toggle
    d_col1, d_col2, d_col3 = st.columns([4, 4, 5])
    valid_dates = df["date_only"].dropna()
    date_range = None
    with d_col1:
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            date_range = st.date_input(
                "Filter by Assessment Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="hist_date_range",
            )
    with d_col2:
        page_size = st.selectbox("Rows per Page", options=[10, 20, 50], index=0, key="hist_page_size")
    with d_col3:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        mask_pii = st.checkbox(
            "Mask personal contact details (Email & Phone) in overview table",
            value=True,
            key="hist_mask_pii",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered = df.copy()
    if search_text.strip():
        q = search_text.strip().lower()
        mask = (
            filtered["display_name"].str.lower().str.contains(q, na=False)
            | filtered["email"].str.lower().str.contains(q, na=False)
            | filtered["phone"].str.lower().str.contains(q, na=False)
            | filtered["address"].str.lower().str.contains(q, na=False)
            | filtered["purpose_label"].str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    if risk_filter == "Low Risk (Class 0)":
        filtered = filtered[filtered["prediction"] == 0]
    elif risk_filter == "High Risk (Class 1)":
        filtered = filtered[filtered["prediction"] == 1]

    if purpose_filter != "All Purposes":
        filtered = filtered[filtered["purpose_label"] == purpose_filter]

    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered = filtered[
            filtered["date_only"].isna()
            | ((filtered["date_only"] >= start_d) & (filtered["date_only"] <= end_d))
        ]

    # Apply sorting
    if sort_option == "Timestamp (Newest First)":
        filtered = filtered.sort_values(by="timestamp_dt", ascending=False, na_position="last")
    elif sort_option == "Timestamp (Oldest First)":
        filtered = filtered.sort_values(by="timestamp_dt", ascending=True, na_position="last")
    elif sort_option == "Class-1 Probability (High → Low)":
        filtered = filtered.sort_values(by="probability", ascending=False)
    elif sort_option == "Class-1 Probability (Low → High)":
        filtered = filtered.sort_values(by="probability", ascending=True)
    elif sort_option == "FICO Score (High → Low)":
        filtered = filtered.sort_values(by="fico", ascending=False)
    elif sort_option == "FICO Score (Low → High)":
        filtered = filtered.sort_values(by="fico", ascending=True)
    elif sort_option == "DTI Ratio (High → Low)":
        filtered = filtered.sort_values(by="dti", ascending=False)

    filtered = filtered.reset_index(drop=True)

    # Filtered KPI Strip
    total_f = len(filtered)
    low_f = int((filtered["prediction"] == 0).sum()) if total_f else 0
    high_f = int((filtered["prediction"] == 1).sum()) if total_f else 0
    low_rate_f = (low_f / total_f * 100.0) if total_f else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("Matching Records", f"{total_f:,}", f"Out of {len(df):,} total", "▤")
    with m2:
        render_kpi_card("Low-Risk (Approved)", f"{low_f:,}", f"{low_rate_f:.1f}% of filtered", "✔", "#10B981")
    with m3:
        render_kpi_card("High-Risk (Denied)", f"{high_f:,}", f"{100 - low_rate_f if total_f else 0:.1f}% of filtered", "▲", "#EF4444")
    with m4:
        csv_bytes = filtered.drop(columns=["timestamp_dt", "date_only"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Export Filtered CSV",
            data=csv_bytes,
            file_name="loanpredict_application_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    if filtered.empty:
        render_empty_state(
            title="No Applications Match Current Filters",
            message="Adjust your search query, risk category filter, or date range to view matching records.",
            icon="🔍",
        )
        return

    # Pagination
    total_pages = max(1, math.ceil(total_f / page_size))
    current_page_num = st.session_state.get("hist_page_num", 1)
    if current_page_num > total_pages:
        current_page_num = 1
        st.session_state.hist_page_num = 1

    start_idx = (current_page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_f)
    page_df = filtered.iloc[start_idx:end_idx]

    # Render Application Table
    rows_html = []
    for idx, row in page_df.iterrows():
        is_low = int(row["prediction"]) == 0
        badge = (
            '<span class="lp-badge-low">● Low Risk (0)</span>'
            if is_low
            else '<span class="lp-badge-high">▲ High Risk (1)</span>'
        )
        disp_email = mask_email(row["email"]) if mask_pii else (row["email"] or "—")
        disp_phone = mask_phone(row["phone"]) if mask_pii else (row["phone"] or "—")
        rows_html.append(
            f"<tr>"
            f"<td class='lp-mono'>#{idx + 1}</td>"
            f"<td class='lp-mono'>{row['timestamp_str']}</td>"
            f"<td><strong>{row['display_name']}</strong></td>"
            f"<td class='lp-mono'>{disp_email}</td>"
            f"<td class='lp-mono'>{disp_phone}</td>"
            f"<td>{row['purpose_label']}</td>"
            f"<td class='lp-mono'>{int(row['fico'])}</td>"
            f"<td class='lp-mono'>{row['dti']:.1f}%</td>"
            f"<td class='lp-mono'>{row['int.rate']:.3f}</td>"
            f"<td class='lp-mono'>{row['prob_pct']:.2f}%</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">Application Records (Showing {start_idx + 1}–{end_idx} of {total_f})</div>
                    <div class="lp-card-subtitle">Page {current_page_num} of {total_pages}</div>
                </div>
            </div>
            <div class="lp-table-wrap">
                <table class="lp-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Timestamp</th>
                            <th>Applicant</th>
                            <th>Email</th>
                            <th>Phone</th>
                            <th>Purpose</th>
                            <th>FICO</th>
                            <th>DTI %</th>
                            <th>Rate</th>
                            <th>Class-1 Prob.</th>
                            <th>Assessment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_html)}
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total_pages > 1:
        p1, p2, p3 = st.columns([2, 6, 2])
        with p1:
            if st.button("← Previous Page", disabled=(current_page_num <= 1), key="hist_prev_btn", use_container_width=True):
                st.session_state.hist_page_num = current_page_num - 1
                st.rerun()
        with p2:
            st.markdown(
                f"<div style='text-align:center; padding-top:0.5rem; color:{T['text_muted']}; font-size:0.85rem;'>"
                f"Page <strong>{current_page_num}</strong> of <strong>{total_pages}</strong></div>",
                unsafe_allow_html=True,
            )
        with p3:
            if st.button("Next Page →", disabled=(current_page_num >= total_pages), key="hist_next_btn", use_container_width=True):
                st.session_state.hist_page_num = current_page_num + 1
                st.rerun()

    # ── Expandable Application Details & Safe Delete Workflow ──────────────
    st.markdown(
        """
        <div class="lp-card">
            <div class="lp-card-header">
                <div>
                    <div class="lp-card-title">Inspect Application Profile &amp; Record Actions</div>
                    <div class="lp-card-subtitle">Select any application from the filtered list to inspect all 13 underwriting parameters or delete a record</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    option_labels = [
        f"#{i + 1} · {r['timestamp_str']} · {r['display_name']} · {r['purpose_label']} · FICO {int(r['fico'])} ({r['risk_label']})"
        for i, r in filtered.iterrows()
    ]
    selected_idx = st.selectbox(
        "Select Application Record to Inspect",
        options=list(range(len(option_labels))),
        format_func=lambda i: option_labels[i],
        key="hist_inspect_select",
    )
    sel_row = filtered.iloc[selected_idx]

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Applicant Contact Info**")
        st.write(f"**Name:** {sel_row['display_name']}")
        st.write(f"**Email:** {sel_row['email'] or '—'}")
        st.write(f"**Phone:** {sel_row['phone'] or '—'}")
        st.write(f"**Address:** {sel_row['address'] or '—'}")
        st.write(f"**Timestamp:** `{sel_row['timestamp_str']}`")
    with d2:
        st.markdown("**Loan & Income Attributes**")
        st.write(f"**Purpose:** {sel_row['purpose_label']} (`{sel_row['purpose']}`)")
        st.write(f"**Credit Policy:** `{int(sel_row['credit.policy'])}`")
        st.write(f"**Interest Rate:** `{sel_row['int.rate']:.3f}` ({sel_row['int.rate']*100:.2f}%)")
        st.write(f"**Installment:** `${sel_row['installment']:,.2f}`")
        st.write(f"**Log Annual Income:** `{sel_row['log.annual.inc']:.2f}`")
        st.write(f"**DTI Ratio:** `{sel_row['dti']:.2f}%`")
    with d3:
        st.markdown("**Credit Bureau & Assessment**")
        st.write(f"**FICO Score:** `{int(sel_row['fico'])}`")
        st.write(f"**Days with Credit Line:** `{sel_row['days.with.cr.line']:,.1f}`")
        st.write(f"**Revolving Bal / Util:** `${sel_row['revol.bal']:,.2f}` / `{sel_row['revol.util']:.1f}%`")
        st.write(
            f"**Inq (6m) / Delinq (2y) / PubRec:** `{int(sel_row['inq.last.6mths'])}` / `{int(sel_row['delinq.2yrs'])}` / `{int(sel_row['pub.rec'])}`"
        )
        st.write(f"**Model Output:** **{sel_row['risk_label']}** (`Class {int(sel_row['prediction'])}`, Prob: `{sel_row['prob_pct']:.2f}%`)")

    st.markdown("<hr style='border-color:" + T["border"] + "; margin: 1rem 0;'>", unsafe_allow_html=True)
    del_col1, del_col2 = st.columns([7, 5])
    with del_col1:
        confirm_delete = st.checkbox(
            f"Confirm permanent deletion of record #{selected_idx + 1} ({sel_row['display_name']}) from MongoDB",
            key=f"confirm_del_{selected_idx}",
        )
    with del_col2:
        if st.button(
            "Delete Selected Application Record",
            key=f"btn_del_{selected_idx}",
            type="secondary",
            disabled=not confirm_delete,
            use_container_width=True,
        ):
            deleted_count, del_err = delete_application_record(
                sel_row["name"], sel_row["email"], sel_row["phone"]
            )
            if del_err:
                st.error(del_err)
            elif deleted_count > 0:
                st.success("Application record deleted from MongoDB.")
                st.rerun()
            else:
                st.warning("No matching document was deleted.")

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANALYTICS DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════════════════
def render_analytics_page(df: pd.DataFrame, db_error: Optional[str]) -> None:
    """Render portfolio-level analytics computed from stored MongoDB assessments."""
    T = get_theme()
    render_page_header(
        eyebrow="Portfolio Analytics",
        title="Application Risk & Financial Analytics Dashboard",
        subtitle="Statistical distributions, credit risk indicators, and loan purpose trends across submitted assessments",
        badge_text="Live MongoDB Aggregation",
    )

    st.markdown(
        f"""
        <div style="background:{T['surface_alt']}; border:1px solid {T['border']}; border-left:4px solid {T['accent_teal']};
                    border-radius:8px; padding:0.75rem 1.1rem; margin-bottom:1.25rem; font-size:0.83rem; color:{T['text_secondary']};">
            <strong>Data Scope Notice:</strong> The analytics below summarize <strong>application-level predictions</strong>
            stored in MongoDB. They reflect historical model risk assessments rather than verified post-funding loan defaults.
            For offline test-set benchmarks of the trained classifiers, see the <strong>Model Insights</strong> page.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_error:
        st.warning(f"Database Notice: {db_error}")

    if df.empty:
        render_empty_state(
            title="No Assessment Data Available for Analytics",
            message="Submit at least one loan application on the New Loan Assessment page to generate portfolio charts and statistical summaries.",
            icon="◔",
        )
        return

    total = len(df)
    approved = int((df["prediction"] == 0).sum())
    denied = int((df["prediction"] == 1).sum())
    approval_rate = (approved / total * 100.0) if total else 0.0
    avg_risk_score = float(df["probability"].mean() * 100.0) if total else 0.0

    # KPI Row (preserving all 5 KPIs from the original analytics page)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Total Submitted", f"{total:,}", "All logged assessments", "▤")
    with c2:
        render_kpi_card("Low Risk (Approved)", f"{approved:,}", "Predicted Class 0", "✔", "#10B981")
    with c3:
        render_kpi_card("High Risk (Denied)", f"{denied:,}", "Predicted Class 1", "▲", "#EF4444")
    with c4:
        render_kpi_card("Low-Risk Rate", f"{approval_rate:.1f}%", "Portfolio approval share", "◈", "#2563EB")
    with c5:
        render_kpi_card("Avg Risk Score", f"{avg_risk_score:.2f}%", "Mean Class-1 probability", "◔", T["accent_teal"])

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Row 1 Charts: Assessment Timeline + FICO Distribution by Risk Category
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Assessment Volume Over Time</div>
                        <div class="lp-card-subtitle">Daily application submissions grouped by predicted risk category</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        valid_ts_df = df.dropna(subset=["date_only"])
        if not valid_ts_df.empty:
            trend_df = (
                valid_ts_df.groupby(["date_only", "risk_label"], as_index=False)
                .size()
                .rename(columns={"size": "Applications", "date_only": "Date"})
            )
            fig_trend = px.bar(
                trend_df,
                x="Date",
                y="Applications",
                color="risk_label",
                barmode="stack",
                color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
                labels={"risk_label": "Risk Category"},
            )
            fig_trend = apply_chart_theme(fig_trend, height=310)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.caption("No valid timestamps available for trend visualization.")
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">FICO Score vs. Predicted Risk Category</div>
                        <div class="lp-card-subtitle">Distribution of applicant FICO credit scores across risk outcomes</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        fig_fico = px.box(
            df,
            x="risk_label",
            y="fico",
            color="risk_label",
            points="all",
            color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
            labels={"risk_label": "Predicted Risk Category", "fico": "FICO Credit Score"},
        )
        fig_fico = apply_chart_theme(fig_fico, height=310)
        fig_fico.update_layout(showlegend=False)
        st.plotly_chart(fig_fico, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2 Charts: Interest Rate vs. Risk Probability Scatter + DTI Distribution
    ch3, ch4 = st.columns(2)
    with ch3:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Interest Rate vs. Predicted Class-1 Probability</div>
                        <div class="lp-card-subtitle">Relationship between loan pricing, FICO, and model default probability</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        fig_scatter = px.scatter(
            df,
            x="int.rate",
            y="prob_pct",
            color="risk_label",
            size="installment",
            hover_data=["display_name", "purpose_label", "fico", "dti"],
            color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
            labels={
                "int.rate": "Interest Rate (Decimal)",
                "prob_pct": "Class-1 Probability (%)",
                "risk_label": "Risk Category",
            },
        )
        fig_scatter.add_hline(y=50.0, line_dash="dash", line_color="#F59E0B", annotation_text="50% Threshold")
        fig_scatter = apply_chart_theme(fig_scatter, height=310)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch4:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Loan Purpose Breakdown</div>
                        <div class="lp-card-subtitle">Application count and mean predicted Class-1 probability by purpose</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        p_summary = (
            df.groupby("purpose_label", as_index=False)
            .agg(Count=("prediction", "count"), MeanProb=("prob_pct", "mean"))
            .sort_values(by="Count", ascending=True)
        )
        fig_purp_prob = px.bar(
            p_summary,
            x="Count",
            y="purpose_label",
            orientation="h",
            color="MeanProb",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            labels={"purpose_label": "Purpose", "Count": "Applications", "MeanProb": "Mean Prob %"},
        )
        fig_purp_prob = apply_chart_theme(fig_purp_prob, height=310)
        st.plotly_chart(fig_purp_prob, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3 Tables: Financial Averages + Credit Risk Indicators (Preserving all original metrics)
    t_col1, t_col2 = st.columns([6, 6])
    with t_col1:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Financial Attribute Summary Statistics</div>
                        <div class="lp-card-subtitle">Portfolio mean, median, and range across submitted loan applications</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        fin_rows = [
            ("Average FICO Score", f"{df['fico'].mean():.0f}", f"{df['fico'].median():.0f}", f"{int(df['fico'].min())} – {int(df['fico'].max())}"),
            ("Average DTI Ratio", f"{df['dti'].mean():.1f}%", f"{df['dti'].median():.1f}%", f"{df['dti'].min():.1f}% – {df['dti'].max():.1f}%"),
            ("Average Interest Rate", f"{df['int.rate'].mean():.3f}", f"{df['int.rate'].median():.3f}", f"{df['int.rate'].min():.3f} – {df['int.rate'].max():.3f}"),
            ("Average Log Annual Income", f"{df['log.annual.inc'].mean():.2f}", f"{df['log.annual.inc'].median():.2f}", f"{df['log.annual.inc'].min():.2f} – {df['log.annual.inc'].max():.2f}"),
            ("Average Monthly Installment", f"${df['installment'].mean():,.2f}", f"${df['installment'].median():,.2f}", f"${df['installment'].min():,.0f} – ${df['installment'].max():,.0f}"),
            ("Average Revolving Balance", f"${df['revol.bal'].mean():,.2f}", f"${df['revol.bal'].median():,.2f}", f"${df['revol.bal'].min():,.0f} – ${df['revol.bal'].max():,.0f}"),
            ("Average Revolving Util", f"{df['revol.util'].mean():.1f}%", f"{df['revol.util'].median():.1f}%", f"{df['revol.util'].min():.1f}% – {df['revol.util'].max():.1f}%"),
        ]
        fin_html = "".join(
            f"<tr><td>{metric}</td><td class='lp-mono'><strong>{mean_v}</strong></td>"
            f"<td class='lp-mono'>{med_v}</td><td class='lp-mono'>{rng_v}</td></tr>"
            for metric, mean_v, med_v, rng_v in fin_rows
        )
        st.markdown(
            f"""
            <div class="lp-table-wrap" style="margin-bottom:0;">
                <table class="lp-table">
                    <thead>
                        <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Min – Max</th></tr>
                    </thead>
                    <tbody>{fin_html}</tbody>
                </table>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col2:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Credit Risk Indicators &amp; Purpose Breakdown</div>
                        <div class="lp-card-subtitle">Incidence of bureau risk flags and outcomes by loan purpose</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        risk_rows = [
            ("High Inquiries (≥3 in 6m)", int((df["inq.last.6mths"] >= 3).sum())),
            ("Delinquencies (2y > 0)", int((df["delinq.2yrs"] > 0).sum())),
            ("Public Records (> 0)", int((df["pub.rec"] > 0).sum())),
            ("Does Not Meet Credit Policy (0)", int((df["credit.policy"] == 0).sum())),
        ]
        risk_html = "".join(
            f"<tr><td>{lbl}</td><td class='lp-mono'><strong>{cnt}</strong></td>"
            f"<td class='lp-mono'>{cnt / total * 100:.1f}%</td></tr>"
            for lbl, cnt in risk_rows
        )
        st.markdown(
            f"""
            <div class="lp-table-wrap">
                <table class="lp-table">
                    <thead>
                        <tr><th>Risk Factor</th><th>Applications</th><th>% of Portfolio</th></tr>
                    </thead>
                    <tbody>{risk_html}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Purpose table
        pc = df["purpose_label"].value_counts().reset_index()
        pc.columns = ["Purpose", "Count"]
        p_rows = []
        for _, r in pc.iterrows():
            p_name = r["Purpose"]
            cnt = int(r["Count"])
            app_cnt = int(((df["purpose_label"] == p_name) & (df["prediction"] == 0)).sum())
            den_cnt = int(((df["purpose_label"] == p_name) & (df["prediction"] == 1)).sum())
            p_rows.append(
                f"<tr><td>{p_name}</td><td class='lp-mono'>{cnt}</td>"
                f"<td class='lp-mono'>{cnt / total * 100:.1f}%</td>"
                f"<td><span class='lp-badge-low'>{app_cnt} Low</span> "
                f"<span class='lp-badge-high'>{den_cnt} High</span></td></tr>"
            )
        st.markdown(
            f"""
            <div class="lp-table-wrap" style="margin-bottom:0;">
                <table class="lp-table">
                    <thead>
                        <tr><th>Purpose</th><th>Count</th><th>Share</th><th>Outcome Breakdown</th></tr>
                    </thead>
                    <tbody>{"".join(p_rows)}</tbody>
                </table>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL INSIGHTS PAGE
# ═══════════════════════════════════════════════════════════════════════════
def render_model_insights_page() -> None:
    """Render offline model evaluation benchmarks from existing reports/ artifacts."""
    T = get_theme()
    render_page_header(
        eyebrow="Offline Evaluation Artifacts",
        title="Model Insights & Ensemble Benchmark Reports",
        subtitle="Performance comparison across the 7 tuned base classifiers, 5-fold cross-validation, and ROC/PR curves",
        badge_text="Read-Only Evaluation Artifacts",
    )

    comp_df = load_model_comparison()
    cv_df = load_cross_validation_results()
    hp_df = load_hyperparameter_results()
    cls_reports = load_classification_reports()

    if comp_df.empty and cv_df.empty:
        render_empty_state(
            title="Evaluation Artifacts Not Found",
            message="No evaluation CSV files were found in the reports/ directory.",
            icon="❖",
        )
        return

    # Architecture & Methodology Disclosure Banner
    st.markdown(
        f"""
        <div class="lp-card" style="background: linear-gradient(135deg, {T['surface']} 0%, {T['surface_alt']} 100%);">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:{T['accent_teal']}; margin-bottom:0.3rem;">
                Ensemble Architecture &amp; Artifact Provenance
            </div>
            <div style="font-size:0.88rem; color:{T['text_secondary']}; line-height:1.6;">
                Live predictions in <strong>LoanPredict</strong> are generated by <code>models/stacking_classifier.pkl</code>,
                a two-level <strong>StackingClassifier</strong> combining all 7 tuned base estimators
                (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gaussian Naive Bayes, and XGBoost)
                using 5-fold cross-validated <code>predict_proba</code> outputs fed into a <strong>Logistic Regression</strong> meta-learner.
                <br><br>
                <em>Evaluation Note:</em> The offline benchmark reports in <code>reports/model_comparison.csv</code> and
                <code>reports/cross_validation_results.csv</code> record test-set and 5-fold cross-validation metrics for the
                <strong>7 individual tuned base classifiers</strong>. The Stacking Classifier itself was not separately logged in
                <code>reports/model_comparison.csv</code> during pipeline execution, so its standalone test metrics are not fabricated here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Benchmark KPIs from comp_df and cv_df
    if not comp_df.empty:
        best_roc_row = comp_df.sort_values(by="ROC AUC", ascending=False).iloc[0]
        best_acc_row = comp_df.sort_values(by="Accuracy", ascending=False).iloc[0]
        best_f1_row = comp_df.sort_values(by="F1 Score", ascending=False).iloc[0]
        best_cv_row = cv_df.sort_values(by="Mean ROC AUC", ascending=False).iloc[0] if not cv_df.empty else None

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                label="Best Test ROC-AUC",
                value=f"{best_roc_row['ROC AUC']:.4f}",
                subtitle=f"Model: {best_roc_row['Model']}",
                icon="❖",
                accent_color="#10B981",
            )
        with k2:
            render_kpi_card(
                label="Best 5-Fold CV ROC-AUC",
                value=f"{best_cv_row['Mean ROC AUC']:.4f}" if best_cv_row is not None else "—",
                subtitle=f"Model: {best_cv_row['Model']} (±{best_cv_row['Standard Deviation']:.4f})" if best_cv_row is not None else "—",
                icon="◈",
                accent_color=T["accent_teal"],
            )
        with k3:
            render_kpi_card(
                label="Highest Test Accuracy",
                value=f"{best_acc_row['Accuracy'] * 100:.2f}%",
                subtitle=f"Model: {best_acc_row['Model']}",
                icon="✔",
            )
        with k4:
            render_kpi_card(
                label="Highest Class-1 F1 Score",
                value=f"{best_f1_row['F1 Score']:.4f}",
                subtitle=f"Model: {best_f1_row['Model']}",
                icon="◔",
            )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Interactive Charts: Test ROC-AUC vs Accuracy + 5-Fold CV ROC-AUC
    c1, c2 = st.columns(2)
    with c1:
        if not comp_df.empty:
            st.markdown(
                """
                <div class="lp-card">
                    <div class="lp-card-header">
                        <div>
                            <div class="lp-card-title">Base Classifier Test-Set Performance</div>
                            <div class="lp-card-subtitle">Comparison of ROC-AUC and Accuracy from reports/model_comparison.csv</div>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            melted = comp_df.melt(
                id_vars=["Model"],
                value_vars=["ROC AUC", "Accuracy", "F1 Score"],
                var_name="Metric",
                value_name="Score",
            )
            fig_comp = px.bar(
                melted,
                x="Score",
                y="Model",
                color="Metric",
                barmode="group",
                orientation="h",
                color_discrete_map={"ROC AUC": "#2563EB", "Accuracy": "#0EA5E9", "F1 Score": "#10B981"},
            )
            fig_comp = apply_chart_theme(fig_comp, height=330)
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        if not cv_df.empty:
            st.markdown(
                """
                <div class="lp-card">
                    <div class="lp-card-header">
                        <div>
                            <div class="lp-card-title">5-Fold Stratified Cross-Validation (ROC-AUC)</div>
                            <div class="lp-card-subtitle">Mean ROC-AUC with ±1 standard deviation error bars across 5 folds</div>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            cv_sorted = cv_df.sort_values(by="Mean ROC AUC", ascending=True)
            fig_cv = go.Figure(
                data=[
                    go.Bar(
                        x=cv_sorted["Mean ROC AUC"],
                        y=cv_sorted["Model"],
                        orientation="h",
                        error_x=dict(type="data", array=cv_sorted["Standard Deviation"], visible=True, color="#94A3B8"),
                        marker_color="#0EA5E9",
                        text=cv_sorted["Mean ROC AUC"].apply(lambda v: f"{v:.4f}"),
                        textposition="auto",
                    )
                ]
            )
            fig_cv = apply_chart_theme(fig_cv, height=330)
            fig_cv.update_xaxes(range=[0.45, 0.75], title="Mean ROC-AUC (5-Fold CV)")
            st.plotly_chart(fig_cv, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Full Test Set Comparison Table
    if not comp_df.empty:
        st.markdown(
            """
            <div class="lp-card">
                <div class="lp-card-header">
                    <div>
                        <div class="lp-card-title">Offline Test-Set Evaluation Table (reports/model_comparison.csv)</div>
                        <div class="lp-card-subtitle">Sorted by Test ROC-AUC across the 1,916 held-out test samples</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        comp_rows = []
        for _, r in comp_df.iterrows():
            comp_rows.append(
                f"<tr>"
                f"<td><strong>{r['Model']}</strong></td>"
                f"<td class='lp-mono'>{r['Accuracy']*100:.2f}%</td>"
                f"<td class='lp-mono'>{r['Precision']:.4f}</td>"
                f"<td class='lp-mono'>{r['Recall']:.4f}</td>"
                f"<td class='lp-mono'>{r['F1 Score']:.4f}</td>"
                f"<td class='lp-mono'><strong>{r['ROC AUC']:.4f}</strong></td>"
                f"<td class='lp-mono'>{r['Log Loss']:.4f}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="lp-table-wrap" style="margin-bottom:0;">
                <table class="lp-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision (Class 1)</th>
                            <th>Recall (Class 1)</th>
                            <th>F1 Score (Class 1)</th>
                            <th>ROC-AUC</th>
                            <th>Log Loss</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(comp_rows)}</tbody>
                </table>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Cross-Validation & Hyperparameter Tuning Tables
    tab_cv, tab_hp, tab_cls = st.tabs(
        [
            "5-Fold Cross-Validation Details",
            "RandomizedSearchCV Tuned Hyperparameters",
            "Per-Model Classification Reports (JSON)",
        ]
    )

    with tab_cv:
        if not cv_df.empty:
            st.dataframe(
                cv_df.style.format(
                    {
                        "Mean ROC AUC": "{:.4f}",
                        "Standard Deviation": "{:.4f}",
                        "Fold 1": "{:.4f}",
                        "Fold 2": "{:.4f}",
                        "Fold 3": "{:.4f}",
                        "Fold 4": "{:.4f}",
                        "Fold 5": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab_hp:
        if not hp_df.empty:
            st.dataframe(hp_df, use_container_width=True, hide_index=True)

    with tab_cls:
        if cls_reports:
            selected_model = st.selectbox(
                "Select Base Classifier to Inspect Detailed Classification Report",
                options=list(cls_reports.keys()),
                key="cls_report_model_select",
            )
            rep = cls_reports[selected_model]
            rep_rows = []
            for label_key, display_lbl in [
                ("0", "Class 0 (Low Risk / Paid)"),
                ("1", "Class 1 (High Risk / Not Fully Paid)"),
                ("macro avg", "Macro Average"),
                ("weighted avg", "Weighted Average"),
            ]:
                if label_key in rep and isinstance(rep[label_key], dict):
                    d = rep[label_key]
                    rep_rows.append(
                        {
                            "Class / Summary": display_lbl,
                            "Precision": f"{d.get('precision', 0.0):.4f}",
                            "Recall": f"{d.get('recall', 0.0):.4f}",
                            "F1-Score": f"{d.get('f1-score', 0.0):.4f}",
                            "Support": int(d.get("support", 0)),
                        }
                    )
            st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

    # Saved ROC & Precision-Recall Curve Plots
    roc_path = os.path.join("reports", "roc_curve.png")
    pr_path = os.path.join("reports", "precision_recall_curve.png")
    if os.path.exists(roc_path) or os.path.exists(pr_path):
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if os.path.exists(roc_path):
                st.markdown(
                    """
                    <div class="lp-card">
                        <div class="lp-card-header">
                            <div>
                                <div class="lp-card-title">Receiver Operating Characteristic (ROC) Curves</div>
                                <div class="lp-card-subtitle">Artifact: reports/roc_curve.png</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(roc_path, use_container_width=True)
        with img_col2:
            if os.path.exists(pr_path):
                st.markdown(
                    """
                    <div class="lp-card">
                        <div class="lp-card-header">
                            <div>
                                <div class="lp-card-title">Precision-Recall Curves</div>
                                <div class="lp-card-subtitle">Artifact: reports/precision_recall_curve.png</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(pr_path, use_container_width=True)

