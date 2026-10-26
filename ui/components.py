"""
Reusable UI components, navigation sidebar, headers, KPI cards, privacy masking,
and Plotly chart styling for the LoanPredict Fintech Dashboard.
"""

from typing import Dict, Optional
import plotly.graph_objects as go
import streamlit as st

from ui.styles import get_theme


NAV_ITEMS = [
    ("overview", "Overview Dashboard", "◈"),
    ("new_prediction", "New Loan Assessment", "✚"),
    ("history", "Application History", "▤"),
    ("analytics", "Analytics Dashboard", "◔"),
    ("model_insights", "Model Insights", "❖"),
]


def mask_email(email: str) -> str:
    """Mask email address for overview tables while preserving domain context."""
    val = (email or "").strip()
    if not val or "@" not in val:
        return val if val else "—"
    local, domain = val.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[:2] + "***" + local[-1]
    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    """Mask phone number leaving only the last 4 digits visible."""
    val = (phone or "").strip()
    if not val:
        return "—"
    digits = [c for c in val if c.isdigit()]
    if len(digits) <= 4:
        return val
    return f"•••-•••-{''.join(digits[-4:])}"


def render_sidebar(total_records: int, db_connected: bool) -> None:
    """Render the institutional sidebar with brand identity, navigation, and controls."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.25rem 0.35rem 1.1rem; border-bottom: 1px solid #1E293B; margin-bottom: 1rem;">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <div style="width:38px; height:38px; border-radius:9px; background:linear-gradient(135deg,#2563EB,#0EA5E9);
                                display:flex; align-items:center; justify-content:center; color:#FFFFFF; font-weight:800; font-size:1.05rem;
                                box-shadow: 0 2px 10px rgba(37,99,235,0.35);">
                        LP
                    </div>
                    <div>
                        <div style="color:#F8FAFC; font-weight:750; font-size:1.05rem; letter-spacing:-0.01em; line-height:1.15;">
                            LoanPredict
                        </div>
                        <div style="color:#38BDF8; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">
                            Credit Risk Intelligence
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="color:#64748B; font-size:0.68rem; font-weight:700; text-transform:uppercase; '
            'letter-spacing:0.08em; padding:0 0.35rem 0.4rem;">Workspace Navigation</div>',
            unsafe_allow_html=True,
        )

        current_page = st.session_state.get("page", "overview")
        for page_key, label, icon in NAV_ITEMS:
            is_active = current_page == page_key
            if st.button(
                f"{icon}   {label}",
                key=f"nav_{page_key}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if current_page != page_key:
                    st.session_state.page = page_key
                    st.rerun()

        st.markdown(
            '<div style="border-top:1px solid #1E293B; margin: 1.1rem 0 0.9rem;"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="color:#64748B; font-size:0.68rem; font-weight:700; text-transform:uppercase; '
            'letter-spacing:0.08em; padding:0 0.35rem 0.45rem;">Interface Preferences</div>',
            unsafe_allow_html=True,
        )

        theme_label = "Switch to Light Mode" if st.session_state.get("dark_mode", True) else "Switch to Dark Mode"
        if st.button(f"◐   {theme_label}", key="sidebar_theme_btn", type="secondary", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.get("dark_mode", True)
            st.rerun()

        if st.button("ℹ   Platform Overview Modal", key="sidebar_welcome_btn", type="secondary", use_container_width=True):
            st.session_state.welcome_shown = False
            st.rerun()

        # System & Model Status panel
        status_dot = "#10B981" if db_connected else "#F59E0B"
        status_text = "MongoDB Connected" if db_connected else "Database Offline"
        st.markdown(
            f"""
            <div style="margin-top:1.25rem; background:#0B1120; border:1px solid #1E293B; border-radius:10px; padding:0.85rem 0.95rem;">
                <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:#64748B; margin-bottom:0.5rem;">
                    System Status
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:#E2E8F0; margin-bottom:0.35rem;">
                    <span style="width:8px; height:8px; border-radius:50%; background:{status_dot}; display:inline-block;"></span>
                    <span>{status_text}</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:#94A3B8; margin-bottom:0.35rem;">
                    <span style="width:8px; height:8px; border-radius:50%; background:#38BDF8; display:inline-block;"></span>
                    <span>Stacking Ensemble Ready</span>
                </div>
                <div style="font-size:0.74rem; color:#64748B; padding-top:0.35rem; border-top:1px solid #1E293B; margin-top:0.4rem;">
                    Stored Assessments: <strong style="color:#CBD5E1;">{total_records}</strong>
                </div>
            </div>

            <div style="margin-top:0.9rem; display:flex; align-items:center; gap:0.65rem; padding:0.6rem 0.4rem;">
                <div style="width:32px; height:32px; border-radius:50%; background:#1E293B; border:1px solid #334155;
                            display:flex; align-items:center; justify-content:center; color:#38BDF8; font-size:0.75rem; font-weight:700;">
                    CA
                </div>
                <div>
                    <div style="color:#E2E8F0; font-size:0.8rem; font-weight:600; line-height:1.2;">Credit Risk Analyst</div>
                    <div style="color:#64748B; font-size:0.7rem;">Underwriting Desk · v1.0.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    badge_text: Optional[str] = None,
) -> None:
    """Render a compact, consistent page header bar."""
    badge_html = f'<div class="lp-header-badge">{badge_text}</div>' if badge_text else ""
    st.markdown(
        f"""
        <div class="lp-page-header">
            <div>
                <div class="lp-header-eyebrow">{eyebrow}</div>
                <h1 class="lp-header-title">{title}</h1>
                <div class="lp-header-sub">{subtitle}</div>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    icon: str = "◈",
    accent_color: Optional[str] = None,
) -> None:
    """Render an institutional KPI metric card."""
    T = get_theme()
    val_color = accent_color if accent_color else T["text_primary"]
    st.markdown(
        f"""
        <div class="lp-kpi-card">
            <div class="lp-kpi-top">
                <span class="lp-kpi-label">{label}</span>
                <span class="lp-kpi-icon">{icon}</span>
            </div>
            <div class="lp-kpi-value" style="color:{val_color};">{value}</div>
            <div class="lp-kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str, icon: str = "▤") -> None:
    """Render a clean, non-cluttered empty state card."""
    T = get_theme()
    st.markdown(
        f"""
        <div class="lp-empty-state">
            <div style="font-size:1.8rem; color:{T['accent_teal']}; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-size:1.05rem; font-weight:700; color:{T['text_primary']}; margin-bottom:0.35rem;">
                {title}
            </div>
            <div style="font-size:0.86rem; color:{T['text_muted']}; max-width:520px; margin:0 auto;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the application footer with developer attribution and version badge."""
    st.markdown(
        """
        <div class="lp-footer">
            <div>
                <strong>LoanPredict</strong> · Enterprise Credit Risk Assessment Platform · © 2026
            </div>
            <div>
                Founded &amp; Developed by <strong>Sagnik Ghosh &amp; Anantika Ghosh</strong> ·
                <span class="lp-mono">v1.0.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply consistent fintech colors, typography, and margins to a Plotly figure."""
    T = get_theme()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=T["text_secondary"],
            size=12,
        ),
        margin=dict(l=16, r=16, t=36, b=24),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=T["text_secondary"], size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", family="Inter, sans-serif", size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=T["chart_grid"],
        zerolinecolor=T["chart_grid"],
        tickfont=dict(color=T["text_muted"], size=11),
        title_font=dict(color=T["text_secondary"], size=12),
    )
    fig.update_yaxes(
        gridcolor=T["chart_grid"],
        zerolinecolor=T["chart_grid"],
        tickfont=dict(color=T["text_muted"], size=11),
        title_font=dict(color=T["text_secondary"], size=12),
    )
    return fig

