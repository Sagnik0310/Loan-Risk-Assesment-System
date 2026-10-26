"""
Design system, color tokens, and CSS stylesheet injection for LoanPredict.
Provides a cohesive institutional fintech aesthetic in both Dark and Light modes.
"""

from typing import Dict
import streamlit as st


DARK_THEME: Dict[str, str] = {
    "name": "dark",
    "app_bg": "#0B1120",
    "sidebar_bg": "#0F172A",
    "sidebar_border": "#1E293B",
    "surface": "#111827",
    "surface_alt": "#162032",
    "surface_hover": "#1E293B",
    "border": "#1E293B",
    "border_strong": "#334155",
    "input_bg": "#0F172A",
    "input_border": "#334155",
    "text_primary": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "text_muted": "#94A3B8",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_soft": "rgba(37, 99, 235, 0.14)",
    "accent_teal": "#0EA5E9",
    "low_risk_bg": "rgba(16, 185, 129, 0.12)",
    "low_risk_border": "rgba(16, 185, 129, 0.35)",
    "low_risk_text": "#34D399",
    "high_risk_bg": "rgba(239, 68, 68, 0.12)",
    "high_risk_border": "rgba(239, 68, 68, 0.35)",
    "high_risk_text": "#F87171",
    "warn_bg": "rgba(245, 158, 11, 0.12)",
    "warn_border": "rgba(245, 158, 11, 0.35)",
    "warn_text": "#FBBF24",
    "shadow": "0 4px 20px -2px rgba(0, 0, 0, 0.35)",
    "chart_grid": "#1E293B",
}

LIGHT_THEME: Dict[str, str] = {
    "name": "light",
    "app_bg": "#F8FAFC",
    "sidebar_bg": "#0F172A",
    "sidebar_border": "#1E293B",
    "surface": "#FFFFFF",
    "surface_alt": "#F1F5F9",
    "surface_hover": "#E2E8F0",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "input_bg": "#FFFFFF",
    "input_border": "#CBD5E1",
    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "accent": "#1D4ED8",
    "accent_hover": "#1E40AF",
    "accent_soft": "rgba(29, 78, 216, 0.08)",
    "accent_teal": "#0284C7",
    "low_risk_bg": "#ECFDF5",
    "low_risk_border": "#A7F3D0",
    "low_risk_text": "#047857",
    "high_risk_bg": "#FEF2F2",
    "high_risk_border": "#FECACA",
    "high_risk_text": "#B91C1C",
    "warn_bg": "#FFFBEB",
    "warn_border": "#FDE68A",
    "warn_text": "#B45309",
    "shadow": "0 2px 10px -2px rgba(15, 23, 42, 0.06)",
    "chart_grid": "#E2E8F0",
}


def get_theme() -> Dict[str, str]:
    """Return active color palette dictionary based on session state."""
    is_dark = st.session_state.get("dark_mode", True)
    return DARK_THEME if is_dark else LIGHT_THEME


def inject_global_styles() -> Dict[str, str]:
    """Inject the complete fintech design system CSS into the Streamlit page."""
    T = get_theme()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ── Global Reset & Chrome ── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        [data-testid="stHeader"] {{
            background: transparent !important;
            height: 2.5rem !important;
        }}

        .stApp {{
            background-color: {T['app_bg']} !important;
            color: {T['text_primary']} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}

        .block-container {{
            max-width: 1380px !important;
            padding-top: 1.4rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
        }}

        /* ── Professional Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {T['sidebar_bg']} !important;
            border-right: 1px solid {T['sidebar_border']} !important;
            min-width: 270px !important;
            max-width: 285px !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {T['sidebar_bg']} !important;
            padding-top: 1.2rem !important;
        }}
        [data-testid="stSidebar"] * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* Sidebar Navigation Buttons */
        [data-testid="stSidebar"] .stButton > button {{
            width: 100% !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding: 0.62rem 0.95rem !important;
            height: 42px !important;
            border-radius: 8px !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            margin-bottom: 0.15rem !important;
            transition: all 0.15s ease !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
            background-color: transparent !important;
            color: #94A3B8 !important;
            border: 1px solid transparent !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
            background-color: rgba(255, 255, 255, 0.06) !important;
            color: #F8FAFC !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
            transform: none !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #3B82F6 !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 10px rgba(37, 99, 235, 0.3) !important;
        }}

        /* ── Main Content Buttons ── */
        .block-container .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {T['accent']} 0%, {T['accent_hover']} 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid {T['accent']} !important;
            border-radius: 8px !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            padding: 0.65rem 1.35rem !important;
            height: 46px !important;
            transition: all 0.16s ease !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.22) !important;
        }}
        .block-container .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.07);
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.32) !important;
        }}
        .block-container .stButton > button[kind="secondary"] {{
            background-color: {T['surface']} !important;
            color: {T['text_primary']} !important;
            border: 1px solid {T['border_strong']} !important;
            border-radius: 8px !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            height: 40px !important;
            transition: all 0.15s ease !important;
        }}
        .block-container .stButton > button[kind="secondary"]:hover {{
            background-color: {T['surface_hover']} !important;
            border-color: {T['accent']} !important;
        }}

        /* ── Form Inputs & Controls ── */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input {{
            background-color: {T['input_bg']} !important;
            border: 1px solid {T['input_border']} !important;
            border-radius: 8px !important;
            color: {T['text_primary']} !important;
            font-size: 0.9rem !important;
            padding: 0.48rem 0.75rem !important;
        }}
        .stSelectbox > div > div {{
            background-color: {T['input_bg']} !important;
            border: 1px solid {T['input_border']} !important;
            border-radius: 8px !important;
            color: {T['text_primary']} !important;
            font-size: 0.9rem !important;
        }}
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {T['accent']} !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.22) !important;
        }}
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stDateInput label,
        .stRadio label,
        .stCheckbox label {{
            color: {T['text_primary']} !important;
            font-size: 0.83rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
        }}

        /* ── Top Page Header ── */
        .lp-page-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 1.25rem 1.5rem;
            background: {T['surface']};
            border: 1px solid {T['border']};
            border-radius: 12px;
            margin-bottom: 1.4rem;
            box-shadow: {T['shadow']};
        }}
        .lp-header-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {T['accent_teal']};
            margin-bottom: 0.25rem;
        }}
        .lp-header-title {{
            font-size: 1.55rem;
            font-weight: 750;
            color: {T['text_primary']};
            margin: 0;
            line-height: 1.25;
            letter-spacing: -0.02em;
        }}
        .lp-header-sub {{
            font-size: 0.88rem;
            color: {T['text_muted']};
            margin-top: 0.25rem;
        }}
        .lp-header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: {T['surface_alt']};
            border: 1px solid {T['border_strong']};
            color: {T['text_secondary']};
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.4rem 0.85rem;
            border-radius: 999px;
            white-space: nowrap;
        }}

        /* ── Surface Cards ── */
        .lp-card {{
            background: {T['surface']};
            border: 1px solid {T['border']};
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: {T['shadow']};
        }}
        .lp-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid {T['border']};
        }}
        .lp-card-title {{
            font-size: 1.0rem;
            font-weight: 700;
            color: {T['text_primary']};
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .lp-card-subtitle {{
            font-size: 0.8rem;
            color: {T['text_muted']};
            margin-top: 0.15rem;
        }}

        /* ── KPI Cards ── */
        .lp-kpi-card {{
            background: {T['surface']};
            border: 1px solid {T['border']};
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            box-shadow: {T['shadow']};
            height: 100%;
            transition: border-color 0.15s ease;
        }}
        .lp-kpi-card:hover {{
            border-color: {T['border_strong']};
        }}
        .lp-kpi-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.55rem;
        }}
        .lp-kpi-label {{
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {T['text_muted']};
        }}
        .lp-kpi-icon {{
            font-size: 0.95rem;
            width: 30px;
            height: 30px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: {T['surface_alt']};
            border: 1px solid {T['border']};
        }}
        .lp-kpi-value {{
            font-size: 1.68rem;
            font-weight: 800;
            color: {T['text_primary']};
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }}
        .lp-kpi-sub {{
            font-size: 0.78rem;
            color: {T['text_muted']};
            margin-top: 0.35rem;
        }}

        /* ── Status Badges ── */
        .lp-badge-low {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: {T['low_risk_bg']};
            border: 1px solid {T['low_risk_border']};
            color: {T['low_risk_text']};
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
        }}
        .lp-badge-high {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: {T['high_risk_bg']};
            border: 1px solid {T['high_risk_border']};
            color: {T['high_risk_text']};
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
        }}
        .lp-badge-neutral {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: {T['surface_alt']};
            border: 1px solid {T['border_strong']};
            color: {T['text_secondary']};
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
        }}

        /* ── Prediction Result Banner ── */
        .lp-result-panel-low {{
            background: {T['low_risk_bg']};
            border: 1px solid {T['low_risk_border']};
            border-left: 5px solid #10B981;
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }}
        .lp-result-panel-high {{
            background: {T['high_risk_bg']};
            border: 1px solid {T['high_risk_border']};
            border-left: 5px solid #EF4444;
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }}

        /* ── Structured Data Tables ── */
        .lp-table-wrap {{
            background: {T['surface']};
            border: 1px solid {T['border']};
            border-radius: 10px;
            overflow-x: auto;
            margin-bottom: 1rem;
        }}
        .lp-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        .lp-table th {{
            background: {T['surface_alt']};
            color: {T['text_muted']};
            font-size: 0.73rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.72rem 1rem;
            text-align: left;
            border-bottom: 1px solid {T['border']};
            white-space: nowrap;
        }}
        .lp-table td {{
            color: {T['text_primary']};
            padding: 0.68rem 1rem;
            border-bottom: 1px solid {T['border']};
            vertical-align: middle;
        }}
        .lp-table tr:last-child td {{
            border-bottom: none;
        }}
        .lp-table tr:hover td {{
            background: {T['surface_alt']};
        }}
        .lp-mono {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
        }}

        /* ── Empty State ── */
        .lp-empty-state {{
            background: {T['surface']};
            border: 1px dashed {T['border_strong']};
            border-radius: 12px;
            padding: 2.8rem 1.8rem;
            text-align: center;
            margin: 1rem 0;
        }}

        /* ── Footer ── */
        .lp-footer {{
            margin-top: 3rem;
            padding-top: 1.25rem;
            border-top: 1px solid {T['border']};
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            color: {T['text_muted']};
            font-size: 0.78rem;
        }}

        /* ── Responsive Adjustments ── */
        @media (max-width: 992px) {{
            .block-container {{
                padding-left: 1.1rem !important;
                padding-right: 1.1rem !important;
            }}
            .lp-page-header {{
                padding: 1rem 1.15rem;
            }}
            .lp-header-title {{
                font-size: 1.3rem;
            }}
            .lp-kpi-value {{
                font-size: 1.38rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return T

