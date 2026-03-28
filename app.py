# =============================================================================
# app.py — InfyVision: AI-Powered Stock Intelligence SaaS
# =============================================================================
# Main Streamlit entry point. Sets up page config and redirects to Landing page.
# Run with:  streamlit run app.py
# =============================================================================

import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="InfyVision — AI Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state defaults ───────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ── Always redirect root (/) to the Landing page ────────────────────────────
st.switch_page("pages/1_Landing.py")
