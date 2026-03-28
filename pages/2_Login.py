# =============================================================================
# pages/2_Login.py — Login / Register page  (JSON-backed auth)
# =============================================================================

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.app_utils import inject_css, authenticate_user, register_user

st.set_page_config(
    page_title="InfyVision — Login",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

# Hide sidebar
st.markdown("""
<style>
section[data-testid="stSidebar"]{display:none}
[data-testid="stSidebarCollapsedControl"]{display:none}
</style>
""", unsafe_allow_html=True)

# Session defaults
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# If already logged in, redirect to dashboard
if st.session_state.logged_in:
    st.switch_page("pages/3_Dashboard.py")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 10px">
    <span style="font-size:40px">📈</span>
    <h1 style="font-size:32px;font-weight:800;margin:8px 0 0;
       background:linear-gradient(135deg,#6366f1,#8b5cf6);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">
       InfyVision</h1>
    <p style="color:#64748b;font-size:14px;margin:4px 0 0">AI-Powered Stock Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ── Centered form container ──────────────────────────────────────────────────
_, center, _ = st.columns([1, 1.2, 1])

with center:
    tab_login, tab_register = st.tabs(["🔐  Login", "📝  Register"])

    # ── Login tab ────────────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            st.markdown("##### Welcome back")
            lu = st.text_input("Username", key="login_user", placeholder="Enter username")
            lp = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
            submit_login = st.form_submit_button("Login →", use_container_width=True)

        if submit_login:
            if not lu or not lp:
                st.error("Please fill in both fields.")
            elif authenticate_user(lu, lp):
                st.session_state.logged_in = True
                st.session_state.username = lu
                st.success(f"Welcome back, **{lu}**!")
                st.switch_page("pages/3_Dashboard.py")
            else:
                st.error("Invalid username or password.")

    # ── Register tab ─────────────────────────────────────────────────────────
    with tab_register:
        with st.form("register_form"):
            st.markdown("##### Create an account")
            ru = st.text_input("Choose Username", key="reg_user", placeholder="Pick a username")
            rp = st.text_input("Choose Password", type="password", key="reg_pass", placeholder="Min 4 characters")
            rp2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Re-enter password")
            submit_reg = st.form_submit_button("Register →", use_container_width=True)

        if submit_reg:
            if not ru or not rp:
                st.error("All fields are required.")
            elif len(rp) < 4:
                st.error("Password must be at least 4 characters.")
            elif rp != rp2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(ru, rp)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# ── Back to landing ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, c2, _ = st.columns([1, 1, 1])
with c2:
    if st.button("← Back to Home", use_container_width=True, key="back_home"):
        st.switch_page("pages/1_Landing.py")
