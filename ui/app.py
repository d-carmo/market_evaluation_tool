"""Streamlit UI entry point: streamlit run ui/app.py"""
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so all project packages resolve
# correctly regardless of which directory Streamlit was launched from.
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ui.api_client import APIClient
from ui.pages import overview, forecasts, accuracy

st.set_page_config(
    page_title="Market Evaluation Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL is read from environment only — not overridable from the UI
# to prevent SSRF. Set API_BASE_URL in your .env file.
_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_API_TOKEN = os.getenv("API_TOKEN", "")
_REFRESH_HOURS = int(os.getenv("ANALYSIS_REFRESH_HOURS", "3"))

# Auto-refresh the UI on the same interval as the server-side analysis job.
st_autorefresh(interval=_REFRESH_HOURS * 60 * 60 * 1000, key="ui_autorefresh")

client = APIClient(_API_BASE_URL, api_token=_API_TOKEN)

_PAGES = ["Overview", "Forecasts", "Accuracy"]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

# Sync the radio widget's own key to current_page before it renders,
# so programmatic navigation (e.g. clicking a ticker) is reflected in the sidebar.
st.session_state["_nav_radio"] = st.session_state["current_page"]

with st.sidebar:
    st.title("📈 Market Eval")
    st.divider()
    page = st.radio(
        "Navigation",
        _PAGES,
        index=_PAGES.index(st.session_state["current_page"]),
        label_visibility="collapsed",
        on_change=lambda: st.session_state.update(current_page=st.session_state["_nav_radio"]),
        key="_nav_radio",
    )
    page = st.session_state["current_page"]
    st.divider()

    # Connection indicator
    try:
        client.health()
        st.success("API connected", icon="✅")
    except Exception:
        st.error("API unreachable", icon="❌")
        st.caption("Check that the API server is running.")

# Page routing
if page == "Overview":
    overview.render(client)
elif page == "Forecasts":
    forecasts.render(client)
elif page == "Accuracy":
    accuracy.render(client)
