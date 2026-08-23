from __future__ import annotations

import base64
import html
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================
FEATURES = [
    "n_rpm",
    "fz_mm_per_tooth",
    "ap_mm",
    "ae_mm",
    "eps_r_um",
    "eps_a_um",
]
MILLING_MODE = "milling_mode"
MILLING_MODES = ["down"]
MODE_FEATURES = []
MODEL_FEATURES = FEATURES.copy()
TARGETS = ["Sa_um", "Sz_um"]

# Frozen D6 EndMill V1 design domain.
DESIGN_RANGES = {
    "n_rpm": {"min": 1600.0, "max": 8000.0},
    "fz_mm_per_tooth": {"min": 0.02, "max": 0.12},
    "ap_mm": {"min": 0.30, "max": 1.50},
    "ae_mm": {"min": 0.30, "max": 3.00},
    "eps_r_um": {"min": -10.0, "max": 10.0},
    "eps_a_um": {"min": 0.0, "max": 5.0},
}
TOOL_DIAMETER_MM = 6.0
NUM_FLUTES = 4
HELIX_ANGLE_DEG = 45.0
FLUTE_LENGTH_MM = 13.0
BOTTOM_DISH_ANGLE_DEG = 2.0
AE_PARTIAL_MIN_MM = 0.3
AE_PARTIAL_MAX_MM = 3.0
AE_SLOT_MM = 6.0

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
LOGO_PATH = APP_DIR / "logo.png"  # User brand logo: keep logo.png beside this app file.

_favicon_path = LOGO_PATH if LOGO_PATH.is_file() else ASSETS_DIR / "millcore_favicon.png"
_page_icon = str(_favicon_path) if _favicon_path.is_file() else "⚙"


# =============================================================================
# Page config and style
# =============================================================================
st.set_page_config(
    page_title="MillTwin-Lite | CNC Surface Intelligence",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",  # native sidebar unused; persistent rail is in the page shell
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --blue: #0A2463;
        --sky: #3E92CC;
        --carbon: #1E1B18;
        --paper: #FFFAFF;
        --hot: #D8315B;
        --blue-050: #EEF6FC;
        --blue-100: #DCECF8;
        --ink-500: #5D5A57;
        --line: rgba(30, 27, 24, 0.28);
        --ok: #0A7B59;
        --warn: #A56412;

        --font-display: "Archivo Black", "Arial Black", sans-serif;
        --font-body: "Inter", "Segoe UI", Arial, sans-serif;
        --font-mono: "IBM Plex Mono", "Cascadia Code", Consolas, monospace;

        --hard-shadow: 6px 6px 0 var(--carbon);
        --small-shadow: 3px 3px 0 var(--carbon);
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
    }

    .stApp {
        color: var(--carbon);
        background-color: var(--paper);
        background-image:
            linear-gradient(rgba(30,27,24,0.055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30,27,24,0.055) 1px, transparent 1px);
        background-size: 32px 32px;
        background-attachment: fixed;
    }

    /* Keep the Streamlit header alive. On Community Cloud the sidebar toggle
       is mounted in the header on narrow/mobile viewports. Hiding the whole
       header can make a collapsed sidebar impossible to reopen. */
    header[data-testid="stHeader"] {
        visibility: visible !important;
        background: transparent !important;
        box-shadow: none !important;
        height: 2.6rem !important;
        z-index: 100000 !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    .block-container {
        max-width: 1540px;
        padding-top: 1.15rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1, h2, h3, h4 {
        font-family: var(--font-display);
        color: var(--carbon);
        letter-spacing: -0.035em;
    }

    p, label, .stCaption {
        color: var(--ink-500);
    }

    a { color: var(--blue); }

    /* ============================= SIDEBAR / MACHINE CONSOLE ============================= */
    [data-testid="stSidebar"] {
        background: var(--blue);
        border-right: 3px solid var(--carbon);
        box-shadow: 5px 0 0 rgba(30,27,24,0.13);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        min-height: 100vh;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] summary {
        color: var(--paper) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: transparent;
        border: 1.5px solid rgba(255,250,255,0.45);
        border-radius: 0 !important;
        overflow: hidden;
        box-shadow: none;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-family: var(--font-mono);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,250,255,0.07) !important;
        color: var(--paper) !important;
        border: 1.5px solid rgba(255,250,255,0.58) !important;
        border-radius: 0 !important;
        font-family: var(--font-mono) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"] {
        background: var(--hot) !important;
        border: 2px solid var(--paper) !important;
        box-shadow: 2px 2px 0 var(--carbon) !important;
    }

    .sidebar-brand {
        margin: 0 0 1rem 0;
        padding: 0 0 1rem 0;
        border-bottom: 1.5px solid rgba(255,250,255,0.45);
    }

    .sidebar-brand__logo-frame {
        background: var(--paper);
        border: 2px solid var(--carbon);
        box-shadow: 4px 4px 0 var(--carbon);
        padding: 0.65rem;
        margin-bottom: 0.9rem;
    }

    .sidebar-brand__logo {
        display: block;
        width: 100%;
        max-height: 116px;
        object-fit: contain;
    }

    .sidebar-brand__mark {
        color: #B9DCF5;
        font-family: var(--font-mono);
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .sidebar-brand__title {
        color: var(--paper);
        font-family: var(--font-display);
        font-size: 1.35rem;
        line-height: 1;
        margin-top: 0.35rem;
        text-transform: uppercase;
    }

    .sidebar-brand__meta {
        color: rgba(255,250,255,0.72);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        margin-top: 0.45rem;
        line-height: 1.45;
    }

    .nameplate {
        margin: 1.5rem 0 0.5rem 0;
        padding: 0.9rem;
        border: 1.5px solid rgba(255,250,255,0.45);
        background: rgba(255,250,255,0.05);
        border-radius: 0;
    }

    .nameplate__label {
        color: rgba(255,250,255,0.65);
        font-family: var(--font-mono);
        font-size: 0.60rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .nameplate__line {
        margin-top: 0.45rem;
        color: var(--paper);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        line-height: 1.55;
    }

    /* ============================= ASYMMETRICAL HERO ============================= */
    .brutal-hero {
        display: grid;
        grid-template-columns: minmax(0, 1.62fr) minmax(310px, 0.72fr);
        border: 2px solid var(--carbon);
        box-shadow: var(--hard-shadow);
        margin-bottom: 1.35rem;
        overflow: hidden;
        background: var(--paper);
    }

    .brutal-hero__main {
        position: relative;
        min-height: 286px;
        padding: 1.65rem 1.8rem 1.55rem 1.8rem;
        background: var(--blue);
        color: var(--paper);
        overflow: hidden;
    }

    .brutal-hero__main::before {
        content: "";
        position: absolute;
        right: 7.5%;
        top: -54px;
        width: 180px;
        height: 180px;
        border: 1px solid rgba(255,250,255,0.27);
        transform: rotate(45deg);
    }

    .brutal-hero__main::after {
        content: "MILLING // SURFACE // INTELLIGENCE";
        position: absolute;
        right: -132px;
        bottom: 122px;
        transform: rotate(90deg);
        color: rgba(255,250,255,0.28);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        white-space: nowrap;
    }

    .brutal-hero__index {
        position: relative;
        z-index: 1;
        color: #B9DCF5;
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .brutal-hero__title {
        position: relative;
        z-index: 1;
        margin-top: 0.8rem;
        color: var(--paper);
        font-family: var(--font-display);
        font-size: clamp(2.45rem, 5vw, 5.2rem);
        line-height: 0.90;
        letter-spacing: -0.065em;
        text-transform: uppercase;
        max-width: 920px;
    }

    .brutal-hero__title span {
        color: var(--sky);
    }

    .brutal-hero__copy {
        position: relative;
        z-index: 1;
        max-width: 840px;
        margin-top: 1rem;
        color: rgba(255,250,255,0.82);
        font-family: var(--font-mono);
        font-size: 0.78rem;
        line-height: 1.65;
    }

    .brutal-hero__chips {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1rem;
    }

    .brutal-chip {
        border: 1px solid rgba(255,250,255,0.62);
        padding: 0.34rem 0.55rem;
        color: var(--paper);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: transparent;
    }

    .brutal-chip--hot {
        background: var(--hot);
        border-color: var(--carbon);
        color: var(--paper);
    }

    .brutal-hero__side {
        display: flex;
        flex-direction: column;
        background: var(--sky);
        border-left: 2px solid var(--carbon);
        padding: 1.25rem;
    }

    .hero-logo-frame {
        background: var(--paper);
        border: 2px solid var(--carbon);
        padding: 0.8rem;
        box-shadow: var(--small-shadow);
    }

    .hero-logo-frame img {
        width: 100%;
        height: 132px;
        object-fit: contain;
        display: block;
    }

    .hero-logo-fallback {
        height: 132px;
        display: grid;
        place-items: center;
        color: var(--blue);
        font-family: var(--font-display);
        font-size: 3rem;
    }

    .hero-system-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        margin-top: 1rem;
        border-left: 1.5px solid var(--carbon);
        border-top: 1.5px solid var(--carbon);
    }

    .hero-system-grid > div {
        min-height: 74px;
        padding: 0.7rem;
        border-right: 1.5px solid var(--carbon);
        border-bottom: 1.5px solid var(--carbon);
        background: rgba(255,250,255,0.86);
    }

    .hero-system-grid span {
        display: block;
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.58rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    .hero-system-grid strong {
        display: block;
        margin-top: 0.35rem;
        color: var(--carbon);
        font-family: var(--font-mono);
        font-size: 0.78rem;
        line-height: 1.3;
    }

    .status-square {
        display: inline-block;
        width: 0.55rem;
        height: 0.55rem;
        margin-right: 0.35rem;
        border: 1px solid var(--carbon);
        vertical-align: -0.03rem;
    }
    .status-square--online { background: var(--sky); }
    .status-square--offline { background: var(--hot); }

    /* ============================= SECTION HEAD / F-SHAPE HIERARCHY ============================= */
    .section-head {
        display: grid;
        grid-template-columns: 118px minmax(0, 1fr) auto;
        align-items: stretch;
        margin: 1.2rem 0 1rem 0;
        border: 2px solid var(--carbon);
        background: var(--paper);
        box-shadow: var(--small-shadow);
    }

    .section-head__index {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 86px;
        padding: 0.6rem;
        background: var(--hot);
        border-right: 2px solid var(--carbon);
        color: var(--paper);
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-align: center;
        text-transform: uppercase;
    }

    .section-head__body {
        padding: 0.9rem 1.05rem 0.85rem 1.05rem;
    }

    .section-head__title {
        color: var(--carbon);
        font-family: var(--font-display);
        font-size: 1.45rem;
        line-height: 1.0;
        letter-spacing: -0.035em;
        text-transform: uppercase;
    }

    .section-head__copy {
        margin-top: 0.5rem;
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.74rem;
        line-height: 1.5;
    }

    .section-head__stamp {
        display: flex;
        align-items: center;
        padding: 0 1rem;
        border-left: 1.5px solid var(--carbon);
        color: var(--blue);
        font-family: var(--font-mono);
        font-size: 0.60rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* ============================= STATUS STRIP / METRICS ============================= */
    div[data-testid="stMetric"] {
        background: var(--paper);
        border: 2px solid var(--carbon);
        border-radius: 0 !important;
        padding: 0.78rem 0.9rem;
        min-height: 102px;
        box-shadow: var(--small-shadow);
    }

    div[data-testid="stMetric"] label {
        color: var(--blue) !important;
        font-family: var(--font-mono);
        font-size: 0.62rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: var(--carbon);
        font-family: var(--font-mono);
        font-weight: 700;
        letter-spacing: -0.04em;
    }

    /* ============================= NOTES / PANELS ============================= */
    .scope-box, .note-box, .technical-note {
        background: var(--paper);
        color: var(--ink-500);
        border: 1.5px solid var(--carbon);
        border-left: 9px solid var(--sky);
        border-radius: 0;
        padding: 0.78rem 0.95rem;
        margin: 0.75rem 0 1rem 0;
        line-height: 1.6;
        font-family: var(--font-mono);
        font-size: 0.73rem;
    }

    .technical-note::before {
        content: "SYS_NOTE // ";
        color: var(--blue);
        font-weight: 700;
    }

    /* ============================= BUTTONS ============================= */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.85rem;
        border-radius: 0 !important;
        border: 2px solid var(--carbon) !important;
        background: var(--paper);
        color: var(--carbon);
        box-shadow: var(--small-shadow);
        font-family: var(--font-mono);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.045em;
        text-transform: uppercase;
        transition: transform 90ms ease, box-shadow 90ms ease, background 90ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--blue-100);
        color: var(--carbon);
        transform: translate(1px, 1px);
        box-shadow: 2px 2px 0 var(--carbon);
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translate(3px, 3px);
        box-shadow: none;
    }

    .stButton > button[kind="primary"] {
        background: var(--hot);
        border-color: var(--carbon) !important;
        color: var(--paper);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--blue);
        color: var(--paper);
    }

    /* ============================= TABS / TOP NAV ============================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--paper);
        border: 2px solid var(--carbon);
        box-shadow: var(--small-shadow);
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        min-height: 3.05rem;
        padding: 0 1rem;
        border-radius: 0 !important;
        border-right: 1.5px solid var(--carbon);
        background: var(--paper);
        color: var(--carbon);
        font-family: var(--font-mono);
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.035em;
        text-transform: uppercase;
    }

    .stTabs [aria-selected="true"] {
        background: var(--blue) !important;
        color: var(--paper) !important;
        box-shadow: inset 0 -6px 0 var(--hot) !important;
    }

    /* ============================= INPUTS / FORM CONTROLS ============================= */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    [data-baseweb="select"] > div {
        border-radius: 0 !important;
        border-color: var(--carbon) !important;
        box-shadow: none !important;
        font-family: var(--font-mono) !important;
    }

    div[data-testid="stNumberInput"] button {
        border-radius: 0 !important;
    }

    div[data-testid="stNumberInput"]:focus-within input,
    div[data-testid="stTextInput"]:focus-within input,
    div[data-testid="stTextArea"]:focus-within textarea {
        outline: 2px solid var(--sky);
        outline-offset: 1px;
    }

    [data-testid="stFileUploader"] {
        background: var(--paper);
        border: 2px dashed var(--carbon);
        border-radius: 0 !important;
        padding: 0.25rem;
    }

    [data-testid="stExpander"] {
        border-radius: 0 !important;
    }

    /* Highlight the two inverse target inputs as commanding controls. */
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sa (µm)"]),
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sz (µm)"]) {
        padding: 0.85rem;
        background: var(--blue-050);
        border: 2px solid var(--carbon);
        border-top: 8px solid var(--sky);
        border-radius: 0 !important;
        box-shadow: var(--small-shadow);
    }

    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sa (µm)"]) label,
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sz (µm)"]) label {
        color: var(--blue) !important;
        font-family: var(--font-mono);
        font-size: 0.70rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sa (µm)"]) input,
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sz (µm)"]) input {
        min-height: 3.35rem;
        color: var(--carbon) !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    /* ============================= DATA / TABLES ============================= */
    [data-testid="stDataFrame"] {
        background: var(--paper);
        border: 2px solid var(--carbon);
        border-radius: 0 !important;
        overflow: hidden;
        box-shadow: var(--small-shadow);
        font-family: var(--font-mono);
    }

    /* ============================= ALERTS ============================= */
    [data-testid="stAlert"] {
        border-radius: 0 !important;
        border: 1.5px solid var(--carbon) !important;
        box-shadow: 3px 3px 0 rgba(30,27,24,0.22);
        font-family: var(--font-mono);
    }

    /* ============================= PREDICTION OUTPUT / CARD-BLOCK LAYOUT ============================= */
    .result-display {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin: 0.7rem 0 1rem 0;
    }

    .result-card {
        position: relative;
        min-height: 188px;
        padding: 1rem 1.15rem 1.05rem 1.15rem;
        overflow: hidden;
        background: var(--paper);
        border: 2px solid var(--carbon);
        border-top: 10px solid var(--sky);
        border-radius: 0;
        box-shadow: var(--hard-shadow);
    }

    .result-card::before {
        content: "";
        position: absolute;
        right: -24px;
        bottom: -24px;
        width: 108px;
        height: 108px;
        background:
            linear-gradient(90deg, transparent 47%, rgba(10,36,99,0.10) 48% 52%, transparent 53%),
            linear-gradient(0deg, transparent 47%, rgba(10,36,99,0.10) 48% 52%, transparent 53%);
        transform: rotate(12deg);
    }

    .result-card--pass { border-top-color: var(--sky); }
    .result-card--fail { border-top-color: var(--hot); }
    .result-card--pending { border-top-color: var(--carbon); }

    .result-card__label {
        position: relative;
        z-index: 1;
        color: var(--blue);
        font-family: var(--font-mono);
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.105em;
        text-transform: uppercase;
    }

    .result-card__value {
        position: relative;
        z-index: 1;
        margin-top: 0.6rem;
        color: var(--carbon);
        font-family: var(--font-display);
        font-size: clamp(3rem, 6vw, 5.7rem);
        font-weight: 400;
        line-height: 0.86;
        letter-spacing: -0.075em;
    }

    .result-card__unit {
        margin-left: 0.25rem;
        color: var(--hot);
        font-family: var(--font-mono);
        font-size: 0.92rem;
        letter-spacing: 0;
    }

    .result-card__meta {
        position: relative;
        z-index: 1;
        margin-top: 0.9rem;
        padding-top: 0.55rem;
        border-top: 1px solid var(--line);
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.68rem;
    }

    .inverse-target-intro {
        margin: 0.2rem 0 0.9rem 0;
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.73rem;
    }

    .best-candidate {
        margin: 1rem 0 0.9rem 0;
        padding: 1rem 1.05rem;
        background: var(--blue);
        border: 2px solid var(--carbon);
        border-left: 12px solid var(--hot);
        border-radius: 0;
        box-shadow: var(--hard-shadow);
    }

    .best-candidate__label {
        color: #B9DCF5;
        font-family: var(--font-mono);
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .best-candidate__value {
        margin-top: 0.42rem;
        color:#0A2463 !important;
        -webkit-text-fill-color:#0A2463 !important;
        font-family: var(--font-mono);
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.75;
    }

    .best-candidate__value b {
        color:#0A2463 !important;
        -webkit-text-fill-color:#0A2463 !important;
        font-weight: 800;
    }

    .roughness-shell {
        border-radius: 16px;
        padding: 0.45rem 0.55rem 0.15rem 0.55rem;
        margin-bottom: 0.35rem;
        overflow: hidden;
    }

    .roughness-shell--pass {
        border: 2px solid #198754;
        background: rgba(25, 135, 84, 0.08);
    }

    .roughness-shell--review {
        border: 2px solid #D8315B;
        background: rgba(216, 49, 91, 0.08);
    }

    .roughness-shell [data-testid="stPlotlyChart"] {
        background: transparent !important;
    }



    /* ============================= RA SHOWCASE / INSPECTION TICKET ============================= */
    .ra-showcase-strip {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(260px, 0.55fr);
        gap: 0;
        margin: 0.85rem 0 1.05rem 0;
        border: 2px solid var(--carbon);
        background: var(--paper);
        box-shadow: var(--hard-shadow);
    }

    .ra-showcase-strip__copy {
        padding: 0.95rem 1rem;
        border-right: 2px solid var(--carbon);
        font-family: var(--font-mono);
        font-size: 0.73rem;
        line-height: 1.6;
        color: var(--ink-500);
    }

    .ra-showcase-strip__tag {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.85rem 1rem;
        background: var(--hot);
        color: var(--paper);
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        text-align: center;
    }

    .ra-ticket {
        display: grid;
        grid-template-columns: minmax(260px, 0.72fr) minmax(0, 1.28fr);
        border: 2px solid var(--carbon);
        box-shadow: var(--hard-shadow);
        margin: 1rem 0 1.1rem 0;
        background: var(--paper);
        overflow: hidden;
    }

    .ra-ticket__status {
        min-height: 238px;
        padding: 1.1rem;
        background: var(--carbon);
        color: var(--paper);
        border-right: 2px solid var(--carbon);
        border-left: 14px solid var(--sky);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .ra-ticket__status--review { border-left-color: var(--hot); }
    .ra-ticket__status--pending { border-left-color: var(--ink-300); }

    .ra-ticket__eyebrow {
        font-family: var(--font-mono);
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,250,255,0.66);
    }

    .ra-ticket__word {
        margin-top: 0.55rem;
        font-family: var(--font-display);
        font-size: clamp(3.2rem, 7vw, 6.4rem);
        line-height: 0.82;
        letter-spacing: -0.075em;
        text-transform: uppercase;
        color: var(--paper);
    }

    .ra-ticket__rule {
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255,250,255,0.28);
        font-family: var(--font-mono);
        font-size: 0.67rem;
        color: rgba(255,250,255,0.72);
        line-height: 1.55;
    }

    .ra-ticket__body {
        display: grid;
        grid-template-rows: auto 1fr auto;
        min-width: 0;
    }

    .ra-ticket__head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.9rem 1rem;
        background: var(--sky);
        border-bottom: 2px solid var(--carbon);
        color: var(--blue);
        font-family: var(--font-mono);
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .ra-ticket__metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .ra-ticket__metric {
        padding: 1.05rem 1rem 0.95rem 1rem;
        min-width: 0;
        border-right: 1.5px solid var(--carbon);
    }
    .ra-ticket__metric:last-child { border-right: 0; }

    .ra-ticket__metric span {
        display: block;
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.61rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .ra-ticket__metric strong {
        display: block;
        margin-top: 0.42rem;
        color: var(--carbon);
        font-family: var(--font-display);
        font-size: clamp(1.45rem, 3vw, 2.65rem);
        line-height: 0.95;
        letter-spacing: -0.055em;
        font-weight: 500;
        white-space: nowrap;
    }

    .ra-ticket__metric strong em {
        font-style: normal;
        color: var(--hot);
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 0;
        margin-left: 0.18rem;
    }

    .ra-ticket__meter-wrap {
        padding: 0.9rem 1rem 1rem 1rem;
        border-top: 1.5px solid var(--carbon);
    }

    .ra-ticket__meter-label {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.42rem;
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .ra-ticket__meter {
        position: relative;
        height: 20px;
        border: 2px solid var(--carbon);
        background:
            repeating-linear-gradient(90deg, var(--paper) 0 18px, rgba(30,27,24,0.13) 18px 19px);
        overflow: hidden;
    }

    .ra-ticket__meter-fill {
        height: 100%;
        background: var(--sky);
        border-right: 2px solid var(--carbon);
    }
    .ra-ticket__meter-fill--review { background: var(--hot); }

    .ra-ticket__foot {
        padding: 0.6rem 1rem;
        border-top: 1.5px solid var(--carbon);
        background: rgba(10,36,99,0.035);
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.63rem;
        line-height: 1.5;
    }


    /* ============================= FOOTER ============================= */
    .app-footer {
        display: grid;
        grid-template-columns: 122px minmax(0,1fr) auto;
        align-items: center;
        gap: 1rem;
        margin-top: 2.8rem;
        padding: 0.9rem;
        border: 2px solid var(--carbon);
        background: var(--paper);
        box-shadow: var(--small-shadow);
    }

    .app-footer__logo-frame {
        background: var(--paper);
        border: 1.5px solid var(--carbon);
        padding: 0.35rem;
    }

    .app-footer__logo {
        width: 100%;
        height: 64px;
        object-fit: contain;
        display: block;
    }

    .app-footer__text {
        color: var(--ink-500);
        font-family: var(--font-mono);
        font-size: 0.70rem;
        line-height: 1.55;
    }

    .app-footer__text strong {
        color: var(--blue);
    }

    .app-footer__stamp {
        color: var(--hot);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    hr {
        border: none;
        border-top: 1.5px solid var(--carbon);
    }

    code, pre {
        font-family: var(--font-mono) !important;
    }

    @media (max-width: 1080px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .brutal-hero {
            grid-template-columns: 1fr;
        }
        .brutal-hero__side {
            border-left: 0;
            border-top: 2px solid var(--carbon);
        }
        .hero-logo-frame img,
        .hero-logo-fallback { height: 100px; }
    }

    @media (max-width: 760px) {
        .result-display { grid-template-columns: 1fr; }
        .section-head { grid-template-columns: 92px minmax(0,1fr); }
        .section-head__stamp { display: none; }
        .app-footer { grid-template-columns: 92px 1fr; }
        .app-footer__stamp { grid-column: 1 / -1; }
        .brutal-hero__title { font-size: clamp(2.2rem, 14vw, 4rem); }
    }

    @media (max-width: 900px) {
        .ra-showcase-strip,
        .ra-ticket { grid-template-columns: 1fr; }
        .ra-showcase-strip__copy,
        .ra-ticket__status { border-right: 0; }
        .ra-showcase-strip__copy { border-bottom: 2px solid var(--carbon); }
        .ra-ticket__status { min-height: 190px; border-bottom: 2px solid var(--carbon); }
        .ra-ticket__metrics { grid-template-columns: 1fr; }
        .ra-ticket__metric { border-right: 0; border-bottom: 1.5px solid var(--carbon); }
        .ra-ticket__metric:last-child { border-bottom: 0; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# MillTwin-Lite UI v3 · editorial cyber-brutalism / minimal interaction layer
# Animation language adapted from the uploaded portfolio ZIP:
# fade-in reveal, hover lift, shimmer sweep, pulse, soft drifting background.
# =============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --blue: #0A2463;
        --blue-deep: #061A48;
        --sky: #3E92CC;
        --carbon: #1E1B18;
        --paper: #FFFAFF;
        --cream: #FFF4E8;
        --hot: #D8315B;
        --line-soft: rgba(62,146,204,.34);
        --line-dark: rgba(10,36,99,.18);
        --muted: #667080;
        --success: #0B8A61;
        --warning: #B36B12;
        --display: "Knockout Cruiserweight", "Bebas Neue", "Arial Narrow", sans-serif;
        --body: "Inter", "Segoe UI", Arial, sans-serif;
        --mono: "IBM Plex Mono", "Cascadia Code", Consolas, monospace;
        --ease: cubic-bezier(.22,.61,.36,1);
    }

    html, body, [class*="css"] { font-family: var(--body); }
    .stApp {
        background-color: var(--paper) !important;
        background-image:
          linear-gradient(rgba(10,36,99,.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(10,36,99,.035) 1px, transparent 1px) !important;
        background-size: 42px 42px !important;
        animation: gridDrift 24s linear infinite;
    }
    @keyframes gridDrift { to { background-position: 42px 42px; } }

    .block-container {
        max-width: 1500px !important;
        padding: 1rem 1.75rem 4rem !important;
        animation: pageReveal .55s var(--ease) both;
    }
    @keyframes pageReveal {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    h1,h2,h3,h4 { color: var(--carbon) !important; }
    h1,h2 { font-family: var(--display) !important; font-weight: 400 !important; }
    h3,h4 { font-family: var(--mono) !important; letter-spacing: .02em !important; }
    p,label,.stCaption { color: var(--muted); }

    /* ---------------- Sidebar ---------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071d4e 0%, var(--blue) 54%, #04183f 100%) !important;
        border-right: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: .65rem !important; }

    /* Deployment-safe sidebar behavior.
       Keep the console wide enough on desktop, but NEVER hide Streamlit's
       collapse/reopen control. Hiding that control can leave Cloud users
       stuck with a collapsed sidebar after a viewport/state change. */
    @media (min-width: 1000px) {
        section[data-testid="stSidebar"] {
            min-width: 350px !important;
            width: 350px !important;
            max-width: 350px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 350px !important;
            min-width: 350px !important;
            max-width: 350px !important;
        }
    }
    [data-testid="stSidebarCollapsedControl"] {
        z-index: 999999 !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        background: var(--blue) !important;
        color: var(--cream) !important;
        border: 1px solid rgba(255,250,255,.35) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(10,36,99,.18) !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        color: var(--cream) !important;
    }

    /* Streamlit Cloud 1.62+: keep sidebar controls usable and visible. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: .75rem !important;
        left: .75rem !important;
        right: auto !important;
        bottom: auto !important;
    }

    /* Paired numeric entry + slider. Users can type a precise value or drag. */
    .machine-input-label {
        margin: .15rem 0 .18rem;
        color: var(--cream);
        font: 600 .70rem/1.25 var(--mono);
        letter-spacing: .035em;
    }
    .machine-input-hint {
        margin: -.18rem 0 .38rem;
        color: rgba(255,244,232,.56);
        font: 500 .58rem/1.2 var(--mono);
        letter-spacing: .025em;
    }
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] {
        margin-bottom: -.15rem !important;
    }
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        font-family: var(--mono) !important;
        font-weight: 600 !important;
        color: var(--cream) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stSlider"] {
        margin-top: -.1rem !important;
        margin-bottom: .55rem !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] summary,
    [data-testid="stSidebar"] .stCaption { color: var(--cream) !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(255,250,255,.24) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,.018) !important;
        box-shadow: none !important;
        transition: border-color .25s ease, background .25s ease;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"]:hover {
        border-color: rgba(62,146,204,.8) !important;
        background: rgba(62,146,204,.05) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div { height: 3px !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border: 1px solid rgba(255,250,255,.22) !important;
        background: rgba(0,0,0,.10) !important;
        border-radius: 7px !important;
        box-shadow: none !important;
        color: var(--cream) !important;
    }
    .sidebar-brand {
        padding: .25rem .1rem 1rem !important;
        margin-bottom: .75rem !important;
        border-bottom: 1px solid rgba(255,250,255,.22) !important;
    }
    .sidebar-brand__logo-frame {
        padding: .7rem !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .sidebar-brand__logo { max-height: 145px !important; object-fit: contain; }
    .sidebar-brand__mark {
        margin-top: .75rem !important;
        color: var(--sky) !important;
        font: 600 .66rem/1.35 var(--mono) !important;
        letter-spacing: .10em !important;
    }
    .sidebar-brand__title { display:none !important; }
    .sidebar-brand__meta { display:none !important; }
    .machine-setup-header {
        display:flex; align-items:center; justify-content:space-between;
        margin:.9rem .05rem .55rem; padding:.2rem 0 .55rem;
        border-bottom:1px solid rgba(255,250,255,.12);
        color:var(--cream); font:700 .72rem/1.25 var(--mono);
        letter-spacing:.10em; text-transform:uppercase;
    }
    .machine-setup-sub {
        margin:0 .05rem .7rem; color:rgba(255,250,255,.62);
        font:500 .60rem/1.4 var(--mono); letter-spacing:.08em; text-transform:uppercase;
    }
    .side-status {
        display:flex; justify-content:space-between; align-items:center;
        margin:.55rem .05rem .9rem; padding:.72rem .75rem;
        border:1px solid rgba(255,250,255,.12); border-radius:9px;
        color:var(--cream); font:600 .68rem var(--mono); letter-spacing:.08em; text-transform:uppercase;
    }
    .side-status__value { display:flex; gap:.45rem; align-items:center; color:#B9F57D; }
    .status-dot-mini { width:8px; height:8px; border-radius:50%; background:#B9F57D; box-shadow:0 0 0 4px rgba(185,245,125,.08); animation:statusPulse 2.1s ease-in-out infinite; }
    .status-dot-mini--off { background:var(--hot); box-shadow:0 0 0 4px rgba(216,49,91,.10); }
    @keyframes statusPulse { 50% { transform:scale(1.25); opacity:.72; } }
    .nameplate { display:none !important; }

    /* ---------------- Hero ---------------- */
    .brutal-hero {
        position:relative !important;
        display:grid !important;
        grid-template-columns: 1fr !important;
        min-height: 202px !important;
        border:0 !important;
        border-radius:13px !important;
        overflow:hidden !important;
        box-shadow:none !important;
        background:
          radial-gradient(circle at 100% 0%, rgba(62,146,204,.22), transparent 35%),
          linear-gradient(100deg, #061A48 0%, var(--blue) 58%, #0A4C88 140%) !important;
        animation: heroEnter .7s var(--ease) both;
    }
    @keyframes heroEnter { from {opacity:0; transform:translateY(-10px)} to {opacity:1; transform:none} }
    .brutal-hero::before {
        content:""; position:absolute; inset:0; pointer-events:none;
        background-image:
          linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
        background-size:38px 38px; opacity:.65;
        animation: heroGrid 18s linear infinite;
    }
    @keyframes heroGrid { to { background-position:38px 0; } }
    .brutal-hero::after {
        content:""; position:absolute; width:390px; height:390px; right:-120px; top:-230px;
        border:1px solid rgba(255,255,255,.26); border-radius:50%;
        box-shadow:0 0 0 42px rgba(255,255,255,.035), 0 0 0 92px rgba(255,255,255,.025);
        animation: arcFloat 9s ease-in-out infinite alternate;
    }
    @keyframes arcFloat { to { transform:translate(-22px,18px) rotate(4deg); } }
    .brutal-hero__main {
        z-index:2 !important; padding:1.35rem 1.7rem !important; display:grid !important;
        grid-template-columns:150px minmax(0, 1fr) !important; column-gap:1.35rem !important; align-items:center !important;
        border:0 !important; background:transparent !important;
    }
    .hero-brand-logo { width:130px; height:130px; object-fit:contain; border-right:1px solid rgba(255,255,255,.30); padding-right:1.25rem; }
    .hero-copy-wrap { min-width:0; }
    .brutal-hero__index {
        color:#80C6F3 !important; font:600 .72rem/1.2 var(--mono) !important;
        letter-spacing:.16em !important; margin-bottom:.25rem !important; text-transform:uppercase;
    }
    .brutal-hero__title {
        color:var(--cream) !important; font-family:var(--display) !important; font-weight:400 !important;
        font-size:clamp(3.2rem,6vw,5.8rem) !important; line-height:.85 !important; letter-spacing:.01em !important;
        text-transform:none !important; margin:0 !important;
    }
    .brutal-hero__title span { color:var(--cream) !important; }
    .brutal-hero__copy { display:none !important; }
    .brutal-hero__chips { margin-top:.85rem !important; gap:.45rem !important; }
    .brutal-chip {
        background:transparent !important; color:var(--cream) !important;
        border:1px solid rgba(255,250,255,.34) !important; border-radius:999px !important;
        padding:.32rem .62rem !important; font:600 .65rem var(--mono) !important; letter-spacing:.05em !important;
        box-shadow:none !important; transition:transform .22s ease,border-color .22s ease,background .22s ease !important;
        position:relative; overflow:hidden;
    }
    .brutal-chip::after {
        content:""; position:absolute; inset:0; left:-130%; width:85%;
        background:linear-gradient(100deg,transparent,rgba(255,255,255,.20),transparent);
        transition:left .55s var(--ease);
    }
    .brutal-chip:hover { transform:translateY(-2px); border-color:var(--sky) !important; background:rgba(62,146,204,.10) !important; }
    .brutal-chip:hover::after { left:140%; }
    .brutal-chip--hot { color:var(--cream) !important; border-color:rgba(62,146,204,.72) !important; background:rgba(62,146,204,.12) !important; }
    .brutal-hero__side { display:none !important; }

    /* ---------------- Tabs ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap:.45rem !important; border-bottom:1px solid rgba(10,36,99,.25) !important;
        padding-top:.35rem !important; overflow:visible !important;
    }
    .stTabs [data-baseweb="tab"] {
        height:3.15rem !important; padding:0 1.25rem !important; border-radius:7px 7px 0 0 !important;
        background:transparent !important; color:#69707D !important;
        font:600 .78rem var(--mono) !important; letter-spacing:.04em !important; text-transform:uppercase !important;
        border:0 !important; transition:color .22s ease,background .22s ease,transform .22s ease !important;
        position:relative !important;
    }
    .stTabs [data-baseweb="tab"]::after {
        content:""; position:absolute; left:18%; right:82%; bottom:-1px; height:3px; background:var(--hot);
        transition:right .30s var(--ease),left .30s var(--ease);
    }
    .stTabs [data-baseweb="tab"]:hover { color:var(--blue) !important; transform:translateY(-1px); }
    .stTabs [data-baseweb="tab"]:hover::after { right:18%; }
    .stTabs [aria-selected="true"] {
        background:linear-gradient(180deg,#0B2C69,#071C4D) !important;
        color:var(--cream) !important; box-shadow:none !important;
        animation:activeTab .28s var(--ease) both;
    }
    .stTabs [aria-selected="true"] p { color:var(--cream) !important; }
    .stTabs [aria-selected="true"]::after { left:0; right:0; }
    @keyframes activeTab { from {transform:translateY(6px); opacity:.55} to {transform:none; opacity:1} }

    /* ---------------- Section title ---------------- */
    .section-head {
        display:grid !important; grid-template-columns:1fr auto !important; gap:1.5rem !important; align-items:end !important;
        border:0 !important; border-bottom:0 !important; padding:.95rem 0 .75rem !important; margin:.2rem 0 .7rem !important;
        background:transparent !important; box-shadow:none !important;
        animation:sectionReveal .55s var(--ease) both;
    }
    @keyframes sectionReveal { from {opacity:0; transform:translateY(18px)} to {opacity:1; transform:none} }
    .section-head__body { display:block !important; }
    .section-head__index {
        grid-column:1 / -1 !important; color:#1978B9 !important; background:transparent !important;
        padding:0 !important; border:0 !important; font:700 .7rem var(--mono) !important; letter-spacing:.14em !important;
        text-transform:uppercase !important; margin-bottom:-.25rem !important;
    }
    .section-head__title {
        font-family:var(--display) !important; font-weight:400 !important; color:#111 !important;
        font-size:clamp(4.2rem,8vw,7.8rem) !important; line-height:.78 !important; letter-spacing:.005em !important;
        text-transform:uppercase !important; max-width:1000px !important;
    }
    .section-head__copy {
        max-width:340px !important; color:#253149 !important; font-size:.86rem !important; line-height:1.45 !important;
        border-left:2px solid var(--hot) !important; padding-left:.8rem !important; margin:0 0 .45rem !important;
    }
    .section-head__stamp { display:none !important; }

    /* ---------------- Generic controls/cards ---------------- */
    div[data-testid="stMetric"],
    [data-testid="stDataFrame"],
    [data-testid="stFileUploader"],
    [data-testid="stAlert"],
    .result-card,
    .best-candidate,
    .ra-ticket,
    .ra-showcase-strip,
    .technical-note {
        border-width:1px !important; box-shadow:none !important;
    }
    div[data-testid="stMetric"] {
        background:rgba(255,255,255,.60) !important; border:1px solid var(--line-soft) !important;
        border-radius:8px !important; padding:.78rem .85rem !important; min-height:88px !important;
        transition:transform .22s ease,border-color .22s ease,background .22s ease;
    }
    div[data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:rgba(62,146,204,.75) !important; background:#fff !important; }
    div[data-testid="stMetric"] label { font:700 .65rem var(--mono) !important; color:#2776A9 !important; letter-spacing:.08em !important; text-transform:uppercase !important; }
    div[data-testid="stMetricValue"] { font-family:var(--mono) !important; color:var(--blue) !important; font-size:1.35rem !important; }

    .stButton > button,
    .stDownloadButton > button {
        border:1px solid rgba(10,36,99,.30) !important; border-radius:7px !important; box-shadow:none !important;
        min-height:2.85rem !important; font:700 .73rem var(--mono) !important; letter-spacing:.045em !important;
        transition:transform .22s ease,background .22s ease,border-color .22s ease,box-shadow .22s ease !important;
        overflow:hidden; position:relative;
    }
    .stButton > button::after,
    .stDownloadButton > button::after { content:"→"; display:inline-block; margin-left:.55rem; transition:transform .22s ease; }
    .stButton > button:hover::after,
    .stDownloadButton > button:hover::after { transform:translateX(5px); }
    .stButton > button:hover,
    .stDownloadButton > button:hover { transform:translateY(-2px); border-color:var(--sky) !important; box-shadow:0 7px 18px rgba(10,36,99,.08) !important; }
    .stButton > button[kind="primary"] {
        background:linear-gradient(90deg,#D8315B,#E52455) !important; border-color:#D8315B !important; color:var(--cream) !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] > div {
        border-color:rgba(10,36,99,.18) !important; border-width:1px !important; border-radius:6px !important;
        box-shadow:none !important; background:rgba(255,255,255,.62) !important;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label {
        font:600 .71rem var(--mono) !important; color:#39475D !important; letter-spacing:.02em !important;
    }
    [data-testid="stDataFrame"] { border:1px solid rgba(10,36,99,.17) !important; border-radius:8px !important; overflow:hidden; }
    [data-testid="stAlert"] { border-radius:7px !important; }
    [data-testid="stExpander"] { border:1px solid rgba(10,36,99,.17) !important; border-radius:8px !important; box-shadow:none !important; }
    hr { border-color:rgba(10,36,99,.14) !important; }

    .ui-card-label {
        color:#1978B9; font:700 .69rem var(--mono); letter-spacing:.11em; text-transform:uppercase; margin:.25rem 0 .7rem;
    }
    .mini-kicker {
        display:flex; align-items:center; gap:.5rem; font:700 .66rem var(--mono); color:#1978B9; text-transform:uppercase; letter-spacing:.10em;
        margin-bottom:.45rem;
    }
    .mini-kicker::before { content:""; width:18px; height:2px; background:var(--hot); }

    /* ---------------- Forward ---------------- */
    .result-display { grid-template-columns:repeat(2,minmax(0,1fr)) !important; gap:1rem !important; margin:0 !important; }
    .result-card {
        min-height:224px !important; padding:1.2rem 1.25rem !important;
        border:1px solid rgba(62,146,204,.55) !important; border-radius:9px !important;
        background:rgba(255,255,255,.64) !important; position:relative; overflow:hidden;
        transition:transform .28s var(--ease), border-color .28s ease, background .28s ease;
        animation:cardReveal .55s var(--ease) both;
    }
    .result-card:hover { transform:translateY(-4px); border-color:var(--sky) !important; background:#fff !important; }
    @keyframes cardReveal { from {opacity:0; transform:translateY(22px)} to {opacity:1; transform:none} }
    .result-card::after {
        width:128px !important; height:128px !important; right:-45px !important; bottom:-56px !important;
        border:13px solid rgba(62,146,204,.06) !important; box-shadow:none !important;
        animation:ringBreath 4s ease-in-out infinite alternate;
    }
    @keyframes ringBreath { to { transform:scale(1.09); opacity:.55; } }
    .result-card__label { color:#1978B9 !important; font:700 .69rem var(--mono) !important; letter-spacing:.10em !important; }
    .result-card__value {
        color:var(--blue) !important; font-family:var(--mono) !important; font-size:clamp(3rem,5.5vw,5rem) !important;
        letter-spacing:-.07em !important; margin-top:2.4rem !important;
    }
    .result-card__unit { color:#1978B9 !important; font-size:1rem !important; }
    .result-card__meta { margin-top:1.1rem !important; font-size:.72rem !important; }
    .result-card--pass { border-color:rgba(11,138,97,.45) !important; }
    .result-card--fail { border-color:rgba(216,49,91,.55) !important; }
    .forward-status {
        margin-top:1rem; padding:.95rem 1.05rem; border:1px solid rgba(62,146,204,.46); border-radius:8px;
        display:flex; align-items:center; justify-content:space-between; gap:1rem; background:rgba(255,255,255,.55);
        font:600 .78rem var(--mono); color:var(--blue); transition:border-color .22s ease,transform .22s ease;
    }
    .forward-status:hover { transform:translateY(-2px); border-color:var(--sky); }
    .forward-status small { color:var(--muted); font-family:var(--body); font-weight:500; }
    .forward-status__arrow { font-size:1.5rem; color:#1978B9; animation:arrowNudge 1.8s ease-in-out infinite; }
    @keyframes arrowNudge { 50% { transform:translateX(6px); } }

    /* ---------------- Inverse ---------------- */
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sa (µm)"]),
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sz (µm)"]) {
        padding:1rem 1.05rem !important; border:1px solid rgba(62,146,204,.55) !important; border-radius:9px !important;
        background:rgba(255,255,255,.62) !important; box-shadow:none !important;
        transition:transform .22s ease,border-color .22s ease;
    }
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sa (µm)"]):hover,
    div[data-testid="stNumberInput"]:has(input[aria-label="Maximum Sz (µm)"]):hover { transform:translateY(-2px); border-color:var(--sky) !important; }
    .best-candidate {
        background:rgba(238,246,252,.68) !important; border:1px solid rgba(62,146,204,.52) !important;
        border-left:3px solid var(--hot) !important; border-radius:8px !important; padding:1rem !important;
        animation:cardReveal .5s var(--ease) both;
    }
    .best-candidate__label { color:#1978B9 !important; }
    .best-candidate__value {
        color:#0A2463 !important;
        -webkit-text-fill-color:#0A2463 !important;
        font-size:.88rem !important;
        line-height:1.65 !important;
    }
    .best-candidate__value b {
        color:#0A2463 !important;
        -webkit-text-fill-color:#0A2463 !important;
        font-weight:800 !important;
    }
    .inverse-placeholder {
        min-height:330px; border:1px solid rgba(62,146,204,.38); border-radius:9px;
        display:grid; place-items:center; text-align:center; padding:2rem; color:var(--muted);
        background:linear-gradient(135deg,rgba(255,255,255,.55),rgba(238,246,252,.34));
    }
    .inverse-placeholder strong { display:block; font-family:var(--display); color:#111; font-size:3rem; font-weight:400; letter-spacing:.02em; }

    /* ---------------- Ra QC ---------------- */
    .ra-showcase-strip { display:none !important; }
    .ra-ticket { border:1px solid rgba(62,146,204,.48) !important; border-radius:9px !important; overflow:hidden !important; animation:cardReveal .5s var(--ease) both; }
    .ra-ticket__status { background:var(--blue) !important; color:var(--cream) !important; }
    .ra-ticket__word { font-family:var(--display) !important; font-weight:400 !important; letter-spacing:.03em !important; }
    .ra-ticket__body { border:0 !important; }
    .ra-ticket__metric { border-right:1px solid rgba(10,36,99,.16) !important; }
    .ra-ticket__meter-fill { transition:width .9s var(--ease) !important; }
    .technical-note { background:rgba(238,246,252,.5) !important; border:1px solid rgba(62,146,204,.24) !important; border-left:2px solid var(--sky) !important; border-radius:7px !important; }

    /* ---------------- Dossier cards ---------------- */
    .dossier-grid { display:grid; grid-template-columns:1.25fr .75fr; gap:1rem; margin-bottom:1rem; }
    .dossier-panel { border:1px solid rgba(62,146,204,.42); border-radius:9px; padding:1rem; background:rgba(255,255,255,.56); }
    .dossier-panel__title { color:#1978B9; font:700 .7rem var(--mono); letter-spacing:.10em; text-transform:uppercase; margin-bottom:.85rem; }
    .quick-steps { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.65rem; }
    .quick-step { min-height:146px; padding:.8rem; border:1px solid rgba(62,146,204,.34); border-radius:7px; transition:transform .22s ease,border-color .22s ease,background .22s ease; }
    .quick-step:hover { transform:translateY(-3px); border-color:var(--sky); background:#fff; }
    .quick-step__n { color:#1978B9; font:700 .72rem var(--mono); }
    .quick-step__name { color:var(--blue); font:700 .72rem var(--mono); text-transform:uppercase; margin-top:1.8rem; }
    .quick-step__copy { color:var(--muted); font-size:.72rem; line-height:1.45; margin-top:.45rem; }
    .motion-list { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; }
    .motion-item { padding:.75rem; border:1px solid rgba(10,36,99,.13); border-radius:7px; font-size:.72rem; color:var(--muted); }
    .motion-item strong { display:block; color:#1978B9; font:700 .68rem var(--mono); margin-bottom:.35rem; text-transform:uppercase; }
    .contact-stack { display:grid; gap:.65rem; }
    .contact-pill {
        display:flex; align-items:center; gap:.8rem; padding:.75rem 1rem; border-radius:999px; background:var(--blue); color:var(--cream) !important;
        text-decoration:none !important; font-size:1rem; transition:transform .22s ease,background .22s ease;
    }
    .contact-pill:hover { transform:translateX(5px); background:#0D3278; }
    .contact-pill__icon { color:var(--cream); font-size:1.1rem; }
    .roadmap-note { border-left:2px solid var(--hot); padding-left:.85rem; color:var(--muted); font-size:.8rem; line-height:1.5; }

    /* ---------------- Footer ---------------- */
    .app-footer { border-top:1px solid rgba(10,36,99,.16) !important; padding-top:1rem !important; margin-top:2rem !important; }
    .app-footer__logo { width:78px !important; }

    /* ---------------- Targeted contrast / logo fixes ---------------- */
    /* Fill the footer brand slot instead of leaving the logo floating inside a white frame. */
    .app-footer__logo-frame {
        padding:0 !important;
        overflow:hidden !important;
        background:var(--blue) !important;
        border:0 !important;
        min-height:84px !important;
        display:flex !important;
        align-items:stretch !important;
    }
    .app-footer__logo {
        width:100% !important;
        height:100% !important;
        min-height:84px !important;
        max-height:none !important;
        object-fit:cover !important;
        display:block !important;
    }

    /* Sidebar expander headings: cream on navy when closed; navy on cream when opened. */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background:transparent !important;
        color:var(--cream) !important;
        transition:background .2s ease,color .2s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        color:inherit !important;
        fill:currentColor !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"][open] > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
        background:var(--cream) !important;
        color:var(--blue) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"][open] > summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"][open] > summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary span {
        color:var(--blue) !important;
    }

    /* Primary CTA text must stay high-contrast over the magenta surface. */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] svg,
    .stDownloadButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] p,
    .stDownloadButton > button[kind="primary"] span,
    .stDownloadButton > button[kind="primary"] svg {
        color:var(--cream) !important;
        fill:currentColor !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover p,
    .stButton > button[kind="primary"]:hover span {
        color:#FFFFFF !important;
    }


    /* ---------------- Final contrast hardening ---------------- */
    /*
       Streamlit 1.62 applies text color to nested <p>/<span> nodes in several
       controls. These rules intentionally sit at the END of the theme so the
       deployed app keeps the same contrast as the design mockups.
    */

    /* 1) Brand artwork: the supplied logo.png contains generous navy margins.
       Crop/scale it visually so the mark actually fills its slot. */
    .sidebar-brand__logo-frame,
    .app-footer__logo-frame {
        overflow: hidden !important;
        background: var(--blue) !important;
    }
    .sidebar-brand__logo {
        width: 100% !important;
        height: 150px !important;
        max-height: none !important;
        object-fit: cover !important;
        object-position: center !important;
        transform: scale(2.15) !important;
        transform-origin: center !important;
        display: block !important;
    }
    .hero-brand-logo {
        width: 138px !important;
        height: 128px !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        object-fit: contain !important;
        object-position: center center !important;
        transform: none !important;
        transform-origin: center !important;
        padding-right: 1rem !important;
        margin: 0 !important;
        display: block !important;
    }
    .app-footer__logo {
        object-fit: cover !important;
        object-position: center !important;
        transform: scale(2.35) !important;
        transform-origin: center !important;
    }

    /* 2) Sidebar expander contrast.
       CLOSED = navy surface / cream type.
       OPEN   = cream surface / navy type. */
    [data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary * {
        background: transparent !important;
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary * {
        background: var(--cream) !important;
        color: var(--blue) !important;
        -webkit-text-fill-color: var(--blue) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary svg,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary svg * {
        color: var(--blue) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary svg,
    [data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary svg * {
        color: var(--cream) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }

    /* Fixed MACHINE SETUP heading is always on navy, so keep it cream. */
    .machine-setup-header,
    .machine-setup-header span,
    .machine-setup-sub {
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
    }

    /* Sidebar editable fields: dark field -> cream value; light +/- steppers -> navy. */
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] input::placeholder,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
        background: rgba(255,255,255,.055) !important;
        border-color: rgba(255,250,255,.16) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg,
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg * {
        color: var(--cream) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }

    /* 4) Cloud sidebar hardening. Streamlit 1.62 supports
       initial_sidebar_state="locked". On desktop this keeps the machining
       console permanently expanded. These selectors only reinforce the
       expanded state and do not fake a sidebar when Streamlit is in mobile
       mode. */
    @media (min-width: 1000px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            transform: none !important;
            margin-left: 0 !important;
            min-width: 350px !important;
            width: 350px !important;
            max-width: 350px !important;
            flex: 0 0 350px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div {
            visibility: visible !important;
            opacity: 1 !important;
        }
    }

    /* Keep the native collapsed control usable on mobile/narrow screens. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 100001 !important;
    }

    /* 3) Magenta primary CTA: NEVER inherit the generic muted paragraph color. */
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
    div[data-testid="stButton"] button[kind="primary"] *,
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] *,
    div[data-testid="stDownloadButton"] button[kind="primary"],
    div[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"],
    div[data-testid="stDownloadButton"] button[kind="primary"] *,
    div[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"] * {
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover,
    div[data-testid="stButton"] button[kind="primary"]:hover *,
    div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Selected navigation tab = navy block + cream type, including nested text. */
    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] * {
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
    }

    /* ---------------- Motion accessibility ---------------- */
    @media (prefers-reduced-motion: reduce) {
        *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
    }
    @media (max-width: 980px) {
        .brutal-hero__main { grid-template-columns:90px 1fr !important; padding:1rem !important; }
        .hero-brand-logo { width:82px; height:82px; padding-right:.8rem; }
        .section-head { grid-template-columns:1fr !important; }
        .section-head__copy { max-width:none !important; }
        .dossier-grid { grid-template-columns:1fr; }
        .quick-steps { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .motion-list { grid-template-columns:1fr; }
        .result-display { grid-template-columns:1fr !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# =============================================================================
# Deploy-stable application shell
# =============================================================================
st.markdown(
    """
    <style>
    /* Native Streamlit sidebar is intentionally not used in the deployed shell.
       A normal left column is used instead so Cloud cannot collapse/lose the
       machining controls because of browser state or Streamlit header changes. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    .block-container {
        max-width: 1840px !important;
        padding: .7rem 1rem 3.5rem !important;
    }

    /* Permanent machining rail: target only the column that contains our marker. */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) {
        background: linear-gradient(180deg, #071d4e 0%, #0A2463 55%, #04183f 100%) !important;
        border: 0 !important;
        border-radius: 14px !important;
        padding: .95rem .9rem 1.15rem !important;
        position: sticky !important;
        top: .55rem !important;
        align-self: flex-start !important;
        max-height: calc(100vh - 1.1rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin;
        scrollbar-color: rgba(255,244,232,.26) transparent;
        box-shadow: none !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker)::-webkit-scrollbar { width: 7px; }
    div[data-testid="stColumn"]:has(.millcore-rail-marker)::-webkit-scrollbar-thumb {
        background: rgba(255,244,232,.30); border-radius: 20px;
    }

    /* Marker itself occupies no space. */
    .millcore-rail-marker { display:none !important; width:0; height:0; }

    /* Rail typography / labels */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) h1,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) h2,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) h3,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) label,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) p,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) span,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) summary,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .stCaption {
        color: #FFF4E8 !important;
        -webkit-text-fill-color: #FFF4E8 !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) .machine-input-label,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .machine-setup-header,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .machine-setup-header *,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .side-status,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .side-status * {
        color: #FFF4E8 !important;
        -webkit-text-fill-color: #FFF4E8 !important;
        text-shadow: none !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) .machine-input-hint,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .machine-setup-sub,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .sidebar-brand__mark {
        color: #91D0FF !important;
        -webkit-text-fill-color: #91D0FF !important;
    }

    /* Inputs in the rail: readable on navy, with thin minimal borders. */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) div[data-testid="stNumberInput"] input,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-baseweb="select"] > div,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) textarea {
        background: rgba(255,244,232,.96) !important;
        color: #0A2463 !important;
        -webkit-text-fill-color: #0A2463 !important;
        border: 1px solid rgba(145,208,255,.75) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) div[data-testid="stNumberInput"] > div,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-baseweb="base-input"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) div[data-testid="stNumberInput"] button {
        background: #0B2C69 !important;
        color: #FFF4E8 !important;
        border: 1px solid rgba(145,208,255,.34) !important;
        box-shadow: none !important;
    }
    div[data-testid="stColumn"]:has(.millcore-rail-marker) div[data-testid="stNumberInput"] button:hover {
        background: #11377F !important;
        border-color: rgba(145,208,255,.68) !important;
        color: #FFF4E8 !important;
    }
    div[data-testid="stColumn"]:has(.millcore-rail-marker) div[data-testid="stNumberInput"] button svg,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) div[data-testid="stNumberInput"] button svg * {
        color: #FFF4E8 !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stSlider"] [role="slider"] {
        background: #D8315B !important;
        border: 2px solid #FFF4E8 !important;
        box-shadow: none !important;
    }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stSlider"] [data-testid="stTickBar"] span,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stSlider"] div {
        color: rgba(255,244,232,.82) !important;
        -webkit-text-fill-color: rgba(255,244,232,.82) !important;
    }

    /* Expander in the rail: closed = navy/cream, opened = cream/navy. */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stExpander"] {
        border: 1px solid rgba(255,244,232,.10) !important;
        background: transparent !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stExpander"] details:not([open]) > summary,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stExpander"] details:not([open]) > summary * {
        background: transparent !important;
        color: #FFF4E8 !important;
        -webkit-text-fill-color: #FFF4E8 !important;
    }
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stExpander"] details[open] > summary,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stExpander"] details[open] > summary * {
        background: #FFF4E8 !important;
        color: #0A2463 !important;
        -webkit-text-fill-color: #0A2463 !important;
    }

    /* Give the right workspace enough room and avoid the old centered-empty-space look. */
    div[data-testid="stColumn"]:has(.millcore-main-marker) {
        min-width: 0 !important;
        padding-left: .15rem !important;
    }
    .millcore-main-marker { display:none !important; width:0; height:0; }

    div[data-testid="stColumn"]:has(.millcore-rail-marker) img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Slightly compact the left logo so the six linked controls fit on common laptops. */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .sidebar-brand__logo-frame {
        margin-bottom: .6rem !important;
        padding: .2rem !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        overflow: hidden !important;
        min-height: auto !important;
        display: grid !important;
        place-items: center !important;
        border-radius: 0 !important;
    }
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .sidebar-brand__logo {
        display: block !important;
        width: 100% !important;
        max-width: min(100%, 260px) !important;
        height: auto !important;
        max-height: 130px !important;
        object-fit: contain !important;
        object-position: center center !important;
        transform: none !important;
        margin: 0 auto !important;
    }


    /* Make tiny helper text and current-value captions readable in the rail. */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stMarkdownContainer"],
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stMarkdownContainer"] *,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .st-emotion-cache-*,
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .stSlider [data-testid="stTickBarMin"],
    div[data-testid="stColumn"]:has(.millcore-rail-marker) .stSlider [data-testid="stTickBarMax"] {
        text-shadow: none !important;
    }

    /* Keep images from bleeding outside any card/frame in the rail. */
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stImage"] {
        overflow: hidden !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stColumn"]:has(.millcore-rail-marker) [data-testid="stImage"] > div {
        overflow: hidden !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    @media (max-width: 1050px) {
        /* Streamlit columns may stack on narrow screens; remove sticky behavior there. */
        div[data-testid="stColumn"]:has(.millcore-rail-marker) {
            position: relative !important;
            top: auto !important;
            max-height: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Loading utilities
# =============================================================================
@st.cache_data(show_spinner=False)
def get_base64_asset(filename: str) -> str:
    """Read a file from the assets folder and return it as a base64 string."""
    path = ASSETS_DIR / filename
    if not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_base64_file(path_value: str) -> str:
    """Read a local image by explicit path; used for same-folder logo.png."""
    path = Path(path_value)
    if not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def project_roots() -> list[Path]:
    """Return likely project/output folders, newest output folders first."""
    base_roots = [APP_DIR]

    cwd = Path.cwd().resolve()
    if cwd != APP_DIR:
        base_roots.append(cwd)

    output_dirs: list[Path] = []
    for base in base_roots:
        try:
            output_dirs.extend(path for path in base.glob("outputs*") if path.is_dir())
        except OSError:
            continue

    output_dirs = sorted(
        {path.resolve() for path in output_dirs},
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )

    roots: list[Path] = []
    seen: set[str] = set()
    for path in [*output_dirs, *base_roots]:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            roots.append(path.resolve())

    return roots


def find_latest_project_file(names: list[str]) -> Path | None:
    """Find the newest matching project file across app and outputs* folders."""
    candidates: list[Path] = []

    for root in project_roots():
        for name in names:
            path = root / name
            if path.is_file():
                candidates.append(path)

    if not candidates:
        return None

    # Prefer the explicit Sa/Sz model name, then newest modification time.
    name_priority = {
        "milltwin_pidl_sasz.onnx": 0,
        "model.onnx": 1,
        "pidl_fixed.onnx": 2,
        "pidl_dynamic.onnx": 3,
        "milltwin.onnx": 4,
        "milltwin_revised.onnx": 5,
    }

    return sorted(
        candidates,
        key=lambda path: (
            name_priority.get(path.name, 99),
            -path.stat().st_mtime,
        ),
    )[0]


@st.cache_data(show_spinner=False)
def load_info(path: str | None = None) -> dict:
    if path is None:
        return {
            "model_type": "unknown",
            "feature_columns": FEATURES,
            "model_feature_columns": MODEL_FEATURES,
            "target_columns": TARGETS,
            "input_ranges": DESIGN_RANGES,
            "limitations": ["info.json was not found; default design ranges are used."],
        }

    info_path = Path(path)
    if not info_path.exists():
        return {
            "model_type": "unknown",
            "feature_columns": FEATURES,
            "model_feature_columns": MODEL_FEATURES,
            "target_columns": TARGETS,
            "input_ranges": DESIGN_RANGES,
            "limitations": [f"info.json was not found at {info_path}."],
        }

    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "model_type": "unknown",
            "feature_columns": FEATURES,
            "target_columns": TARGETS,
            "input_ranges": DESIGN_RANGES,
            "limitations": [f"Could not read info.json: {error}"],
        }


def clean_range(name: str, raw_ranges: dict) -> tuple[float, float]:
    """
    Use clean design ranges for UI readability instead of LHS-generated decimal limits.
    """
    if name in DESIGN_RANGES:
        return float(DESIGN_RANGES[name]["min"]), float(DESIGN_RANGES[name]["max"])

    if name in raw_ranges:
        return float(raw_ranges[name]["min"]), float(raw_ranges[name]["max"])

    raise KeyError(f"Unknown range name: {name}")


def validate_onnx_schema(session) -> None:
    """Validate the D6 6-input, 2-output ONNX schema."""
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected exactly one ONNX input tensor, received {len(inputs)}.")

    input_shape = inputs[0].shape
    input_dim = input_shape[-1] if input_shape else None
    if isinstance(input_dim, int) and input_dim != len(MODEL_FEATURES):
        raise ValueError(
            f"Expected {len(MODEL_FEATURES)} D6 model inputs, received {input_dim}."
        )

    outputs = session.get_outputs()
    if len(outputs) < 1:
        raise ValueError("The ONNX model has no output tensor.")

    rough_shape = outputs[0].shape
    rough_dim = rough_shape[-1] if rough_shape else None
    if isinstance(rough_dim, int) and rough_dim != len(TARGETS):
        raise ValueError(
            f"Expected first output dimension {len(TARGETS)} for [Sa, Sz], "
            f"received {rough_dim}."
        )


def _create_onnx_session(model_source):
    return ort.InferenceSession(
        model_source,
        providers=["CPUExecutionProvider"],
    )


def load_onnx_model(uploaded_model):
    if not ONNX_AVAILABLE:
        return None, "ONNX Runtime is not installed. Run: pip install onnxruntime"

    if uploaded_model is not None:
        try:
            session = _create_onnx_session(uploaded_model.getvalue())
            validate_onnx_schema(session)
            output_name = session.get_outputs()[0].name
            return (
                session,
                f"Uploaded direct Sa/Sz PIDL ONNX loaded. Output: {output_name}.",
            )
        except Exception as error:
            return None, f"Uploaded ONNX failed validation: {error}"

    model_names = [
        "milltwin_pidl_sasz.onnx",
        "model.onnx",
        "pidl_fixed.onnx",
        "pidl_dynamic.onnx",
        "milltwin.onnx",
        "milltwin_revised.onnx",
    ]

    candidates: list[Path] = []
    for root in project_roots():
        for name in model_names:
            path = root / name
            if path.is_file():
                candidates.append(path)

    name_priority = {name: index for index, name in enumerate(model_names)}
    candidates = sorted(
        {path.resolve() for path in candidates},
        key=lambda path: (
            name_priority.get(path.name, 99),
            -path.stat().st_mtime,
        ),
    )

    validation_errors: list[str] = []

    for path in candidates:
        try:
            session = _create_onnx_session(str(path))
            validate_onnx_schema(session)
            output_name = session.get_outputs()[0].name
            return (
                session,
                f"Direct Sa/Sz PIDL loaded from {path}. Output: {output_name}.",
            )
        except Exception as error:
            validation_errors.append(f"{path.name}: {error}")

    searched = ", ".join(str(root) for root in project_roots())
    message = (
        "No compatible direct Sa/Sz ONNX model was found. "
        "Expected milltwin_pidl_sasz.onnx or model.onnx. "
        f"Searched: {searched}."
    )

    if validation_errors:
        message += " Validation details: " + " | ".join(validation_errors[:3])

    return None, message


def _run_onnx(session, values: np.ndarray) -> tuple[np.ndarray, float]:
    if session is None:
        raise RuntimeError(
            "No compatible direct Sa/Sz ONNX model is available. "
            "Train the PIDL model or upload milltwin_pidl_sasz.onnx/model.onnx."
        )

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(MODEL_FEATURES):
        raise ValueError(
            f"Expected encoded input shape [batch, {len(MODEL_FEATURES)}], received {values.shape}."
        )

    start = time.perf_counter()
    outputs = session.run(
        None,
        {session.get_inputs()[0].name: values},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if len(outputs) < 1:
        raise ValueError("The ONNX model returned no output tensor.")

    roughness = np.asarray(outputs[0], dtype=float)
    if roughness.ndim == 1 and roughness.size == len(TARGETS):
        roughness = roughness.reshape(1, -1)

    if roughness.ndim != 2 or roughness.shape[1] != len(TARGETS):
        raise ValueError(
            f"Unexpected roughness output shape {roughness.shape}; "
            f"expected [batch, {len(TARGETS)}]."
        )

    if not np.isfinite(roughness).all():
        raise ValueError("The ONNX model returned NaN or infinite roughness values.")

    return roughness, elapsed_ms


def encode_model_inputs(values: np.ndarray, modes="down") -> np.ndarray:
    """Return raw six-input D6 vector; milling mode is fixed to down."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise ValueError(f"Expected six continuous inputs, received {values.shape}.")
    if isinstance(modes, str):
        mode_list = [modes] * len(values)
    else:
        mode_list = [str(m) for m in modes]
    if len(mode_list) != len(values):
        raise ValueError("milling_mode count must match input row count.")
    if any(str(m).strip().lower() != "down" for m in mode_list):
        raise ValueError("This trained model supports down milling only.")
    return values


def validate_mode_ae(mode: str, ae_mm: float) -> None:
    mode = str(mode).strip().lower()
    ae_mm = float(ae_mm)
    if mode != "down":
        raise ValueError("This trained model supports down milling only.")
    if not (AE_PARTIAL_MIN_MM <= ae_mm <= AE_PARTIAL_MAX_MM):
        raise ValueError("Current D6 down-milling model requires 0.3 <= ae <= 3.0 mm.")


def predict(session, values: np.ndarray) -> tuple[np.ndarray, float]:
    return _run_onnx(session, values)


def append_prediction_history(
    params: dict,
    sa_pred_um: float,
    sz_pred_um: float,
    target_sa_um: float,
    target_sz_um: float,
    inference_ms: float,
) -> None:
    """Store one forward-prediction record in the current Streamlit session."""
    passed = sa_pred_um <= target_sa_um and sz_pred_um <= target_sz_um
    st.session_state.prediction_history.append(
        {
            "record_id": uuid.uuid4().hex[:10],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_rpm": float(params["n_rpm"]),
            "fz_mm_per_tooth": float(params["fz_mm_per_tooth"]),
            "ap_mm": float(params["ap_mm"]),
            "ae_mm": float(params["ae_mm"]),
            "eps_r_um": float(params["eps_r_um"]),
            "eps_a_um": float(params["eps_a_um"]),
            "milling_mode": str(params["milling_mode"]),
            "Sa_pred_um": float(sa_pred_um),
            "Sz_pred_um": float(sz_pred_um),
            "Sa_limit_um": float(target_sa_um),
            "Sz_limit_um": float(target_sz_um),
            "decision": "PASS" if passed else "REVIEW",
            "inference_ms": float(inference_ms),
            "note": "",
        }
    )


def reuse_history_record(record: dict) -> None:
    """Load a saved machining condition back into the Forward-model sidebar."""
    set_linked_machine_value("forward_n", record["n_rpm"], integer=True)
    set_linked_machine_value("forward_fz", record["fz_mm_per_tooth"])
    set_linked_machine_value("forward_ap", record["ap_mm"])
    mode = str(record.get("milling_mode", "down")).lower()
    st.session_state["forward_mode"] = mode
    if mode != "slot":
        set_linked_machine_value("forward_ae_partial", record["ae_mm"])
    set_linked_machine_value("forward_er", record["eps_r_um"])
    set_linked_machine_value("forward_ea", record["eps_a_um"])
    st.session_state["target_sa"] = float(record["Sa_limit_um"])
    st.session_state["target_sz"] = float(record["Sz_limit_um"])
    st.session_state.last_prediction = None


# =============================================================================
# Math and display utilities
# =============================================================================
def in_range_badges(params: dict) -> pd.DataFrame:
    rows = []

    for feature, value in params.items():
        if feature not in DESIGN_RANGES:
            continue

        low = float(DESIGN_RANGES[feature]["min"])
        high = float(DESIGN_RANGES[feature]["max"])

        rows.append(
            {
                "Feature": feature,
                "Value": value,
                "Design min": low,
                "Design max": high,
                "Inside design range": low <= float(value) <= high,
            }
        )

    return pd.DataFrame(rows)


def section_header(index: str, title: str, description: str = "") -> None:
    copy_html = f'<div class="section-head__copy">{html.escape(description)}</div>' if description else ""
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-head__index">{html.escape(index)}</div>
            <div class="section-head__body">
                <div class="section-head__title">{html.escape(title)}</div>
            </div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def roughness_chart(title: str, value: float, limit: float):
    """Compact engineering chart showing prediction against the acceptance limit."""
    upper = max(limit * 1.38, value * 1.18, 1e-6)
    passed = value <= limit
    state_color = "#198754" if passed else "#D8315B"
    state_label = "PASS" if passed else "REVIEW"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[value],
            y=[""],
            orientation="h",
            marker={
                "color": state_color,
                "line": {"color": "#FFFFFF", "width": 2.2},
            },
            width=0.42,
            hovertemplate=f"{title}: %{{x:.3f}} µm · {state_label}<extra></extra>",
        )
    )
    fig.add_vline(
        x=limit,
        line_width=2.5,
        line_dash="dash",
        line_color="#1E1B18",
    )
    fig.add_annotation(
        x=limit,
        y=1.16,
        xref="x",
        yref="paper",
        text=f"Target {limit:.3f} µm",
        showarrow=False,
        yanchor="bottom",
        font={
            "size": 12,
            "color": "#5C6472",
            "family": "JetBrains Mono, Cascadia Code, monospace",
        },
    )
    fig.add_annotation(
        x=0.985,
        y=0.97,
        xref="paper",
        yref="paper",
        text=f"<b>{state_label}</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={
            "size": 11,
            "color": state_color,
            "family": "JetBrains Mono, Cascadia Code, monospace",
        },
    )

    inside_threshold = 0.22 * upper
    if value >= inside_threshold:
        fig.add_annotation(
            x=max(value - 0.02 * upper, 0.02 * upper),
            y="",
            xref="x",
            yref="y",
            text=f"<b>{value:.3f} µm</b>",
            showarrow=False,
            xanchor="right",
            font={
                "size": 12,
                "color": "#FFFFFF",
                "family": "JetBrains Mono, Cascadia Code, monospace",
            },
        )
    else:
        fig.add_annotation(
            x=value + 0.015 * upper,
            y="",
            xref="x",
            yref="y",
            text=f"<b>{value:.3f} µm</b>",
            showarrow=False,
            xanchor="left",
            font={
                "size": 12,
                "color": state_color,
                "family": "JetBrains Mono, Cascadia Code, monospace",
            },
        )

    fig.add_annotation(
        x=0.5,
        y=-0.18,
        xref="paper",
        yref="paper",
        text=f"<b>{title}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font={
            "size": 18,
            "color": "#1E1B18",
            "family": "Inter, Arial, sans-serif",
        },
    )

    fig.update_layout(
        height=215,
        margin=dict(l=14, r=14, t=56, b=54),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.55,
        xaxis={
            "range": [0, upper],
            "title": "Surface roughness (µm)",
            "gridcolor": "#D9D6D9",
            "zeroline": False,
            "showline": False,
            "tickfont": {"color": "#5C6472", "size": 11},
            "title_font": {"color": "#5C6472", "size": 11},
            "title_standoff": 18,
        },
        yaxis={
            "showgrid": False,
            "showline": False,
            "showticklabels": False,
        },
        font={
            "family": "JetBrains Mono, Cascadia Code, monospace",
            "color": "#5C6472",
        },
    )
    return fig


def normalize_fem_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Sa_FEM_um": "Sa_um",
        "Sz_FEM_um": "Sz_um",
        "Sa_fem_um": "Sa_um",
        "Sz_fem_um": "Sz_um",
        "Sa": "Sa_um",
        "Sz": "Sz_um",
        "mode": "milling_mode",
    }

    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in frame.columns and new not in frame.columns
    }

    out = frame.rename(columns=rename_map)
    if MILLING_MODE in out.columns:
        out[MILLING_MODE] = out[MILLING_MODE].astype(str).str.strip().str.lower()
    return out


def fem_metrics(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []

    for idx, target in enumerate(TARGETS):
        truth = frame[target].to_numpy(float)
        p = pred[:, idx]

        denom = np.maximum(np.abs(truth), 1e-8)
        ss_res = np.sum((truth - p) ** 2)
        ss_tot = np.sum((truth - np.mean(truth)) ** 2)

        rows.append(
            {
                "target": target,
                "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                "MAE_um": np.mean(np.abs(truth - p)),
                "RMSE_um": np.sqrt(np.mean((truth - p) ** 2)),
                "MAPE_percent": np.mean(np.abs((truth - p) / denom)) * 100.0,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Linked machine-input controls
# =============================================================================
def _coerce_linked_value(value, integer: bool):
    return int(round(float(value))) if integer else float(value)


def _sync_machine_input(source_key: str, target_key: str, canonical_key: str, integer: bool) -> None:
    """Synchronize a number input and slider through one canonical session value."""
    value = _coerce_linked_value(st.session_state[source_key], integer)
    st.session_state[target_key] = value
    st.session_state[canonical_key] = value


def set_linked_machine_value(canonical_key: str, value, integer: bool = False) -> None:
    """Set canonical, typed and slider states together (used by History -> Reuse)."""
    value = _coerce_linked_value(value, integer)
    st.session_state[canonical_key] = value
    st.session_state[f"{canonical_key}__number"] = value
    st.session_state[f"{canonical_key}__slider"] = value


def linked_number_slider(
    label: str,
    min_value,
    max_value,
    default,
    step,
    *,
    key: str,
    fmt: str | None = None,
    unit_hint: str = "TYPE OR DRAG",
    integer: bool = False,
):
    """Render a bidirectionally-linked number input and slider in the sidebar."""
    number_key = f"{key}__number"
    slider_key = f"{key}__slider"

    default_value = _coerce_linked_value(default, integer)
    min_v = _coerce_linked_value(min_value, integer)
    max_v = _coerce_linked_value(max_value, integer)
    step_v = _coerce_linked_value(step, integer)

    # Respect a canonical value already loaded by History/reuse; otherwise initialize it.
    if key not in st.session_state:
        st.session_state[key] = default_value
    canonical = _coerce_linked_value(st.session_state[key], integer)
    canonical = max(min_v, min(max_v, canonical))
    st.session_state[key] = canonical

    if number_key not in st.session_state:
        st.session_state[number_key] = canonical
    if slider_key not in st.session_state:
        st.session_state[slider_key] = canonical

    # If an external action updated only the canonical value, bring both controls with it.
    if st.session_state[number_key] != canonical and st.session_state[slider_key] != canonical:
        st.session_state[number_key] = canonical
        st.session_state[slider_key] = canonical

    # IMPORTANT: this helper can be called from the main script, so every widget
    # must be explicitly mounted in CONTROL_PANEL.  Using plain st.number_input /
    # st.slider here renders them in the main page on Streamlit Cloud.
    CONTROL_PANEL.markdown(f'<div class="machine-input-label">{label}</div>', unsafe_allow_html=True)
    CONTROL_PANEL.number_input(
        label,
        min_value=min_v,
        max_value=max_v,
        step=step_v,
        format=fmt,
        key=number_key,
        label_visibility="collapsed",
        on_change=_sync_machine_input,
        args=(number_key, slider_key, key, integer),
    )
    CONTROL_PANEL.slider(
        f"{label} slider",
        min_value=min_v,
        max_value=max_v,
        step=step_v,
        format=fmt,
        key=slider_key,
        label_visibility="collapsed",
        on_change=_sync_machine_input,
        args=(slider_key, number_key, key, integer),
    )
    CONTROL_PANEL.markdown(f'<div class="machine-input-hint">{unit_hint}</div>', unsafe_allow_html=True)

    return _coerce_linked_value(st.session_state[key], integer)


# =============================================================================
# Load info/model and application shell
# =============================================================================
brand_logo_b64 = get_base64_file(str(LOGO_PATH))

info_path = find_latest_project_file(["info.json"])
info = load_info(str(info_path) if info_path is not None else None)
raw_ranges = info.get("input_ranges", DESIGN_RANGES)

# Deploy-stable shell: use a normal left column instead of Streamlit's native sidebar.
# This avoids Community Cloud restoring a hidden/collapsed sidebar state.
CONTROL_PANEL, APP_MAIN = st.columns([0.235, 0.765], gap="medium")
CONTROL_PANEL.markdown('<div class="millcore-rail-marker"></div>', unsafe_allow_html=True)
APP_MAIN.markdown('<div class="millcore-main-marker"></div>', unsafe_allow_html=True)

sidebar_logo_html = (
    f'<img class="sidebar-brand__logo" src="data:image/png;base64,{brand_logo_b64}" alt="Millcore logo">'
    if brand_logo_b64
    else '<div style="color:#FFF4E8;font-family:Impact,sans-serif;font-size:2rem;text-align:center;">MILLCORE</div>'
)

CONTROL_PANEL.markdown(
    f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand__logo-frame">{sidebar_logo_html}</div>
        <div class="sidebar-brand__mark">MILLCORE SYSTEMS / CNC SURFACE INTELLIGENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with CONTROL_PANEL.expander("01 / MODEL INTERFACE", expanded=False):
    uploaded_model = st.file_uploader(
        "Forward PIDL model",
        type=["onnx"],
        key="uploaded_onnx_model",
        help="Expected schema: six process inputs and direct [Sa, Sz] outputs.",
    )

model_session, model_status = load_onnx_model(uploaded_model)
model_ready = model_session is not None
status_label = "ONLINE" if model_ready else "OFFLINE"
status_dot = "" if model_ready else " status-dot-mini--off"

CONTROL_PANEL.markdown(
    f"""
    <div class="side-status">
        <span>MODEL STATUS</span>
        <span class="side-status__value"><span class="status-dot-mini{status_dot}"></span>{status_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep the six model inputs permanently visible inside the sidebar.
# This intentionally avoids placing MACHINE SETUP inside an expander because
# Streamlit Cloud can restore a collapsed expander/sidebar state per browser.
CONTROL_PANEL.markdown(
    '<div class="machine-setup-header"><span>02 / MACHINE SETUP</span><span>6 INPUTS</span></div>'
    '<div class="machine-setup-sub">Type an exact value or drag the linked slider.</div>',
    unsafe_allow_html=True,
)

n_min_ui, n_max_ui = clean_range("n_rpm", raw_ranges)
fz_min_ui, fz_max_ui = clean_range("fz_mm_per_tooth", raw_ranges)
ap_min_ui, ap_max_ui = clean_range("ap_mm", raw_ranges)
er_min_ui, er_max_ui = clean_range("eps_r_um", raw_ranges)
ea_min_ui, ea_max_ui = clean_range("eps_a_um", raw_ranges)

n = linked_number_slider(
    "Spindle speed n [rpm]", int(n_min_ui), int(n_max_ui), 4000, 100,
    key="forward_n", fmt="%d", unit_hint="RPM · TYPE OR DRAG", integer=True,
)
fz = linked_number_slider(
    "Feed per tooth fz [mm/tooth]", float(fz_min_ui), float(fz_max_ui), 0.07, 0.005,
    key="forward_fz", fmt="%.3f", unit_hint="MM/TOOTH · TYPE OR DRAG",
)
ap = linked_number_slider(
    "Axial depth ap [mm]", float(ap_min_ui), float(ap_max_ui), 0.90, 0.05,
    key="forward_ap", fmt="%.2f", unit_hint="MM · TYPE OR DRAG",
)

milling_mode = CONTROL_PANEL.selectbox(
    "Milling mode", MILLING_MODES,
    index=MILLING_MODES.index(st.session_state.get("forward_mode", "down"))
    if st.session_state.get("forward_mode", "down") in MILLING_MODES else 0,
    key="forward_mode",
)

ae = linked_number_slider(
    "Radial engagement ae [mm]", AE_PARTIAL_MIN_MM, AE_PARTIAL_MAX_MM, 1.5, 0.1,
    key="forward_ae_partial", fmt="%.1f", unit_hint="MM · TYPE OR DRAG",
)
eps_r = linked_number_slider(
    "Radial runout εr [µm]", float(er_min_ui), float(er_max_ui), 0.0, 0.1,
    key="forward_er", fmt="%.1f", unit_hint="µM · TYPE OR DRAG",
)
eps_a = linked_number_slider(
    "Axial runout εa [µm]", float(ea_min_ui), float(ea_max_ui), 2.5, 0.1,
    key="forward_ea", fmt="%.1f", unit_hint="µM · TYPE OR DRAG",
)

with APP_MAIN:
    hero_logo = (
        f'<img class="hero-brand-logo" src="data:image/png;base64,{brand_logo_b64}" alt="Millcore logo">'
        if brand_logo_b64
        else '<div class="hero-brand-logo" style="display:grid;place-items:center;color:#FFF4E8;font:5rem Impact,sans-serif;">M</div>'
    )

    st.markdown(
        f"""
        <div class="brutal-hero">
            <div class="brutal-hero__main">
                {hero_logo}
                <div class="hero-copy-wrap">
                    <div class="brutal-hero__index">A MILLCORE TEAM ENGINEERING PLATFORM</div>
                    <div class="brutal-hero__title">MillTwin-Lite</div>
                    <div class="brutal-hero__chips">
                        <div class="brutal-chip brutal-chip--hot">● PIDL ENGINE</div>
                        <div class="brutal-chip">ONNX INFERENCE</div>
                        <div class="brutal-chip">CNC MILLING</div>
                        <div class="brutal-chip">Sa / Sz</div>
                        <div class="brutal-chip">Ra QC</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # =============================================================================
    # Session state
    # =============================================================================
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "inverse_results" not in st.session_state:
        st.session_state.inverse_results = None

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if "ra_check_result" not in st.session_state:
        st.session_state.ra_check_result = None


    # =============================================================================
    # Tabs
    # =============================================================================
    forward_tab, inverse_tab, history_tab, ra_tab, fem_tab, info_tab = st.tabs(
        [
            "Forward",
            "Inverse",
            "History",
            "Ra QC",
            "Validation",
            "Dossier",
        ]
    )


    # =============================================================================
    # Forward prediction tab
    # =============================================================================
    with forward_tab:
        section_header("01 / PREDICTIVE MACHINING", "Forward Prediction", "Predict surface results from one machining condition.")

        control_col, result_col = st.columns([0.78, 2.22], gap="large")

        with control_col:
            st.markdown('<div class="ui-card-label">QUALITY CEILINGS</div>', unsafe_allow_html=True)
            target_sa = st.number_input(
                "Target Sa [µm]", min_value=0.001, max_value=20.0, value=1.0,
                step=0.05, format="%.3f", key="target_sa",
            )
            target_sz = st.number_input(
                "Target Sz [µm]", min_value=0.001, max_value=100.0, value=5.0,
                step=0.1, format="%.3f", key="target_sz",
            )
            run_clicked = st.button(
                "RUN PIDL INFERENCE", type="primary", width="stretch",
                disabled=model_session is None,
            )

            if model_session is None:
                st.caption("ONNX model required.")
            else:
                st.caption("Six process inputs are set in Machine Setup.")

        if run_clicked:
            raw_values = np.array([[n, fz, ap, ae, eps_r, eps_a]], dtype=np.float32)
            try:
                validate_mode_ae(milling_mode, ae)
                values = encode_model_inputs(raw_values, milling_mode)
                prediction, elapsed = predict(model_session, values)
                prediction_params = dict(zip(FEATURES, raw_values[0]))
                prediction_params["milling_mode"] = milling_mode
                sa_pred = float(prediction[0, 0])
                sz_pred = float(prediction[0, 1])
                st.session_state.last_prediction = {"Sa": sa_pred, "Sz": sz_pred, "time": elapsed, "params": prediction_params}
                append_prediction_history(
                    params=prediction_params,
                    sa_pred_um=sa_pred,
                    sz_pred_um=sz_pred,
                    target_sa_um=float(target_sa),
                    target_sz_um=float(target_sz),
                    inference_ms=elapsed,
                )
            except Exception as error:
                st.error(str(error))

        result = st.session_state.last_prediction
        target_met = bool(result and result["Sa"] <= float(target_sa) and result["Sz"] <= float(target_sz))
        sa_value = f"{result['Sa']:.3f}" if result else "—"
        sz_value = f"{result['Sz']:.3f}" if result else "—"
        sa_card_state = "result-card--pass" if result and result["Sa"] <= float(target_sa) else "result-card--fail" if result else "result-card--pending"
        sz_card_state = "result-card--pass" if result and result["Sz"] <= float(target_sz) else "result-card--fail" if result else "result-card--pending"

        with result_col:
            st.markdown(
                f"""
                <div class="result-display">
                    <div class="result-card {sa_card_state}">
                        <div class="result-card__label">PREDICTED AREAL MEAN ROUGHNESS · SA</div>
                        <div class="result-card__value">{sa_value}<span class="result-card__unit">µm</span></div>
                        <div class="result-card__meta">Sa ceiling · {float(target_sa):.3f} µm</div>
                    </div>
                    <div class="result-card {sz_card_state}">
                        <div class="result-card__label">PREDICTED MAXIMUM SURFACE HEIGHT · SZ</div>
                        <div class="result-card__value">{sz_value}<span class="result-card__unit">µm</span></div>
                        <div class="result-card__meta">Sz ceiling · {float(target_sz):.3f} µm</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            state = "PASS" if target_met else ("REVIEW" if result else "READY FOR INFERENCE")
            detail = f"Latency {result['time']:.3f} ms" if result else "Configure machining parameters and run the PIDL model."
            st.markdown(
                f'<div class="forward-status"><div><strong>{state}</strong><br><small>{detail}</small></div><div class="forward-status__arrow">→</div></div>',
                unsafe_allow_html=True,
            )

        params = {
            "n_rpm": n, "fz_mm_per_tooth": fz, "ap_mm": ap, "ae_mm": ae,
            "eps_r_um": eps_r, "eps_a_um": eps_a, "milling_mode": milling_mode,
        }
        range_df = in_range_badges(params).rename(columns={
            "Feature": "Process variable", "Value": "Current value", "Design min": "Domain minimum",
            "Design max": "Domain maximum", "Inside design range": "In domain",
        })
        in_domain = bool(range_df.empty or range_df["In domain"].all())

        with st.expander("PROCESS DOMAIN / DIAGNOSTICS", expanded=False):
            st.dataframe(range_df, width="stretch", hide_index=True)
            if result:
                d1, d2 = st.columns(2)
                with d1:
                    sa_pass = result["Sa"] <= float(target_sa)
                    sa_shell_class = "roughness-shell roughness-shell--pass" if sa_pass else "roughness-shell roughness-shell--review"
                    st.markdown(f'<div class="{sa_shell_class}">', unsafe_allow_html=True)
                    st.plotly_chart(
                        roughness_chart("Sa", result["Sa"], target_sa),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                with d2:
                    sz_pass = result["Sz"] <= float(target_sz)
                    sz_shell_class = "roughness-shell roughness-shell--pass" if sz_pass else "roughness-shell roughness-shell--review"
                    st.markdown(f'<div class="{sz_shell_class}">', unsafe_allow_html=True)
                    st.plotly_chart(
                        roughness_chart("Sz", result["Sz"], target_sz),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                physics_ok = result["Sa"] > 0 and result["Sz"] >= 2.0 * result["Sa"]
                st.caption(f"Geometry consistency: {'PASS' if physics_ok else 'REVIEW'} · Model domain: {'IN' if in_domain else 'OUT'}")


    # =============================================================================
    # Inverse search tab
    # =============================================================================
    with inverse_tab:
        section_header("02 / PARAMETER SYNTHESIS", "Inverse Search", "Sa and Sz are the commanding inputs.")

        target_sa_col, target_sz_col = st.columns(2, gap="medium")
        with target_sa_col:
            i_sa = st.number_input("Maximum Sa (µm)", min_value=0.001, value=1.0, step=0.05, format="%.3f", key="i_sa")
        with target_sz_col:
            i_sz = st.number_input("Maximum Sz (µm)", min_value=0.001, value=5.0, step=0.1, format="%.3f", key="i_sz")

        search_col, ranked_col = st.columns([0.86, 1.64], gap="large")

        with search_col:
            st.markdown('<div class="ui-card-label">SEARCH ENVELOPE</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                n_min = st.number_input("n min [rpm]", 1000, 8000, 1000, 100, key="inverse_n_min")
                fz_min = st.number_input("fz min [mm/tooth]", 0.02, 0.12, 0.02, 0.005, format="%.3f", key="inverse_fz_min")
            with d2:
                n_max = st.number_input("n max [rpm]", 1000, 8000, 8000, 100, key="inverse_n_max")
                fz_max = st.number_input("fz max [mm/tooth]", 0.02, 0.12, 0.12, 0.005, format="%.3f", key="inverse_fz_max")

            st.markdown('<div class="ui-card-label">FIXED PROCESS STATE</div>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                i_ap = st.number_input("ap [mm]", 0.3, 1.5, float(ap), 0.10, format="%.2f", key="inverse_ap")
                i_er = st.number_input("εr [µm]", -10.0, 10.0, float(eps_r), 0.1, format="%.1f", key="inverse_er")
            with f2:
                i_ae = st.number_input("ae [mm]", 0.3, 3.0, min(max(float(ae), 0.3), 3.0), 0.1, format="%.1f", key="inverse_ae_partial")
                i_ea = st.number_input("εa [µm]", 0.0, 5.0, float(eps_a), 0.1, format="%.1f", key="inverse_ea")
            i_mode = "down"
            teeth = NUM_FLUTES
            top_k = st.number_input("Ranked results", 1, 100, 10, 1, key="inverse_top_k")

            with st.expander("GRID / PRODUCTIVITY LIMITS", expanded=False):
                n_points = st.number_input("n grid points", 2, 500, 120, 1, key="inverse_n_points")
                fz_points = st.number_input("fz grid points", 2, 500, 120, 1, key="inverse_fz_points")
                feed_max = st.number_input("Max feed F [mm/min] · 0 = off", min_value=0.0, value=0.0, step=100.0, key="inverse_feed_max")
                mrr_max = st.number_input("Max MRR [mm³/min] · 0 = off", min_value=0.0, value=0.0, step=1000.0, key="inverse_mrr_max")

            run_inverse = st.button("RUN INVERSE SEARCH", type="primary", width="stretch")

        if run_inverse:
            if model_session is None:
                st.error("No ONNX model is available. Train or upload a model first.")
            elif n_min >= n_max:
                st.error("n minimum must be smaller than n maximum.")
            elif fz_min >= fz_max:
                st.error("fz minimum must be smaller than fz maximum.")
            else:
                n_grid = np.linspace(float(n_min), float(n_max), int(n_points))
                fz_grid = np.linspace(float(fz_min), float(fz_max), int(fz_points))
                n_mesh, fz_mesh = np.meshgrid(n_grid, fz_grid, indexing="ij")
                count = n_mesh.size
                raw_X = np.column_stack([
                    n_mesh.ravel(), fz_mesh.ravel(), np.full(count, i_ap), np.full(count, i_ae),
                    np.full(count, i_er), np.full(count, i_ea),
                ]).astype(np.float32)
                try:
                    validate_mode_ae(i_mode, i_ae)
                    X = encode_model_inputs(raw_X, i_mode)
                    prediction, elapsed = predict(model_session, X)
                    result_df = pd.DataFrame(raw_X, columns=FEATURES)
                    result_df["milling_mode"] = i_mode
                    result_df["Sa_pred_um"] = prediction[:, 0]
                    result_df["Sz_pred_um"] = prediction[:, 1]
                    result_df["F_mm_min"] = result_df["n_rpm"] * result_df["fz_mm_per_tooth"] * int(teeth)
                    result_df["MRR_mm3_min"] = result_df["ap_mm"] * result_df["ae_mm"] * result_df["F_mm_min"]
                    result_df["roughness_pass"] = (result_df["Sa_pred_um"] <= float(i_sa)) & (result_df["Sz_pred_um"] <= float(i_sz))
                    feasible_mask = result_df["roughness_pass"].copy()
                    if feed_max > 0: feasible_mask &= result_df["F_mm_min"] <= float(feed_max)
                    if mrr_max > 0: feasible_mask &= result_df["MRR_mm3_min"] <= float(mrr_max)
                    result_df["feasible"] = feasible_mask
                    result_df["violation"] = (
                        np.maximum(0, (result_df["Sa_pred_um"] - float(i_sa)) / float(i_sa)) ** 2
                        + np.maximum(0, (result_df["Sz_pred_um"] - float(i_sz)) / float(i_sz)) ** 2
                    )
                    feasible = result_df[result_df["feasible"]].copy()
                    if len(feasible) > 0:
                        ranked = feasible.sort_values(["MRR_mm3_min", "Sa_pred_um", "Sz_pred_um"], ascending=[False, True, True]).head(int(top_k))
                    else:
                        ranked = result_df.sort_values(["violation", "Sa_pred_um", "Sz_pred_um"], ascending=[True, True, True]).head(int(top_k))
                    ranked.attrs["elapsed_ms"] = elapsed
                    ranked.attrs["feasible_count"] = len(feasible)
                    st.session_state.inverse_results = ranked
                except Exception as error:
                    st.error(str(error))

        with ranked_col:
            st.markdown('<div class="ui-card-label">RANKED MACHINING SOLUTION</div>', unsafe_allow_html=True)
            if st.session_state.inverse_results is None:
                st.markdown('<div class="inverse-placeholder"><div><strong>AWAITING SEARCH</strong>Set the target surface and run inverse search.</div></div>', unsafe_allow_html=True)
            else:
                display_df = st.session_state.inverse_results.copy()
                best = display_df.iloc[0]
                best_state = "FEASIBLE" if bool(best.get("feasible", False)) else "CLOSEST AVAILABLE"
                st.markdown(
                    f"""
                    <div class="best-candidate">
                        <div class="best-candidate__label">TOP CANDIDATE / {best_state}</div>
                        <div class="best-candidate__value">
                            n <b>{best['n_rpm']:.0f}</b> rpm · fz <b>{best['fz_mm_per_tooth']:.3f}</b> mm/tooth ·
                            ap <b>{best['ap_mm']:.2f}</b> mm · ae <b>{best['ae_mm']:.1f}</b> mm ·
                            Sa <b>{best['Sa_pred_um']:.3f}</b> µm · Sz <b>{best['Sz_pred_um']:.3f}</b> µm ·
                            MRR <b>{best['MRR_mm3_min']:.1f}</b> mm³/min
                        </div>
                    </div>
                    """, unsafe_allow_html=True,
                )
                table_df = display_df[[
                    "n_rpm","fz_mm_per_tooth","ap_mm","ae_mm","Sa_pred_um","Sz_pred_um","F_mm_min","MRR_mm3_min","feasible"
                ]]
                st.dataframe(table_df, width="stretch", hide_index=True)
                st.download_button("EXPORT RESULTS · CSV", table_df.to_csv(index=False), "inverse_results.csv", "text/csv", width="stretch")


    # =============================================================================
    # Ra showcase check tab
    # =============================================================================
    with ra_tab:
        section_header(
            "04 / QUALITY GATE",
            "Ra Quality Check",
            "Measured Ra versus the drawing limit.",
        )

        ra_col1, ra_col2, ra_col3 = st.columns([1, 1, 1.15])
        with ra_col1:
            ra_measured = st.number_input(
                "Measured Ra [µm]",
                min_value=0.001,
                max_value=100.0,
                value=0.800,
                step=0.050,
                format="%.3f",
                key="ra_measured_um",
                help="Enter the Ra value reported by the profilometer, or use a manual demo value for showcasing.",
            )
        with ra_col2:
            ra_limit = st.number_input(
                "Maximum allowable Ra [µm]",
                min_value=0.001,
                max_value=100.0,
                value=1.600,
                step=0.050,
                format="%.3f",
                key="ra_limit_um",
            )
        with ra_col3:
            ra_source = st.selectbox(
                "Inspection source",
                ["Profilometer / external measurement", "Manual demo entry"],
                key="ra_source",
            )

        ra_ref_col, ra_action_col = st.columns([1.6, 1])
        with ra_ref_col:
            ra_sample_ref = st.text_input(
                "Part / sample reference",
                value="DEMO-01",
                key="ra_sample_ref",
                help="Display label only; it is not used in any calculation.",
            )
        with ra_action_col:
            st.write("")
            st.write("")
            ra_check_clicked = st.button(
                "CHECK RA QUALITY",
                type="primary",
                width="stretch",
                key="run_ra_quality_check",
            )

        if ra_check_clicked:
            ra_value = float(ra_measured)
            ra_max = float(ra_limit)
            ra_passed = ra_value <= ra_max
            ra_margin = ra_max - ra_value
            ra_utilization = (ra_value / ra_max) * 100.0
            st.session_state.ra_check_result = {
                "value": ra_value,
                "limit": ra_max,
                "passed": ra_passed,
                "margin": ra_margin,
                "utilization": ra_utilization,
                "source": str(ra_source),
                "sample_ref": str(ra_sample_ref).strip() or "UNNAMED",
            }

        ra_result = st.session_state.ra_check_result

        if ra_result is None:
            st.markdown(
                """
                <div class="ra-ticket">
                    <div class="ra-ticket__status ra-ticket__status--pending">
                        <div>
                            <div class="ra-ticket__eyebrow">Ra inspection state</div>
                            <div class="ra-ticket__word">READY</div>
                        </div>
                        <div class="ra-ticket__rule">Rule / Ra measured ≤ Ra maximum<br>Waiting for inspection input.</div>
                    </div>
                    <div class="ra-ticket__body">
                        <div class="ra-ticket__head"><span>Inspection ticket</span><span>MILLTWIN // RA-QC</span></div>
                        <div class="ra-ticket__metrics">
                            <div class="ra-ticket__metric"><span>Measured Ra</span><strong>—<em>µm</em></strong></div>
                            <div class="ra-ticket__metric"><span>Ra limit</span><strong>—<em>µm</em></strong></div>
                            <div class="ra-ticket__metric"><span>Margin</span><strong>—<em>µm</em></strong></div>
                        </div>
                        <div class="ra-ticket__foot">Manual quality-gate showcase. The deployed ONNX model remains Sa/Sz-only.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            ra_passed = bool(ra_result["passed"])
            status_word = "PASS" if ra_passed else "REVIEW"
            status_class = "" if ra_passed else "ra-ticket__status--review"
            fill_class = "" if ra_passed else "ra-ticket__meter-fill--review"
            meter_width = max(0.0, min(float(ra_result["utilization"]), 100.0))
            margin_label = "Margin" if ra_passed else "Exceedance"
            margin_value = abs(float(ra_result["margin"]))
            ref_text = html.escape(str(ra_result["sample_ref"]))
            source_text = html.escape(str(ra_result["source"]))
            utilization = float(ra_result["utilization"])

            st.markdown(
                f"""
                <div class="ra-ticket">
                    <div class="ra-ticket__status {status_class}">
                        <div>
                            <div class="ra-ticket__eyebrow">Ra inspection state</div>
                            <div class="ra-ticket__word">{status_word}</div>
                        </div>
                        <div class="ra-ticket__rule">
                            Rule / Ra measured ≤ Ra maximum<br>
                            Sample / {ref_text}
                        </div>
                    </div>
                    <div class="ra-ticket__body">
                        <div class="ra-ticket__head"><span>Inspection ticket</span><span>MILLTWIN // RA-QC</span></div>
                        <div class="ra-ticket__metrics">
                            <div class="ra-ticket__metric"><span>Measured Ra</span><strong>{ra_result['value']:.3f}<em>µm</em></strong></div>
                            <div class="ra-ticket__metric"><span>Ra limit</span><strong>{ra_result['limit']:.3f}<em>µm</em></strong></div>
                            <div class="ra-ticket__metric"><span>{margin_label}</span><strong>{margin_value:.3f}<em>µm</em></strong></div>
                        </div>
                        <div>
                            <div class="ra-ticket__meter-wrap">
                                <div class="ra-ticket__meter-label"><span>Limit utilization</span><span>{utilization:.1f}%</span></div>
                                <div class="ra-ticket__meter"><div class="ra-ticket__meter-fill {fill_class}" style="width:{meter_width:.1f}%"></div></div>
                            </div>
                            <div class="ra-ticket__foot">Source / {source_text} · Reference / {ref_text} · This QC result is not a PIDL prediction.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ra_passed:
                st.success(
                    f"Ra accepted — {ra_result['value']:.3f} ≤ {ra_result['limit']:.3f} µm. "
                    f"Remaining margin: {margin_value:.3f} µm."
                )
            else:
                st.error(
                    f"Ra requires review — {ra_result['value']:.3f} > {ra_result['limit']:.3f} µm. "
                    f"Exceedance: {margin_value:.3f} µm."
                )

        st.caption("QC uses measured/manual Ra only; PIDL inference remains Sa/Sz-only.")


    # =============================================================================
    # Prediction history tab
    # =============================================================================
    with history_tab:
        section_header("03 / MACHINING LOGBOOK", "History", "Reuse previous predictions as a compact reference log.")
        history = st.session_state.prediction_history

        if not history:
            st.markdown('<div class="inverse-placeholder"><div><strong>NO SAVED RUNS</strong>Forward predictions will appear here automatically.</div></div>', unsafe_allow_html=True)
        else:
            history_df = pd.DataFrame(history)
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("SAVED RUNS", f"{len(history_df)}")
            with m2: st.metric("PASS", f"{int((history_df['decision'] == 'PASS').sum())}")
            with m3: st.metric("REVIEW", f"{int((history_df['decision'] == 'REVIEW').sum())}")
            with m4: st.metric("LATEST", str(history_df.iloc[-1]["created_at"])[11:16])

            filter_col, export_col = st.columns([1, 1])
            with filter_col:
                decision_filter = st.selectbox("Decision filter", ["ALL", "PASS", "REVIEW"], key="history_decision_filter")
            with export_col:
                st.write("")
                st.download_button("EXPORT HISTORY · CSV", history_df.to_csv(index=False), "milltwin_prediction_history.csv", "text/csv", width="stretch")

            view_df = history_df.copy()
            if decision_filter != "ALL": view_df = view_df[view_df["decision"] == decision_filter]
            display_columns = ["created_at","n_rpm","fz_mm_per_tooth","ap_mm","ae_mm","Sa_pred_um","Sz_pred_um","decision","note"]
            st.dataframe(view_df[display_columns].sort_values("created_at", ascending=False), width="stretch", hide_index=True)

            st.markdown('<div class="ui-card-label">SAVED RUN ACTIONS</div>', unsafe_allow_html=True)
            record_indices = list(range(len(history) - 1, -1, -1))
            selected_index = st.selectbox(
                "Saved prediction",
                record_indices,
                format_func=lambda idx: f"{history[idx]['created_at']} · n {history[idx]['n_rpm']:.0f} · fz {history[idx]['fz_mm_per_tooth']:.3f} · Sa {history[idx]['Sa_pred_um']:.3f} · Sz {history[idx]['Sz_pred_um']:.3f} · {history[idx]['decision']}",
                key="history_selected_index",
            )
            selected_record = history[int(selected_index)]
            d1,d2,d3,d4 = st.columns(4)
            with d1: st.metric("Sa", f"{selected_record['Sa_pred_um']:.3f} µm")
            with d2: st.metric("Sz", f"{selected_record['Sz_pred_um']:.3f} µm")
            with d3: st.metric("DECISION", selected_record["decision"])
            with d4: st.metric("LATENCY", f"{selected_record['inference_ms']:.3f} ms")
            note_key = f"history_note_{selected_record['record_id']}"
            note_value = st.text_area("Engineering note", value=str(selected_record.get("note", "")), key=note_key, placeholder="Process note…")
            a1,a2,a3,a4 = st.columns(4)
            with a1:
                st.button("REUSE PARAMETERS", type="primary", width="stretch", on_click=reuse_history_record, args=(selected_record,))
            with a2:
                if st.button("SAVE NOTE", width="stretch"):
                    st.session_state.prediction_history[int(selected_index)]["note"] = note_value
                    st.success("Note saved.")
            with a3:
                if st.button("DELETE RECORD", width="stretch"):
                    del st.session_state.prediction_history[int(selected_index)]
                    st.session_state.pop("history_selected_index", None)
                    st.rerun()
            with a4:
                if st.button("CLEAR HISTORY", width="stretch"):
                    st.session_state.prediction_history = []
                    st.session_state.pop("history_selected_index", None)
                    st.rerun()
            st.caption("Session log only. A persistent machining handbook can be added in a later product phase.")


    # =============================================================================
    # FEM comparison tab
    # =============================================================================
    with fem_tab:
        section_header(
            "05 / EVIDENCE CHECK",
            "Validation",
            "Compare predictions against labelled FEM or measurement data.",
        )

        fem_file = st.file_uploader(
            "Reference dataset [CSV]",
            type=["csv"],
            key="fem_upload",
        )

        if fem_file is not None:
            try:
                fem = pd.read_csv(fem_file)
                fem = normalize_fem_columns(fem)

                required = [*FEATURES, *TARGETS]
                missing = [column for column in required if column not in fem.columns]

                if missing:
                    st.error(f"Missing required FEM columns: {missing}")
                elif model_session is None:
                    st.error("No ONNX model is available. Train or upload a model first.")
                else:
                    raw_X_fem = fem[FEATURES].to_numpy(dtype=np.float32)
                    X_fem = encode_model_inputs(raw_X_fem, "down")
                    pred_fem, elapsed_fem = predict(model_session, X_fem)

                    result_fem = fem.copy()
                    result_fem["Sa_pred_um"] = pred_fem[:, 0]
                    result_fem["Sz_pred_um"] = pred_fem[:, 1]
                    result_fem["Sa_error_um"] = result_fem["Sa_pred_um"] - result_fem["Sa_um"]
                    result_fem["Sz_error_um"] = result_fem["Sz_pred_um"] - result_fem["Sz_um"]
                    result_fem["Sa_abs_error_um"] = result_fem["Sa_error_um"].abs()
                    result_fem["Sz_abs_error_um"] = result_fem["Sz_error_um"].abs()

                    metrics_fem = fem_metrics(fem, pred_fem)

                    st.success(f"FEM comparison completed. Inference time: {elapsed_fem:.3f} ms.")

                    st.markdown("#### Validation summary")
                    for _, metric_row in metrics_fem.iterrows():
                        target_name = str(metric_row["target"]).replace("_um", "")
                        v1, v2, v3, v4 = st.columns(4)
                        with v1:
                            st.metric(f"{target_name} · R²", f"{metric_row['R2']:.4f}")
                        with v2:
                            st.metric(f"{target_name} · MAE", f"{metric_row['MAE_um']:.4f} µm")
                        with v3:
                            st.metric(f"{target_name} · RMSE", f"{metric_row['RMSE_um']:.4f} µm")
                        with v4:
                            st.metric(f"{target_name} · MAPE", f"{metric_row['MAPE_percent']:.2f}%")

                    with st.expander("Detailed validation metrics", expanded=False):
                        st.dataframe(
                            metrics_fem.style.format(
                                {
                                    "R2": "{:.6f}",
                                    "MAE_um": "{:.6f}",
                                    "RMSE_um": "{:.6f}",
                                    "MAPE_percent": "{:.3f}",
                                }
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                    with st.expander("Pointwise comparison · raw rows", expanded=False):
                        st.dataframe(
                            result_fem.style.format(
                                {
                                    "n_rpm": "{:.0f}",
                                    "fz_mm_per_tooth": "{:.2f}",
                                    "ap_mm": "{:.2f}",
                                    "ae_mm": "{:.1f}",
                                    "eps_r_um": "{:.1f}",
                                    "eps_a_um": "{:.1f}",
                                    "Sa_um": "{:.3f}",
                                    "Sz_um": "{:.3f}",
                                    "Sa_pred_um": "{:.3f}",
                                    "Sz_pred_um": "{:.3f}",
                                    "Sa_error_um": "{:.3f}",
                                    "Sz_error_um": "{:.3f}",
                                    "Sa_abs_error_um": "{:.3f}",
                                    "Sz_abs_error_um": "{:.3f}",
                                }
                            ),
                            width="stretch",
                        )

                    st.download_button(
                        label="EXPORT VALIDATION RESULTS · CSV",
                        data=result_fem.to_csv(index=False),
                        file_name="fem_comparison.csv",
                        mime="text/csv",
                        width="stretch",
                    )

            except Exception as error:
                st.error(str(error))
        else:
            st.info("Upload FEM/external CSV to compare model predictions against external labels.")


    # =============================================================================
    # Dossier / guide / model scope
    # =============================================================================
    with info_tab:
        section_header("06 / SYSTEM GUIDE", "Dossier", "Quick start, model scope and contact in one place.")

        st.markdown(
            """
            <div class="dossier-grid">
              <div class="dossier-panel">
                <div class="dossier-panel__title">01 / QUICK START</div>
                <div class="quick-steps">
                  <div class="quick-step"><div class="quick-step__n">01 →</div><div class="quick-step__name">Forward</div><div class="quick-step__copy">Predict Sa / Sz from machining inputs.</div></div>
                  <div class="quick-step"><div class="quick-step__n">02 →</div><div class="quick-step__name">Inverse</div><div class="quick-step__copy">Find candidate parameters for a target surface.</div></div>
                  <div class="quick-step"><div class="quick-step__n">03 →</div><div class="quick-step__name">History</div><div class="quick-step__copy">Reuse prior predictions as a reference log.</div></div>
                  <div class="quick-step"><div class="quick-step__n">04 →</div><div class="quick-step__name">Ra QC</div><div class="quick-step__copy">Check measured Ra against a drawing limit.</div></div>
                  <div class="quick-step"><div class="quick-step__n">05 →</div><div class="quick-step__name">Validation</div><div class="quick-step__copy">Compare model output with labelled data.</div></div>
                </div>
              </div>
              <div class="dossier-panel">
                <div class="dossier-panel__title">02 / CONTACT</div>
                <div class="contact-stack">
                  <a class="contact-pill" href="tel:0703166078"><span class="contact-pill__icon">☎</span><span>0703166078</span></a>
                  <a class="contact-pill" href="mailto:phuc.lerobotic@hcmut.edu.vn"><span class="contact-pill__icon">✉</span><span>phuc.lerobotic@hcmut.edu.vn</span></a>
                </div>
                <div class="roadmap-note" style="margin-top:1rem;">G-code generation is intentionally excluded. The current product stops at decision support and expert-level parameter reference. A persistent machining handbook can be considered later.</div>
              </div>
            </div>
            <div class="dossier-panel" style="margin-bottom:1rem;">
              <div class="dossier-panel__title">03 / MOTION LANGUAGE</div>
              <div class="motion-list">
                <div class="motion-item"><strong>Hover lift</strong>Controls rise slightly to signal interactivity.</div>
                <div class="motion-item"><strong>Underline sweep</strong>Navigation reveals the magenta accent on hover.</div>
                <div class="motion-item"><strong>Soft drift</strong>The engineering grid moves slowly in the background.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("MODEL SCHEMA / DESIGN DOMAIN", expanded=False):
            schema_rows = []
            for feature in FEATURES:
                low, high = clean_range(feature, raw_ranges)
                schema_rows.append({
                    "Column": feature,
                    "Type": "Input",
                    "Unit": {"n_rpm":"rpm","fz_mm_per_tooth":"mm/tooth","ap_mm":"mm","ae_mm":"mm","eps_r_um":"µm","eps_a_um":"µm"}.get(feature, ""),
                    "UI range": f"{low:g} – {high:g}",
                })
            schema_rows.append({"Column": MILLING_MODE, "Type": "Fixed condition", "Unit": "categorical", "UI range": "down (fixed)"})
            for target in TARGETS:
                schema_rows.append({"Column": target, "Type": "Output", "Unit": "µm", "UI range": "-"})
            st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

        with st.expander("MODEL INFORMATION", expanded=False):
            info_summary = {
                "model_type": info.get("model_type", "unknown"),
                "selected_model": info.get("selected_model", info.get("selected_pidl_model", "unknown")),
                "best_validation_accuracy_model": info.get("best_validation_accuracy_model", "unknown"),
                "dataset_file": info.get("dataset_file", "dataset_endmill_v1_valid.csv"),
                "dataset_rows": info.get("dataset_rows", "unknown"),
                "split_method": info.get("split_method", "unknown"),
                "split": info.get("split", {}),
                "feature_columns": info.get("raw_input_columns", [*FEATURES, MILLING_MODE]),
                "model_feature_columns": info.get("model_feature_columns", MODEL_FEATURES),
                "target_columns": info.get("target_columns", TARGETS),
                "fixed_tool": info.get("fixed_tool", {"D_mm": 6.0, "Zn": 4, "beta_deg": 45.0}),
                "physical_scope": info.get("physical_scope", "unknown"),
            }
            st.json(info_summary)

        with st.expander("ACCURACY / PHYSICS DIAGNOSTICS", expanded=False):
            metrics_path = find_latest_project_file(["model_metrics.csv", "ablation_results.csv", "metrics.csv"])
            if metrics_path is not None:
                try:
                    st.dataframe(pd.read_csv(metrics_path), width="stretch", hide_index=True)
                except Exception as error:
                    st.warning(f"Could not read {metrics_path}: {error}")
            else:
                st.info("No model metrics file found.")
            physics_path = find_latest_project_file(["physics_diagnostics.csv", "physics_metrics.csv"])
            if physics_path is not None:
                try:
                    st.dataframe(pd.read_csv(physics_path), width="stretch", hide_index=True)
                except Exception as error:
                    st.warning(f"Could not read {physics_path}: {error}")

        with st.expander("LIMITATIONS / CLOUD DEPLOYMENT", expanded=False):
            limitations = info.get("limitations", [
                "Trained on D6 EndMill FSM synthetic labels only.",
                "Current physics excludes vibration, chatter, material constitutive behavior and tool wear.",
                "FEM or experimental validation is required before claiming real machining accuracy.",
            ])
            for item in limitations:
                st.write(f"- {item}")
            required_files = [
                "app.py",
                "logo.png",
                "milltwin_pidl_sasz.onnx",
                "info.json",
                "model_metrics.csv",
                "physics_diagnostics.csv",
                "ablation_results.csv",
                "model_card.md",
                "requirements.txt",
            ]
            file_rows = []
            for file in required_files:
                located_file = find_latest_project_file([file])
                file_rows.append({"File":file,"Exists":located_file is not None,"Location":str(located_file) if located_file else ""})
            st.dataframe(pd.DataFrame(file_rows), width="stretch", hide_index=True)


    # =============================================================================
    # Team credit
    # =============================================================================
    if brand_logo_b64:
        st.markdown(
            f"""
            <div class="app-footer">
                <div class="app-footer__logo-frame">
                    <img class="app-footer__logo" src="data:image/png;base64,{brand_logo_b64}" alt="Millcore logo">
                </div>
                <div class="app-footer__text">
                    <strong>Engineered by Millcore.</strong><br>
                    MillTwin-Lite · physics-informed CNC surface intelligence / deployment interface.
                </div>
                <div class="app-footer__stamp">MILLCORE // SMART MACHINING</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Millcore Systems · Smart machining. Precise results.")
