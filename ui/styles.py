import streamlit as st


def apply_styles(C: dict, DARK: bool) -> None:
    _shadow = "0 1px 4px rgba(0,0,0,.35)" if DARK else "0 1px 3px rgba(0,0,0,.08)"
    _toggle_glow_on = "rgba(79,195,247,.55)" if DARK else "rgba(79,195,247,.45)"
    _toggle_glow_off = "rgba(253,193,53,.55)" if DARK else "rgba(251,140,0,.45)"
    _track_off_color = "#fb8c00"

    st.markdown(
        f"""
<style>
  [data-testid="stAppViewContainer"] {{ background:{C["bg"]}; }}
  [data-testid="block-container"]    {{ padding:0.6rem 2.5rem 2rem; }}
  #MainMenu, footer, header          {{ visibility:hidden; }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background:{C["card"]}; border:1px solid {C["border"]};
    border-radius:10px; padding:.85rem 1.1rem;
    box-shadow:{_shadow}; transition:border-color .2s;
  }}
  [data-testid="metric-container"]:hover {{ border-color:{C["blue"]}40; }}
  [data-testid="stMetricLabel"] {{
    font-size:.72rem; text-transform:uppercase; letter-spacing:.07em;
    color:{C["faint"]}; font-weight:500;
  }}
  [data-testid="stMetricValue"] {{ color:{C["text"]}; font-size:1.55rem; font-weight:600; }}
  [data-testid="stMetricDelta"]  {{ display:none; }}

  /* Tabs */
  button[data-baseweb="tab"] {{
    background:transparent!important; color:{C["faint"]}!important;
    font-size:.82rem; letter-spacing:.04em; font-weight:500;
    border-bottom:2px solid transparent!important; padding:.5rem 1.1rem;
  }}
  button[aria-selected="true"][data-baseweb="tab"] {{
    color:{C["blue"]}!important; border-bottom:2px solid {C["blue"]}!important;
  }}

  /* Typography */
  h1 {{ color:{C["text"]}!important; font-size:1.3rem!important; font-weight:600!important; margin:0!important; }}
  h2 {{ color:{C["text"]}!important; font-size:.98rem!important; font-weight:500!important; }}
  h3, h4 {{ color:{C["text"]}!important; }}
  p, li {{ color:{C["muted"]}!important; font-size:.84rem!important; }}
  hr {{ border-color:{C["border"]}!important; opacity:1!important; margin:.8rem 0!important; }}

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {{
    background:{C["card"]}; border-color:{C["border"]}; color:{C["text"]}; border-radius:8px;
  }}

  /* ── Premium toggle switch ── */
  [data-testid="stToggle"] {{
    display:flex; justify-content:center; align-items:center;
    padding-top:1.1rem;
    transition: filter 0.4s cubic-bezier(.4,0,.2,1);
  }}
  [data-testid="stToggle"]:has(input:checked) {{
    filter: drop-shadow(0 0 9px {_toggle_glow_on});
  }}
  [data-testid="stToggle"]:has(input:not(:checked)) {{
    filter: drop-shadow(0 0 9px {_toggle_glow_off});
  }}
  [data-testid="stToggle"] label {{
    transform: scale(1.35);
    transform-origin: center;
    cursor: pointer;
  }}
  [data-testid="stToggle"] input:not(:checked) + div {{
    background-color: {_track_off_color} !important;
    transition: background-color 0.3s ease !important;
  }}
  [data-testid="stToggle"] input + div > div {{
    box-shadow: 0 2px 8px rgba(0,0,0,.35), 0 0 0 1px rgba(255,255,255,.08) !important;
    transition: transform 0.3s cubic-bezier(.4,0,.2,1) !important;
  }}

  /* Segmented control — période & horizon */
  [data-testid="stSegmentedControl"] {{
    background: {C["card"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
    padding: 3px !important;
  }}
  [data-testid="stSegmentedControl"] button {{
    background: transparent !important;
    color: {C["muted"]} !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
    transition: background .15s, color .15s !important;
  }}
  [data-testid="stSegmentedControl"] button:hover {{
    background: {C["border"]} !important;
    color: {C["text"]} !important;
  }}
  [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
    background: {C["blue"]}22 !important;
    color: {C["blue"]} !important;
    box-shadow: 0 0 0 1px {C["blue"]}44 !important;
  }}
</style>
""",
        unsafe_allow_html=True,
    )
