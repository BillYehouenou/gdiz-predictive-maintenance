from pathlib import Path
import streamlit as st
from ui.helpers import rgba

_ROOT = Path(__file__).resolve().parent.parent.parent

_DUCKDB_COLOR = "#e6a817"


def _css(C: dict) -> str:
    return f"""
    <style>
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseDot {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50%       {{ transform: scale(1.35); opacity: 0.55; }}
    }}
    @keyframes slideRight {{
        from {{ transform: scaleX(0); }}
        to   {{ transform: scaleX(1); }}
    }}
    .abt-kpi {{
        animation: fadeInUp .45s ease both;
        border-radius: 14px;
        padding: 1.2rem 1.1rem 1rem;
        box-sizing: border-box;
    }}
    .sec-label {{
        font-size: .68rem;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: {C['faint']};
        font-weight: 600;
        margin: 1.6rem 0 .6rem;
    }}
    .live-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        background: {C['green']};
        border-radius: 50%;
        animation: pulseDot 2s ease infinite;
        vertical-align: middle;
        margin-right: 6px;
    }}
    .pipe-desc {{
        font-size: .85rem;
        line-height: 1.80;
        color: {C['muted']};
    }}
    .pipe-desc strong {{
        color: {C['text']};
        font-weight: 600;
    }}
    .pipe-desc code {{
        font-size: .78rem;
        background: {C['border']};
        border-radius: 4px;
        padding: .05rem .3rem;
        color: {C['text']};
    }}
    .pipe-progress-track {{
        margin-top: .9rem;
        height: 3px;
        border-radius: 2px;
        background: {C['border']};
        overflow: hidden;
    }}
    .pipe-progress-fill {{
        height: 100%;
        border-radius: 2px;
        transform-origin: left;
        animation: slideRight .55s cubic-bezier(.22,1,.36,1) both;
    }}
    </style>
    """


def _kpi_card(icon: str, label: str, val: str, sub: str, color: str, delay: float, C: dict) -> str:
    bg = f"linear-gradient(135deg,{rgba(color,0.22)} 0%,{rgba(color,0.08)} 100%)"
    muted = C["muted"]
    label_color = rgba(color, 0.75)
    return (
        f"<div class='abt-kpi' style='background:{bg};animation-delay:{delay:.2f}s;"
        f"display:flex;align-items:center;gap:1rem;'>"
        f"<img src='{icon}' width='52' height='52' style='flex-shrink:0;opacity:.92;'/>"
        f"<div>"
        f"<div style='font-size:.6rem;letter-spacing:.08em;text-transform:uppercase;"
        f"color:{label_color};font-weight:700;margin-bottom:.1rem;'>{label}</div>"
        f"<div style='font-size:2rem;font-weight:800;color:{color};line-height:1.05;'>{val}</div>"
        f"<div style='font-size:.69rem;color:{muted};margin-top:.15rem;'>{sub}</div>"
        f"</div>"
        f"</div>"
    )


def _pipe_card(idx: int, total: int, color: str, desc: str, C: dict) -> str:
    pct = round((idx + 1) / total * 100)
    muted = C["muted"]
    border = C["border"]

    dots = "".join(
        f"<div style='width:{12 if i == idx else 5}px;height:4px;border-radius:2px;"
        f"background:{muted if i == idx else border};'></div>"
        for i in range(total)
    )

    return (
        f"<div style='animation:fadeInUp .35s ease both;margin-top:.7rem;'>"
        f"<div class='pipe-desc'>{desc}</div>"
        f"<div class='pipe-progress-track'>"
        f"<div class='pipe-progress-fill' style='width:{pct}%;background:{color};'></div>"
        f"</div>"
        f"<div style='display:flex;gap:5px;margin-top:.5rem;'>{dots}</div>"
        f"</div>"
    )


def render(C: dict) -> None:
    st.markdown(_css(C), unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("<p class='sec-label' style='margin-top:0;'>Modèle</p>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin:.4rem 0 1.5rem;'>"
        f"<p style='color:{C['muted']};font-size:.87rem;margin:0;'>"
        f"<span class='live-dot'></span>"
        f"Pipeline de maintenance prédictive par IA"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # ── Rangée 1 : Contexte (Machines + Données) ──────────────────────────────
    row1 = [
        ("https://img.icons8.com/?size=100&id=Ouhuf6HfyHeJ&format=png&color=FFFFFF",
         "Machines", "30", "Textiles", C["green"]),
        ("https://img.icons8.com/?size=100&id=KZHjwwenS7oK&format=png&color=000000",
         "Données", "1.05M", "enrégistrements sur 2 ans", C["teal"]),
    ]
    c1, c2 = st.columns(2)
    for col, (icon, label, val, sub, color), delay in zip([c1, c2], row1, [0.0, 0.07]):
        col.markdown(_kpi_card(icon, label, val, sub, color, delay, C), unsafe_allow_html=True)

    st.markdown("<div style='margin:.5rem 0;'></div>", unsafe_allow_html=True)

    # ── Rangée 2 : Métriques (ROC-AUC + F2 + Seuil) ──────────────────────────
    row2 = [
        ("https://img.icons8.com/?size=100&id=Yt084riMRP1m&format=png&color=FFFFFF",
         "ROC-AUC", "0.825", "Split temporel 2024 - 2025", C["blue"]),
        ("https://img.icons8.com/?size=100&id=0ZNXJnXqu3Lh&format=png&color=FFFFFF",
         "F2-Score", "0.055", "Métrique de performance", C["orange"]),
        ("https://img.icons8.com/?size=100&id=hvx78MXvVF91&format=png&color=000000",
         "Seuil F2", "0.015", "Logit-normalisé à 50 %", C["red"]),
    ]
    m1, m2, m3 = st.columns(3)
    for col, (icon, label, val, sub, color), delay in zip([m1, m2, m3], row2, [0.14, 0.21, 0.28]):
        col.markdown(_kpi_card(icon, label, val, sub, color, delay, C), unsafe_allow_html=True)

    # ── Pipeline interactif ────────────────────────────────────────────────────
    st.markdown("<p class='sec-label' style='margin-top:1.8rem;'>Pipeline</p>", unsafe_allow_html=True)

    STEPS = [
        ("DuckDB", _DUCKDB_COLOR,
         "Les données sont <strong>générées synthétiquement</strong> selon le framework AI4I 2020 "
         "adapté au parc textile de la GDIZ. Architecture en 3 étapes : <strong>A</strong> — contexte "
         "exogène global partagé par toutes les machines (saisons du Bénin, température ambiante, "
         "cycle électrique SBEE avec délestages, pluies par chaîne de Markov) ; <strong>B</strong> — "
         "évolution séquentielle par machine avec dépendance N−1 (usure cumulative, inertie thermique "
         "α=0,55, vibrations, probabilité de panne) ; <strong>C</strong> — agrégation, injection de "
         "bruit IoT (2 % de valeurs manquantes) et ingestion dans une base <strong>DuckDB</strong> "
         "embarquée indexée sur <code>(machine_id, timestamp)</code>. "
         "Résultat : 30 machines × 730 jours × 48 pas = <strong>1,05 M lignes</strong>."),
        ("Features", C["teal"],
         "Trois <strong>features temporelles</strong> calculées sur fenêtre glissante de 24 h "
         "(48 pas × 30 min) enrichissent chaque observation : "
         "<code>tool_wear_delta_24h</code> — taux d'accumulation d'usure sur 24 h, clampé à 0 "
         "pour absorber les remises à zéro post-maintenance ; "
         "<code>vibration_max_24h</code> — pire vibration récente, plus robuste qu'un point "
         "instantané ; <code>process_temp_max_24h</code> — pic thermique sur 24 h, reflet du "
         "stress mécanique cumulé. Ces trois signaux rendent visible la dynamique de dégradation "
         "qu'un snapshot instantané ne peut pas capturer."),
        ("Preprocessor", C["muted"],
         "Le <code>Preprocessor</code> est un <code>BaseEstimator</code> scikit-learn encapsulant "
         "un <code>ColumnTransformer</code> : les features numériques passent par "
         "<code>SimpleImputer(median)</code> puis <code>StandardScaler</code> ; les variables "
         "catégorielles (<code>benin_season</code>, <code>machine_type</code>) par "
         "<code>SimpleImputer(most_frequent)</code> puis <code>OneHotEncoder</code>. "
         "Fitté une seule fois sur les données 2024 et sérialisé dans <code>model_pipeline.pkl</code>, "
         "il garantit une transformation <strong>strictement identique</strong> à l'entraînement "
         "et à l'inférence, sans data leakage."),
        ("LightGBM", C["orange"],
         "<strong>LightGBM</strong> entraîné avec <code>class_weight='balanced'</code> pour "
         "compenser le déséquilibre extrême (~0,02 % de pannes). <strong>Split temporel strict</strong> : "
         "entraînement sur 2024, évaluation sur 2025 — aucune donnée future n'est vue pendant le fit. "
         "Le seuil de décision est optimisé sur le <strong>F2-Score</strong> (recall pondéré 2×) "
         "sur une grille [0,005 ; 0,50], aboutissant à <strong>0,015</strong>. La probabilité brute "
         "est ensuite normalisée via logit pour que 0,015 → 50 % affiché. "
         "Métriques, artefacts et importances de features tracés via <strong>MLflow</strong>."),
        ("FastAPI", C["green"],
         "Le modèle est exposé en <strong>API REST</strong> via FastAPI (<code>POST /api/v1/predict</code>). "
         "Chaque requête enrichit le vecteur reçu avec les features temporelles "
         "(<code>enrich_inference_point</code>), applique le pipeline de prétraitement et retourne "
         "la probabilité logit-normalisée avec la décision de maintenance (alerte / nominal). "
         "En cas d'indisponibilité de MLflow, un repli automatique sur "
         "<code>model_pipeline.pkl</code> garantit la continuité de service."),
        ("Streamlit", C["blue"],
         "Dashboard temps réel : vue globale des 30 machines avec indicateurs d'alerte, "
         "détail par machine avec séries historiques, probabilités normalisées et recommandations "
         "de maintenance. Les données sont interrogées à chaque interaction directement depuis "
         "<strong>DuckDB</strong> via des lectures vectorisées en colonnes pour une réponse rapide."),
    ]

    step_names = [s[0] for s in STEPS]
    selected = st.select_slider(
        "Étape du pipeline",
        options=step_names,
        label_visibility="collapsed",
    )

    idx = step_names.index(selected)
    _, color, desc = STEPS[idx]

    st.markdown(
        _pipe_card(idx, len(STEPS), color, desc, C),
        unsafe_allow_html=True,
    )
