import streamlit as st

from ui.helpers import rgba


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
    @keyframes cardPop {{
        0%   {{ opacity: 0; transform: translateY(10px) scale(.97); }}
        60%  {{ opacity: 1; transform: translateY(-2px) scale(1.008); }}
        100% {{ opacity: 1; transform: translateY(0) scale(1); }}
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
        color: {C["faint"]};
        font-weight: 600;
        margin: 1.6rem 0 .6rem;
    }}
    .live-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        background: {C["green"]};
        border-radius: 50%;
        animation: pulseDot 2s ease infinite;
        vertical-align: middle;
        margin-right: 6px;
    }}
    .pipe-resume {{
        font-size: .98rem;
        font-weight: 650;
        line-height: 1.4;
        margin: 0 0 1rem;
    }}
    .pipe-list {{
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: .6rem;
    }}
    .pipe-item {{
        display: flex;
        align-items: flex-start;
        gap: .65rem;
        font-size: .85rem;
        line-height: 1.5;
        color: {C["muted"]};
    }}
    .pipe-item .dot {{
        flex-shrink: 0;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-top: .5rem;
    }}
    .pipe-item strong {{
        color: {C["text"]};
        font-weight: 650;
    }}
    .pipe-item code {{
        font-size: .78rem;
        background: {C["border"]};
        border-radius: 4px;
        padding: .05rem .3rem;
        color: {C["text"]};
    }}
    .pipe-card {{
        position: relative;
        border-radius: 15px;
        padding: 1.25rem 1.5rem 1.2rem;
        margin-top: 1.1rem;
        min-height: 230px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
    }}
    [class^="pipe-anim-"] {{
        animation: cardPop .4s cubic-bezier(.22,1,.36,1) both;
    }}
    .pipe-title {{
        font-size: 1.05rem;
        font-weight: 750;
        letter-spacing: -.01em;
        margin: 0 0 .65rem;
    }}
    .st-key-pipeline_pills button {{
        padding: .15rem .7rem !important;
        min-height: 1.65rem !important;
        height: 1.65rem !important;
        font-size: .74rem !important;
        font-weight: 600 !important;
        letter-spacing: .01em;
        border-radius: 999px !important;
        transition: transform .15s cubic-bezier(.22,1,.36,1), background-color .2s ease, color .2s ease;
    }}
    .st-key-pipeline_pills button:active {{
        transform: scale(.88);
    }}
    .st-key-pipeline_pills [data-testid="stButtonGroup"] {{
        gap: .3rem !important;
    }}
    </style>
    """


def _carte_kpi(icon: str, label: str, val: str, sub: str, color: str, delay: float, C: dict) -> str:
    bg = f"linear-gradient(135deg,{rgba(color, 0.22)} 0%,{rgba(color, 0.08)} 100%)"
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


def _carte_etape(idx: int, color: str, title: str, resume: str, points: list[str], C: dict) -> str:
    bg = f"linear-gradient(160deg,{rgba(color, 0.10)} 0%,{rgba(color, 0.03)} 55%,transparent 100%)"
    # Le nom de classe varie avec idx (pipe-anim-0, pipe-anim-1, ...) pour forcer le
    # navigateur à rejouer l'animation à chaque changement d'étape — une animation CSS
    # ne se relance pas tant que le nom de classe qui la déclenche reste identique.
    items = "".join(
        f"<li class='pipe-item'><span class='dot' style='background:{color};'></span><span>{p}</span></li>" for p in points
    )
    return (
        f"<div class='pipe-card pipe-anim-{idx}' style='background:{bg};'>"
        f"<p class='pipe-title' style='color:{C['text']};'>{title}</p>"
        f"<p class='pipe-resume' style='color:{color};'>{resume}</p>"
        f"<ul class='pipe-list'>{items}</ul>"
        f"</div>"
    )


def afficher(C: dict) -> None:
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
        ("https://img.icons8.com/?size=100&id=Ouhuf6HfyHeJ&format=png&color=FFFFFF", "Machines", "50", "Textiles", C["green"]),
        (
            "https://img.icons8.com/?size=100&id=KZHjwwenS7oK&format=png&color=000000",
            "Données",
            "1.75M",
            "enrégistrements sur 2 ans",
            C["teal"],
        ),
    ]
    c1, c2 = st.columns(2)
    for col, (icon, label, val, sub, color), delay in zip([c1, c2], row1, [0.0, 0.07]):
        col.markdown(_carte_kpi(icon, label, val, sub, color, delay, C), unsafe_allow_html=True)

    st.markdown("<div style='margin:.5rem 0;'></div>", unsafe_allow_html=True)

    # ── Rangée 2 : Métriques (recall événementiel + fausses alertes + seuil) ──
    row2 = [
        (
            "https://img.icons8.com/?size=100&id=Yt084riMRP1m&format=png&color=FFFFFF",
            "Recall événementiel",
            "84.3%",
            "Split temporel 2024 → 2025",
            C["blue"],
        ),
        (
            "https://img.icons8.com/?size=100&id=0ZNXJnXqu3Lh&format=png&color=FFFFFF",
            "Fausses alertes",
            "0.22",
            "Épisodes / machine / jour",
            C["orange"],
        ),
        (
            "https://img.icons8.com/?size=100&id=hvx78MXvVF91&format=png&color=000000",
            "Préavis médian",
            "12.5h",
            "Sur les pannes correctement anticipées",
            C["red"],
        ),
    ]
    m1, m2, m3 = st.columns(3)
    for col, (icon, label, val, sub, color), delay in zip([m1, m2, m3], row2, [0.14, 0.21, 0.28]):
        col.markdown(_carte_kpi(icon, label, val, sub, color, delay, C), unsafe_allow_html=True)

    # ── Pipeline interactif ────────────────────────────────────────────────────
    st.markdown("<p class='sec-label' style='margin-top:1.8rem;'>Pipeline</p>", unsafe_allow_html=True)

    STEPS = [
        (
            "Données",
            C["teal"],
            "Un jumeau numérique du parc GDIZ, pas un jeu de données générique.",
            [
                "<strong>50 machines textiles</strong> simulées pas à pas sur 2 ans, inspirées du référentiel "
                "industriel AI4I 2020.",
                "<strong>Climat béninois</strong> et coupures électriques SBEE reproduits fidèlement.",
                "<strong>Seuil de casse individuel</strong> par machine (65 à 99 % d'usure) — un parc hétérogène, "
                "pas un seuil unique irréaliste.",
                "<strong>5 causes de panne</strong> possibles : usure, surchauffe, surcharge mécanique, électrique, aléatoire.",
                "Stocké dans <code>DuckDB</code>, optimisé pour l'analyse de séries temporelles à grande échelle.",
            ],
        ),
        (
            "Indicateurs",
            C["blue"],
            "La tendance récente compte plus que l'instant présent.",
            [
                "Chaque mesure est comparée à son <strong>évolution récente</strong>, jamais lue isolément.",
                "<strong>Fenêtre 4h</strong> : détecte une accélération brutale (delta rapide d'usure, de température).",
                "<strong>Fenêtre 24h</strong> : capte une tendance plus lente (moyenne, écart-type, pic).",
                "Une dégradation progressive se voit dans la <strong>dynamique</strong>, pas dans une valeur brute.",
            ],
        ),
        (
            "Préparation des données",
            C["orange"],
            "Même traitement à l'entraînement et en production — zéro écart silencieux.",
            [
                "<strong>Contrôle de sanité</strong> : bornes physiques vérifiées par capteur avant tout traitement.",
                "<strong>Valeurs manquantes</strong> imputées automatiquement (capteurs défaillants, pertes IoT).",
                "<strong>Catégories encodées nativement</strong> pour LightGBM — pas de mise à l'échelle, pas de one-hot.",
                "Un <strong>artefact unique et figé</strong> garantit l'identité entraînement / dashboard / API.",
            ],
        ),
        (
            "ML",
            C["red"],
            "84 % des pannes prévisibles détectées, 5 jours à l'avance.",
            [
                "<code>LightGBM</code> (gradient boosting), calibré par régression isotonique pour des "
                "probabilités interprétables.",
                "Entraîné sur <strong>2024</strong>, testé sur <strong>2025</strong> — généralisation dans le "
                "temps, pas par cœur.",
                "Horizon de prédiction : <strong>5 jours</strong>, sur usure / surchauffe / surcharge mécanique.",
                "Pannes électriques et aléatoires <strong>exclues</strong> — aucun signal précurseur, vérifié empiriquement.",
                "<strong>84 % de recall</strong>, ~1 fausse alerte tous les 4-5 jours par machine.",
            ],
        ),
        (
            "Moteur de calcul",
            C["green"],
            "Une API qui répond en millisecondes, sans jamais dévier du modèle entraîné.",
            [
                "<code>FastAPI</code>, endpoint unitaire + endpoint par lot pour un relevé de parc complet.",
                "<strong>Bornes physiques validées</strong> à l'entrée avant toute prédiction.",
                "Mêmes indicateurs 4h/24h, <strong>même pipeline</strong> qu'à l'entraînement.",
                "Suivi <code>MLflow</code> asynchrone — ne ralentit jamais la réponse.",
                "<strong>Rechargement à chaud</strong> après ré-entraînement, sans redémarrer le service.",
                "Repli automatique sur copie locale si le registre MLflow est indisponible.",
            ],
        ),
        (
            "Interface",
            C["blue"],
            "Ce qu'il faut savoir, pas tout ce qu'il y a à voir.",
            [
                "Liste <strong>« à traiter aujourd'hui »</strong>, triée par risque décroissant.",
                "<strong>Cause probable</strong> par alerte (SHAP) — usure, surchauffe, surcharge mécanique...",
                "Risque toujours exprimé <strong>sur 5 jours</strong>, jamais une heure inventée.",
                "Suivi historique détaillé par machine et coûts de panne chiffrés.",
                "<code>Streamlit</code> + <code>DuckDB</code> en colonnes → affichage instantané.",
            ],
        ),
    ]

    step_names = [s[0] for s in STEPS]
    selected = st.pills(
        "Étape du pipeline",
        options=step_names,
        default=step_names[0],
        label_visibility="collapsed",
        key="pipeline_pills",
    )
    if selected is None:
        selected = step_names[0]

    idx = step_names.index(selected)
    title, color, resume, points = STEPS[idx]

    st.markdown(
        _carte_etape(idx, color, title, resume, points, C),
        unsafe_allow_html=True,
    )
