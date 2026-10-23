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
    .pipe-para strong {{
        color: {C["text"]};
        font-weight: 650;
    }}
    .pipe-para code {{
        font-size: .78rem;
        background: {C["border"]};
        border-radius: 4px;
        padding: .05rem .3rem;
        color: {C["text"]};
    }}
    .pipe-para {{
        font-size: .85rem;
        line-height: 1.6;
        color: {C["muted"]};
        margin: 0;
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


def _carte_etape(idx: int, color: str, title: str, resume: str, paragraphe: str, C: dict) -> str:
    bg = f"linear-gradient(160deg,{rgba(color, 0.10)} 0%,{rgba(color, 0.03)} 55%,transparent 100%)"
    return (
        f"<div class='pipe-card pipe-anim-{idx}' style='background:{bg};'>"
        f"<p class='pipe-title' style='color:{C['text']};'>{title}</p>"
        f"<p class='pipe-resume' style='color:{color};'>{resume}</p>"
        f"<p class='pipe-para'>{paragraphe}</p>"
        f"</div>"
    )


def afficher(C: dict) -> None:
    st.markdown(_css(C), unsafe_allow_html=True)

    # Section metriques
    st.markdown("<p class='sec-label' style='margin-top:0;'>Modèle</p>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin:.4rem 0 1.5rem;'>"
        f"<p style='color:{C['muted']};font-size:.87rem;margin:0;'>"
        f"<span class='live-dot'></span>"
        f"Pipeline de maintenance prédictive"
        f"</p></div>",
        unsafe_allow_html=True,
    )

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

    # Pipeline interactif
    st.markdown("<p class='sec-label' style='margin-top:1.8rem;'>Pipeline</p>", unsafe_allow_html=True)

    STEPS = [
        (
            "Données",
            C["teal"],
            "Un jumeau numérique du parc GDIZ, pas un jeu de données générique.",
            "Plusieurs machines textiles sont simulées pas à pas sur 2 ans, sous la <strong>météo réelle</strong> de Glo-Djigbé "
            "(Open-Meteo : température, humidité, pluie, poussière d'Harmattan) et des coupures SBEE simulées. Chaque machine "
            "casse à son propre seuil d'usure, avec 5 causes de panne possibles (usure, surchauffe, surcharge mécanique, "
            "électrique, aléatoire). L'usure elle-même reste <strong>cachée</strong> : comme en usine, le modèle ne voit que "
            "les capteurs et les compteurs de maintenance. Le tout est stocké dans une base DuckDB.",
        ),
        (
            "Indicateurs",
            C["blue"],
            "La tendance récente compte plus que l'instant présent.",
            "Plutôt qu'une mesure isolée, le modèle regarde <strong>l'évolution récente </strong> des mesures récupérées sur "
            "chaque machine : une fenêtre de <strong>4h</strong> pour repérer une accélération brutale (usure ou température "
            "qui grimpe vite), et une fenêtre de <strong>24h</strong> pour capter une tendance plus lente. C'est cette "
            "dynamique, plus que la valeur brute, qui trahit une dégradation progressive.",
        ),
        (
            "Préparation des données",
            C["orange"],
            "Même traitement à l'entraînement et en production pour éviter un quelconque écart silencieux.",
            "Avant tout traitement, chaque capteur passe un <strong>contrôle de sanité</strong> (bornes physiques "
            "vérifiées) et les valeurs manquantes sont imputées automatiquement. Les catégories sont encodées "
            "nativement pour LightGBM, sans mise à l'échelle superflue. Un <strong>artefact unique et figé</strong> "
            "garantit que l'entraînement, le dashboard et l'API appliquent exactement le même traitement.",
        ),
        (
            "ML",
            C["red"],
            "65 % des pannes prévisibles détectées, jusqu'à 5 jours à l'avance.",
            "Le modèle est un LightGBM, calibré pour donner des probabilités fiables et interprétables. Entraîné sur 2024 et "
            "testé sur 2025, il apprend à généraliser dans le temps plutôt qu'à retenir par cœur, et prédit les pannes "
            "d'usure, de surchauffe et de surcharge mécanique. Les pannes électriques et aléatoires étant exclues pour "
            "l'instant du périmètre détectable car sans signal précurseur. Le recall est mis en face d'une métrique de coût "
            "d'exploitation du modèle. Ici, on accepte une fausse alerte tous les 4 à 5 jours par machine.",
        ),
        (
            "Moteur de calcul",
            C["green"],
            "Une API qui répond en millisecondes, sans jamais dévier du modèle entraîné.",
            "L'API répond en millisecondes, avec un mode unitaire et un mode par lot pour analyser tout le parc d'un coup. "
            "Chaque valeur reçue est validée avant toute prédiction, avec exactement le même pipeline qu'à l'entraînement. "
            "Le modèle peut être <strong>rechargé à chaud</strong> après un ré-entraînement, sans jamais interrompre le "
            "service, et bascule automatiquement sur une copie locale si le registre MLflow est indisponible.",
        ),
        (
            "Interface",
            C["blue"],
            "Ce qu'il faut savoir, pas tout ce qu'il y a à voir.",
            "Le dashboard affiche une liste <strong>« à traiter aujourd'hui »</strong>, triée par risque décroissant, avec "
            "pour chaque alerte sa <strong>cause probable</strong>. Le risque est toujours exprimé sur l'horizon de "
            "<strong>5 jours</strong> du modèle, et chaque machine dispose d'un historique détaillé avec ses coûts de panne "
            "chiffrés.",
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
