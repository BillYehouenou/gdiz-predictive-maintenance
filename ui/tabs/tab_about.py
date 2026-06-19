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


def _pipe_card(idx: int, color: str, title: str, desc: str, C: dict) -> str:
    bg = f"linear-gradient(160deg,{rgba(color, 0.10)} 0%,{rgba(color, 0.03)} 55%,transparent 100%)"
    # Le nom de classe varie avec idx (pipe-anim-0, pipe-anim-1, ...) pour forcer le
    # navigateur à rejouer l'animation à chaque changement d'étape — une animation CSS
    # ne se relance pas tant que le nom de classe qui la déclenche reste identique.
    return (
        f"<div class='pipe-card pipe-anim-{idx}' style='background:{bg};'>"
        f"<p class='pipe-title' style='color:{C['text']};'>{title}</p>"
        f"<div class='pipe-desc'>{desc}</div>"
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
         "ROC-AUC", "0.843", "Split temporel 2024 → 2025", C["blue"]),
        ("https://img.icons8.com/?size=100&id=0ZNXJnXqu3Lh&format=png&color=FFFFFF",
         "F2-Score", "0.382", "Recall 56 % · Précision 17 %", C["orange"]),
        ("https://img.icons8.com/?size=100&id=hvx78MXvVF91&format=png&color=000000",
         "Seuil F2", "0.06", "Logit-normalisé à 50 %", C["red"]),
    ]
    m1, m2, m3 = st.columns(3)
    for col, (icon, label, val, sub, color), delay in zip([m1, m2, m3], row2, [0.14, 0.21, 0.28]):
        col.markdown(_kpi_card(icon, label, val, sub, color, delay, C), unsafe_allow_html=True)

    # ── Pipeline interactif ────────────────────────────────────────────────────
    st.markdown("<p class='sec-label' style='margin-top:1.8rem;'>Pipeline</p>", unsafe_allow_html=True)

    STEPS = [
        ("Données", C["teal"],
         "Les données sont entièrement simulées pour reproduire fidèlement le parc textile de la "
         "<strong>GDIZ</strong>, en s'inspirant du référentiel industriel AI4I 2020. Chaque machine évolue pas à "
         "pas sur 2 ans (température, vibrations, usure) en tenant compte de la météo du Bénin et des coupures "
         "électriques de la SBEE. Chaque machine a aussi son propre <strong>seuil de casse</strong> : certaines "
         "tiennent jusqu'à 95-99 % d'usure, d'autres cassent dès 65-70 %, comme dans un vrai parc industriel "
         "hétérogène plutôt qu'avec un seuil unique irréaliste. Une panne peut survenir pour "
         "<strong>5 causes différentes</strong> : usure de l'outil, surchauffe, surcharge mécanique, panne "
         "électrique ou panne purement aléatoire. Le tout est stocké dans une base de données ultra-rapide, "
         "<strong>DuckDB</strong>, taillée pour l'analyse de gros volumes de séries temporelles."
        ),

        ("Indicateurs", C["blue"],
         "Pour repérer une panne avant qu'elle n'arrive, le système ne se contente pas de regarder l'instant "
         "présent : une vibration élevée à un moment donné peut être normale ou anormale selon le contexte. Il "
         "compare donc chaque mesure à son <strong>évolution récente</strong> sur deux fenêtres glissantes — "
         "<strong>4 heures</strong> pour capter une accélération brutale (delta rapide d'usure ou de "
         "température) et <strong>24 heures</strong> pour capter une tendance plus lente (moyenne, écart-type, "
         "pic maximal). C'est cette dynamique dans le temps, plus que la valeur brute du moment, qui révèle une "
         "machine en train de se dégrader progressivement."
        ),

        ("Préparation des données", C["orange"],
         "Avant d'entraîner l'IA, les données brutes doivent être nettoyées et standardisées automatiquement : "
         "les valeurs manquantes (capteurs défaillants, pertes de signal IoT) sont imputées, toutes les mesures "
         "numériques sont mises à la même échelle (StandardScaler) pour qu'aucune variable ne soit favorisée "
         "simplement parce qu'elle a de grandes valeurs, et les variables catégorielles comme le type de machine "
         "sont converties en colonnes numériques (OneHotEncoder). Cette chaîne de préparation est figée dans un "
         "artefact unique, pour garantir que le dashboard et l'API traitent toujours les données "
         "<strong>exactement comme lors de l'entraînement</strong> — un point critique pour éviter tout écart "
         "silencieux entre entraînement et production."
        ),

        ("ML", C["red"],
         "Le modèle utilisé est <strong>LightGBM</strong>, un algorithme d'arbres de décision en gradient "
         "boosting reconnu pour sa rapidité sur des données tabulaires. Il est entraîné sur l'année 2024 et "
         "testé sur 2025, pour vérifier qu'il généralise bien dans le temps plutôt que d'apprendre le passé par "
         "cœur. Il vise spécifiquement les pannes qui ont un véritable signal annonciateur — usure, surchauffe, "
         "surcharge mécanique — sur un horizon de <strong>5 jours</strong> ; les pannes électriques ou "
         "purement aléatoires sont volontairement exclues de la cible, car elles n'ont, par nature, rien à "
         "apprendre, et les inclure aurait seulement dilué le signal utile. Résultat mesuré : plus d'une panne "
         "prévisible sur deux est détectée (recall 56 %), avec près d'une alerte sur six réellement fondée "
         "(précision 17 %)."
        ),

        ("Moteur de calcul", C["green"],
         "Une fois entraîné, le modèle est exposé via une API web (<strong>FastAPI</strong>) ultra-rapide, "
         "indépendante du dashboard. Dès qu'une mesure de machine est envoyée, ce service recalcule à la volée "
         "les indicateurs temporels (4h/24h), applique exactement le même pipeline de préparation que celui "
         "utilisé à l'entraînement, puis renvoie en quelques millisecondes la probabilité de panne et la "
         "décision de maintenance. Le modèle est chargé depuis le registre <strong>MLflow</strong> ; en cas "
         "d'indisponibilité de celui-ci, l'API bascule automatiquement sur une copie locale du modèle pour "
         "garantir une continuité de service."
        ),

        ("Interface", C["blue"],
         "Le dashboard final donne une vue d'ensemble du parc des 30 machines avec leurs alertes en temps réel, "
         "un suivi historique détaillé par machine et des indicateurs de coût liés aux pannes (production "
         "perdue, pièces, matière gâchée). Le risque affiché pour une machine correspond toujours à la "
         "probabilité de panne prévisible <strong>dans les 5 prochains jours</strong> ; le sélecteur 24h/7j/30j "
         "ne change que la période d'historique consultée pour repérer le pic de risque, jamais l'horizon de "
         "prédiction lui-même. L'affichage reste instantané car <strong>Streamlit</strong> interroge directement "
         "<strong>DuckDB</strong> avec des requêtes optimisées en colonnes."
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
    title, color, desc = STEPS[idx]

    st.markdown(
        _pipe_card(idx, color, title, desc, C),
        unsafe_allow_html=True,
    )
