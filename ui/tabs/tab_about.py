from pathlib import Path
import streamlit as st
from ui.helpers import rgba

_ROOT = Path(__file__).resolve().parent.parent.parent

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
        ("DuckDB", "#e6d817",
         "Les données simulent fidèlement le parc textile de la <strong>GDIZ</strong>, inspiré du dataset AI4I 2020 grâce à un modèle basé sur 3 piliers :"
         "<ul style='margin-top: 5px; padding-left: 20px;'>"
         "<li>La météo du Bénin et les coupures électriques de la SBEE qui impactent toutes les machines ;</li>"
         "<li>Chaque machine accumule sa propre usure, sa chaleur et ses vibrations d'un moment à l'autre ;</li>"
         "<li>Le tout est stocké dans une base de données ultra-rapide <strong>DuckDB</strong> après l'ajout de bruits de capteurs (bruits capteurs IoT).</li>"
         "</ul>"
        ),
         
        ("Indicateurs", "#e6d817",
         "Pour repérer les pannes avant qu'elles n'arrivent, nous calculons des indicateurs clés combinant les <strong>mesures instantanées</strong> des capteurs et leur <strong>historique sur une fenêtre glissante de 24 heures</strong>. "
         "Ces analyses temporelles permettent à l'IA de voir venir une dégradation lente qu'un simple contrôle ponctuel ne détecterait pas."
        ),
         
        ("Préparation des données", "#e6d817",
         "Avant d'alimenter l'IA, les données brutes sont nettoyées automatiquement. Les valeurs manquantes sont imputées, les chiffres sont mis à la même échelle (StandardScaler) "
         "et les variables catégrorielles comme le type de machine sont traduits en valeurs numériques (OneHotEncoder). "
         "Ce bloc de préparation est figé dans un artefact pour garantir que les données du dashboard soient traitées <strong>exactement de la même manière</strong> que lors de l'entraînement de l'IA."
        ),
         
        ("ML", "#e6d817",
         "Nous utilisons l'algorithme <strong>LightGBM</strong> configuré pour surmonter le déséquilibre extrême des données. L'IA a été entraînée sur l'année 2024 "
         "et testée sur 2025. Le seuil d'alerte a été réglé à <strong>1,5 % de suspicion</strong> : "
         "c'est le point d'équilibre idéal (F2-Score) pour intercepter un maximum de pannes sans déclencher "
         "trop de fausses alertes. Toutes les performances sont sauvegardées et tracées dans <strong>MLflow</strong>."),
         
        ("Moteur de calcul", "#e6d817",
         "L'IA est propulsée sous forme d'un service web privé API REST ultra-rapide. "
         "Dès que le dashboard lui envoie les mesures d'une machine, ce moteur calcule les indicateurs sur 24h, applique les filtres et renvoie instantanément la probabilité de panne et la décision "
         "de maintenance. Si l'outil de suivi MLflow est en panne, l'API bascule automatiquement sur un fichier de secours pour garantir 100% de disponibilité."
        ),
         
        ("Interface", C["blue"],
         "L'application finale offre une vue d'ensemble du parc des 30 machines avec leurs voyants d'alerte, un suivi historique détaillé par machine et des conseils de maintenance. "
         "L'affichage est instantané car <strong>Streamlit</strong> interroge directement la base de données avec des requêtes optimisées en colonnes."
        )
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
