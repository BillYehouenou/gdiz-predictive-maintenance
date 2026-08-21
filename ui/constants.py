from pathlib import Path

# ─── FEATURES ML ──────────────────────────────────────────────────────────────
FEAT_COLS: list[str] = [
    "saison_benin",
    "type_machine",
    "temperature_ambiante",
    "temperature_process",
    "vitesse_rotation",
    "couple",
    "usure_outil",
    "niveau_activite",
    "vibration",
    "humidite",
    "concentration_poussiere",
    "indicateur_pluie",
    "indicateur_coupure",
    "niveau_tension",
    "stabilite_tension",
    # Features temporelles — calculées par enrichir_serie_inference(), doit rester
    # synchronisé avec config/config.yaml::jeu_donnees.features (une divergence ici
    # envoie un jeu de features incomplet au modèle, cf. docs/model-decisions.md).
    "usure_outil_delta_24h",
    "usure_outil_delta_4h",
    "vibration_max_24h",
    "vibration_moyenne_24h",
    "vibration_ecart_type_24h",
    "vibration_delta_4h",
    "temperature_process_max_24h",
    "temperature_process_moyenne_24h",
    "temperature_process_delta_4h",
    "ecart_thermique",
    "ecart_thermique_max_24h",
    "poussiere_moyenne_24h",
    "poussiere_max_24h",
    "couple_max_24h",
]

# ─── COÛTS DE MAINTENANCE — constantes métier GDIZ Bénin ─────────────────────
# C_panne = C_production + C_technique + C_qualite
#
# Challenge : taux technicien à 2 500 XOF/h est conservateur (marché : 3 000–5 000 XOF/h).
# Pièces Type H : 250 000 XOF est le minimum OEM — peut atteindre 500 000 XOF haut de gamme.
# Tissu Type H : 15 m semble conservateur à 35 m/h (inertie mécanique ~26 min).
_TECH_RATE: int = 2_500  # XOF/h — taux horaire technicien (commun à tous types)
_MAT_COST: int = 800  # XOF/m — coût de revient matière textile

# C_production : capacité × marge nette par mètre
_PROD: dict = {
    "L": {"capacity_m_per_h": 15, "margin_xof_per_m": 1_200},  # 18 000 XOF/h d'arrêt
    "M": {"capacity_m_per_h": 20, "margin_xof_per_m": 1_500},  # 30 000 XOF/h d'arrêt
    "H": {"capacity_m_per_h": 35, "margin_xof_per_m": 2_000},  # 70 000 XOF/h d'arrêt
}
# C_technique : pièce de rechange (fixe par panne) + technicien (variable)
_PARTS: dict = {"L": 45_000, "M": 90_000, "H": 250_000}  # XOF par panne
# C_qualite : tissu gâché au crash × coût matière
_WASTE: dict = {"L": 5, "M": 8, "H": 15}  # mètres perdus

# Coefficients précalculés
_PROD_COEFF = {t: _PROD[t]["capacity_m_per_h"] * _PROD[t]["margin_xof_per_m"] for t in ("L", "M", "H")}
_QUAL_FIX = {t: _WASTE[t] * _MAT_COST for t in ("L", "M", "H")}

# ─── BASE DE DONNÉES ──────────────────────────────────────────────────────────
DB_PATH = Path("data/raw/gdiz_maintenance.db")
TABLE = "factory_telemetry"

# ─── PÉRIODES ─────────────────────────────────────────────────────────────────
PERIOD_MAP: dict[str, tuple[str, str]] = {
    "30j": ("2025-12-01", "2025-12-30"),
    "T4 2025": ("2025-10-01", "2025-12-30"),
    "S2 2025": ("2025-07-01", "2025-12-30"),
    "Année 2025": ("2025-01-01", "2025-12-30"),
    "2024-2025": ("2024-01-01", "2025-12-30"),
}
PERIOD_PREV_MAP: dict[str, tuple[str, str] | None] = {
    "30j": ("2025-11-01", "2025-11-30"),
    "T4 2025": ("2025-07-01", "2025-09-30"),
    "S2 2025": ("2025-01-01", "2025-06-30"),
    "Année 2025": ("2024-01-01", "2024-12-31"),
    "2024-2025": None,
}
PERIOD_PREV_LABEL: dict[str, str] = {
    "30j": "Nov 2025",
    "T4 2025": "T3 2025",
    "S2 2025": "S1 2025",
    "Année 2025": "2024",
    "2024-2025": "",
}
STEPS_MAP: dict[str, int] = {"24h": 48, "7j": 336, "30j": 1440}

# ─── SEUILS ENVIRONNEMENTAUX ──────────────────────────────────────────────────
HARMATTAN_THRESHOLD = 600  # µg/m³ — seuil extrême (non encore atteint)
HARMATTAN_WARN = 300  # µg/m³ — alerte Harmattan fort

# ─── PALETTES ─────────────────────────────────────────────────────────────────
_DARK = {
    "blue": "#4fc3f7",
    "green": "#52c41a",
    "orange": "#fa8c16",
    "red": "#f5222d",
    "teal": "#13c2c2",
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#21262d",
    "text": "#e6edf3",
    "muted": "#8b97a8",
    "faint": "#6e7681",
    "plot_bg": "rgba(13,17,23,.55)",
    "grid": "rgba(255,255,255,.035)",
}
_LIGHT = {
    "blue": "#0969da",
    "green": "#1a7f37",
    "orange": "#bc4c00",
    "red": "#cf222e",
    "teal": "#0a7d7d",
    "bg": "#ffffff",
    "card": "#f6f8fa",
    "border": "#d0d7de",
    "text": "#1f2328",
    "muted": "#636c76",
    "faint": "#8c959f",
    "plot_bg": "rgba(246,248,250,.8)",
    "grid": "rgba(0,0,0,.06)",
}
