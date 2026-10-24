// Commun à toutes les pages : compteurs de la navigation, recherche d'une machine (⌘K), thème.
import { dateHeure, esc, lire } from "./api.js";

const champ = document.getElementById("recherche-champ");

async function chargerEtat() {
  try {
    const etat = await lire("/etat");
    const c = etat.comptes;
    const alertes = (c.en_panne ?? 0) + (c.panne_predite ?? 0) + (c.surveillance ?? 0);
    document.getElementById("compteur-parc").textContent = alertes || "";
    document.getElementById("compteur-alertes").textContent = etat.interventions_a_traiter || "";
    document.getElementById("fin-donnees").textContent = `Données au ${dateHeure(etat.fin_donnees)}`;
    document.getElementById("liste-machines").innerHTML = etat.machines
      .map((m) => `<option value="M${m.numero}">${esc(m.nom)} · gamme ${esc(m.gamme)}</option>`)
      .join("");
  } catch {
    document.getElementById("fin-donnees").textContent = "Données indisponibles";
  }
}

document.getElementById("recherche").addEventListener("submit", (e) => {
  e.preventDefault();
  const n = parseInt(champ.value.replace(/\D/g, ""), 10);
  if (n >= 1 && n <= 50) window.location.href = `/machines/GDIZ_TEX_M_${String(n).padStart(2, "0")}`;
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    champ.focus();
  }
});

document.getElementById("theme").addEventListener("click", () => {
  const sombre = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = sombre ? "light" : "dark";
  try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch {}
});

// Les pages rappellent ce chargement après une action qui change les compteurs
window.addEventListener("etat-change", chargerEtat);
chargerEtat();
