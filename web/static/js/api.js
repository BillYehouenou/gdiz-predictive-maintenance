// Accès à l'API et petits outils d'affichage partagés par les pages.

export async function lire(chemin) {
  const reponse = await fetch(`/api/v1${chemin}`);
  if (!reponse.ok) throw new Error(`${chemin} : ${reponse.status} ${(await reponse.json().catch(() => ({}))).detail ?? ""}`);
  return reponse.json();
}

export async function envoyer(chemin, corps = {}) {
  const reponse = await fetch(`/api/v1${chemin}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(corps),
  });
  if (!reponse.ok) throw new Error((await reponse.json().catch(() => ({}))).detail ?? `Erreur ${reponse.status}`);
  return reponse.json();
}

// Échappe tout texte venu des données avant de l'insérer dans le HTML
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const fr = (d) => new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: d });
export const nombre = (v, d = 0) => (v == null ? "—" : fr(d).format(v));
export const pourcent = (v, d = 1) => (v == null ? "—" : `${fr(d).format(v * 100)} %`);
export const millions = (v, d = 1) => `${fr(d).format(v / 1e6)} M`;
export const dateCourte = (iso) => new Date(iso).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" });
export const dateHeure = (iso) =>
  new Date(iso).toLocaleString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).replace(" ", " · ");

export const mouvementReduit = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Les nombres marqués data-compte défilent jusqu'à leur valeur (depuis data-depart, 0 par défaut)
export function compter(racine = document) {
  racine.querySelectorAll("[data-compte]").forEach((el) => {
    const cible = Number(el.dataset.compte), depart = Number(el.dataset.depart ?? 0), dec = Number(el.dataset.dec ?? 0);
    const format = (v) => `${nombre(v, dec)}${el.dataset.suffixe ?? ""}`;
    if (mouvementReduit() || !Number.isFinite(cible)) { el.textContent = format(cible); return; }
    const debut = performance.now(), duree = 1100;
    const pas = (t) => {
      const k = Math.min(1, (t - debut) / duree);
      el.textContent = format(depart + (cible - depart) * (1 - Math.pow(1 - k, 4)));
      if (k < 1) requestAnimationFrame(pas);
    };
    requestAnimationFrame(pas);
  });
}

export const LIBELLES_STATUT = { normal: "Normal", surveillance: "Surveillance", panne_predite: "Panne prédite", en_panne: "En panne" };
export const CAUSES = {
  TWF: { libelle: "Usure", couleur: "var(--yellow)" },
  HDF: { libelle: "Surchauffe", couleur: "var(--orange)" },
  OSF: { libelle: "Surcharge méca.", couleur: "var(--teal)" },
  PWF: { libelle: "Électrique", couleur: "var(--blue)" },
  RNF: { libelle: "Aléatoire", couleur: "var(--gray-c)" },
};
export const FAMILLES = { "Usure et entretien": "var(--yellow)", Surchauffe: "var(--orange)", "Surcharge mécanique": "var(--teal)", "Environnement / réseau": "var(--blue)" };

export const squelette = (h = 120) => `<div class="squelette" style="--h: ${h}px" aria-label="Chargement"></div>`;
export const vide = (texte) => `<p class="vide">${esc(texte)}</p>`;

// Affiche l'erreur dans le conteneur, avec un bouton qui relance le chargement
export function erreur(conteneur, e, relancer) {
  conteneur.innerHTML = `<div class="erreur" role="alert"><span>Chargement impossible : ${esc(e.message)}</span>
    <button type="button" class="btn btn-contour">Réessayer</button></div>`;
  conteneur.querySelector("button").addEventListener("click", relancer);
}
