// Commun aux deux pages : accès à l'API, formats, animations, puis l'en-tête (onglets, compte, thème, export PDF).

export async function lire(chemin) {
  const reponse = await fetch(`/api/v1${chemin}`);
  if (!reponse.ok) throw new Error(`${chemin} : ${reponse.status} ${(await reponse.json().catch(() => ({}))).detail ?? ""}`);
  return reponse.json();
}

// Échappe tout texte venu des données avant de l'insérer dans le HTML
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const fr = (d) => new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: d });
export const nombre = (v, d = 0) => (v == null ? "—" : fr(d).format(v));
export const pourcent = (v, d = 1) => (v == null ? "—" : `${fr(d).format(v * 100)} %`);
export const millions = (v, d = 1) => `${fr(d).format(v / 1e6)} M`;
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

// Pastille de sélection qui glisse sous l'élément actif d'un groupe de pilules ; renvoie la fonction qui la replace
export function pastille(groupe, actif) {
  const curseur = document.createElement("span");
  curseur.className = "curseur immobile";
  curseur.setAttribute("aria-hidden", "true");
  groupe.classList.add("glisse");
  groupe.prepend(curseur);
  const placer = (cible = groupe.querySelector(actif), anime = true) => {
    if (!cible) return;
    curseur.classList.toggle("immobile", !anime);
    curseur.style.width = `${cible.offsetWidth}px`;
    curseur.style.transform = `translateX(${cible.offsetLeft}px)`;
  };
  const replacer = () => placer(undefined, false);
  replacer();
  window.addEventListener("resize", replacer);
  document.fonts?.ready.then(replacer);
  return placer;
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

// Affiche l'erreur dans le conteneur, avec un bouton qui relance le chargement
export function erreur(conteneur, e, relancer) {
  conteneur.innerHTML = `<div class="erreur" role="alert"><span>Chargement impossible : ${esc(e.message)}</span>
    <button type="button" class="btn btn-contour">Réessayer</button></div>`;
  conteneur.querySelector("button").addEventListener("click", relancer);
}

const racine = document.documentElement;
const bouton = document.getElementById("compte");
const menu = document.getElementById("menu-compte");
const theme = document.getElementById("theme");

const majInterrupteur = () => theme.setAttribute("aria-checked", String(racine.dataset.theme === "dark"));

function ouvrirMenu(ouvert) {
  menu.hidden = !ouvert;
  bouton.setAttribute("aria-expanded", String(ouvert));
}

bouton.addEventListener("click", () => ouvrirMenu(menu.hidden));
document.addEventListener("click", (e) => { if (!e.target.closest(".compte")) ouvrirMenu(false); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { ouvrirMenu(false); bouton.focus(); } });

theme.addEventListener("click", () => {
  racine.dataset.theme = racine.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", racine.dataset.theme); } catch {}
  majInterrupteur();
});
majInterrupteur();

// Onglets : la pastille glisse vers l'onglet cliqué, puis la page change
const onglets = document.querySelector(".onglets");
const placerOnglet = pastille(onglets, '[aria-current="page"]');
const ongletDeLaPage = onglets.querySelector('[aria-current="page"]');
onglets.querySelectorAll("a").forEach((lien) =>
  lien.addEventListener("click", (e) => {
    if (lien.getAttribute("aria-current") === "page" || e.metaKey || e.ctrlKey || e.shiftKey || mouvementReduit()) return;
    e.preventDefault();
    onglets.querySelector('[aria-current="page"]')?.removeAttribute("aria-current");
    lien.setAttribute("aria-current", "page");
    placerOnglet(lien);
    setTimeout(() => { window.location.href = lien.href; }, 320);
  }),
);
// Retour arrière depuis le cache du navigateur : l'onglet de la page redevient l'onglet actif
window.addEventListener("pageshow", (e) => {
  if (!e.persisted || !ongletDeLaPage) return;
  onglets.querySelectorAll("a").forEach((a) => a.removeAttribute("aria-current"));
  ongletDeLaPage.setAttribute("aria-current", "page");
  placerOnglet(ongletDeLaPage, false);
});

// L'impression du navigateur produit le PDF ; le nom du fichier vient du titre de la page
document.querySelectorAll("[data-export]").forEach((b) =>
  b.addEventListener("click", () => {
    const { title } = document, choix = racine.dataset.theme;
    document.title = `Rapport GDIZ ${b.dataset.export} ${new Date().toISOString().slice(0, 10)}`;
    racine.dataset.theme = "light";
    window.print();
    document.title = title;
    racine.dataset.theme = choix;
  }),
);
