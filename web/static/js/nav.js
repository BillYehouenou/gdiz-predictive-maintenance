// Commun aux deux pages : onglets, menu du compte, thème, export du rapport en PDF.
import { mouvementReduit, pastille } from "./api.js";

const racine = document.documentElement;
const bouton = document.getElementById("compte");
const menu = document.getElementById("menu-compte");
const theme = document.getElementById("theme");

const estSombre = () => (racine.dataset.theme ? racine.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
const majInterrupteur = () => theme.setAttribute("aria-checked", String(estSombre()));

function ouvrirMenu(ouvert) {
  menu.hidden = !ouvert;
  bouton.setAttribute("aria-expanded", String(ouvert));
}

bouton.addEventListener("click", () => ouvrirMenu(menu.hidden));
document.addEventListener("click", (e) => { if (!e.target.closest(".compte")) ouvrirMenu(false); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { ouvrirMenu(false); bouton.focus(); } });

theme.addEventListener("click", () => {
  racine.dataset.theme = estSombre() ? "light" : "dark";
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
    if (choix) racine.dataset.theme = choix;
    else delete racine.dataset.theme;
  }),
);
