// Commun aux deux pages : bulle d'alertes sur « Parc », thème, export du rapport en PDF.
import { lire } from "./api.js";

async function chargerBulle() {
  try {
    const c = (await lire("/etat")).comptes;
    const n = (c.en_panne ?? 0) + (c.panne_predite ?? 0) + (c.surveillance ?? 0);
    const bulle = document.getElementById("bulle-parc");
    bulle.textContent = n || "";
    bulle.setAttribute("aria-label", `${n} machine${n > 1 ? "s" : ""} en alerte`);
  } catch {}
}

document.getElementById("theme").addEventListener("click", () => {
  const sombre = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = sombre ? "light" : "dark";
  try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch {}
  document.querySelector(".compte").open = false;
});

// L'impression du navigateur produit le PDF ; le nom du fichier vient du titre de la page
document.querySelectorAll("[data-export]").forEach((b) =>
  b.addEventListener("click", () => {
    const { title } = document, theme = document.documentElement.dataset.theme;
    document.title = `Rapport GDIZ ${b.dataset.export} ${new Date().toISOString().slice(0, 10)}`;
    document.documentElement.dataset.theme = "light";
    window.print();
    document.title = title;
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }),
);

chargerBulle();
