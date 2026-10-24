import { FAMILLES, LIBELLES_STATUT, erreur, esc, lire, millions, nombre, pourcent, squelette } from "./api.js";
import { courbeRisque, frise, jauge } from "./charts.js";

const conteneur = document.getElementById("fiche");
const idMachine = conteneur.dataset.machine;
const choix = document.getElementById("choix-machine");
let jours = 60;

const carte = (contenu, style = "") => `<section class="carte" style="${style}">${contenu}</section>`;
const titre = (texte, couleur, icone, droite = "") => `<div class="carte-tete"><div class="carte-titre">
  <span class="tuile${couleur === "var(--yellow)" ? " sombre" : ""}" style="background: ${couleur}" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${icone}"></path></svg></span>
  <h2>${esc(texte)}</h2></div>${droite}</div>`;

function valeurContribution(c) {
  if (c.valeur == null) return "—";
  if (typeof c.valeur === "string") return esc(c.valeur);
  return `${nombre(c.valeur, Math.abs(c.valeur) >= 100 ? 0 : 2)} ${esc(c.unite)}`;
}

function afficher(f) {
  const max = Math.max(...f.contributions.map((c) => c.contribution), 1);
  const familles = [...new Set(f.contributions.map((c) => c.famille).filter(Boolean))];
  const fois = f.risque / f.seuil;
  conteneur.innerHTML = `
  <div style="display: flex; flex-direction: column; gap: 16px">
    <section class="grille" style="--min: 260px" aria-label="État de la machine M${esc(f.numero)}">
      ${carte(`<div style="display: flex; align-items: center; gap: 12px">
          <span class="tuile" style="width: 44px; height: 44px; flex-basis: 44px; background: var(--surface-2); border: 1px solid var(--border); color: var(--text)" aria-hidden="true"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h16M6 20V10l4 2.5V10l4 2.5V6h4v14"></path></svg></span>
          <span class="mono" style="font-size: 40px; font-weight: 600; letter-spacing: -0.03em; line-height: 1">M${esc(f.numero)}</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px">
          <span class="badge statut-${esc(f.statut)}" style="align-self: flex-start">${esc(LIBELLES_STATUT[f.statut])}</span>
          <div style="display: flex; flex-wrap: wrap; gap: 6px"><span class="badge doux">${esc(f.nom)}</span><span class="badge doux">Gamme ${esc(f.gamme)}</span></div>
        </div>`, "justify-content: space-between")}
      ${carte(`<div class="carte-tete"><h2 style="font-size: 15px">Santé</h2><span class="mono discret">100 × (1 − risque)⁴</span></div>${jauge(f.sante)}`, "gap: 6px")}
      ${carte(`<h2 style="font-size: 15px">Risque de panne à 5 jours</h2>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px">
          <span style="font-size: 42px; font-weight: 500; letter-spacing: -0.03em; line-height: 1; color: ${f.risque >= f.seuil ? "var(--danger)" : "var(--text)"}">${pourcent(f.risque)}</span>
          ${f.risque >= f.seuil ? `<span class="badge jaune">× ${nombre(fois, fois >= 10 ? 0 : 1)} le seuil</span>` : `<span class="badge doux">sous le seuil de ${pourcent(f.seuil)}</span>`}
        </div>
        <span style="font-size: 13px; color: var(--text-2); padding-top: 12px; border-top: 1px solid var(--border)">Cause probable : <b style="color: var(--text)">${esc((f.cause ?? "aucune dominante").toLowerCase())}</b></span>`, "justify-content: space-between")}
    </section>

    <div class="grille" style="--min: 460px">
      ${carte(`${titre("Pourquoi ce risque", "var(--yellow)", "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14M12 17.2v.1", '<span class="discret">contributions SHAP</span>')}
        ${f.contributions.length ? f.contributions.map((c) => `<div style="display: flex; flex-direction: column; gap: 6px">
          <div style="display: flex; justify-content: space-between; gap: 12px; font-size: 13px"><span style="font-weight: 500">${esc(c.libelle)} <span style="font-weight: 400; color: var(--text-3)">· ${valeurContribution(c)}</span></span><b>+${nombre(c.contribution, 2)}</b></div>
          <div role="img" aria-label="${esc(c.libelle)} : contribution +${nombre(c.contribution, 2)}" style="height: 22px; border-radius: 7px; background: var(--surface-2)"><div style="width: ${((c.contribution / max) * 100).toFixed(1)}%; height: 100%; border-radius: 7px; background-color: ${FAMILLES[c.famille] ?? "var(--gray-c)"}; background-image: repeating-linear-gradient(135deg, rgba(255,255,255,.24) 0 2px, transparent 2px 7px)"></div></div>
        </div>`).join("") : '<p class="vide">Aucune variable ne pousse vers la panne : le risque est au plancher.</p>'}
        <ul class="legende">${familles.map((fa) => `<li><span class="temoin" style="background: ${FAMILLES[fa]}"></span>${esc(fa)}</li>`).join("")}</ul>`)}
      ${carte(`${titre("Historique", "var(--blue)", "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3.5 2", '<span class="discret">24 mois · pannes</span>')}
        <div class="defile">${frise(f.historique, f.fin_donnees, f.risque)}</div>
        <div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px">${[
          [pourcent(f.douze_mois.disponibilite, 2), "disponibilité"],
          [f.douze_mois.pannes, `panne${f.douze_mois.pannes > 1 ? "s" : ""} en 12 mois`],
          [`${nombre(f.douze_mois.heures_arret, 1)} h`, "d'arrêt subi"],
          [`${nombre(f.douze_mois.cout / 1000)} k`, "FCFA de coût"],
        ].map(([v, l]) => `<div style="background: var(--surface-2); border-radius: 14px; padding: 12px 14px; display: flex; flex-direction: column; gap: 2px"><span style="font-size: 19px; font-weight: 500; letter-spacing: -0.02em">${v}</span><span class="discret">${l}</span></div>`).join("")}</div>`)}
    </div>

    <section aria-labelledby="t-capteurs" style="display: flex; flex-direction: column; gap: 12px">
      <div class="carte-tete"><div style="display: flex; align-items: baseline; gap: 12px"><h2 id="t-capteurs" style="font-size: 18px">Capteurs</h2><span class="discret">30 derniers jours</span></div><span class="discret">limite = 99e centile du parc (1er pour la tension)</span></div>
      <div class="kpis" style="grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px">${[
        ...f.capteurs_30j.map((c) => ({ libelle: c.libelle, valeur: nombre(c.valeur, c.unite === "mm/s" ? 2 : c.valeur > 100 ? 0 : 1), unite: c.unite, note: c.hors_limite ? `${c.libelle.includes("min") ? "↓ sous" : "↑ au-dessus de"} la limite (${nombre(c.limite, 1)})` : `limite ${nombre(c.limite, 1)}`, alerte: c.hors_limite })),
        { libelle: "Pannes", valeur: f.pannes_30j, unite: "", note: "sur les 30 derniers jours", alerte: f.pannes_30j > 0 },
      ].map((c) => `<div class="kpi"><div class="kpi-tete" style="font-size: 13px; font-weight: 500; color: var(--text-2)"><span aria-hidden="true" style="width: 8px; height: 8px; border-radius: 50%; background: ${c.alerte ? "var(--danger)" : "var(--good)"}"></span>${esc(c.libelle)}</div>
        <div class="kpi-corps" style="gap: 8px"><span class="kpi-valeur" style="font-size: 26px">${c.valeur}<small> ${esc(c.unite)}</small></span><span style="font-size: 12px; font-weight: 500; color: ${c.alerte ? "var(--danger)" : "var(--text-3)"}">${esc(c.note)}</span></div></div>`).join("")}</div>
    </section>

    ${carte(`${titre("Risque de panne", "var(--orange)", "M3 12h4l3-8 4 16 3-8h4", `<div class="segmente" role="group" aria-label="Fenêtre"><button type="button" data-jours="30" aria-pressed="${jours === 30}">30 j</button><button type="button" data-jours="60" aria-pressed="${jours === 60}">60 j</button></div>`)}
      <div class="defile" id="courbe">${squelette(220)}</div>
      <ul class="legende"><li><span class="temoin" style="background: repeating-linear-gradient(135deg, var(--hatch) 0 1.5px, transparent 1.5px 4px)"></span>Délestage SBEE (pas de mesure)</li><li><span class="temoin" style="width: 9px; height: 9px; border-radius: 0; transform: rotate(45deg); background: var(--danger)"></span>Panne constatée</li></ul>`)}
  </div>`;
  conteneur.querySelectorAll("[data-jours]").forEach((b) => b.addEventListener("click", () => { jours = Number(b.dataset.jours); chargerCourbe(); }));
  chargerCourbe();
}

async function chargerCourbe() {
  const zone = document.getElementById("courbe");
  conteneur.querySelectorAll("[data-jours]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.jours) === jours)));
  zone.innerHTML = squelette(220);
  try {
    const points = await lire(`/machines/${idMachine}/risque?jours=${jours}`);
    zone.innerHTML = points.length ? courbeRisque(points) : '<p class="vide">Pas de relevé sur la période.</p>';
  } catch (e) {
    erreur(zone, e, chargerCourbe);
  }
}

async function charger() {
  conteneur.innerHTML = squelette(180) + '<div style="height: 16px"></div>' + squelette(320);
  try {
    const [fiche, etat] = await Promise.all([lire(`/machines/${idMachine}`), lire("/etat")]);
    const ordre = { panne_predite: 0, en_panne: 1, surveillance: 2, normal: 3 };
    choix.innerHTML = [...etat.machines]
      .sort((a, b) => ordre[a.statut] - ordre[b.statut] || b.risque - a.risque || a.numero.localeCompare(b.numero))
      .map((m) => `<option value="${esc(m.id_machine)}" ${m.id_machine === idMachine ? "selected" : ""}>M${esc(m.numero)} · ${esc(LIBELLES_STATUT[m.statut].toLowerCase())}</option>`)
      .join("");
    afficher({ ...fiche, fin_donnees: etat.fin_donnees });
  } catch (e) {
    erreur(conteneur, e, charger);
  }
}

choix.addEventListener("change", () => { window.location.href = `/machines/${choix.value}`; });
charger();
