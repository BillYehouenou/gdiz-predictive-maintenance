import { LIBELLES_STATUT, dateCourte, dateHeure, envoyer, erreur, esc, lire, millions, nombre, pourcent, squelette, vide } from "./api.js";

const colonne = document.getElementById("colonne");
const cote = document.getElementById("cote");
const etapes = document.getElementById("etapes");
const CONSTATS = {
  corrige: { titre: "Défaut trouvé et corrigé", sens: "l'alerte était fondée", libelle: "Fondée · défaut corrigé" },
  rien: { titre: "Rien d'anormal", sens: "c'était une fausse alerte", libelle: "Fausse alerte · rien d'anormal" },
  panne: { titre: "Panne avant l'intervention", sens: "fondée, mais traitée trop tard", libelle: "Fondée · panne survenue" },
  ecartee: { libelle: "Fausse alerte · écartée" },
};
let donnees = null;
let ouvert = {}; // formulaire ouvert par intervention : "plan" ou "constat"
let choix = {}; // créneau, technicien et type choisis avant confirmation
let reglage = null;

const point = (couleur) => `<span aria-hidden="true" style="width: 10px; height: 10px; border-radius: 50%; background: ${couleur}"></span>`;
const pastille = (n, fond, encre = "#ffffff") => `<span aria-hidden="true" style="width: 26px; height: 26px; flex: 0 0 26px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: ${encre}; background: ${fond}">${n}</span>`;

function resume(i) {
  const m = i.machine;
  if (!m) return { titre: esc(i.nom ?? ""), detail: "" };
  return {
    titre: `${esc(m.nom)} · criticité ${nombre(m.criticite, m.criticite % 1 ? 1 : 0)} / 10 · risque ${pourcent(m.risque)}`,
    detail: `${m.cause ? `Cause probable : ${esc(m.cause.toLowerCase())} · ` : ""}en alerte depuis le ${dateCourte(i.debut_alerte)}`,
  };
}

function formulairePlan(i) {
  const c = choix[i.id] ?? (choix[i.id] = { creneau: donnees.creneaux[0], technicien: i.technicien_conseille, type: i.type_conseille });
  return `<div style="border-top: 1px solid var(--border); padding-top: 14px; display: flex; flex-direction: column; gap: 14px">
    <div style="display: flex; flex-wrap: wrap; gap: 14px 20px; align-items: flex-end">
      <div style="display: flex; flex-direction: column; gap: 6px; flex: 1 1 200px"><label for="creneau-${i.id}" class="discret" style="font-weight: 500; color: var(--text-2)">Quand</label>
        <select class="champ" id="creneau-${i.id}" data-champ="creneau" data-id="${i.id}">${donnees.creneaux.map((x) => `<option ${x === c.creneau ? "selected" : ""}>${esc(x)}</option>`).join("")}</select></div>
      <div style="display: flex; flex-direction: column; gap: 6px; flex: 1 1 200px"><label for="tech-${i.id}" class="discret" style="font-weight: 500; color: var(--text-2)">Technicien</label>
        <select class="champ" id="tech-${i.id}" data-champ="technicien" data-id="${i.id}">${donnees.techniciens.map((t) => `<option value="${esc(t.id)}" ${t.id === c.technicien ? "selected" : ""}>${esc(t.id)} · ${esc(t.specialite)} · ${esc(t.equipe)}</option>`).join("")}</select></div>
      <div role="group" aria-label="Type d'intervention" style="display: flex; flex-direction: column; gap: 6px"><span class="discret" style="font-weight: 500; color: var(--text-2)">Type</span>
        <div class="segmente">${donnees.types.map((t) => `<button type="button" data-type="${esc(t)}" data-id="${i.id}" aria-pressed="${t === c.type}">${esc(t)}</button>`).join("")}</div></div>
    </div>
    <div style="display: flex; justify-content: flex-end; gap: 8px">
      <button type="button" class="btn btn-texte" data-action="fermer" data-id="${i.id}">Annuler</button>
      <button type="button" class="btn btn-noir" data-action="planifier" data-id="${i.id}">Confirmer l'intervention</button>
    </div></div>`;
}

function formulaireConstat(i) {
  return `<div role="group" aria-label="Constat de l'intervention sur M${esc(i.numero)}" style="border-top: 1px solid var(--border); padding-top: 14px; display: flex; flex-direction: column; gap: 10px">
    <span style="font-size: 13px; font-weight: 500">Qu'a-t-on trouvé sur M${esc(i.numero)} ?</span>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px">${["corrige", "rien", "panne"]
      .map((k) => `<button type="button" class="btn btn-contour" data-action="qualifier" data-constat="${k}" data-id="${i.id}" style="min-height: 64px; border-radius: 14px; flex-direction: column; align-items: flex-start; justify-content: center; text-align: left; gap: 2px"><b style="font-size: 13px">${CONSTATS[k].titre}</b><span class="discret">${CONSTATS[k].sens}</span></button>`)
      .join("")}</div></div>`;
}

function carteAlerte(i) {
  const r = resume(i);
  let actions = "";
  if (i.statut === "a_traiter") {
    actions = `<div style="display: flex; flex-wrap: wrap; gap: 8px"><button type="button" class="btn btn-contour" data-action="ecarter" data-id="${i.id}">Fausse alerte</button>
      <button type="button" class="btn btn-noir" data-action="ouvrir-plan" data-id="${i.id}" aria-expanded="${ouvert[i.id] === "plan"}">Planifier une intervention</button></div>`;
  } else if (i.statut === "planifiee") {
    actions = `<button type="button" class="btn btn-noir" data-action="ouvrir-constat" data-id="${i.id}" aria-expanded="${ouvert[i.id] === "constat"}">Saisir le constat</button>`;
  } else {
    actions = `<div style="display: flex; align-items: center; gap: 8px"><span class="badge ${i.fondee ? "bleu" : "doux"}">${esc(CONSTATS[i.constat]?.libelle ?? i.constat)}</span>
      <button type="button" class="btn btn-texte" data-action="rouvrir" data-id="${i.id}">Modifier</button></div>`;
  }
  const plan = i.technicien && i.statut !== "a_traiter"
    ? `<div class="discret" style="padding: 10px 12px; border-radius: 12px; background: var(--surface-2); color: var(--text-2); font-size: 13px">${esc(i.type ?? "")} planifiée · ${esc(i.creneau ?? "")} · <span class="mono">${esc(i.technicien)}</span></div>` : "";
  return `<article class="carte" style="border-radius: 18px; padding: 16px; gap: 14px">
    <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px">
      <a href="/machines/${esc(i.id_machine)}" class="mono" style="font-size: 20px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); text-decoration: none">M${esc(i.numero)}</a>
      ${i.machine ? `<span class="badge statut-${esc(i.machine.statut)}">${esc(LIBELLES_STATUT[i.machine.statut])}</span>` : ""}
      <div style="flex: 1 1 220px; min-width: 0; display: flex; flex-direction: column"><span style="font-weight: 500">${r.titre}</span><span class="discret">${r.detail}</span></div>
      ${actions}
    </div>
    ${plan}
    ${ouvert[i.id] === "plan" && i.statut === "a_traiter" ? formulairePlan(i) : ""}
    ${ouvert[i.id] === "constat" && i.statut === "planifiee" ? formulaireConstat(i) : ""}
  </article>`;
}

function section(titre, aide, couleur, liste, texteVide) {
  return `<section aria-label="${esc(titre)}" style="display: flex; flex-direction: column; gap: 10px">
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap">${point(couleur)}<h2 style="font-size: 17px; letter-spacing: -0.01em">${esc(titre)}</h2>
      <span class="badge doux">${liste.length}</span><span class="discret">${esc(aide)}</span></div>
    ${liste.length ? liste.map(carteAlerte).join("") : vide(texteVide)}</section>`;
}

function afficherReglage() {
  const points = donnees.reglage;
  const p = points[reglage];
  const positif = p.bilan >= 0;
  document.getElementById("reglage-valeurs").innerHTML = `
    <div style="background: var(--surface); border-radius: 12px; padding: 10px 12px"><b style="font-size: 18px; font-weight: 500">${pourcent(p.part_pannes_annoncees, 0)}</b><div class="discret">pannes annoncées</div></div>
    <div style="border-radius: 12px; padding: 10px 12px; background: ${positif ? "var(--accent-weak)" : "var(--warn-weak)"}; color: ${positif ? "var(--accent-ink)" : "var(--warn-ink)"}"><b style="font-size: 18px">${positif ? "+" : "−"}${millions(Math.abs(p.bilan))}</b><div style="font-size: 12px; color: var(--text-2)">FCFA nets / an</div></div>`;
  document.getElementById("reglage-alertes").textContent = `${nombre(p.alertes_jour, 1)} / jour`;
}

function afficher() {
  const d = donnees;
  const enCours = d.planifiees.length;
  etapes.innerHTML = [
    [1, "À traiter", "alertes du modèle", d.a_traiter.length, "var(--orange)"],
    [2, "Intervention planifiée", "en attente du constat", enCours, "var(--ink-btn)"],
    [3, "Qualifiée", "fondée ou fausse alerte", d.qualifiees.length, "var(--accent)"],
  ].map(([n, t, s, c, f], k) => `<li style="flex: 1 1 200px; display: flex; align-items: center; gap: 8px; min-width: 0">
      <div style="flex: 1; min-width: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 10px 12px; display: flex; align-items: center; gap: 10px">
        ${pastille(n, f, f === "var(--ink-btn)" ? "var(--ink-btn-text)" : "#ffffff")}<div style="display: flex; flex-direction: column; line-height: 1.3"><b style="font-size: 13px">${t}</b><span class="discret">${s}</span></div>
        <span style="margin-left: auto; font-size: 20px; font-weight: 500">${c}</span></div>
      ${k < 2 ? '<svg aria-hidden="true" width="16" height="12" viewBox="0 0 16 12"><path d="M1 6h12M9 2l4 4-4 4" fill="none" stroke="var(--text-3)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>' : ""}</li>`).join("");

  colonne.innerHTML =
    section("À traiter", "planifier une intervention, ou écarter l'alerte", "var(--orange)", d.a_traiter, "Toutes les alertes ont été prises en charge.") +
    section("Interventions planifiées", "saisir le constat une fois sur place", "var(--ink-btn)", d.planifiees, "Aucune intervention planifiée.") +
    section("Qualifiées aujourd'hui", "fondée ou fausse alerte, avec le technicien", "var(--accent)", d.qualifiees, "Aucun constat saisi pour l'instant.") +
    `<section aria-labelledby="t-histo" style="display: flex; flex-direction: column; gap: 10px">
      <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap">${point("var(--text-3)")}<h2 id="t-histo" style="font-size: 17px">Dernières interventions</h2><span class="discret">historique 2025 reconstitué à partir des alertes et des pannes réelles</span></div>
      <div class="carte defile" style="padding: 0; border-radius: 18px"><table style="min-width: 560px"><thead><tr><th scope="col">Alerte</th><th scope="col">Machine</th><th scope="col">Technicien</th><th scope="col">Constat</th></tr></thead><tbody>
      ${d.historique.map((i) => `<tr><td style="color: var(--text-2); white-space: nowrap">${dateHeure(i.debut_alerte)}</td><td><a href="/machines/${esc(i.id_machine)}" class="mono" style="font-weight: 600; color: var(--text); text-decoration: none">M${esc(i.numero)}</a> <span class="discret">${esc(i.nom ?? "")}</span></td><td class="mono discret" style="color: var(--text-2)">${esc(i.technicien ?? "—")}</td><td><span class="badge ${i.fondee ? "bleu" : "doux"}">${esc(CONSTATS[i.constat]?.libelle ?? "")}</span></td></tr>`).join("")}
      </tbody></table></div></section>`;

  const actuel = d.reglage.findIndex((p) => p.actuel);
  if (reglage === null) reglage = actuel;
  const a = d.reglage[actuel];
  const f = d.fiabilite;
  const terrain = d.qualifiees;
  cote.innerHTML = `
    <section class="carte" aria-labelledby="t-equipe" style="gap: 12px">
      <div class="carte-titre"><span class="tuile" style="background: var(--teal)" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="9" cy="8" r="3.2"/><path d="M3.5 19a5.5 5.5 0 0 1 11 0"/><circle cx="17" cy="9" r="2.4"/><path d="M16 14.2a4.5 4.5 0 0 1 5 4.8"/></svg></span>
        <div><h2 id="t-equipe">Équipe maintenance</h2><span class="discret">${d.techniciens.length} techniciens · 3 équipes</span></div></div>
      <ul style="list-style: none; margin: 0; padding: 0">${d.techniciens.map((t) => `<li style="display: grid; grid-template-columns: 70px minmax(0, 1fr) auto; align-items: center; gap: 10px; padding: 9px 0; border-top: 1px solid var(--border)">
        <b class="mono" style="font-size: 12px">${esc(t.id)}</b><span style="font-size: 13px; color: var(--text-2)">${esc(t.specialite)} · ${esc(t.equipe)}</span>
        <span class="badge ${t.interventions_planifiees ? "attente" : "doux"}">${t.interventions_planifiees ? `${t.interventions_planifiees} planifiée${t.interventions_planifiees > 1 ? "s" : ""}` : "disponible"}</span></li>`).join("")}</ul>
    </section>
    <section class="carte" aria-labelledby="t-fiab" style="background: var(--surface-2); box-shadow: none; gap: 14px">
      <div><h2 id="t-fiab" style="font-size: 15px">Peut-on croire une alerte ?</h2><span class="discret">bilan 2025, au réglage actuel</span></div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px">
        <div style="background: var(--surface); border-radius: 14px; padding: 12px 14px"><b style="font-size: 24px; font-weight: 500">${Math.round(a.part_pannes_annoncees * 10)} / 10</b><div style="font-size: 12px; color: var(--text-2)">pannes annoncées à l'avance</div></div>
        <div style="background: var(--surface); border-radius: 14px; padding: 12px 14px"><b style="font-size: 24px; font-weight: 500">1 / ${f.part_fondees ? Math.round(1 / f.part_fondees) : "—"}</b><div style="font-size: 12px; color: var(--text-2)">interventions suivies d'une panne</div></div>
      </div>
      <ul style="list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; font-size: 13px">
        <li style="display: flex; justify-content: space-between; gap: 8px"><span style="color: var(--text-2)">Interventions en 2025</span><b>${nombre(f.alertes_qualifiees)} · dont ${nombre(f.alertes_fondees)} fondées</b></li>
        <li style="display: flex; justify-content: space-between; gap: 8px"><span style="color: var(--text-2)">Préavis médian</span><b>${nombre(a.preavis_median_h, 1)} h</b></li>
      </ul>
      <div style="border-top: 1px solid var(--border-strong); padding-top: 14px; display: flex; flex-direction: column; gap: 10px">
        <label for="reglage" style="display: flex; justify-content: space-between; gap: 8px; font-size: 13px"><span style="font-weight: 500">Réglage du nombre d'alertes</span><b id="reglage-alertes"></b></label>
        <input id="reglage" type="range" min="0" max="${d.reglage.length - 1}" step="1" value="${reglage}" style="width: 100%; min-height: 44px; margin: 0; accent-color: var(--ink-btn)">
        <div style="display: flex; justify-content: space-between" class="discret"><span>plus de pannes vues</span><span>moins d'alertes</span></div>
        <div id="reglage-valeurs" style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px"></div>
        <span class="discret">Simulation sur 2025 : le seuil appliqué se règle à l'entraînement. Fausse alerte = 1 h de technicien ; panne annoncée = 3 h d'arrêt évitées.</span>
      </div>
    </section>
    <section class="carte" aria-labelledby="t-retour" style="gap: 14px">
      <div><h2 id="t-retour" style="font-size: 16px">Retour terrain</h2><span class="discret">constats saisis dans le tableau de bord</span></div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px">
        <div style="background: var(--accent-weak); border-radius: 14px; padding: 12px 14px"><b style="font-size: 26px; font-weight: 500; color: var(--accent-ink)">${terrain.filter((i) => i.fondee).length}</b><div style="font-size: 12px; color: var(--text-2)">alertes fondées</div></div>
        <div style="background: var(--surface-2); border-radius: 14px; padding: 12px 14px"><b style="font-size: 26px; font-weight: 500">${terrain.filter((i) => !i.fondee).length}</b><div style="font-size: 12px; color: var(--text-2)">fausses alertes</div></div>
      </div>
      <p style="margin: 0; font-size: 13px; color: var(--text-2)">Chaque constat est enregistré avec la machine, le technicien et la date. Les constats ne sont pas encore utilisés à l'entraînement : ils mesurent d'abord la fiabilité des alertes sur le terrain.</p>
    </section>`;
  const curseur = document.getElementById("reglage");
  curseur.addEventListener("input", () => { reglage = Number(curseur.value); afficherReglage(); });
  afficherReglage();
}

async function charger() {
  if (!donnees) { colonne.innerHTML = squelette(160) + squelette(160); cote.innerHTML = squelette(300); }
  try {
    donnees = await lire("/alertes");
    afficher();
  } catch (e) {
    erreur(colonne, e, charger);
  }
}

async function agir(action, id, corps) {
  try {
    await envoyer(`/alertes/${id}/${action}`, corps);
    delete ouvert[id];
    delete choix[id];
    await charger();
    window.dispatchEvent(new Event("etat-change"));
  } catch (e) {
    window.alert(`Action impossible : ${e.message}`);
  }
}

colonne.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const id = Number(b.dataset.id);
  if (b.dataset.type) { choix[id].type = b.dataset.type; afficher(); return; }
  switch (b.dataset.action) {
    case "ouvrir-plan": ouvert[id] = ouvert[id] === "plan" ? null : "plan"; afficher(); break;
    case "ouvrir-constat": ouvert[id] = ouvert[id] === "constat" ? null : "constat"; afficher(); break;
    case "fermer": ouvert[id] = null; afficher(); break;
    case "planifier": agir("planifier", id, choix[id]); break;
    case "qualifier": agir("qualifier", id, { constat: b.dataset.constat }); break;
    case "ecarter": agir("qualifier", id, { constat: "ecartee" }); break;
    case "rouvrir": agir("rouvrir", id); break;
  }
});

colonne.addEventListener("change", (e) => {
  const s = e.target.closest("select[data-champ]");
  if (s) choix[Number(s.dataset.id)][s.dataset.champ] = s.value;
});

charger();
