import { CAUSES, FAMILLES, LIBELLES_STATUT, compter, erreur, esc, lire, nombre, pourcent, squelette } from "./api.js";
import { anneau, courbeRisque } from "./charts.js";

const conteneur = document.getElementById("fiche");
const idMachine = conteneur.dataset.machine;
const choix = document.getElementById("choix-machine");
const JOURS = 60;

const carte = (contenu, classe = "", style = "") => `<section class="carte ${classe}" style="${style}">${contenu}</section>`;
const titre = (texte, droite = "") => `<div class="carte-tete"><h2>${esc(texte)}</h2>${droite}</div>`;
const COULEURS_STATUT = { normal: "var(--gray-c)", surveillance: "var(--warn)", panne_predite: "var(--orange)", en_panne: "var(--danger)" };
const ENCRES_STATUT = { normal: "var(--text-3)", surveillance: "var(--warn-ink)", panne_predite: "var(--orange-ink)", en_panne: "var(--danger-ink)" };

function valeurContribution(c) {
  if (c.valeur == null) return "—";
  if (typeof c.valeur === "string") return esc(c.valeur);
  return `${nombre(c.valeur, Math.abs(c.valeur) >= 100 ? 0 : 2)} ${esc(c.unite)}`;
}

function sante(f) {
  if (f.sante < 40) return { libelle: "Critique", couleur: "var(--danger)", fond: "var(--danger-weak)", encre: "var(--danger-ink)" };
  if (f.sante < 70) return { libelle: "À surveiller", couleur: "var(--orange)", fond: "var(--orange-weak)", encre: "var(--orange-ink)" };
  return { libelle: "Bonne", couleur: "var(--good)", fond: "var(--good-weak)", encre: "var(--good-ink)" };
}

// Frise des 24 derniers mois : dates au-dessus de la ligne, causes en dessous ; deux rangées quand les pannes sont proches
function frise(historique, finIso, statut) {
  const fin = new Date(finIso).getTime(), debut = fin - 730 * 864e5;
  const position = (iso) => Math.min(100, Math.max(0, ((new Date(iso).getTime() - debut) / (fin - debut)) * 100));
  const derniers = [-Infinity, -Infinity];
  const evenements = historique.map((e) => {
    const x = position(e.date);
    const rang = x - derniers[0] >= 12 || x - derniers[0] >= x - derniers[1] ? 0 : 1;
    derniers[rang] = x;
    return { x, rang, date: new Date(e.date).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit", year: "2-digit" }), cause: (CAUSES[e.cause]?.libelle ?? e.cause).toLowerCase(), couleur: CAUSES[e.cause]?.couleur ?? "var(--gray-c)", encre: "var(--text)", encreCause: "var(--text-3)" };
  });
  const deuxRangs = evenements.some((e) => e.rang === 1);
  evenements.push({ x: 100, rang: 0, date: "Aujourd'hui", cause: LIBELLES_STATUT[statut].toLowerCase(), couleur: COULEURS_STATUT[statut], encre: ENCRES_STATUT[statut], encreCause: ENCRES_STATUT[statut] });
  const haut = deuxRangs ? 18 : 0;
  const aria = historique.map((e) => `${new Date(e.date).toLocaleDateString("fr-FR")} ${(CAUSES[e.cause]?.libelle ?? e.cause).toLowerCase()}`).join(", ") || "aucune panne";
  return `<div class="frise" role="img" aria-label="Pannes des 24 derniers mois : ${esc(aria)} ; aujourd'hui ${esc(LIBELLES_STATUT[statut].toLowerCase())}" style="height: ${60 + 2 * haut}px">
    <div class="frise-ligne" style="top: ${27 + haut}px"></div>
    ${evenements.map((e, i) => `<div class="frise-evt${e.x === 100 ? " fin" : e.x < 6 ? " debut" : ""}" style="--i: ${i}; left: ${e.x.toFixed(1)}%; top: ${e.rang ? 0 : haut}px">
      <b style="color: ${e.encre}">${esc(e.date)}</b>
      <i aria-hidden="true" style="background: ${e.couleur}; ${e.rang ? `margin-top: ${6 + haut}px; margin-bottom: ${6 + haut}px` : ""}"></i>
      <span style="color: ${e.encreCause}">${esc(e.cause)}</span>
    </div>`).join("")}
  </div>`;
}

// Plage d'un capteur : vert jusqu'à la limite, rouge au-delà (l'inverse pour un minimum) ; le point est la pire valeur
function capteur(c, i) {
  const minimum = c.sens === "min";
  const bas = c.limite * (minimum ? 0.9 : 0.5), haut = c.limite * (minimum ? 1.15 : 1.25);
  const pos = (v) => Math.min(98, Math.max(2, ((v - bas) / (haut - bas)) * 100)).toFixed(1);
  const l = pos(c.limite);
  const decimales = c.unite === "mm/s" ? 2 : c.valeur > 100 ? 0 : 1;
  const piste = minimum
    ? `linear-gradient(90deg, var(--danger-weak) ${l}%, var(--good-weak) ${l}%)`
    : `linear-gradient(90deg, var(--good-weak) ${l}%, var(--danger-weak) ${l}%)`;
  return `<div class="capteur" style="--i: ${i}">
    <div class="capteur-nom"><b>${esc(c.libelle)}</b><span>${minimum ? "minimum" : "limite"} ${nombre(c.limite, 1)} ${esc(c.unite)}</span></div>
    <div class="plage" role="img" aria-label="${esc(c.libelle)} ${c.valeur == null ? "sans mesure" : `${nombre(c.valeur, decimales)} ${esc(c.unite)}, ${c.hors_limite ? "hors limite" : "dans la limite"}`} (${nombre(c.limite, 1)})" style="background: ${piste}">
      <span class="limite" aria-hidden="true" style="left: ${l}%"></span>
      ${c.valeur == null ? "" : `<span class="valeur" aria-hidden="true" style="left: ${pos(c.valeur)}%; background: ${c.hors_limite ? "var(--danger)" : "var(--good)"}"></span>`}
    </div>
    <span class="capteur-valeur" style="color: ${c.hors_limite ? "var(--danger-ink)" : "var(--text)"}">${c.valeur == null ? "—" : nombre(c.valeur, decimales)}<small> ${esc(c.unite)}</small></span>
  </div>`;
}

function afficher(f) {
  const max = Math.max(...f.contributions.map((c) => c.contribution), 1);
  const familles = [...new Set(f.contributions.map((c) => c.famille).filter(Boolean))];
  const s = sante(f);
  const principal = f.contributions[0];
  document.getElementById("badges").innerHTML = `<span class="badge statut-${esc(f.statut)}">${esc(LIBELLES_STATUT[f.statut])}</span>
    <span class="badge doux">${esc(f.nom)}</span><span class="badge doux">Gamme ${esc(f.gamme)}</span>`;
  conteneur.innerHTML = `
  <div style="display: flex; flex-direction: column; gap: 20px">
    <section class="grille" style="--min: 260px" aria-label="État de la machine M${esc(f.numero)}">
      ${carte(`${anneau(f.sante, s.couleur, s.fond)}
        <div style="display: flex; flex-direction: column; gap: 4px">
          <h2 class="discret" style="font-size: 13px; font-weight: 500">Santé</h2>
          <p class="grand" style="color: ${s.encre}">${s.libelle}</p>
          <p class="discret" style="margin: 0">Score sur 100, déduit du risque de panne.</p>
        </div>`, "", "flex-direction: row; align-items: center; gap: 18px")}
      ${carte(`<h2 class="discret" style="font-size: 13px; font-weight: 500">Risque de panne à 5 jours</h2>
        <span style="font-size: 38px; font-weight: 700; letter-spacing: -0.04em; line-height: 1" data-compte="${f.risque * 100}" data-dec="1" data-suffixe=" %">${pourcent(f.risque)}</span>
        <div class="risque-barre" aria-hidden="true"><div style="background: var(--danger); width: ${Math.max(f.risque * 100, 1).toFixed(1)}%"></div></div>
        <p class="discret" style="margin: 0">Probabilité que la machine tombe en panne dans les 120 prochaines heures.</p>`, "", "gap: 10px")}
      ${f.cause
        ? carte(`<h2 style="font-size: 13px; font-weight: 500">Cause probable</h2>
          <p class="grand">${esc(f.cause)}</p>
          ${principal ? `<p style="margin: 0; font-size: 12px; color: var(--cause-ink)">Premier signal : ${esc(principal.libelle.toLowerCase())} à ${valeurContribution(principal)}.</p>` : ""}`, "cause", "gap: 8px")
        : carte(`<h2 class="discret" style="font-size: 13px; font-weight: 500">Cause probable</h2>
          <p class="grand">Aucune dominante</p>
          <p class="discret" style="margin: 0">La machine est sous le seuil d'alerte.</p>`, "", "gap: 8px")}
    </section>

    <div class="grille">
      ${carte(`${titre("Pourquoi ce risque", '<span class="discret">Contributions SHAP</span>')}
        ${f.contributions.length ? f.contributions.map((c, i) => `<div class="shap" style="--i: ${i}">
          <div class="shap-tete"><span><span style="font-weight: 500">${esc(c.libelle)}</span><span style="color: var(--text-3)"> · ${valeurContribution(c)}</span></span><b style="font-weight: 600">+${nombre(c.contribution, 2)}</b></div>
          <div class="shap-piste" role="img" aria-label="${esc(c.libelle)} : contribution +${nombre(c.contribution, 2)}"><div style="width: ${((c.contribution / max) * 100).toFixed(1)}%; background: ${FAMILLES[c.famille] ?? "var(--gray-c)"}"></div></div>
        </div>`).join("") : '<p class="vide">Aucune variable ne pousse vers la panne.</p>'}
        ${familles.length ? `<ul class="legende trait">${familles.map((fa) => `<li><span class="point" style="background: ${FAMILLES[fa]}"></span>${esc(fa)}</li>`).join("")}</ul>` : ""}`)}
      ${carte(`${titre("Historique", '<span class="discret">24 derniers mois</span>')}
        ${frise(f.historique, f.fin_donnees, f.statut)}
        <div class="chiffres">${[
          [pourcent(f.douze_mois.disponibilite, 2), "disponibilité"],
          [f.douze_mois.pannes, `panne${f.douze_mois.pannes > 1 ? "s" : ""} en 12 mois`],
          [`${nombre(f.douze_mois.heures_arret, 1)} h`, "d'arrêt subi"],
          [`${nombre(f.douze_mois.cout / 1000)} k`, "FCFA de coût"],
        ].map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join("")}</div>`, "", "gap: 18px")}
    </div>

    ${carte(`<div class="carte-tete">
        <div><h2 id="t-capteurs">Capteurs</h2><span class="discret">Pire valeur des 30 derniers jours face à la limite du parc</span></div>
        <span class="puce ${f.pannes_30j ? "mauvais" : "bon"}">${f.pannes_30j} panne${f.pannes_30j > 1 ? "s" : ""} sur 30 jours</span>
      </div>
      <div class="capteurs">${f.capteurs_30j.map((c, i) => capteur(c, i)).join("")}</div>`)}

    ${carte(`${titre("Risque de panne", `<span class="discret">${JOURS} derniers jours</span>`)}
      <div class="defile" id="courbe">${squelette(220)}</div>
      <ul class="legende trait">
        <li><span class="temoin-courbe" aria-hidden="true"></span>Risque au pas de 2 h</li>
        <li><span class="temoin-delestage" aria-hidden="true"></span>Délestage SBEE (pas de mesure)</li>
        <li><span class="temoin-panne" aria-hidden="true">✕</span>Panne constatée</li>
      </ul>`, "", "gap: 12px")}
  </div>`;
  compter(conteneur);
  chargerCourbe();
}

async function chargerCourbe() {
  const zone = document.getElementById("courbe");
  zone.innerHTML = squelette(220);
  try {
    const points = await lire(`/machines/${idMachine}/risque?jours=${JOURS}`);
    zone.innerHTML = points.length ? courbeRisque(points) : '<p class="vide">Pas de relevé sur la période.</p>';
  } catch (e) {
    erreur(zone, e, chargerCourbe);
  }
}

async function charger() {
  conteneur.innerHTML = squelette(160) + '<div style="height: 20px"></div>' + squelette(320);
  try {
    const [fiche, etat] = await Promise.all([lire(`/machines/${idMachine}`), lire("/etat")]);
    const ordre = { panne_predite: 0, en_panne: 1, surveillance: 2, normal: 3 };
    choix.innerHTML = [...etat.machines]
      .sort((a, b) => ordre[a.statut] - ordre[b.statut] || b.risque - a.risque || a.numero.localeCompare(b.numero))
      .map((m) => `<option value="${esc(m.id_machine)}" ${m.id_machine === idMachine ? "selected" : ""}>M${esc(m.numero)}</option>`)
      .join("");
    afficher({ ...fiche, fin_donnees: etat.fin_donnees });
  } catch (e) {
    erreur(conteneur, e, charger);
  }
}

choix.addEventListener("change", () => { window.location.href = `/machines/${choix.value}`; });
charger();
