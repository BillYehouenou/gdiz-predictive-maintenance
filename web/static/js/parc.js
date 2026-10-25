import { CAUSES, LIBELLES_STATUT, compter, dateHeure, erreur, esc, lire, millions, nombre, pastille, pourcent, squelette } from "./api.js";
import { matrice } from "./charts.js";

const $ = (id) => document.getElementById(id);
const dialogue = $("detail");
let indicateurs = []; // cartes de la période affichée, relues à l'ouverture du détail
const valeursAffichees = {}; // les compteurs repartent de la valeur de la période précédente

const signe = (v, texte) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${texte}`;

// Explication métier de chaque indicateur, avec les chiffres de la période
function details(a) {
  const heuresPanne = a.disponibilite ? (a.heures_marche * (1 - a.disponibilite)) / a.disponibilite : 0;
  return {
    dispo: [
      ["Ce que ça mesure", "La part du temps où les machines devaient tourner et tournaient vraiment. Les délestages SBEE et les entretiens planifiés sont exclus : ils ne disent rien de l'état des machines."],
      ["Calcul", `${nombre(a.heures_marche)} h de marche ÷ (${nombre(a.heures_marche)} h + ${nombre(heuresPanne)} h en panne) = ${pourcent(a.disponibilite, 2)}.`],
      ["Pourquoi ça compte", `Un point de disponibilité représente environ ${nombre((a.heures_marche + heuresPanne) / 100)} h de production sur la période.`],
    ],
    mtbf: [
      ["Ce que ça mesure", "Le temps moyen de marche d'une machine entre deux pannes (Mean Time Between Failures)."],
      ["Calcul", `${nombre(a.heures_marche)} h de marche ÷ ${a.pannes} pannes = ${nombre(a.mtbf_h)} h.`],
      ["Pourquoi ça compte", `Une panne tous les ${nombre((a.mtbf_h ?? 0) / 24)} jours de marche par machine : plus il monte, moins la production s'arrête.`],
    ],
    cout: [
      ["Ce que ça mesure", "Ce que les pannes ont réellement coûté à l'usine sur la période."],
      ["Calcul", `${a.pannes} pannes × ${millions(a.pannes ? a.cout / a.pannes : 0, 2)} FCFA en moyenne. Chaque panne compte les pièces, la production perdue et le technicien pendant la durée réelle d'arrêt, et le tissu rebuté.`],
      ["Pourquoi ça compte", "C'est le montant que le modèle cherche à réduire : une panne annoncée devient un entretien planifié."],
    ],
    arret: [
      ["Ce que ça mesure", "Le temps total de réparation après une panne."],
      ["Calcul", `${nombre(a.heures_arret)} h sur ${a.pannes} pannes : ${nombre(a.pannes ? a.heures_arret / a.pannes : 0, 1)} h de réparation en moyenne (MTTR).`],
      ["Pourquoi ça compte", "Chaque heure d'arrêt est de la production perdue et un technicien mobilisé."],
    ],
  };
}

function indicateur(cle, libelle, couleur, encre, [valeur, num, dec, suffixe], unite, actuel, precedent, plusEstMieux, formatDelta, libellePrec, explication) {
  let delta = "";
  if (precedent != null && actuel != null) {
    const ecart = actuel - precedent;
    const sens = Math.abs(ecart) < 1e-9 ? "" : (ecart > 0) === plusEstMieux ? "bon" : "mauvais";
    const fleche = Math.abs(ecart) < 1e-9 ? "=" : ecart > 0 ? "▲" : "▼";
    delta = `<span class="delta ${sens}">${fleche} ${esc(formatDelta(ecart))}</span> vs ${esc(libellePrec)}`;
  }
  return { cle, libelle, couleur, encre, valeur, num, dec, suffixe, unite, delta, explication };
}

function afficherIndicateurs(t) {
  const a = t.indicateurs, p = t.indicateurs_precedents ?? {};
  const prec = t.periode.libelle_precedente;
  const d = details(a);
  indicateurs = [
    indicateur("dispo", "Disponibilité", "var(--good)", "var(--good-ink)", [nombre(a.disponibilite * 100, 1), a.disponibilite * 100, 1, ""], "%", a.disponibilite, p.disponibilite, true, (e) => signe(e, `${nombre(Math.abs(e) * 100, 2)} pt`), prec, d.dispo),
    indicateur("mtbf", "MTBF moyen", "var(--blue)", "var(--blue)", [nombre(a.mtbf_h), a.mtbf_h, 0, ""], "h de marche", a.mtbf_h, p.mtbf_h, true, (e) => signe(e, `${nombre(Math.abs(e))} h`), prec, d.mtbf),
    indicateur("cout", "Coût des pannes", "var(--orange)", "var(--orange-ink)", [millions(a.cout), a.cout / 1e6, 1, " M"], "FCFA", a.cout, p.cout, false, (e) => signe(e, millions(Math.abs(e))), prec, d.cout),
    indicateur("arret", "Heures d'arrêt subies", "var(--indigo)", "var(--indigo)", [nombre(a.heures_arret), a.heures_arret, 0, ""], "h", a.heures_arret, p.heures_arret, false, (e) => signe(e, `${nombre(Math.abs(e))} h`), prec, d.arret),
  ];
  const compteur = (k) => `<span data-compte="${k.num}" data-depart="${valeursAffichees[k.cle] ?? 0}" data-dec="${k.dec}" data-suffixe="${k.suffixe}">${esc(k.valeur)}</span>`;
  $("kpis").innerHTML = indicateurs.map((k, i) => `<article class="kpi" style="--i: ${i}">
    <h3><span class="point" style="background: ${k.couleur}" aria-hidden="true"></span>${esc(k.libelle)}</h3>
    <p class="kpi-valeur">${compteur(k)}<small> ${esc(k.unite)}</small></p>
    <p class="kpi-pied">${k.delta}</p>
    <button type="button" class="plus" data-kpi="${k.cle}" aria-label="Détail : ${esc(k.libelle)}" aria-haspopup="dialog">
      <span aria-hidden="true"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></span>
    </button>
  </article>`).join("");
  compter($("kpis"));
  indicateurs.forEach((k) => { if (Number.isFinite(k.num)) valeursAffichees[k.cle] = k.num; });
}

// Détail au premier plan : <dialog> modal, fermé par la croix, Échap ou un clic sur le voile
$("kpis").addEventListener("click", (e) => {
  const bouton = e.target.closest(".plus");
  const k = bouton && indicateurs.find((i) => i.cle === bouton.dataset.kpi);
  if (!k) return;
  dialogue.innerHTML = `<button type="button" class="fermer" aria-label="Fermer"><svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
    <div>
      <h2 id="detail-titre" style="color: ${k.encre}"><span class="point" style="width: 10px; height: 10px; background: ${k.couleur}" aria-hidden="true"></span>${esc(k.libelle)}</h2>
      <p class="detail-valeur"><span data-compte="${k.num}" data-dec="${k.dec}" data-suffixe="${k.suffixe}">${esc(k.valeur)}</span><small> ${esc(k.unite)}</small></p>
      ${k.delta ? `<p class="kpi-pied">${k.delta}</p>` : ""}
    </div>
    ${k.explication.map(([titre, texte]) => `<section><h3 style="color: ${k.encre}">${esc(titre)}</h3><p>${esc(texte)}</p></section>`).join("")}`;
  dialogue.querySelector(".fermer").addEventListener("click", () => dialogue.close());
  dialogue.showModal();
  compter(dialogue);
});
dialogue.addEventListener("click", (e) => { if (e.target === dialogue) dialogue.close(); });

function afficherCauses(t) {
  const causes = Object.keys(CAUSES);
  const parGamme = { L: {}, M: {}, H: {} };
  t.pannes_par_gamme.forEach((p) => { parGamme[p.gamme][p.cause] = p.pannes; });
  const max = Math.max(1, ...t.pannes_par_gamme.map((p) => p.pannes));
  const total = (g) => Object.values(parGamme[g]).reduce((a, b) => a + b, 0);
  $("causes-sous-titre").textContent = `${t.periode.libelle} · ${t.indicateurs.pannes} pannes`;
  $("causes-legende").innerHTML = causes.map((c) => `<li><span class="point" style="background: ${CAUSES[c].couleur}"></span>${esc(CAUSES[c].libelle)}</li>`).join("");
  $("causes").innerHTML = `<div class="groupes" role="img" aria-label="Pannes par gamme et par cause : ${["L", "M", "H"].map((g) => `gamme ${g} ${total(g)}`).join(", ")}">${["L", "M", "H"]
    .map((g) => `<div class="groupe"><div class="groupe-tete"><b>Gamme ${g}</b><span class="discret">${total(g)} pannes</span></div><div class="barres">${causes
      .map((c) => {
        const n = parGamme[g][c] ?? 0;
        const h = Math.max(6, Math.round((n / max) * 120));
        return `<div class="barre" style="--h: ${h}px; --i: ${causes.indexOf(c)}" data-bulle="${esc(CAUSES[c].libelle)} · ${n} panne${n > 1 ? "s" : ""}"><span>${n}</span><div style="height: ${h}px; background: ${CAUSES[c].couleur}"></div></div>`;
      })
      .join("")}</div></div>`)
    .join("")}</div>`;
}

// Coût par mois (par semaine sur 3 mois ou moins) : période précédente en gris à gauche de chaque barre
function afficherCout(t) {
  const serie = t.cout_serie, prec = t.cout_serie_precedente.slice(0, serie.length);
  const mensuel = serie.length > 1 && serie.every((p) => p.debut.endsWith("-01"));
  $("cout-total").innerHTML = `<span data-compte="${t.indicateurs.cout / 1e6}" data-dec="1" data-suffixe=" M">${millions(t.indicateurs.cout)}</span> <span style="font-size: 13px; font-weight: 500; color: var(--text-3)">FCFA · ${esc(t.periode.libelle)}</span>`;
  $("cout-legende").innerHTML = `<li><span class="temoin" style="background: var(--orange)"></span>${esc(t.periode.libelle)}</li>` +
    (prec.length ? `<li><span class="temoin" style="background: var(--barre-prec)"></span>${esc(t.periode.libelle_precedente)}</li>` : "");
  if (!serie.length) { $("cout").innerHTML = '<p class="vide">Aucune panne sur la période.</p>'; return; }
  const max = Math.max(1, ...serie.map((p) => p.cout), ...prec.map((p) => p.cout));
  const pic = serie.reduce((iMax, p, i) => (p.cout > serie[iMax].cout ? i : iMax), 0);
  const h = (v) => Math.round((v / max) * 150);
  const etiquette = (iso) => new Date(iso).toLocaleDateString("fr-FR", mensuel ? { month: "short" } : { day: "2-digit", month: "2-digit" });
  const valeurs = serie.length <= 13;
  $("cout").innerHTML = `<div class="cout barres" style="--n: ${serie.length}" role="img" aria-label="Coût des pannes par ${mensuel ? "mois" : "semaine"}, pic à ${millions(serie[pic].cout)} FCFA">${serie
    .map((p, i) => {
      const avant = prec[i];
      const bulle = `${etiquette(p.debut)} · ${millions(p.cout)}${avant ? ` (${millions(avant.cout)} avant)` : ""}`;
      return `<div class="barre${i === pic ? " pic" : ""}" style="--h: ${Math.max(h(p.cout), avant ? h(avant.cout) : 0)}px; --i: ${i}" data-bulle="${esc(bulle)}">
        ${valeurs ? `<span>${nombre(p.cout / 1e6, 1)}</span>` : ""}
        <div class="paire">${avant ? `<div class="prec" style="height: ${h(avant.cout)}px"></div>` : ""}<div style="height: ${h(p.cout)}px"></div></div>
      </div>`;
    })
    .join("")}</div>
    <div class="cout-mois" style="--n: ${serie.length}">${serie.map((p, i) => `<span>${valeurs || i % Math.ceil(serie.length / 12) === 0 ? esc(etiquette(p.debut)) : ""}</span>`).join("")}</div>`;
  compter($("cout-total"));
}

let finDonnees = "";
let enAlerte = null;
const majSousTitre = () => {
  const attention = enAlerte == null ? "" : enAlerte ? `${enAlerte} machine${enAlerte > 1 ? "s demandent" : " demande"} votre attention` : "Aucune machine en alerte";
  $("sous-titre").textContent = [attention, finDonnees && `données au ${dateHeure(finDonnees)}`].filter(Boolean).join(" · ") || "État des 50 machines et coût des pannes.";
};

// Pastille bleue sous la période choisie, qui glisse au changement
const placerCurseur = pastille($("periodes"), '[aria-pressed="true"]');

async function chargerPeriode(code) {
  document.querySelectorAll("#periodes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.periode === code)));
  placerCurseur();
  ["kpis", "causes", "cout"].forEach((id) => { $(id).innerHTML = squelette(id === "kpis" ? 110 : 200); });
  try {
    const t = await lire(`/parc?periode=${code}`);
    afficherIndicateurs(t);
    afficherCauses(t);
    afficherCout(t);
  } catch (e) {
    erreur($("kpis"), e, () => chargerPeriode(code));
    $("causes").innerHTML = $("cout").innerHTML = "";
  }
}

async function chargerEtat() {
  $("plan").innerHTML = squelette(300);
  $("matrice").innerHTML = squelette(240);
  try {
    const etat = await lire("/etat");
    finDonnees = etat.fin_donnees;
    enAlerte = etat.machines.filter((m) => m.statut !== "normal").length;
    majSousTitre();
    $("plan").innerHTML = `<div class="plan">${etat.machines
      .map((m, i) => `<a href="/machines/${esc(m.id_machine)}" class="case statut-${esc(m.statut)}" style="--i: ${i}" aria-label="Machine ${esc(m.numero)}, ${esc(m.nom)}, ${esc(LIBELLES_STATUT[m.statut].toLowerCase())}" title="M${esc(m.numero)} · ${esc(m.nom)} · ${esc(LIBELLES_STATUT[m.statut])}">${esc(m.numero)}</a>`)
      .join("")}</div>
      <ul class="etats" aria-label="Légende des états">${Object.entries(LIBELLES_STATUT)
        .map(([s, l], i) => `<li class="statut-${s}" style="--i: ${i}">${s === "normal" ? "" : `<span class="point" style="width: 7px; height: 7px; background: var(--${{ surveillance: "warn", panne_predite: "orange", en_panne: "danger" }[s]})" aria-hidden="true"></span>`}${l} <b>${etat.comptes[s] ?? 0}</b></li>`)
        .join("")}</ul>`;
    $("matrice").innerHTML = matrice(etat.machines, etat.seuil);
  } catch (e) {
    erreur($("plan"), e, chargerEtat);
    $("matrice").innerHTML = "";
  }
}

let periode = new URLSearchParams(window.location.search).get("periode") ?? "annee";
document.querySelectorAll("#periodes button").forEach((b) =>
  b.addEventListener("click", () => {
    periode = b.dataset.periode;
    history.replaceState(null, "", `?periode=${periode}`);
    chargerPeriode(periode);
  }),
);
chargerPeriode(periode);
chargerEtat();
