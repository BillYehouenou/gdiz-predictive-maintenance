import { CAUSES, LIBELLES_STATUT, dateHeure, erreur, esc, lire, millions, nombre, pourcent, squelette } from "./api.js";
import { courbeCout, matrice } from "./charts.js";

const $ = (id) => document.getElementById(id);
let kpiOuvert = null; // clé de l'indicateur dont le détail est ouvert, gardée d'une période à l'autre

const signe = (v, texte) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${texte}`;

// Explication métier de chaque indicateur, avec les chiffres de la période
function details(a) {
  const heuresPanne = a.disponibilite ? (a.heures_marche * (1 - a.disponibilite)) / a.disponibilite : 0;
  return {
    dispo: [
      ["Ce que ça mesure", "La part du temps où les machines devaient tourner et tournaient vraiment. Les délestages SBEE et les entretiens planifiés sont exclus : ils ne disent rien de l'état des machines."],
      ["Calcul sur la période", `<b>${nombre(a.heures_marche)} h</b> de marche ÷ (${nombre(a.heures_marche)} h + <b>${nombre(heuresPanne)} h</b> en panne) = <b>${pourcent(a.disponibilite, 2)}</b>, tous équipements confondus.`],
      ["Pourquoi ça compte", `Un point de disponibilité représente environ <b>${nombre((a.heures_marche + heuresPanne) / 100)} h</b> de production sur la période. C'est la promesse de la maintenance prédictive : moins de temps perdu en pannes subies.`],
    ],
    mtbf: [
      ["Ce que ça mesure", "Le temps moyen de marche d'une machine entre deux pannes (Mean Time Between Failures)."],
      ["Calcul sur la période", `<b>${nombre(a.heures_marche)} h</b> de marche ÷ <b>${a.pannes}</b> pannes = <b>${nombre(a.mtbf_h)} h</b>.`],
      ["Pourquoi ça compte", `Une machine tombe en panne en moyenne tous les <b>${nombre((a.mtbf_h ?? 0) / 24)} jours</b> de marche. Quand des pannes sont évitées par un entretien au bon moment, le MTBF monte.`],
    ],
    cout: [
      ["Ce que ça mesure", "Ce que les pannes ont réellement coûté à l'usine, en FCFA."],
      ["Calcul sur la période", `<b>${a.pannes}</b> pannes × <b>${millions(a.pannes ? a.cout / a.pannes : 0, 2)}</b> FCFA en moyenne = <b>${millions(a.cout)}</b> FCFA. Chaque panne compte les pièces, la production perdue et le technicien pendant la durée réelle de réparation, et le tissu rebuté.`],
      ["Pourquoi ça compte", "C'est le montant que le modèle cherche à réduire : une panne annoncée à l'avance devient un entretien planifié, plus court et sans rebut. Les pannes des gammes hautes pèsent le plus."],
    ],
    arret: [
      ["Ce que ça mesure", "Le temps total pendant lequel des machines étaient arrêtées pour réparer une panne."],
      ["Calcul sur la période", `<b>${nombre(a.heures_arret)} h</b> sur <b>${a.pannes}</b> pannes, soit <b>${nombre(a.pannes ? a.heures_arret / a.pannes : 0, 1)} h</b> de réparation en moyenne par panne (MTTR).`],
      ["Pourquoi ça compte", "Chaque heure d'arrêt est de la production perdue et un technicien mobilisé. Réduire le nombre de pannes et leur durée libère de la capacité sans acheter de machine."],
    ],
  };
}

function carteKpi(cle, libelle, valeur, actuel, precedent, plusEstMieux, formatDelta, libellePrec, explication) {
  let delta = "<span></span>";
  if (precedent != null && actuel != null) {
    const ecart = actuel - precedent;
    const sens = Math.abs(ecart) < 1e-9 ? "neutre" : (ecart > 0) === plusEstMieux ? "bon" : "mauvais";
    const fleche = Math.abs(ecart) < 1e-9 ? "=" : ecart > 0 ? "▲" : "▼";
    delta = `<span><span class="delta ${sens}">${fleche} ${esc(formatDelta(ecart))}</span> vs ${esc(libellePrec)}</span>`;
  }
  const ouverte = kpiOuvert === cle;
  return `<article class="kpi${ouverte ? " ouverte" : ""}" data-kpi="${cle}">
    <h3>${esc(libelle)}</h3>
    <p class="kpi-valeur">${valeur}</p>
    <div class="kpi-pied">${delta}<button type="button" class="voir" aria-expanded="${ouverte}" aria-controls="detail-${cle}">${ouverte ? "Réduire ↙" : "Voir le détail ↗"}</button></div>
    <div class="kpi-detail" id="detail-${cle}" ${ouverte ? "" : "hidden"}>${explication.map(([t, p]) => `<section><h4>${t}</h4><p>${p}</p></section>`).join("")}</div>
  </article>`;
}

function afficherIndicateurs(t) {
  const a = t.indicateurs, p = t.indicateurs_precedents ?? {};
  const prec = t.periode.libelle_precedente;
  const d = details(a);
  $("kpis").innerHTML = [
    carteKpi("dispo", "Disponibilité", pourcent(a.disponibilite), a.disponibilite, p.disponibilite, true, (e) => signe(e, `${nombre(Math.abs(e) * 100, 2)} pt`), prec, d.dispo),
    carteKpi("mtbf", "MTBF moyen", `${nombre(a.mtbf_h)} <small>h de marche</small>`, a.mtbf_h, p.mtbf_h, true, (e) => signe(e, `${nombre(Math.abs(e))} h`), prec, d.mtbf),
    carteKpi("cout", "Coût des pannes", `${millions(a.cout)} <small>FCFA</small>`, a.cout, p.cout, false, (e) => signe(e, `${millions(Math.abs(e))}`), prec, d.cout),
    carteKpi("arret", "Heures d'arrêt subies", `${nombre(a.heures_arret)} <small>h</small>`, a.heures_arret, p.heures_arret, false, (e) => signe(e, `${nombre(Math.abs(e))} h`), prec, d.arret),
  ].join("");
}

$("kpis").addEventListener("click", (e) => {
  const bouton = e.target.closest(".voir");
  if (!bouton) return;
  const cle = bouton.closest(".kpi").dataset.kpi;
  kpiOuvert = kpiOuvert === cle ? null : cle;
  $("kpis").querySelectorAll(".kpi").forEach((carte) => {
    const ouverte = carte.dataset.kpi === kpiOuvert;
    carte.classList.toggle("ouverte", ouverte);
    carte.querySelector(".kpi-detail").hidden = !ouverte;
    const b = carte.querySelector(".voir");
    b.setAttribute("aria-expanded", String(ouverte));
    b.textContent = ouverte ? "Réduire ↙" : "Voir le détail ↗";
  });
});

function afficherCauses(t) {
  const causes = Object.keys(CAUSES);
  const parGamme = { L: {}, M: {}, H: {} };
  t.pannes_par_gamme.forEach((p) => { parGamme[p.gamme][p.cause] = p.pannes; });
  const max = Math.max(1, ...t.pannes_par_gamme.map((p) => p.pannes));
  const total = (g) => Object.values(parGamme[g]).reduce((a, b) => a + b, 0);
  const noms = { L: "entrée de gamme", M: "milieu de gamme", H: "haut de gamme" };
  $("causes-sous-titre").textContent = `${t.periode.libelle} · ${t.indicateurs.pannes} pannes`;
  $("causes-legende").innerHTML = causes.map((c) => `<li><span class="temoin" style="background: ${CAUSES[c].couleur}"></span>${esc(CAUSES[c].libelle)}</li>`).join("");
  $("causes").innerHTML = `<div class="groupes" role="img" aria-label="Pannes par gamme et par cause : ${["L", "M", "H"].map((g) => `gamme ${g} ${total(g)}`).join(", ")}">${["L", "M", "H"]
    .map((g) => `<div class="groupe"><div class="barres">${causes
      .map((c) => {
        const n = parGamme[g][c] ?? 0;
        return `<div class="barre" title="${esc(CAUSES[c].libelle)} : ${n}"><span>${n}</span><div style="height: ${Math.max(6, Math.round((n / max) * 170))}px; background-color: ${CAUSES[c].couleur}"></div></div>`;
      })
      .join("")}</div><div class="nom"><b>Gamme ${g}</b><span class="discret">${noms[g]} · ${total(g)} pannes</span></div></div>`)
    .join("")}</div>`;
}

function afficherCout(t) {
  $("cout-sous-titre").textContent = `${t.periode.libelle} · ${t.cout_serie.length > 13 ? "par semaine" : "par mois"}, millions de FCFA`;
  const prec = t.indicateurs_precedents;
  $("cout-legende").innerHTML = `<li><span class="temoin" style="border-radius: 50%; background: var(--text)"></span>${esc(t.periode.libelle)} · <b>${millions(t.indicateurs.cout)}</b></li>` +
    (prec ? `<li><span class="temoin" style="border-radius: 50%; border: 1.5px dashed var(--hatch)"></span>${esc(t.periode.libelle_precedente)} · ${millions(prec.cout)}</li>` : "");
  $("cout").innerHTML = t.cout_serie.length ? courbeCout(t.cout_serie, t.cout_serie_precedente) : `<p class="vide">Aucune panne sur la période.</p>`;
}

let finDonnees = "";
let libellePeriode = "";
const majSousTitre = () => {
  $("sous-titre").textContent = ["État des 50 machines et coût des pannes", libellePeriode, finDonnees && `données au ${dateHeure(finDonnees)}`].filter(Boolean).join(" · ");
};

async function chargerPeriode(code) {
  document.querySelectorAll("#periodes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.periode === code)));
  ["kpis", "causes", "cout"].forEach((id) => { $(id).innerHTML = squelette(id === "kpis" ? 130 : 240); });
  try {
    const t = await lire(`/parc?periode=${code}`);
    libellePeriode = t.periode.libelle;
    majSousTitre();
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
  $("matrice").innerHTML = squelette(260);
  try {
    const etat = await lire("/etat");
    finDonnees = etat.fin_donnees;
    majSousTitre();
    $("plan").innerHTML = `<div class="plan">${etat.machines
      .map((m) => `<a href="/machines/${esc(m.id_machine)}" class="case statut-${esc(m.statut)}" aria-label="Machine ${esc(m.numero)}, ${esc(m.nom)}, ${esc(LIBELLES_STATUT[m.statut].toLowerCase())}" title="M${esc(m.numero)} · ${esc(m.nom)} · ${esc(LIBELLES_STATUT[m.statut])}">${esc(m.numero)}</a>`)
      .join("")}</div>
      <ul class="legende" aria-label="Légende des états" style="margin-top: 18px">${Object.entries(LIBELLES_STATUT)
        .map(([s, l]) => `<li><span class="temoin statut-${s}" aria-hidden="true"></span>${l} <b>${etat.comptes[s] ?? 0}</b></li>`)
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
