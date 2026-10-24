import { CAUSES, LIBELLES_STATUT, erreur, esc, lire, millions, nombre, pourcent, squelette } from "./api.js";
import { courbeCout, matrice } from "./charts.js";

const $ = (id) => document.getElementById(id);
const ICONES = {
  activite: "M3 12h4l3-8 4 16 3-8h4",
  horloge: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3.5 2",
  pieces: "M12 4c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 7v5c0 1.7 3.6 3 8 3s8-1.3 8-3V7M4 12v5c0 1.7 3.6 3 8 3s8-1.3 8-3v-5",
  pause: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM10 9v6M14 9v6",
};
const FLECHES = { haut: "M8 11.5V4.8M5.2 7.4 8 4.6l2.8 2.8", bas: "M8 4.5v6.7M5.2 8.6 8 11.4l2.8-2.8", egal: "M5 8h6" };

// Une hausse est bonne pour la disponibilité et le MTBF, mauvaise pour le coût et les heures d'arrêt
function carteKpi(libelle, icone, couleur, valeur, actuel, precedent, plusEstMieux, formatDelta) {
  let delta = "";
  if (precedent != null && actuel != null) {
    const ecart = actuel - precedent;
    const sens = Math.abs(ecart) < 1e-9 ? "egal" : ecart > 0 ? "haut" : "bas";
    const ton = sens === "egal" ? "neutre" : (ecart > 0) === plusEstMieux ? "bon" : "mauvais";
    delta = `<span class="delta ${ton}"><svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="currentColor"></circle><path d="${FLECHES[sens]}" fill="none" stroke="var(--surface)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>${esc(formatDelta(ecart))}</span>`;
  }
  return `<div class="kpi">
    <div class="kpi-tete"><span class="tuile${couleur === "var(--yellow)" ? " sombre" : ""}" style="background: ${couleur}" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${ICONES[icone]}"></path></svg></span><h3>${esc(libelle)}</h3></div>
    <div class="kpi-corps"><span class="kpi-valeur">${valeur}</span><div class="kpi-pied">${delta || "<span></span>"}<span>${delta ? "vs période précédente" : ""}</span></div></div>
  </div>`;
}

const signe = (v, texte) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${texte}`;

function afficherIndicateurs(t) {
  const a = t.indicateurs, p = t.indicateurs_precedents ?? {};
  $("kpis").innerHTML = [
    carteKpi("Disponibilité", "activite", "var(--teal)", pourcent(a.disponibilite), a.disponibilite, p.disponibilite, true, (e) => signe(e, `${nombre(Math.abs(e) * 100, 2)} pt`)),
    carteKpi("MTBF moyen", "horloge", "var(--blue)", `${nombre(a.mtbf_h)} <small>h de marche</small>`, a.mtbf_h, p.mtbf_h, true, (e) => signe(e, `${nombre(Math.abs(e))} h`)),
    carteKpi("Coût des pannes", "pieces", "var(--yellow)", `${millions(a.cout)} <small>FCFA</small>`, a.cout, p.cout, false, (e) => signe(e, `${millions(Math.abs(e))} FCFA`)),
    carteKpi("Heures d'arrêt subies", "pause", "var(--orange)", `${nombre(a.heures_arret)} <small>h</small>`, a.heures_arret, p.heures_arret, false, (e) => signe(e, `${nombre(Math.abs(e))} h`)),
  ].join("");
}

function afficherCauses(t) {
  const causes = Object.keys(CAUSES);
  const parGamme = { L: {}, M: {}, H: {} };
  t.pannes_par_gamme.forEach((p) => { parGamme[p.gamme][p.cause] = p.pannes; });
  const max = Math.max(1, ...t.pannes_par_gamme.map((p) => p.pannes));
  const total = (g) => Object.values(parGamme[g]).reduce((a, b) => a + b, 0);
  const noms = { L: "entrée", M: "milieu", H: "haut" };
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
  $("cout-legende").innerHTML = `<li class="pilule"><span class="temoin" style="width: 8px; height: 8px; border-radius: 50%; background: var(--text)"></span>${esc(t.periode.libelle)} · <b>${millions(t.indicateurs.cout)}</b></li>` +
    (prec ? `<li class="pilule"><span class="temoin" style="width: 8px; height: 8px; border-radius: 50%; border: 1.5px solid var(--hatch)"></span>${esc(t.periode.libelle_precedente)} · ${millions(prec.cout)}</li>` : "");
  $("cout").innerHTML = t.cout_serie.length ? courbeCout(t.cout_serie, t.cout_serie_precedente) : `<p class="vide">Aucune panne sur la période.</p>`;
}

async function chargerPeriode(code) {
  document.querySelectorAll("#periodes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.periode === code)));
  ["kpis", "causes", "cout"].forEach((id) => { $(id).innerHTML = squelette(id === "kpis" ? 130 : 240); });
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
  $("matrice").innerHTML = squelette(260);
  try {
    const etat = await lire("/etat");
    const comptes = etat.comptes;
    $("plan").innerHTML = `<div class="plan">${etat.machines
      .map((m) => `<a href="/machines/${esc(m.id_machine)}" class="case statut-${esc(m.statut)}" aria-label="Machine ${esc(m.numero)}, ${esc(m.nom)}, ${esc(LIBELLES_STATUT[m.statut].toLowerCase())}" title="M${esc(m.numero)} · ${esc(m.nom)} · ${esc(LIBELLES_STATUT[m.statut])}">${esc(m.numero)}</a>`)
      .join("")}</div>
      <ul class="legende" aria-label="Légende des états" style="margin-top: 18px">${Object.entries(LIBELLES_STATUT)
        .map(([s, l]) => `<li><span class="temoin statut-${s}" aria-hidden="true"></span>${l} <b>${comptes[s] ?? 0}</b></li>`)
        .join("")}</ul>`;
    $("matrice").innerHTML = matrice(etat.machines, etat.seuil);
  } catch (e) {
    erreur($("plan"), e, chargerEtat);
    $("matrice").innerHTML = "";
  }
}

const parametres = new URLSearchParams(window.location.search);
let periode = parametres.get("periode") ?? "annee";
document.querySelectorAll("#periodes button").forEach((b) =>
  b.addEventListener("click", () => {
    periode = b.dataset.periode;
    history.replaceState(null, "", `?periode=${periode}`);
    chargerPeriode(periode);
  }),
);
chargerPeriode(periode);
chargerEtat();
