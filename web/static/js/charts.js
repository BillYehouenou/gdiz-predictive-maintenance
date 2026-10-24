// Graphiques en SVG écrit à la main : chaque fonction renvoie une chaîne SVG, couleurs lues dans les tokens CSS.
import { CAUSES, esc, nombre, pourcent } from "./api.js";

const texte = (x, y, contenu, attrs = "") => `<text x="${x}" y="${y}" font-size="11" fill="var(--text-3)" ${attrs}>${esc(contenu)}</text>`;
const ligne = (x1, y1, x2, y2, attrs) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}></line>`;
const pilule = (x, y, contenu, fond, encre, l = 64) =>
  `<rect x="${x - l / 2}" y="${y - 16}" width="${l}" height="22" rx="11" fill="${fond}"></rect>` +
  `<text x="${x}" y="${y - 1}" font-size="11.5" font-weight="600" fill="${encre}" text-anchor="middle">${esc(contenu)}</text>`;

// Courbe lissée sans dépassement (interpolation cubique monotone) : les pics sont arrondis, jamais inventés
export function lisse(pts) {
  const n = pts.length;
  if (n < 2) return n ? `M${pts[0][0]} ${pts[0][1]}` : "";
  const dx = [], m = [];
  for (let i = 0; i < n - 1; i++) { dx.push(pts[i + 1][0] - pts[i][0]); m.push((pts[i + 1][1] - pts[i][1]) / dx[i]); }
  const t = [m[0]];
  for (let i = 1; i < n - 1; i++) t.push(m[i - 1] * m[i] <= 0 ? 0 : (m[i - 1] + m[i]) / 2);
  t.push(m[n - 2]);
  for (let i = 0; i < n - 1; i++) {
    if (m[i] === 0) { t[i] = 0; t[i + 1] = 0; continue; }
    const a = t[i] / m[i], b = t[i + 1] / m[i], s = a * a + b * b;
    if (s > 9) { const k = 3 / Math.sqrt(s); t[i] = k * a * m[i]; t[i + 1] = k * b * m[i]; }
  }
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3;
    d += ` C${(pts[i][0] + h).toFixed(1)} ${(pts[i][1] + t[i] * h).toFixed(1)} ${(pts[i + 1][0] - h).toFixed(1)} ${(pts[i + 1][1] - t[i + 1] * h).toFixed(1)} ${pts[i + 1][0].toFixed(1)} ${pts[i + 1][1].toFixed(1)}`;
  }
  return d;
}

const echelleLog = (v, x0, largeur) => x0 + ((Math.log10(Math.min(Math.max(v, 0.01), 1)) + 2) / 2) * largeur;

// Risque (échelle log) × criticité. Les machines saines sont regroupées par criticité, les alertes nommées.
export function matrice(machines, seuil) {
  const X = (r) => echelleLog(r, 40, 430);
  const Y = (c) => 220 - c * 20;
  const xs = X(seuil), x3 = X(3 * seuil);
  const saines = new Map();
  machines.filter((m) => m.statut === "normal").forEach((m) => {
    const g = saines.get(m.criticite) ?? { n: 0, nom: m.nom };
    saines.set(m.criticite, { ...g, n: g.n + 1 });
  });
  const alertes = machines.filter((m) => m.statut === "surveillance" || m.statut === "panne_predite");
  let svg = `<svg viewBox="0 0 492 262" role="img" aria-label="Matrice risque criticité : ${alertes.map((m) => `M${m.numero} en zone ${m.zone}`).join(", ") || "aucune machine en alerte"}">
    <rect x="${xs}" y="140" width="${x3 - xs - 2}" height="80" rx="10" fill="var(--surface-2)"></rect>
    <rect x="${xs}" y="80" width="${x3 - xs - 2}" height="58" rx="10" fill="var(--warn-weak)"></rect>
    <path d="M${xs + 10} 12 H460 A10 10 0 0 1 470 22 V210 A10 10 0 0 1 460 220 H${x3 + 10} A10 10 0 0 1 ${x3} 210 V88 A10 10 0 0 0 ${x3 - 10} 78 H${xs + 10} A10 10 0 0 1 ${xs} 68 V22 A10 10 0 0 1 ${xs + 10} 12 Z" fill="var(--danger-weak)"></path>
    ${texte(xs + 10, 211, "SURVEILLER", 'font-weight="600" letter-spacing="0.06em" fill="var(--text-2)"')}
    ${texte(xs + 10, 98, "PLANIFIER", 'font-weight="600" letter-spacing="0.06em" fill="var(--warn-ink)"')}
    ${texte(xs + 10, 30, "AGIR", 'font-weight="600" letter-spacing="0.06em" fill="var(--danger)"')}`;
  [0, 4, 7, 10].forEach((c) => {
    svg += ligne(40, Y(c), 470, Y(c), `stroke="var(--border-strong)" ${c ? 'stroke-dasharray="2 5"' : ""}`) + texte(34, Y(c) + 4, c, 'text-anchor="end"');
  });
  [[0.01, "1 %"], [0.03, "3 %"], [0.1, "10 %"], [0.3, "30 %"], [1, "100 %"]].forEach(([r, l]) => { svg += texte(X(r), 250, l, `text-anchor="${r === 1 ? "end" : "middle"}"`); });
  saines.forEach((g, c) => {
    svg += `<circle cx="${xs - 18}" cy="${Y(c)}" r="9" fill="var(--surface-2)" stroke="var(--border-strong)"></circle>
      ${texte(xs - 18, Y(c) + 3.5, g.n, 'font-size="9" font-weight="600" fill="var(--text-2)" text-anchor="middle"')}`;
  });
  alertes.forEach((m) => {
    const x = X(m.risque), y = Y(m.criticite);
    if (m.statut === "panne_predite") {
      svg += `<path d="M${x} ${y - 7} L${x + 7.5} ${y + 6} L${x - 7.5} ${y + 6} Z" fill="var(--orange)" stroke="var(--surface)" stroke-width="1.5" stroke-linejoin="round"></path>
        <rect x="${x + 11}" y="${y - 10}" width="40" height="20" rx="10" fill="var(--ink-btn)"></rect>
        <text x="${x + 31}" y="${y + 4}" font-size="11" font-weight="600" font-family="var(--police-mono)" fill="var(--ink-btn-text)" text-anchor="middle">M${esc(m.numero)}</text>`;
    } else {
      svg += `<circle cx="${x}" cy="${y}" r="5.5" fill="var(--warn-weak)" stroke="var(--warn)" stroke-width="2"></circle>
        <text x="${x + 10}" y="${y + 4}" font-size="11" font-family="var(--police-mono)" fill="var(--text)">M${esc(m.numero)}</text>`;
    }
  });
  svg += ligne(xs, 10, xs, 220, 'stroke="var(--text)" stroke-width="1.25" stroke-dasharray="4 3"') + texte(xs, 236, `seuil ${pourcent(seuil)}`, 'font-size="10" fill="var(--text-2)" text-anchor="middle"');
  return svg + "</svg>";
}

// Coût des pannes : période en cours (trait plein, lissé) et précédente (pointillés)
export function courbeCout(serie, precedente) {
  const valeurs = serie.map((p) => p.cout / 1e6);
  const prec = precedente.slice(0, serie.length).map((p) => p.cout / 1e6);
  const max = Math.max(1, ...valeurs, ...prec) * 1.15;
  const pas = max > 8 ? 4 : max > 4 ? 2 : 1;
  const haut = Math.ceil(max / pas) * pas;
  const n = Math.max(serie.length - 1, 1);
  const X = (i) => 50 + (i * 890) / n;
  const Y = (v) => 190 - (v / haut) * 170;
  const mensuel = serie.length > 1 && serie.every((p) => p.debut.endsWith("-01"));
  const etiquette = (iso) => new Date(iso).toLocaleDateString("fr-FR", mensuel ? { month: "short" } : { day: "2-digit", month: "2-digit" });
  const pts = valeurs.map((v, i) => [X(i), Y(v)]);
  const trace = lisse(pts);
  const pic = valeurs.indexOf(Math.max(...valeurs));
  let svg = `<svg viewBox="0 0 960 220" role="img" aria-label="Coût des pannes par ${mensuel ? "mois" : "semaine"} : ${nombre(valeurs.reduce((a, b) => a + b, 0), 1)} millions de FCFA, pic à ${nombre(valeurs[pic] ?? 0, 1)} millions" style="min-width: 640px">`;
  for (let v = 0; v <= haut; v += pas) svg += ligne(50, Y(v), 940, Y(v), `stroke="var(--border${v ? "" : "-strong"})" ${v ? 'stroke-dasharray="2 5"' : ""}`) + texte(40, Y(v) + 4, v, 'text-anchor="end"');
  if (prec.length > 1) svg += `<path d="${lisse(prec.map((v, i) => [X(i), Y(v)]))}" fill="none" stroke="var(--hatch)" stroke-width="1.75" stroke-dasharray="5 4"></path>`;
  if (pts.length > 1) svg += `<path d="${trace} L${X(serie.length - 1)} 190 L50 190 Z" fill="var(--text)" opacity="0.05"></path><path d="${trace}" fill="none" stroke="var(--text)" stroke-width="1.75"></path>`;
  const tous = serie.length <= 13;
  serie.forEach((p, i) => {
    svg += `<circle cx="${X(i)}" cy="${Y(valeurs[i])}" r="${tous ? 4 : 2.5}" fill="var(--text)" stroke="var(--surface)" stroke-width="2"></circle>`;
    if (tous || i % Math.ceil(serie.length / 12) === 0) svg += texte(X(i), 210, etiquette(p.debut), 'text-anchor="middle"');
  });
  if (valeurs.length && valeurs[pic] > 0) {
    svg += ligne(X(pic), Y(valeurs[pic]), X(pic), 190, 'stroke="var(--text-3)" stroke-dasharray="3 3"');
    svg += pilule(X(pic), Math.max(Y(valeurs[pic]) - 12, 28), `${nombre(valeurs[pic], 1)} M`, "var(--yellow)", "#17181b");
  }
  return svg + "</svg>";
}

// Risque de panne par pas de 2 h, échelle log, sans lissage ; délestages en hachures, pannes en losanges
export function courbeRisque(points) {
  const n = Math.max(points.length - 1, 1);
  const X = (i) => (i * 960) / n;
  const Y = (r) => 200 - ((Math.log10(Math.min(Math.max(r, 0.01), 1)) + 2) / 2) * 200;
  let svg = `<svg viewBox="-48 -8 1028 240" role="img" aria-label="Risque de panne sur ${Math.round(points.length / 12)} jours, de ${pourcent(Math.min(...points.map((p) => p.risque)))} à ${pourcent(Math.max(...points.map((p) => p.risque)))}" style="min-width: 720px">`;
  points.forEach((p, i) => { if (p.delestage) svg += `<rect x="${X(i) - 1.5}" y="0" width="3" height="200" fill="var(--hatch)" opacity="0.4"></rect>`; });
  [[0.01, "1 %", 200], [0.1, "10 %", 100], [1, "100 %", 0]].forEach(([, l, y]) => {
    svg += ligne(0, y, 960, y, `stroke="var(--border${y === 200 ? "-strong" : ""})" ${y === 200 ? "" : 'stroke-dasharray="2 5"'}`) + texte(-8, y + 4, l, 'text-anchor="end"');
  });
  const segments = [[]];
  points.forEach((p, i) => (p.delestage ? segments.push([]) : segments[segments.length - 1].push(`${X(i).toFixed(1)},${Y(p.risque).toFixed(1)}`)));
  segments.filter((s) => s.length).forEach((s) => {
    const x0 = s[0].split(",")[0], x1 = s[s.length - 1].split(",")[0];
    svg += `<polygon points="${s.join(" ")} ${x1},200 ${x0},200" fill="var(--orange)" opacity="0.1"></polygon><polyline points="${s.join(" ")}" fill="none" stroke="var(--orange)" stroke-width="1.75" stroke-linejoin="round"></polyline>`;
  });
  points.forEach((p, i) => { if (p.panne) svg += `<path d="M${X(i)} ${Y(p.risque) - 7} l7 7 -7 7 -7 -7 Z" fill="var(--danger)" stroke="var(--surface)" stroke-width="1.5"></path>`; });
  let derniereSemaine = -1;
  points.forEach((p, i) => {
    const d = new Date(p.timestamp);
    const semaine = Math.floor(i / 84);
    if (semaine !== derniereSemaine && d.getHours() < 2) {
      derniereSemaine = semaine;
      svg += texte(X(i), 222, d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" }), 'text-anchor="middle"');
    }
  });
  const fin = points[points.length - 1];
  if (fin) {
    svg += ligne(960, Y(fin.risque), 960, 200, 'stroke="var(--text-3)" stroke-dasharray="3 3"');
    svg += `<circle cx="960" cy="${Y(fin.risque)}" r="5" fill="var(--orange)" stroke="var(--surface)" stroke-width="2"></circle>`;
    svg += pilule(921, Math.max(Y(fin.risque) - 12, 22), pourcent(fin.risque), "var(--yellow)", "#17181b", 62);
  }
  return svg + "</svg>";
}

// Frise des pannes sur 24 mois, avec le risque actuel au bout
export function frise(historique, finIso, risque) {
  const fin = new Date(finIso).getTime();
  const debut = fin - 730 * 864e5;
  const X = (iso) => 20 + ((new Date(iso).getTime() - debut) / (fin - debut)) * 600;
  let svg = `<svg viewBox="0 -40 640 152" role="img" aria-label="Pannes des 24 derniers mois : ${historique.map((e) => `${new Date(e.date).toLocaleDateString("fr-FR")} ${CAUSES[e.cause]?.libelle ?? e.cause}`).join(", ") || "aucune"} ; risque actuel ${pourcent(risque)}" style="min-width: 520px">
    <rect x="320" y="40" width="300" height="40" rx="12" fill="var(--surface-2)"></rect>
    ${texte(470, 96, "12 derniers mois", 'font-size="10" text-anchor="middle"')}
    ${ligne(20, 60, 620, 60, 'stroke="var(--border-strong)" stroke-width="2" stroke-linecap="round"')}
    ${texte(20, 110, new Date(debut).getFullYear() + 1 === new Date(fin).getFullYear() ? new Date(debut).getFullYear() : "")}`;
  let precedent = -Infinity, rang = 0;
  historique.forEach((e) => {
    const x = X(e.date);
    rang = x - precedent < 90 ? rang + 1 : 0;
    precedent = x;
    const y = 40 - rang * 24;
    svg += `<path d="M${x} 52 L${x + 8} 60 L${x} 68 L${x - 8} 60 Z" fill="var(--danger)"></path>`;
    if (rang) svg += ligne(x, y + 4, x, 50, 'stroke="var(--border-strong)"');
    svg += `<text x="${x}" y="${y}" font-size="11" fill="var(--text)" text-anchor="middle">${esc(new Date(e.date).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" }))} · ${esc((CAUSES[e.cause]?.libelle ?? e.cause).toLowerCase())}</text>`;
    svg += texte(x, y - 12, `${nombre(e.duree_h, 1)} h d'arrêt`, 'font-size="10" text-anchor="middle"');
  });
  svg += `<path d="M620 50 L628 66 L612 66 Z" fill="var(--orange)" stroke="var(--surface)" stroke-width="1.5" stroke-linejoin="round"></path>`;
  svg += `<rect x="532" y="16" width="88" height="22" rx="11" fill="var(--ink-btn)"></rect><text x="576" y="31" font-size="11" font-weight="600" fill="var(--ink-btn-text)" text-anchor="middle">auj. · ${esc(pourcent(risque))}</text>`;
  return svg + "</svg>";
}

// Jauge de santé en demi-cercle
export function jauge(sante) {
  const longueur = Math.PI * 76;
  const couleur = sante < 40 ? "var(--danger)" : sante < 70 ? "var(--orange)" : "var(--good)";
  const libelle = sante < 40 ? "critique" : sante < 70 ? "à surveiller" : "bonne";
  return `<svg viewBox="0 0 200 112" role="img" aria-label="Santé ${sante} sur 100, ${libelle}" style="width: 100%; max-width: 230px; height: auto; display: block; margin: 0 auto">
    <path d="M24 100 A76 76 0 0 1 176 100" fill="none" stroke="var(--track)" stroke-width="16" stroke-linecap="round"></path>
    <path d="M24 100 A76 76 0 0 1 176 100" fill="none" stroke="${couleur}" stroke-width="16" stroke-linecap="round" stroke-dasharray="${((sante / 100) * longueur).toFixed(1)} ${longueur.toFixed(1)}"></path>
    <text x="100" y="88" font-size="38" font-weight="500" fill="var(--text)" text-anchor="middle">${sante}</text>
    <text x="100" y="106" font-size="12" fill="var(--text-3)" text-anchor="middle">sur 100 · ${libelle}</text></svg>`;
}
