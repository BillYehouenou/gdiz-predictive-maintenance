// Graphiques en SVG écrit à la main : chaque fonction renvoie une chaîne SVG, couleurs lues dans les tokens CSS.
import { esc, mouvementReduit, pourcent } from "./commun.js";

const texte = (x, y, contenu, attrs = "") => `<text x="${x}" y="${y}" font-size="10" fill="var(--text-3)" ${attrs}>${esc(contenu)}</text>`;
const ligne = (x1, y1, x2, y2, attrs) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}></line>`;

// Risque en échelle log de 0,1 % à 100 % : les machines saines restent lisibles, même très bas
const echelleLog = (r, x0, largeur) => x0 + ((Math.log10(Math.min(Math.max(r, 0.001), 1)) + 3) / 3) * largeur;
const GRADUATIONS = [[0.001, "0,1 %"], [0.01, "1 %"], [0.1, "10 %"], [1, "100 %"]];

// Étiquettes de la matrice : pour chaque point, la première position autour de lui qui ne touche ni une étiquette déjà
// posée ni un point, dans le cadre ; au-delà des positions voisines, l'étiquette s'éloigne et un trait la relie au point
function placerEtiquettes(ordre, pts) {
  const poses = [];
  const touche = (a, b) => a.g < b.g + b.l + 2 && a.g + a.l + 2 > b.g && a.h < b.h + b.ht + 2 && a.h + a.ht + 2 > b.h;
  const surPoint = (e) => pts.some((q) => q.x + q.r > e.g && q.x - q.r < e.g + e.l && q.y + q.r > e.h && q.y - q.r < e.h + e.ht);
  const dedans = (e) => e.g >= 47 && e.g + e.l <= 470 && e.h >= 10 && e.h + e.ht <= 259;
  ordre.forEach((p) => {
    const l = p.m.statut === "panne_predite" ? 40 : 30, ht = p.m.statut === "panne_predite" ? 20 : 16, d = p.r + 4;
    const voisines = [[d, -ht / 2], [-d - l, -ht / 2], [-l / 2, -d - ht], [-l / 2, d], [d, -d - ht], [d, d], [-d - l, -d - ht], [-d - l, d]];
    const lointaines = [];
    for (let k = 1; k <= 6; k++) {
      const e = d + k * 18;
      lointaines.push([-l / 2, -e - ht], [-l / 2, e], [e, -e - ht / 2], [e, e - ht / 2], [-e - l, -e - ht / 2], [-e - l, e - ht / 2]);
    }
    const candidats = [...voisines.map((c) => [...c, false]), ...lointaines.map((c) => [...c, true])].map(([dx, dy, loin]) => ({ p, g: p.x + dx, h: p.y + dy, l, ht, loin }));
    const libre = (e) => dedans(e) && !surPoint(e) && !poses.some((q) => touche(e, q));
    poses.push(candidats.find(libre) ?? candidats.find((e) => dedans(e) && !poses.some((q) => touche(e, q))) ?? candidats[0]);
  });
  return poses;
}

// Risque × criticité des machines en alerte. Trois rectangles : agir (risque ≥ 3 × seuil), puis planifier (criticité ≥ 4)
// ou surveiller ; risque en échelle log de 1 % à 100 %
export function matrice(machines, seuil) {
  const X = (r) => 44 + ((Math.log10(Math.min(Math.max(r, 0.01), 1)) + 2) / 2) * 426;
  const Y = (c) => 262 - c * 24;
  const x3 = X(3 * seuil), y4 = Y(4);
  const alertes = machines.filter((m) => m.statut === "surveillance" || m.statut === "panne_predite");
  const etiquette = (x, y, mot, encre) => texte(x, y, mot, `font-weight="700" letter-spacing="0.06em" fill="${encre}"`);
  let svg = `<svg viewBox="0 0 480 290" role="img" aria-label="Risque × criticité : ${alertes.map((m) => `M${m.numero} en zone ${m.zone}`).join(", ") || "aucune machine en alerte"}">
    <rect class="zone" style="--i: 0" x="47" y="10" width="${x3 - 49}" height="${y4 - 12}" rx="10" fill="var(--zone-planifier)"></rect>
    <rect class="zone" style="--i: 1" x="47" y="${y4 + 2}" width="${x3 - 49}" height="${259 - y4 - 2}" rx="10" fill="var(--zone-surveiller)"></rect>
    <rect class="zone" style="--i: 2" x="${x3 + 2}" y="10" width="${470 - x3 - 2}" height="249" rx="10" fill="var(--zone-agir)"></rect>
    ${etiquette(57, y4 - 12, "PLANIFIER", "var(--warn-ink)")}
    ${etiquette(57, 249, "SURVEILLER", "var(--text-3)")}
    ${etiquette(x3 + 12, 249, "AGIR", "var(--danger-ink)")}
    ${ligne(44, 6, 44, 262, 'stroke="var(--text-3)" stroke-linecap="round"')}
    ${ligne(44, 262, 474, 262, 'stroke="var(--text-3)" stroke-linecap="round"')}
    ${texte(14, 140, "criticité", 'text-anchor="middle" transform="rotate(-90 14 140)"')}`;
  [0, 4, 10].forEach((c) => { svg += ligne(40, Y(c), 44, Y(c), 'stroke="var(--text-3)"') + texte(36, Y(c) + 4, c, 'text-anchor="end"'); });
  [[0.01, "1 %"], [0.1, "10 %"], [1, "100 %"]].forEach(([r, l], i) => {
    svg += ligne(X(r), 262, X(r), 266, 'stroke="var(--text-3)"') + texte(X(r), 280, l, `text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}"`);
  });
  // Les points d'abord, puis les étiquettes posées sans chevauchement (panne prédite d'abord, puis risque décroissant)
  const pts = alertes.map((m, i) => ({ m, i, x: X(m.risque), y: Y(m.criticite), r: m.statut === "panne_predite" ? 7 : 5.5 }));
  pts.forEach(({ m, i, x, y, r }) => {
    const couleur = m.statut === "panne_predite" ? "var(--orange)" : "var(--warn)";
    if (m.statut === "panne_predite") svg += `<circle class="onde" cx="${x}" cy="${y}" r="7" fill="var(--orange)"></circle>`;
    svg += `<g class="pt" style="--i: ${i}"><circle cx="${x}" cy="${y}" r="${r}" fill="${couleur}" stroke="var(--surface)" stroke-width="2"></circle></g>`;
  });
  const ordre = [...pts].sort((a, b) => (b.m.statut === "panne_predite") - (a.m.statut === "panne_predite") || b.m.risque - a.m.risque);
  placerEtiquettes(ordre, pts).forEach(({ p, g, h, l, ht, loin }) => {
    const pilule = p.m.statut === "panne_predite";
    const cx = g + l / 2, cy = h + ht / 2;
    if (loin) svg += ligne(p.x, p.y, cx, cy, 'stroke="var(--text-3)" stroke-width="1"');
    svg += `<g class="pt" style="--i: ${p.i}">${pilule
      ? `<rect x="${g}" y="${h}" width="${l}" height="${ht}" rx="${ht / 2}" fill="var(--ink-btn)"></rect>
        <text x="${cx}" y="${cy + 4}" font-size="10.5" font-weight="600" fill="var(--ink-btn-text)" text-anchor="middle">M${esc(p.m.numero)}</text>`
      : `<rect x="${g}" y="${h}" width="${l}" height="${ht}" rx="4" fill="var(--surface)" opacity="${loin ? 0.9 : 0}"></rect>
        <text x="${cx}" y="${cy + 4}" font-size="10.5" font-weight="600" fill="var(--text)" text-anchor="middle">M${esc(p.m.numero)}</text>`}</g>`;
  });
  return svg + "</svg>";
}

// Risque de panne au pas de 2 h, sans lissage, échelle log ; pendant l'arrêt du week-end, l'état de la machine est figé
// et la courbe garde le dernier risque. Délestages en barres grises, pannes constatées en croix
export function courbeRisque(brut) {
  let dernier = null; // le week-end prolonge le dernier risque noté
  const points = brut.map((p) => {
    if (p.risque != null) dernier = p.risque;
    return p.risque == null && p.arret && dernier != null ? { ...p, risque: dernier } : p;
  });
  const n = Math.max(points.length - 1, 1);
  const X = (i) => (i * 960) / n;
  const Y = (r) => 200 - ((Math.log10(Math.min(Math.max(r, 0.001), 1)) + 3) / 3) * 200;
  // Pas de risque en panne ou en délestage : la courbe s'interrompt
  const notes = points.map((p, i) => [i, p.risque]).filter(([, r]) => r != null);
  const risques = notes.map(([, r]) => r);
  const derniere = notes[notes.length - 1];
  let svg = `<svg viewBox="-44 -10 1004 236" role="img" aria-label="Risque de panne sur ${Math.round(points.length / 12)} jours, de ${pourcent(Math.min(...risques))} à ${pourcent(Math.max(...risques))}, ${pourcent(derniere?.[1])} au dernier relevé en marche" style="min-width: 640px">`;
  GRADUATIONS.forEach(([r, l]) => {
    const y = Y(r);
    svg += ligne(0, y, 960, y, r === 0.001 ? 'stroke="var(--border-strong)"' : 'stroke="var(--border)" stroke-dasharray="2 4"') + texte(-8, y + 4, l, 'font-size="11" text-anchor="end"');
  });
  points.forEach((p, i) => { if (p.delestage) svg += `<rect x="${(X(i) - 1.5).toFixed(1)}" y="0" width="3" height="200" rx="1.5" fill="var(--gray-c)" opacity="0.6"></rect>`; });
  // Un rideau découvre la courbe de gauche à droite (absent si les animations sont réduites)
  const rideau = mouvementReduit() ? "" : '<animate attributeName="width" from="0" to="1004" dur="1.6s" calcMode="spline" keySplines="0.2 0.8 0.2 1" fill="freeze"></animate>';
  svg += `<clipPath id="revele"><rect x="-44" y="-10" width="1004" height="236">${rideau}</rect></clipPath><g clip-path="url(#revele)">`;
  const segments = [[]];
  points.forEach((p, i) => (p.delestage || p.risque == null ? segments.push([]) : segments[segments.length - 1].push([X(i).toFixed(1), Y(p.risque).toFixed(1)])));
  segments.filter((s) => s.length).forEach((s) => {
    const trace = s.map((pt) => pt.join(",")).join(" ");
    svg += `<polygon points="${trace} ${s[s.length - 1][0]},200 ${s[0][0]},200" fill="var(--danger)" opacity="0.1"></polygon>
      <polyline points="${trace}" fill="none" stroke="var(--danger)" stroke-width="1.75" stroke-linejoin="round"></polyline>`;
  });
  let connu = risques[0] ?? 0.001; // une panne se place au dernier risque connu : la machine ne tourne plus
  points.forEach((p, i) => {
    if (p.risque != null) connu = p.risque;
    if (!p.panne) return;
    const x = X(i), y = Y(connu), croix = `M${x - 5} ${y - 5} L${x + 5} ${y + 5} M${x + 5} ${y - 5} L${x - 5} ${y + 5}`;
    svg += `<path d="${croix}" stroke="var(--surface)" stroke-width="5" stroke-linecap="round"></path><path d="${croix}" stroke="var(--text)" stroke-width="2.25" stroke-linecap="round"></path>`;
  });
  svg += "</g>";
  for (let k = 0; k <= 4; k++) {
    const i = Math.round((k * n) / 4);
    const d = new Date(points[Math.min(i, points.length - 1)].timestamp);
    svg += texte(X(i), 222, d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" }), `font-size="11" text-anchor="${k === 0 ? "start" : k === 4 ? "end" : "middle"}"`);
  }
  // Le point « maintenant » seulement si la machine tourne au dernier relevé ; sinon la croix ou le trou parlent d'eux-mêmes
  if (derniere && derniere[0] === points.length - 1) svg += `<circle class="fin" cx="${X(derniere[0]).toFixed(1)}" cy="${Y(derniere[1]).toFixed(1)}" r="5" fill="var(--danger)" stroke="var(--surface)" stroke-width="2"></circle>`;
  return svg + "</svg>";
}

// Anneau de santé : la part remplie suit le score sur 100
export function anneau(sante, couleur, fond) {
  const tour = 2 * Math.PI * 50;
  return `<svg class="anneau" width="104" height="104" viewBox="0 0 120 120" role="img" aria-label="Santé ${sante} sur 100">
    <circle cx="60" cy="60" r="50" fill="none" stroke="${fond}" stroke-width="14"></circle>
    <circle class="arc" cx="60" cy="60" r="50" fill="none" stroke="${couleur}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${((Math.max(sante, 1) / 100) * tour).toFixed(1)} ${tour.toFixed(1)}" transform="rotate(-90 60 60)"></circle>
    <text x="60" y="68" font-size="26" font-weight="700" fill="var(--text)" text-anchor="middle" data-compte="${sante}">${sante}</text>
  </svg>`;
}
