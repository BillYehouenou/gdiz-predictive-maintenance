// Graphiques en SVG écrit à la main : chaque fonction renvoie une chaîne SVG, couleurs lues dans les tokens CSS.
import { esc, mouvementReduit, pourcent } from "./api.js";

const texte = (x, y, contenu, attrs = "") => `<text x="${x}" y="${y}" font-size="10" fill="var(--text-3)" ${attrs}>${esc(contenu)}</text>`;
const ligne = (x1, y1, x2, y2, attrs) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}></line>`;

// Risque en échelle log de 0,1 % à 100 % : les machines saines restent lisibles, même très bas
const echelleLog = (r, x0, largeur) => x0 + ((Math.log10(Math.min(Math.max(r, 0.001), 1)) + 3) / 3) * largeur;
const GRADUATIONS = [[0.001, "0,1 %"], [0.01, "1 %"], [0.1, "10 %"], [1, "100 %"]];

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
  alertes.forEach((m, i) => {
    const x = X(m.risque), y = Y(m.criticite);
    const gauche = x > 420;
    if (m.statut === "panne_predite") {
      const xl = gauche ? x - 52 : x + 12;
      svg += `<circle class="onde" cx="${x}" cy="${y}" r="7" fill="var(--orange)"></circle>
        <g class="pt" style="--i: ${i}"><circle cx="${x}" cy="${y}" r="7" fill="var(--orange)" stroke="var(--surface)" stroke-width="2"></circle>
        <rect x="${xl}" y="${y - 10}" width="40" height="20" rx="10" fill="var(--ink-btn)"></rect>
        <text x="${xl + 20}" y="${y + 4}" font-size="10.5" font-weight="600" fill="var(--ink-btn-text)" text-anchor="middle">M${esc(m.numero)}</text></g>`;
    } else {
      svg += `<g class="pt" style="--i: ${i}"><circle cx="${x}" cy="${y}" r="5.5" fill="var(--warn)" stroke="var(--surface)" stroke-width="2"></circle>
        <text x="${gauche ? x - 9 : x + 9}" y="${y + 4}" font-size="10.5" font-weight="600" fill="var(--text)" text-anchor="${gauche ? "end" : "start"}">M${esc(m.numero)}</text></g>`;
    }
  });
  return svg + "</svg>";
}

// Risque de panne au pas de 2 h, sans lissage, échelle log ; délestages en barres grises, pannes constatées en losanges
export function courbeRisque(points) {
  const n = Math.max(points.length - 1, 1);
  const X = (i) => (i * 960) / n;
  const Y = (r) => 200 - ((Math.log10(Math.min(Math.max(r, 0.001), 1)) + 3) / 3) * 200;
  const risques = points.map((p) => p.risque);
  let svg = `<svg viewBox="-44 -10 1004 236" role="img" aria-label="Risque de panne sur ${Math.round(points.length / 12)} jours, de ${pourcent(Math.min(...risques))} à ${pourcent(Math.max(...risques))}, ${pourcent(risques[risques.length - 1])} aujourd'hui" style="min-width: 640px">`;
  GRADUATIONS.forEach(([r, l]) => {
    const y = Y(r);
    svg += ligne(0, y, 960, y, r === 0.001 ? 'stroke="var(--border-strong)"' : 'stroke="var(--border)" stroke-dasharray="2 4"') + texte(-8, y + 4, l, 'font-size="11" text-anchor="end"');
  });
  points.forEach((p, i) => { if (p.delestage) svg += `<rect x="${(X(i) - 1.5).toFixed(1)}" y="0" width="3" height="200" rx="1.5" fill="var(--gray-c)" opacity="0.6"></rect>`; });
  // Un rideau découvre la courbe de gauche à droite (absent si les animations sont réduites)
  const rideau = mouvementReduit() ? "" : '<animate attributeName="width" from="0" to="1004" dur="1.6s" calcMode="spline" keySplines="0.2 0.8 0.2 1" fill="freeze"></animate>';
  svg += `<clipPath id="revele"><rect x="-44" y="-10" width="1004" height="236">${rideau}</rect></clipPath><g clip-path="url(#revele)">`;
  const segments = [[]];
  points.forEach((p, i) => (p.delestage ? segments.push([]) : segments[segments.length - 1].push([X(i).toFixed(1), Y(p.risque).toFixed(1)])));
  segments.filter((s) => s.length).forEach((s) => {
    const trace = s.map((pt) => pt.join(",")).join(" ");
    svg += `<polygon points="${trace} ${s[s.length - 1][0]},200 ${s[0][0]},200" fill="var(--danger)" opacity="0.1"></polygon>
      <polyline points="${trace}" fill="none" stroke="var(--danger)" stroke-width="1.75" stroke-linejoin="round"></polyline>`;
  });
  points.forEach((p, i) => { if (p.panne) svg += `<path d="M${X(i)} ${Y(p.risque) - 7} l7 7 -7 7 -7 -7 Z" fill="var(--text)" stroke="var(--surface)" stroke-width="1.5"></path>`; });
  svg += "</g>";
  for (let k = 0; k <= 4; k++) {
    const i = Math.round((k * n) / 4);
    const d = new Date(points[Math.min(i, points.length - 1)].timestamp);
    svg += texte(X(i), 222, d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" }), `font-size="11" text-anchor="${k === 0 ? "start" : k === 4 ? "end" : "middle"}"`);
  }
  const fin = points[points.length - 1];
  if (fin) svg += `<circle class="fin" cx="960" cy="${Y(fin.risque).toFixed(1)}" r="5" fill="var(--danger)" stroke="var(--surface)" stroke-width="2"></circle>`;
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
