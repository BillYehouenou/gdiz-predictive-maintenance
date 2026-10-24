// Graphiques en SVG écrit à la main : chaque fonction renvoie une chaîne SVG, couleurs lues dans les tokens CSS.
import { esc, pourcent } from "./api.js";

const texte = (x, y, contenu, attrs = "") => `<text x="${x}" y="${y}" font-size="10" fill="var(--text-3)" ${attrs}>${esc(contenu)}</text>`;
const ligne = (x1, y1, x2, y2, attrs) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}></line>`;

// Risque en échelle log de 0,1 % à 100 % : les machines saines restent lisibles, même très bas
const echelleLog = (r, x0, largeur) => x0 + ((Math.log10(Math.min(Math.max(r, 0.001), 1)) + 3) / 3) * largeur;
const GRADUATIONS = [[0.001, "0,1 %"], [0.01, "1 %"], [0.1, "10 %"], [1, "100 %"]];

// Risque × criticité. Les zones partent des axes ; seules les machines en alerte sont nommées.
export function matrice(machines, seuil) {
  const X = (r) => echelleLog(r, 40, 430);
  const Y = (c) => 220 - c * 20;
  const x3 = X(3 * seuil);
  const alertes = machines.filter((m) => m.statut === "surveillance" || m.statut === "panne_predite");
  const saines = machines.filter((m) => m.statut === "normal");
  let svg = `<svg viewBox="0 0 480 246" role="img" aria-label="Risque × criticité : ${alertes.map((m) => `M${m.numero} en zone ${m.zone}`).join(", ") || "aucune machine en alerte"} ; ${saines.length} machines saines">
    <path d="M50 10 H460 Q470 10 470 20 V210 Q470 220 460 220 H${x3 + 13} Q${x3 + 3} 220 ${x3 + 3} 210 V90 Q${x3 + 3} 80 ${x3 - 7} 80 H50 Q40 80 40 70 V20 Q40 10 50 10 Z" fill="var(--zone-agir)"></path>
    <rect x="40" y="83" width="${x3 - 40}" height="55" rx="10" fill="var(--zone-planifier)"></rect>
    <rect x="40" y="141" width="${x3 - 40}" height="79" rx="10" fill="var(--zone-surveiller)"></rect>
    ${texte(x3 + 13, 211, "AGIR", 'font-weight="700" letter-spacing="0.06em" fill="var(--danger-ink)"')}
    ${texte(x3 - 8, 99, "PLANIFIER", 'font-weight="700" letter-spacing="0.06em" fill="var(--warn-ink)" text-anchor="end"')}
    ${texte(x3 - 8, 211, "SURVEILLER", 'font-weight="700" letter-spacing="0.06em" fill="var(--text-3)" text-anchor="end"')}`;
  [0, 4, 7, 10].forEach((c) => { svg += texte(32, Y(c) + 4, c, 'text-anchor="end"'); });
  GRADUATIONS.forEach(([r, l], i) => { svg += texte(X(r), 240, l, `text-anchor="${i === 0 ? "start" : i === GRADUATIONS.length - 1 ? "end" : "middle"}"`); });
  saines.forEach((m) => {
    svg += `<circle cx="${X(m.risque).toFixed(1)}" cy="${Y(m.criticite)}" r="4" fill="var(--gray-c)" stroke="var(--surface)" stroke-width="1.5"><title>M${esc(m.numero)} · ${esc(pourcent(m.risque))}</title></circle>`;
  });
  alertes.forEach((m) => {
    const x = X(m.risque), y = Y(m.criticite);
    const gauche = x > 420;
    if (m.statut === "panne_predite") {
      const xl = gauche ? x - 52 : x + 12;
      svg += `<circle cx="${x}" cy="${y}" r="7" fill="var(--orange)" stroke="var(--surface)" stroke-width="2"></circle>
        <rect x="${xl}" y="${y - 10}" width="40" height="20" rx="10" fill="var(--ink-btn)"></rect>
        <text x="${xl + 20}" y="${y + 4}" font-size="10.5" font-weight="600" fill="var(--ink-btn-text)" text-anchor="middle">M${esc(m.numero)}</text>`;
    } else {
      svg += `<circle cx="${x}" cy="${y}" r="5.5" fill="var(--warn)" stroke="var(--surface)" stroke-width="2"></circle>
        <text x="${gauche ? x - 9 : x + 9}" y="${y + 4}" font-size="10.5" font-weight="600" fill="var(--text)" text-anchor="${gauche ? "end" : "start"}">M${esc(m.numero)}</text>`;
    }
  });
  return svg + "</svg>";
}

// Risque de panne au pas de 2 h, sans lissage, échelle log ; délestages en barres grises, pannes en losanges
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
  const segments = [[]];
  points.forEach((p, i) => (p.delestage ? segments.push([]) : segments[segments.length - 1].push([X(i).toFixed(1), Y(p.risque).toFixed(1)])));
  segments.filter((s) => s.length).forEach((s) => {
    const trace = s.map((pt) => pt.join(",")).join(" ");
    svg += `<polygon points="${trace} ${s[s.length - 1][0]},200 ${s[0][0]},200" fill="var(--orange)" opacity="0.12"></polygon>
      <polyline points="${trace}" fill="none" stroke="var(--orange)" stroke-width="1.75" stroke-linejoin="round"></polyline>`;
  });
  points.forEach((p, i) => { if (p.panne) svg += `<path d="M${X(i)} ${Y(p.risque) - 7} l7 7 -7 7 -7 -7 Z" fill="var(--danger)" stroke="var(--surface)" stroke-width="1.5"></path>`; });
  for (let k = 0; k <= 4; k++) {
    const i = Math.round((k * n) / 4);
    const d = new Date(points[Math.min(i, points.length - 1)].timestamp);
    svg += texte(X(i), 222, d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" }), `font-size="11" text-anchor="${k === 0 ? "start" : k === 4 ? "end" : "middle"}"`);
  }
  const fin = points[points.length - 1];
  if (fin) svg += `<circle cx="960" cy="${Y(fin.risque).toFixed(1)}" r="5" fill="var(--orange)" stroke="var(--surface)" stroke-width="2"></circle>`;
  return svg + "</svg>";
}

// Anneau de santé : la part remplie suit le score sur 100
export function anneau(sante, couleur, fond) {
  const tour = 2 * Math.PI * 50;
  return `<svg class="anneau" width="104" height="104" viewBox="0 0 120 120" role="img" aria-label="Santé ${sante} sur 100">
    <circle cx="60" cy="60" r="50" fill="none" stroke="${fond}" stroke-width="14"></circle>
    <circle cx="60" cy="60" r="50" fill="none" stroke="${couleur}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${((Math.max(sante, 1) / 100) * tour).toFixed(1)} ${tour.toFixed(1)}" transform="rotate(-90 60 60)"></circle>
    <text x="60" y="68" font-size="26" font-weight="700" fill="var(--text)" text-anchor="middle">${sante}</text>
  </svg>`;
}
