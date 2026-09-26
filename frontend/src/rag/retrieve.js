// Keyword retrieval (BM25) over the campus notes, run in the browser.
// The on-device build uses nomic-embed-text embeddings in FAISS; the hosted
// demo has no embedding model, so it ranks notes by keyword overlap instead.
import campusDocs from '../data/campusDocs.js';

const STOP = new Set(
  'a an and are as at be by can do does for from get go how i if in is it me my of on or so the to what when where which who why will with you your about there their this that have has any campus cu boulder university colorado'.split(' ')
);

function stem(w) {
  if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 5 && w.endsWith('ing')) return w.slice(0, -3);
  if (w.length > 5 && w.endsWith('ed')) return w.slice(0, -2);
  if (w.length > 4 && w.endsWith('es') && !w.endsWith('ses')) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) return w.slice(0, -1);
  return w;
}

export function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter((w) => !STOP.has(w))
    .map(stem);
}

// Hand-written synonyms so everyday wording finds the right note.
const RAW_SYNONYMS = {
  bus: ['transit', 'rtd'],
  transit: ['bus', 'rtd'],
  ride: ['transit', 'rtd'],
  airport: ['skyride'],
  therapy: ['counsel', 'caps'],
  therapist: ['counsel', 'caps'],
  mental: ['counsel', 'caps'],
  stress: ['counsel', 'caps'],
  anxiety: ['counsel', 'caps'],
  depress: ['counsel', 'caps'],
  doctor: ['medical', 'wardenburg'],
  sick: ['medical', 'wardenburg'],
  clinic: ['medical', 'wardenburg'],
  food: ['pantry'],
  hungry: ['pantry', 'food'],
  groceries: ['pantry', 'food'],
  job: ['career', 'handshake'],
  internship: ['career', 'handshake'],
  resume: ['career'],
  tutor: ['tutor', 'asap'],
  homework: ['tutor'],
  gym: ['recreation', 'rec'],
  workout: ['recreation', 'rec'],
  swim: ['pool', 'recreation'],
  mascot: ['ralphie'],
  buffalo: ['ralphie'],
  id: ['onecard'],
  card: ['onecard'],
  study: ['library', 'norlin'],
  book: ['library'],
  computing: ['computer', 'science'],
  cs: ['computer', 'science'],
  car: ['parking', 'permit'],
  drive: ['parking', 'permit'],
};
const SYNONYMS = Object.fromEntries(
  Object.entries(RAW_SYNONYMS).map(([k, v]) => [stem(k), v.map(stem)])
);

const index = campusDocs.map((doc) => {
  const tokens = tokenize(`${doc.title} ${doc.title} ${doc.text}`);
  const tf = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
  return { doc, tf, len: tokens.length };
});
const avgLen = index.reduce((s, d) => s + d.len, 0) / index.length;
const df = new Map();
for (const d of index) for (const t of d.tf.keys()) df.set(t, (df.get(t) || 0) + 1);

function idf(t) {
  const n = df.get(t) || 0;
  return Math.log(1 + (index.length - n + 0.5) / (n + 0.5));
}

export function retrieve(question, k = 3) {
  const base = tokenize(question);
  const terms = new Set(base);
  for (const t of base) for (const s of SYNONYMS[t] || []) terms.add(s);
  const k1 = 1.4;
  const b = 0.75;
  const scored = index.map(({ doc, tf, len }) => {
    let score = 0;
    for (const t of terms) {
      const f = tf.get(t);
      if (!f) continue;
      score += idf(t) * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * len) / avgLen)));
    }
    return { doc, score };
  });
  scored.sort((a, b) => b.score - a.score);
  const top = scored[0]?.score || 0;
  // Drop weak matches: keep notes scoring at least 35% of the best one.
  return scored.filter((s) => s.score > 0 && s.score >= top * 0.35).slice(0, k);
}
