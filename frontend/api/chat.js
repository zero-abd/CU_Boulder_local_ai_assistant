// Stateless relay for "bring your own key" live mode.
//
// The browser sends the visitor's key in the Authorization header with each
// request. This function forwards that one request to a fixed provider host
// and streams the answer back. It never logs, stores or caches the key, and it
// only talks to the hosts below (no user-supplied URLs), so it is not an open
// proxy. It exists because some providers (OpenAI) omit CORS headers on error
// responses, which hides "invalid key" errors from the browser.

const PROVIDERS = {
  openai: 'https://api.openai.com/v1/chat/completions',
  openrouter: 'https://openrouter.ai/api/v1/chat/completions',
  groq: 'https://api.groq.com/openai/v1/chat/completions',
};

const MAX_BODY_CHARS = 20000;

function error(status, message) {
  return new Response(JSON.stringify({ error: { message } }), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  });
}

export async function POST(request) {
  const auth = request.headers.get('authorization') || '';
  if (!/^Bearer \S{8,}$/.test(auth)) return error(401, 'Missing API key.');

  let body;
  try {
    const raw = await request.text();
    if (raw.length > MAX_BODY_CHARS) return error(413, 'Request too large.');
    body = JSON.parse(raw);
  } catch {
    return error(400, 'Invalid JSON.');
  }

  const url = PROVIDERS[body.provider];
  if (!url) return error(400, 'Unknown provider.');
  if (typeof body.model !== 'string' || !body.model || body.model.length > 120) return error(400, 'Invalid model.');
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const valid =
    messages.length > 0 &&
    messages.length <= 4 &&
    messages.every((m) => ['system', 'user'].includes(m?.role) && typeof m.content === 'string');
  if (!valid) return error(400, 'Invalid messages.');

  let upstream;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: auth },
      body: JSON.stringify({ model: body.model, messages, stream: true, temperature: 0.3, max_tokens: 400 }),
      signal: request.signal,
    });
  } catch {
    return error(502, 'Could not reach the provider.');
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') || 'application/json',
      'Cache-Control': 'no-store',
    },
  });
}
