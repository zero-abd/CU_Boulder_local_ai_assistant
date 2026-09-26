// API clients for BuffAdvisor.
//
// askLocal: the on-device Flask backend (backend/server.py), which runs
//   DeepSeek-R1-Distill-Llama-8B through ONNX Runtime GenAI and streams
//   sentences over Server-Sent Events. Only used when VITE_API_URL is set
//   or in `npm run dev`.
// askLive: the hosted "bring your own key" mode. For the listed providers the
//   request goes through api/chat.js, a stateless relay to a fixed provider
//   host that never stores or logs the key. A custom base URL is called
//   directly from the browser.

export const LOCAL_API_URL =
  import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:5001/api' : '');

// Same style instructions as LangChain.choose_solution_style in backend/bot.py.
const STYLE_TEXT = {
  brief: 'Provide the shortest possible answers. One sentence is ideal.',
  detailed: 'Provide concise but informative answers in 2-3 sentences. No longer.',
  supportive: 'Be encouraging but extremely brief. Keep answers to 1-2 sentences.',
  balanced: 'Balance information with brevity. Maximum 2 sentences.',
};

// Reads an SSE response body and calls onEvent with each `data:` payload.
async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (line.startsWith('data:')) {
        if (onEvent(line.slice(5).trim()) === false) return;
      }
    }
  }
}

// One POST that streams the answer back. (The hackathon client sent a POST
// and also opened an EventSource GET for the same question, so the model ran
// twice per message.)
export async function askLocal({ question, style, onChunk, signal }) {
  const res = await fetch(`${LOCAL_API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: question, style, new_session: false, streaming: true }),
    signal,
  });
  if (!res.ok) {
    let msg = `Local backend returned ${res.status}`;
    try {
      const j = await res.json();
      msg = j.message || j.error || msg;
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  let error = null;
  await readSSE(res.body, (data) => {
    let evt;
    try {
      evt = JSON.parse(data);
    } catch {
      return true;
    }
    if (evt.status === 'generating' && evt.chunk) onChunk(evt.chunk);
    if (evt.status === 'error') error = new Error(evt.message || 'Generation failed');
    return evt.status !== 'complete' && evt.status !== 'error';
  });
  if (error) throw error;
}

export async function checkLocal() {
  if (!LOCAL_API_URL) return false;
  try {
    const res = await fetch(`${LOCAL_API_URL}/status`, { signal: AbortSignal.timeout(4000) });
    const j = await res.json();
    return Boolean(j.ready);
  } catch {
    return false;
  }
}

// Prompt adapted from LangChain.prompt_template in backend/bot.py.
export function buildMessages({ question, style, notes }) {
  const context = notes.length
    ? notes.map((n, i) => `[${i + 1}] ${n.title}\n${n.text}`).join('\n\n')
    : '(no matching campus notes)';
  const system = [
    'You are BuffAdvisor, an AI assistant for University of Colorado Boulder students.',
    'Provide quick, concise information about CU Boulder. BE EXTREMELY BRIEF. Answer in 1-3 sentences only.',
    `Advisory Style: ${STYLE_TEXT[style] || STYLE_TEXT.balanced}`,
    'Answer only from the Reference Information. If it does not cover the question, say you do not have that in your campus notes and suggest checking colorado.edu.',
    'Plain text only, no markdown.',
  ].join('\n');
  const user = `Student Query: ${question}\n\nReference Information:\n${context}`;
  return [
    { role: 'system', content: system },
    { role: 'user', content: user },
  ];
}

export async function askLive({ provider, baseUrl, apiKey, model, question, style, notes, onChunk, signal }) {
  const messages = buildMessages({ question, style, notes });
  const viaRelay = provider !== 'custom' && !import.meta.env.DEV;
  const url = viaRelay ? '/api/chat' : `${baseUrl.replace(/\/+$/, '')}/chat/completions`;
  const payload = viaRelay
    ? { provider, model, messages }
    : { model, messages, stream: true, temperature: 0.3, max_tokens: 400 };
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new Error(
      viaRelay
        ? 'Network error. Try again.'
        : `No readable response from ${new URL(url, location.href).host}. Check the base URL and key; the provider must allow browser (CORS) requests.`
    );
  }
  if (!res.ok) {
    let detail = '';
    try {
      const j = await res.json();
      detail = j.error?.message || j.message || '';
    } catch {
      /* not JSON */
    }
    const hint = res.status === 401 || res.status === 403 ? ' Check your API key.' : '';
    detail = detail.replace(/\.\s*$/, '');
    throw new Error(`Provider returned ${res.status}${detail ? `: ${detail}` : ''}.${hint}`);
  }

  // DeepSeek-R1-style models can emit <think>...</think>; hide that part.
  let raw = '';
  let shown = '';
  const emit = () => {
    const visible = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/^\s+/, '');
    if (visible.length > shown.length) {
      onChunk(visible.slice(shown.length));
      shown = visible;
    }
  };
  await readSSE(res.body, (data) => {
    if (data === '[DONE]') return false;
    try {
      const j = JSON.parse(data);
      const delta = j.choices?.[0]?.delta?.content;
      if (delta) {
        raw += delta;
        emit();
      }
    } catch {
      /* ignore keep-alive lines */
    }
    return true;
  });
  if (!shown) throw new Error('The model returned an empty answer. Try another model.');
}
