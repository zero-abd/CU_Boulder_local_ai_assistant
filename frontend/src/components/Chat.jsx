import React, { useEffect, useRef, useState } from 'react';
import samples from '../data/samples';
import { retrieve, tokenize } from '../rag/retrieve';
import { askLive, askLocal, checkLocal, LOCAL_API_URL } from '../api/buffAdvisor';
import KeyPanel from './KeyPanel';
import { PROVIDERS } from '../config';
import buffLogo from '../assets/cu-boulder-logo.svg';

const KEY_STORE = 'buffadvisor.byok';

function loadSaved() {
  try {
    const raw = sessionStorage.getItem(KEY_STORE);
    if (raw) return { config: JSON.parse(raw), remember: true };
  } catch {
    /* storage unavailable */
  }
  const p = PROVIDERS.openai;
  return { config: { provider: 'openai', baseUrl: p.baseUrl, model: p.model, apiKey: '' }, remember: false };
}

// Closest sample question by token overlap (Jaccard).
function matchSample(question) {
  const q = new Set(tokenize(question));
  let best = null;
  let bestScore = 0;
  for (const s of samples) {
    const t = new Set(tokenize(s.q));
    const inter = [...q].filter((x) => t.has(x)).length;
    const score = inter / (q.size + t.size - inter || 1);
    if (score > bestScore) {
      best = s;
      bestScore = score;
    }
  }
  return bestScore >= 0.3 ? best : null;
}

const MODES = [
  { id: 'demo', label: 'Sample answers' },
  { id: 'live', label: 'Live, your key' },
  ...(LOCAL_API_URL ? [{ id: 'local', label: 'On-device backend' }] : []),
];

let nextId = 1;

export default function Chat() {
  const saved = useRef(loadSaved()).current;
  const [mode, setMode] = useState('demo');
  const [config, setConfig] = useState(saved.config);
  const [remember, setRemember] = useState(saved.remember);
  const [style, setStyle] = useState('brief');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [speakReplies, setSpeakReplies] = useState(false);
  const [localReady, setLocalReady] = useState(null);
  const abortRef = useRef(null);
  const recRef = useRef(null);
  const endRef = useRef(null);

  const SpeechRecognition = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    try {
      if (remember) sessionStorage.setItem(KEY_STORE, JSON.stringify(config));
      else sessionStorage.removeItem(KEY_STORE);
    } catch {
      /* storage unavailable */
    }
  }, [config, remember]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages]);

  useEffect(() => {
    if (mode === 'local') checkLocal().then(setLocalReady);
  }, [mode]);

  const patch = (id, fn) => setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...fn(m) } : m)));

  const speak = (text) => {
    if (!speakReplies || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  };

  async function ask(question) {
    question = question.trim();
    if (!question || busy) return;
    setInput('');
    const hits = retrieve(question, 3);
    const notes = hits.map((h) => h.doc);
    const userMsg = { id: nextId++, role: 'user', text: question };
    const botId = nextId++;
    const label =
      mode === 'demo' ? 'Sample answer, not from the model' : mode === 'live' ? `Live: ${config.model}` : 'On-device: DeepSeek-R1-Distill-Llama-8B';
    setMessages((ms) => [
      ...ms,
      userMsg,
      { id: botId, role: 'assistant', text: '', pending: true, label, sources: mode === 'local' ? [] : notes },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const onChunk = (c) => patch(botId, (m) => ({ text: m.text + c, pending: false }));

    try {
      if (mode === 'demo') {
        const s = matchSample(question);
        const text = s
          ? s.a
          : 'Sample mode only has answers for the questions listed below. Switch to "Live, your key" to ask anything, or run BuffAdvisor on your own AMD AI PC.';
        if (!s) patch(botId, () => ({ label: 'Sample mode', sources: [] }));
        const words = text.split(/(\s+)/);
        for (const w of words) {
          if (controller.signal.aborted) break;
          onChunk(w);
          await new Promise((r) => setTimeout(r, 22));
        }
        speak(text);
      } else if (mode === 'live') {
        if (!config.apiKey.trim()) throw new Error('Add your API key above to use live mode.');
        if (!config.baseUrl.trim() || !config.model.trim()) throw new Error('Set a base URL and model.');
        let full = '';
        await askLive({
          provider: config.provider,
          baseUrl: config.baseUrl.trim(),
          apiKey: config.apiKey.trim(),
          model: config.model.trim(),
          question,
          style,
          notes,
          signal: controller.signal,
          onChunk: (c) => {
            full += c;
            onChunk(c);
          },
        });
        speak(full);
      } else {
        let full = '';
        await askLocal({
          question,
          style,
          signal: controller.signal,
          onChunk: (c) => {
            full += c;
            onChunk(c);
          },
        });
        speak(full);
      }
      patch(botId, () => ({ pending: false }));
    } catch (e) {
      if (e.name === 'AbortError') {
        patch(botId, (m) => ({ pending: false, text: m.text + (m.text ? ' ' : '') + '[stopped]' }));
      } else {
        patch(botId, () => ({ role: 'error', pending: false, text: e.message, sources: [] }));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function toggleMic() {
    if (!SpeechRecognition) return;
    if (listening) {
      recRef.current?.stop();
      return;
    }
    const rec = new SpeechRecognition();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.onresult = (e) => ask(e.results[0][0].transcript);
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    recRef.current = rec;
    setListening(true);
    rec.start();
  }

  const banner = {
    demo: (
      <>
        <strong className="text-gray-200">Sample mode.</strong> BuffAdvisor's model runs only on a local AMD AI PC, so
        nothing here is a live on-device run. These answers are samples written from the campus notes for this page,
        not recordings of the model. The retrieval step (which notes match your question) runs live in your browser.
      </>
    ),
    live: (
      <>
        <strong className="text-gray-200">Live mode.</strong> Your question is matched against the campus notes in your
        browser, then answered by the model you choose with your own key. On the AMD AI PC this step runs
        DeepSeek-R1-Distill-Llama-8B locally instead.
      </>
    ),
    local: (
      <>
        <strong className="text-gray-200">On-device backend</strong> at <code>{LOCAL_API_URL}</code>:{' '}
        {localReady === null ? 'checking...' : localReady ? 'ready.' : 'not reachable or still loading the model.'}
      </>
    ),
  }[mode];

  return (
    <section className="flex min-h-[640px] flex-col overflow-hidden rounded-xl border border-gray-700 bg-gray-800/80 shadow-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center gap-2">
          <img src={buffLogo} alt="" className="h-8 w-8" />
          <h2 className="text-lg font-semibold text-white">BuffAdvisor</h2>
        </div>
        <div role="tablist" className="flex rounded-lg bg-gray-900 p-1 text-sm">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="tab"
              aria-selected={mode === m.id}
              onClick={() => !busy && setMode(m.id)}
              className={`rounded-md px-3 py-1.5 transition-colors ${
                mode === m.id ? 'bg-[#CFB87C] font-medium text-gray-900' : 'text-gray-300 hover:text-white'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3 border-b border-gray-700 px-4 py-3">
        <p className="text-xs leading-relaxed text-gray-400">{banner}</p>
        {mode === 'live' && (
          <KeyPanel config={config} setConfig={setConfig} remember={remember} setRemember={setRemember} />
        )}
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4" style={{ maxHeight: '52vh', minHeight: 220 }}>
        {messages.length === 0 && (
          <div className="py-6 text-center text-gray-400">
            <p className="text-base text-gray-200">Ask about CU Boulder: buses, food, health, tutoring, careers.</p>
            <p className="mt-1 text-sm">Pick a question below, type one, or use the mic.</p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-[#CFB87C] text-gray-900'
                  : m.role === 'error'
                    ? 'border border-red-500/50 bg-red-950/60 text-red-200'
                    : 'bg-gray-700 text-gray-100'
              }`}
            >
              {m.role === 'assistant' && (
                <div className="mb-1 text-[11px] uppercase tracking-wide text-[#CFB87C]">{m.label}</div>
              )}
              {m.pending ? <span className="animate-pulse text-gray-400">Retrieving notes, answering...</span> : m.text}
              {m.role === 'assistant' && m.sources?.length > 0 && !m.pending && (
                <div className="mt-2 border-t border-gray-600 pt-2 text-[11px] text-gray-400">
                  Retrieved notes:{' '}
                  {m.sources.map((s, i) => (
                    <span key={s.id}>
                      {i > 0 && ' · '}
                      <a className="underline hover:text-gray-200" href={s.source} target="_blank" rel="noreferrer">
                        {s.title}
                      </a>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="border-t border-gray-700 px-4 py-3">
        <div className="mb-3 flex flex-wrap gap-2">
          {samples.map((s) => (
            <button
              key={s.q}
              disabled={busy}
              onClick={() => ask(s.q)}
              className="rounded-full border border-gray-600 px-3 py-1 text-xs text-gray-300 transition-colors hover:border-[#CFB87C] hover:text-white disabled:opacity-50"
            >
              {s.q}
            </button>
          ))}
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (busy) abortRef.current?.abort();
            else ask(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={listening ? 'Listening...' : 'Ask a question about CU Boulder...'}
            className="min-w-0 flex-1 rounded-lg border border-gray-600 bg-gray-900 px-3 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:border-[#CFB87C] focus:outline-none"
          />
          {SpeechRecognition && (
            <button
              type="button"
              onClick={toggleMic}
              disabled={busy}
              title="Voice input (Web Speech API)"
              aria-label="Voice input"
              className={`rounded-lg border px-3 py-2.5 text-sm ${
                listening ? 'border-red-400 text-red-300' : 'border-gray-600 text-gray-300 hover:text-white'
              }`}
            >
              {listening ? 'Stop' : 'Mic'}
            </button>
          )}
          <button
            type="submit"
            className="rounded-lg bg-[#CFB87C] px-4 py-2.5 text-sm font-medium text-gray-900 hover:bg-[#dccb96]"
          >
            {busy ? 'Stop' : 'Ask'}
          </button>
        </form>
        <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-gray-400">
          {mode !== 'demo' && (
          <label className="flex items-center gap-1.5">
            Style
            <select
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              className="rounded border border-gray-600 bg-gray-900 px-1.5 py-0.5 text-gray-200"
            >
              <option value="brief">Brief</option>
              <option value="balanced">Balanced</option>
              <option value="detailed">Detailed</option>
              <option value="supportive">Supportive</option>
            </select>
          </label>
          )}
          {'speechSynthesis' in window && (
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={speakReplies} onChange={(e) => setSpeakReplies(e.target.checked)} />
              Speak replies
            </label>
          )}
          {messages.length > 0 && (
            <button type="button" className="underline hover:text-gray-200" onClick={() => !busy && setMessages([])}>
              Clear chat
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
