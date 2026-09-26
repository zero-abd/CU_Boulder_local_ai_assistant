# BuffAdvisor frontend

React 19 + Vite 6 + Tailwind 4. Deployed at https://buffadvisor.vercel.app (Vercel project root: `frontend/`).

```bash
npm install
npm run dev      # http://localhost:5173
npm run build
```

## Modes

- **Sample answers**: `src/data/samples.js`, labelled as samples on the page. Retrieval over `src/data/campusDocs.js` runs in the browser (`src/rag/retrieve.js`).
- **Live, your key**: the visitor's own OpenAI-compatible key. Preset providers go through `api/chat.js` (a Vercel function that relays to a fixed host and stores nothing); a custom base URL is called from the browser. Under `npm run dev` presets are called directly, since Vite does not serve `api/`; use `vercel dev` to exercise the relay.
- **On-device backend**: shown when `VITE_API_URL` is set or under `npm run dev` (defaults to `http://localhost:5001/api`). Streams from `backend/server.py` over SSE with a single POST.
