import React from 'react';
import './App.css';
import Chat from './components/Chat';
import { REPO_URL } from './config';

function Code({ children }) {
  return (
    <pre className="mt-2 overflow-x-auto rounded-md border border-gray-700 bg-gray-950 p-3 text-xs leading-relaxed text-gray-200">
      <code>{children}</code>
    </pre>
  );
}

function Step({ n, title, children }) {
  return (
    <li className="relative pl-9">
      <span className="absolute left-0 top-0 flex h-6 w-6 items-center justify-center rounded-full bg-[#CFB87C] text-xs font-semibold text-gray-900">
        {n}
      </span>
      <h4 className="font-medium text-gray-100">{title}</h4>
      <div className="mt-1 text-sm leading-relaxed text-gray-400">{children}</div>
    </li>
  );
}

const App = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-200">
      <div className="pattern-grid pointer-events-none fixed inset-0 opacity-5" />
      <div className="relative mx-auto max-w-6xl px-4 py-8 sm:py-12">
        <header className="mb-8">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#CFB87C]/40 bg-[#CFB87C]/10 px-3 py-1 text-xs text-[#CFB87C]">
            1st place, AMD AI PC track, HackCU 11 (2025)
          </div>
          <h1 className="text-3xl font-bold text-white sm:text-4xl">BuffAdvisor</h1>
          <p className="mt-3 max-w-3xl text-base leading-relaxed text-gray-300">
            A CU Boulder campus assistant that runs DeepSeek-R1-Distill-Llama-8B entirely on a local AMD AI PC: AMD Quark
            quantization, ONNX Runtime GenAI, local RAG with Ollama embeddings and FAISS, and a React voice UI streaming
            from Flask over Server-Sent Events. No cloud model is involved on the device.
          </p>
          <div className="mt-4 flex flex-wrap gap-3 text-sm">
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-gray-600 px-3 py-1.5 text-gray-200 hover:border-[#CFB87C] hover:text-white"
            >
              Source on GitHub
            </a>
            <a
              href="#run-locally"
              className="rounded-lg border border-gray-600 px-3 py-1.5 text-gray-200 hover:border-[#CFB87C] hover:text-white"
            >
              Run it on your own AMD AI PC
            </a>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          <div className="min-w-0 lg:col-span-3">
            <Chat />
          </div>

          <aside className="min-w-0 space-y-6 lg:col-span-2">
            <section className="rounded-xl border border-gray-700 bg-gray-800/60 p-5">
              <h3 className="text-lg font-semibold text-white">How it works on the device</h3>
              <ol className="mt-3 space-y-2 text-sm leading-relaxed text-gray-400">
                <li>
                  <span className="text-gray-200">1. Retrieve.</span> A CU Boulder info PDF is split into 1,000-character
                  chunks, embedded with <code>nomic-embed-text</code> through Ollama and indexed in FAISS. Each question
                  pulls the closest chunks.
                </li>
                <li>
                  <span className="text-gray-200">2. Generate.</span> DeepSeek-R1-Distill-Llama-8B, quantized with AMD
                  Quark, runs through ONNX Runtime GenAI on the AMD hardware, falling back to CPU if the device can't
                  schedule it.
                </li>
                <li>
                  <span className="text-gray-200">3. Stream.</span> Flask streams sentences to the React UI over SSE;
                  the Web Speech API handles voice in and out.
                </li>
              </ol>
              <p className="mt-3 text-xs leading-relaxed text-gray-500">
                This page can't run an 8B model on a server, so it keeps steps 1 and 3 in your browser (keyword retrieval
                over short campus notes summarized from colorado.edu, since the original PDF isn't in the repo) and
                shows sample answers, or uses your own key for step 2.
              </p>
            </section>

            <section id="run-locally" className="rounded-xl border border-gray-700 bg-gray-800/60 p-5">
              <h3 className="text-lg font-semibold text-white">Run it on your own AMD AI PC</h3>
              <p className="mt-1 text-sm text-gray-400">
                Built for a Windows AMD AI PC. Needs Python 3.8+, Node 18+ and Ollama.
              </p>
              <ol className="mt-4 space-y-5">
                <Step n={1} title="Get the code">
                  <Code>{`git clone ${REPO_URL}.git
cd CU_Boulder_local_ai_assistant`}</Code>
                </Step>
                <Step n={2} title="Quantize the model">
                  Run <code>amd_llm_quantization.ipynb</code> (Colab) to quantize{' '}
                  <code>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</code> with AMD Quark, then export it as an ONNX Runtime
                  GenAI model folder. Point <code>model_path</code> in <code>backend/bot.py</code> at that folder.
                </Step>
                <Step n={3} title="Start the backend">
                  <Code>{`cd backend
pip install flask flask-cors -r requirements.txt
ollama pull nomic-embed-text
# put a CU Boulder info PDF in backend/
python server.py        # http://localhost:5001`}</Code>
                </Step>
                <Step n={4} title="Start the voice UI">
                  <Code>{`cd frontend
npm install
npm run dev             # http://localhost:5173`}</Code>
                  In dev mode the chat gets an extra <span className="text-gray-200">On-device backend</span> tab that
                  streams from <code>localhost:5001</code>. Set <code>VITE_API_URL</code> to point elsewhere.
                </Step>
              </ol>
            </section>
          </aside>
        </div>

        <footer className="mt-10 border-t border-gray-800 pt-6 text-xs text-gray-500">
          Built at HackCU 11 (2025). Campus notes summarize public colorado.edu pages and may be out of date; check the
          linked source. Not affiliated with or endorsed by the University of Colorado.
        </footer>
      </div>
    </div>
  );
};

export default App;
