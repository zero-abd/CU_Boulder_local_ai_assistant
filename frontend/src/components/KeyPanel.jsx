import React from 'react';

import { PROVIDERS, REPO_URL } from '../config';

const input =
  'w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:border-[#CFB87C] focus:outline-none';

export default function KeyPanel({ config, setConfig, remember, setRemember }) {
  const update = (patch) => setConfig({ ...config, ...patch });
  const pickProvider = (id) => {
    const p = PROVIDERS[id];
    update({ provider: id, baseUrl: p.baseUrl, model: p.model || config.model });
  };

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/70 p-4 text-sm">
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block">
          <span className="mb-1 block text-xs text-gray-400">Provider</span>
          <select className={input} value={config.provider} onChange={(e) => pickProvider(e.target.value)}>
            {Object.entries(PROVIDERS).map(([id, p]) => (
              <option key={id} value={id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-gray-400">Model</span>
          <input
            className={input}
            value={config.model}
            onChange={(e) => update({ model: e.target.value })}
            placeholder="model id"
            spellCheck={false}
          />
        </label>
        {config.provider === 'custom' && (
          <label className="block sm:col-span-2">
            <span className="mb-1 block text-xs text-gray-400">Base URL</span>
            <input
              className={input}
              value={config.baseUrl}
              onChange={(e) => update({ baseUrl: e.target.value })}
              placeholder="https://api.example.com/v1"
              spellCheck={false}
            />
          </label>
        )}
        <label className="block sm:col-span-2">
          <span className="mb-1 block text-xs text-gray-400">API key</span>
          <input
            className={input}
            type="password"
            autoComplete="off"
            value={config.apiKey}
            onChange={(e) => update({ apiKey: e.target.value })}
            placeholder="sk-..."
            spellCheck={false}
          />
        </label>
      </div>
      <label className="mt-3 flex items-center gap-2 text-xs text-gray-400">
        <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
        Keep the key for this browser tab (sessionStorage, cleared when the tab closes)
      </label>
      <p className="mt-3 text-xs leading-relaxed text-gray-400">
        Your key stays in your browser and is sent only with your own requests. Nothing is saved. This project is
        open source, so you can check the code:{' '}
        <a className="text-[#CFB87C] underline" href={REPO_URL} target="_blank" rel="noreferrer">
          {REPO_URL.replace('https://', '')}
        </a>
      </p>
      <p className="mt-2 text-xs leading-relaxed text-gray-500">
        {config.provider === 'custom'
          ? 'A custom base URL is called straight from your browser, so that provider must allow browser (CORS) requests.'
          : `Requests pass through this site's stateless relay (api/chat.js), which only forwards to ${PROVIDERS[config.provider].label} and keeps nothing.`}{' '}
        Live mode swaps the on-device model for your cloud model; retrieval over the campus notes stays the same.
      </p>
    </div>
  );
}
