export const REPO_URL = 'https://github.com/zero-abd/CU_Boulder_local_ai_assistant';

export const PROVIDERS = {
  openai: { label: 'OpenAI', baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  openrouter: { label: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o-mini' },
  groq: { label: 'Groq', baseUrl: 'https://api.groq.com/openai/v1', model: 'llama-3.1-8b-instant' },
  custom: { label: 'Other (OpenAI-compatible)', baseUrl: '', model: '' },
};
