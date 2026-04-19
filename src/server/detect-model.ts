/**
 * Model detection for the Cerebras Inference adapter.
 *
 * Cerebras exposes OpenAI-compatible `/v1/models`. We hit that endpoint
 * to verify a given model id is available on the caller's account
 * (free-tier vs paid) and return the canonical provider metadata
 * Paperclip expects.
 */

import {
  AUTH_ENV_VAR,
  CEREBRAS_MODELS_URL,
  DEFAULT_MODEL,
  PROVIDER_SLUG,
} from "../shared/constants.js";

export interface DetectModelResult {
  model: string;
  provider: string;
  source: string;
  candidates?: string[];
}

async function fetchAvailableModels(apiKey: string): Promise<string[] | null> {
  try {
    const resp = await fetch(CEREBRAS_MODELS_URL, {
      headers: {
        accept: "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return null;
    const body = (await resp.json()) as {
      data?: Array<{ id?: string }>;
    };
    if (!body || !Array.isArray(body.data)) return null;
    const ids: string[] = [];
    for (const m of body.data) {
      if (m && typeof m.id === "string" && m.id.length > 0) ids.push(m.id);
    }
    return ids;
  } catch {
    return null;
  }
}

/**
 * Verify the configured model is available on the caller's Cerebras
 * account. If no `CEREBRAS_API_KEY` is present we can't call the API —
 * we still return the compiled-in default so Paperclip has *something*.
 */
export async function detectModel(): Promise<DetectModelResult | null> {
  const apiKey = (process.env[AUTH_ENV_VAR] ?? "").trim();
  if (!apiKey) {
    return {
      model: DEFAULT_MODEL,
      provider: PROVIDER_SLUG,
      source: "default",
    };
  }

  const candidates = await fetchAvailableModels(apiKey);
  if (!candidates || candidates.length === 0) {
    return {
      model: DEFAULT_MODEL,
      provider: PROVIDER_SLUG,
      source: "default",
    };
  }

  const preferred = candidates.includes(DEFAULT_MODEL) ? DEFAULT_MODEL : candidates[0];
  return {
    model: preferred,
    provider: PROVIDER_SLUG,
    source: "api_models_endpoint",
    candidates,
  };
}
