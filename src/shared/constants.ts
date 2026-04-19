/**
 * Cerebras Inference adapter — shared constants.
 *
 * Cerebras runs its own silicon (CS-3 wafer-scale) and is the fastest
 * LLM inference backend on Earth — typical throughput is ~1500 tok/sec
 * for frontier open-weights models.
 *
 * v0.7 adds OpenAI-style tool calling and a full model catalog (free +
 * paid/pro tier). The free tier currently exposes
 * `qwen-3-235b-a22b-instruct-2507` only; the pro tier adds Llama 3.3 70B,
 * Llama 4 Scout 17B, and gpt-oss-120b.
 */

export const ADAPTER_TYPE = "cerebras_local";
export const ADAPTER_LABEL = "cerebras_local";
export const PROVIDER_SLUG = "cerebras";
export const BILLER_SLUG = "cerebras";

export const CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1";
export const CEREBRAS_CHAT_COMPLETIONS_URL = `${CEREBRAS_BASE_URL}/chat/completions`;
export const CEREBRAS_MODELS_URL = `${CEREBRAS_BASE_URL}/models`;

export const DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507";
export const DEFAULT_TIMEOUT_SEC = 300;
export const DEFAULT_GRACE_SEC = 10;

/** Hard cap on tool-call loop iterations. */
export const TOOL_CALL_MAX_ITERATIONS = 10;

/** Friendly 429 message surfaced mid-loop when the free tier throttles. */
export const RATE_LIMIT_MESSAGE =
  "Cerebras rate limit hit (429). The free tier throttles aggressively — " +
  "wait ~60s and retry, or upgrade to the pro tier for higher limits.";

export const DEFAULT_PROMPT_TEMPLATE = `{{instructions}}

{{paperclipContext}}

{{taskBody}}`;

/**
 * Free models known to work on Cerebras Inference (as of 2026-04).
 */
export const FREE_MODELS = [
  "qwen-3-235b-a22b-instruct-2507",
] as const;

/**
 * Paid / pro-tier models on Cerebras Inference (as of 2026-04).
 * Keep in sync with https://inference.cerebras.ai/
 */
export const PAID_MODELS = [
  "llama-3.3-70b",
  "llama-4-scout-17b-16e-instruct",
  "gpt-oss-120b",
] as const;

/**
 * Known context windows for model metadata. Fallback = 8192.
 */
export const CONTEXT_WINDOWS: Record<string, number> = {
  "qwen-3-235b-a22b-instruct-2507": 131072,
  "llama-3.3-70b": 128000,
  "llama-4-scout-17b-16e-instruct": 131072,
  "gpt-oss-120b": 131072,
};

export const AUTH_ENV_VAR = "CEREBRAS_API_KEY";
