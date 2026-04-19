/**
 * Cerebras Inference adapter — shared constants.
 *
 * Cerebras runs its own silicon (CS-3 wafer-scale) and is the fastest
 * LLM inference backend on Earth — typical throughput is ~1500 tok/sec
 * for frontier open-weights models. The free tier currently exposes
 * a single model: `qwen-3-235b-a22b-instruct-2507`.
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

export const DEFAULT_PROMPT_TEMPLATE = `{{instructions}}

{{paperclipContext}}

{{taskBody}}`;

/**
 * Free models known to work on Cerebras Inference (as of 2026-04).
 * Cerebras only exposes one model on the free tier; add more as they launch.
 */
export const FREE_MODELS = [
  "qwen-3-235b-a22b-instruct-2507",
] as const;

export const AUTH_ENV_VAR = "CEREBRAS_API_KEY";
