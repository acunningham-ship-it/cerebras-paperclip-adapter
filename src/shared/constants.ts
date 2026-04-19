/**
 * Cerebras Inference adapter — shared constants.
 */

export const ADAPTER_TYPE = "cerebras_local";
export const ADAPTER_LABEL = "cerebras_local";
export const PROVIDER_SLUG = "cerebras";
export const BILLER_SLUG = "cerebras";

export const CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1";

export const DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507";
export const DEFAULT_TIMEOUT_SEC = 300;
export const DEFAULT_GRACE_SEC = 10;

export const DEFAULT_PROMPT_TEMPLATE = `{{instructions}}

{{paperclipContext}}

{{taskBody}}`;

/**
 * Free models known to work on Cerebras Inference.
 */
export const FREE_MODELS = [
  "qwen-3-235b-a22b-instruct-2507"
] as const;

export const AUTH_ENV_VAR = "CEREBRAS_API_KEY";
