/**
 * Cerebras Inference Paperclip adapter — main entry.
 *
 * Cerebras runs LLMs on its own CS-3 wafer-scale silicon and is the
 * fastest inference backend in the world — typical throughput is
 * ~1500 tokens/second on frontier open-weights models. That raw speed
 * is the whole reason this adapter exists: when latency matters more
 * than model choice, point a Paperclip agent here and watch it rip.
 *
 * Strategy: direct OpenAI-compatible HTTP calls to
 *   https://api.cerebras.ai/v1/chat/completions
 * using `Authorization: Bearer $CEREBRAS_API_KEY`.
 *
 * v0.0.1 is single-turn; see TODOs in `server/execute.ts` for
 * tool-calling, session resume, and retry/backoff work items.
 *
 * @packageDocumentation
 */

import type {
  AdapterConfigSchema,
  AdapterEnvironmentTestContext,
  AdapterEnvironmentTestResult,
  AdapterModel,
  ServerAdapterModule,
} from "@paperclipai/adapter-utils";
import {
  ADAPTER_LABEL,
  ADAPTER_TYPE,
  AUTH_ENV_VAR,
  CEREBRAS_MODELS_URL,
  DEFAULT_MODEL,
  DEFAULT_PROMPT_TEMPLATE,
  DEFAULT_TIMEOUT_SEC,
  FREE_MODELS,
} from "./shared/constants.js";
import {
  detectModel,
  execute,
  sessionCodec,
} from "./server/index.js";

export const type = ADAPTER_TYPE;
export const label = ADAPTER_LABEL;

/**
 * Static fallback model list. Cerebras only ships one model on the
 * free tier right now, but we keep this in an array so paid-tier
 * models can be dropped in without touching the factory.
 */
const STATIC_FALLBACK: AdapterModel[] = FREE_MODELS.map((id) => ({
  id,
  label: `${id} — free (Cerebras ~1500 tok/sec)`,
}));

async function loadModels(): Promise<AdapterModel[]> {
  const apiKey = (process.env[AUTH_ENV_VAR] ?? "").trim();
  if (!apiKey) return STATIC_FALLBACK;
  try {
    const resp = await fetch(CEREBRAS_MODELS_URL, {
      headers: {
        accept: "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return STATIC_FALLBACK;
    const body = (await resp.json()) as { data?: Array<{ id?: string }> };
    if (!body || !Array.isArray(body.data)) return STATIC_FALLBACK;
    const out: AdapterModel[] = [];
    for (const m of body.data) {
      if (!m || typeof m.id !== "string" || !m.id) continue;
      out.push({
        id: m.id,
        label: `${m.id} — Cerebras (~1500 tok/sec)`,
      });
    }
    return out.length > 0 ? out : STATIC_FALLBACK;
  } catch {
    return STATIC_FALLBACK;
  }
}

export const models: AdapterModel[] = await loadModels();

export const agentConfigurationDoc = `# Cerebras Inference Adapter

This adapter routes an agent's single-turn requests to
[Cerebras Inference](https://cerebras.ai/inference), the fastest LLM
inference backend in the world (~1500 tokens/second on frontier
open-weights models running on their CS-3 wafer-scale silicon).

## Prerequisites

- A Cerebras Inference API key from https://cloud.cerebras.ai/

## Core Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | string | \`${DEFAULT_MODEL}\` | Cerebras model id. Free tier currently: \`${DEFAULT_MODEL}\`. |
| temperature | number | 0.7 | Sampling temperature. |
| maxTokens | number | _(unset)_ | Max completion tokens. 0 = provider default. |
| systemPrompt | string | _(none)_ | Optional system prompt prepended to the conversation. |
| timeoutSec | number | ${DEFAULT_TIMEOUT_SEC} | Hard timeout for a single run. |
| promptTemplate | string | _(default)_ | Mustache-style template for the user message. |
| env | object | \`{}\` | Extra env. \`CEREBRAS_API_KEY\` here is preferred over process env. |

## Environment Variables

- \`${AUTH_ENV_VAR}\` — required. Read from \`config.env.${AUTH_ENV_VAR}\` first, then \`process.env.${AUTH_ENV_VAR}\`.

## Notes / Limitations

- **v0.0.1 is single-turn.** Tool calling and session resume are TODO.
- Billing is reported as \`provider=cerebras, biller=cerebras,
  billingType=credits\`. On the free tier \`costUsd=0\`.
`;

const configSchema: AdapterConfigSchema = {
  fields: [
    {
      key: "model",
      label: "Model",
      type: "combobox",
      default: DEFAULT_MODEL,
      required: false,
      options: STATIC_FALLBACK.map((m) => ({ label: m.label, value: m.id })),
      hint: "Cerebras model id. Free tier is limited to qwen-3-235b-a22b-instruct-2507.",
    },
    {
      key: "temperature",
      label: "Temperature",
      type: "number",
      default: 0.7,
      required: false,
    },
    {
      key: "maxTokens",
      label: "Max completion tokens",
      type: "number",
      default: 0,
      required: false,
      hint: "0 = provider default.",
    },
    {
      key: "systemPrompt",
      label: "System prompt",
      type: "textarea",
      default: "",
      required: false,
    },
    {
      key: "timeoutSec",
      label: "Timeout (seconds)",
      type: "number",
      default: DEFAULT_TIMEOUT_SEC,
      required: false,
    },
    {
      key: "promptTemplate",
      label: "Prompt template",
      type: "textarea",
      default: DEFAULT_PROMPT_TEMPLATE,
      required: false,
    },
  ],
};

/**
 * Minimal testEnvironment — we don't spawn anything, so the only
 * meaningful check is "is CEREBRAS_API_KEY present?" plus a cheap
 * ping of `/v1/models`.
 */
async function testEnvironment(
  ctx: AdapterEnvironmentTestContext,
): Promise<AdapterEnvironmentTestResult> {
  const envConfig =
    ctx.config && typeof ctx.config === "object" && ctx.config !== null
      ? ((ctx.config as Record<string, unknown>).env as Record<string, unknown> | undefined) ?? {}
      : {};
  const fromConfig =
    typeof envConfig[AUTH_ENV_VAR] === "string"
      ? (envConfig[AUTH_ENV_VAR] as string).trim()
      : "";
  const apiKey = fromConfig || (process.env[AUTH_ENV_VAR] ?? "").trim();

  const checks: AdapterEnvironmentTestResult["checks"] = [];
  const testedAt = new Date().toISOString();

  if (!apiKey) {
    checks.push({
      code: "cerebras_api_key_missing",
      level: "error",
      message: `Missing ${AUTH_ENV_VAR}.`,
      hint: `Set ${AUTH_ENV_VAR} in the adapter env config or the process environment.`,
    });
    return { adapterType: ADAPTER_TYPE, status: "fail", checks, testedAt };
  }

  try {
    const resp = await fetch(CEREBRAS_MODELS_URL, {
      headers: {
        accept: "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) {
      checks.push({
        code: "cerebras_models_http_error",
        level: "error",
        message: `Cerebras /v1/models returned HTTP ${resp.status}`,
        detail: resp.statusText,
      });
      return { adapterType: ADAPTER_TYPE, status: "fail", checks, testedAt };
    }
    checks.push({
      code: "cerebras_api_reachable",
      level: "info",
      message: "Cerebras API reachable and API key accepted.",
    });
    return { adapterType: ADAPTER_TYPE, status: "pass", checks, testedAt };
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    checks.push({
      code: "cerebras_network_error",
      level: "error",
      message: "Failed to reach Cerebras API.",
      detail: reason,
    });
    return { adapterType: ADAPTER_TYPE, status: "fail", checks, testedAt };
  }
}

/**
 * Factory invoked by the Paperclip plugin loader.
 */
export function createServerAdapter(): ServerAdapterModule {
  return {
    type: ADAPTER_TYPE,
    execute,
    testEnvironment,
    sessionCodec,
    models,
    agentConfigurationDoc,
    detectModel,
    getConfigSchema: () => configSchema,
    supportsInstructionsBundle: false,
  };
}

export default createServerAdapter;
