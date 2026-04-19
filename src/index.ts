/**
 * Cerebras Inference Paperclip adapter — main entry.
 *
 * Cerebras runs LLMs on its own CS-3 wafer-scale silicon and is the
 * fastest inference backend in the world — typical throughput is
 * ~1500 tokens/second on frontier open-weights models.
 *
 * v0.7:
 *   - OpenAI-style tool calling (see `server/execute.ts`).
 *   - Full model catalog: queries `/v1/models` at boot and sorts
 *     free-tier models before pro/paid ones. Falls back to the hardcoded
 *     constants list when the API is unreachable.
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
  CONTEXT_WINDOWS,
  DEFAULT_MODEL,
  DEFAULT_PROMPT_TEMPLATE,
  DEFAULT_TIMEOUT_SEC,
  FREE_MODELS,
  PAID_MODELS,
} from "./shared/constants.js";
import {
  detectModel,
  execute,
  sessionCodec,
} from "./server/index.js";

export const type = ADAPTER_TYPE;
export const label = ADAPTER_LABEL;

/**
 * Extended AdapterModel with tier + context-window metadata. Paperclip's
 * AdapterModel type only requires { id, label }, but we tag each entry with
 * the extra fields so downstream tools (pricing, config UIs, etc.) can key
 * on them without a second lookup.
 */
interface CerebrasModelMeta extends AdapterModel {
  free: boolean;
  contextWindow: number;
}

const FREE_SET = new Set<string>(FREE_MODELS);

function isFreeModel(id: string): boolean {
  return FREE_SET.has(id);
}

function contextWindowFor(id: string): number {
  return CONTEXT_WINDOWS[id] ?? 8192;
}

function labelFor(id: string, free: boolean, ctx: number): string {
  const tier = free ? "free" : "pro";
  const ctxLabel = ctx >= 1000 ? `${Math.round(ctx / 1024)}K` : String(ctx);
  return `${id} — ${tier} · ${ctxLabel} ctx (Cerebras ~1500 tok/sec)`;
}

function toModelMeta(id: string): CerebrasModelMeta {
  const free = isFreeModel(id);
  const ctx = contextWindowFor(id);
  return { id, label: labelFor(id, free, ctx), free, contextWindow: ctx };
}

/**
 * Hardcoded fallback — used when `/v1/models` is unreachable (offline or
 * missing API key). Free models come first, paid/pro second.
 */
const STATIC_FALLBACK: CerebrasModelMeta[] = [
  ...FREE_MODELS.map((id) => toModelMeta(id)),
  ...PAID_MODELS.map((id) => toModelMeta(id)),
];

function sortModels(models: CerebrasModelMeta[]): CerebrasModelMeta[] {
  return [...models].sort((a, b) => {
    if (a.free !== b.free) return a.free ? -1 : 1;
    return a.id.localeCompare(b.id);
  });
}

async function loadModels(): Promise<CerebrasModelMeta[]> {
  const apiKey = (process.env[AUTH_ENV_VAR] ?? "").trim();
  if (!apiKey) return sortModels(STATIC_FALLBACK);
  try {
    const resp = await fetch(CEREBRAS_MODELS_URL, {
      headers: {
        accept: "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return sortModels(STATIC_FALLBACK);
    const body = (await resp.json()) as { data?: Array<{ id?: string; context_length?: number }> };
    if (!body || !Array.isArray(body.data)) return sortModels(STATIC_FALLBACK);
    const out: CerebrasModelMeta[] = [];
    for (const m of body.data) {
      if (!m || typeof m.id !== "string" || !m.id) continue;
      const free = isFreeModel(m.id);
      const ctx =
        typeof m.context_length === "number" && m.context_length > 0
          ? m.context_length
          : contextWindowFor(m.id);
      out.push({ id: m.id, label: labelFor(m.id, free, ctx), free, contextWindow: ctx });
    }
    return out.length > 0 ? sortModels(out) : sortModels(STATIC_FALLBACK);
  } catch {
    return sortModels(STATIC_FALLBACK);
  }
}

export const models: AdapterModel[] = await loadModels();

export const agentConfigurationDoc = `# Cerebras Inference Adapter

This adapter routes Paperclip agents to
[Cerebras Inference](https://cerebras.ai/inference), the fastest LLM
inference backend in the world (~1500 tokens/second on frontier
open-weights models running on their CS-3 wafer-scale silicon).

## Prerequisites

- A Cerebras Inference API key from https://cloud.cerebras.ai/

## Core Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | string | \`${DEFAULT_MODEL}\` | Cerebras model id. Free tier default: \`${DEFAULT_MODEL}\`. |
| temperature | number | 0.7 | Sampling temperature. |
| maxTokens | number | _(unset)_ | Max completion tokens. 0 = provider default. |
| systemPrompt | string | _(none)_ | Optional system prompt prepended to the conversation. |
| timeoutSec | number | ${DEFAULT_TIMEOUT_SEC} | Hard timeout for a single run. |
| promptTemplate | string | _(default)_ | Mustache-style template for the user message. |
| env | object | \`{}\` | Extra env. \`CEREBRAS_API_KEY\` here is preferred over process env. |

## Environment Variables

- \`${AUTH_ENV_VAR}\` — required. Read from \`config.env.${AUTH_ENV_VAR}\` first, then \`process.env.${AUTH_ENV_VAR}\`.

## Features

- **Streaming single-turn** (no tools): SSE chunks are mirrored straight
  to the log for live ~1500 tok/sec throughput.
- **Tool calling** (when \`ctx.tools\` is populated): OpenAI-style loop,
  up to 10 iterations. Each \`tool_call\` is invoked via the adapter
  context; \`tool\` messages are appended and the conversation continues
  until \`finish_reason === "stop"\`.
- **Rate-limit handling**: HTTP 429 surfaces a friendly message
  prompting the caller to back off or upgrade to the pro tier.

## Notes

- Billing is reported as \`provider=cerebras, biller=cerebras, billingType=credits\`.
- Free tier \`costUsd=0\`. Paid rate-card integration is TODO.
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
      hint: "Cerebras model id. Free tier currently: qwen-3-235b-a22b-instruct-2507.",
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
