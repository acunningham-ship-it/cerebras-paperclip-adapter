/**
 * Execute a single Cerebras Inference run.
 *
 * Strategy: direct OpenAI-compatible HTTP call to
 *   POST https://api.cerebras.ai/v1/chat/completions
 * with `Authorization: Bearer $CEREBRAS_API_KEY` and `stream: true`.
 *
 * Cerebras runs on their own CS-3 wafer-scale silicon and is the fastest
 * LLM inference backend anywhere (~1500 tok/sec on frontier open-weights
 * models like Qwen 3 235B). That speed is the whole point of this
 * adapter — we stream tokens straight through the SSE parser to keep
 * first-byte latency minimal.
 *
 * v0.0.1 is single-turn. Tool calling, session resume, and multi-turn
 * history are left as TODOs below.
 */

import {
  asString,
  asNumber,
  parseObject,
  renderTemplate,
  joinPromptSections,
} from "@paperclipai/adapter-utils/server-utils";
import type {
  AdapterExecutionContext,
  AdapterExecutionResult,
} from "@paperclipai/adapter-utils";
import {
  parseOpenAiSseStream,
  describeOpenAiError,
} from "./parse.js";
import {
  ADAPTER_TYPE,
  AUTH_ENV_VAR,
  BILLER_SLUG,
  CEREBRAS_CHAT_COMPLETIONS_URL,
  DEFAULT_MODEL,
  DEFAULT_PROMPT_TEMPLATE,
  DEFAULT_TIMEOUT_SEC,
  PROVIDER_SLUG,
} from "../shared/constants.js";

interface ResolvedKey {
  key: string | null;
  source: "config_env" | "process_env" | "missing";
}

function resolveCerebrasApiKey(envConfig: Record<string, unknown>): ResolvedKey {
  const fromConfig =
    typeof envConfig[AUTH_ENV_VAR] === "string"
      ? (envConfig[AUTH_ENV_VAR] as string).trim()
      : "";
  if (fromConfig) return { key: fromConfig, source: "config_env" };

  const fromProc = (process.env[AUTH_ENV_VAR] ?? "").trim();
  if (fromProc) return { key: fromProc, source: "process_env" };

  return { key: null, source: "missing" };
}

export async function execute(
  ctx: AdapterExecutionContext,
): Promise<AdapterExecutionResult> {
  const { runId, agent, config, context, onLog, onMeta } = ctx;

  const promptTemplate = asString(config.promptTemplate, DEFAULT_PROMPT_TEMPLATE);
  const model = asString(config.model, DEFAULT_MODEL);
  const timeoutSec = asNumber(config.timeoutSec, DEFAULT_TIMEOUT_SEC);
  const temperature = asNumber(config.temperature, 0.7);
  const maxTokensCfg = asNumber(config.maxTokens, 0);
  const systemPrompt = asString(config.systemPrompt, "").trim();

  const envConfig = parseObject(config.env);
  const { key: apiKey, source: apiKeySource } = resolveCerebrasApiKey(envConfig);

  // ------------------------------------------------------------------
  // Prompt assembly — mirror the (minimal) subset of claude_local that
  // makes sense for a single-turn HTTP call: render the configured
  // template, then join any wake / handoff prelude the server injected.
  // ------------------------------------------------------------------
  const templateData = {
    agentId: agent.id,
    companyId: agent.companyId,
    runId,
    company: { id: agent.companyId },
    agent,
    run: { id: runId, source: "on_demand" },
    context,
  };
  const renderedPrompt = renderTemplate(promptTemplate, templateData);
  const sessionHandoffNote = asString(context.paperclipSessionHandoffMarkdown, "").trim();
  const userPrompt = joinPromptSections([sessionHandoffNote, renderedPrompt]).trim();

  // ------------------------------------------------------------------
  // Invocation meta — nothing is spawned as a child, but we still want
  // Paperclip to log what HTTP call it is about to make.
  // ------------------------------------------------------------------
  if (onMeta) {
    await onMeta({
      adapterType: ADAPTER_TYPE,
      command: `POST ${CEREBRAS_CHAT_COMPLETIONS_URL}`,
      cwd: process.cwd(),
      commandArgs: [model, `timeout=${timeoutSec}s`],
      commandNotes: [
        `Cerebras Inference — ~1500 tok/sec streaming`,
        `API key source: ${apiKeySource}`,
      ],
      env: {
        [AUTH_ENV_VAR]: apiKey ? "<redacted>" : "<missing>",
      },
      prompt: userPrompt,
      promptMetrics: {
        promptChars: userPrompt.length,
        systemChars: systemPrompt.length,
      },
      context,
    });
  }

  if (!apiKey) {
    const msg = `Missing ${AUTH_ENV_VAR}. Set it in the adapter's env config or the process environment.`;
    await onLog("stderr", `[paperclip-cerebras] ${msg}\n`);
    return {
      exitCode: 1,
      signal: null,
      timedOut: false,
      errorMessage: msg,
      errorCode: "cerebras_auth_required",
      provider: PROVIDER_SLUG,
      biller: BILLER_SLUG,
      model,
      billingType: "credits",
    };
  }

  // ------------------------------------------------------------------
  // Build request. Cerebras follows the OpenAI chat-completions schema.
  // ------------------------------------------------------------------
  const messages: Array<{ role: "system" | "user"; content: string }> = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  messages.push({ role: "user", content: userPrompt });

  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    // Ask Cerebras to include a final usage frame — this is optional in
    // the OpenAI spec, opt-in on most providers.
    stream_options: { include_usage: true },
    temperature,
  };
  if (maxTokensCfg > 0) body.max_tokens = maxTokensCfg;

  const abort = new AbortController();
  const timeoutHandle = setTimeout(() => abort.abort(), timeoutSec * 1000);

  let resp: Response;
  try {
    resp = await fetch(CEREBRAS_CHAT_COMPLETIONS_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "text/event-stream",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: abort.signal,
    });
  } catch (err) {
    clearTimeout(timeoutHandle);
    const timedOut = (err as { name?: string })?.name === "AbortError";
    const reason = err instanceof Error ? err.message : String(err);
    const errMsg = timedOut
      ? `Timed out after ${timeoutSec}s`
      : `Cerebras request failed: ${reason}`;
    await onLog("stderr", `[paperclip-cerebras] ${errMsg}\n`);
    return {
      exitCode: timedOut ? 124 : 1,
      signal: null,
      timedOut,
      errorMessage: errMsg,
      errorCode: timedOut ? "timeout" : "cerebras_request_failed",
      provider: PROVIDER_SLUG,
      biller: BILLER_SLUG,
      model,
      billingType: "credits",
    };
  }

  // Non-2xx: read the body (JSON, not SSE) and surface the provider's error.
  if (!resp.ok) {
    clearTimeout(timeoutHandle);
    const rawText = await resp.text().catch(() => "");
    let parsedErr: unknown = null;
    try {
      parsedErr = JSON.parse(rawText);
    } catch {
      parsedErr = null;
    }
    const providerMsg = describeOpenAiError(parsedErr) ?? rawText.slice(0, 500);
    const msg = `Cerebras HTTP ${resp.status}: ${providerMsg || resp.statusText}`;
    await onLog("stderr", `[paperclip-cerebras] ${msg}\n`);

    let errorCode: string = "cerebras_http_error";
    if (resp.status === 401 || resp.status === 403) errorCode = "cerebras_auth_required";
    else if (resp.status === 429) errorCode = "cerebras_rate_limited";
    else if (resp.status >= 500) errorCode = "cerebras_upstream_error";

    return {
      exitCode: 1,
      signal: null,
      timedOut: false,
      errorMessage: msg,
      errorCode,
      provider: PROVIDER_SLUG,
      biller: BILLER_SLUG,
      model,
      billingType: "credits",
    };
  }

  // ------------------------------------------------------------------
  // Stream and accumulate. Cerebras is blisteringly fast, so chunks
  // tend to arrive in big bursts — we stream each decoded chunk to the
  // log so callers can watch tokens land in real time.
  // ------------------------------------------------------------------
  const reader = resp.body?.getReader();
  if (!reader) {
    clearTimeout(timeoutHandle);
    const msg = "Cerebras response had no body stream";
    await onLog("stderr", `[paperclip-cerebras] ${msg}\n`);
    return {
      exitCode: 1,
      signal: null,
      timedOut: false,
      errorMessage: msg,
      errorCode: "cerebras_empty_response",
      provider: PROVIDER_SLUG,
      biller: BILLER_SLUG,
      model,
      billingType: "credits",
    };
  }

  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let timedOut = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      // Mirror stream chunks straight to the log so operators see
      // the ~1500 tok/sec throughput live.
      await onLog("stdout", chunk);
    }
    buffer += decoder.decode();
  } catch (err) {
    timedOut = (err as { name?: string })?.name === "AbortError";
    const reason = err instanceof Error ? err.message : String(err);
    await onLog("stderr", `[paperclip-cerebras] stream error: ${reason}\n`);
  } finally {
    clearTimeout(timeoutHandle);
  }

  const parsed = parseOpenAiSseStream(buffer);

  if (timedOut) {
    return {
      exitCode: 124,
      signal: null,
      timedOut: true,
      errorMessage: `Timed out after ${timeoutSec}s`,
      errorCode: "timeout",
      provider: PROVIDER_SLUG,
      biller: BILLER_SLUG,
      model: parsed.model ?? model,
      billingType: "credits",
      usage: {
        inputTokens: parsed.inputTokens,
        outputTokens: parsed.outputTokens,
      },
      summary: parsed.text,
    };
  }

  return {
    exitCode: 0,
    signal: null,
    timedOut: false,
    errorMessage: null,
    usage: {
      inputTokens: parsed.inputTokens,
      outputTokens: parsed.outputTokens,
    },
    sessionId: parsed.completionId,
    sessionParams: parsed.completionId ? { sessionId: parsed.completionId } : null,
    sessionDisplayId: parsed.completionId,
    provider: PROVIDER_SLUG,
    biller: BILLER_SLUG,
    model: parsed.model ?? model,
    billingType: "credits",
    // Free tier — cost is zero. TODO(Dev Team): once paid tier lands,
    // compute costUsd from Cerebras's per-model rate card.
    costUsd: 0,
    resultJson: {
      text: parsed.text,
      finishReason: parsed.finishReason,
      completionId: parsed.completionId,
      model: parsed.model,
    },
    summary: parsed.text,
  };
}

// TODO(Dev Team):
// 1. Tool calling — translate Paperclip's tool protocol to OpenAI
//    `tools` + `tool_choice`, parse `tool_calls` deltas from the stream,
//    and loop until `finish_reason === "stop"`.
// 2. Session resume — persist conversation history and replay on
//    subsequent runs. Cerebras has no server-side session concept, so
//    this is pure client-side bookkeeping.
// 3. Cost tracking — currently hard-coded to 0 (free tier only).
// 4. Retry/backoff on 429 (rate-limit) and 5xx.
// 5. testEnvironment() — ping `/v1/models` and surface auth/network checks.
