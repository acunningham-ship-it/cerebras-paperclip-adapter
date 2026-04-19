/**
 * Execute a Cerebras Inference run.
 *
 * v0.7: adds OpenAI-style tool calling. When the execution context
 * exposes tools (`ctx.tools`), we switch to a non-streaming chat.completions
 * loop, translating each entry to OpenAI `tools`/`tool_choice`, invoking
 * any `tool_calls` via `ctx.tools.invoke()`, appending tool_result messages,
 * and looping until `finish_reason === "stop"` (or `TOOL_CALL_MAX_ITERATIONS`
 * is reached). When no tools are configured we keep the v0.0.1 streaming
 * single-turn fast-path untouched — Cerebras is ~1500 tok/sec and the whole
 * point is to stream those tokens live.
 *
 * Cerebras throttles aggressively on the free tier; 429s mid-loop surface a
 * short, friendly message rather than a cryptic HTTP error.
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
  RATE_LIMIT_MESSAGE,
  TOOL_CALL_MAX_ITERATIONS,
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

// ---------------------------------------------------------------------
// Tool plumbing
// ---------------------------------------------------------------------

/**
 * Minimal duck-typed shape the adapter uses to read tools from
 * `AdapterExecutionContext`. The current `@paperclipai/adapter-utils`
 * release does not declare `tools` on the context, so we treat it as
 * optional and check at runtime. This lets v0.7 compile cleanly against
 * today's SDK and light up tool calling as soon as the host plugin
 * starts populating `ctx.tools`.
 */
interface ToolDescriptor {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  inputSchema?: Record<string, unknown>;
}
interface ToolRegistry {
  list?: () => ToolDescriptor[] | Promise<ToolDescriptor[]>;
  invoke: (name: string, args: Record<string, unknown>) => Promise<unknown>;
}

async function readToolsFromContext(ctx: AdapterExecutionContext): Promise<{
  registry: ToolRegistry | null;
  descriptors: ToolDescriptor[];
}> {
  const maybe = (ctx as unknown as { tools?: unknown }).tools;
  if (!maybe || typeof maybe !== "object") return { registry: null, descriptors: [] };
  const reg = maybe as Partial<ToolRegistry> & {
    entries?: ToolDescriptor[];
    available?: ToolDescriptor[];
  };
  if (typeof reg.invoke !== "function") return { registry: null, descriptors: [] };
  let descriptors: ToolDescriptor[] = [];
  try {
    if (typeof reg.list === "function") {
      const listed = await reg.list();
      if (Array.isArray(listed)) descriptors = listed;
    } else if (Array.isArray(reg.entries)) {
      descriptors = reg.entries;
    } else if (Array.isArray(reg.available)) {
      descriptors = reg.available;
    }
  } catch {
    descriptors = [];
  }
  const valid = descriptors.filter(
    (t): t is ToolDescriptor => !!t && typeof t.name === "string" && t.name.length > 0,
  );
  return { registry: reg as ToolRegistry, descriptors: valid };
}

function toolDescriptorsToOpenAi(
  tools: ToolDescriptor[],
): Array<Record<string, unknown>> {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description ?? "",
      parameters: t.parameters ?? t.inputSchema ?? {
        type: "object",
        properties: {},
      },
    },
  }));
}

// ---------------------------------------------------------------------
// Non-streaming chat.completions call used by the tool-calling loop.
// ---------------------------------------------------------------------

interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
}

interface OpenAiChoice {
  index: number;
  finish_reason: string | null;
  message: {
    role: "assistant";
    content: string | null;
    tool_calls?: Array<{
      id: string;
      type: "function";
      function: { name: string; arguments: string };
    }>;
  };
}

interface OpenAiNonStreamingResponse {
  id?: string;
  model?: string;
  choices?: OpenAiChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface ChatCallOutcome {
  ok: boolean;
  status: number;
  statusText: string;
  body: OpenAiNonStreamingResponse | null;
  rawError: string | null;
}

async function callChatCompletionsJson(
  apiKey: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
): Promise<ChatCallOutcome> {
  const resp = await fetch(CEREBRAS_CHAT_COMPLETIONS_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    const rawText = await resp.text().catch(() => "");
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(rawText);
    } catch {
      parsed = null;
    }
    return {
      ok: false,
      status: resp.status,
      statusText: resp.statusText,
      body: null,
      rawError: describeOpenAiError(parsed) ?? rawText.slice(0, 500),
    };
  }
  const parsedBody = (await resp.json().catch(() => null)) as OpenAiNonStreamingResponse | null;
  return {
    ok: true,
    status: resp.status,
    statusText: resp.statusText,
    body: parsedBody,
    rawError: null,
  };
}

function safeParseJsonArgs(raw: string): Record<string, unknown> {
  if (!raw) return {};
  try {
    const v = JSON.parse(raw);
    if (v && typeof v === "object" && !Array.isArray(v)) {
      return v as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

function stringifyToolResult(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
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

  // Prompt assembly
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

  // Tools discovery — duck-typed; absent on SDKs that don't expose ctx.tools.
  const { registry: toolsRegistry, descriptors: toolDescriptors } =
    await readToolsFromContext(ctx);
  const toolCallingEnabled = !!toolsRegistry && toolDescriptors.length > 0;

  if (onMeta) {
    await onMeta({
      adapterType: ADAPTER_TYPE,
      command: `POST ${CEREBRAS_CHAT_COMPLETIONS_URL}`,
      cwd: process.cwd(),
      commandArgs: [
        model,
        `timeout=${timeoutSec}s`,
        toolCallingEnabled
          ? `tools=${toolDescriptors.length}`
          : "tools=0",
      ],
      commandNotes: [
        `Cerebras Inference — ~1500 tok/sec${toolCallingEnabled ? " (tool-calling loop)" : " streaming"}`,
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

  if (toolCallingEnabled && toolsRegistry) {
    return await executeToolCallingLoop({
      apiKey,
      model,
      systemPrompt,
      userPrompt,
      temperature,
      maxTokensCfg,
      timeoutSec,
      toolDescriptors,
      toolsRegistry,
      onLog,
    });
  }

  return await executeStreaming({
    apiKey,
    model,
    systemPrompt,
    userPrompt,
    temperature,
    maxTokensCfg,
    timeoutSec,
    onLog,
  });
}

// ---------------------------------------------------------------------
// Streaming single-turn (v0.0.1 fast-path, preserved verbatim).
// ---------------------------------------------------------------------

async function executeStreaming(args: {
  apiKey: string;
  model: string;
  systemPrompt: string;
  userPrompt: string;
  temperature: number;
  maxTokensCfg: number;
  timeoutSec: number;
  onLog: AdapterExecutionContext["onLog"];
}): Promise<AdapterExecutionResult> {
  const { apiKey, model, systemPrompt, userPrompt, temperature, maxTokensCfg, timeoutSec, onLog } = args;

  const messages: Array<{ role: "system" | "user"; content: string }> = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  messages.push({ role: "user", content: userPrompt });

  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
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

  if (!resp.ok) {
    clearTimeout(timeoutHandle);
    const rawText = await resp.text().catch(() => "");
    let parsedErr: unknown = null;
    try { parsedErr = JSON.parse(rawText); } catch { parsedErr = null; }
    const providerMsg = describeOpenAiError(parsedErr) ?? rawText.slice(0, 500);
    const baseMsg = `Cerebras HTTP ${resp.status}: ${providerMsg || resp.statusText}`;
    const msg = resp.status === 429 ? `${baseMsg} — ${RATE_LIMIT_MESSAGE}` : baseMsg;
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

// ---------------------------------------------------------------------
// Tool-calling loop (non-streaming; up to TOOL_CALL_MAX_ITERATIONS turns).
// ---------------------------------------------------------------------

async function executeToolCallingLoop(args: {
  apiKey: string;
  model: string;
  systemPrompt: string;
  userPrompt: string;
  temperature: number;
  maxTokensCfg: number;
  timeoutSec: number;
  toolDescriptors: ToolDescriptor[];
  toolsRegistry: ToolRegistry;
  onLog: AdapterExecutionContext["onLog"];
}): Promise<AdapterExecutionResult> {
  const {
    apiKey, model, systemPrompt, userPrompt, temperature, maxTokensCfg,
    timeoutSec, toolDescriptors, toolsRegistry, onLog,
  } = args;

  const openAiTools = toolDescriptorsToOpenAi(toolDescriptors);
  const messages: ChatMessage[] = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  messages.push({ role: "user", content: userPrompt });

  const abort = new AbortController();
  const timeoutHandle = setTimeout(() => abort.abort(), timeoutSec * 1000);

  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let lastCompletionId: string | null = null;
  let lastModel: string | null = null;
  let lastFinishReason: string | null = null;
  let finalText = "";

  try {
    for (let iter = 0; iter < TOOL_CALL_MAX_ITERATIONS; iter++) {
      const reqBody: Record<string, unknown> = {
        model,
        messages,
        temperature,
        stream: false,
        tools: openAiTools,
        tool_choice: "auto",
      };
      if (maxTokensCfg > 0) reqBody.max_tokens = maxTokensCfg;

      let outcome: ChatCallOutcome;
      try {
        outcome = await callChatCompletionsJson(apiKey, reqBody, abort.signal);
      } catch (err) {
        const timedOut = (err as { name?: string })?.name === "AbortError";
        const reason = err instanceof Error ? err.message : String(err);
        const msg = timedOut
          ? `Timed out after ${timeoutSec}s`
          : `Cerebras request failed: ${reason}`;
        await onLog("stderr", `[paperclip-cerebras] ${msg}\n`);
        return {
          exitCode: timedOut ? 124 : 1,
          signal: null,
          timedOut,
          errorMessage: msg,
          errorCode: timedOut ? "timeout" : "cerebras_request_failed",
          provider: PROVIDER_SLUG,
          biller: BILLER_SLUG,
          model,
          billingType: "credits",
          usage: { inputTokens: totalInputTokens, outputTokens: totalOutputTokens },
          summary: finalText,
        };
      }

      if (!outcome.ok) {
        const baseMsg = `Cerebras HTTP ${outcome.status}: ${outcome.rawError || outcome.statusText}`;
        const msg = outcome.status === 429
          ? `${baseMsg} — ${RATE_LIMIT_MESSAGE}`
          : baseMsg;
        await onLog("stderr", `[paperclip-cerebras] ${msg}\n`);
        let errorCode: string = "cerebras_http_error";
        if (outcome.status === 401 || outcome.status === 403) errorCode = "cerebras_auth_required";
        else if (outcome.status === 429) errorCode = "cerebras_rate_limited";
        else if (outcome.status >= 500) errorCode = "cerebras_upstream_error";
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
          usage: { inputTokens: totalInputTokens, outputTokens: totalOutputTokens },
          summary: finalText,
        };
      }

      const body = outcome.body;
      if (!body || !Array.isArray(body.choices) || body.choices.length === 0) {
        await onLog("stderr", `[paperclip-cerebras] Empty chat.completions response on iteration ${iter}\n`);
        return {
          exitCode: 1,
          signal: null,
          timedOut: false,
          errorMessage: "Cerebras returned an empty chat.completions response",
          errorCode: "cerebras_empty_response",
          provider: PROVIDER_SLUG,
          biller: BILLER_SLUG,
          model,
          billingType: "credits",
          usage: { inputTokens: totalInputTokens, outputTokens: totalOutputTokens },
          summary: finalText,
        };
      }

      if (typeof body.id === "string") lastCompletionId = body.id;
      if (typeof body.model === "string") lastModel = body.model;
      if (body.usage) {
        if (typeof body.usage.prompt_tokens === "number") totalInputTokens += body.usage.prompt_tokens;
        if (typeof body.usage.completion_tokens === "number") totalOutputTokens += body.usage.completion_tokens;
      }

      const choice = body.choices[0];
      lastFinishReason = choice.finish_reason;
      const assistantMsg = choice.message;
      const toolCalls = assistantMsg.tool_calls ?? [];
      const assistantContent = assistantMsg.content ?? "";

      // Mirror assistant text to the log so operators see progress.
      if (assistantContent) {
        await onLog("stdout", assistantContent);
      }

      // Push the assistant turn into history verbatim.
      messages.push({
        role: "assistant",
        content: assistantContent || null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });

      if (choice.finish_reason === "tool_calls" && toolCalls.length > 0) {
        for (const tc of toolCalls) {
          const toolName = tc.function?.name ?? "";
          const rawArgs = tc.function?.arguments ?? "";
          const parsedArgs = safeParseJsonArgs(rawArgs);
          await onLog(
            "stdout",
            `\n[paperclip-cerebras] tool_call ${toolName}(${rawArgs.slice(0, 200)})\n`,
          );
          let resultText: string;
          try {
            const result = await toolsRegistry.invoke(toolName, parsedArgs);
            resultText = stringifyToolResult(result);
          } catch (err) {
            const reason = err instanceof Error ? err.message : String(err);
            resultText = JSON.stringify({ error: reason });
            await onLog(
              "stderr",
              `[paperclip-cerebras] tool ${toolName} failed: ${reason}\n`,
            );
          }
          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: resultText,
          });
        }
        // Continue loop for follow-up assistant turn.
        continue;
      }

      // finish_reason === "stop" (or length/content_filter/etc.) — we're done.
      finalText = assistantContent;
      break;
    }

    if (lastFinishReason === "tool_calls") {
      // Hit the iteration cap.
      await onLog(
        "stderr",
        `[paperclip-cerebras] tool-call loop hit max iterations (${TOOL_CALL_MAX_ITERATIONS})\n`,
      );
      return {
        exitCode: 1,
        signal: null,
        timedOut: false,
        errorMessage: `Tool-call loop exceeded ${TOOL_CALL_MAX_ITERATIONS} iterations without a final answer.`,
        errorCode: "cerebras_tool_loop_exhausted",
        provider: PROVIDER_SLUG,
        biller: BILLER_SLUG,
        model: lastModel ?? model,
        billingType: "credits",
        usage: { inputTokens: totalInputTokens, outputTokens: totalOutputTokens },
        sessionId: lastCompletionId,
        sessionParams: lastCompletionId ? { sessionId: lastCompletionId } : null,
        sessionDisplayId: lastCompletionId,
        summary: finalText,
      };
    }
  } finally {
    clearTimeout(timeoutHandle);
  }

  return {
    exitCode: 0,
    signal: null,
    timedOut: false,
    errorMessage: null,
    usage: {
      inputTokens: totalInputTokens,
      outputTokens: totalOutputTokens,
    },
    sessionId: lastCompletionId,
    sessionParams: lastCompletionId ? { sessionId: lastCompletionId } : null,
    sessionDisplayId: lastCompletionId,
    provider: PROVIDER_SLUG,
    biller: BILLER_SLUG,
    model: lastModel ?? model,
    billingType: "credits",
    costUsd: 0,
    resultJson: {
      text: finalText,
      finishReason: lastFinishReason,
      completionId: lastCompletionId,
      model: lastModel,
    },
    summary: finalText,
  };
}

// TODO(Dev Team):
// 1. Session resume — persist multi-turn history across runs (pure
//    client-side bookkeeping; Cerebras has no server-side session concept).
// 2. Cost tracking — hard-coded to 0. Compute per-model costUsd from
//    Cerebras's paid rate card once billing is enabled.
// 3. Retry/backoff on transient 429 / 5xx inside the tool-call loop.
