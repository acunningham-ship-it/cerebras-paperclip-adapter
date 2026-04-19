/**
 * Session state codec + SSE parser for the Cerebras Inference adapter.
 *
 * v0.0.1 is single-turn: we don't persist a real session across runs yet,
 * but we still implement the `AdapterSessionCodec` shape so Paperclip's
 * session machinery is happy.
 */

import type { AdapterSessionCodec } from "@paperclipai/adapter-utils";

export const sessionCodec: AdapterSessionCodec = {
  deserialize(raw: unknown): Record<string, unknown> | null {
    if (raw && typeof raw === "object") return raw as Record<string, unknown>;
    return null;
  },
  serialize(params: Record<string, unknown> | null): Record<string, unknown> | null {
    if (!params) return null;
    return { ...params };
  },
  getDisplayId(params: Record<string, unknown> | null): string | null {
    if (!params) return null;
    const v = params.sessionId;
    return typeof v === "string" && v.length > 0 ? v : null;
  },
};

/**
 * A single aggregated streaming response.
 */
export interface ParsedStream {
  text: string;
  model: string | null;
  finishReason: string | null;
  inputTokens: number;
  outputTokens: number;
  /** Unique id returned by Cerebras for this completion, if any. */
  completionId: string | null;
  raw: Array<Record<string, unknown>>;
}

/**
 * Parse an OpenAI-compatible SSE stream body (as a single string buffer).
 *
 * Cerebras returns standard `data: {...}\n\n` frames terminated by
 * `data: [DONE]`. We tolerate partial/ragged frames — any line that
 * isn't JSON is skipped.
 */
export function parseOpenAiSseStream(buffer: string): ParsedStream {
  let text = "";
  let model: string | null = null;
  let finishReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let completionId: string | null = null;
  const raw: Array<Record<string, unknown>> = [];

  // SSE frames are separated by blank lines. Split on double-newlines, then
  // fall back to line-by-line for safety.
  const frames = buffer.split(/\r?\n\r?\n/);
  for (const frame of frames) {
    for (const line of frame.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      let obj: Record<string, unknown>;
      try {
        obj = JSON.parse(payload) as Record<string, unknown>;
      } catch {
        continue;
      }
      raw.push(obj);

      if (typeof obj.id === "string" && !completionId) completionId = obj.id;
      if (typeof obj.model === "string" && !model) model = obj.model;

      // Usage frames (sent at the very end when stream_options.include_usage).
      const usage = obj.usage as Record<string, unknown> | undefined;
      if (usage && typeof usage === "object") {
        const pt = usage.prompt_tokens;
        const ct = usage.completion_tokens;
        if (typeof pt === "number") inputTokens = pt;
        if (typeof ct === "number") outputTokens = ct;
      }

      const choices = obj.choices as Array<Record<string, unknown>> | undefined;
      if (!Array.isArray(choices)) continue;
      for (const choice of choices) {
        const delta = choice.delta as Record<string, unknown> | undefined;
        if (delta && typeof delta.content === "string") text += delta.content;
        // Some providers emit final content under `message` instead of `delta`.
        const message = choice.message as Record<string, unknown> | undefined;
        if (message && typeof message.content === "string") text += message.content;
        const fr = choice.finish_reason;
        if (typeof fr === "string" && fr) finishReason = fr;
      }
    }
  }

  return { text, model, finishReason, inputTokens, outputTokens, completionId, raw };
}

/**
 * Map a Cerebras/OpenAI error response body to a short human-readable message.
 */
export function describeOpenAiError(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const err = (body as Record<string, unknown>).error;
  if (err && typeof err === "object") {
    const msg = (err as Record<string, unknown>).message;
    if (typeof msg === "string" && msg) return msg;
  }
  const msg = (body as Record<string, unknown>).message;
  if (typeof msg === "string" && msg) return msg;
  return null;
}
