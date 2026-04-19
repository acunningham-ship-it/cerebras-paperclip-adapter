/**
 * Cerebras Inference Paperclip adapter — main entry.
 *
 * v0.0.1: scaffold. Implementation TBD.
 */

import {
  ADAPTER_LABEL,
  ADAPTER_TYPE,
  DEFAULT_MODEL,
  AUTH_ENV_VAR,
} from "./shared/constants.js";

export const type = ADAPTER_TYPE;
export const label = ADAPTER_LABEL;

export const models = [];

export const agentConfigurationDoc = `# Cerebras Inference Adapter Configuration

Free LLM access via Cerebras Inference. Requires \`CEREBRAS_API_KEY\` env var.

## Core configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | string | qwen-3-235b-a22b-instruct-2507 | Model id |
| timeoutSec | number | 300 | Execution timeout |

See FREE_MODELS in src/shared/constants.ts for available free models.
`;

// TODO(Dev Team): implement createServerAdapter() factory
