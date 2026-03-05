export interface AgentOption {
  name: string;
  description?: string;
}

export interface PlaygroundConfig {
  apiBase: string;
  agents: AgentOption[];
  debug: boolean;
  setup_type?: "single" | "multi" | "dynamic_pipeline" | "pipeline";
}

export interface BudgetData {
  limit?: number;
  remaining?: number;
  spent?: number;
  percent_used?: number;
}

export interface DescribeData {
  name: string;
  description: string;
  tools: string[] | Array<{ name: string; description?: string }>;
  budget?: BudgetData;
  internal_agents?: string[];
  setup_type?: "single" | "multi" | "dynamic_pipeline" | "pipeline";
}

export interface StreamChunkText {
  type?: "text";
  text?: string;
  accumulated?: string;
}

export interface StreamChunkDone {
  type?: "done";
  done?: boolean;
  cost?: number;
  budget_remaining?: number;
  tokens?: {
    total?: number;
    total_tokens?: number;
    input_tokens?: number;
    output_tokens?: number;
  };
  events?: Array<{ hook: string; ctx: Record<string, unknown> }>;
}

export interface StreamChunkStatus {
  type: "status";
  message: string;
}

export interface StreamChunkHook {
  type: "hook";
  hook: string;
  ctx: Record<string, unknown>;
}

export type StreamChunk =
  | StreamChunkText
  | StreamChunkDone
  | StreamChunkStatus
  | StreamChunkHook
  | (StreamChunkText & { done?: boolean });

export async function fetchConfig(baseUrl = ""): Promise<PlaygroundConfig> {
  const url = `${baseUrl || ""}./config`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch config");
  const data: PlaygroundConfig = await res.json();
  const apiBase = (data.apiBase || "").replace(/\/+$/, "");
  return { ...data, apiBase: apiBase ? `${apiBase}/` : "/" };
}

export async function fetchBudget(apiBase: string, agentName?: string): Promise<BudgetData | null> {
  const path = agentName ? `${agentName}/budget` : "budget";
  const url = `${apiBase.replace(/\/$/, "")}/${path}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.json();
}

export async function fetchDescribe(
  apiBase: string,
  agentName?: string
): Promise<DescribeData | null> {
  const path = agentName ? `${agentName}/describe` : "describe";
  const url = `${apiBase.replace(/\/$/, "")}/${path}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.json();
}

/** Remote config: field schema with optional response-only values (baseline_value, current_value, overridden) */
export interface ConfigFieldSchema {
  name: string;
  path: string;
  type: string;
  default: unknown;
  description: string | null;
  constraints: Record<string, number | string>;
  enum_values: string[] | null;
  children: ConfigFieldSchema[] | null;
  remote_excluded: boolean;
  baseline_value?: unknown;
  current_value?: unknown;
  overridden?: boolean;
}

export interface ConfigSectionSchema {
  section: string;
  class_name: string;
  fields: ConfigFieldSchema[];
}

/** Full agent config from GET /config (schema + baseline, overrides, current) */
export interface AgentConfigResponse {
  agent_id: string;
  agent_name: string;
  class_name: string;
  sections: Record<string, ConfigSectionSchema>;
  baseline_values: Record<string, unknown>;
  overrides: Record<string, unknown>;
  current_values: Record<string, unknown>;
}

export interface ConfigOverrideItem {
  path: string;
  value: unknown;
}

export interface PatchConfigPayload {
  agent_id: string;
  version: number;
  overrides: ConfigOverrideItem[];
}

export interface PatchConfigResult {
  accepted: string[];
  rejected: Array<[string, string]>;
  pending_restart: string[];
}

export async function fetchAgentConfig(
  apiBase: string,
  agentName?: string
): Promise<AgentConfigResponse | null> {
  const path = agentName ? `${agentName}/config` : "config";
  const url = `${apiBase.replace(/\/$/, "")}/${path}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.json();
}

export async function patchAgentConfig(
  apiBase: string,
  agentId: string,
  overrides: ConfigOverrideItem[],
  agentName?: string
): Promise<PatchConfigResult | null> {
  const path = agentName ? `${agentName}/config` : "config";
  const url = `${apiBase.replace(/\/$/, "")}/${path}`;
  const res = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_id: agentId, version: Date.now(), overrides }),
  });
  if (!res.ok) return null;
  return res.json();
}
