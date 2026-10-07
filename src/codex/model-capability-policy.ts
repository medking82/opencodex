import type { OcxConfig, OcxParsedRequest } from "../types";
import { slugEquals, slugsEquivalent } from "../providers/slug-codec";
import { toolChoiceToolPredicate } from "../types/tools";

type ModelIdentity = { providerName: string; modelId: string };
const AGENT_NAMES = new Set([
  "spawn_agent", "send_input", "resume_agent", "close_agent", "wait",
  "send_message", "followup_task", "interrupt_agent", "list_agents", "wait_agent",
]);
const AGENT_NAMESPACES = new Set(["functions", "collaboration"]);

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

/** Select the resolved destination, not a display name or caller-supplied role. */
export function isSingleAgentModel(config: Pick<OcxConfig, "singleAgentModels">, route: ModelIdentity): boolean {
  return config.singleAgentModels?.some(selector => slugEquals(selector, route.providerName, route.modelId)) === true;
}

const CATALOG_ORIGIN = "opencodex_single_agent_origin";

/** Final projection after the global collaboration mode; restore only matching owned fields. */
export function applySingleAgentCatalogPolicy(entries: Record<string, unknown>[], config: Pick<OcxConfig, "singleAgentModels">): void {
  for (const entry of entries) {
    const origin = entry[CATALOG_ORIGIN];
    if (object(origin) && origin.slug === entry.slug
      && (origin.version === null || ["v1", "v2", "disabled"].includes(String(origin.version)))
      && (origin.parallel === null || typeof origin.parallel === "boolean")) {
      if (entry.multi_agent_version === "disabled") {
        if (origin.version === null) delete entry.multi_agent_version;
        else entry.multi_agent_version = origin.version;
      }
      if (entry.supports_parallel_tool_calls === false) {
        if (origin.parallel === null) delete entry.supports_parallel_tool_calls;
        else entry.supports_parallel_tool_calls = origin.parallel;
      }
      delete entry[CATALOG_ORIGIN];
    }
    if (typeof entry.slug !== "string" || !config.singleAgentModels?.some(selector => slugsEquivalent(selector, entry.slug as string))) continue;
    entry[CATALOG_ORIGIN] = {
      slug: entry.slug,
      version: typeof entry.multi_agent_version === "string" ? entry.multi_agent_version : null,
      parallel: typeof entry.supports_parallel_tool_calls === "boolean" ? entry.supports_parallel_tool_calls : null,
    };
    entry.multi_agent_version = "disabled";
    entry.supports_parallel_tool_calls = false;
  }
}

export class SingleAgentWorkerDisabledError extends Error {
  readonly code = "single_agent_worker_disabled";
  constructor() {
    super("This model can only run as the main agent. Execute the work sequentially in the main agent.");
  }
}

export function isNativeAgentTool(name: unknown, namespace?: unknown): boolean {
  if (typeof name !== "string") return false;
  if (typeof namespace === "string" && namespace.length > 0 && !AGENT_NAMESPACES.has(namespace)) return false;
  if (namespace === "collaboration" || name.startsWith("collaboration.") || name.startsWith("collaboration__")) return true;
  if (AGENT_NAMES.has(name)) return true;
  for (const prefix of ["functions.", "functions__", "collaboration.", "collaboration__"]) {
    if (name.startsWith(prefix) && AGENT_NAMES.has(name.slice(prefix.length))) return true;
  }
  return false;
}

function filterDeclarations(value: unknown, namespace?: unknown): unknown {
  if (!Array.isArray(value)) return value;
  const filtered: unknown[] = [];
  for (const tool of value) {
    if (!object(tool)) { filtered.push(tool); continue; }
    if (tool.type === "namespace" && Array.isArray(tool.tools)) {
      const children = filterDeclarations(tool.tools, tool.name) as unknown[];
      if (children.length > 0) filtered.push(children === tool.tools ? tool : { ...tool, tools: children });
      continue;
    }
    const name = object(tool.function) ? tool.function.name : tool.name;
    if (!isNativeAgentTool(name, namespace ?? tool.namespace)) filtered.push(tool);
  }
  return filtered.length === value.length && filtered.every((tool, i) => tool === value[i]) ? value : filtered;
}

/** Restrict protocol declarations only; leave user text and executed-call history intact. */
export function restrictSingleAgentDeclarations(body: unknown): void {
  if (!object(body)) return;
  if (Array.isArray(body.tools)) body.tools = filterDeclarations(body.tools);
  if (Array.isArray(body.input)) {
    body.input = body.input.map(item => {
      if (!object(item) || (item.type !== "additional_tools" && item.type !== "tool_search_output")) return item;
      const tools = filterDeclarations(item.tools);
      return tools === item.tools ? item : { ...item, tools };
    });
  }
  if (object(body.tool_choice) && body.tool_choice.type === "allowed_tools") {
    body.tool_choice = { ...body.tool_choice, tools: filterDeclarations(body.tool_choice.tools) };
  }
  body.parallel_tool_calls = false;
}

/** A forced forbidden choice must fail rather than silently nominate a different tool. */
export function applySingleAgentRequestPolicy(parsed: OcxParsedRequest): string | undefined {
  const tools = parsed.context.tools ?? [];
  const remaining = tools.filter(tool => !isNativeAgentTool(tool.name, tool.namespace));
  const choice = parsed.options.toolChoice;
  const allowed = toolChoiceToolPredicate(choice, tools);
  const required = choice === "required"
    || typeof choice === "object" && ("name" in choice || "allowedTools" in choice && choice.mode === "required");
  if (required && !remaining.some(allowed)) {
    return "single_agent_delegation_disabled: the selected model cannot delegate; use an ordinary tool in the main agent.";
  }
  parsed.context.tools = remaining;
  if (choice && typeof choice === "object" && "allowedTools" in choice) {
    parsed.options.toolChoice = {
      ...choice,
      allowedTools: choice.allowedTools.filter(name => remaining.some(toolChoiceToolPredicate({ ...choice, allowedTools: [name] }, remaining))),
    };
  }
  restrictSingleAgentDeclarations(parsed._rawBody);
  parsed.options.parallelToolCalls = false;
  return undefined;
}
