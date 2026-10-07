import { isNativeAgentTool } from "../codex/model-capability-policy";
import { sseDataPayload, type SseBlockRewrite } from "./sse-payload-rewrite";
import { markResponseNonReplayable } from "../lib/upstream-retry";

type SingleAgentErrorCode =
  | "single_agent_worker_disabled" | "single_agent_delegation_disabled" | "single_agent_parallel_tools_disabled"
  | "single_agent_invalid_response" | "single_agent_invalid_tool_identity" | "single_agent_incomplete_tool_batch"
  | "single_agent_late_tool_call" | "single_agent_duplicate_terminal" | "single_agent_inconsistent_tool_batch"
  | "single_agent_unsupported_transport";

/** A local capability refusal must neither become an arbitrary upstream code nor trigger model retries. */
export function singleAgentErrorResponse(code: SingleAgentErrorCode, message = "This model runs as a single agent. Use one ordinary tool at a time in the main agent; delegation is disabled."): Response {
  const response = Response.json({ error: { type: "invalid_request_error", code, message } }, { status: 400 });
  markResponseNonReplayable(response);
  return response;
}

const CLIENT_CALLS = new Set(["function_call", "custom_tool_call", "local_shell_call", "computer_call", "tool_search_call"]);
function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
function clientCalls(response: unknown): Record<string, unknown>[] {
  if (!object(response) || !Array.isArray(response.output)) return [];
  return response.output.filter((item): item is Record<string, unknown> => object(item) && CLIENT_CALLS.has(String(item.type)));
}

export function singleAgentResponseViolation(response: unknown): SingleAgentErrorCode | undefined {
  if (!object(response) || !Array.isArray(response.output)) return "single_agent_invalid_response";
  const calls = clientCalls(response);
  if (calls.some(call => isNativeAgentTool(call.name, call.namespace))) return "single_agent_delegation_disabled";
  if (calls.length > 1) return "single_agent_parallel_tools_disabled";
  if (calls.length && (response.status !== "completed" || calls.some(call => call.status !== undefined && call.status !== "completed"))) return "single_agent_incomplete_tool_batch";
  if (calls.some(call => typeof call.id !== "string" || call.id.length === 0 || call.id.length > 256)) return "single_agent_invalid_tool_identity";
  return undefined;
}

/**
 * The client can execute output_item.done before the model's terminal frame. Hold client-call
 * lifecycle events until the terminal snapshot proves the whole batch is one ordinary call.
 * Keep text streaming; retain only bounded identity metadata, never call arguments or blocks.
 */
export function createSingleAgentToolGuardBlockRewrite(options: {
  onCompletedResponse?: (response: Record<string, unknown>) => void;
  onFailure?: () => void;
} = {}): SseBlockRewrite {
  let stopped = false;
  let sawCall = false;
  let completed = false;
  const identities = new Set<string>();

  const failure = (code: SingleAgentErrorCode): readonly string[] => {
    stopped = true;
    identities.clear();
    options.onFailure?.();
    return [`event: response.failed\ndata: ${JSON.stringify({
      type: "response.failed",
      response: {
        status: "failed", output: [],
        error: { type: "invalid_request_error", code, message: "This model runs as a single agent. Use one ordinary tool at a time in the main agent; delegation is disabled." },
      },
    })}`];
  };
  const remember = (id: unknown): readonly string[] | undefined => {
    sawCall = true;
    if (typeof id !== "string" || id.length === 0 || id.length > 256) return failure("single_agent_invalid_tool_identity");
    identities.add(id);
    if (identities.size > 1) return failure("single_agent_parallel_tools_disabled");
    return undefined;
  };

  const rewrite: SseBlockRewrite = block => {
    if (stopped) return [];
    const data = sseDataPayload(block);
    if (data === null) return [block];
    if (data === "[DONE]") {
      if (sawCall && !completed) return failure("single_agent_incomplete_tool_batch");
      return [block];
    }
    let payload: unknown;
    try { payload = JSON.parse(data); } catch { return [block]; }
    if (!object(payload)) return [block];
    const type = payload.type;
    const item = payload.item;
    if ((type === "response.output_item.added" || type === "response.output_item.done")
      && object(item) && CLIENT_CALLS.has(String(item.type))) {
      if (completed) return failure("single_agent_late_tool_call");
      if (isNativeAgentTool(item.name, item.namespace)) return failure("single_agent_delegation_disabled");
      return remember(item.id) ?? [];
    }
    if (typeof type === "string" && (type.startsWith("response.function_call_arguments.") || type.startsWith("response.custom_tool_call_input."))) {
      if (completed) return failure("single_agent_late_tool_call");
      return remember(payload.item_id) ?? [];
    }
    if (type === "response.failed" || type === "response.incomplete") {
      identities.clear();
      if (sawCall || clientCalls(payload.response).length > 0) return failure("single_agent_incomplete_tool_batch");
      stopped = true;
      options.onFailure?.();
      return [block];
    }
    if (type !== "response.completed") return [block];
    if (completed) return failure("single_agent_duplicate_terminal");
    const violation = singleAgentResponseViolation(payload.response);
    if (violation) return failure(violation);
    const calls = clientCalls(payload.response);
    if (sawCall && (calls.length !== 1 || !identities.has(String(calls[0]?.id)))) {
      return failure("single_agent_inconsistent_tool_batch");
    }
    identities.clear();
    completed = true;
    options.onCompletedResponse?.(payload.response as Record<string, unknown>);
    if (calls.length === 0) return [block];
    const call = calls[0]!;
    // Use the existing bounded-JSON lifecycle contract: one complete item before the terminal.
    // The undeclared-tool guard and identity restores run before this gate.
    const output = (payload.response as { output: unknown[] }).output;
    const itemDone = { type: "response.output_item.done", output_index: output.indexOf(call), item: call };
    return [`event: response.output_item.done\ndata: ${JSON.stringify(itemDone)}`, block];
  };
  rewrite.dispose = () => { identities.clear(); stopped = true; };
  return rewrite;
}
