/** Single-agent capability admission and client delivery with synthetic, offline transports. */
import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
import { handleResponses } from "../../src/server/responses";
import { createSingleAgentToolGuardBlockRewrite, singleAgentResponseViolation } from "../../src/server/responses-single-agent-guard";
import { applySingleAgentRequestPolicy, isSingleAgentModel, applySingleAgentCatalogPolicy } from "../../src/codex/model-capability-policy";
import { parseRequest } from "../../src/responses/parser";
import type { OcxConfig } from "../../src/types";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { validateConfigCandidate, configDiagnosticsFromRaw } from "../../src/config/diagnostics";
import { applyConfiguredAutoReviewModelOverride } from "../../src/codex/catalog/auto-review";
import * as responseState from "../../src/responses/state";
import { isNonReplayableResponse } from "../../src/lib/upstream-retry";
import { createTempHome, type TempHome } from "../helpers/temp-home";
import * as adapterResolve from "../../src/server/adapter-resolve";
import { fakeChatGptJwt } from "../helpers/agent-task-recovery";
import { mkdirSync } from "node:fs";

let releaseSpendHome: (() => void) | undefined;
let testHome: TempHome;
beforeEach(() => { testHome = createTempHome("ocx-single-agent-"); mkdirSync(testHome.codexHome); });
const takeSpendHome = (): void => { releaseSpendHome ??= acquireOwnedSpendHome(); };
afterEach(() => { releaseSpendHome?.(); releaseSpendHome = undefined; testHome.remove(); });
function sse(type: string, payload: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...payload })}\n\n`;
}

describe("single-agent tool delivery", () => {
  test("config admission and file diagnostics retain the policy and independent reviewer", () => {
    const configured = {
      port: 10100, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      providers: { fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key", autoReviewModelOverrides: { "local-model": "gpt-6.1-sol" } } },
    };
    const candidate = validateConfigCandidate(configured);
    expect(candidate.ok).toBe(true);
    if (!candidate.ok) throw new Error(candidate.error);
    expect(candidate.config.singleAgentModels).toEqual(configured.singleAgentModels);
    expect(configDiagnosticsFromRaw(JSON.stringify(configured)).config.singleAgentModels).toEqual(configured.singleAgentModels);
    expect(validateConfigCandidate({ ...configured, singleAgentModels: ["local-model"] }).ok).toBe(false);
    expect(validateConfigCandidate({ ...configured, singleAgentModels: "fixture/local-model" }).ok).toBe(false);
    const local = { slug: "fixture/local-model", auto_review_model_override: null, multi_agent_version: "v2" };
    const native = { slug: "gpt-6.1-sol", auto_review_model_override: null, multi_agent_version: "v2" };
    expect(applyConfiguredAutoReviewModelOverride([local, native], null, candidate.config)).toBe("applied");
    applySingleAgentCatalogPolicy([local, native], candidate.config);
    expect(local.auto_review_model_override).toBe("gpt-6.1-sol");
    expect(local.multi_agent_version).toBe("disabled");
    expect(native.auto_review_model_override).toBeNull();
    expect(native.multi_agent_version).toBe("v2");
  });

  test("reserved collaboration namespaces and blank legacy namespaces cannot reopen delegation", () => {
    for (const tool of [
      { type: "namespace", name: "collaboration", tools: [{ type: "function", name: "future_worker", parameters: { type: "object" } }] },
      { type: "function", name: "collaboration.future_worker", parameters: { type: "object" } },
      { type: "function", name: "spawn_agent", namespace: "", parameters: { type: "object" } },
    ]) {
      const parsed = parseRequest({ model: "fixture/local-model", input: "fixture", tools: [tool] });
      expect(applySingleAgentRequestPolicy(parsed)).toBeUndefined();
      expect((parsed._rawBody as { tools: unknown[] }).tools).toEqual([]);
    }
    expect(singleAgentResponseViolation({ status: "completed", output: [{ type: "function_call", id: "fc_worker", name: "future_worker", namespace: "collaboration" }] })).toBe("single_agent_delegation_disabled");
  });

  test("the catalog excludes only the selected routed worker and restores owned fields", () => {
    const local = { slug: "fixture/vendor-local", multi_agent_version: "v2", supports_parallel_tool_calls: true };
    const native = { slug: "gpt-6.1-sol", multi_agent_version: "v2", supports_parallel_tool_calls: true };
    const rows = [local, native];
    applySingleAgentCatalogPolicy(rows, { singleAgentModels: ["fixture/vendor/local"] });
    applySingleAgentCatalogPolicy(rows, { singleAgentModels: ["fixture/vendor/local"] });
    expect(local.multi_agent_version).toBe("disabled");
    expect(local.supports_parallel_tool_calls).toBe(false);
    expect(native).toEqual({ slug: "gpt-6.1-sol", multi_agent_version: "v2", supports_parallel_tool_calls: true });
    applySingleAgentCatalogPolicy(rows, {});
    expect(local).toEqual({ slug: "fixture/vendor-local", multi_agent_version: "v2", supports_parallel_tool_calls: true });
  });

  const call = { type: "function_call", id: "fc_one", call_id: "call_one", name: "exec_command", arguments: '{"cmd":"fixture"}', status: "completed" };
  const block = (value: unknown) => `data: ${JSON.stringify(value)}`;

  test("holds executable events while preserving text progress, then emits one complete call", () => {
    const rewrite = createSingleAgentToolGuardBlockRewrite();
    expect(rewrite(block({ type: "response.output_item.added", item: { ...call, arguments: "" } }))).toEqual([]);
    const text = block({ type: "response.output_text.delta", delta: "Inspecting" });
    expect(rewrite(text)).toEqual([text]);
    expect(rewrite(block({ type: "response.output_item.done", item: call }))).toEqual([]);
    const terminal = block({ type: "response.completed", response: { status: "completed", output: [call] } });
    const delivered = rewrite(terminal);
    expect(delivered).toHaveLength(2);
    expect(delivered[0]).toContain('"type":"response.output_item.done"');
    expect(delivered[0]).toContain('"call_id":"call_one"');
    expect(delivered[1]).toBe(terminal);
    rewrite.dispose?.();
  });

  test("a second call never releases the first executable event", () => {
    const rewrite = createSingleAgentToolGuardBlockRewrite();
    expect(rewrite(block({ type: "response.output_item.done", item: call }))).toEqual([]);
    const denied = rewrite(block({ type: "response.output_item.added", item: { ...call, id: "fc_two", call_id: "call_two" } }));
    expect(denied.join("\n")).toContain("single_agent_parallel_tools_disabled");
    expect(denied.join("\n")).not.toContain("response.output_item.done");
    expect(rewrite(block({ type: "response.completed", response: { output: [call] } }))).toEqual([]);
    rewrite.dispose?.();
  });

  test("sparse terminals, native delegation and incomplete batches fail before dispatch", () => {
    expect(singleAgentResponseViolation({ output: [call, { ...call, id: "fc_two" }] })).toBe("single_agent_parallel_tools_disabled");
    expect(singleAgentResponseViolation({ output: [{ ...call, name: "spawn_agent", namespace: "collaboration" }] })).toBe("single_agent_delegation_disabled");
    const rewrite = createSingleAgentToolGuardBlockRewrite();
    expect(rewrite(block({ type: "response.output_item.done", item: call }))).toEqual([]);
    const denied = rewrite("data: [DONE]").join("\n");
    expect(denied).toContain("single_agent_incomplete_tool_batch");
    expect(denied).not.toContain("response.output_item.done");
    rewrite.dispose?.();
  });

  test("cancellation releases no held call and disposed rewrites remain inert", () => {
    const rewrite = createSingleAgentToolGuardBlockRewrite();
    expect(rewrite(block({ type: "response.output_item.done", item: call }))).toEqual([]);
    rewrite.dispose?.();
    rewrite.dispose?.();
    expect(rewrite(block({ type: "response.completed", response: { output: [call] } }))).toEqual([]);
  });

  test("strips current, deferred and discovered native agent tools, retaining foreign namespaces and history", () => {
    const spawn = { type: "function", name: "spawn_agent", parameters: { type: "object" } };
    const shell = { type: "function", name: "exec_command", parameters: { type: "object" } };
    const foreign = { type: "namespace", name: "foreign", tools: [spawn] };
    const history = { type: "function_call", name: "spawn_agent", call_id: "old", arguments: '{"message":"historical"}' };
    const parsed = parseRequest({
      model: "fixture/local-model", tools: [spawn, shell, foreign],
      input: [history, { type: "additional_tools", role: "developer", tools: [spawn, shell] }, { type: "tool_search_output", call_id: "search", tools: [spawn, shell] }],
    });
    expect(applySingleAgentRequestPolicy(parsed)).toBeUndefined();
    const raw = parsed._rawBody as { tools: unknown[]; input: unknown[]; parallel_tool_calls: boolean };
    expect(raw.tools).toEqual([shell, foreign]);
    expect(raw.input[0]).toEqual(history);
    expect(raw.input[1]).toMatchObject({ tools: [shell] });
    expect(raw.input[2]).toMatchObject({ tools: [shell] });
    expect(raw.parallel_tool_calls).toBe(false);
  });

  test("forced delegation is refused without changing user intent or native model policy", () => {
    const parsed = parseRequest({ model: "fixture/local-model", tools: [{ type: "function", name: "spawn_agent", parameters: { type: "object" } }], tool_choice: { type: "function", name: "spawn_agent" }, input: "fixture" });
    expect(applySingleAgentRequestPolicy(parsed)).toContain("single_agent_delegation_disabled");
    expect(isSingleAgentModel({ singleAgentModels: ["fixture/local-model"] }, { providerName: "openai", modelId: "gpt-6.1-sol" })).toBe(false);
  });
});

describe("configured single-agent model admission", () => {
  const fixtureCall = { type: "function_call", id: "fc_one", call_id: "call_one", name: "exec_command", arguments: '{"cmd":"fixture"}', status: "completed" };
  test.each([
    ["one ordinary call", "fixture/local-model", ["exec_command"], 200],
    ["a forbidden native call", "fixture/local-model", ["spawn_agent"], 502],
    ["two ordinary calls", "fixture/local-model", ["exec_command", "exec_command"], 400],
    ["an unlisted model", "fixture/unrestricted", ["exec_command", "exec_command"], 200],
  ] as const)("buffered delivery: %s", async (_label, model, names, expectedStatus) => {
    const config = {
      port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      providers: { fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" } },
    } as OcxConfig;
    const output = names.map((name, i) => ({ type: "function_call", id: `fc_${i}`, call_id: `call_${i}`, name, arguments: '{"cmd":"fixture"}', status: "completed" }));
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async () => Response.json({ id: "resp_fixture", status: "completed", output })) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, stream: false, input: "fixture", tools: [{ type: "function", name: "exec_command", parameters: { type: "object" } }] }),
      }), config, { model: "", provider: "" });
      const delivered = await response.json() as { output?: unknown[]; error?: unknown };
      expect(response.status).toBe(expectedStatus);
      if (expectedStatus === 200) expect(delivered.output).toHaveLength(output.length);
      else expect(delivered.output).toBeUndefined();
    } finally { globalThis.fetch = savedFetch; }
  });

  test("an alias retains one complete streaming call and strict request fields", async () => {
    const config = {
      port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      providers: { fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key", models: ["local-model"], modelAliases: { "local-model": "local-alias" } } },
    } as OcxConfig;
    const call = { type: "function_call", id: "fc_one", call_id: "call_one", name: "exec_command", arguments: '{"cmd":"fixture"}', status: "completed" };
    const format = { type: "json_schema", name: "fixture_result", strict: true, schema: { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"], additionalProperties: false } };
    let sent: Record<string, unknown> | undefined;
    let completions = 0;
    const cache = spyOn(responseState, "rememberResponseState").mockImplementation(() => {});
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async (_url, init) => {
      sent = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response([
        sse("response.output_item.added", { output_index: 0, item: { ...call, arguments: "" } }),
        sse("response.function_call_arguments.delta", { item_id: "fc_one", output_index: 0, delta: call.arguments }),
        sse("response.output_item.done", { output_index: 0, item: call }),
        sse("response.completed", { response: { id: "resp_fixture", model: "local-model", status: "completed", output: [call] } }),
        "data: [DONE]\n\n",
      ].join(""), { headers: { "content-type": "text/event-stream" } });
    }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "local-alias", stream: true, input: "fixture", text: { format }, tools: [{ type: "function", name: "exec_command", parameters: { type: "object" } }] }),
      }), config, { model: "", provider: "" }, { onResponseComplete: () => { completions++; } });
      const delivered = await response.text();
      expect(response.status).toBe(200);
      expect(sent?.parallel_tool_calls).toBe(false);
      expect(sent?.text).toMatchObject({ format });
      expect(delivered.match(/"type":"response.output_item.done"/g)).toHaveLength(1);
      expect(delivered).toContain(call.call_id);
      expect(delivered).toContain('"type":"response.completed"');
      expect(delivered).not.toContain('"type":"response.function_call_arguments.delta"');
      expect(completions).toBe(1);
      expect(cache).toHaveBeenCalledTimes(1);
    } finally { cache.mockRestore(); globalThis.fetch = savedFetch; }
  });

  test.each(["openai-responses", "openai-chat"] as const)("the %s streaming pipeline refuses two ordinary calls before delivering either", async adapter => {
    const config = {
      port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      providers: { fixture: { adapter, baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" } },
    } as OcxConfig;
    const one = { type: "function_call", id: "fc_one", call_id: "call_one", name: "exec_command", arguments: '{"cmd":"fixture"}', status: "completed" };
    const two = { ...one, id: "fc_two", call_id: "call_two" };
    let sent: Record<string, unknown> | undefined;
    let completions = 0;
    const cache = spyOn(responseState, "rememberResponseState").mockImplementation(() => {});
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async (_url, init) => {
      sent = JSON.parse(String(init?.body)) as Record<string, unknown>;
      const nativeFrames = [
        sse("response.output_item.added", { output_index: 0, item: { ...one, arguments: "" } }),
        sse("response.output_item.done", { output_index: 0, item: one }),
        sse("response.output_item.added", { output_index: 1, item: { ...two, arguments: "" } }),
        sse("response.output_item.done", { output_index: 1, item: two }),
        sse("response.completed", { response: { id: "resp_batch", status: "completed", output: [one, two] } }),
        "data: [DONE]\n\n",
      ].join("");
      const translatedFrames = [
        { choices: [{ index: 0, delta: { tool_calls: [
          { index: 0, id: one.call_id, type: "function", function: { name: one.name, arguments: one.arguments } },
          { index: 1, id: two.call_id, type: "function", function: { name: two.name, arguments: two.arguments } },
        ] }, finish_reason: null }] },
        { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
      ].map(payload => `data: ${JSON.stringify(payload)}\n\n`).join("") + "data: [DONE]\n\n";
      return new Response(adapter === "openai-chat" ? translatedFrames : nativeFrames, { headers: { "content-type": "text/event-stream" } });
    }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "fixture/local-model", stream: true, input: "fixture", tools: [
          { type: "function", name: "spawn_agent", parameters: { type: "object" } },
          { type: "function", name: "exec_command", parameters: { type: "object", properties: { cmd: { type: "string" } } } },
        ] }),
      }), config, { model: "", provider: "" }, { onResponseComplete: () => { completions++; } });
      const delivered = await response.text();
      expect(sent?.parallel_tool_calls).toBe(false);
      expect(JSON.stringify(sent?.tools)).not.toContain("spawn_agent");
      expect(delivered).toContain("single_agent_parallel_tools_disabled");
      expect(delivered).not.toContain('"type":"response.output_item.done"');
      expect(delivered).not.toContain('"type":"response.completed"');
      expect(completions).toBe(0);
      expect(cache).not.toHaveBeenCalled();
    } finally { cache.mockRestore(); globalThis.fetch = savedFetch; }
  });

  test.each(["legacy", "metadata"])("refuses a spawned local turn before any inference or fallback send (%s)", async marker => {
    const config = {
      port: 0,
      defaultProvider: "fixture",
      subagentModelFallback: ["fixture/local-model", "backup/cloud-model"],
      singleAgentModels: ["fixture/local-model"],
      providers: {
        fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" },
        backup: { adapter: "openai-responses", baseUrl: "https://backup.test/v1", authMode: "key", apiKey: "fixture-key" },
      },
    } as OcxConfig;
    let sends = 0;
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      sends++;
      return Response.json({ id: "resp_local_child", status: "completed", output: [] });
    }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json", ...(marker === "legacy"
          ? { "x-openai-subagent": "collab_spawn" }
          : { "x-codex-turn-metadata": JSON.stringify({ subagent_kind: "thread_spawn" }) }) },
        body: JSON.stringify({ model: "fixture/local-model", stream: false, input: "synthetic child assignment" }),
      }), config, { model: "", provider: "" });
      const body = await response.text();
      expect(sends).toBe(0);
      expect(response.ok).toBe(false);
      expect(body).toContain("single_agent_worker_disabled");
      expect(response.status).toBe(400);
      expect(JSON.parse(body).error.code).toBe("single_agent_worker_disabled");
      expect(isNonReplayableResponse(response)).toBe(true);
    } finally {
      globalThis.fetch = savedFetch;
    }
  });

  test("a native reviewer failure preserves strict format and never dispatches to the local model", async () => {
    const config = { port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      subagentModelFallback: ["fixture/local-model"], providers: {
        fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key", autoReviewModelOverrides: { "local-model": "gpt-6.1-sol" } },
        openai: { adapter: "openai-responses", baseUrl: "https://chatgpt.com/backend-api/codex", authMode: "forward", codexAccountMode: "direct" },
      } } as OcxConfig;
    const format = { type: "json_schema", name: "permission_result", strict: true, schema: {
      type: "object", properties: { allowed: { type: "boolean" } }, required: ["allowed"], additionalProperties: false,
    } };
    const sends: string[] = [];
    let sent: Record<string, unknown> | undefined;
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async (url, init) => {
      sends.push(String(url)); sent = JSON.parse(String(init?.body));
      return Response.json({ error: { type: "invalid_request_error", code: "fixture_reviewer_unavailable", param: "text.format", message: "Synthetic reviewer refusal" } }, { status: 400 });
    }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${fakeChatGptJwt("fixture-reviewer")}`, "chatgpt-account-id": "fixture-reviewer" },
        body: JSON.stringify({ model: "gpt-6.1-sol", stream: false, input: "synthetic permission review", text: { format }, parallel_tool_calls: true,
          tools: [{ type: "function", name: "exec_command", parameters: { type: "object" } }] }),
      }), config, { model: "", provider: "" });
      expect(response.status).toBe(400);
      expect(await response.text()).toContain("fixture_reviewer_unavailable");
      expect(sends).toHaveLength(1);
      expect(sends[0]).toContain("chatgpt.com");
      expect(sent?.text).toMatchObject({ format });
      expect(sent?.parallel_tool_calls).toBe(true);
    } finally { globalThis.fetch = savedFetch; }
  });

  test("native collaboration admission keeps its forwarded declarations and output", async () => {
    const config = { port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"], providers: {
      fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" },
      openai: { adapter: "openai-responses", baseUrl: "https://chatgpt.com/backend-api/codex", authMode: "forward", codexAccountMode: "direct" },
    } } as OcxConfig;
    const nativeTool = { type: "namespace", name: "collaboration", tools: [{ type: "function", name: "spawn_agent", parameters: { type: "object" } }] };
    const nativeCall = { ...fixtureCall, name: "spawn_agent", namespace: "collaboration" };
    const sends: string[] = [];
    let sent: Record<string, unknown> | undefined;
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async (url, init) => {
      sends.push(String(url)); sent = JSON.parse(String(init?.body));
      return Response.json({ id: "resp_native_collaboration", model: "gpt-6.1-sol", status: "completed", output: [nativeCall] });
    }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${fakeChatGptJwt("fixture-native")}`, "chatgpt-account-id": "fixture-native", "x-openai-subagent": "collab_spawn" },
        body: JSON.stringify({ model: "gpt-6.1-sol", stream: false, input: "synthetic native collaboration", tools: [nativeTool], parallel_tool_calls: true }),
      }), config, { model: "", provider: "" });
      expect(response.status).toBe(200);
      expect((await response.json() as { output: unknown[] }).output).toEqual([nativeCall]);
      expect(sends).toHaveLength(1);
      expect(sends[0]).toContain("chatgpt.com");
      expect(sent?.tools).toEqual([nativeTool]);
      expect(sent?.parallel_tool_calls).toBe(true);
    } finally { globalThis.fetch = savedFetch; }
  });

  test.each([false, true])("a forbidden call cannot move a combo onto a replacement cloud target (stream=%s)", async stream => {
    const config = {
      port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      providers: {
        fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" },
        backup: { adapter: "openai-responses", baseUrl: "https://backup.test/v1", authMode: "key", apiKey: "fixture-key" },
      },
      combos: { policy: { strategy: "failover", targets: [{ provider: "fixture", model: "local-model" }, { provider: "backup", model: "cloud-model" }] } },
    } as OcxConfig;
    const sends: string[] = [];
    const savedFetch = globalThis.fetch;
    globalThis.fetch = (async input => {
      sends.push(String(input));
      const terminal = { id: "resp_forbidden_combo", model: "local-model", status: "completed", output: [{ ...fixtureCall, name: "spawn_agent" }] };
      return stream
        ? new Response(sse("response.completed", { response: terminal }), { headers: { "content-type": "text/event-stream" } })
        : Response.json(terminal);
    }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "combo/policy", stream, input: "fixture", tools: [{ type: "function", name: "exec_command", parameters: { type: "object" } }] }),
      }), config, { model: "", provider: "" });
      await response.text();
      expect(sends).toHaveLength(1);
      expect(sends[0]).toContain("fixture.test");
      expect(isNonReplayableResponse(response)).toBe(true);
      expect(response.ok).toBe(false);
    } finally { globalThis.fetch = savedFetch; }
  });

  test.each(["chat", "anthropic"] as const)("unsupported %s ingress is refused before sending", async inboundWire => {
    const config = { port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"], providers: {
      fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" },
    } } as OcxConfig;
    const savedFetch = globalThis.fetch;
    const transport = mock(async () => Response.json({ status: "completed", output: [] }));
    globalThis.fetch = transport as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", body: JSON.stringify({ model: "fixture/local-model", input: "fixture" }),
      }), config, { model: "", provider: "" }, { inboundWire });
      expect(response.status).toBe(400);
      expect(await response.text()).toContain("single_agent_unsupported_transport");
      expect(isNonReplayableResponse(response)).toBe(true);
      expect(transport).not.toHaveBeenCalled();
    } finally { globalThis.fetch = savedFetch; }
  });

  test("an adapter-owned loop is refused before execute or send", async () => {
    const config = { port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"], providers: {
      fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" },
    } } as OcxConfig;
    const execute = mock(async function* () { throw new Error("owned loop must never execute"); });
    const resolved = adapterResolve.resolveAdapter(config.providers.fixture!);
    const adapter = spyOn(adapterResolve, "resolveAdapter").mockReturnValue({ ...resolved, runTurn: execute });
    const savedFetch = globalThis.fetch;
    const transport = mock(async () => Response.json({ status: "completed", output: [] }));
    globalThis.fetch = transport as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", body: JSON.stringify({ model: "fixture/local-model", input: "fixture" }),
      }), config, { model: "", provider: "" });
      expect(response.status).toBe(400);
      expect(await response.text()).toContain("single_agent_unsupported_transport");
      expect(execute).not.toHaveBeenCalled();
      expect(transport).not.toHaveBeenCalled();
    } finally { adapter.mockRestore(); globalThis.fetch = savedFetch; }
  });

  test.each(["identity mismatch", "forbidden call then valid terminal"])("SSE %s cannot publish a successful completion or cache", async kind => {
    const config = {
      port: 0, defaultProvider: "fixture", singleAgentModels: ["fixture/local-model"],
      providers: { fixture: { adapter: "openai-responses", baseUrl: "https://fixture.test/v1", authMode: "key", apiKey: "fixture-key" } },
    } as OcxConfig;
    const first = kind === "identity mismatch" ? fixtureCall : { ...fixtureCall, name: "spawn_agent" };
    const terminalCall = kind === "identity mismatch" ? { ...fixtureCall, id: "fc_other", call_id: "call_other" } : fixtureCall;
    const wire = sse("response.output_item.done", { item: first })
      + sse("response.completed", { response: { id: "resp_policy_mismatch", model: "local-model", status: "completed", output: [terminalCall] } });
    let completions = 0;
    let sends = 0;
    const savedFetch = globalThis.fetch;
    const cache = spyOn(responseState, "rememberResponseState").mockImplementation(() => {});
    globalThis.fetch = (async () => { sends++; return new Response(wire, { headers: { "content-type": "text/event-stream" } }); }) as typeof fetch;
    try {
      takeSpendHome();
      const response = await handleResponses(new Request("http://localhost/v1/responses", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "fixture/local-model", stream: true, input: "fixture", store: true, tools: [{ type: "function", name: "exec_command", parameters: { type: "object" } }] }),
      }), config, { model: "", provider: "" }, { onResponseComplete: () => { completions++; } });
      const delivered = await response.text();
      expect(delivered).toContain("response.failed");
      expect(delivered).not.toContain('"type":"response.output_item.done"');
      expect(delivered).not.toContain('"type":"response.completed"');
      expect(completions).toBe(0);
      expect(cache).not.toHaveBeenCalled();
      expect(sends).toBe(1);
    } finally { cache.mockRestore(); globalThis.fetch = savedFetch; }
  });
});
