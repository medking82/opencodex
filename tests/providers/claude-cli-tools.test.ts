import { beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import { existsSync, readFileSync } from "node:fs";
import { PassThrough, Readable, Writable } from "node:stream";
import type { ChildProcess } from "node:child_process";
import { createClaudeCliAdapter } from "../../src/adapters/claude-cli/adapter";
import { CLAUDE_CLI_PROFILE, clearClaudeCliBinaryCache } from "../../src/adapters/claude-cli/profiles";
import { buildResponseJSON } from "../../src/bridge/response-json";
import type { AdapterEvent, OcxParsedRequest, OcxProviderConfig } from "../../src/types";
import { createTestTranslatorBudget } from "../helpers/translator-budget";

beforeEach(() => clearClaudeCliBinaryCache());
const provider = { adapter: "claude-cli", baseUrl: CLAUDE_CLI_PROFILE.canonicalBaseUrl } as OcxProviderConfig;
const init = { type: "system", subtype: "init", mcp_servers: [{ name: "opencodex", status: "connected" }] };
const stop = { type: "stream_event", event: { type: "message_stop" } };
const terminal = { type: "result", subtype: "success", is_error: false, result: "done" };
const tool = { name: "lookup", description: "Lookup fixture", parameters: { type: "object", properties: { value: { type: "string" } } } };
function request(): OcxParsedRequest {
  return { modelId: "claude-opus-5-5", stream: true, options: {},
    context: { messages: [{ role: "user", content: "Call lookup", timestamp: 0 }], tools: [tool] } } as OcxParsedRequest;
}
function childWith(frames: unknown[], parked = false) {
  const child = new EventEmitter() as ChildProcess;
  const stdout = new PassThrough();
  for (const frame of frames) stdout.write(JSON.stringify(frame) + "\n");
  if (!parked) stdout.end();
  child.stdout = stdout;
  child.stderr = Readable.from([]);
  let written = "";
  child.stdin = new Writable({ write(chunk, _encoding, cb) { written += String(chunk); cb(); } });
  child.killed = false;
  child.exitCode = null;
  child.kill = () => {
    child.killed = true;
    child.exitCode = 0;
    queueMicrotask(() => { stdout.destroy(); child.emit("close", 0); });
    return true;
  };
  if (!parked) setTimeout(() => { child.exitCode = 0; child.emit("close", 0); }, 3);
  return { child, written: () => written };
}
const call = (name = "lookup", id = "native-id") => ({ type: "tool_use", id, name: `mcp__opencodex__${name}`, input: { value: "fixture" } });
const complete = (blocks = [call()]) => ({ type: "assistant", message: { stop_reason: "tool_use", content: blocks } });
async function run(frames: unknown[], parsed = request(), parked = false) {
  const fake = childWith(frames, parked);
  const paths: string[] = [];
  let argsSeen: readonly string[] = [];
  let catalog: unknown;
  const adapter = createClaudeCliAdapter(provider, { which: () => "/test/claude", killGraceMs: 10,
    spawn: (_binary, args) => {
      argsSeen = args;
      const prompt = args[args.indexOf("--system-prompt-file") + 1]!;
      paths.push(prompt);
      if (args.includes("--mcp-config")) {
        const configPath = args[args.indexOf("--mcp-config") + 1]!;
        const config = JSON.parse(readFileSync(configPath, "utf8"));
        expect(Object.keys(config.mcpServers)).toEqual(["opencodex"]);
        const catalogPath = config.mcpServers.opencodex.args.at(-1);
        catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
        paths.push(configPath, catalogPath);
      }
      return fake.child;
    } });
  const events: AdapterEvent[] = [];
  const budget = createTestTranslatorBudget();
  await adapter.runTurn!(parsed, { headers: new Headers(), translatorBudget: budget }, e => events.push(e));
  expect(paths.every(path => !existsSync(path))).toBe(true);
  expect(budget.snapshot()).toMatchObject({ currentBytes: 0, activeCalls: 0 });
  return { ...fake, events, args: argsSeen, catalog };
}

describe("Claude CLI capture-only client tools", () => {
  test("captures structured calls, preserves native IDs, and keeps exact isolated permissions", async () => {
    const result = await run([init, complete(), stop], request(), true);
    expect(result.args[result.args.indexOf("--tools") + 1]).toBe("");
    expect(result.args).toContain("--strict-mcp-config");
    expect(result.args[result.args.indexOf("--setting-sources") + 1]).toBe("");
    expect(result.args).not.toContain("--dangerously-skip-permissions");
    expect(result.args).not.toContain("--permission-mode");
    expect(result.args[result.args.indexOf("--allowedTools") + 1]).toBe("mcp__opencodex__lookup");
    expect(result.catalog).toEqual([{ name: "lookup", description: tool.description, inputSchema: tool.parameters }]);
    expect(result.events).toContainEqual({ type: "tool_call_start", id: "native-id", name: "lookup" });
    expect(result.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use", endTurn: false });
    expect(buildResponseJSON(result.events, "fixture").output).toContainEqual(expect.objectContaining({
      type: "function_call", call_id: "native-id", name: "lookup", arguments: '{"value":"fixture"}',
    }));
    expect(result.child.killed).toBe(true);
  });

  test("partial streaming plus complete snapshot emits a single tool call", async () => {
    const partial = [
      { type: "stream_event", event: { type: "content_block_start", index: 0, content_block: { ...call(), input: {} } } },
      { type: "stream_event", event: { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '{"value":"fixture"}' } } },
      { type: "stream_event", event: { type: "content_block_stop", index: 0 } },
    ];
    const result = await run([init, ...partial, complete(), stop], request(), true);
    expect(result.events.filter(e => e.type === "tool_call_start")).toHaveLength(1);
    expect(result.events.filter(e => e.type === "tool_call_end")).toHaveLength(1);
    expect(result.events.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use" });
  });

  test("preserves namespace wire identity and replayed result without executing history", async () => {
    const parsed = request();
    parsed.context.tools = [{ ...tool, namespace: "functions" }];
    parsed.context.messages = [
      { role: "assistant", content: [{ type: "toolCall", id: "previous", name: "functions__lookup", arguments: { value: "old" } }], timestamp: 0 },
      { role: "toolResult", toolCallId: "previous", content: "RECEIPT-7291", timestamp: 1 },
    ];
    const result = await run([init, complete([call("functions__lookup")]), stop], parsed, true);
    expect(result.events).toContainEqual({ type: "tool_call_start", id: "native-id", name: "functions__lookup" });
    expect(result.written()).toContain("mcp__opencodex__functions__lookup");
    expect(result.written()).toContain("RECEIPT-7291");
  });

  test("custom-tool wire input restores to a Responses custom_tool_call", async () => {
    const parsed = request();
    parsed.context.tools = [{ name: "exec", namespace: "functions", description: "Code input",
      parameters: { type: "object", properties: { input: { type: "string" } }, required: ["input"] } }];
    const block = { type: "tool_use", id: "custom-id", name: "mcp__opencodex__functions__exec", input: { input: "print(7291)" } };
    const result = await run([init, complete([block as ReturnType<typeof call>]), stop], parsed, true);
    const response = buildResponseJSON(result.events, "fixture", {
      toolNsMap: new Map([["functions__exec", { namespace: "functions", name: "exec", freeform: true }]]),
      freeformToolNames: new Set(["functions__exec"]),
    });
    expect(response.output).toContainEqual(expect.objectContaining({ type: "custom_tool_call",
      call_id: "custom-id", namespace: "functions", name: "exec", input: "print(7291)" }));
  });
  test("tool_choice none advertises no MCP catalog", async () => {
    const parsed = request();
    parsed.options.toolChoice = "none";
    const result = await run([terminal], parsed);
    expect(result.args).not.toContain("--mcp-config");
    expect(result.catalog).toBeUndefined();
    expect(result.events.at(-1)).toMatchObject({ type: "done" });
  });

  test("required tool cannot be replaced by text or simulated XML", async () => {
    const parsed = request();
    parsed.options.toolChoice = "required";
    const result = await run([init, { ...terminal, result: '<invoke name="lookup">fixture</invoke>' }], parsed);
    expect(result.events.at(-1)).toMatchObject({ type: "error", code: "tool_call_required" });
    expect(result.events.some(e => e.type === "done" || e.type === "tool_call_start")).toBe(false);
  });

  test("invalid catalogs fail before spawning or exposing argument text", async () => {
    const parsed = request();
    parsed.context.tools = [{ ...tool, name: "bad,permission" }];
    const result = await run([], parsed);
    expect(result.args).toEqual([]);
    expect(result.events).toEqual([expect.objectContaining({ type: "error", status: 400, code: "tool_catalog_invalid" })]);
  });

  test("foreign init and undeclared calls fail closed", async () => {
    for (const frames of [
      [{ ...init, mcp_servers: [{ name: "foreign", status: "connected" }] }, complete(), stop],
      [init, complete([call("undeclared")]), stop],
      [complete(), stop],
    ]) {
      const result = await run(frames, request(), true);
      expect(result.events.at(-1)?.type).toBe("error");
      expect(result.events.some(e => e.type === "done" || e.type === "tool_call_start")).toBe(false);
      expect(result.child.killed).toBe(true);
    }
  });

  test("abort reaps parked child and removes staged catalog and prompt", async () => {
    const abort = new AbortController();
    const fake = childWith([init], true);
    const paths: string[] = [];
    const adapter = createClaudeCliAdapter(provider, { which: () => "/test/claude", killGraceMs: 10,
      spawn: (_binary, args) => {
        paths.push(args[args.indexOf("--mcp-config") + 1]!, args[args.indexOf("--system-prompt-file") + 1]!);
        setTimeout(() => abort.abort(), 5);
        return fake.child;
      } });
    const events: AdapterEvent[] = [];
    await adapter.runTurn!(request(), { headers: new Headers(), abortSignal: abort.signal, translatorBudget: createTestTranslatorBudget() }, e => events.push(e));
    expect(fake.child.killed).toBe(true);
    expect(paths.every(path => !existsSync(path))).toBe(true);
    expect(events.some(e => e.type === "done")).toBe(false);
  });
});