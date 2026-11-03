import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PersistedCooldownMap } from "../../src/combos/cooldown-store";
import { clearComboTargetCooldowns, coolComboTarget, isComboTargetInCooldown } from "../../src/combos/failover";
import { classifyFailure, classifySoftSuccess, RECOVERY_ACTION, type FailureClass } from "../../src/routing/failure-taxonomy";

let home: string;
let previous: string | undefined;
beforeEach(() => {
  previous = process.env.OPENCODEX_HOME;
  home = mkdtempSync(join(tmpdir(), "ocx-cooldown-"));
  process.env.OPENCODEX_HOME = home;
});
afterEach(() => {
  clearComboTargetCooldowns();
  if (previous === undefined) delete process.env.OPENCODEX_HOME; else process.env.OPENCODEX_HOME = previous;
  rmSync(home, { recursive: true, force: true });
});

test("every failure class has exactly one recovery action", () => {
  const classes: FailureClass[] = ["rate_limit", "quota_exhausted", "auth", "model_entitlement", "timeout", "network", "protocol", "overloaded", "request_shape", "unknown"];
  for (const c of classes) expect(typeof RECOVERY_ACTION[c]).toBe("string");
  expect(Object.keys(RECOVERY_ACTION).sort()).toEqual([...classes].sort());
});

test("classification separates the classes", () => {
  expect(classifyFailure({ status: 429, message: "Rate limit reached for requests" })).toBe("rate_limit");
  expect(classifyFailure({ status: 429, code: "insufficient_quota" })).toBe("quota_exhausted");
  expect(classifyFailure({ status: 502, message: "The usage limit has been reached" })).toBe("quota_exhausted");
  expect(classifyFailure({ status: 401 })).toBe("auth");
  expect(classifyFailure({ status: 403, message: "model is not available for this account" })).toBe("model_entitlement");
  expect(classifyFailure({ timedOut: true })).toBe("timeout");
  expect(classifyFailure({ transportError: true })).toBe("network");
  expect(classifyFailure({ status: 503 })).toBe("overloaded");
  expect(classifyFailure({ status: 400, code: "context_length_exceeded" })).toBe("request_shape");
});

test("an HTTP 200 body that only carries a router error or timeout is a failure", () => {
  expect(classifySoftSuccess({ error: { message: "Request timed out", code: "timeout" } })).toBe("timeout");
  expect(classifySoftSuccess({ error: { message: "upstream router error" } })).toBe("protocol");
  expect(classifySoftSuccess({ type: "error", error: { type: "overloaded_error", message: "Overloaded" } })).toBe("protocol");
  expect(classifySoftSuccess({ choices: [{ message: { content: "ok" } }] })).toBeNull();
  expect(classifySoftSuccess({ output: [{ type: "message" }], error: null })).toBeNull();
});

test("a combo cooldown is written to disk and reloaded by a fresh map after a restart", () => {
  expect(coolComboTarget("combo-a", { provider: "Codex", model: "gpt-6.1-sol" }, { retryAfter: "600", status: 429 })).toBe(true);
  const store = new PersistedCooldownMap<{ cooldownUntil: number; revision: number; status?: number }>(() => 1);
  // The production write is debounced by 250 ms; wait it out, then load the file with a fresh map.
  const written = join(home, "combo-cooldowns.json");
  const { setTimeout: wait } = require("node:timers/promises") as typeof import("node:timers/promises");
  return wait(400).then(() => {
    const rows = JSON.parse(readFileSync(written, "utf8")) as Record<string, { cooldownUntil: number; status?: number }>;
    const key = Object.keys(rows).find(k => k.includes("Codex/gpt-6.1-sol"))!;
    expect(rows[key]!.cooldownUntil).toBeGreaterThan(Date.now() + 500_000);
    expect(rows[key]!.status).toBe(429);
    expect(store.has(key)).toBe(true);
    expect(isComboTargetInCooldown("combo-a", { provider: "Codex", model: "gpt-6.1-sol" })).toBe(true);
  });
});
