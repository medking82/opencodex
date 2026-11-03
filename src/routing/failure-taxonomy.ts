/**
 * Failure taxonomy: one class per failure, exactly one recovery action per class.
 *
 * Adopted from Hermes-agent (MIT, NousResearch) error classification: a router must decide from the
 * class alone whether to wait, rotate the credential, hop to the next target, or give up, never from
 * ad-hoc string matches at each call site. Also adopts its rule that an HTTP 200 whose body only
 * carries a router error or timeout is a failure (`classifySoftSuccess`).
 */
export type FailureClass =
  | "rate_limit"          // short request-rate limit: wait out Retry-After, then retry the same target
  | "quota_exhausted"     // spent window: cool the target until its reset, hop now
  | "auth"                // credential rejected: rotate/refresh the credential, then retry once
  | "model_entitlement"   // this account may not use this model: hop, and cool this (account, model)
  | "timeout"             // no/stalled response: retry once on the same target, then hop
  | "network"             // connection failure: hop now, short cooldown
  | "protocol"            // malformed or router-error body: hop now, do not cool the target
  | "overloaded"          // 5xx / 529: hop now, short cooldown
  | "request_shape"       // this request is invalid for the target: stop, return the error (no hop)
  | "unknown";            // anything else: hop once, short cooldown

export type RecoveryAction = "wait_retry" | "cool_and_hop" | "rotate_credential" | "retry_then_hop" | "hop" | "stop";

export const RECOVERY_ACTION: Readonly<Record<FailureClass, RecoveryAction>> = {
  rate_limit: "wait_retry",
  quota_exhausted: "cool_and_hop",
  auth: "rotate_credential",
  model_entitlement: "cool_and_hop",
  timeout: "retry_then_hop",
  network: "hop",
  protocol: "hop",
  overloaded: "hop",
  request_shape: "stop",
  unknown: "hop",
};

export interface FailureInput {
  status?: number;
  code?: string | null;
  message?: string;
  /** Set when the transport failed before any status (connect error, reset). */
  transportError?: boolean;
  timedOut?: boolean;
}

const QUOTA_CODES = new Set(["insufficient_quota", "usage_limit_exceeded", "usage_limit_reached", "1308", "billing_error", "insufficient_balance", "payment_required"]);
const ENTITLEMENT = /not (?:entitled|available|supported) (?:for|on) (?:this )?(?:account|plan)|model[^.]{0,40}(?:not found|does not exist|not available)|does not have access to (?:the )?model/i;
const QUOTA_TEXT = /usage limit (?:has been )?reached|quota (?:has been )?(?:exceeded|exhausted)|out of credits|weekly limit/i;
const SHAPE_CODES = new Set(["context_length_exceeded", "input_admission_refused", "tool_catalog_too_large", "unsupported_input_modality", "invalid_request_error"]);

export function classifyFailure(input: FailureInput): FailureClass {
  const code = (input.code ?? "").trim().toLowerCase().replaceAll("-", "_");
  const text = input.message ?? "";
  if (input.timedOut || input.status === 408 || input.status === 504 || code === "timeout_error") return "timeout";
  if (input.transportError && input.status === undefined) return "network";
  if (QUOTA_CODES.has(code) || (input.status !== 400 && input.status !== 422 && QUOTA_TEXT.test(text))) return "quota_exhausted";
  if (input.status === 401 || input.status === 403 && !ENTITLEMENT.test(text)) return "auth";
  if (ENTITLEMENT.test(text) || input.status === 404 && /model/i.test(text)) return "model_entitlement";
  if (input.status === 429) return "rate_limit";
  if (input.status !== undefined && (input.status >= 500 || input.status === 529)) return "overloaded";
  if (SHAPE_CODES.has(code) || input.status === 400 || input.status === 413 || input.status === 422) return "request_shape";
  if (input.status === undefined) return "protocol";
  return "unknown";
}

/**
 * HTTP 200 is not success when the body only carries a router error or a router timeout
 * (for example `{"error": {...}}`, `{"type":"error"}`, or a bare timeout notice with no output).
 * Returns the failure class, or null when the body looks like a real completion.
 */
export function classifySoftSuccess(body: unknown): FailureClass | null {
  if (body === null || typeof body !== "object" || Array.isArray(body)) return null;
  const rec = body as Record<string, unknown>;
  const hasOutput = Array.isArray(rec.choices) && rec.choices.length > 0
    || Array.isArray(rec.output) && rec.output.length > 0
    || Array.isArray(rec.content) && rec.content.length > 0;
  if (hasOutput) return null;
  const err = rec.error ?? (rec.type === "error" ? rec : undefined);
  if (err === undefined || err === null || err === false) return null;
  const e = (typeof err === "object" ? err : { message: String(err) }) as Record<string, unknown>;
  const message = typeof e.message === "string" ? e.message : typeof rec.message === "string" ? rec.message : "";
  const code = typeof e.code === "string" ? e.code : typeof e.type === "string" ? e.type : undefined;
  const status = typeof e.status === "number" ? e.status : typeof e.status_code === "number" ? e.status_code : undefined;
  if (/time(?:d)?[- ]?out|timed out|deadline exceeded/i.test(message)) return "timeout";
  const cls = classifyFailure({ status, code, message });
  return cls === "unknown" ? "protocol" : cls;
}
