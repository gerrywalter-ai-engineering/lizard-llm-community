// Shared local-request hardening for BOTH port-5567 servers
// (scripts/benchmark-dashboard.ts and src/ui/server.js) — Phase 6 of
// .Agent/auth_phases.md. Applied before route-specific logic.
//
// Two independent protections:
//
//  1. Loopback hardening (every request): the dashboard binds only to
//     127.0.0.1, but a malicious web page can still point a DNS name at
//     127.0.0.1 and try to drive it from the victim's browser
//     (DNS-rebinding). We therefore require the Host header to be an expected
//     loopback authority and never emit wildcard CORS.
//
//  2. Mutation protection (state-changing methods): a valid local session,
//     a matching CSRF token, an exact same-origin/host Origin (when present),
//     and a JSON content-type. The classification of which routes are
//     public / community / paid / local-admin lives in route-classes.js.
//
// Enforcement is gated by config.auth.enabled: while auth is disabled
// (default, Phase 9 dark rollout) loopback hardening still runs but session/
// CSRF checks are skipped so current behavior is unchanged.

const MUTATING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

// The DNS-rebinding defense rejects HOSTNAMES, not ports: an attacker makes
// `evil.example` resolve to 127.0.0.1 and the browser sends `Host: evil.example`.
// So the check is that the host authority is a loopback IP literal (or the
// reserved `localhost`, which an attacker cannot own) on any port. The dashboard
// still binds only to 127.0.0.1, so the port is free to vary (5567, test ports).
const LOOPBACK_HOSTNAMES = new Set(["127.0.0.1", "::1", "localhost"]);

function loopbackHostname(hostAuthority) {
  if (!hostAuthority) return null;
  const value = String(hostAuthority).trim().toLowerCase();
  // IPv6 literal: [::1] or [::1]:port
  const v6 = value.match(/^\[([0-9a-f:]+)\](?::\d+)?$/);
  if (v6) return v6[1];
  // hostname[:port]
  const host = value.split(":")[0];
  return host || null;
}

/**
 * @param {object} options
 * @param {number} [options.port] retained for signature compatibility (unused
 *   for host matching — any loopback port is accepted).
 * @param {string[]} [options.allowedHostnames] override the loopback set.
 */
export function createLoopbackGuard({ port = 7817, allowedHostnames = null } = {}) {
  void port;
  const hostnames = new Set((allowedHostnames ?? [...LOOPBACK_HOSTNAMES]).map((h) => h.toLowerCase()));

  function isAllowedHost(hostHeader) {
    if (!hostHeader) {
      // A missing Host header is HTTP/1.0-ish and not something a browser
      // fetch produces; reject to be safe.
      return false;
    }
    const hostname = loopbackHostname(hostHeader);
    return Boolean(hostname) && hostnames.has(hostname);
  }

  // The Origin, when present, must be a loopback origin. Non-browser clients
  // (curl, the desktop shell) omit Origin and pass the Origin check — they
  // cannot be driven by a hostile web page.
  function isAllowedOrigin(origin) {
    if (!origin) return true;
    let parsed;
    try {
      parsed = new URL(origin);
    } catch {
      return false;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false;
    const hostname = parsed.hostname.replace(/^\[|\]$/g, "").toLowerCase();
    return hostnames.has(hostname);
  }

  /**
   * Check a request. Returns { ok:true } or { ok:false, status, code, message }.
   * `session` and `csrfOk` are supplied by the caller (they come from the auth
   * runtime's session manager); pass enforceMutations=false to skip the
   * session/CSRF layer while auth is disabled.
   */
  function check({ method, headers = {}, requiresSession = false, session = null, csrfOk = false, enforceMutations = true }) {
    const host = headers.host;
    if (!isAllowedHost(host)) {
      return { ok: false, status: 403, code: "BAD_HOST", message: "Unexpected Host header (loopback only)." };
    }
    if (!isAllowedOrigin(headers.origin)) {
      return { ok: false, status: 403, code: "BAD_ORIGIN", message: "Cross-origin request refused." };
    }
    const isMutation = MUTATING_METHODS.has(String(method).toUpperCase());
    if (isMutation && enforceMutations) {
      // Content-type must be JSON for API mutations: blocks simple-form
      // cross-site POSTs (which cannot set application/json without a
      // preflight the loopback server never approves).
      const contentType = String(headers["content-type"] ?? "");
      if (headers["content-length"] && headers["content-length"] !== "0" && !contentType.includes("application/json")) {
        return { ok: false, status: 415, code: "UNSUPPORTED_MEDIA_TYPE", message: "Mutations must send application/json." };
      }
      if (requiresSession) {
        if (!session) {
          return { ok: false, status: 401, code: "AUTH_REQUIRED", message: "A valid local session is required." };
        }
        if (!csrfOk) {
          return { ok: false, status: 403, code: "CSRF_INVALID", message: "Missing or invalid CSRF token." };
        }
      }
    }
    return { ok: true };
  }

  return { check, isAllowedHost, isAllowedOrigin, allowedHostnames: [...hostnames] };
}
