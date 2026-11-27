export function createRequestCancellation({ req, res, timeoutMs = null } = {}) {
  const controller = new AbortController();
  const now = () => globalThis.performance?.now?.() ?? Date.now();
  let session = null;
  let closePromise = null;
  let reason = null;
  let sessionBoundAt = null;
  let cancelRequestedAt = null;
  let sessionCloseStartedAt = null;
  let sessionCloseCompletedAt = null;

  const closeSession = () => {
    if (!session || closePromise) return closePromise;
    sessionCloseStartedAt = now();
    closePromise = Promise.resolve(session.close()).catch(() => {}).finally(() => {
      sessionCloseCompletedAt = now();
    });
    return closePromise;
  };
  const cancel = (nextReason = "cancelled") => {
    if (controller.signal.aborted) return closePromise;
    reason = nextReason;
    cancelRequestedAt = now();
    controller.abort(nextReason);
    closeSession();
    return closePromise;
  };
  const onAborted = () => cancel("client_disconnect");
  const onClose = () => {
    if (!res.writableEnded) cancel("client_disconnect");
  };
  req?.once?.("aborted", onAborted);
  res?.once?.("close", onClose);
  const timer = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? setTimeout(() => cancel("deadline_exceeded"), timeoutMs)
    : null;
  timer?.unref?.();

  return {
    signal: controller.signal,
    get reason() { return reason; },
    get timings() {
      return {
        sessionBoundAt,
        cancelRequestedAt,
        sessionCloseStartedAt,
        sessionCloseCompletedAt,
        cancelToCloseStartMs: cancelRequestedAt == null || sessionCloseStartedAt == null
          ? null
          : Math.max(0, sessionCloseStartedAt - cancelRequestedAt),
        cancelToCloseCompleteMs: cancelRequestedAt == null || sessionCloseCompletedAt == null
          ? null
          : Math.max(0, sessionCloseCompletedAt - cancelRequestedAt),
      };
    },
    bindSession(nextSession) {
      session = nextSession;
      sessionBoundAt = now();
      if (controller.signal.aborted) closeSession();
    },
    cancel,
    async dispose() {
      if (timer) clearTimeout(timer);
      req?.off?.("aborted", onAborted);
      res?.off?.("close", onClose);
      await closePromise;
    },
  };
}
