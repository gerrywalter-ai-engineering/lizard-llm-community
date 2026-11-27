export async function writeWithBackpressure(res, chunk, signal = null) {
  if (signal?.aborted || res.destroyed || res.writableEnded) return false;
  if (res.write(chunk)) return true;
  await new Promise((resolve) => {
    const finish = () => {
      res.off("drain", finish);
      res.off("close", finish);
      res.off("error", finish);
      signal?.removeEventListener?.("abort", finish);
      resolve();
    };
    res.once("drain", finish);
    res.once("close", finish);
    res.once("error", finish);
    signal?.addEventListener?.("abort", finish, { once: true });
  });
  return !signal?.aborted && !res.destroyed && !res.writableEnded;
}
