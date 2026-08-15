/**
 * NDJSON stream reader: one JSON object per line, held open indefinitely.
 *
 * Two things it has to survive, both of which happen in practice. A chunk can
 * land mid-line, so the tail is buffered until its newline arrives; and the
 * connection drops when the laptop sleeps, so it reconnects with backoff until
 * the caller aborts. Errors travel as frames, not status codes, so a non-200 on
 * connect is a transport problem and always worth retrying.
 */

export type NdjsonHandlers<T> = {
  onMessage: (value: T) => void;
  /** connection state, for the "reconnecting…" strip */
  onOpen?: () => void;
  onClose?: (reason: "aborted" | "ended" | "error", error?: unknown) => void;
};

const BACKOFF_MS = [500, 1000, 2000, 4000, 8000];

/**
 * Consume `url` until `signal` aborts. Resolves when aborted or when the caller
 * is told to stop; never rejects transport failures are retried.
 */
export async function readNdjson<T>(
  url: string,
  signal: AbortSignal,
  handlers: NdjsonHandlers<T>,
): Promise<void> {
  let attempt = 0;

  while (!signal.aborted) {
    try {
      const response = await fetch(url, {
        signal,
        headers: { accept: "application/x-ndjson" },
        cache: "no-store",
      });
      if (!response.ok || !response.body) {
        throw new Error(`events stream failed (${response.status})`);
      }

      attempt = 0;
      handlers.onOpen?.();

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Everything before the last newline is complete; the rest is a
        // partial line and waits for the next chunk.
        let newline = buffer.indexOf("\n");
        while (newline !== -1) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (line) emit(line, handlers.onMessage);
          newline = buffer.indexOf("\n");
        }
      }

      const tail = buffer.trim();
      if (tail) emit(tail, handlers.onMessage);
      if (signal.aborted) break;
      // A clean end of stream still means the server hung up; reconnect.
    } catch (error) {
      if (signal.aborted) break;
      handlers.onClose?.("error", error);
    }

    if (signal.aborted) break;
    const wait = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
    attempt += 1;
    await sleep(wait, signal);
  }

  handlers.onClose?.("aborted");
}

function emit<T>(line: string, onMessage: (value: T) => void): void {
  try {
    onMessage(JSON.parse(line) as T);
  } catch {
    // A malformed line is the server's bug, not a reason to drop the stream.
  }
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(finish, ms);
    signal.addEventListener("abort", finish, { once: true });
    function finish() {
      clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    }
  });
}
