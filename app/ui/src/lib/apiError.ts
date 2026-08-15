/**
 * The server's error envelope, as a throwable.
 *
 * It lives in a module of its own rather than in `api.ts` because both sides of
 * the seam need it: `api.request` builds one from `{"error":{code,message}}`,
 * and the fixture backend has to *throw the same thing* a fixture that
 * rejects with a plain `Error` is a second, kinder server whose failures have
 * no code, no status, and none of the branching the real ones get (the 404 that
 * means "no dub.wav yet" above all). Importing it from `api.ts` would be a
 * cycle: `api` imports `fixtures`, so `fixtures` cannot import `api`.
 */

import type { ErrorCode } from "./types";

export class ApiError extends Error {
  readonly code: ErrorCode;
  readonly status: number;

  constructor(code: ErrorCode, message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }

  /**
   * `busy` is the one-job rule refusing an edit never a queue notice.
   *
   * The server raises it for *refusals* only (a structural edit while a job
   * runs, a second install), and those are not queued behind anything; the
   * model actions queue and answer 202. So there is nothing to say here that
   * the server's own message does not say better, and this is a predicate, not
   * a place to invent copy.
   */
  get isBusy(): boolean {
    return this.code === "busy";
  }
}
