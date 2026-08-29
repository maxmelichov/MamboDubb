/**
 * Opening a file the run produced, in whichever environment the app is in.
 * Shared by the editor header and the run menu's file list.
 */

import { ExternalLink, FolderOpen } from "lucide-react";
import { Button } from "./ui";
import { api } from "../lib/api";
import { isDesktop, revealRunFile } from "../lib/desktop";

/**
 * Open a file the run produced.
 *
 * Two environments, one intent. In the desktop shell the useful thing is the
 * file itself the user wants to play it in QuickTime, drop it into a chat,
 * put it somewhere so it is revealed in Finder. In a browser tab there is no
 * Finder and the server is already serving the run directory, so the URL opens
 * in a new tab. The shell falls back to the tab if the reveal could not
 * happen, which covers an older shell with no `workspace` in its handshake.
 *
 * It is a function rather than a hook because both callers the header button
 * and the run menu's file list want the same three lines and neither wants
 * to think about which environment it is in.
 */
export async function openRunFile(name: string, relPath: string): Promise<void> {
  if (await revealRunFile(name, relPath)) return;
  const href = api.mediaUrl(name, relPath);
  if (href) window.open(href, "_blank", "noopener");
}

/**
 * "Where is the file?" as a button.
 *
 * The single most-asked question at the end of a run, and until now the answer
 * was a Finder window and a memory of the run directory's name. Header-sized
 * and labelled for the environment it is in "Show in Finder" is a promise a
 * browser tab cannot keep, and "Open" is a weaker one than the shell can.
 */
export function OpenFileButton(
  { name, path, title }: { name: string; path: string; title?: string },
) {
  const desktop = isDesktop();
  return (
    <Button
      size="sm"
      // When the file is behind the edits, say so on the button that opens it —
      // this is the last moment before the user watches five minutes of a video
      // that does not contain the correction they just made.
      title={
        title
          ? `${desktop ? `Show ${path} in Finder` : `Open ${path} in a new tab`} (${title})`
          : desktop
            ? `Show ${path} in Finder`
            : `Open ${path} in a new tab`
      }
      onClick={() => void openRunFile(name, path)}
    >
      {desktop ? (
        <FolderOpen className="h-3.5 w-3.5" aria-hidden />
      ) : (
        <ExternalLink className="h-3.5 w-3.5" aria-hidden />
      )}
      {desktop ? "Show in Finder" : "Open preview"}
    </Button>
  );
}
