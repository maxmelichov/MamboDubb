/**
 * The receipt behind the one destructive verdict, lifted out of `EditorPage`
 * so the page reads as the page.
 */

import { Button, StateIcon } from "./ui";
import { cn } from "../lib/classNames";

/**
 * The whole of what a flip to keep destroys, held for {@link UNDO_MS}.
 *
 * A fresh object every time, so a second keep re-arms the timer rather than
 * inheriting whatever was left of the first one's.
 */
export type KeptUndo = { uid: string; id: number; text_en: string | null };

/**
 * "Kept #17 Undo."
 *
 * The whole guard on the one destructive verdict, and deliberately not a dialog:
 * the flip has to stay a keystroke, because checking a run is a hundred of them.
 * A strip that costs nothing to ignore and one click to reverse is the trade a
 * confirm cannot make.
 *
 * The title says the part the strip has no room for and the user has no way to
 * guess: the restored translation comes back *locked*, because restoring it is
 * writing it, and writing a line is what makes it the user's. That is a better
 * outcome than losing it and it is not the same state as before, so it is said
 * rather than glossed.
 */
export function UndoKeepStrip({ undo, onUndo }: { undo: KeptUndo; onUndo: () => void }) {
  return (
    <div
      role="status"
      data-undo-toast
      title={
        undo.text_en
          ? "Undo switches this line back to Dub and puts its translation back. A restored " +
            "line counts as hand-written from then on, so a re-run will not replace it."
          : "Undo switches this line back to Dub and queues the work it needs."
      }
      className={cn(
        // Above the 7rem timeline strip, centred, out of the script's way.
        "fixed bottom-[8.5rem] left-1/2 z-50 -translate-x-1/2",
        "flex items-center gap-2 rounded-full border border-border bg-raised px-3 py-1.5",
        "text-[12.5px] text-primary shadow-pop",
      )}
    >
      <StateIcon state="kept" className="h-2.5 w-2.5" />
      <span className="font-mono tabular-nums">Kept #{undo.id}</span>
      <span aria-hidden className="text-muted">
        ·
      </span>
      <Button size="xs" variant="ghost" data-undo-keep onClick={onUndo}>
        Undo
      </Button>
    </div>
  );
}
