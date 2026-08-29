/**
 * The editor's keyboard, behind one button, and the list it is drawn from.
 */

import { HelpCircle } from "lucide-react";
import { Kbd, Popover } from "./ui";

/**
 * Every shortcut the editor binds, in the order they are learned.
 *
 * `a` and `b` name the buttons they press, which now say Orig and Dub on their
 * faces. The keys did not change a comparison tool's A and B are worth keeping
 * as *keys* but a help list that says "play the original" next to a button
 * labelled "Orig" is one vocabulary, and it used to be two.
 */
const SHORTCUTS: [string[], string][] = [
  [["space"], "play / pause"],
  [["↑", "↓"], "previous / next line"],
  [["↵"], "edit the selected translation"],
  [["esc"], "leave the field without saving"],
  [["a"], "play Orig: the original for this line"],
  [["b"], "play Dub: the dubbed clip for this line"],
  [["k"], "switch between dub and keep"],
  [["s"], "split the selection at the playhead"],
  [["⌫"], "remove the selected line (⌘Z restores it)"],
  [["+", "−"], "zoom the timeline"],
  [["⌘", "f"], "search the script"],
  [["⌘", "z"], "undo the last edit"],
  [["⌘", "⇧", "z"], "redo the edit you undid"],
];

/**
 * The keyboard, behind one button.
 *
 * It used to share this popover with a colour legend. The legend is gone: every
 * script row now carries its state as a word, which is the same information in
 * the place it is needed, and a key to an encoding that is already spelled out
 * is a key to nothing.
 *
 * One mark did not get that treatment and could not: the timeline's diagonal
 * hatch has no row, no word and no chip anywhere in the app, so a reviewer who
 * wonders what the striped blocks are has nothing to read. The spans themselves
 * now carry the sentence as a tooltip; this is the same sentence in the place
 * somebody goes when they are looking for an explanation rather than pointing at
 * the thing they want explained.
 */
export function ShortcutHelp() {
  return (
    <Popover
      label="Keyboard shortcuts"
      title="Keyboard"
      trigger={<HelpCircle className="h-3.5 w-3.5" />}
      className="w-[19rem]"
    >
      <dl className="flex flex-col gap-1.5 text-[11px]">
        {SHORTCUTS.map(([keys, does]) => (
          <div key={does} className="flex items-baseline gap-2">
            <dt className="flex shrink-0 items-center gap-1">
              {keys.map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </dt>
            <dd className="min-w-0 flex-1 text-right text-muted">{does}</dd>
          </div>
        ))}
      </dl>
      <p
        data-hatch-note
        className="mt-3 border-t border-border pt-2.5 text-[11px] leading-relaxed text-muted"
      >
        <span
          aria-hidden
          className="hatch-unclaimed mr-1.5 inline-block h-2.5 w-5 -mb-0.5 rounded-[2px] border border-dashed"
          style={{ borderColor: "color-mix(in srgb, var(--color-unclaimed) 45%, transparent)" }}
        />
        The hatched blocks on the timeline are <span className="text-secondary">unclaimed
        time</span>: no segment covers them, so the dub plays the original there.
      </p>
    </Popover>
  );
}
