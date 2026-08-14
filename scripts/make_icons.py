#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Render the Dubbing Studio app icon ladder from its one SVG source.

    uv run --script scripts/make_icons.py            # regenerate every raster
    uv run --script scripts/make_icons.py --check    # fail if the rasters are stale

`app/desktop/src-tauri/icons/icon.svg` is the source of truth; every PNG, the .icns
and the .ico in that directory are outputs. Edit the SVG, run this, commit both.

## Why QuickLook does the rasterising

macOS ships no SVG rasteriser that a script can call directly: `sips` cannot read SVG,
and `rsvg-convert` / Inkscape / ImageMagick are all extra installs. `cairosvg` needs
libcairo, which is one more Homebrew dependency. What every Mac *does* have is
QuickLook, whose WebKit-backed SVG generator `qlmanage -t -s 1024` will write a
1024x1024 PNG. Its one flaw is that it composites onto opaque white and throws the
alpha channel away — so we re-cut the alpha ourselves from the tile silhouette, which
we know exactly because the SVG's tile is a plain rounded rect on the macOS icon grid
(824x824 inset in a 1024 canvas, corner radius 185). The SVG paints 4 px of bleed
past that silhouette so the white-blended edge pixels fall outside the cut.

Everything below the 1024 master is a Lanczos downscale (Pillow), which is sharper at
16-32 px than re-rasterising the SVG at those sizes would be. `iconutil` assembles the
.icns from a standard .iconset; Pillow writes the multi-resolution .ico.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ICONS = ROOT / "app" / "desktop" / "src-tauri" / "icons"
SOURCE = ICONS / "icon.svg"

MASTER = 1024
# The macOS icon grid, and the geometry the SVG's tile is drawn on.
TILE_INSET = 100
TILE_SIZE = 824
TILE_RADIUS = 185
SUPERSAMPLE = 4

# Tauri reads these by name (bundle.icon + the Windows Store tiles the CLI scaffolds).
PNG_LADDER = {
    "32x32.png": 32,
    "64x64.png": 64,
    "128x128.png": 128,
    "128x128@2x.png": 256,
    "icon.png": 1024,
    "Square30x30Logo.png": 30,
    "Square44x44Logo.png": 44,
    "Square71x71Logo.png": 71,
    "Square89x89Logo.png": 89,
    "Square107x107Logo.png": 107,
    "Square142x142Logo.png": 142,
    "Square150x150Logo.png": 150,
    "Square284x284Logo.png": 284,
    "Square310x310Logo.png": 310,
    "StoreLogo.png": 50,
}

# The full macOS ladder. iconutil rejects an .iconset with a size it does not know.
ICONSET = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def rasterize(svg: Path) -> Image.Image:
    """SVG -> a 1024x1024 RGBA master, alpha re-cut from the tile silhouette."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run(
            ["qlmanage", "-t", "-s", str(MASTER), "-o", str(out), str(svg)],
            check=True,
            capture_output=True,
        )
        rendered = out / f"{svg.name}.png"
        if not rendered.is_file():
            raise SystemExit(
                f"error: qlmanage produced no thumbnail for {svg}. The usual cause is "
                f"invalid SVG (an XML comment containing a double hyphen will do it) — "
                f"open the file in Quick Look to see the parser error."
            )
        image = Image.open(rendered).convert("RGBA")
    if image.size != (MASTER, MASTER):
        image = image.resize((MASTER, MASTER), Image.LANCZOS)
    image.putalpha(tile_mask())
    return image


def tile_mask() -> Image.Image:
    """The rounded-square silhouette, drawn oversized and downsampled for clean edges."""
    scale = SUPERSAMPLE
    mask = Image.new("L", (MASTER * scale, MASTER * scale), 0)
    draw = ImageDraw.Draw(mask)
    lo = TILE_INSET * scale
    hi = (TILE_INSET + TILE_SIZE) * scale - 1
    draw.rounded_rectangle([lo, lo, hi, hi], radius=TILE_RADIUS * scale, fill=255)
    return mask.resize((MASTER, MASTER), Image.LANCZOS)


def scaled(master: Image.Image, size: int) -> Image.Image:
    if size == master.width:
        return master.copy()
    return master.resize((size, size), Image.LANCZOS)


def render_all(master: Image.Image, dest: Path) -> dict[str, bytes]:
    """Every output as name -> bytes, so --check can compare without writing."""
    out: dict[str, bytes] = {}
    for name, size in PNG_LADDER.items():
        out[name] = png_bytes(scaled(master, size))

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for name, size in ICONSET:
            scaled(master, size).save(iconset / name, "PNG")
        icns = Path(tmp) / "icon.icns"
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
            check=True,
            capture_output=True,
        )
        out["icon.icns"] = icns.read_bytes()

        ico = Path(tmp) / "icon.ico"
        master.save(ico, "ICO", sizes=[(s, s) for s in ICO_SIZES])
        out["icon.ico"] = ico.read_bytes()
    return out


def png_bytes(image: Image.Image) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.png"
        image.save(path, "PNG")
        return path.read_bytes()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed rasters match the SVG instead of rewriting them",
    )
    args = parser.parse_args(argv)

    if not SOURCE.is_file():
        raise SystemExit(f"error: {SOURCE} is missing")
    if not shutil.which("qlmanage") or not shutil.which("iconutil"):
        raise SystemExit("error: this script needs macOS (qlmanage + iconutil)")

    outputs = render_all(rasterize(SOURCE), ICONS)

    if args.check:
        stale = [
            name
            for name, data in outputs.items()
            if not (ICONS / name).is_file() or (ICONS / name).read_bytes() != data
        ]
        if stale:
            print("stale icons (re-run without --check): " + ", ".join(sorted(stale)))
            return 1
        print(f"{len(outputs)} icons match {SOURCE.name}")
        return 0

    for name, data in sorted(outputs.items()):
        (ICONS / name).write_bytes(data)
        print(f"  {name:<24} {len(data):>8} bytes  {digest(data)}")
    print(f"\nwrote {len(outputs)} icons to {ICONS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
