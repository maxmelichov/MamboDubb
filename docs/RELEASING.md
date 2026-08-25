# Releasing the macOS .dmg

> Windows and Linux installers come out of
> [`.github/workflows/build-desktop.yml`](../.github/workflows/build-desktop.yml)
> instead, unsigned and with their own caveats. See
> [CROSS_PLATFORM.md](CROSS_PLATFORM.md).

The whole point of the desktop app is that a fresh Mac never needs a terminal. An
unsigned .dmg breaks that promise at step 1: Gatekeeper quarantines every download,
and an app with no Developer ID signature and no notarization ticket gets the
**"MamboDubb is damaged and can't be opened. You should move it to the Trash"**
dialog on any Mac but the one that built it. A signed, notarized .dmg is the only
way dragging the app to Applications just works.

Until the Developer ID exists, unsigned releases are still shippable: `release_dmg.py`
ad-hoc-signs the bundle, puts `install.sh` in the .dmg as **Install MamboDubb.command**,
and users install with

```bash
curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh
```

instead of dragging. Attach `install.sh` to the GitHub release as well as the `.dmg`.

The one command:

```bash
uv run --script scripts/release_dmg.py
```

It gates on the Qwen3-TTS submodule, builds the UI, stages the payload, runs
`pnpm tauri build` (Tauri signs every binary inside-out with the hardened runtime,
submits the .app to Apple, waits, and staples), then notarizes and staples the .dmg
container itself, verifies everything Gatekeeper will check (`codesign --verify`,
`spctl --assess`, `stapler validate`), and prints the final artifact path.
`--dry-run` prints the plan without running anything.

## Prerequisites (one-time)

### 1. A Developer ID Application certificate, the one human-only step

Nothing in this repo can conjure this; it takes a paid Apple Developer Program
enrollment ($99/year) and **only the Account Holder** can create Developer ID
certificates.

1. Keychain Access → Certificate Assistant → *Request a Certificate from a
   Certificate Authority* → save the `.certSigningRequest` to disk (this drops the
   private key into your **login** keychain).
2. [developer.apple.com → Certificates](https://developer.apple.com/account/resources/certificates/list)
   → create a **Developer ID Application** certificate (G2 Sub-CA), upload the CSR,
   download the `.cer`, double-click it into the **login** keychain.
3. Install the intermediates from
   [Apple's certificate authority page](https://www.apple.com/certificateauthority/):
   **Developer ID - G2** and, if missing, **Apple Root CA - G2**. macOS does not
   always ship these, and without them the next step reports
   `1 matching identity, 0 valid identities`. It looks like a permissions problem
   but is a broken trust chain.
4. Verify:

   ```bash
   security find-identity -v -p codesigning
   # → 1 valid identity found
   #   "Developer ID Application: Your Name (TEAMID)"
   ```

### 2. An app-specific password for notarization

Apple refuses the account password here. Create one at
[account.apple.com](https://account.apple.com) → Sign-In and Security →
App-Specific Passwords.

### 3. The four environment variables

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="your@email.com"
export APPLE_PASSWORD="the-app-specific-password"
export APPLE_TEAM_ID="TEAMID"        # the 10 characters in the identity's parentheses
```

Tauri reads all four itself during `tauri build`; the script only checks they exist
and warns loudly when they do not. With none set it builds **unsigned**, ad-hoc-signs
the bundle, injects `Install MamboDubb.command`, and prints the `install.sh` one-liner;
with only the identity set it warns that signed-but-unnotarized is still blocked by
current macOS ("Apple could not verify…" instead of "damaged": politer, equally dead).

## The release

```bash
# once per clone
git submodule update --init --recursive

# bump the version in the THREE files that must move together (see app/desktop/README.md):
#   app/desktop/src-tauri/tauri.conf.json  → "version"   (names the .dmg, CFBundleShortVersionString)
#   app/desktop/src-tauri/Cargo.toml       → [package] version
#   app/desktop/package.json               → "version"

uv run --script scripts/release_dmg.py
```

Notarization is a round-trip to Apple's servers. Usually minutes; the **first**
notarization for a new app or certificate can take hours, which is normal. If the
wait was interrupted, the submission keeps processing on Apple's side. Check with:

```bash
xcrun notarytool history --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
xcrun notarytool log <submission-id> --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
```

(the log, with rejection reasons if any, appears only after processing finishes),
then staple by hand: `xcrun stapler staple <the .dmg>` and the .app inside the
bundle dir if its own stapling was the interrupted step.

## Afterwards, before uploading anywhere

The script already ran these; anything re-signed or re-stapled by hand should pass
them again:

```bash
APP=app/desktop/src-tauri/target/release/bundle/macos/MamboDubb.app
codesign --verify --deep --strict --verbose=2 "$APP"
spctl --assess --type exec -v "$APP"     # "accepted, source=Notarized Developer ID"
xcrun stapler validate "$APP"
xcrun stapler validate <the .dmg>
```

The end-to-end check that actually settles it: copy the .dmg to **another Mac** (or
re-quarantine locally with `xattr -w com.apple.quarantine "0083;;;" <dmg>`), mount,
drag to Applications, launch from Finder. It must open with no dialog beyond the
standard first-run confirmation: never "damaged", never "could not verify".

Then attach the .dmg to the GitHub release; the README's install link points at
`releases/latest`.
