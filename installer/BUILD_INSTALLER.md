# Building `Ariadne_setup.exe`

This folder turns the Ariadne project into a Windows installer using **Inno Setup**.
The installer ships the application files only — the Python dependencies and the
language/embedding models are installed by the user afterwards (via `setup.bat`
and Ollama), exactly as the README describes.

---

## What you need (once)

- **Inno Setup 6** — free, from <https://jrsoftware.org/isdl.php>. Install it normally.

That's the only build-time requirement. Inno Setup's compiler (`ISCC.exe`) is
Windows-only, so the installer must be compiled on Windows.

---

## Folder layout

Keep this `installer\` folder at the **project root**, alongside `backend\`,
`scripts\`, and the `.bat` files:

```
Ariadne\
├── backend\
├── scripts\
├── tests\  tools\
├── config\
├── setup.bat  ingest.bat  start.bat  start_lan.bat  stop.bat
├── requirements.txt
├── README.md
└── installer\
    ├── Ariadne.iss          <- the installer script
    ├── Ariadne.ico          <- the app icon
    └── BUILD_INSTALLER.md    <- this file
```

The script uses relative paths (`..\backend`, `..\scripts`, …), so this layout
matters.

---

## Build it

**Option A — double-click:** right-click `Ariadne.iss` → **Compile**.

**Option B — command line:**

```bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\Ariadne.iss
```

The finished installer appears at:

```
installer\dist\Ariadne_setup.exe
```

Upload that to your GitHub release. It is small (just source + scripts), so it
stays well under the 2 GB release-asset limit.

---

## What the installer does

- Installs to `%LOCALAPPDATA%\Programs\Ariadne` by default (per-user, no admin
  needed, and fully writable — important, because Ariadne creates its `.venv`
  and writes its database/indexes inside its own folder).
- Creates Start-Menu shortcuts in run order:
  1. **Setup (run first)** → `setup.bat`
  2. **Index documents** → `ingest.bat`
  3. **Ariadne** → `start.bat` (and a "share on network" entry for `start_lan.bat`)
  plus an **Uninstall** entry and an "Open Ariadne folder" shortcut.
- Optionally adds a desktop shortcut and can run `setup.bat` right after install.
- Registers a clean uninstaller that also removes generated files (`.venv`,
  `storage\`, processed data, overrides).

## What the installer deliberately leaves out

- **Python dependencies** (`torch`, `sentence-transformers`, etc.) — installed by
  `setup.bat` into a local `.venv`.
- **The virtual environment** (`.venv\`) — never shipped.
- **Models** — the Ollama model (`ollama pull llama3.1:8b`) and the Hugging Face
  embedding/reranker models, fetched on first use.
- **Generated runtime data** — the database, indexes, logs, processed files, and
  `config\ui_overrides.yaml` are excluded; the empty folders are created so the
  app has somewhere to write.

---

## Customising

Open `Ariadne.iss` and edit the `#define` lines at the top:

```
#define AppVersion   "1.0.0"      ; bump this each release
#define AppPublisher "ZenithAI"
#define AppURL       "https://github.com/your-org/ariadne"
```

Keep the `AppId` GUID **unchanged** across versions — it is how Windows
recognises an upgrade of the same application rather than a second copy.

### Optional: a cleaner first-run experience

The installer points shortcuts at the `.bat` files, which open a console window
(useful — it shows progress and server logs). If you later want a windowless
launcher or a bundled environment, that is a separate, larger packaging effort.

### Optional: code signing

An unsigned installer triggers a Windows SmartScreen warning ("Windows protected
your PC"). To avoid it, sign `Ariadne_setup.exe` with a code-signing certificate
using `signtool`. This is optional; without it the installer still works, users
just click **More info → Run anyway**.
