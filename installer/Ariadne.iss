; ============================================================================
;  Ariadne — Inno Setup installer script
;  Builds Ariadne_setup.exe, which installs the application files (Python
;  sources, scripts, config, and UI). It deliberately does NOT bundle the
;  Python dependencies or the language/embedding models — the user installs
;  those by running "1. Setup" (which runs setup.bat) and pulling the models
;  with Ollama, exactly as described in the README.
;
;  HOW TO BUILD:
;    1. Install Inno Setup 6  ->  https://jrsoftware.org/isdl.php
;    2. Place this file in an "installer\" folder at the project root,
;       so that "..\backend", "..\scripts", etc. resolve correctly.
;    3. Right-click Ariadne.iss -> "Compile", or run:
;         "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\Ariadne.iss
;    4. The finished installer appears in installer\dist\Ariadne_setup.exe
; ============================================================================

#define AppName        "Ariadne"
#define AppVersion     "1.0.0"
#define AppPublisher   "ZenithAI"
#define AppURL         "https://github.com/your-org/ariadne"

[Setup]
; A unique identity for this application (keep this GUID stable across versions).
AppId={{C4DCCDDA-0A74-4F8C-9A30-535B126B8356}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}

; Per-user install under %LOCALAPPDATA%\Programs\Ariadne.
; This is intentional: Ariadne creates its own virtual environment (.venv) and
; writes its database, indexes, and logs INSIDE its own folder at runtime.
; Installing under Program Files would require admin rights AND break those
; runtime writes for standard users. A per-user location stays fully writable.
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Output
OutputDir=dist
OutputBaseFilename=Ariadne_setup
SetupIconFile=Ariadne.ico
UninstallDisplayIcon={app}\Ariadne.ico

; Appearance & behavior
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
DisableWelcomePage=no
DisableProgramGroupPage=yes
ShowLanguageDialog=no
ArchitecturesInstallIn64BitMode=x64compatible
; Optional pages — uncomment if you add the files:
; LicenseFile=..\LICENSE.txt
; InfoBeforeFile=..\README.md

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut to start Ariadne"; GroupDescription: "Additional shortcuts:"

[Files]
; --- Application source tree ---------------------------------------------
; recursesubdirs copies subfolders; Excludes strips dev/runtime junk so the
; installer never ships a virtual environment, caches, the database, the
; built indexes, logs, private documents, or machine-local overrides.
Source: "..\backend\*"; DestDir: "{app}\backend"; Flags: recursesubdirs createallsubdirs ignoreversion; \
    Excludes: "__pycache__\*,*.pyc,*.pyo,_preview.html,_admin.html"
Source: "..\scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs createallsubdirs ignoreversion; \
    Excludes: "__pycache__\*,*.pyc,*.pyo"
Source: "..\tests\*";   DestDir: "{app}\tests";   Flags: recursesubdirs createallsubdirs ignoreversion; \
    Excludes: "__pycache__\*,*.pyc,*.pyo"
Source: "..\tools\*";   DestDir: "{app}\tools";   Flags: recursesubdirs createallsubdirs ignoreversion; \
    Excludes: "__pycache__\*,*.pyc,*.pyo"

; --- Configuration baseline (NOT the machine-local ui_overrides.yaml) -----
Source: "..\config\client.yaml"; DestDir: "{app}\config"; Flags: ignoreversion
; If you keep other static config files, add them explicitly here.

; --- Launch scripts -------------------------------------------------------
Source: "..\setup.bat";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\ingest.bat";       DestDir: "{app}"; Flags: ignoreversion
Source: "..\start.bat";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\start_lan.bat";    DestDir: "{app}"; Flags: ignoreversion
Source: "..\stop.bat";         DestDir: "{app}"; Flags: ignoreversion

; --- Project metadata & icon ---------------------------------------------
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\.gitignore";       DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "Ariadne.ico";         DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; Create the empty runtime folders the app expects to write into.
Name: "{app}\data\input"
Name: "{app}\data\processed"
Name: "{app}\storage\logs"
Name: "{app}\storage\vector"
Name: "{app}\storage\incremental_work"
Name: "{app}\reports"

[Icons]
; Numbered so the run-order is obvious in the Start Menu.
Name: "{group}\1. Ariadne - Setup (run first)"; Filename: "{app}\setup.bat";  WorkingDir: "{app}"; IconFilename: "{app}\Ariadne.ico"; Comment: "Create the environment and install dependencies (needs internet once)"
Name: "{group}\2. Ariadne - Index documents";   Filename: "{app}\ingest.bat"; WorkingDir: "{app}"; IconFilename: "{app}\Ariadne.ico"; Comment: "Ingest your documents and build the search indexes"
Name: "{group}\Ariadne";                        Filename: "{app}\start.bat";  WorkingDir: "{app}"; IconFilename: "{app}\Ariadne.ico"; Comment: "Start Ariadne and open it in your browser"
Name: "{group}\Ariadne (share on network)";     Filename: "{app}\start_lan.bat"; WorkingDir: "{app}"; IconFilename: "{app}\Ariadne.ico"; Comment: "Start Ariadne so other devices on your network can reach it"
Name: "{group}\Open Ariadne folder";            Filename: "{app}"
Name: "{group}\Uninstall Ariadne";              Filename: "{uninstallexe}"
; Optional desktop shortcut
Name: "{autodesktop}\Ariadne"; Filename: "{app}\start.bat"; WorkingDir: "{app}"; IconFilename: "{app}\Ariadne.ico"; Tasks: desktopicon

[Run]
; Offer to open the README and to run setup immediately after installation.
Filename: "{app}\README.md"; Description: "Open the README"; Flags: postinstall shellexec skipifsilent unchecked
Filename: "{app}\setup.bat"; Description: "Run Setup now (creates the environment and installs dependencies)"; WorkingDir: "{app}"; Flags: postinstall shellexec skipifsilent

[UninstallDelete]
; Remove everything the app generated so uninstall leaves nothing behind.
; (Comment these out if you would rather preserve a user's indexed data.)
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\storage"
Type: filesandordirs; Name: "{app}\data\processed"
Type: filesandordirs; Name: "{app}\data\input\_extracted"
Type: filesandordirs; Name: "{app}\config\ui_overrides.yaml"
Type: filesandordirs; Name: "{app}\reports"
Type: filesandordirs; Name: "{app}\backend\__pycache__"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nAriadne installs the application files only. After installing, run "1. Ariadne - Setup" to create the environment and install dependencies, make sure Ollama is running with your model pulled, then index your documents and launch.
