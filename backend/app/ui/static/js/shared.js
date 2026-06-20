(() => {
  const PREFERENCE_KEY = "ariadne.uiPreferences.v3";
  const DEFAULT_PREFERENCES = {
    appearance: "light",
    answerMode: "balanced",
    searchDepth: "standard",
    sourceVisibility: "sources",
  };
  const DEPTH_TO_TOP_K = { fast: 5, standard: 8, deep: 12 };

  const api = async (url, options = {}) => {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
    if (!response.ok) {
      const detail = data?.detail || data?.error || response.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  };

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function loadPreferences() {
    try { return { ...DEFAULT_PREFERENCES, ...JSON.parse(localStorage.getItem(PREFERENCE_KEY) || "{}") }; }
    catch { return { ...DEFAULT_PREFERENCES }; }
  }
  function savePreferences(prefs) {
    localStorage.setItem(PREFERENCE_KEY, JSON.stringify({ ...loadPreferences(), ...prefs }));
    applyAppearance(loadPreferences().appearance);
  }
  function resetPreferences() {
    localStorage.setItem(PREFERENCE_KEY, JSON.stringify(DEFAULT_PREFERENCES));
    applyAppearance(DEFAULT_PREFERENCES.appearance);
  }
  function applyAppearance(value) {
    const mode = value || loadPreferences().appearance || "dark";
    if (mode === "system") {
      const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.dataset.theme = dark ? "dark" : "light";
      return;
    }
    document.documentElement.dataset.theme = mode === "dark" ? "dark" : "light";
  }
  applyAppearance(loadPreferences().appearance);
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    if (loadPreferences().appearance === "system") applyAppearance("system");
  });

  const toast = (message) => {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message;
    el.classList.add("show");
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.remove("show"), 3300);
  };

  const setBusy = (button, busy, label = "Working...") => {
    if (!button) return;
    if (busy) {
      button.dataset.originalText = button.textContent;
      button.textContent = label;
      button.disabled = true;
    } else {
      button.textContent = button.dataset.originalText || button.textContent;
      button.disabled = false;
    }
  };

  function humanStatus(value) {
    const v = String(value || "unknown").toLowerCase();
    const map = {
      ok: "Ready", fresh: "Search Ready", high: "Reliable", medium: "Needs Review",
      low: "Low Confidence", no_answer: "Not Enough Evidence", warning: "Check Answer",
      running: "Running", completed: "Completed", failed: "Failed", error: "Error"
    };
    return map[v] || v.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
  }

  const renderPills = (status) => {
    const el = document.getElementById("statusPills");
    if (!el) return;
    const overall = status?.index_lifecycle?.overall_status || "unknown";
    const docs = status?.metadata_db?.documents ?? "-";
    const chunks = status?.metadata_db?.chunks ?? "-";
    const model = status?.llm?.model || "local model";
    el.innerHTML = `
      <span class="pill ${overall}">${escapeHtml(humanStatus(overall))}</span>
      <span class="pill muted">${escapeHtml(model)}</span>
      <span class="pill muted">${docs} docs</span>
      <span class="pill muted">${chunks} excerpts</span>`;
  };

  function initChoiceMenus(root = document) {
    root.querySelectorAll(".choice-menu").forEach(menu => {
      if (menu.dataset.ready === "1") return;
      menu.dataset.ready = "1";
      const trigger = menu.querySelector(".choice-trigger");
      const label = menu.querySelector(".choice-label");
      const options = [...menu.querySelectorAll(".choice-popover button")];
      function set(value) {
        const option = options.find(x => x.dataset.value === value) || options[0];
        menu.dataset.value = option?.dataset.value || "";
        if (label) label.textContent = option?.dataset.label || option?.textContent || "Choose";
        options.forEach(x => x.classList.toggle("active", x === option));
        menu.dispatchEvent(new CustomEvent("choice:changed", { bubbles: true, detail: { value: menu.dataset.value } }));
      }
      set(menu.dataset.value || options[0]?.dataset.value || "");
      trigger?.addEventListener("click", (event) => {
        event.stopPropagation();
        document.querySelectorAll(".choice-menu.open").forEach(other => { if (other !== menu) other.classList.remove("open"); });
        menu.classList.toggle("open");
      });
      options.forEach(option => option.addEventListener("click", () => { set(option.dataset.value || ""); menu.classList.remove("open"); }));
    });
  }
  document.addEventListener("click", () => document.querySelectorAll(".choice-menu.open").forEach(menu => menu.classList.remove("open")));
  function getChoiceValue(id) { return document.querySelector(`.choice-menu[data-choice-id="${id}"]`)?.dataset.value || ""; }
  function setChoiceValue(id, value) {
    const menu = document.querySelector(`.choice-menu[data-choice-id="${id}"]`);
    if (!menu) return;
    const option = [...menu.querySelectorAll(".choice-popover button")].find(x => x.dataset.value === value) || menu.querySelector(".choice-popover button");
    if (!option) return;
    menu.dataset.value = option.dataset.value || "";
    menu.querySelector(".choice-label").textContent = option.dataset.label || option.textContent;
    menu.querySelectorAll(".choice-popover button").forEach(x => x.classList.toggle("active", x === option));
  }

  function extractCitations(text) {
    return [...String(text || "").matchAll(/\[(DOCS|MEALS|HEADS|GEAR|IMAGES|HR|UNCLASSIFIED):[^\]]+\]/g)].map(m => m[0]);
  }
  function stripDisplaySections(text) {
    let value = String(text || "").replace(/\r\n/g, "\n").trim();
    value = value.replace(/^###\s*Answer\s*\n/i, "");
    value = value.replace(/^##\s*Answer\s*\n/i, "");
    value = value.replace(/\n?###\s*Supporting Evidence\s*\n[\s\S]*?(?=\n###\s*(Confidence|Missing Information)\s*\n|$)/gi, "\n");
    value = value.replace(/^.*\bis not used in this answer\b.*$/gim, "");
    value = value.replace(/\n{3,}/g, "\n\n").trim();
    return value;
  }

  function repairMarkdownTables(text) {
    let value = String(text || "");
    value = value.replace(/\|\s+\|\s+(?=[A-Za-z0-9\[\-])/g, "|\n| ");
    value = value.replace(/(\|[^\n]+\|)\s*(\|\s*:?-{3,}:?\s*\|)/g, "$1\n$2");
    value = value.replace(/(\|\s*:?-{3,}:?[^\n]*\|)\s*(\|\s*[A-Za-z0-9\[])/g, "$1\n$2");
    return value;
  }
  function normalizeTableLine(line) { return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim()); }
  function isTableLine(line) { return /^\s*\|.+\|\s*$/.test(line || ""); }
  function isSeparatorLine(line) { return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line || ""); }

  function displayCitation(match, citationMap = {}) {
    const mapped = citationMap?.[match];
    const label = mapped?.display_label || "Source";
    const name = mapped?.display_name || "Local document";
    const sourceIndex = mapped?.source_index || "";
    return `<button class="citation-pill" type="button" data-citation="${escapeHtml(match)}" data-source-index="${escapeHtml(sourceIndex)}" title="${escapeHtml(name)} — ${escapeHtml(match)}">${escapeHtml(label)}</button>`;
  }

  function inlineFormat(text, citationMap = {}) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\[(DOCS|MEALS|HEADS|GEAR|IMAGES|HR|UNCLASSIFIED):([^\]]+)\]/g, match => displayCitation(match, citationMap));
    return html;
  }

  function markdownLite(value, citationMap = {}) {
    const lines = repairMarkdownTables(stripDisplaySections(value)).split("\n");
    const out = [];
    let i = 0, paragraph = [];
    const flush = () => { if (paragraph.length) { out.push(`<p>${inlineFormat(paragraph.join(" ").trim(), citationMap)}</p>`); paragraph = []; } };
    while (i < lines.length) {
      const trimmed = lines[i].trim();
      if (!trimmed) { flush(); i++; continue; }
      if (/^###\s+/.test(trimmed)) { flush(); out.push(`<h3>${inlineFormat(trimmed.replace(/^###\s+/, ""), citationMap)}</h3>`); i++; continue; }
      if (/^##\s+/.test(trimmed)) { flush(); out.push(`<h3>${inlineFormat(trimmed.replace(/^##\s+/, ""), citationMap)}</h3>`); i++; continue; }
      if (isTableLine(trimmed) && isSeparatorLine(lines[i + 1] || "")) {
        flush();
        const header = normalizeTableLine(trimmed); i += 2;
        const rows = [];
        while (i < lines.length && isTableLine(lines[i])) { rows.push(normalizeTableLine(lines[i])); i++; }
        out.push(`<div class="table-wrap"><table class="ariadne-table"><thead><tr>${header.map(c => `<th>${inlineFormat(c, citationMap)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${header.map((_, idx) => `<td>${inlineFormat(row[idx] || "", citationMap)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      if (/^[-*]\s+/.test(trimmed)) {
        flush(); const items=[];
        while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) { items.push(lines[i].trim().replace(/^[-*]\s+/, "")); i++; }
        out.push(`<ul>${items.map(item => `<li>${inlineFormat(item, citationMap)}</li>`).join("")}</ul>`); continue;
      }
      if (/^\d+[.)]\s+/.test(trimmed)) {
        flush(); const items=[];
        while (i < lines.length && /^\d+[.)]\s+/.test(lines[i].trim())) { items.push(lines[i].trim().replace(/^\d+[.)]\s+/, "")); i++; }
        out.push(`<ol>${items.map(item => `<li>${inlineFormat(item, citationMap)}</li>`).join("")}</ol>`); continue;
      }
      paragraph.push(trimmed); i++;
    }
    flush(); return out.join("\n");
  }

  function scoreLabel(score) {
    if (typeof score !== "number") return "Relevance";
    if (score >= 0.75) return "Strong match";
    if (score >= 0.45) return "Useful match";
    return "Related";
  }
  const evidenceCard = (item, index = 0) => {
    const score = item.combined_score ?? item.score ?? null;
    const scoreText = typeof score === "number" ? score.toFixed(3) : "-";
    const title = item.title || item.citation_label || `Source ${index + 1}`;
    return `<article class="evidence-card" id="source-${index + 1}" data-citation="${escapeHtml(item.citation_label || "")}">
      <div class="evidence-title"><div><strong>Source ${index + 1}: ${escapeHtml(title)}</strong><p class="source-file">${escapeHtml(item.source_file || "Local source")}</p></div><span class="pill muted">${escapeHtml(scoreLabel(score))}</span></div>
      <div class="evidence-meta"><span class="code-pill">${escapeHtml(item.source_system || "Source")}</span><span class="code-pill">${escapeHtml(item.record_type || "excerpt")}</span><span class="code-pill">score ${escapeHtml(scoreText)}</span></div>
      <details class="technical-details"><summary>Technical citation</summary><div class="citations-row"><span class="citation-pill">${escapeHtml(item.citation_label || "")}</span></div></details>
      <pre class="preview">${escapeHtml(item.text_preview || "")}</pre>
    </article>`;
  };

  const sourceDocumentCard = (doc, index = 0, showTechnical = false) => {
    const sourceIndex = doc.source_index || index + 1;
    const systems = (doc.source_systems || []).join(", ") || "Local source";
    const recordTypes = (doc.record_types || []).join(", ") || "indexed content";
    const used = doc.chunk_count || (doc.citation_labels || []).length || 1;
    return `<article class="evidence-card document-source-card" id="doc-source-${escapeHtml(sourceIndex)}" data-source-index="${escapeHtml(sourceIndex)}">
      <div class="evidence-title"><div><strong>${escapeHtml(doc.display_label || `Source ${sourceIndex}`)}: ${escapeHtml(doc.display_name || doc.title || "Local document")}</strong><p class="source-file">Document-level source</p></div><span class="pill muted">${escapeHtml(used)} excerpt${used === 1 ? "" : "s"} used</span></div>
      <div class="evidence-meta"><span class="code-pill">${escapeHtml(systems)}</span><span class="code-pill">${escapeHtml(recordTypes)}</span></div>
      ${doc.preview ? `<pre class="preview">${escapeHtml(doc.preview)}</pre>` : ""}
      ${showTechnical ? `<details class="technical-details"><summary>Technical details</summary><p class="source-file">${escapeHtml(doc.full_path || doc.source_file || "")}</p><div class="citations-row">${(doc.citation_labels || []).map(label => `<span class="citation-pill">${escapeHtml(label)}</span>`).join("")}</div></details>` : ""}
    </article>`;
  };

  const initSidebar = () => {
    const shell = document.querySelector(".app-shell");
    const toggle = document.getElementById("sidebarToggle");
    const collapsed = localStorage.getItem("ariadne.sidebarCollapsed") === "1";
    function sync() {
      const isCollapsed = shell?.classList.contains("sidebar-collapsed");
      if (toggle) { toggle.textContent = isCollapsed ? "⇥" : "⇤"; toggle.title = isCollapsed ? "Expand sidebar" : "Collapse sidebar"; toggle.setAttribute("aria-label", toggle.title); }
    }
    if (shell && collapsed) shell.classList.add("sidebar-collapsed");
    sync();
    toggle?.addEventListener("click", () => { shell?.classList.toggle("sidebar-collapsed"); localStorage.setItem("ariadne.sidebarCollapsed", shell?.classList.contains("sidebar-collapsed") ? "1" : "0"); sync(); });
  };
  function getTopKFromPreferences() { return DEPTH_TO_TOP_K[loadPreferences().searchDepth] || DEPTH_TO_TOP_K.standard; }

  function segmented(name, current, options) {
    return `<div class="segmented" data-segmented="${name}">${options.map(([value,label]) => `<button type="button" data-value="${value}" class="${current === value ? "active" : ""}">${label}</button>`).join("")}</div>`;
  }
  function readSegmented(root, name) { return root.querySelector(`[data-segmented="${name}"] button.active`)?.dataset.value; }

  const CONFIG_GROUP_META = {
    "Local text model": {
      icon: "◉",
      title: "Ariadne's voice",
      description: "Choose the local model that speaks only from the evidence Ariadne is given.",
      tone: "model"
    },
    "Ingestion and chunking": {
      icon: "⇄",
      title: "The doorway",
      description: "Choose where new files enter and how Ariadne divides them into searchable strands.",
      tone: "ingestion"
    },
    "OCR fallback": {
      icon: "◇",
      title: "Reading faded script",
      description: "Let Ariadne read scanned pages when ordinary text extraction cannot find the thread.",
      tone: "fallback"
    },
    "Vision fallback": {
      icon: "◌",
      title: "Second sight",
      description: "Use a local vision model only when an image still needs a description.",
      tone: "fallback"
    },
    "Embeddings and index lifecycle": {
      icon: "◎",
      title: "The thread map",
      description: "Shape the map Ariadne uses to follow related ideas across local files.",
      tone: "index"
    },
    "Search behavior": {
      icon: "⌕",
      title: "The pathfinder",
      description: "Balance exact names and codes with meaning search, without hidden topic shortcuts.",
      tone: "retrieval"
    },
    "Answer generation": {
      icon: "✦",
      title: "The answer thread",
      description: "Decide how much evidence Ariadne carries forward and when she should ask for another clue.",
      tone: "answer"
    },
    "Runtime and LAN safety": {
      icon: "◆",
      title: "The boundary",
      description: "Keep Ariadne's work paced and keep model paths inside this host or approved LAN machines.",
      tone: "safety"
    }
  };

  function displayFieldValue(field) {
    if (Array.isArray(field.value)) return field.value.join(", ");
    if (field.kind === "bool") return field.value ? "On" : "Off";
    return field.value ?? "";
  }

  function fieldControl(field) {
    const path = escapeHtml(field.path);
    const value = field.value ?? "";
    const common = `data-setting-path="${path}" data-setting-kind="${escapeHtml(field.kind)}"`;
    if (field.kind === "bool") {
      return `<label class="ariadne-switch"><input ${common} type="checkbox" ${value ? "checked" : ""}><span aria-hidden="true"></span><em>${value ? "Enabled" : "Disabled"}</em></label>`;
    }
    if (field.kind === "select") {
      return `<select ${common} class="ariadne-select">${(field.choices || []).map(choice => `<option value="${escapeHtml(choice)}" ${String(value) === String(choice) ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>`;
    }
    if (field.kind === "text" || field.kind === "csv") {
      const shown = Array.isArray(value) ? value.join(", ") : value;
      return `<input ${common} class="ariadne-input" type="text" value="${escapeHtml(shown)}" placeholder="${escapeHtml(field.placeholder || "")}" />`;
    }
    return `<input ${common} class="ariadne-input" type="number" value="${escapeHtml(value)}" min="${field.min ?? ""}" max="${field.max ?? ""}" step="${field.step ?? "1"}" />`;
  }

  function renderConfigField(field) {
    const value = displayFieldValue(field);
    const kindLabel = field.kind === "bool" ? "toggle" : field.kind === "csv" ? "list" : field.kind;
    return `<article class="config-field-card" data-config-field="${escapeHtml(field.path)}">
      <div class="config-field-main">
        <div>
          <label class="config-label" for="cfg-${escapeHtml(field.path).replace(/[^a-zA-Z0-9_-]/g, "-")}">${escapeHtml(field.label)}</label>
          <p class="config-help">${escapeHtml(field.help || "")}</p>
        </div>
        <span class="config-kind">${escapeHtml(kindLabel)}</span>
      </div>
      <div class="config-control-wrap">${fieldControl(field)}</div>
      <div class="config-current"><span>Current</span><strong>${escapeHtml(value === "" ? "blank" : value)}</strong></div>
    </article>`;
  }

  function renderConfigGroups(configPayload) {
    const groups = configPayload?.schema?.editable_groups || configPayload?.editable_groups || {};
    return Object.entries(groups).map(([group, fields], index) => {
      const meta = CONFIG_GROUP_META[group] || { icon: "◇", title: group, description: "A local dial for this part of Ariadne's loom.", tone: "default" };
      const open = index < 2 ? "open" : "";
      return `<details class="config-section config-tone-${escapeHtml(meta.tone)}" ${open}>
        <summary>
          <span class="config-section-icon">${escapeHtml(meta.icon)}</span>
          <span class="config-section-copy"><strong>${escapeHtml(meta.title)}</strong><em>${escapeHtml(meta.description)}</em></span>
          <span class="config-section-count">${fields.length} controls</span>
        </summary>
        <div class="config-fields-grid">${fields.map(renderConfigField).join("")}</div>
      </details>`;
    }).join("");
  }

  async function renderAdminConfigEditor(container) {
    if (!container) return null;
    container.innerHTML = `<div class="config-loading"><span class="signal-dot"></span><p>Reading the loom settings...</p></div>`;
    const schema = await api("/api/config/effective");
    const groups = schema?.schema?.editable_groups || {};
    const groupCount = Object.keys(groups).length;
    const fieldCount = Object.values(groups).reduce((total, fields) => total + fields.length, 0);
    container.innerHTML = `
      <section class="config-hero-panel">
        <div>
          <p class="eyebrow">Ariadne's loom</p>
          <h3>Set the path; let the thread follow evidence.</h3>
          <p class="setting-subtle">These controls guide how files enter the maze, how difficult pages are read, how the thread map is renewed, and which local voice Ariadne uses. The baseline stays protected; changes here are kept as a local overlay you can restore at any time.</p>
        </div>
        <div class="config-hero-stats" aria-label="Editable configuration summary">
          <span><strong>${groupCount}</strong><em>Thread Rooms</em></span>
          <span><strong>${fieldCount}</strong><em>Tunable Dials</em></span>
        </div>
      </section>
      <section class="config-action-strip">
        <button id="detectAdminModels" class="secondary-button" type="button">Find local voices</button>
        <div id="adminModelDetection" class="config-detection-note">After changing a model path, let Ariadne look again for local voices she can use.</div>
      </section>
      <div class="config-section-stack">${renderConfigGroups(schema)}</div>`;
    container.querySelectorAll(".config-hero-stats span:nth-child(n+3)").forEach(node => node.remove());
    container.querySelectorAll(".ariadne-switch input").forEach(input => input.addEventListener("change", () => {
      const label = input.closest(".ariadne-switch")?.querySelector("em");
      if (label) label.textContent = input.checked ? "Enabled" : "Disabled";
    }));
    return schema;
  }

  async function renderStandaloneConfigEditor(container) {
    return renderAdminConfigEditor(container);
  }



  function parseConfigFieldValue(input) {
    const kind = input.dataset.settingKind || "text";
    if (kind === "bool") return Boolean(input.checked);
    if (kind === "int") {
      const raw = String(input.value || "").trim();
      if (raw === "") return null;
      const parsed = Number.parseInt(raw, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    if (kind === "float") {
      const raw = String(input.value || "").trim();
      if (raw === "") return null;
      const parsed = Number.parseFloat(raw);
      return Number.isFinite(parsed) ? parsed : null;
    }
    if (kind === "csv") {
      return String(input.value || "")
        .split(",")
        .map(value => value.trim())
        .filter(Boolean);
    }
    return String(input.value ?? "").trim();
  }

  function collectConfigOverrides(root = document) {
    const overrides = {};
    if (!root) return overrides;
    root.querySelectorAll("[data-setting-path]").forEach(input => {
      const path = input.dataset.settingPath;
      if (!path) return;
      overrides[path] = parseConfigFieldValue(input);
    });
    return overrides;
  }

  async function initSettingsDrawer() {
    const drawer = document.getElementById("settingsDrawer"), backdrop = document.getElementById("settingsBackdrop"), open = document.getElementById("settingsOpen"), close = document.getElementById("settingsClose"), content = document.getElementById("settingsContent"), save = document.getElementById("saveSettings"), reset = document.getElementById("resetSettings");
    if (!drawer || !content) return;
    const show = async () => { drawer.classList.add("open"); backdrop?.classList.add("open"); await renderSettings(); };
    const hide = () => { drawer.classList.remove("open"); backdrop?.classList.remove("open"); };
    open?.addEventListener("click", show); close?.addEventListener("click", hide); backdrop?.addEventListener("click", hide); document.addEventListener("keydown", e => { if (e.key === "Escape") hide(); });

    async function renderSettings() {
      const prefs = loadPreferences();
      content.innerHTML = `<p class="setting-subtle">Opening the view settings...</p>`;
      content.innerHTML = `
        <section class="setting-panel"><h3>Appearance</h3><p class="setting-subtle">Choose the light Ariadne uses on this device.</p>${segmented("appearance", prefs.appearance, [["light","Light"],["dark","Dark"],["system","Match device"]])}</section>
        <section class="setting-panel"><h3>Answer style</h3><p class="setting-subtle">Choose how much of the thread Ariadne lays out by default.</p>${segmented("answerMode", prefs.answerMode, [["brief","Brief"],["balanced","Balanced"],["detailed","Detailed"]])}</section>
        <section class="setting-panel"><h3>Source trail</h3><p class="setting-subtle">Choose how much of the evidence trail appears under each answer.</p>${segmented("sourceVisibility", prefs.sourceVisibility, [["compact","Compact"],["sources","Source cards"],["technical","Technical"]])}</section>
        <section class="setting-panel"><h3>Search depth</h3><p class="setting-subtle">Choose how far Ariadne looks into the source trail. This never adds hidden topic hints.</p>${segmented("searchDepth", prefs.searchDepth, [["fast","Fast"],["standard","Standard"],["deep","Deep"]])}</section>`;
      content.querySelectorAll("[data-segmented] button").forEach(btn => btn.addEventListener("click", () => { btn.parentElement.querySelectorAll("button").forEach(x => x.classList.remove("active")); btn.classList.add("active"); if (btn.parentElement.dataset.segmented === "appearance") applyAppearance(btn.dataset.value); }));
    }

    save?.addEventListener("click", async () => {
      try {
        const prefs = { appearance: readSegmented(content,"appearance"), answerMode: readSegmented(content,"answerMode"), sourceVisibility: readSegmented(content,"sourceVisibility"), searchDepth: readSegmented(content,"searchDepth") };
        savePreferences(prefs);
        setChoiceValue("answerMode", loadPreferences().answerMode);
        toast("View settings saved locally"); window.dispatchEvent(new CustomEvent("ariadne:settings-saved")); window.dispatchEvent(new CustomEvent("rags:settings-saved"));
      } catch (err) { toast(`View settings failed: ${err.message}`); } finally { setBusy(save, false); }
    });
    reset?.addEventListener("click", async () => { resetPreferences(); await renderSettings(); setChoiceValue("answerMode", DEFAULT_PREFERENCES.answerMode); toast("View reset"); });
  }

  function addButtonMicroInteractions() {
    document.addEventListener("pointerdown", event => {
      const button = event.target.closest("button, .nav-item, .operation-card"); if (!button) return;
      button.animate([{ transform: getComputedStyle(button).transform }, { transform: "scale(0.976)" }, { transform: getComputedStyle(button).transform }], { duration: 220, easing: "cubic-bezier(.22,1,.36,1)" });
    });
  }
  addButtonMicroInteractions();

  window.RAGS = { api, escapeHtml, markdownLite, stripDisplaySections, extractCitations, toast, setBusy, renderPills, evidenceCard, sourceDocumentCard, initSidebar, initSettingsDrawer, initChoiceMenus, getChoiceValue, setChoiceValue, loadPreferences, savePreferences, getTopKFromPreferences, humanStatus, applyAppearance, renderConfigGroups, renderAdminConfigEditor, renderStandaloneConfigEditor, collectConfigOverrides };
})();
