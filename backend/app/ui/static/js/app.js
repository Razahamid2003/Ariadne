(async () => {
  const {
    api, toast, setBusy, renderPills, evidenceCard, sourceDocumentCard, markdownLite,
    stripDisplaySections, extractCitations, escapeHtml, initSidebar, initSettingsDrawer,
    initChoiceMenus, getChoiceValue, setChoiceValue, loadPreferences, getTopKFromPreferences,
    humanStatus,
  } = window.RAGS;

  initSidebar();
  initChoiceMenus();
  await initSettingsDrawer();

  let activeChatId = null;
  let currentChats = [];
  let currentView = "chat";

  const pageTitle = document.getElementById("pageTitle");
  const pageSubtitle = document.getElementById("pageSubtitle");
  const views = { chat: document.getElementById("view-chat"), threads: document.getElementById("view-threads"), search: document.getElementById("view-search"), sources: document.getElementById("view-sources") };
  const titleByView = {
    chat: ["Ariadne", "Follow the thread. Navigate uncertainty."],
    threads: ["Threads", "Peer into past threads. Continue the journey."],
    search: ["Sources", "Look into the source of each thread."],
    sources: ["Source trail", "See the evidence Ariadne can follow."],
  };

  function switchView(view, options = {}) {
    currentView = view || "chat";
    document.querySelectorAll("[data-view]").forEach(x => x.classList.toggle("active", x.dataset.view === currentView));
    Object.values(views).forEach(v => v?.classList.remove("active"));
    views[currentView]?.classList.add("active");
    if (pageTitle) pageTitle.textContent = titleByView[currentView]?.[0] || "Ariadne";
    if (pageSubtitle) pageSubtitle.textContent = titleByView[currentView]?.[1] || "Follow the thread. Navigate uncertainty.";
    if (currentView === "sources") refreshStatus();
    if (currentView === "threads") renderThreadArchive();
    if (currentView === "chat" && options.freshHome) startFreshHomeThread({ focus: true });
  }

  document.querySelectorAll("[data-view]").forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.view, { freshHome: btn.dataset.view === "chat" }));
  });

  function syncUiPreferences() {
    const prefs = loadPreferences();
    setChoiceValue("answerMode", prefs.answerMode);
  }
  syncUiPreferences();

  async function refreshStatus() {
    try {
      const status = await api("/api/status");
      renderPills(status);
      const sourceStatus = document.getElementById("sourceStatus");
      if (sourceStatus) sourceStatus.textContent = JSON.stringify(status, null, 2);
      renderSourceSummary(status);
      renderPipeline(status);
      const railModel = document.getElementById("railModelName");
      if (railModel) {
        const modelName = status?.llm?.model || "configured";
        if (railModel.hasAttribute("data-rail-model")) {
          railModel.innerHTML = `Local voice: ${escapeHtml(modelName)} &#8594;`;
        } else {
          railModel.textContent = modelName;
        }
      }
    } catch (err) { toast(`Status failed: ${err.message}`); }
  }

  function renderPipeline(status) {
    const map = {
      ingestion: (status && status.metadata_db && status.metadata_db.documents != null)
        ? { ok: status.metadata_db.documents > 0, note: status.metadata_db.documents + " documents" }
        : { ok: false, note: "No data yet" },
      keyword: lifecycleState(status && status.index_lifecycle && status.index_lifecycle.keyword_index_status),
      vector: lifecycleState(status && status.index_lifecycle && status.index_lifecycle.vector_index_status),
      model: (status && status.llm && status.llm.model)
        ? { ok: true, note: status.llm.model }
        : { ok: false, note: "Not configured" },
    };
    Object.keys(map).forEach((key) => {
      const info = map[key];
      const note = document.querySelector('[data-pipe="' + key + '"]');
      const state = document.querySelector('[data-pipe-state="' + key + '"]');
      if (note) note.textContent = info.note;
      if (state) {
        state.className = info.ok ? "health-ok" : "health-warn";
        state.innerHTML = '<i class="hdot"></i>' + (info.ok ? "Ready" : "Pending");
      }
    });
  }

  function lifecycleState(s) {
    if (s === "ready" || s === "current" || s === "completed") return { ok: true, note: "Ready" };
    if (!s || s === "unknown") return { ok: false, note: "Not built" };
    return { ok: false, note: humanStatus(s) };
  }

  function renderSourceSummary(status) {
    const target = document.getElementById("sourceSummary"); if (!target) return;
    const cards = [
      ["Documents", status?.metadata_db?.documents ?? "-", "Files currently indexed"],
      ["Evidence strands", status?.metadata_db?.chunks ?? "-", "Passages, rows, captions, and pages Ariadne can trace"],
      ["Thread readiness", humanStatus(status?.index_lifecycle?.overall_status), "Name/code matching and meaning search"],
      ["Local voice", status?.llm?.model || "local model", "Served by the host machine"],
      ["OCR", status?.ocr?.enabled ? "Enabled" : "Off", "Local text extraction from images"],
      ["Second sight", status?.vision?.enabled ? "Enabled" : "Off", "Local help for images and scanned pages"],
    ];
    target.innerHTML = cards.map(([label, value, help]) => `<article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p class="setting-help">${escapeHtml(help)}</p></article>`).join("");
  }

  async function refreshChats() {
    try {
      const data = await api("/api/chats");
      currentChats = data.chats || [];
      renderChatList();
    } catch (err) {
      toast(`Could not load chats: ${err.message}`);
    }
  }

  function chatListItemHtml(chat, compact = false) {
    return `
      <button class="chat-list-item ${compact ? "compact-thread" : "archive-thread"} ${chat.chat_id === activeChatId ? "active" : ""}" type="button" data-chat-id="${escapeHtml(chat.chat_id)}">
        <span class="chat-title">${escapeHtml(chat.title || "New thread")}</span>
        <span class="chat-meta">${escapeHtml(chat.message_count || 0)} messages${chat.updated_at ? ` · ${escapeHtml(chat.updated_at)}` : ""}</span>
        <span class="delete-chat" data-delete-chat="${escapeHtml(chat.chat_id)}" title="Delete thread">×</span>
      </button>
    `;
  }

  function bindThreadList(root) {
    if (!root) return;
    root.querySelectorAll("[data-chat-id]").forEach(button => {
      button.addEventListener("click", event => {
        if (event.target.closest("[data-delete-chat]")) return;
        loadChat(button.dataset.chatId);
        switchView("chat", { freshHome: false });
      });
    });
    root.querySelectorAll("[data-delete-chat]").forEach(button => {
      button.addEventListener("click", event => {
        event.preventDefault(); event.stopPropagation();
        deleteChat(button.dataset.deleteChat);
      });
    });
  }

  function renderChatList() {
    const list = document.getElementById("chatList");
    if (!list) return;
    if (!currentChats.length) {
      list.innerHTML = `<div class="chat-empty">No past threads yet.</div>`;
    } else {
      const recent = currentChats.slice(0, 4);
      list.innerHTML = recent.map(chat => chatListItemHtml(chat, true)).join("") + (currentChats.length > 4 ? `<button class="chat-list-item view-all-threads" type="button" data-open-thread-archive>Open all ${currentChats.length} threads</button>` : "");
      list.querySelector("[data-open-thread-archive]")?.addEventListener("click", () => document.querySelector('[data-view="threads"]')?.click());
    }
    bindThreadList(list);
    renderThreadArchive();
  }

  function renderThreadArchive() {
    const full = document.getElementById("fullThreadList");
    if (!full) return;
    if (!currentChats.length) {
      full.innerHTML = `<div class="chat-empty">No past threads yet. Begin a new path from the sidebar.</div>`;
      return;
    }
    full.innerHTML = currentChats.map(chat => chatListItemHtml(chat, false)).join("");
    bindThreadList(full);
  }

  function showWelcome() {
    const welcome = document.getElementById("welcomeState");
    const panel = document.getElementById("answerPanel");
    if (panel) panel.innerHTML = "";
    if (welcome) welcome.style.display = "grid";
  }

  function isFreshHomeThread() {
    const panel = document.getElementById("answerPanel");
    const welcome = document.getElementById("welcomeState");
    return !activeChatId && (!panel || panel.children.length === 0) && (!welcome || welcome.style.display !== "none");
  }

  function startFreshHomeThread(options = {}) {
    if (isFreshHomeThread()) {
      if (options.focus) document.getElementById("chatQuery")?.focus();
      return false;
    }
    activeChatId = null;
    localStorage.removeItem("ariadne.activeChatId");
    showWelcome();
    renderChatList();
    if (options.focus) document.getElementById("chatQuery")?.focus();
    return true;
  }

  function newChat() {
    const changed = startFreshHomeThread({ focus: true });
    switchView("chat", { freshHome: false });
    if (changed) toast("A fresh thread is ready");
  }

  async function loadChat(chatId) {
    if (!chatId) return;
    try {
      const data = await api(`/api/chats/${encodeURIComponent(chatId)}`);
      activeChatId = chatId;
      renderChatList();
      renderConversation(data.chat);
    } catch (err) {
      toast(`Could not open thread: ${err.message}`);
      if (activeChatId === chatId) { activeChatId = null; localStorage.removeItem("ariadne.activeChatId"); }
    }
  }

  async function deleteChat(chatId) {
    if (!chatId) return;
    try {
      await api(`/api/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });
      if (activeChatId === chatId) startFreshHomeThread();
      await refreshChats();
      toast("Thread deleted");
    } catch (err) { toast(`Delete failed: ${err.message}`); }
  }

  function renderConversation(chat) {
    const welcome = document.getElementById("welcomeState");
    const panel = document.getElementById("answerPanel");
    if (!panel || !welcome) return;
    const messages = chat?.messages || [];
    if (!messages.length) { showWelcome(); return; }
    welcome.style.display = "none";
    panel.innerHTML = messages.map(message => {
      if (message.role === "user") return userMessageHtml(message.content);
      const payload = message.payload || { answer: message.content, status: "ok", confidence: "medium", evidence: [], source_documents: [] };
      return answerCardHtml(payload);
    }).join("");
    bindCitationButtons(panel);
    panel.scrollTop = panel.scrollHeight;
  }

  function userMessageHtml(text) {
    return `<article class="message-row user-row"><div class="user-bubble">${escapeHtml(text)}</div></article>`;
  }

  function citationMapFromData(data) {
    const direct = data.citation_source_map || {};
    if (Object.keys(direct).length) return direct;
    const map = {};
    (data.source_documents || []).forEach(doc => {
      (doc.citation_labels || []).forEach(label => {
        map[label] = { display_label: doc.display_label, display_name: doc.display_name, source_index: doc.source_index };
      });
    });
    return map;
  }

  function documentSourcesFromData(data) {
    if (data.source_documents?.length) return data.source_documents;
    const grouped = new Map();
    (data.evidence || []).forEach(item => {
      const key = item.document_id || item.source_file || item.title || item.citation_label;
      if (!grouped.has(key)) grouped.set(key, { source_index: grouped.size + 1, display_label: `Source ${grouped.size + 1}`, display_name: item.source_file || item.title || "Local source", title: item.title, source_file: item.source_file, source_systems: [], record_types: [], citation_labels: [], chunk_count: 0, preview: item.text_preview || "" });
      const doc = grouped.get(key); doc.chunk_count += 1;
      if (item.source_system && !doc.source_systems.includes(item.source_system)) doc.source_systems.push(item.source_system);
      if (item.record_type && !doc.record_types.includes(item.record_type)) doc.record_types.push(item.record_type);
      if (item.citation_label && !doc.citation_labels.includes(item.citation_label)) doc.citation_labels.push(item.citation_label);
    });
    return [...grouped.values()];
  }

  function inferAnswerTitle(data) {
    const text = String(data?.answer || "").trim();
    if (data?.status === "no_answer") return "Not enough evidence";
    if (data?.status === "error") return "Could not complete";
    if (/^subject:/im.test(text) || /\bdear\s+/i.test(text)) return "Draft";
    if (/^\s*\|.+\|/m.test(text)) return "Results";
    return "Response";
  }

  function answerCardHtml(data) {
    const prefs = loadPreferences();
    const statusClass = data.status || "ok";
    const confidenceClass = data.confidence || "low";
    const displayText = stripDisplaySections(data.answer || "");
    const sourceDocs = documentSourcesFromData(data);
    const citationMap = citationMapFromData({ ...data, source_documents: sourceDocs });
    const showSources = prefs.sourceVisibility !== "compact";
    const showTechnical = prefs.sourceVisibility === "technical";
    const sourceCards = sourceDocs.map((doc, index) => sourceDocumentCard(doc, index, showTechnical)).join("");
    const sourcePills = sourceDocs.map(doc => `<button class="source-pill" type="button" data-source-index="${escapeHtml(doc.source_index)}">${escapeHtml(doc.display_label)} <span>${escapeHtml(doc.display_name || "Local source")}</span></button>`).join("");

    return `<article class="answer-bubble assistant-message">
      <div class="answer-top"><div><p class="eyebrow">Ariadne</p><h2>${escapeHtml(inferAnswerTitle(data))}</h2></div><div class="status-pills"><span class="pill ${statusClass}">${escapeHtml(humanStatus(data.status))}</span><span class="pill ${confidenceClass}">${escapeHtml(humanStatus(data.confidence))}</span></div></div>
      <div class="answer-body">${markdownLite(displayText, citationMap)}</div>
      <div class="sources-block"><h3>Source trail</h3><div class="source-pills-row">${sourcePills || '<span class="muted-text">No source trail was needed for this response.</span>'}</div></div>
      ${showSources ? `<details class="sources-block" ${prefs.sourceVisibility === "sources" ? "open" : ""}><summary class="clean-summary">View source trail (${sourceDocs.length})</summary><div class="cards-grid">${sourceCards || '<p class="subtitle">No source documents surfaced.</p>'}</div></details>` : ""}
      ${showTechnical ? `<details class="sources-block" open><summary class="clean-summary">Technical details</summary><pre class="json-block">${escapeHtml(JSON.stringify({ validation: data.validation, retrieval_diagnostics: data.retrieval_diagnostics, latency_ms: data.api_latency_ms, citations: data.citations }, null, 2))}</pre></details>` : ""}
    </article>`;
  }

  function bindCitationButtons(root) {
    root.querySelectorAll("[data-source-index]").forEach(button => button.addEventListener("click", () => {
      const sourceIndex = button.dataset.sourceIndex;
      const card = root.querySelector(`.document-source-card[data-source-index="${CSS.escape(sourceIndex)}"]`);
      if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    }));
  }

  function setLoadingAnswer(userText) {
    const welcome = document.getElementById("welcomeState");
    const panel = document.getElementById("answerPanel");
    welcome.style.display = "none";
    panel.insertAdjacentHTML("beforeend", userMessageHtml(userText));
    panel.insertAdjacentHTML("beforeend", `<article class="answer-bubble loading-card" id="loadingAnswer"><div class="answer-top"><div><p class="eyebrow">Following the thread</p><h2>Tracing the path...</h2><p class="subtitle">Ariadne checks the current thread first, then turns to local sources only when new evidence is needed.</p></div><span class="pill running">Thinking</span></div></article>`);
    panel.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function renderAnswer(data) {
    const loading = document.getElementById("loadingAnswer");
    if (loading) loading.outerHTML = answerCardHtml(data);
    else document.getElementById("answerPanel").insertAdjacentHTML("beforeend", answerCardHtml(data));
    bindCitationButtons(document.getElementById("answerPanel"));
    document.getElementById("answerPanel").lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  async function askQuestion(event) {
    event?.preventDefault();
    const button = document.getElementById("askButton");
    const input = document.getElementById("chatQuery");
    const query = input.value.trim();
    if (!query) { toast("Give Ariadne a thread to follow first"); return; }
    const prefs = loadPreferences();
    const body = { chat_id: activeChatId, query, top_k: getTopKFromPreferences(), source_system: getChoiceValue("chatSource") || null, record_type: null, show_evidence: true, answer_mode: getChoiceValue("answerMode") || prefs.answerMode, preview_chars: prefs.sourceVisibility === "technical" ? 1600 : 1000 };
    try {
      button.disabled = true; button.classList.add("is-busy");
      input.value = ""; input.dispatchEvent(new Event("input"));
      setLoadingAnswer(query);
      const data = await api("/api/chat", { method: "POST", body: JSON.stringify(body) });
      activeChatId = data.chat_id || activeChatId;
      renderAnswer(data);
      await refreshChats();
      await refreshStatus();
    } catch (err) {
      toast(`Answer failed: ${err.message}`);
      const loading = document.getElementById("loadingAnswer");
      if (loading) loading.outerHTML = `<article class="answer-bubble"><h2>Something went wrong</h2><p class="subtitle">${escapeHtml(err.message)}</p></article>`;
    } finally { button.disabled = false; button.classList.remove("is-busy"); }
  }

  async function runSearch() {
    const button = document.getElementById("searchButton");
    const body = { query: document.getElementById("searchQuery").value.trim(), top_k: getTopKFromPreferences(), preview_chars: 1000 };
    if (!body.query) { toast("Enter something to search"); return; }
    try {
      setBusy(button, true, "Searching...");
      const data = await api("/api/search", { method: "POST", body: JSON.stringify(body) });
      const target = document.getElementById("searchResults");
      target.innerHTML = (data.results || []).map((item, index) => evidenceCard(item, index)).join("") || `<div class="section-card glass"><h2>No matching source trail found</h2><p class="subtitle">Try a source name, ID, model, person, code, or phrase from the documents.</p></div>`;
      toast(`${data.result_count || 0} source strands found`);
    } catch (err) { toast(`Search failed: ${err.message}`); } finally { setBusy(button, false); }
  }

  function autoGrowTextarea() {
    const textarea = document.getElementById("chatQuery"); if (!textarea) return;
    const resize = () => { textarea.style.height = "auto"; textarea.style.height = Math.min(textarea.scrollHeight, 180) + "px"; };
    textarea.addEventListener("input", resize); resize();
  }

  document.getElementById("chatForm")?.addEventListener("submit", askQuestion);
  document.getElementById("newChatButton")?.addEventListener("click", newChat);
  document.getElementById("newChatFromThreads")?.addEventListener("click", newChat);

  document.getElementById("deleteAllThreads")?.addEventListener("click", async () => {
    if (!currentChats.length) { toast("No threads to delete."); return; }
    const confirmed = window.confirm(`Delete all ${currentChats.length} thread${currentChats.length === 1 ? "" : "s"}? This cannot be undone.`);
    if (!confirmed) return;
    try {
      const res = await api("/api/chats", { method: "DELETE" });
      if (activeChatId) startFreshHomeThread();
      await refreshChats();
      toast(`${res.count ?? "All"} thread${res.count === 1 ? "" : "s"} deleted.`);
    } catch (err) { toast(`Failed to clear threads: ${err.message}`); }
  });
  document.getElementById("searchButton")?.addEventListener("click", runSearch);
  document.getElementById("chatQuery")?.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") askQuestion(event); });
  window.addEventListener("ariadne:settings-saved", () => { syncUiPreferences(); refreshStatus(); });
  window.addEventListener("rags:settings-saved", () => { syncUiPreferences(); refreshStatus(); });

  autoGrowTextarea();
  await refreshStatus();
  await refreshChats();
  startFreshHomeThread();
})();
