(async () => {
  const { api, toast, setBusy, renderPills, escapeHtml, initSidebar, initSettingsDrawer, humanStatus, renderAdminConfigEditor, collectConfigOverrides } = window.RAGS;

  initSidebar();
  await initSettingsDrawer();

  const views = {
    overview: document.getElementById("admin-overview"),
    data: document.getElementById("admin-data"),
    config: document.getElementById("admin-config"),
    jobs: document.getElementById("admin-jobs"),
    logs: document.getElementById("admin-logs"),
  };

  async function activateAdminView(view, updateHash = true) {
    const selected = views[view] ? view : "overview";
    document.querySelectorAll("[data-admin-view]").forEach(x => x.classList.toggle("active", x.dataset.adminView === selected));
    Object.entries(views).forEach(([name, node]) => node?.classList.toggle("active", name === selected));
    if (updateHash) history.replaceState(null, "", selected === "overview" ? location.pathname : `${location.pathname}#${selected}`);
    if (selected === "jobs") await refreshJobs();
    if (selected === "logs") await refreshLogs();
    if (selected === "config") await loadAdminConfig();
    if (selected === "overview") await refreshStatus();
  }

  document.querySelectorAll("[data-admin-view]").forEach(btn => {
    btn.addEventListener("click", async () => activateAdminView(btn.dataset.adminView));
  });
  window.addEventListener("hashchange", () => activateAdminView(location.hash.replace("#", "") || "overview", false));

  async function refreshStatus() {
    const status = await api("/api/status");
    renderPills(status);
    const json = document.getElementById("adminStatusJson");
    if (json) json.textContent = JSON.stringify(status, null, 2);
    const metrics = [
      ["Documents", status.metadata_db?.documents ?? "-", "Files Ariadne can see"],
      ["Evidence strands", status.metadata_db?.chunks ?? "-", "Passages, rows, captions, and pages"],
      ["Name/code path", humanStatus(status.keyword_index?.status || status.index_lifecycle?.keyword_index?.status || "unknown"), "Exact-match index readiness"],
      ["Meaning path", humanStatus(status.index_lifecycle?.vector_index?.status || "unknown"), "Semantic index readiness"],
      ["Tracked files", status.file_tracking?.registry?.total_files_tracked ?? "-", "The local file registry"],
      ["Overall", humanStatus(status.index_lifecycle?.overall_status || "unknown"), "Thread readiness"],
    ];
    document.getElementById("statusCards").innerHTML = metrics.map(([label, value, help]) => `
      <article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p class="setting-help">${escapeHtml(help)}</p></article>
    `).join("");
  }

  async function postAdmin(url, body = {}) {
    return api(url, { method: "POST", body: JSON.stringify(body) });
  }

  function friendlyPlanSummary(data) {
    const plan = data?.plan || data;
    const counts = [
      ["New strands", plan?.files_new?.length ?? plan?.new_files?.length ?? 0],
      ["Changed strands", plan?.files_changed?.length ?? plan?.changed_files?.length ?? 0],
      ["Missing strands", plan?.files_missing?.length ?? plan?.missing_files?.length ?? 0],
      ["Unchanged strands", plan?.files_unchanged?.length ?? plan?.unchanged_files?.length ?? 0],
    ];
    return `${counts.map(([label, value]) => `${label}: ${value}`).join("\n")}\n\n${JSON.stringify(data, null, 2)}`;
  }

  async function planIngest() {
    const btn = document.getElementById("planIngestBtn");
    try {
      setBusy(btn, true, "Inspecting...");
      const data = await postAdmin("/api/admin/plan-ingest");
      document.getElementById("adminOutput").textContent = friendlyPlanSummary(data);
      toast("The doorway has been inspected");
    } catch (err) { toast(`Doorway inspection failed: ${err.message}`); }
    finally { setBusy(btn, false); }
  }

  async function startJob(url, body, button, message) {
    try {
      setBusy(button, true, "Weaving...");
      const job = await postAdmin(url, body);
      document.getElementById("adminOutput").textContent = JSON.stringify(job, null, 2);
      toast(message);
      await refreshJobs();
      pollJob(job.job_id);
    } catch (err) { toast(`Work failed: ${err.message}`); }
    finally { setBusy(button, false); }
  }

  async function pollJob(jobId) {
    if (!jobId) return;
    for (let i = 0; i < 240; i++) {
      const job = await api(`/api/admin/jobs/${jobId}`);
      document.getElementById("adminOutput").textContent = JSON.stringify(job, null, 2);
      await refreshJobs();
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        await refreshStatus();
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  async function refreshJobs() {
    const data = await api("/api/admin/jobs?limit=12");
    document.getElementById("jobsList").innerHTML = (data.jobs || []).map(job => `
      <article class="evidence-card">
        <div class="evidence-title"><strong>${escapeHtml(job.name)}</strong><span class="pill ${escapeHtml(job.status)}">${escapeHtml(humanStatus(job.status))}</span></div>
        <div class="evidence-meta"><span class="code-pill">${escapeHtml(job.job_id)}</span><span class="code-pill">${escapeHtml(job.created_at || "")}</span></div>
        ${job.error ? `<p class="subtitle">${escapeHtml(job.error)}</p>` : ""}
      </article>
    `).join("") || `<p class="subtitle">No work is running in the loom yet.</p>`;
  }

  async function loadAdminConfig() {
    const container = document.getElementById("adminConfigContent");
    try {
      await renderAdminConfigEditor(container);
      bindAdminConfigActions();
    } catch (err) {
      container.innerHTML = `<p class="subtitle">Ariadne could not read the loom settings: ${escapeHtml(err.message)}</p>`;
    }
  }

  function bindAdminConfigActions() {
    document.getElementById("detectAdminModels")?.addEventListener("click", detectAdminModels);
  }

  async function detectAdminModels() {
    const target = document.getElementById("adminModelDetection");
    const container = document.getElementById("adminConfigContent");
    const btn = document.getElementById("detectAdminModels");
    const llmUrl = container?.querySelector('[data-setting-path="llm.base_url"]')?.value;
    const visionUrl = container?.querySelector('[data-setting-path="vision.base_url"]')?.value;
    if (target) target.textContent = "Listening for local voices from this machine...";
    try {
      setBusy(btn, true, "Listening...");
      const data = await api("/api/models/detect", { method: "POST", body: JSON.stringify({ llm_base_url: llmUrl, vision_base_url: visionUrl }) });
      const describe = (label, info) => `${label}: ${info?.models?.length ? info.models.join(", ") : "no local voices found"}`;
      if (target) target.textContent = `${describe("Text", data.llm)}\n${describe("Vision", data.vision)}`;
      toast("Local voices checked");
    } catch (err) {
      if (target) target.textContent = `Listening failed: ${err.message}`;
      toast(`Local voice check failed: ${err.message}`);
    } finally {
      setBusy(btn, false);
    }
  }

  async function saveAdminConfig() {
    const btn = document.getElementById("saveAdminConfigBtn");
    const container = document.getElementById("adminConfigContent");
    const output = document.getElementById("adminConfigOutput");
    try {
      setBusy(btn, true, "Saving...");
      const overrides = collectConfigOverrides(container);
      const data = await api("/api/config/overrides", { method: "PATCH", body: JSON.stringify({ overrides }) });
      output.textContent = JSON.stringify({ status: data.status, message: data.message, overrides: data.overrides }, null, 2);
      toast("Loom settings saved locally");
      await loadAdminConfig();
      await refreshStatus();
      window.dispatchEvent(new CustomEvent("ariadne:settings-saved"));
    } catch (err) {
      toast(`Loom settings save failed: ${err.message}`);
      if (output) output.textContent = `Loom settings save failed: ${err.message}`;
    } finally {
      setBusy(btn, false);
    }
  }

  async function clearAdminConfig() {
    if (!confirm("Restore the baseline and clear the loom settings saved from the UI?")) return;
    const output = document.getElementById("adminConfigOutput");
    try {
      const data = await api("/api/config/overrides", { method: "DELETE" });
      output.textContent = JSON.stringify(data, null, 2);
      toast("Baseline restored");
      await loadAdminConfig();
      await refreshStatus();
      window.dispatchEvent(new CustomEvent("ariadne:settings-saved"));
    } catch (err) { toast(`Could not restore baseline: ${err.message}`); }
  }


  async function clearLogs() {
    if (!confirm("Clear local question echoes? This removes the local query log, not saved threads.")) return;
    const btn = document.getElementById("clearLogsBtn");
    try {
      setBusy(btn, true, "Clearing...");
      const data = await api("/api/admin/query-log", { method: "DELETE" });
      toast(`${data.records_removed || 0} log record${data.records_removed === 1 ? "" : "s"} removed`);
      await refreshLogs();
    } catch (err) { toast(`Could not clear history: ${err.message}`); }
    finally { setBusy(btn, false); }
  }

  async function refreshLogs() {
    const data = await api("/api/admin/query-log?limit=30");
    document.getElementById("queryLog").innerHTML = (data.records || []).map(row => `
      <article class="evidence-card">
        <div class="evidence-title"><strong>${escapeHtml(row.query || row.type || "Question")}</strong><span class="pill ${escapeHtml(row.confidence || row.status || "muted")}">${escapeHtml(humanStatus(row.status || row.confidence || ""))}</span></div>
        <div class="evidence-meta"><span class="code-pill">${escapeHtml(row.timestamp || "")}</span><span class="code-pill">${escapeHtml(row.latency_ms || "-")} ms</span></div>
        <pre class="preview">${escapeHtml(JSON.stringify(row, null, 2))}</pre>
      </article>
    `).join("") || `<p class="subtitle">No question echoes yet.</p>`;
  }

  document.getElementById("planIngestBtn")?.addEventListener("click", planIngest);
  document.getElementById("ingestBtn")?.addEventListener("click", (e) => startJob("/api/admin/ingest", { force: false }, e.currentTarget, "Weaving new strands"));
  document.getElementById("rebuildBtn")?.addEventListener("click", (e) => startJob("/api/admin/rebuild", { fresh: false }, e.currentTarget, "Retying the thread map"));
  document.getElementById("freshRebuildBtn")?.addEventListener("click", (e) => {
    if (confirm("Reweave from the beginning? This clears generated search data and rebuilds the source trail from local files.")) {
      startJob("/api/admin/rebuild", { fresh: true }, e.currentTarget, "Reweaving from the beginning");
    }
  });
  document.getElementById("refreshJobsBtn")?.addEventListener("click", refreshJobs);
  document.getElementById("saveAdminConfigBtn")?.addEventListener("click", saveAdminConfig);
  document.getElementById("reloadConfigBtn")?.addEventListener("click", loadAdminConfig);
  document.getElementById("clearConfigBtn")?.addEventListener("click", clearAdminConfig);
  document.getElementById("refreshLogsBtn")?.addEventListener("click", refreshLogs);
  document.getElementById("clearLogsBtn")?.addEventListener("click", clearLogs);
  document.getElementById("refreshOverviewBtn")?.addEventListener("click", refreshStatus);
  window.addEventListener("ariadne:settings-saved", refreshStatus);

  await activateAdminView(location.hash.replace("#", "") || "overview", false);
})();
