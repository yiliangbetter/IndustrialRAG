const $ = (sel) => document.querySelector(sel);

const envForm = $("#env-form");
const envStatus = $("#env-status");
const btnSaveEnv = $("#btn-save-env");
const envLoadProgress = $("#env-load-progress");
const envLoadProgressBar = $("#env-load-progress-bar");
const envLoadProgressLabel = $("#env-load-progress-label");
const envLoadLog = $("#env-load-log");
const checkRuntime = $("#check-runtime");
const checkModels = $("#check-models");
const ragStatus = $("#rag-status");
const finishChecklist = $("#finish-checklist");
const finishStatus = $("#finish-status");
const btnFinish = $("#btn-finish");
const btnReloadRag = $("#btn-reload-rag");
const ragLoadProgress = $("#rag-load-progress");
const ragLoadProgressBar = $("#rag-load-progress-bar");
const ragLoadProgressLabel = $("#rag-load-progress-label");
const ragLoadLog = $("#rag-load-log");
const btnIngest = $("#btn-ingest");
const btnStopIngest = $("#btn-stop-ingest");
const fileInput = $("#file-input");
const uploadZone = $("#upload-zone");
const fileList = $("#file-list");
const fileListToolbar = $("#file-list-toolbar");
const fileListCount = $("#file-list-count");
const btnClearFiles = $("#btn-clear-files");
const ingestGateHint = $("#ingest-gate-hint");
const ingestStatus = $("#ingest-status");
const ingestProgress = $("#ingest-progress");
const ingestProgressBar = $("#ingest-progress-bar");
const ingestProgressLabel = $("#ingest-progress-label");
const ingestLog = $("#ingest-log");
const setupSteps = $("#setup-steps");

let pendingFiles = [];
let latestStatus = null;
let ingestBusy = false;
let ingestStopping = false;
let engineLoadBusy = false;
let ragEnsurePromise = null;

const ENV_SAVE_STEPS = [
  { label: "保存配置到本地…", log: "写入 config/.env", pct: 12 },
  { label: "应用环境变量…", log: "设置 HF 模型路径与工作目录", pct: 28 },
  { label: "加载向量模型 (bge-m3)…", log: "从本地缓存加载 embedding 模型（首次较慢）", pct: 52 },
  { label: "加载 Rerank 模型…", log: "从本地缓存加载 reranker 模型", pct: 72 },
  { label: "初始化 RAG 引擎…", log: "构建 LightRAG 实例并挂载存储", pct: 88 },
];

const RAG_RELOAD_STEPS = [
  { label: "重新读取配置…", log: "加载 config/.env", pct: 18 },
  { label: "卸载旧引擎实例…", log: "释放上一轮 RAG 资源", pct: 32 },
  { label: "加载向量模型 (bge-m3)…", log: "从本地缓存加载 embedding 模型", pct: 55 },
  { label: "加载 Rerank 模型…", log: "从本地缓存加载 reranker 模型", pct: 74 },
  { label: "初始化 RAG 引擎…", log: "构建 LightRAG 实例", pct: 90 },
];

const STEP_LABELS = [
  "LLM 配置",
  "环境检测",
  "文档灌库",
  "完成",
];

function renderSteps(status) {
  const flags = [
    status?.env?.ok,
    status?.runtime_deps_ok && status?.bundled_models_ok && status?.rag_ready,
    status?.knowledge_base_ok,
    status?.can_finish_setup,
  ];
  setupSteps.innerHTML = STEP_LABELS.map((label, i) => {
    let cls = "setup-step-pill";
    if (flags[i]) cls += " done";
    if (i === flags.findIndex((f) => !f)) cls += " active";
    return `<span class="${cls}">${i + 1}. ${label}</span>`;
  }).join("");
}

function renderCheckList(el, items) {
  el.innerHTML = items
    .map(
      (item) =>
        `<li><span class="check-icon ${item.ok ? "ok" : "bad"}">${item.ok ? "✓" : "✗"}</span>${item.label}</li>`
    )
    .join("");
}

function renderEnvForm(fields) {
  envForm.innerHTML = fields
    .map((f) => {
      const inputType = f.type === "password" ? "password" : "text";
      if (f.type === "select") {
        const opts = (f.options || [])
          .map(
            (o) =>
              `<option value="${o}" ${o === f.value ? "selected" : ""}>${o}</option>`
          )
          .join("");
        return `
          <label>
            <span class="label-text">${f.label}</span>
            <p class="field-hint">${f.hint || ""}</p>
            <select name="${f.key}" ${f.required ? "required" : ""}>${opts}</select>
          </label>`;
      }
      return `
        <label>
          <span class="label-text">${f.label}</span>
          <p class="field-hint">${f.hint || ""}</p>
          <input name="${f.key}" type="${inputType}" value="${escapeAttr(f.value || "")}"
            placeholder="${escapeAttr(f.placeholder || "")}" ${f.required ? "required" : ""} />
        </label>`;
    })
    .join("");
}

function escapeAttr(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function collectEnvForm() {
  const values = {};
  envForm.querySelectorAll("[name]").forEach((el) => {
    values[el.name] = el.value.trim();
  });
  return values;
}

async function refreshStatus() {
  const res = await fetch("/api/setup/status");
  if (!res.ok) throw new Error(`status ${res.status}`);
  latestStatus = await res.json();
  renderSteps(latestStatus);

  renderCheckList(checkRuntime, latestStatus.runtime_deps || []);
  renderCheckList(checkModels, latestStatus.bundled_models || []);

  if (latestStatus.rag_ready) {
    ragStatus.textContent = `RAG 引擎已就绪 · 工作目录：${latestStatus.working_dir}`;
    ragStatus.className = "hint status-ok";
  } else if (latestStatus.rag_error) {
    ragStatus.textContent = `RAG 未就绪：${latestStatus.rag_error}`;
    ragStatus.className = "hint error";
  } else {
    ragStatus.textContent = "请先保存 LLM 配置以加载 RAG 引擎。";
    ragStatus.className = "hint";
  }

  const finishItems = [
    { ok: latestStatus.env?.ok, text: "LLM 配置已保存" },
    { ok: latestStatus.bundled_models_ok, text: "内置模型就绪（向量 / Rerank / PDF 解析）" },
    { ok: latestStatus.rag_ready, text: "RAG 引擎加载成功" },
    {
      ok: latestStatus.knowledge_base_ok,
      text: latestStatus.knowledge_base_ok
        ? `知识库已灌库（${latestStatus.kb_doc_count || 0} 篇文档）`
        : latestStatus.kb_partial
          ? `知识库未完成灌库（${latestStatus.kb_doc_count || 0} 篇残留，需重新灌库）`
          : "知识库未完成灌库",
    },
  ];
  finishChecklist.innerHTML = finishItems
    .map((it) => `<li class="${it.ok ? "ok" : "bad"}">${it.ok ? "✓" : "○"} ${it.text}</li>`)
    .join("");

  btnFinish.disabled = !latestStatus.can_finish_setup;
  updateIngestControls();
}

async function ensureRagReady() {
  if (latestStatus?.rag_ready) return true;
  if (!latestStatus?.env?.ok) return false;
  if (engineLoadBusy) return Boolean(latestStatus?.rag_ready);
  if (ragEnsurePromise) return ragEnsurePromise;

  ragEnsurePromise = (async () => {
    updateIngestControls();
    try {
      const data = await runStagedProgressTask({
        steps: RAG_RELOAD_STEPS,
        progressEl: ragLoadProgress,
        barEl: ragLoadProgressBar,
        labelEl: ragLoadProgressLabel,
        logEl: ragLoadLog,
        buttons: [btnReloadRag],
        forms: [envForm],
        task: async () => {
          const res = await fetch("/api/setup/reload-rag", { method: "POST" });
          const payload = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
          return payload;
        },
      });
      await refreshStatus();
      return Boolean(data.rag_ready ?? latestStatus?.rag_ready);
    } catch (err) {
      if (ingestGateHint) {
        ingestGateHint.textContent = `RAG 引擎未就绪：${err.message || err}（请检查步骤 1 配置，或在步骤 2 点击「重新加载 RAG 引擎」）`;
        ingestGateHint.className = "hint ingest-gate-hint error";
      }
      return false;
    } finally {
      ragEnsurePromise = null;
    }
  })();

  return ragEnsurePromise;
}

function updateIngestControls() {
  const hasFiles = pendingFiles.length > 0;
  const ragReady = Boolean(latestStatus?.rag_ready);
  const envOk = Boolean(latestStatus?.env?.ok);

  if (fileListToolbar) {
    fileListToolbar.classList.toggle("hidden", !hasFiles);
  }
  if (fileListCount) {
    fileListCount.textContent = hasFiles ? `已选 ${pendingFiles.length} 个文件` : "";
  }
  if (btnClearFiles) {
    btnClearFiles.disabled = !hasFiles || ingestBusy;
  }

  btnIngest.disabled = !hasFiles || !ragReady || ingestBusy || engineLoadBusy;
  if (btnStopIngest) {
    btnStopIngest.classList.toggle("hidden", !ingestBusy);
    btnStopIngest.disabled = !ingestBusy || ingestStopping;
  }

  if (!ingestGateHint) return;

  if (ingestBusy) {
    ingestGateHint.textContent = "灌库进行中，请勿关闭窗口…";
    ingestGateHint.className = "hint ingest-gate-hint";
    return;
  }

  if (!hasFiles) {
    ingestGateHint.textContent = "";
    ingestGateHint.className = "hint ingest-gate-hint";
  } else if (!envOk) {
    ingestGateHint.textContent = "请先在步骤 1 保存 LLM 配置，再开始灌库。";
    ingestGateHint.className = "hint ingest-gate-hint error";
  } else if (!ragReady) {
    if (ragEnsurePromise) {
      ingestGateHint.textContent = "正在加载 RAG 引擎…";
    } else if (latestStatus?.rag_error) {
      ingestGateHint.textContent = `RAG 引擎未就绪：${latestStatus.rag_error}（请在步骤 2 点击「重新加载 RAG 引擎」）`;
    } else {
      ingestGateHint.textContent = "正在等待 RAG 引擎加载…";
    }
    ingestGateHint.className = "hint ingest-gate-hint error";
  } else {
    ingestGateHint.textContent = "文件已就绪，可以开始灌库。";
    ingestGateHint.className = "hint ingest-gate-hint status-ok";
  }
}

async function loadEnvForm() {
  const res = await fetch("/api/setup/env");
  if (!res.ok) throw new Error(`env ${res.status}`);
  const data = await res.json();
  renderEnvForm(data.fields || []);
}

envForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (engineLoadBusy) return;

  engineLoadBusy = true;
  envStatus.textContent = "正在保存并加载引擎，请勿关闭窗口…";
  envStatus.className = "hint";

  try {
    const data = await runStagedProgressTask({
      steps: ENV_SAVE_STEPS,
      progressEl: envLoadProgress,
      barEl: envLoadProgressBar,
      labelEl: envLoadProgressLabel,
      logEl: envLoadLog,
      buttons: [btnSaveEnv, btnReloadRag],
      forms: [envForm],
      task: async () => {
        const res = await fetch("/api/setup/env", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ values: collectEnvForm() }),
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
        return payload;
      },
    });
    envStatus.textContent = data.rag_ready
      ? "配置已保存，RAG 引擎已加载。"
      : `配置已保存，但引擎未就绪：${data.rag_error || "未知错误"}`;
    envStatus.className = data.rag_ready ? "hint status-ok" : "hint error";
    await refreshStatus();
  } catch (err) {
    envStatus.textContent = `保存失败：${err.message || err}`;
    envStatus.className = "hint error";
  } finally {
    engineLoadBusy = false;
  }
});

btnReloadRag.addEventListener("click", async () => {
  if (engineLoadBusy) return;

  engineLoadBusy = true;
  ragStatus.textContent = "正在重新加载 RAG 引擎…";
  ragStatus.className = "hint";

  try {
    await runStagedProgressTask({
      steps: RAG_RELOAD_STEPS,
      progressEl: ragLoadProgress,
      barEl: ragLoadProgressBar,
      labelEl: ragLoadProgressLabel,
      logEl: ragLoadLog,
      buttons: [btnReloadRag, btnSaveEnv],
      forms: [envForm],
      task: async () => {
        const res = await fetch("/api/setup/reload-rag", { method: "POST" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
        return payload;
      },
    });
    await refreshStatus();
  } catch (err) {
    ragStatus.textContent = `加载失败：${err.message || err}`;
    ragStatus.className = "hint error";
  } finally {
    engineLoadBusy = false;
  }
});

function renderFileList() {
  renderPendingFileList(fileList, pendingFiles, {
    readOnly: ingestBusy,
    onRemove: (index) => {
      pendingFiles.splice(index, 1);
      renderFileList();
      void ensureRagReady();
    },
  });
  updateIngestControls();
}

function clearPendingFiles() {
  pendingFiles = [];
  renderFileList();
}

function addFiles(list) {
  for (const f of list) {
    if (!pendingFiles.some((p) => p.name === f.name && p.size === f.size)) {
      pendingFiles.push(f);
    }
  }
  renderFileList();
  void ensureRagReady();
}

btnClearFiles?.addEventListener("click", () => {
  if (ingestBusy) return;
  clearPendingFiles();
});

fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});

btnIngest.addEventListener("click", async () => {
  if (!pendingFiles.length || ingestBusy) return;
  if (!latestStatus?.rag_ready) {
    const ready = await ensureRagReady();
    if (!ready) return;
  }

  ingestBusy = true;
  ingestStopping = false;
  updateIngestControls();
  ingestStatus.textContent = "灌库进行中，请稍候…";
  if (ingestProgressBar) ingestProgressBar.classList.remove("error");
  try {
    const data = await streamIngest({
      files: pendingFiles,
      logEl: ingestLog,
      progressEl: ingestProgress,
      progressBarEl: ingestProgressBar,
      progressLabelEl: ingestProgressLabel,
      statusEl: ingestStatus,
      onActiveChange: (active) => {
        ingestBusy = active;
        if (!active) ingestStopping = false;
        updateIngestControls();
      },
    });
    if (data.cancelled) {
      ingestStatus.textContent = "已停止灌库并清空知识库";
      pendingFiles = [];
    } else {
      ingestStatus.textContent = `完成：成功 ${data.ok}，失败 ${data.fail}`;
      if (data.fail === 0) {
        pendingFiles = [];
      }
    }
    await refreshStatus();
  } catch (err) {
    ingestStatus.textContent = `灌库失败：${err.message || err}`;
    appendIngestLog(ingestLog, `错误：${err.message || err}`);
  } finally {
    ingestBusy = false;
    ingestStopping = false;
    renderFileList();
  }
});

btnStopIngest?.addEventListener("click", async () => {
  if (!ingestBusy || ingestStopping) return;
  ingestStopping = true;
  updateIngestControls();
  try {
    await requestStopIngest({ logEl: ingestLog, statusEl: ingestStatus });
  } catch (err) {
    ingestStopping = false;
    updateIngestControls();
    appendIngestLog(ingestLog, `停止失败：${err.message || err}`);
    ingestStatus.textContent = `停止失败：${err.message || err}`;
  }
});

btnFinish.addEventListener("click", async () => {
  finishStatus.textContent = "正在完成…";
  try {
    const res = await fetch("/api/setup/complete", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    window.location.href = data.redirect || "/";
  } catch (err) {
    finishStatus.textContent = `无法完成：${err.message || err}`;
  }
});

(async function init() {
  try {
    await loadEnvForm();
    await refreshStatus();
    if (latestStatus?.env?.ok && !latestStatus?.rag_ready) {
      await ensureRagReady();
    }
    setInterval(refreshStatus, 8000);
  } catch (err) {
    envStatus.textContent = `初始化失败：${err.message || err}`;
  }
})();
