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
const checkLanguage = $("#check-language");
const setupSubtitle = $("#setup-subtitle");
const kbSummary = $("#kb-summary");
const btnClearKb = $("#btn-clear-kb");
const enableMultimodal = $("#enable-multimodal");
const multimodalHint = $("#multimodal-hint");
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
const btnIngestAppend = $("#btn-ingest-append");
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
let statusPollTimer = null;
let engineLoadBusy = false;
let ragEnsurePromise = null;
let multimodalSyncBusy = false;

function envFieldValue(name, fallback = "") {
  const el = envForm?.querySelector(`[name="${name}"]`);
  return (el?.value || fallback).trim();
}

function buildEnvSaveSteps() {
  const embed = latestStatus?.embedding_model || envFieldValue("EMBEDDING_MODEL", "BAAI/bge-m3");
  const rerank = latestStatus?.rerank_model || envFieldValue("RERANK_MODEL", "BAAI/bge-reranker-base");
  return [
    { label: "保存配置到本地…", log: "写入 config/.env", pct: 12 },
    { label: "应用环境变量…", log: "设置 HF 模型路径与工作目录", pct: 28 },
    { label: `加载向量模型 (${embed})…`, log: "从本地缓存加载 embedding 模型（首次较慢）", pct: 52 },
    { label: `加载 Rerank 模型 (${rerank})…`, log: "从本地缓存加载 reranker 模型", pct: 72 },
    { label: "初始化 RAG 引擎…", log: "构建 LightRAG 实例并挂载存储", pct: 88 },
  ];
}

function buildRagReloadSteps() {
  const embed = latestStatus?.embedding_model || envFieldValue("EMBEDDING_MODEL", "BAAI/bge-m3");
  const rerank = latestStatus?.rerank_model || envFieldValue("RERANK_MODEL", "BAAI/bge-reranker-base");
  return [
    { label: "重新读取配置…", log: "加载 config/.env", pct: 18 },
    { label: "卸载旧引擎实例…", log: "释放上一轮 RAG 资源", pct: 32 },
    { label: `加载向量模型 (${embed})…`, log: "从本地缓存加载 embedding 模型", pct: 55 },
    { label: `加载 Rerank 模型 (${rerank})…`, log: "从本地缓存加载 reranker 模型", pct: 74 },
    { label: "初始化 RAG 引擎…", log: "构建 LightRAG 实例", pct: 90 },
  ];
}

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

function optionValue(o) {
  return typeof o === "object" && o !== null ? o.value : o;
}

function optionLabel(o) {
  return typeof o === "object" && o !== null ? o.label || o.value : o;
}

function renderEnvForm(fields) {
  envForm.innerHTML = fields
    .map((f) => {
      const inputType = f.type === "password" ? "password" : "text";
      if (f.type === "select") {
        const opts = (f.options || [])
          .map((o) => {
            const v = optionValue(o);
            const lbl = optionLabel(o);
            const selected = v === f.value ? "selected" : "";
            return `<option value="${escapeAttr(v)}" ${selected}>${escapeAttr(lbl)}</option>`;
          })
          .join("");
        const customVisible = f.allow_custom && f.value === "__custom__";
        const customHtml = f.allow_custom
          ? `<input class="env-custom-input${customVisible ? "" : " hidden"}" name="${f.key}__custom"
              type="text" value="${escapeAttr(f.custom_value || "")}"
              placeholder="输入 API 网关支持的视觉模型名称" />`
          : "";
        return `
          <label class="env-field${f.allow_custom ? " env-field-custom" : ""}">
            <span class="label-text">${f.label}</span>
            <p class="field-hint">${f.hint || ""}</p>
            <select name="${f.key}" ${f.required ? "required" : ""}>${opts}</select>
            ${customHtml}
          </label>`;
      }
      return `
        <label class="env-field">
          <span class="label-text">${f.label}</span>
          <p class="field-hint">${f.hint || ""}</p>
          <input name="${f.key}" type="${inputType}" value="${escapeAttr(f.value || "")}"
            placeholder="${escapeAttr(f.placeholder || "")}" ${f.required ? "required" : ""} />
        </label>`;
    })
    .join("");
  bindEnvFormHandlers();
}

function bindEnvFormHandlers() {
  envForm.querySelectorAll('select[name="VISION_MODEL"]').forEach((sel) => {
    sel.addEventListener("change", () => {
      const custom = envForm.querySelector('[name="VISION_MODEL__custom"]');
      if (custom) {
        custom.classList.toggle("hidden", sel.value !== "__custom__");
        if (sel.value === "__custom__") custom.focus();
      }
    });
  });
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
  syncIngestUiWithServer();
  renderSteps(latestStatus);

  renderCheckList(checkRuntime, latestStatus.runtime_deps || []);
  renderCheckList(checkModels, latestStatus.bundled_models || []);

  const configItems = [
    {
      ok: Boolean(latestStatus.env?.ok),
      label: latestStatus.env?.ok
        ? "LLM 配置已保存"
        : latestStatus.env?.message || "LLM 配置未保存",
    },
    {
      ok: Boolean(latestStatus.llm_model),
      label: latestStatus.llm_model
        ? `文本 LLM：${latestStatus.llm_model}（LLM_MODEL）`
        : "文本 LLM：未配置",
    },
    {
      ok: Boolean(latestStatus.embedding_model),
      label: latestStatus.embedding_model
        ? `向量模型：${latestStatus.embedding_model}（EMBEDDING_MODEL / ${latestStatus.embedding_backend || "hf"}）`
        : "向量模型：未配置",
    },
    {
      ok: Boolean(latestStatus.rerank_model),
      label: latestStatus.rerank_model
        ? `Rerank：${latestStatus.rerank_model}（RERANK_MODEL / ${latestStatus.rerank_binding || "hf"}）`
        : "Rerank：未配置",
    },
    {
      ok: Boolean(latestStatus.rag_query_mode),
      label: `默认 RAG 模式：${latestStatus.rag_query_mode || "mix"}（RAG_QUERY_MODE）`,
    },
    {
      ok: Boolean(latestStatus.hf_home_effective),
      label: `HF 模型缓存：${latestStatus.hf_home_effective || latestStatus.models_dir || "—"}`,
    },
  ];
  const langItems = [
    {
      ok: Boolean(latestStatus.chinese_ingest),
      label: `灌库图谱语言：${latestStatus.ingest_language || "English"}（SUMMARY_LANGUAGE，控制实体/关系抽取与摘要）`,
    },
    {
      ok: (latestStatus.prompt_language || "").toLowerCase().startsWith("zh"),
      label: `多模态 Prompt：${latestStatus.prompt_language || "en"}（RAG_PROMPT_LANGUAGE）`,
    },
    {
      ok: Boolean(latestStatus.vision_model),
      label: latestStatus.vision_model
        ? `视觉模型：${latestStatus.vision_model}（多模态灌库时使用）`
        : "视觉模型：未配置（步骤 1 保存配置后自动写入）",
    },
    {
      ok: Boolean(latestStatus.multimodal_enabled),
      label: latestStatus.multimodal_enabled
        ? "多模态灌库：已开启（图片/表格/公式）"
        : "多模态灌库：关闭（仅文本，更快）",
    },
  ];
  if (checkLanguage) renderCheckList(checkLanguage, [...configItems, ...langItems]);

  if (setupSubtitle) {
    setupSubtitle.textContent = latestStatus.setup_complete
      ? "已完成安装；添加文档或整批重灌请从步骤 3 操作"
      : "首次使用请按步骤完成配置与灌库";
  }

  if (enableMultimodal && !multimodalSyncBusy) {
    enableMultimodal.checked = Boolean(latestStatus.multimodal_enabled);
  }
  if (multimodalHint) {
    const vision = latestStatus.vision_model ? `视觉模型 ${latestStatus.vision_model}` : "请先在步骤 1 选择视觉模型";
    multimodalHint.textContent = latestStatus.multimodal_enabled
      ? `当前引擎已启用图片/表格/公式处理（${vision}）；切换后会重新加载 RAG 引擎。`
      : `默认仅文本灌库；勾选后将重新加载引擎并启用多模态处理（需已在步骤 1 配置 ${vision}）。`;
  }

  if (kbSummary) {
    const kb = latestStatus.knowledge_base || {};
    const success = kb.success_count ?? latestStatus.kb_success_count ?? 0;
    const failed = kb.failed_count ?? latestStatus.kb_failed_count ?? 0;
    const hasUsableKb = Boolean(latestStatus.knowledge_base_ok);
    const hasPartialKb = Boolean(latestStatus.kb_partial && success > 0);
    const hasFailedOnly = failed > 0 && success === 0;

    if (success > 0 && failed > 0) {
      const failedNames = (kb.failed_docs || [])
        .map((d) => d.name || d.file_path)
        .filter(Boolean)
        .join("、");
      kbSummary.className = "kb-summary warn error";
      kbSummary.innerHTML = `<strong>当前知识库：</strong>已成功 ${success} 篇，<strong>${failed} 篇失败</strong>。${
        failedNames ? `失败文件：${escapeAttr(failedNames)}。` : ""
      }请查看灌库日志后追加灌库重试失败文件。`;
    } else if (hasFailedOnly) {
      kbSummary.className = "kb-summary warn error";
      kbSummary.innerHTML = `<strong>当前知识库：</strong>${kb.message || "灌库失败"}。请检查 LLM 配额或配置后重新灌库。`;
    } else if (hasUsableKb || hasPartialKb) {
      kbSummary.className = "kb-summary warn";
      kbSummary.innerHTML = `<strong>当前知识库：</strong>${kb.message || "已有数据"}。添加新文档请点「追加灌库」；整批重灌请点「重新灌库」。`;
    } else if (latestStatus.kb_partial) {
      kbSummary.className = "kb-summary warn error";
      kbSummary.innerHTML = `<strong>当前知识库：</strong>${kb.message || "索引未完成"}。请清空后重新灌库。`;
    } else {
      kbSummary.className = "kb-summary ok";
      kbSummary.innerHTML = `<strong>当前知识库：</strong>空。保存配置后将按 <code>${escapeAttr(latestStatus.ingest_language || "Chinese")}</code> 生成图谱。`;
    }
  }

  if (btnClearKb) {
    const kb = latestStatus.knowledge_base || {};
    const hasKb = Boolean(
      latestStatus.knowledge_base_ok ||
        latestStatus.kb_partial ||
        (kb.success_count ?? 0) > 0 ||
        (kb.failed_count ?? 0) > 0 ||
        (kb.doc_count ?? 0) > 0
    );
    btnClearKb.disabled = ingestBusy || engineLoadBusy || !hasKb;
    btnClearKb.title = hasKb
      ? "仅清空知识库，不上传文件"
      : "当前知识库为空，无需清空";
  }

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
        ? formatKbFinishText(latestStatus)
        : latestStatus.kb_partial
          ? formatKbPartialText(latestStatus)
          : "知识库未完成灌库",
    },
  ];
  finishChecklist.innerHTML = finishItems
    .map((it) => `<li class="${it.ok ? "ok" : "bad"}">${it.ok ? "✓" : "○"} ${it.text}</li>`)
    .join("");

  btnFinish.disabled = !latestStatus.can_finish_setup;
  updateIngestControls();
}

function formatKbFinishText(status) {
  const ok = status.kb_success_count ?? status.kb_unique_doc_count ?? status.kb_doc_count ?? 0;
  return `知识库已灌库（${ok} 篇文档）`;
}

function formatKbPartialText(status) {
  const ok = status.kb_success_count ?? 0;
  const fail = status.kb_failed_count ?? 0;
  if (ok > 0 && fail > 0) {
    return `知识库部分完成（成功 ${ok} 篇，失败 ${fail} 篇，请查看灌库日志后追加重试）`;
  }
  if (fail > 0 && ok === 0) {
    return `知识库灌库失败（${fail} 篇均未成功，请检查配置后重灌）`;
  }
  const total = status.kb_doc_count ?? ok;
  if (ok > 0 && total > ok) {
    return `知识库未完成灌库（${total} 条残留记录 / ${ok} 个 PDF，需清空后重灌）`;
  }
  return `知识库未完成灌库（请重新灌库）`;
}

function syncIngestUiWithServer() {
  const serverActive = Boolean(latestStatus?.ingest_active);
  if (ingestBusy && !serverActive) {
    ingestBusy = false;
    ingestStopping = false;
    if (ingestStatus && /进行中|正在停止|处理完成后/.test(ingestStatus.textContent || "")) {
      ingestStatus.textContent = "灌库任务已结束（服务端）。如状态未更新，请查看下方日志。";
      ingestStatus.className = "hint status-ok";
    }
  }
  if (ingestBusy) {
    if (!statusPollTimer) {
      statusPollTimer = setInterval(() => {
        refreshStatus().catch(() => {});
      }, 2500);
    }
  } else if (statusPollTimer) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }
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
        steps: buildRagReloadSteps(),
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

function knowledgeBaseHasData() {
  const kb = latestStatus?.knowledge_base || {};
  return Boolean(
    latestStatus?.knowledge_base_ok ||
      (kb.success_count > 0 && kb.chunk_count > 0) ||
      latestStatus?.kb_partial
  );
}

function updateIngestControls() {
  const hasFiles = pendingFiles.length > 0;
  const ragReady = Boolean(latestStatus?.rag_ready);
  const envOk = Boolean(latestStatus?.env?.ok);
  const hasKb = knowledgeBaseHasData();
  const ingestDisabled = !hasFiles || !ragReady || ingestBusy || engineLoadBusy || multimodalSyncBusy;

  if (fileListToolbar) {
    fileListToolbar.classList.toggle("hidden", !hasFiles);
  }
  if (fileListCount) {
    fileListCount.textContent = hasFiles ? `已选 ${pendingFiles.length} 个文件` : "";
  }
  if (btnClearFiles) {
    btnClearFiles.disabled = !hasFiles || ingestBusy;
  }

  if (btnIngest) {
    btnIngest.textContent = hasKb ? "重新灌库（清空后）" : "开始灌库";
    btnIngest.classList.toggle("danger", hasKb);
    btnIngest.classList.toggle("secondary", !hasKb);
    btnIngest.disabled = ingestDisabled;
  }
  if (btnIngestAppend) {
    btnIngestAppend.classList.toggle("hidden", !hasKb);
    btnIngestAppend.disabled = ingestDisabled;
  }
  if (btnClearKb) {
    btnClearKb.disabled = ingestBusy || engineLoadBusy || multimodalSyncBusy || !hasKb;
    btnClearKb.title = hasKb
      ? "仅清空知识库，不上传文件"
      : "当前知识库为空，无需清空";
  }
  if (btnStopIngest) {
    btnStopIngest.classList.toggle("hidden", !ingestBusy);
    btnStopIngest.disabled = !ingestBusy || ingestStopping;
  }
  if (typeof updateIngestTerminalStopState === "function") {
    updateIngestTerminalStopState(ingestBusy, ingestStopping);
  }

  if (enableMultimodal) {
    enableMultimodal.disabled = ingestBusy || engineLoadBusy || multimodalSyncBusy;
  }

  if (!ingestGateHint) return;

  if (ingestBusy) {
    ingestGateHint.textContent = ingestStopping
      ? "已请求停止；当前文件处理完成后将终止，并仅清理本次未完成的灌库残余（请稍候）…"
      : "灌库进行中，请勿关闭窗口…";
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
  } else if (hasKb) {
    ingestGateHint.textContent = "文件已就绪：追加新文档点「追加灌库」；整批重灌点「重新灌库（清空后）」。";
    ingestGateHint.className = "hint ingest-gate-hint status-ok";
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
  if (engineLoadBusy || ingestBusy) {
    envStatus.textContent = ingestBusy
      ? "灌库进行中，请等待结束或停止后再保存配置。"
      : "正在加载引擎，请稍候…";
    envStatus.className = "hint error";
    return;
  }

  engineLoadBusy = true;
  envStatus.textContent = "正在保存并加载引擎，请勿关闭窗口…";
  envStatus.className = "hint";

  try {
    const data = await runStagedProgressTask({
      steps: buildEnvSaveSteps(),
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
  if (engineLoadBusy || ingestBusy) {
    ragStatus.textContent = ingestBusy
      ? "灌库进行中，请等待结束或停止后再重新加载引擎。"
      : "正在加载引擎，请稍候…";
    ragStatus.className = "hint error";
    return;
  }

  engineLoadBusy = true;
  ragStatus.textContent = "正在重新加载 RAG 引擎…";
  ragStatus.className = "hint";

  try {
    await runStagedProgressTask({
      steps: buildRagReloadSteps(),
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

async function applyMultimodalSetting(enabled) {
  if (ingestBusy || engineLoadBusy || multimodalSyncBusy) return;
  multimodalSyncBusy = true;
  updateIngestControls();
  if (multimodalHint) {
    multimodalHint.textContent = enabled
      ? "正在开启多模态并重新加载引擎…"
      : "正在关闭多模态并重新加载引擎…";
  }
  try {
    await runStagedProgressTask({
      steps: buildRagReloadSteps(),
      progressEl: ragLoadProgress,
      barEl: ragLoadProgressBar,
      labelEl: ragLoadProgressLabel,
      logEl: ragLoadLog,
      buttons: [btnReloadRag, btnSaveEnv],
      forms: [envForm],
      task: async () => {
        const res = await fetch("/api/setup/multimodal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled }),
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || payload.message || `HTTP ${res.status}`);
        return payload;
      },
    });
    await refreshStatus();
    if (multimodalHint) {
      multimodalHint.textContent = enabled
        ? "多模态已开启：灌库时将处理图片、表格与公式。"
        : "多模态已关闭：仅文本灌库。";
      multimodalHint.className = "hint status-ok";
    }
  } catch (err) {
    if (enableMultimodal) enableMultimodal.checked = !enabled;
    if (multimodalHint) {
      multimodalHint.textContent = `切换多模态失败：${err.message || err}`;
      multimodalHint.className = "hint error";
    }
  } finally {
    multimodalSyncBusy = false;
    updateIngestControls();
  }
}

enableMultimodal?.addEventListener("change", () => {
  void applyMultimodalSetting(enableMultimodal.checked);
});

async function clearKnowledgeBase({ logEl, statusEl } = {}) {
  const res = await fetch("/api/setup/clear-knowledge-base", { method: "POST" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
  if (logEl) appendIngestLog(logEl, data.message || "知识库已清空");
  if (statusEl) statusEl.textContent = data.message || "知识库已清空";
  await refreshStatus();
  return data;
}

btnClearKb?.addEventListener("click", async () => {
  if (ingestBusy || engineLoadBusy) return;
  if (!window.confirm("确定清空现有知识库？此操作不可恢复，清空后需重新上传文档灌库。")) {
    return;
  }
  ingestStatus.textContent = "正在清空知识库…";
  ingestStatus.className = "hint";
  try {
    btnClearKb.disabled = true;
    await clearKnowledgeBase({ logEl: ingestLog, statusEl: ingestStatus });
    ingestStatus.textContent = "知识库已清空，请上传文档后重新灌库。";
    ingestStatus.className = "hint status-ok";
  } catch (err) {
    ingestStatus.textContent = `清空失败：${err.message || err}`;
    ingestStatus.className = "hint error";
  } finally {
    updateIngestControls();
  }
});

async function startIngest({ clearFirst = false } = {}) {
  if (!pendingFiles.length || ingestBusy) return;
  if (!latestStatus?.rag_ready) {
    const ready = await ensureRagReady();
    if (!ready) return;
  }

  const hasKb = knowledgeBaseHasData();
  if (clearFirst && hasKb) {
    const ok = window.confirm(
      "将清空现有知识库，再灌入您选择的文件。\n\n适合整批重灌；若只想添加新 PDF，请改用「追加灌库」。\n\n是否继续？"
    );
    if (!ok) return;
  } else if (!clearFirst && hasKb) {
    const ok = window.confirm(
      "新文档将追加到现有知识库，不会删除其它文档；同名 PDF 会自动替换已有索引。\n\n是否继续？"
    );
    if (!ok) return;
  }

  ingestBusy = true;
  ingestStopping = false;
  updateIngestControls();
  ingestStatus.textContent = "灌库进行中，请稍候…";
  if (ingestProgressBar) ingestProgressBar.classList.remove("error");
  try {
    if (clearFirst && hasKb) {
      ingestStatus.textContent = "正在清空旧知识库…";
      appendIngestLog(ingestLog, "重新灌库：先清空现有知识库…");
      await clearKnowledgeBase();
    } else if (!clearFirst && hasKb) {
      appendIngestLog(ingestLog, "追加灌库：保留现有知识库，写入新文件…");
    }
    ingestStatus.textContent = "灌库进行中，请稍候…";
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
      ingestStatus.textContent = data.message || "已停止灌库（已有文档已保留）";
      ingestStatus.className = "hint error";
      pendingFiles = [];
    } else if (data.fail > 0) {
      ingestStatus.className = "hint error";
      ingestStatus.textContent =
        data.ok > 0
          ? `灌库完成：成功 ${data.ok} 篇，失败 ${data.fail} 篇（失败原因见下方日志）`
          : `灌库失败：${data.fail} 篇均未成功（详见下方日志）`;
      if (ingestProgressBar) ingestProgressBar.classList.add("error");
    } else {
      ingestStatus.className = "hint status-ok";
      ingestStatus.textContent = `灌库完成：成功 ${data.ok} 篇`;
      pendingFiles = [];
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
}

btnIngest.addEventListener("click", async () => {
  await startIngest({ clearFirst: knowledgeBaseHasData() });
});

btnIngestAppend?.addEventListener("click", async () => {
  await startIngest({ clearFirst: false });
});

async function handleStopIngest() {
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
}

btnStopIngest?.addEventListener("click", handleStopIngest);
document.addEventListener("ingest-stop-request", handleStopIngest);

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
