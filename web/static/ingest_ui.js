/** Shared streaming ingest UI for setup wizard and chat sidebar. */

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * @param {HTMLElement} listEl
 * @param {File[]} files
 * @param {{ onRemove?: (index: number) => void, readOnly?: boolean }} options
 */
function renderPendingFileList(listEl, files, options = {}) {
  const { onRemove, readOnly = false } = options;
  listEl.innerHTML = "";
  files.forEach((file, index) => {
    const li = document.createElement("li");
    li.className = "file-list-item";

    const name = document.createElement("span");
    name.className = "file-list-name";
    name.textContent = `${file.name} (${formatFileSize(file.size)})`;

    li.appendChild(name);

    if (!readOnly && onRemove) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "file-list-remove";
      btn.title = "移除此文件";
      btn.setAttribute("aria-label", `移除 ${file.name}`);
      btn.textContent = "×";
      btn.addEventListener("click", () => onRemove(index));
      li.appendChild(btn);
    }

    listEl.appendChild(li);
  });
}

function parseSseLines(buffer, onEvent) {
  const parts = buffer.split("\n");
  const rest = parts.pop() ?? "";
  for (const line of parts) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const raw = trimmed.slice(5).trim();
    if (!raw) continue;
    try {
      onEvent(JSON.parse(raw));
    } catch {
      /* ignore */
    }
  }
  return rest;
}

function appendIngestLog(logEl, text) {
  if (!text) return;
  const targets = [];
  if (logEl) targets.push(logEl);
  const modalLog = getIngestTerminalModal()?.logEl;
  if (modalLog && modalLog !== logEl) targets.push(modalLog);
  for (const el of targets) {
    el.textContent += `${text}\n`;
    el.scrollTop = el.scrollHeight;
  }
}

/** @type {{ backdrop: HTMLElement, panel: HTMLElement, logEl: HTMLElement, progressEl: HTMLElement, progressBarEl: HTMLElement, progressLabelEl: HTMLElement, stopBtn: HTMLButtonElement, closeBtn: HTMLButtonElement, reopenBtn: HTMLButtonElement } | null} */
let ingestTerminalModal = null;

function getIngestTerminalModal() {
  return ingestTerminalModal;
}

function ensureIngestTerminalModal() {
  if (ingestTerminalModal) return ingestTerminalModal;

  const backdrop = document.createElement("div");
  backdrop.className = "ingest-terminal-backdrop hidden";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.setAttribute("aria-labelledby", "ingest-terminal-title");

  backdrop.innerHTML = `
    <div class="ingest-terminal-panel">
      <header class="ingest-terminal-header">
        <div class="ingest-terminal-title-wrap">
          <span class="ingest-terminal-dot" aria-hidden="true"></span>
          <h2 id="ingest-terminal-title">灌库终端</h2>
        </div>
        <div class="ingest-terminal-header-actions">
          <button type="button" class="btn danger ingest-terminal-stop hidden" title="停止灌库并清空知识库">
            停止灌库
          </button>
          <button type="button" class="btn ghost ingest-terminal-close" title="关闭窗口（灌库在后台继续）">
            关闭
          </button>
        </div>
      </header>
      <div class="ingest-terminal-progress">
        <div class="ingest-progress-label ingest-terminal-progress-label">等待开始…</div>
        <div class="ingest-progress-track">
          <div class="ingest-progress-bar ingest-terminal-progress-bar"></div>
        </div>
      </div>
      <pre class="ingest-log ingest-terminal-log" aria-live="polite"></pre>
      <p class="hint ingest-terminal-hint">解析、建图谱与向量索引的实时日志。可点「停止灌库」终止并清空知识库；仅关闭窗口不会中断灌库。</p>
    </div>
  `;

  document.body.appendChild(backdrop);

  const reopenBtn = document.createElement("button");
  reopenBtn.type = "button";
  reopenBtn.className = "ingest-terminal-reopen hidden";
  reopenBtn.textContent = "灌库日志";
  reopenBtn.title = "打开灌库日志窗口";
  document.body.appendChild(reopenBtn);

  const panel = backdrop.querySelector(".ingest-terminal-panel");
  const closeBtn = backdrop.querySelector(".ingest-terminal-close");
  const stopBtn = backdrop.querySelector(".ingest-terminal-stop");
  ingestTerminalModal = {
    backdrop,
    panel,
    logEl: backdrop.querySelector(".ingest-terminal-log"),
    progressEl: backdrop.querySelector(".ingest-terminal-progress"),
    progressBarEl: backdrop.querySelector(".ingest-terminal-progress-bar"),
    progressLabelEl: backdrop.querySelector(".ingest-terminal-progress-label"),
    stopBtn,
    closeBtn,
    reopenBtn,
  };

  stopBtn?.addEventListener("click", () => {
    document.dispatchEvent(new CustomEvent("ingest-stop-request"));
  });
  closeBtn?.addEventListener("click", () => hideIngestTerminalModal());
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) hideIngestTerminalModal();
  });
  reopenBtn.addEventListener("click", () => showIngestTerminalModal());

  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || !ingestTerminalModal) return;
    if (ingestTerminalModal.backdrop.classList.contains("hidden")) return;
    hideIngestTerminalModal();
  });

  return ingestTerminalModal;
}

function showIngestTerminalModal() {
  const modal = ensureIngestTerminalModal();
  modal.backdrop.classList.remove("hidden");
  modal.reopenBtn.classList.add("hidden");
  document.body.classList.add("ingest-terminal-open");
}

function hideIngestTerminalModal() {
  if (!ingestTerminalModal) return;
  ingestTerminalModal.backdrop.classList.add("hidden");
  document.body.classList.remove("ingest-terminal-open");
  if (ingestTerminalModal.reopenBtn && !ingestTerminalModal.reopenBtn.dataset.forceHide) {
    ingestTerminalModal.reopenBtn.classList.remove("hidden");
  }
}

function openIngestTerminalModal() {
  showIngestTerminalModal();
}

function setIngestTerminalReopenVisible(visible) {
  const modal = ensureIngestTerminalModal();
  if (visible) {
    modal.reopenBtn.classList.remove("hidden");
    delete modal.reopenBtn.dataset.forceHide;
  } else {
    modal.reopenBtn.classList.add("hidden");
    modal.reopenBtn.dataset.forceHide = "1";
  }
}

function updateIngestTerminalStopState(busy, stopping) {
  const modal = getIngestTerminalModal();
  if (!modal?.stopBtn) return;
  modal.stopBtn.classList.toggle("hidden", !busy);
  modal.stopBtn.disabled = !busy || stopping;
  modal.stopBtn.textContent = stopping ? "正在停止…" : "停止灌库";
}

function setIngestProgressBarError(progressBarEl, hasError) {
  const bars = [progressBarEl, getIngestTerminalModal()?.progressBarEl].filter(Boolean);
  for (const bar of bars) {
    if (hasError) bar.classList.add("error");
    else bar.classList.remove("error");
  }
}

function setIngestProgress(progressEl, barEl, labelEl, current, total, message) {
  const apply = (pEl, bEl, lEl) => {
    if (!pEl) return;
    pEl.classList.remove("hidden");
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    if (bEl) bEl.style.width = `${pct}%`;
    if (lEl) {
      lEl.textContent = message || (total ? `进度 ${current}/${total}（${pct}%）` : "灌库进行中…");
    }
  };
  apply(progressEl, barEl, labelEl);
  const modal = getIngestTerminalModal();
  if (modal) {
    apply(modal.progressEl, modal.progressBarEl, modal.progressLabelEl);
  }
}

/**
 * @param {Object} opts
 * @param {File[]} opts.files
 * @param {HTMLElement|null} opts.logEl
 * @param {HTMLElement|null} opts.progressEl
 * @param {HTMLElement|null} opts.progressBarEl
 * @param {HTMLElement|null} opts.progressLabelEl
 * @param {HTMLElement|null} opts.statusEl
 * @param {(active: boolean) => void} [opts.onActiveChange]
 */
async function streamIngest(opts) {
  const {
    files,
    logEl,
    progressEl,
    progressBarEl,
    progressLabelEl,
    statusEl,
    onActiveChange,
  } = opts;

  if (logEl) logEl.textContent = "";
  const modal = ensureIngestTerminalModal();
  if (modal.logEl) modal.logEl.textContent = "";
  if (modal.progressBarEl) setIngestProgressBarError(modal.progressBarEl, false);
  openIngestTerminalModal();
  setIngestProgress(progressEl, progressBarEl, progressLabelEl, 0, files.length, "上传文件中…");
  onActiveChange?.(true);

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));

  try {
    const res = await fetch("/api/ingest/stream", { method: "POST", body: fd });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let result = { ok: 0, fail: 0, errors: [], cancelled: false };
    let streamError = null;

    const handleEvent = (ev) => {
      if (ev.type === "ingest_saved") {
        appendIngestLog(logEl, ev.message || "文件已上传");
        if (statusEl) statusEl.textContent = ev.message || "";
        return;
      }
      if (ev.type === "ingest_start") {
        appendIngestLog(logEl, `共 ${ev.total} 个文件待灌库`);
        setIngestProgress(progressEl, progressBarEl, progressLabelEl, 0, ev.total, "开始灌库…");
        return;
      }
      if (ev.type === "file_start") {
        appendIngestLog(logEl, ev.message || `开始：${ev.file}`);
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          Math.max(0, (ev.current || 1) - 1),
          ev.total || files.length,
          ev.message
        );
        return;
      }
      if (ev.type === "log") {
        appendIngestLog(logEl, ev.message);
        return;
      }
      if (ev.type === "file_ok") {
        appendIngestLog(logEl, `✓ 成功：${ev.file}`);
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          ev.current || 0,
          ev.total || files.length,
          `已完成 ${ev.current}/${ev.total}`
        );
        return;
      }
      if (ev.type === "file_fail") {
        const errText = ev.error || "灌库失败";
        appendIngestLog(logEl, `✗ 失败：${ev.file}\n  ${errText}`);
        if (statusEl) {
          statusEl.textContent = `✗ ${ev.file}：${errText}`;
          statusEl.className = "hint error";
        }
        if (progressBarEl) setIngestProgressBarError(progressBarEl, true);
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          ev.current || 0,
          ev.total || files.length,
          `失败 ${ev.current}/${ev.total}：${ev.file}`
        );
        return;
      }
      if (ev.type === "error") {
        streamError = new Error(ev.message || "灌库失败");
        return;
      }
      if (ev.type === "cancelled") {
        result = {
          ok: ev.ok ?? 0,
          fail: ev.fail ?? 0,
          errors: ev.errors || [],
          cancelled: true,
        };
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          result.ok + result.fail,
          files.length,
          "已停止"
        );
        if (progressBarEl) setIngestProgressBarError(progressBarEl, true);
        appendIngestLog(logEl, ev.message || "—— 已停止灌库并清空知识库 ——");
        if (statusEl) statusEl.textContent = ev.message || "已停止灌库并清空知识库";
        return;
      }
      if (ev.type === "done") {
        result = { ok: ev.ok ?? 0, fail: ev.fail ?? 0, errors: ev.errors || [], cancelled: false };
        const label =
          result.fail > 0
            ? result.ok > 0
              ? `完成：成功 ${result.ok} 篇，失败 ${result.fail} 篇`
              : `灌库失败：${result.fail} 篇均未成功`
            : `灌库完成：成功 ${result.ok} 篇`;
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          files.length,
          files.length,
          label
        );
        if (result.fail > 0) {
          if (progressBarEl) setIngestProgressBarError(progressBarEl, true);
          if (statusEl) {
            statusEl.textContent =
              result.ok > 0
                ? `灌库完成：成功 ${result.ok} 篇，失败 ${result.fail} 篇（失败原因见下方日志）`
                : `灌库失败：${result.fail} 篇均未成功（详见下方日志）`;
            statusEl.className = "hint error";
          }
        } else if (statusEl) {
          statusEl.textContent = `灌库完成：成功 ${result.ok} 篇`;
          statusEl.className = "hint status-ok";
        }
        appendIngestLog(logEl, `—— ${label} ——`);
        if (result.errors?.length) {
          result.errors.forEach((e) => {
            appendIngestLog(logEl, `  ${e.file}: ${e.error}`);
          });
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      buf = parseSseLines(buf, handleEvent);
      if (streamError) throw streamError;
    }
    buf = parseSseLines(buf + "\n", handleEvent);
    if (streamError) throw streamError;
    return result;
  } finally {
    onActiveChange?.(false);
    setIngestTerminalReopenVisible(true);
  }
}

async function requestStopIngest(opts = {}) {
  const { logEl, statusEl } = opts;
  appendIngestLog(logEl, "正在请求停止灌库…");
  if (statusEl) statusEl.textContent = "正在停止灌库（当前文件处理完成后终止）…";
  const res = await fetch("/api/ingest/cancel", { method: "POST" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
  appendIngestLog(logEl, data.message || "停止请求已发送");
  if (statusEl) statusEl.textContent = data.message || "正在停止灌库…";
  return data;
}

/**
 * Staged progress UI while a long-running setup task executes.
 * Steps advance on a timer until the task finishes (no backend streaming).
 */
async function runStagedProgressTask(opts) {
  const {
    steps,
    task,
    progressEl,
    barEl,
    labelEl,
    logEl,
    buttons = [],
    forms = [],
    stepIntervalMs = 3200,
  } = opts;

  const appendLog = (text) => {
    if (!logEl || !text) return;
    logEl.textContent += `${text}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  };

  const setStep = (index) => {
    const step = steps[index];
    if (!step) return;
    progressEl?.classList.remove("hidden");
    if (labelEl) labelEl.textContent = step.label;
    if (barEl) {
      barEl.classList.remove("indeterminate", "error");
      barEl.style.width = `${step.pct}%`;
    }
    if (step.log) appendLog(step.log);
  };

  const disableFormControls = (disabled) => {
    forms.forEach((form) => {
      form?.querySelectorAll("input, select, textarea, button").forEach((el) => {
        el.disabled = disabled;
      });
    });
  };

  buttons.forEach((btn) => {
    if (btn) btn.disabled = true;
  });
  disableFormControls(true);

  if (logEl) logEl.textContent = "";
  setStep(0);

  let stepIndex = 0;
  const timer = setInterval(() => {
    if (stepIndex < steps.length - 1) {
      stepIndex += 1;
      setStep(stepIndex);
    } else if (barEl) {
      barEl.classList.add("indeterminate");
    }
  }, stepIntervalMs);

  try {
    const result = await task();
    clearInterval(timer);
    if (barEl) {
      barEl.classList.remove("indeterminate");
      barEl.style.width = "100%";
    }
    if (labelEl) labelEl.textContent = "完成";
    appendLog("—— 加载完成 ——");
    return result;
  } catch (err) {
    clearInterval(timer);
    if (barEl) {
      barEl.classList.remove("indeterminate");
      barEl.classList.add("error");
      barEl.style.width = "100%";
    }
    if (labelEl) labelEl.textContent = "加载失败";
    appendLog(`错误：${err.message || err}`);
    throw err;
  } finally {
    clearInterval(timer);
    buttons.forEach((btn) => {
      if (btn) btn.disabled = false;
    });
    disableFormControls(false);
  }
}
