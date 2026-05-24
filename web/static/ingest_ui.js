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
  if (!logEl || !text) return;
  logEl.textContent += `${text}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setIngestProgress(progressEl, barEl, labelEl, current, total, message) {
  if (!progressEl) return;
  progressEl.classList.remove("hidden");
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  if (barEl) barEl.style.width = `${pct}%`;
  if (labelEl) {
    labelEl.textContent = message || (total ? `进度 ${current}/${total}（${pct}%）` : "灌库进行中…");
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
        appendIngestLog(logEl, `✗ 失败：${ev.file}\n  ${ev.error || ""}`);
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          ev.current || 0,
          ev.total || files.length,
          `失败 ${ev.current}/${ev.total}`
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
        if (progressBarEl) progressBarEl.classList.add("error");
        appendIngestLog(logEl, ev.message || "—— 已停止灌库并清空知识库 ——");
        if (statusEl) statusEl.textContent = ev.message || "已停止灌库并清空知识库";
        return;
      }
      if (ev.type === "done") {
        result = { ok: ev.ok ?? 0, fail: ev.fail ?? 0, errors: ev.errors || [], cancelled: false };
        setIngestProgress(
          progressEl,
          progressBarEl,
          progressLabelEl,
          files.length,
          files.length,
          `完成：成功 ${result.ok}，失败 ${result.fail}`
        );
        appendIngestLog(logEl, `—— 灌库结束：成功 ${result.ok}，失败 ${result.fail} ——`);
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
