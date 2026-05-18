const $ = (sel) => document.querySelector(sel);

const messagesEl = $("#messages");
const queryInput = $("#query-input");
const queryMode = $("#query-mode");
const composer = $("#composer");
const btnSend = $("#btn-send");
const btnClear = $("#btn-clear");
const btnIngest = $("#btn-ingest");
const fileInput = $("#file-input");
const uploadZone = $("#upload-zone");
const fileList = $("#file-list");
const ingestStatus = $("#ingest-status");

let pendingFiles = [];
/** When false, user scrolled up — do not auto-jump to bottom on every token. */
let scrollPinnedToBottom = true;
let scrollRaf = 0;

function getMessagesEnd() {
  let end = document.getElementById("messages-end");
  if (!end && messagesEl) {
    end = document.createElement(`d` + `iv`);
    end.id = "messages-end";
    end.className = "messages-end";
    end.setAttribute("aria-hidden", "true");
    messagesEl.appendChild(end);
  }
  return end;
}

function pinMessagesEnd() {
  const end = getMessagesEnd();
  if (end && messagesEl) messagesEl.appendChild(end);
}

function scrollMessages(force = false) {
  if (!messagesEl) return;
  const gap =
    messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
  if (!force && !scrollPinnedToBottom && gap > 80) return;

  if (scrollRaf) cancelAnimationFrame(scrollRaf);
  scrollRaf = requestAnimationFrame(() => {
    scrollRaf = requestAnimationFrame(() => {
      pinMessagesEnd();
      const end = getMessagesEnd();
      if (end) {
        end.scrollIntoView({ block: "end", behavior: "instant" });
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  });
}

function initMessagesScroll() {
  if (!messagesEl) return;
  messagesEl.addEventListener(
    "scroll",
    () => {
      const gap =
        messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      scrollPinnedToBottom = gap < 80;
    },
    { passive: true }
  );
}

function watchMessageResize(el) {
  if (!el || typeof ResizeObserver === "undefined") return;
  const ro = new ResizeObserver(() => scrollMessages(scrollPinnedToBottom));
  ro.observe(el);
}

function renderMarkdown(el, md) {
  if (!md) {
    el.innerHTML = "";
    return;
  }
  if (typeof marked === "undefined") {
    el.textContent = md;
    return;
  }
  marked.setOptions({ breaks: true, gfm: true });
  const html = marked.parse(md);
  el.innerHTML =
    typeof DOMPurify !== "undefined" ? DOMPurify.sanitize(html) : html;
}

function appendMessage(role, text, extraClass = "") {
  const el = document.createElement(`d` + `iv`);
  el.className = `msg ${role} ${extraClass}`.trim();
  const label = role === "user" ? "你" : role === "assistant" ? "助手" : "系统";
  const roleEl = document.createElement(`d` + `iv`);
  roleEl.className = "role";
  roleEl.textContent = label;
  const bodyEl = document.createElement(`d` + `iv`);
  bodyEl.className = "body";
  bodyEl.textContent = text;
  el.appendChild(roleEl);
  el.appendChild(bodyEl);
  messagesEl.insertBefore(el, getMessagesEnd());
  pinMessagesEnd();
  scrollMessages(true);
  return el;
}

function createLoadingMessage() {
  const el = document.createElement(`d` + `iv`);
  el.className = "msg assistant loading";
  el.innerHTML = `
    <div class="loading-body">
      <span class="loading-text">正在检索知识库…</span>
      <span class="loading-dots" aria-hidden="true"><i></i><i></i><i></i></span>
    </div>
  `;
  messagesEl.insertBefore(el, getMessagesEnd());
  pinMessagesEnd();
  scrollMessages(true);
  watchMessageResize(el);
  return el;
}

function setLoadingStatus(el, text) {
  const textEl = el.querySelector(".loading-text");
  if (textEl && text) textEl.textContent = text;
}

function stopLoadingMessage(el) {
  el.classList.remove("loading");
}

/** Build assistant bubble with collapsible thinking + markdown answer area. */
function prepareAssistantStream(el) {
  stopLoadingMessage(el);
  el.innerHTML = "";
  el.classList.remove("loading");
  el.classList.add("streaming", "has-structure");

  const roleEl = document.createElement(`d` + `iv`);
  roleEl.className = "role";
  roleEl.textContent = "助手";

  const bodyEl = document.createElement(`d` + `iv`);
  bodyEl.className = "body assistant-body";

  const thinkingBlock = document.createElement("details");
  thinkingBlock.className = "thinking-block";
  thinkingBlock.hidden = true;
  const thinkingSummary = document.createElement("summary");
  thinkingSummary.textContent = "思考过程";
  const thinkingText = document.createElement("div");
  thinkingText.className = "thinking-text";
  thinkingBlock.append(thinkingSummary, thinkingText);

  const scopeEl = document.createElement(`d` + `iv`);
  scopeEl.className = "retrieval-scope";
  scopeEl.hidden = true;

  const answerLabel = document.createElement(`d` + `iv`);
  answerLabel.className = "answer-label";
  answerLabel.textContent = "回答";
  answerLabel.hidden = true;

  const answerMd = document.createElement(`d` + `iv`);
  answerMd.className = "answer-md markdown-body";

  bodyEl.append(thinkingBlock, scopeEl, answerLabel, answerMd);
  el.append(roleEl, bodyEl);
  pinMessagesEnd();
  scrollMessages(true);
  watchMessageResize(el);

  return {
    thinkingBlock,
    thinkingText,
    scopeEl,
    answerLabel,
    answerMd,
    thinkingRaw: "",
    answerRaw: "",
  };
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showRetrievalScope(ui, ev) {
  if (!ui?.scopeEl || !ev?.active) return;
  ui.scopeEl.hidden = false;
  const summary = ev.text || "";
  const kept = (ev.kept_sources || []).slice(0, 5);
  const removed = (ev.removed_sources || []).slice(0, 5);
  let html = `<strong>${escapeHtml(summary)}</strong>`;
  if (kept.length) {
    html += `<div class="scope-kept">参考：${kept.map((s) => escapeHtml(s)).join("、")}</div>`;
  }
  if (removed.length) {
    const items = removed
      .map((r) => escapeHtml(r.title || r.path || ""))
      .join("、");
    html += `<div class="scope-removed">已排除：${items}</div>`;
  }
  ui.scopeEl.innerHTML = html;
  scrollMessages();
}

function appendThinkingDelta(ui, text) {
  if (!ui || !text) return;
  ui.thinkingRaw += text;
  ui.thinkingBlock.hidden = false;
  ui.thinkingText.textContent = ui.thinkingRaw;
  if (!ui.thinkingBlock.open) ui.thinkingBlock.open = true;
  scrollMessages();
}

function appendAnswerDelta(ui, text) {
  if (!ui || !text) return;
  ui.answerRaw += text;
  ui.answerLabel.hidden = false;
  renderMarkdown(ui.answerMd, ui.answerRaw);
  scrollMessages();
}

async function fetchHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}

async function refreshStatus() {
  try {
    const h = await fetchHealth();
    $("#meta-ready").textContent = h.ready ? "是" : "否";
    $("#meta-ready").className = h.ready ? "status-ok" : "status-bad";
    $("#meta-wd").textContent = h.working_dir || "—";
    $("#meta-mode").textContent = h.query_mode || "—";
    if (h.query_mode) queryMode.value = h.query_mode;
    const errEl = $("#meta-error");
    if (h.init_error) {
      errEl.textContent = h.init_error;
      errEl.classList.remove("hidden");
    } else {
      errEl.classList.add("hidden");
    }
    btnSend.disabled = !h.ready;
  } catch (e) {
    $("#meta-ready").textContent = "无法连接";
    $("#meta-error").textContent = String(e);
    $("#meta-error").classList.remove("hidden");
    btnSend.disabled = true;
  }
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
      /* ignore malformed */
    }
  }
  return rest;
}

async function streamQuery(query, loadingEl) {
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode: queryMode.value, stream: true }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = data.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || d).join("; ")
          : `HTTP ${res.status}`;
    throw new Error(msg);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let ui = null;
  let gotContent = false;
  let streamError = null;

  const handleEvent = (ev) => {
    if (ev.type === "status") {
      if (!gotContent) setLoadingStatus(loadingEl, ev.text);
      return;
    }
    if (ev.type === "retrieval_scope") {
      if (!ui) {
        ui = prepareAssistantStream(loadingEl);
        gotContent = true;
      }
      showRetrievalScope(ui, ev);
      return;
    }
    if (ev.type === "error") {
      streamError = new Error(ev.message || "查询失败");
      return;
    }
    if (ev.type === "thinking_delta" && ev.text) {
      if (!gotContent) {
        ui = prepareAssistantStream(loadingEl);
        gotContent = true;
      }
      appendThinkingDelta(ui, ev.text);
      return;
    }
    if (ev.type === "answer_delta" && ev.text) {
      if (!gotContent) {
        ui = prepareAssistantStream(loadingEl);
        gotContent = true;
      }
      appendAnswerDelta(ui, ev.text);
      return;
    }
    // Legacy single-stream delta (no CoT split)
    if (ev.type === "delta" && ev.text) {
      if (!gotContent) {
        ui = prepareAssistantStream(loadingEl);
        gotContent = true;
      }
      appendAnswerDelta(ui, ev.text);
    }
    if (ev.type === "done") {
      loadingEl.classList.remove("streaming");
      if (ui?.thinkingBlock && ui.thinkingRaw) {
        ui.thinkingBlock.open = false;
      }
      scrollMessages(true);
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

  if (!gotContent) {
    stopLoadingMessage(loadingEl);
    loadingEl.innerHTML = "";
    const roleEl = document.createElement(`d` + `iv`);
    roleEl.className = "role";
    roleEl.textContent = "助手";
    const bodyEl = document.createElement(`d` + `iv`);
    bodyEl.className = "body";
    bodyEl.textContent = "(空回答)";
    loadingEl.appendChild(roleEl);
    loadingEl.appendChild(bodyEl);
  } else {
    stopLoadingMessage(loadingEl);
    loadingEl.classList.remove("streaming");
    if (ui?.answerMd && ui.answerRaw) {
      renderMarkdown(ui.answerMd, ui.answerRaw);
    }
    scrollMessages(true);
  }
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;

  appendMessage("user", q);
  queryInput.value = "";
  scrollPinnedToBottom = true;
  btnSend.disabled = true;
  const loading = createLoadingMessage();

  try {
    await streamQuery(q, loading);
  } catch (err) {
    loading.remove();
    appendMessage("system", `错误：${err.message || err}`);
  } finally {
    btnSend.disabled = false;
    queryInput.focus();
  }
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

btnClear.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  getMessagesEnd();
  appendMessage("system", "对话已清空。");
  scrollPinnedToBottom = true;
});

function renderFileList() {
  fileList.innerHTML = "";
  pendingFiles.forEach((f) => {
    const li = document.createElement("li");
    li.textContent = `${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
    fileList.appendChild(li);
  });
  btnIngest.disabled = pendingFiles.length === 0;
}

function addFiles(fileListLike) {
  for (const f of fileListLike) {
    if (!pendingFiles.some((p) => p.name === f.name && p.size === f.size)) {
      pendingFiles.push(f);
    }
  }
  renderFileList();
}

fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});

btnIngest.addEventListener("click", async () => {
  if (!pendingFiles.length) return;
  btnIngest.disabled = true;
  ingestStatus.textContent = "灌库进行中，请稍候…";

  const fd = new FormData();
  pendingFiles.forEach((f) => fd.append("files", f));

  try {
    const res = await fetch("/api/ingest", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    ingestStatus.textContent = `完成：成功 ${data.ok}，失败 ${data.fail}`;
    appendMessage("system", `灌库完成：成功 ${data.ok} 个文件，失败 ${data.fail} 个。`);
    pendingFiles = [];
    renderFileList();
  } catch (err) {
    ingestStatus.textContent = `失败：${err.message || err}`;
    appendMessage("system", `灌库失败：${err.message || err}`);
  } finally {
    btnIngest.disabled = pendingFiles.length === 0;
  }
});

appendMessage("system", "欢迎使用 RAG-Anything。左侧可查看工作目录与模式，下方输入问题开始对话。");
initMessagesScroll();
pinMessagesEnd();
refreshStatus();
setInterval(refreshStatus, 15000);
