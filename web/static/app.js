const $ = (sel) => document.querySelector(sel);

const messagesEl = $("#messages");
const queryInput = $("#query-input");
const queryMode = $("#query-mode");
const composer = $("#composer");
const btnSend = $("#btn-send");
const btnClear = $("#btn-clear");
const btnIngest = $("#btn-ingest");
const btnStopIngest = $("#btn-stop-ingest");
const fileInput = $("#file-input");
const uploadZone = $("#upload-zone");
const fileList = $("#file-list");
const fileListToolbar = $("#file-list-toolbar");
const fileListCount = $("#file-list-count");
const btnClearFiles = $("#btn-clear-files");
const ingestStatus = $("#ingest-status");
const ingestProgress = $("#ingest-progress");
const ingestProgressBar = $("#ingest-progress-bar");
const ingestProgressLabel = $("#ingest-progress-label");
const ingestLog = $("#ingest-log");
const enableMultimodal = $("#enable-multimodal");
const multimodalHint = $("#multimodal-hint");
const queryDebugDump = $("#query-debug-dump");
const queryDebugHint = $("#query-debug-hint");

let pendingFiles = [];
let ingestBusy = false;
let ingestStopping = false;
let multimodalSyncBusy = false;
let queryDebugSyncBusy = false;
let latestSetupStatus = null;
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

  const imagesEl = document.createElement(`d` + `iv`);
  imagesEl.className = "related-images";
  imagesEl.hidden = true;

  const answerLabel = document.createElement(`d` + `iv`);
  answerLabel.className = "answer-label";
  answerLabel.textContent = "回答";
  answerLabel.hidden = true;

  const answerMd = document.createElement(`d` + `iv`);
  answerMd.className = "answer-md markdown-body";

  bodyEl.append(thinkingBlock, scopeEl, imagesEl, answerLabel, answerMd);
  el.append(roleEl, bodyEl);
  pinMessagesEnd();
  scrollMessages(true);
  watchMessageResize(el);

  return {
    thinkingBlock,
    thinkingText,
    scopeEl,
    imagesEl,
    answerLabel,
    answerMd,
    thinkingRaw: "",
    answerRaw: "",
    inlineFiguresApplied: false,
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

function figureHtml(img) {
  const cap = img.caption ? escapeHtml(img.caption) : "相关图片";
  const page =
    img.page != null
      ? `<span class="img-page">第 ${escapeHtml(String(img.page))} 页</span>`
      : "";
  const url = escapeHtml(img.url || "");
  return `<figure class="inline-answer-figure">
    <a href="${url}" target="_blank" rel="noopener noreferrer">
      <img src="${url}" alt="${cap}" loading="lazy" />
    </a>
    <figcaption>${cap}${page}</figcaption>
  </figure>`;
}

function anchorNeedles(anchor) {
  let bare = (anchor || "").replace(/\*\*/g, "").trim();
  bare = bare.replace(/^[*\-•]\s+/, "").trim();
  const needles = [];
  if (bare) needles.push(bare);
  const bold = (anchor || "").match(/\*\*([^*]+)\*\*/);
  if (bold) {
    const term = bold[1].trim();
    if (term.length >= 2 && !needles.includes(term)) needles.push(term);
  }
  const head = bare.split(/[：:（(]/)[0].trim();
  if (head && head.length >= 2 && !needles.includes(head)) needles.push(head);
  return needles;
}

function lineAtMatchStart(rawText, matchStart) {
  if (!rawText || matchStart == null || matchStart < 0) return "";
  const lineStart = rawText.lastIndexOf("\n", matchStart) + 1;
  const lineEnd = rawText.indexOf("\n", matchStart);
  return rawText
    .slice(lineStart, lineEnd < 0 ? rawText.length : lineEnd)
    .trim();
}

function findBlockForAnchor(root, anchor, matchStart, rawText) {
  if (!root) return null;
  const lineAnchor = lineAtMatchStart(rawText, matchStart);
  const needles = [];
  if (lineAnchor && lineAnchor.length >= 4) needles.push(lineAnchor);
  for (const needle of anchorNeedles(anchor)) {
    if (!needles.includes(needle)) needles.push(needle);
  }
  for (const needle of needles) {
    const candidates = root.querySelectorAll("li, p");
    for (const el of candidates) {
      if ((el.textContent || "").includes(needle)) return el;
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
      if ((node.textContent || "").includes(needle)) {
        let el = node.parentElement;
        while (el && el !== root) {
          if (el.matches("li, p")) return el;
          el = el.parentElement;
        }
        return node.parentElement;
      }
      node = walker.nextNode();
    }
  }
  return null;
}

function lineSegmentMatches(segment, lineAnchor) {
  const seg = (segment || "").replace(/\*\*/g, "").trim();
  const line = (lineAnchor || "").replace(/\*\*/g, "").trim();
  if (!seg || !line) return false;
  if (seg.includes(line) || line.includes(seg)) return true;
  const head = line.split(/[：:（(]/)[0].trim();
  return head.length >= 4 && seg.includes(head);
}

/** Insert after a logical line inside a single <p> (marked breaks:true), else after block. */
function findInsertAfterForPlacement(block, matchStart, rawText) {
  if (!block) return null;
  if (block.matches("li")) return block;
  const lineAnchor = lineAtMatchStart(rawText, matchStart);
  if (!lineAnchor || !block.matches("p")) return block;

  let segment = "";
  for (const child of block.childNodes) {
    if (child.nodeName === "BR") {
      if (lineSegmentMatches(segment, lineAnchor)) return child;
      segment = "";
      continue;
    }
    segment += child.textContent || "";
  }
  if (lineSegmentMatches(segment, lineAnchor)) {
    const last = block.lastChild;
    return last && last.nodeType === Node.TEXT_NODE ? last : block;
  }
  return block;
}

function applyInlineImages(ui, ev) {
  if (!ui?.answerMd || !ev?.placements?.length || !ev?.images?.length) return;
  if (ui.imagesEl) ui.imagesEl.hidden = true;
  renderMarkdown(ui.answerMd, ui.answerRaw || "");
  const ordered = [...ev.placements].sort(
    (a, b) => (a.match_start || 0) - (b.match_start || 0)
  );
  let inserted = 0;
  for (const pl of ordered) {
    const img = ev.images[pl.image_index];
    if (!img || !pl.anchor_text) continue;
    const block = findBlockForAnchor(
      ui.answerMd,
      pl.anchor_text,
      pl.match_start,
      ui.answerRaw || ""
    );
    if (!block) continue;
    const insertAfter = findInsertAfterForPlacement(
      block,
      pl.match_start,
      ui.answerRaw || ""
    );
    if (!insertAfter) continue;
    const wrapper = document.createElement("div");
    wrapper.innerHTML = figureHtml(img);
    const fig = wrapper.firstElementChild;
    if (!fig) continue;
    insertAfter.insertAdjacentElement("afterend", fig);
    inserted += 1;
  }
  ui.inlineFiguresApplied = inserted > 0;
  if (!ui.inlineFiguresApplied && ui.imagesEl && ev.images?.length) {
    ui.imagesEl.hidden = false;
    ui.imagesEl.innerHTML = ev.images.map(figureHtml).join("");
  }
  scrollMessages();
}

function showRelatedImages(ui, ev) {
  if (ev?.type === "inline_images" || ev?.placements?.length) {
    applyInlineImages(ui, ev);
    return;
  }
  if (!ui?.imagesEl || !ev?.images?.length) return;
  ui.imagesEl.hidden = false;
  const cards = ev.images
    .map((img) => {
      const cap = img.caption ? escapeHtml(img.caption) : "相关图片";
      const page =
        img.page != null
          ? `<span class="img-page">第 ${escapeHtml(String(img.page))} 页</span>`
          : "";
      const url = escapeHtml(img.url || "");
      return `<figure class="related-img-card">
        <a href="${url}" target="_blank" rel="noopener noreferrer">
          <img src="${url}" alt="${cap}" loading="lazy" />
        </a>
        <figcaption>${cap}${page}</figcaption>
      </figure>`;
    })
    .join("");
  ui.imagesEl.innerHTML = `<div class="related-images-head">参考资料图片（来自文档原文，便于对照）</div><div class="related-images-grid">${cards}</div>`;
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

async function fetchQueryDebugStatus() {
  const res = await fetch("/api/dev/query-debug");
  if (!res.ok) throw new Error(`query debug ${res.status}`);
  return res.json();
}

function renderQueryDebugHint(data) {
  if (!queryDebugHint || !data) return;
  const dir = data.dump_dir || "logs/query_dumps";
  const recent = Array.isArray(data.recent) ? data.recent : [];
  const recentLine = recent.length
    ? `最近：${recent[0].name}`
    : "尚无 dump 文件";
  queryDebugHint.textContent = data.enabled
    ? `已开启。目录：${dir}。${recentLine}`
    : `关闭时不在磁盘写入。开启后每次问答会在 ${dir} 生成 JSON，便于排查检索与配图。`;
}

async function refreshQueryDebugPanel() {
  if (!queryDebugDump) return;
  try {
    const data = await fetchQueryDebugStatus();
    if (!queryDebugSyncBusy) {
      queryDebugDump.checked = Boolean(data.enabled);
    }
    renderQueryDebugHint(data);
  } catch {
    if (queryDebugHint) {
      queryDebugHint.textContent = "无法读取调试开关状态。";
    }
  }
}

async function fetchHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}

async function fetchSetupStatus() {
  const res = await fetch("/api/setup/status");
  if (!res.ok) throw new Error(`setup status ${res.status}`);
  return res.json();
}

function knowledgeBaseHasData() {
  return Boolean(latestSetupStatus?.knowledge_base_ok || latestSetupStatus?.kb_partial);
}

async function refreshStatus() {
  try {
    const [h, setup] = await Promise.all([fetchHealth(), fetchSetupStatus().catch(() => null)]);
    latestSetupStatus = setup;
    $("#meta-ready").textContent = h.ready ? "是" : "否";
    $("#meta-ready").className = h.ready ? "status-ok" : "status-bad";
    $("#meta-wd").textContent = h.working_dir || "—";
    $("#meta-mode").textContent = h.query_mode || "—";
    if ($("#meta-multimodal")) {
      $("#meta-multimodal").textContent = h.multimodal_enabled
        ? "多模态（图/表/公式）"
        : "仅文本";
    }
    if (enableMultimodal && !multimodalSyncBusy) {
      enableMultimodal.checked = Boolean(h.multimodal_enabled);
    }
    if (multimodalHint) {
      multimodalHint.textContent = h.multimodal_enabled
        ? "多模态已开启；切换后会重新加载引擎。"
        : "默认仅文本灌库；勾选以启用多模态。";
    }
    if (h.query_mode) queryMode.value = h.query_mode;
    const errEl = $("#meta-error");
    if (h.init_error) {
      errEl.textContent = h.init_error;
      errEl.classList.remove("hidden");
    } else {
      errEl.classList.add("hidden");
    }
    const setupLinkWrap = $("#setup-link-wrap");
    if (setupLinkWrap && h.client_mode) {
      setupLinkWrap.classList.remove("hidden");
    }
    btnSend.disabled = !h.ready;
    updateIngestControls();
    await refreshQueryDebugPanel();
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
  const q = (typeof query === "string" ? query : query?.query || "").trim();
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: q,
      mode: queryMode.value,
      stream: true,
    }),
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
    if (ev.type === "related_images") {
      if (!ui) {
        ui = prepareAssistantStream(loadingEl);
        gotContent = true;
      }
      showRelatedImages(ui, ev);
      return;
    }
    if (ev.type === "inline_images") {
      if (!ui) {
        ui = prepareAssistantStream(loadingEl);
        gotContent = true;
      }
      applyInlineImages(ui, ev);
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
    if (ev.type === "query_debug_saved") {
      const name = ev.name || "query dump";
      appendMessage("system", `已保存查询调试日志：${name}`);
      refreshQueryDebugPanel();
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
    bodyEl.textContent = "未收到回答，请查看服务端日志或稍后重试。";
    loadingEl.appendChild(roleEl);
    loadingEl.appendChild(bodyEl);
  } else {
    stopLoadingMessage(loadingEl);
    loadingEl.classList.remove("streaming");
    if (ui?.answerMd && ui.answerRaw && !ui.inlineFiguresApplied) {
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
  queryInput.dataset.lastQuery = q;
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

function updateIngestControls() {
  const hasFiles = pendingFiles.length > 0;
  const hasKb = knowledgeBaseHasData();
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
    btnIngest.textContent = hasKb ? "追加灌库" : "开始灌库";
  }
  btnIngest.disabled = !hasFiles || ingestBusy || multimodalSyncBusy;
  if (enableMultimodal) {
    enableMultimodal.disabled = ingestBusy || multimodalSyncBusy;
  }
  if (btnStopIngest) {
    btnStopIngest.classList.toggle("hidden", !ingestBusy);
    btnStopIngest.disabled = !ingestBusy || ingestStopping;
  }
}

function renderFileList() {
  renderPendingFileList(fileList, pendingFiles, {
    readOnly: ingestBusy,
    onRemove: (index) => {
      pendingFiles.splice(index, 1);
      renderFileList();
    },
  });
  updateIngestControls();
}

function clearPendingFiles() {
  pendingFiles = [];
  renderFileList();
}

function addFiles(fileListLike) {
  for (const f of fileListLike) {
    if (!pendingFiles.some((p) => p.name === f.name && p.size === f.size)) {
      pendingFiles.push(f);
    }
  }
  renderFileList();
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

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});

btnIngest.addEventListener("click", async () => {
  if (!pendingFiles.length || ingestBusy) return;

  if (knowledgeBaseHasData()) {
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
      ingestStatus.className = "hint error";
      appendMessage("system", "灌库已停止，知识库已清空。");
      pendingFiles = [];
    } else if (data.fail > 0) {
      ingestStatus.className = "hint error";
      ingestStatus.textContent =
        data.ok > 0
          ? `灌库完成：成功 ${data.ok} 篇，失败 ${data.fail} 篇（详见下方日志）`
          : `灌库失败：${data.fail} 篇均未成功（详见下方日志）`;
      if (ingestProgressBar) ingestProgressBar.classList.add("error");
      appendMessage(
        "system",
        data.ok > 0
          ? `灌库部分完成：成功 ${data.ok} 篇，失败 ${data.fail} 篇。`
          : `灌库失败：${data.fail} 篇均未成功。`
      );
    } else {
      ingestStatus.className = "hint status-ok";
      ingestStatus.textContent = `灌库完成：成功 ${data.ok} 篇`;
      appendMessage("system", `灌库完成：成功 ${data.ok} 篇。`);
      pendingFiles = [];
    }
    renderFileList();
  } catch (err) {
    ingestStatus.textContent = `失败：${err.message || err}`;
    appendMessage("system", `灌库失败：${err.message || err}`);
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

enableMultimodal?.addEventListener("change", async () => {
  if (multimodalSyncBusy || ingestBusy) return;
  const enabled = enableMultimodal.checked;
  multimodalSyncBusy = true;
  updateIngestControls();
  ingestStatus.textContent = enabled ? "正在开启多模态…" : "正在关闭多模态…";
  try {
    const res = await fetch("/api/setup/multimodal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    ingestStatus.textContent = data.message || (enabled ? "多模态已开启" : "多模态已关闭");
    await refreshStatus();
  } catch (err) {
    enableMultimodal.checked = !enabled;
    ingestStatus.textContent = `切换失败：${err.message || err}`;
  } finally {
    multimodalSyncBusy = false;
    updateIngestControls();
  }
});

queryDebugDump?.addEventListener("change", async () => {
  if (queryDebugSyncBusy) return;
  const enabled = queryDebugDump.checked;
  queryDebugSyncBusy = true;
  try {
    const res = await fetch("/api/dev/query-debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
    renderQueryDebugHint(data);
    appendMessage("system", data.message || (enabled ? "已开启查询调试日志。" : "已关闭查询调试日志。"));
  } catch (err) {
    queryDebugDump.checked = !enabled;
    appendMessage("system", `调试开关保存失败：${err.message || err}`);
  } finally {
    queryDebugSyncBusy = false;
  }
});

appendMessage("system", "欢迎使用南兴知识库问答助手。左侧可查看工作目录与模式，下方输入问题开始对话。");
initMessagesScroll();
pinMessagesEnd();
refreshStatus();
refreshQueryDebugPanel();
setInterval(refreshStatus, 15000);
