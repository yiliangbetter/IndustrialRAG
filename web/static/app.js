const $ = (sel) => document.querySelector(sel);

const messagesEl = $("#messages");
const queryInput = $("#query-input");
const queryMode = $("#query-mode");
const composer = $("#composer");
const btnSend = $("#btn-send");
const btnClear = $("#btn-clear");
const btnIngest = $("#btn-ingest");
const btnIngestLog = $("#btn-ingest-log");
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
const kbBaseDir = $("#kb-base-dir");
const btnKbApply = $("#btn-kb-apply");
const kbPathStatus = $("#kb-path-status");
const ingestKbBase = $("#ingest-kb-base");

let pendingFiles = [];
let ingestBusy = false;
let ingestStopping = false;
let multimodalSyncBusy = false;
let queryDebugSyncBusy = false;
let kbPathSyncBusy = false;
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

const PHASE_STATUS_TEXT = {
  retrieve: "正在检索知识库…",
  rerank: "正在 Rerank…",
  generate: "正在生成回答…",
  bundle: "正在复用检索结果…",
  clarify: "正在生成推荐问法…",
};

function phaseStatusText(phase, fallback) {
  if (fallback) return fallback;
  if (phase && PHASE_STATUS_TEXT[phase]) return PHASE_STATUS_TEXT[phase];
  return "正在处理您的问题…";
}

function createLoadingMessage() {
  const el = document.createElement(`d` + `iv`);
  el.className = "msg assistant loading";
  el.innerHTML = `
    <div class="loading-body">
      <span class="loading-text">正在处理您的问题…</span>
      <span class="loading-dots" aria-hidden="true"><i></i><i></i><i></i></span>
    </div>
  `;
  messagesEl.insertBefore(el, getMessagesEnd());
  pinMessagesEnd();
  scrollMessages(true);
  watchMessageResize(el);
  return el;
}

function setLoadingStatus(el, text, phase) {
  const textEl = el.querySelector(".loading-text");
  if (textEl) textEl.textContent = phaseStatusText(phase, text);
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

  const pendingScope = el.dataset.pendingRetrievalScope;
  if (pendingScope) {
    delete el.dataset.pendingRetrievalScope;
    try {
      showRetrievalScope(
        {
          scopeEl,
          thinkingBlock,
          thinkingText,
          imagesEl,
          answerLabel,
          answerMd,
          thinkingRaw: "",
          answerRaw: "",
          inlineFiguresApplied: false,
        },
        JSON.parse(pendingScope),
      );
    } catch {
      /* ignore */
    }
  }

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

function answerBodyForPlacement(rawText) {
  const text = (rawText || "").trim();
  const cut = text.search(/\n###\s*References\b/i);
  return cut >= 0 ? text.slice(0, cut).trim() : text;
}

/** ### heading title containing match_start (machine section in listing answers). */
function machineSectionTitleAtOffset(rawText, matchStart) {
  const body = answerBodyForPlacement(rawText);
  if (matchStart == null || matchStart < 0) return "";
  const before = body.slice(0, matchStart);
  const matches = [...before.matchAll(/^###\s+([^\n]+)/gm)];
  if (!matches.length) return "";
  return matches[matches.length - 1][1].trim();
}

function normalizeSectionTitle(title) {
  return (title || "").replace(/\s+/g, "").trim();
}

/** 0-based ### index in answer body (excludes References) at matchStart. */
function machineSectionHeadingIndexAtOffset(rawText, matchStart) {
  const body = answerBodyForPlacement(rawText);
  if (matchStart == null || matchStart < 0) return -1;
  const matches = [...body.slice(0, matchStart).matchAll(/^###\s+([^\n]+)/gm)];
  if (!matches.length) return -1;
  return matches.length - 1;
}

/** h3 nodes for answer body only (skip References heading in full render). */
function answerBodyH3Elements(root) {
  if (!root) return [];
  return [...root.querySelectorAll("h3")].filter((h3) => {
    const t = normalizeSectionTitle(h3.textContent || "");
    return t && !/^references$/i.test(t);
  });
}

function findSectionHeading(root, sectionTitle, rawText, matchStart) {
  if (!root || !sectionTitle) return null;
  const want = normalizeSectionTitle(sectionTitle);
  const h3s = answerBodyH3Elements(root);
  for (const h3 of h3s) {
    const got = normalizeSectionTitle(h3.textContent || "");
    if (got && got === want) return h3;
  }
  if (rawText != null && matchStart != null && matchStart >= 0) {
    const idx = machineSectionHeadingIndexAtOffset(rawText, matchStart);
    if (idx >= 0 && idx < h3s.length) return h3s[idx];
  }
  return null;
}

function machineHintFromSourceKey(sourceKey) {
  return (sourceKey || "")
    .replace(/\s+/g, "")
    .replace(/维护保养手册.*$/i, "")
    .replace(/手册.*$/i, "")
    .trim();
}

function sectionForImageIndex(imageIndex, placements, rawText, images) {
  const pl = (placements || []).find((p) => p.image_index === imageIndex);
  if (pl) return sectionKeyForPlacement(rawText, pl);
  const img = images?.[imageIndex];
  if (!img) return "";
  const hint = machineHintFromSourceKey(img.source_key);
  if (!hint) return "";
  const body = answerBodyForPlacement(rawText);
  for (const m of body.matchAll(/^###\s+([^\n]+)/gm)) {
    const title = normalizeSectionTitle(m[1]);
    if (title === hint) return m[1].trim();
  }
  return "";
}

/** Dedup within a section: manual (source_key) + caption; URL when caption is too short. */
function imageDisplayDedupKey(img) {
  const manual = normalizeSectionTitle(
    machineHintFromSourceKey(img?.source_key || "")
  );
  const cap = normalizePlacementLine(img?.caption || "");
  const url = (img?.url || "").trim();
  if (cap.length >= 8) {
    return manual ? `${manual}::${cap}` : cap;
  }
  return manual ? `${manual}::${url}` : url;
}

function isAnswerItemLine(text) {
  const t = (text || "").trim();
  if (!t) return false;
  if (/^###\s/.test(t)) return false;
  return true;
}

/** Start offsets of each non-empty answer line before References (excludes ### headings). */
function answerItemLineStarts(rawText) {
  const body = answerBodyForPlacement(rawText);
  const starts = [];
  let offset = 0;
  for (const line of body.split("\n")) {
    if (isAnswerItemLine(line)) {
      const lead = line.search(/\S/);
      starts.push(offset + (lead >= 0 ? lead : 0));
    }
    offset += line.length + 1;
  }
  return starts;
}

/** 0-based answer line index for a placement offset (backend match_start). */
function itemIndexForMatchStart(rawText, matchStart) {
  if (matchStart == null || matchStart < 0) return -1;
  const starts = answerItemLineStarts(rawText);
  if (!starts.length) return -1;
  let idx = 0;
  for (let i = 0; i < starts.length; i++) {
    if (starts[i] <= matchStart) idx = i;
    else break;
  }
  return idx;
}

function sectionKeyForPlacement(rawText, placement) {
  const titled = machineSectionTitleAtOffset(rawText, placement?.match_start ?? 0);
  if (titled) return normalizeSectionTitle(titled);
  const idx = itemIndexForMatchStart(rawText, placement?.match_start ?? -1);
  if (idx >= 0) return `line:${idx}`;
  return "_";
}

function elementBeforeReferences(el, refsH3) {
  if (!el || !refsH3) return true;
  return !!(
    el.compareDocumentPosition(refsH3) & Node.DOCUMENT_POSITION_FOLLOWING
  );
}

/** Ordered DOM anchors (insert after) for each answer line before References. */
function collectAnswerLineAnchors(root) {
  if (!root) return [];
  const refs = findReferencesHeading(root);
  const anchors = [];

  function addParagraphLineAnchors(p) {
    const nodes = [...p.childNodes];
    const hasBr = nodes.some((n) => n.nodeName === "BR");
    if (!hasBr) {
      if ((p.textContent || "").trim()) anchors.push(p);
      return;
    }
    let segment = "";
    for (const node of nodes) {
      if (node.nodeName === "BR") {
        if (segment.trim()) anchors.push(node);
        segment = "";
        continue;
      }
      segment += node.textContent || "";
    }
    if (segment.trim()) anchors.push(p);
  }

  function walk(parent) {
    for (const child of parent.children) {
      if (refs && !elementBeforeReferences(child, refs)) break;
      if (child.matches?.("h3")) {
        const t = normalizeSectionTitle(child.textContent || "");
        if (/^references$/i.test(t)) break;
        continue;
      }
      if (child.matches?.("ul, ol")) {
        for (const li of child.children) {
          if (!li.matches?.("li")) continue;
          if (refs && !elementBeforeReferences(li, refs)) break;
          if ((li.textContent || "").trim()) anchors.push(li);
        }
        continue;
      }
      if (child.matches?.("p")) {
        addParagraphLineAnchors(child);
        continue;
      }
      if (child.children?.length) walk(child);
    }
  }

  walk(root);
  return anchors;
}

function findReferencesHeading(root) {
  if (!root) return null;
  for (const h3 of root.querySelectorAll("h3")) {
    const t = normalizeSectionTitle(h3.textContent || "");
    if (/^references$/i.test(t)) return h3;
  }
  return null;
}

function findInsertPointByItemIndex(root, rawText, placement) {
  const idx = itemIndexForMatchStart(rawText, placement?.match_start ?? -1);
  if (idx < 0) return null;
  const anchors = collectAnswerLineAnchors(root);
  return idx < anchors.length ? anchors[idx] : null;
}

function filterPlacementsBySectionDedup(placements, images, rawText) {
  const seenBySection = new Map();
  const sorted = [...placements].sort(
    (a, b) => (a.match_start || 0) - (b.match_start || 0)
  );
  const kept = [];
  for (const pl of sorted) {
    const img = images[pl.image_index];
    if (!img) continue;
    const section = sectionKeyForPlacement(rawText, pl);
    const key = imageDisplayDedupKey(img);
    if (!seenBySection.has(section)) seenBySection.set(section, new Set());
    const sectionSeen = seenBySection.get(section);
    if (key && sectionSeen.has(key)) continue;
    if (key) sectionSeen.add(key);
    kept.push(pl);
  }
  return kept;
}

function dedupImagesForDisplay(images, rawText, placements) {
  const seenBySection = new Map();
  return (images || []).filter((img, idx) => {
    const section =
      sectionForImageIndex(idx, placements, rawText, images) || "_";
    const key = imageDisplayDedupKey(img);
    if (!key) return true;
    if (!seenBySection.has(section)) seenBySection.set(section, new Set());
    const sectionSeen = seenBySection.get(section);
    if (sectionSeen.has(key)) return false;
    sectionSeen.add(key);
    return true;
  });
}

function findInsertPointInSection(root, rawText, placement) {
  if (!root) return null;
  const sectionTitle = machineSectionTitleAtOffset(
    rawText,
    placement?.match_start ?? 0
  );
  const sectionH3 = findSectionHeading(
    root,
    sectionTitle,
    rawText,
    placement?.match_start ?? -1
  );
  if (!sectionH3) return null;

  const rawLine = lineAtPlacementOffset(rawText, placement);
  if (!rawLine) return null;

  let el = sectionH3.nextElementSibling;
  while (el && el.nodeName !== "H3") {
    const lis = el.matches?.("li") ? [el] : [...(el.querySelectorAll?.("li") || [])];
    for (const li of lis) {
      if (placementLinesMatch(li.textContent || "", rawLine)) return li;
    }
    if (el.matches?.("p")) {
      let segment = "";
      for (const child of el.childNodes) {
        if (child.nodeName === "BR") {
          if (placementLinesMatch(segment, rawLine)) return child;
          segment = "";
          continue;
        }
        segment += child.textContent || "";
      }
      if (placementLinesMatch(segment, rawLine)) return el;
    }
    el = el.nextElementSibling;
  }
  return null;
}

function lineAtMatchStart(rawText, matchStart) {
  if (!rawText || matchStart == null || matchStart < 0) return "";
  const body = answerBodyForPlacement(rawText);
  const lineStart = body.lastIndexOf("\n", matchStart) + 1;
  const lineEnd = body.indexOf("\n", matchStart);
  return body
    .slice(lineStart, lineEnd < 0 ? body.length : lineEnd)
    .trim();
}

function normalizePlacementLine(text) {
  return (text || "")
    .replace(/\*\*/g, "")
    .replace(/^\s*(?:\d+\.\s*)?[-*•]\s+/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function lineHeadForMatch(text) {
  return normalizePlacementLine(text).split(/[：:（(]/)[0].trim();
}

function placementLinesMatch(domText, rawLine) {
  const a = normalizePlacementLine(domText);
  const b = normalizePlacementLine(rawLine);
  if (!a || !b) return false;
  if (a === b) return true;
  const headA = lineHeadForMatch(domText);
  const headB = lineHeadForMatch(rawLine);
  return headB.length >= 4 && headA === headB;
}

/** Resolve the answer line at placement.match_start (offset-primary). */
function lineAtPlacementOffset(rawText, placement) {
  const matchStart = placement?.match_start;
  if (rawText && matchStart != null && matchStart >= 0) {
    const line = lineAtMatchStart(rawText, matchStart);
    if (line) return line;
  }
  return (placement?.anchor_text || "").trim();
}

/**
 * Phase B: insert after the DOM line that corresponds to match_start in answerRaw.
 * Falls back to anchor_text block search when offset line cannot be located.
 */
function findInsertPointByOffset(root, rawText, placement) {
  if (!root) return null;
  const rawLine = lineAtPlacementOffset(rawText, placement);
  if (!rawLine) return null;

  for (const li of root.querySelectorAll("li")) {
    if (placementLinesMatch(li.textContent || "", rawLine)) return li;
  }

  for (const p of root.querySelectorAll("p")) {
    let segment = "";
    for (const child of p.childNodes) {
      if (child.nodeName === "BR") {
        if (placementLinesMatch(segment, rawLine)) return child;
        segment = "";
        continue;
      }
      segment += child.textContent || "";
    }
    if (placementLinesMatch(segment, rawLine)) {
      return p;
    }
  }

  return null;
}

function findInsertPointForPlacement(root, rawText, placement) {
  const inSection = findInsertPointInSection(root, rawText, placement);
  if (inSection) return inSection;
  const byItem = findInsertPointByItemIndex(root, rawText, placement);
  if (byItem) return byItem;
  const byOffset = findInsertPointByOffset(root, rawText, placement);
  if (byOffset) return byOffset;
  const rawLine = lineAtPlacementOffset(rawText, placement);
  if (rawLine && (placement?.match_start || 0) <= 1) {
    const firstP = root.querySelector("p");
    if (firstP && placementLinesMatch(firstP.textContent || "", rawLine)) {
      return firstP;
    }
  }
  const block = findBlockForAnchor(
    root,
    placement.anchor_text,
    placement.match_start,
    rawText
  );
  if (!block) return null;
  return findInsertAfterForPlacement(block, placement.match_start, rawText);
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
  const seg = normalizePlacementLine(segment);
  const line = normalizePlacementLine(lineAnchor);
  if (!seg || !line) return false;
  if (seg === line) return true;
  const headSeg = lineHeadForMatch(segment);
  const headLine = lineHeadForMatch(lineAnchor);
  return headLine.length >= 4 && headSeg === headLine;
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
    return block;
  }
  return block;
}

/** Insert block figure after a paragraph/list item or a line break within a paragraph. */
function insertFigureAfterAnchor(anchor, fig) {
  if (!anchor || !fig) return false;
  let el = anchor;
  if (el.nodeType === Node.TEXT_NODE) {
    el = el.parentElement;
  }
  if (el?.nodeName === "BR") {
    el.insertAdjacentElement("afterend", fig);
    return true;
  }
  if (el?.matches?.("p, li")) {
    el.insertAdjacentElement("afterend", fig);
    return true;
  }
  if (el) {
    el.insertAdjacentElement("afterend", fig);
    return true;
  }
  return false;
}

function appendFiguresToAnswerEnd(answerMd, images) {
  if (!answerMd || !images?.length) return 0;
  const frag = document.createDocumentFragment();
  let count = 0;
  for (const img of images) {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = figureHtml(img);
    const fig = wrapper.firstElementChild;
    if (!fig) continue;
    frag.appendChild(fig);
    count += 1;
  }
  if (!count) return 0;
  const refsH3 = findReferencesHeading(answerMd);
  if (refsH3) {
    refsH3.insertAdjacentElement("beforebegin", frag);
  } else {
    answerMd.appendChild(frag);
  }
  return count;
}

function showAnswerImageFallback(ui, images) {
  if (!ui?.imagesEl || !images?.length) return;
  ui.imagesEl.hidden = false;
  ui.imagesEl.innerHTML = images.map(figureHtml).join("");
}

function applyInlineImages(ui, ev) {
  if (!ui?.answerMd || !ev?.placements?.length || !ev?.images?.length) return;
  if (ui.imagesEl) ui.imagesEl.hidden = true;
  const rawText = ui.answerRaw || "";
  renderMarkdown(ui.answerMd, rawText);
  const ordered = filterPlacementsBySectionDedup(
    ev.placements,
    ev.images,
    rawText
  );
  const pending = [];
  for (const pl of ordered) {
    const img = ev.images[pl.image_index];
    if (!img || !pl.anchor_text) continue;
    const insertAfter = findInsertPointForPlacement(ui.answerMd, rawText, pl);
    if (!insertAfter) continue;
    const wrapper = document.createElement("div");
    wrapper.innerHTML = figureHtml(img);
    const fig = wrapper.firstElementChild;
    if (!fig) continue;
    pending.push({ insertAfter, fig });
  }
  let inserted = 0;
  for (let i = pending.length - 1; i >= 0; i--) {
    if (insertFigureAfterAnchor(pending[i].insertAfter, pending[i].fig)) {
      inserted += 1;
    }
  }
  ui.inlineFiguresApplied = inserted > 0;
  const displayImages = dedupImagesForDisplay(
    ev.images,
    rawText,
    ev.placements
  );
  if (!ui.inlineFiguresApplied && displayImages.length) {
    const appended = appendFiguresToAnswerEnd(ui.answerMd, displayImages);
    ui.inlineFiguresApplied = appended > 0;
  }
  if (!ui.inlineFiguresApplied && displayImages.length) {
    showAnswerImageFallback(ui, displayImages);
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

function syncKnowledgeBasePathFields(h) {
  if (!h) return;
  const base =
    h.kb_base_dir ||
    (h.working_dir && h.parser_output_dir
      ? `${h.working_dir} + ${h.parser_output_dir}`
      : "");
  if (kbBaseDir && !kbPathSyncBusy && base && h.kb_base_dir) {
    kbBaseDir.value = h.kb_base_dir;
  }
  const metaKb = $("#meta-kb-base");
  if (metaKb) {
    metaKb.textContent = h.kb_base_dir || "（自定义子路径）";
  }
  if ($("#meta-wd")) {
    $("#meta-wd").textContent = h.working_dir || "—";
  }
  if ($("#meta-pod")) {
    $("#meta-pod").textContent = h.parser_output_dir || "—";
  }
  if (ingestKbBase) {
    ingestKbBase.textContent = h.kb_base_dir || h.working_dir || "—";
  }
}

async function applyKnowledgeBasePath() {
  if (!kbBaseDir || kbPathSyncBusy || ingestBusy) return;
  const base_dir = (kbBaseDir.value || "").trim();
  if (!base_dir) {
    if (kbPathStatus) {
      kbPathStatus.textContent = "请填写知识库根目录，例如 data/kb_new";
      kbPathStatus.classList.remove("hidden");
      kbPathStatus.className = "hint status-bad";
    }
    return;
  }
  kbPathSyncBusy = true;
  if (btnKbApply) btnKbApply.disabled = true;
  if (kbPathStatus) {
    kbPathStatus.textContent = "正在切换知识库并重新加载引擎…";
    kbPathStatus.classList.remove("hidden", "status-bad", "status-ok");
  }
  try {
    const res = await fetch("/api/knowledge-base/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_dir }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      const msg =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || d).join("; ")
            : data.message || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    if (kbPathStatus) {
      kbPathStatus.textContent = data.message || "知识库路径已切换。";
      kbPathStatus.classList.add("status-ok");
    }
    appendMessage("system", data.message || `已切换知识库：${base_dir}`);
    await refreshStatus();
  } catch (err) {
    if (kbPathStatus) {
      kbPathStatus.textContent = `切换失败：${err.message || err}`;
      kbPathStatus.classList.add("status-bad");
    }
    appendMessage("system", `知识库路径切换失败：${err.message || err}`);
  } finally {
    kbPathSyncBusy = false;
    if (btnKbApply) btnKbApply.disabled = false;
    updateIngestControls();
  }
}

async function refreshStatus() {
  try {
    const [h, setup] = await Promise.all([fetchHealth(), fetchSetupStatus().catch(() => null)]);
    latestSetupStatus = setup;
    $("#meta-ready").textContent = h.ready ? "是" : "否";
    $("#meta-ready").className = h.ready ? "status-ok" : "status-bad";
    syncKnowledgeBasePathFields(h);
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

function renderClarificationPanel(loadingEl, data) {
  stopLoadingMessage(loadingEl);
  loadingEl.classList.add("clarification-panel");
  loadingEl.classList.remove("streaming");
  loadingEl.innerHTML = "";

  const clarificationId = String(data?.clarification_id || "").trim();
  const originalQuery = String(data?.original_query || "").trim();
  if (clarificationId) {
    loadingEl.dataset.clarificationId = clarificationId;
  }
  if (originalQuery) {
    loadingEl.dataset.originalQuery = originalQuery;
  }

  const roleEl = document.createElement(`d` + `iv`);
  roleEl.className = "role";
  roleEl.textContent = "助手";
  loadingEl.appendChild(roleEl);

  const bodyEl = document.createElement(`d` + `iv`);
  bodyEl.className = "body";

  const unrelated = Boolean(data?.unrelated);
  const unanswerable = Boolean(data?.unanswerable);
  const gateOutcome = data?.gate_outcome;
  const showOptions =
    gateOutcome === "offer" ||
    (!gateOutcome && !unrelated && !unanswerable && (data?.candidates || []).length > 0);
  const intro = document.createElement("p");
  intro.className =
    unrelated || unanswerable ? "clarify-intro clarify-unanswerable" : "clarify-intro";
  if (unrelated || unanswerable || gateOutcome === "reject") {
    intro.textContent =
      data?.message ||
      "您的问题与当前知识库内容关联度较低，暂无法基于知识库作答。请尝试换种说法，或联系技术支持。";
  } else {
    const k = data?.generation?.k_answerable ?? (data?.candidates || []).length;
    const hasKeepOriginal = Boolean(data?.keep_original?.query);
    intro.textContent =
      k > 0
        ? hasKeepOriginal
          ? "请从下列推荐问法中选择一条，或保持原问继续作答。"
          : "请从下列已验证可检索的推荐问法中选择一条继续问答。"
        : "正在准备澄清选项…";
  }
  bodyEl.appendChild(intro);

  const preview = document.createElement(`d` + `iv`);
  preview.className = "clarify-preview";
  preview.innerHTML = `原问：<pre>${escapeHtml(data?.original_query || "")}</pre>`;
  bodyEl.appendChild(preview);

  if (showOptions) {
    const options = document.createElement(`d` + `iv`);
    options.className = "clarify-options";

    const keepOriginal = data?.keep_original;
    if (keepOriginal?.query) {
      const keepBtn = document.createElement("button");
      keepBtn.type = "button";
      keepBtn.className = "clarify-option clarify-keep-original";
      const keepScore =
        keepOriginal.final_score != null ? ` · final ${keepOriginal.final_score}` : "";
      const keepHint =
        keepOriginal.bundle_cached === false ? "（将重新检索）" : "";
      keepBtn.textContent = `保持原问继续${keepScore}${keepHint}`;
      keepBtn.addEventListener("click", () => {
        void submitClarifiedQuery({
          query: keepOriginal.query,
          clarify_choice: "keep_original",
          clarification_id: clarificationId || loadingEl.dataset.clarificationId,
        });
      });
      options.appendChild(keepBtn);
    }

    for (const cand of data?.candidates || []) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "clarify-option";
      const scoreTxt =
        cand.final_score != null
          ? ` · final ${cand.final_score}`
          : cand.max_rerank_score != null
            ? ` · rerank ${cand.max_rerank_score}`
            : cand.score != null
              ? ` · ${cand.score}`
              : "";
      const chunkTxt =
        cand.chunk_count != null ? ` · ${cand.chunk_count} chunks` : "";
      btn.textContent = `${cand.text}${chunkTxt}${scoreTxt}`;
      btn.dataset.candidateId = cand.id;
      btn.addEventListener("click", () => {
        void submitClarifiedQuery({
          query: cand.text,
          clarify_choice: "use_candidate",
          clarification_id: clarificationId || loadingEl.dataset.clarificationId,
          candidate_id: cand.id,
        });
      });
      options.appendChild(btn);
    }

    bodyEl.appendChild(options);
  }

  loadingEl.appendChild(bodyEl);
  scrollMessages(true);
}

async function submitClarifiedQuery(opts) {
  const q = (opts?.query || "").trim();
  if (!q) return;

  appendMessage("user", q);
  scrollPinnedToBottom = true;
  btnSend.disabled = true;
  const loading = createLoadingMessage();
  markClarifyOptionUsed(opts);

  try {
    await streamQuery(opts, loading);
  } catch (err) {
    loading.remove();
    appendMessage("system", `错误：${err.message || err}`);
  } finally {
    btnSend.disabled = false;
    queryInput.focus();
  }
}

function markClarifyOptionUsed(opts) {
  const cid = String(opts?.clarification_id || "").trim();
  if (!cid) return;
  const panels = document.querySelectorAll(".clarification-panel");
  for (const panel of panels) {
    if (String(panel.dataset.clarificationId || "").trim() !== cid) continue;
    if (opts.clarify_choice === "keep_original") {
      const btn = panel.querySelector(".clarify-keep-original");
      if (btn) {
        btn.disabled = true;
        btn.classList.add("clarify-option-used");
        btn.textContent = "原问已回答";
      }
      continue;
    }
    if (opts.clarify_choice === "use_candidate" && opts.candidate_id) {
      for (const btn of panel.querySelectorAll(".clarify-option")) {
        if (btn.dataset.candidateId === opts.candidate_id) {
          btn.disabled = true;
          btn.classList.add("clarify-option-used");
        }
      }
    }
  }
}

async function streamQuery(queryOrOpts, loadingEl) {
  const opts =
    typeof queryOrOpts === "string" ? { query: queryOrOpts } : { ...queryOrOpts };
  const q = (opts.query || "").trim();
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: q,
      mode: queryMode.value,
      stream: true,
      clarify_choice: opts.clarify_choice || null,
      clarification_id: opts.clarification_id || null,
      candidate_id: opts.candidate_id || null,
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
      if (!gotContent) setLoadingStatus(loadingEl, ev.text, ev.phase);
      return;
    }
    if (ev.type === "retrieval_scope") {
      if (loadingEl.classList.contains("loading")) {
        loadingEl.dataset.pendingRetrievalScope = JSON.stringify(ev);
        return;
      }
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
    if (ev.type === "clarification_required") {
      renderClarificationPanel(loadingEl, ev.data);
      gotContent = true;
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
      if (ev.clarification_only) {
        scrollMessages(true);
        return;
      }
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
  } else if (!loadingEl.classList.contains("clarification-panel")) {
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
  if (btnKbApply) {
    btnKbApply.disabled = ingestBusy || kbPathSyncBusy;
  }
  if (kbBaseDir) {
    kbBaseDir.disabled = ingestBusy || kbPathSyncBusy;
  }
  if (enableMultimodal) {
    enableMultimodal.disabled = ingestBusy || multimodalSyncBusy;
  }
  if (btnStopIngest) {
    btnStopIngest.classList.toggle("hidden", !ingestBusy);
    btnStopIngest.disabled = !ingestBusy || ingestStopping;
  }
  if (btnIngestLog) {
    const modal = typeof getIngestTerminalModal === "function" ? getIngestTerminalModal() : null;
    const hasLog =
      (ingestLog && (ingestLog.textContent || "").trim().length > 0) ||
      (modal?.logEl && (modal.logEl.textContent || "").trim().length > 0);
    btnIngestLog.classList.toggle("hidden", !ingestBusy && !hasLog);
  }
  if (typeof updateIngestTerminalStopState === "function") {
    updateIngestTerminalStopState(ingestBusy, ingestStopping);
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

btnIngestLog?.addEventListener("click", () => {
  if (typeof showIngestTerminalModal === "function") showIngestTerminalModal();
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

btnKbApply?.addEventListener("click", () => {
  void applyKnowledgeBasePath();
});

kbBaseDir?.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    void applyKnowledgeBasePath();
  }
});

appendMessage("system", "欢迎使用南兴知识库问答助手。左侧可查看工作目录与模式，下方输入问题开始对话。");
initMessagesScroll();
pinMessagesEnd();
refreshStatus();
refreshQueryDebugPanel();
setInterval(refreshStatus, 15000);
