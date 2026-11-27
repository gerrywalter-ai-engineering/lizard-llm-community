// Lizard LLM Community Edition - onboarding and chat client.
// Four steps: machine -> model -> download -> chat. No framework, no build.

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  step: "machine",
  hardware: null,
  task: "",
  models: [],
  local: [],
  active: null,
  pick: null, // { repo, file, label, sizeGB, quant, id? }
  engine: "lizard-native",
  warmSource: null,
  warm: "cold",
  session: crypto.randomUUID(),
  busy: false,
};

const STEPS = ["machine", "model", "download", "chat"];

// ---------- helpers ----------
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error ?? `Request failed (${response.status})`);
    error.body = body;
    throw error;
  }
  return body;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function gb(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? `${number.toFixed(number >= 10 ? 0 : digits)} GB` : "unknown";
}

function bytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "";
  if (number >= 1024 ** 3) return `${(number / 1024 ** 3).toFixed(2)} GB`;
  return `${Math.round(number / 1024 ** 2)} MB`;
}

function duration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds % 60)} s`;
}

const FIT_LABEL = { fits: "Fits well", tight: "Tight fit", "too-large": "Too large", "no-fit": "Too large" };

function fitBadge(fit) {
  if (!fit) return null;
  return el("span", { class: "fit", "data-fit": fit, text: FIT_LABEL[fit] ?? fit });
}

// ---------- steps ----------
function stepReachable(step) {
  if (step === "machine" || step === "model") return Boolean(state.hardware) || step === "machine";
  if (step === "download") return Boolean(state.pick);
  if (step === "chat") return Boolean(state.active);
  return false;
}

function show(step, { focus = true } = {}) {
  state.step = step;
  for (const name of STEPS) {
    $(`#panel-${name}`).hidden = name !== step;
  }
  for (const button of $$("#steps button")) {
    const name = button.dataset.step;
    button.toggleAttribute("aria-current", name === step);
    if (name === step) button.setAttribute("aria-current", "step");
    button.disabled = !stepReachable(name);
  }
  if (step === "chat") enterChat();
  if (focus) $("#main").focus({ preventScroll: true });
  history.replaceState(null, "", step === "machine" ? location.pathname : `#${step}`);
}

function markDone(step, hint) {
  const button = $(`#steps button[data-step="${step}"]`);
  button.dataset.done = "true";
  if (hint) $(`#hint-${step}`).textContent = hint;
}

for (const button of $$("#steps button")) {
  button.addEventListener("click", () => show(button.dataset.step));
}

// ---------- heat capsule ----------
function setHeat(next, text) {
  state.warm = next;
  $("#heat").dataset.state = next;
  $("#heat-text").textContent = text;
  const ready = next === "warm";
  $("#prompt").disabled = !ready;
  $("#send").disabled = !ready || state.busy;
}

// ---------- 1. machine ----------
async function scan(fresh = false) {
  $("#specs").setAttribute("aria-busy", "true");
  $("#to-model").disabled = true;
  try {
    const { hardware } = await api(fresh ? "/api/scan?fresh=1" : "/api/scan");
    state.hardware = hardware;
    renderMachine(hardware);
    markDone("machine", `${gb(hardware.memory?.totalGB, 0)} RAM, ${hardware.primaryGpu?.name ? shortGpu(hardware.primaryGpu.name) : "CPU only"}`);
    $("#to-model").disabled = false;
  } catch (error) {
    $("#machine-verdict").textContent = `The hardware scan failed: ${error.message}. Lizard can still run models on the processor.`;
  } finally {
    $("#specs").setAttribute("aria-busy", "false");
    show(state.step, { focus: false });
  }
}

function shortGpu(name) {
  return String(name).replace(/\((R|TM)\)/gi, "").replace(/\s+/g, " ").replace(/^(NVIDIA|AMD|Intel)\s+/i, "").trim();
}

function renderMachine(hardware) {
  const cpu = hardware.cpu ?? {};
  const memory = hardware.memory ?? {};
  const gpu = hardware.primaryGpu;
  const rows = [
    ["Processor", String(cpu.model ?? "Unknown").replace(/\((R|TM)\)/gi, "").replace(/\s+/g, " "), cpu.physicalCores ? `${cpu.physicalCores} cores, ${cpu.logicalThreads ?? cpu.physicalCores} threads` : ""],
    ["Memory", gb(memory.totalGB, 0), memory.availableGB ? `${gb(memory.availableGB)} free right now` : ""],
    ["Graphics", gpu?.name ? shortGpu(gpu.name) : "No graphics card found", gpu?.vramGB ? `${gb(gpu.vramGB)} dedicated memory` : "Models run on the processor"],
    ["Model budget", gb(memory.plannerBudgetGB), "Room for weights plus the context cache"],
  ];
  $("#specs").replaceChildren(...rows.map(([label, value, note]) => el("div", {},
    el("dt", { text: label }),
    el("dd", {}, value, note ? el("small", { text: note }) : null),
  )));
  const budget = Number(memory.plannerBudgetGB) || 0;
  let verdict;
  if (budget >= 24) verdict = "Plenty of room. Models up to about 14 billion parameters run comfortably here.";
  else if (budget >= 12) verdict = "Good for everyday models: 4 to 8 billion parameters at Q4 run well.";
  else if (budget >= 6) verdict = "Best with small models of 1 to 4 billion parameters. They are quick and good at chat.";
  else verdict = "Memory is tight. Start with the smallest models; close other apps for more room.";
  $("#machine-verdict").textContent = verdict;
}

$("#rescan").addEventListener("click", () => scan(true));
$("#to-model").addEventListener("click", () => { show("model"); loadModels(); });

// ---------- 2. model ----------
async function loadModels() {
  const list = $("#model-list");
  list.setAttribute("aria-busy", "true");
  if (!state.models.length) list.replaceChildren(el("li", { class: "empty-line", text: "Finding models that fit..." }));
  try {
    const params = new URLSearchParams({ mode: "balanced" });
    if (state.task) params.set("task", state.task);
    const { models } = await api(`/api/models?${params}`);
    state.models = models;
    renderModels();
  } catch (error) {
    list.replaceChildren(el("li", { class: "empty-line", text: `Could not load the model list: ${error.message}` }));
  } finally {
    list.setAttribute("aria-busy", "false");
  }
}

function renderModels() {
  const order = { fits: 0, tight: 1 };
  const models = [...state.models].sort((a, b) => (order[a.fit] ?? 2) - (order[b.fit] ?? 2) || (b.featured ? 1 : 0) - (a.featured ? 1 : 0));
  if (!models.length) {
    $("#model-list").replaceChildren(el("li", { class: "empty-line", text: "No curated model matches this filter. Try Everything, or search Hugging Face below." }));
    return;
  }
  $("#model-list").replaceChildren(...models.map((model) => {
    const options = model.quantOptions ?? [];
    const select = el("select", { "aria-label": `Size of ${model.label}` },
      options.map((option) => el("option", {
        value: option.filename,
        selected: option.filename === model.recommendedFile,
        text: `${option.quant}, ${gb(option.sizeGB)}`,
      })));
    const button = el("button", {
      class: "btn primary",
      type: "button",
      text: "Choose",
      onclick: () => {
        const option = options.find((entry) => entry.filename === select.value) ?? options[0];
        choose({
          repo: model.hfRepo,
          file: option.filename,
          label: model.label,
          sizeGB: option.sizeGB,
          quant: option.quant,
          memoryGB: option.estimate?.totalGB,
          license: model.license,
          id: model.id,
          fit: option.fit,
        });
      },
    });
    return el("li", {},
      el("div", {},
        el("div", { class: "row-title" }, model.label, fitBadge(model.fit)),
        el("div", { class: "row-meta", text: [
          model.paramsB ? `${model.paramsB}B parameters` : null,
          model.estimate?.totalGB ? `needs about ${gb(model.estimate.totalGB)} while running` : null,
          model.tasks?.length ? `good for ${model.tasks.slice(0, 3).join(", ")}` : null,
        ].filter(Boolean).join(", ") }),
      ),
      el("div", { class: "row-actions" }, options.length > 1 ? select : null, button),
    );
  }));
}

for (const chip of $$(".filters .chip")) {
  chip.addEventListener("click", () => {
    for (const other of $$(".filters .chip")) other.setAttribute("aria-pressed", String(other === chip));
    state.task = chip.dataset.task;
    loadModels();
  });
}

async function loadLocal() {
  const { active, catalog, proOffer } = await api("/api/active");
  state.active = active;
  state.local = catalog ?? [];
  if (proOffer) renderPro(proOffer);
  renderLocal();
  return active;
}

function renderLocal() {
  const wrap = $("#local-wrap");
  wrap.hidden = state.local.length === 0;
  $("#local-list").replaceChildren(...state.local.map((entry) => {
    const isActive = state.active?.id === entry.id;
    return el("li", {},
      el("div", {},
        el("div", { class: "row-title" }, entry.id, isActive ? el("span", { class: "tag", text: "active" }) : null),
        el("div", { class: "row-meta", text: [entry.quant, entry.hfRepo].filter(Boolean).join(", ") }),
      ),
      el("div", { class: "row-actions" },
        el("button", { class: "btn primary", type: "button", text: isActive ? "Open chat" : "Use", onclick: () => useLocal(entry) }),
        el("button", { class: "btn quiet danger", type: "button", text: "Delete", onclick: () => removeLocal(entry) }),
      ),
    );
  }));
}

async function useLocal(entry) {
  try {
    const { active } = await api("/api/activate", { method: "POST", body: JSON.stringify({ id: entry.id, runtimeProvider: state.engine }) });
    state.active = active;
    markDone("model", entry.id);
    markDone("download", "Ready on disk");
    show("chat");
  } catch (error) {
    alertLine(`Could not use ${entry.id}: ${error.message}`);
  }
}

async function removeLocal(entry) {
  if (!window.confirm(`Delete ${entry.id} from this computer? The file is removed from disk.`)) return;
  try {
    await api("/api/models/delete", { method: "POST", body: JSON.stringify({ id: entry.id }) });
    await loadLocal();
  } catch (error) {
    alertLine(`Could not delete ${entry.id}: ${error.message}`);
  }
}

function alertLine(message) {
  const panel = $(`#panel-${state.step}`);
  let line = $(".error.inline", panel);
  if (!line) {
    line = el("p", { class: "error inline", role: "alert" });
    panel.append(line);
  }
  line.textContent = message;
}

// Hugging Face search
$("#hub-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("#hub-q").value.trim();
  const list = $("#hub-list");
  list.replaceChildren(el("li", { class: "empty-line", text: "Searching Hugging Face..." }));
  try {
    const params = new URLSearchParams({ q: query, limit: "12", chatOnly: "true" });
    const { models } = await api(`/api/hf/search?${params}`);
    if (!models.length) {
      list.replaceChildren(el("li", { class: "empty-line", text: `No GGUF model found for "${query}".` }));
      return;
    }
    list.replaceChildren(...models.map((model) => el("li", {},
      el("div", {},
        el("div", { class: "row-title" }, model.repo, fitBadge(model.compatibility?.fit), model.verified ? el("span", { class: "tag", text: "tested" }) : null),
        el("div", { class: "row-meta", text: [
          model.paramsB ? `${model.paramsB}B parameters` : null,
          Number.isFinite(model.downloads) ? `${model.downloads.toLocaleString()} downloads` : null,
        ].filter(Boolean).join(", ") }),
      ),
      el("div", { class: "row-actions" }, el("button", { class: "btn", type: "button", text: "See files", onclick: (event) => showFiles(model, event.currentTarget.closest("li")) })),
    )));
  } catch (error) {
    list.replaceChildren(el("li", { class: "empty-line", text: `Search failed: ${error.message}` }));
  }
});

async function showFiles(model, row) {
  $(".files", row)?.remove();
  const box = el("div", { class: "files" }, el("p", { class: "row-meta", text: "Reading the file list..." }));
  row.append(box);
  try {
    const detail = await api(`/api/hf/model?repo=${encodeURIComponent(model.repo)}`);
    const files = detail.ggufFiles ?? [];
    if (!files.length) {
      box.replaceChildren(el("p", { class: "row-meta", text: "This repository has no single-file GGUF download." }));
      return;
    }
    const note = detail.nativeCompat && !detail.nativeCompat.ready
      ? el("p", { class: "row-note", text: `The native engines may not support this model yet: ${detail.nativeCompat.reason ?? "unsupported architecture"}.` })
      : null;
    box.replaceChildren(note, ...files.slice(0, 8).map((file) => el("div", { class: "file" },
      el("span", {}, `${file.quant ?? file.filename}, ${gb(file.sizeGB)}`, " ", fitBadge(file.compatibility?.fit)),
      el("button", { class: "btn small", type: "button", text: "Choose", onclick: () => choose({
        repo: model.repo,
        file: file.filename,
        label: model.repo.split("/").pop(),
        sizeGB: file.sizeGB,
        quant: file.quant,
        memoryGB: file.compatibility?.estimate?.totalGB,
        license: detail.license,
        fit: file.compatibility?.fit,
      }) }),
    )));
  } catch (error) {
    box.replaceChildren(el("p", { class: "row-meta", text: `Could not read the files: ${error.message}` }));
  }
}

function choose(pick) {
  state.pick = pick;
  markDone("model", `${pick.label}, ${pick.quant ?? ""}`.trim());
  $("#picked").replaceChildren(
    el("div", { class: "row-title" }, pick.label, fitBadge(pick.fit)),
    el("div", { class: "row-meta", text: [
      pick.quant,
      pick.sizeGB ? `${gb(pick.sizeGB)} download` : null,
      pick.memoryGB ? `about ${gb(pick.memoryGB)} while running` : null,
      pick.license ? `licence: ${pick.license}` : null,
    ].filter(Boolean).join(", ") }),
    el("div", { class: "row-meta", text: `huggingface.co/${pick.repo}` }),
  );
  $("#dl-error").hidden = true;
  $("#dl-progress").hidden = true;
  $("#dl-start").disabled = false;
  $("#dl-start").textContent = "Download and load";
  show("download");
}

// ---------- 3. download ----------
for (const radio of $$('input[name="engine"]')) {
  radio.addEventListener("change", () => { state.engine = radio.value; $("#chat-engine").value = radio.value; });
}
$("#dl-back").addEventListener("click", () => show("model"));

$("#dl-start").addEventListener("click", () => {
  const pick = state.pick;
  if (!pick) return;
  const params = new URLSearchParams({ repo: pick.repo, file: pick.file });
  if (pick.id) params.set("id", pick.id);
  if (pick.license) params.set("license", pick.license);
  const source = new EventSource(`/api/download?${params}`);
  const startedAt = performance.now();
  $("#dl-start").disabled = true;
  $("#dl-error").hidden = true;
  $("#dl-progress").hidden = false;
  $("#dl-status").textContent = "Connecting to Hugging Face";
  $("#dl-meta").textContent = "";

  source.addEventListener("phase", (event) => {
    const phase = JSON.parse(event.data);
    $("#dl-status").textContent = phase.message;
  });
  source.addEventListener("progress", (event) => {
    const { downloadedBytes, totalBytes, ratio } = JSON.parse(event.data);
    const percent = ratio ? Math.min(100, ratio * 100) : 0;
    $("#dl-bar").style.width = `${percent}%`;
    $(".bar").setAttribute("aria-valuenow", String(Math.round(percent)));
    const seconds = (performance.now() - startedAt) / 1000;
    const speed = downloadedBytes / Math.max(seconds, 0.1);
    const remaining = totalBytes ? (totalBytes - downloadedBytes) / Math.max(speed, 1) : NaN;
    $("#dl-status").textContent = `${Math.floor(percent)}%, ${bytes(downloadedBytes)} of ${bytes(totalBytes)}`;
    $("#dl-meta").textContent = [speed > 0 ? `${bytes(speed)}/s` : "", remaining ? `${duration(remaining)} left` : ""].filter(Boolean).join(", ");
  });
  source.addEventListener("done", async (event) => {
    source.close();
    const { record } = JSON.parse(event.data);
    $("#dl-bar").style.width = "100%";
    $("#dl-status").textContent = "Verified. Activating";
    $("#dl-meta").textContent = "";
    try {
      const { active } = await api("/api/activate", { method: "POST", body: JSON.stringify({ id: record.id, runtimeProvider: state.engine }) });
      state.active = active;
      markDone("download", "Verified and saved");
      await loadLocal();
      show("chat");
    } catch (error) {
      failDownload(`Downloaded, but activation failed: ${error.message}`);
    }
  });
  // "error" fires for our own error event and for a dropped connection.
  source.addEventListener("error", (event) => {
    source.close();
    let message = "The connection to the app was lost during the download.";
    try { message = JSON.parse(event.data).message ?? message; } catch {}
    failDownload(message);
  });
});

function failDownload(message) {
  $("#dl-error").textContent = `${message} Check your connection and free disk space, then try again.`;
  $("#dl-error").hidden = false;
  $("#dl-start").disabled = false;
  $("#dl-start").textContent = "Try again";
}

// ---------- 4. chat ----------
function enterChat() {
  if (!state.active) return;
  $("#chat-model").textContent = `${state.active.id}${state.active.quant ? `, ${state.active.quant}` : ""}`;
  $("#chat-engine").value = state.active.runtimeProvider ?? state.engine;
  markDone("chat", "Running on this computer");
  if (state.warm === "cold" || state.warm === "error") warm();
}

function warm() {
  state.warmSource?.close();
  setHeat("warming", `Loading ${state.active?.id ?? "model"}`);
  const startedAt = performance.now();
  const source = new EventSource("/api/warm-model");
  state.warmSource = source;
  source.addEventListener("warming", () => setHeat("warming", `Loading ${state.active.id} into memory`));
  source.addEventListener("verifying", () => setHeat("warming", "Checking the first answer"));
  source.addEventListener("ready", (event) => {
    const data = JSON.parse(event.data);
    const seconds = ((data.loadMs ?? performance.now() - startedAt) / 1000).toFixed(1);
    setHeat("warm", `${data.modelId} ready on ${data.runtimeProvider}, loaded in ${seconds} s`);
    renderApi(data.endpoint);
    $("#prompt").focus();
  });
  source.addEventListener("idle", () => setHeat("cold", "Unloaded. Send a message to load it again"));
  source.addEventListener("load-error", (event) => {
    const data = JSON.parse(event.data);
    source.close();
    setHeat("error", "Could not load the model");
    addMessage("bot", `The model did not load: ${data.message}. Try the other engine, or pick a smaller model.`);
  });
  // The stream stays open on purpose: it keeps the model warm while this page is open.
}

function renderApi(endpoint) {
  const address = endpoint?.selected ?? endpoint;
  const host = address?.host ?? "127.0.0.1";
  const port = address?.port ?? (state.active?.runtimeProvider === "caterpillar" ? 8014 : 8013);
  const base = `http://${host}:${port}`;
  $("#api-snippet").textContent = `curl ${base}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${state.active?.id ?? "active"}", "messages": [{"role": "user", "content": "Hello"}]}'

# OpenAI SDK:  base_url="${base}/v1", api_key="local"
# Ollama clients: ${base}/api/chat`;
}

$("#api-copy").addEventListener("click", async () => {
  await navigator.clipboard?.writeText($("#api-snippet").textContent);
  $("#api-copy").textContent = "Copied";
  setTimeout(() => { $("#api-copy").textContent = "Copy"; }, 1500);
});

function addMessage(role, text) {
  $("#chat-empty")?.remove();
  const node = el("div", { class: `msg ${role}` }, el("div", { class: "msg-text", text }));
  $("#log").append(node);
  $("#log").scrollTop = $("#log").scrollHeight;
  return node;
}

function send(text) {
  if (!text || state.busy) return;
  if (state.warm !== "warm") { warm(); return; }
  state.busy = true;
  $("#send").disabled = true;
  addMessage("user", text);
  const bot = addMessage("bot", "");
  bot.classList.add("pending");
  const out = $(".msg-text", bot);
  const params = new URLSearchParams({ prompt: text, session: state.session });
  const source = new EventSource(`/api/quick-chat?${params}`);
  const finish = () => {
    source.close();
    bot.classList.remove("pending");
    state.busy = false;
    $("#send").disabled = state.warm !== "warm";
    $("#prompt").focus();
  };
  source.addEventListener("delta", (event) => {
    out.textContent += JSON.parse(event.data).text ?? "";
    $("#log").scrollTop = $("#log").scrollHeight;
  });
  source.addEventListener("done", (event) => {
    const data = JSON.parse(event.data);
    if (!out.textContent && data.response) out.textContent = data.response;
    const meta = [
      Number.isFinite(data.decodeTokensPerSecond) ? `${data.decodeTokensPerSecond.toFixed(1)} tokens/s` : (Number.isFinite(data.userVisibleTokensPerSecond) ? `${data.userVisibleTokensPerSecond.toFixed(1)} tokens/s` : null),
      Number.isFinite(data.ttftMs) ? `first word after ${(data.ttftMs / 1000).toFixed(2)} s` : null,
      data.runtimeProvider,
    ].filter(Boolean).join(", ");
    if (meta) bot.append(el("div", { class: "msg-meta", text: meta }));
    finish();
  });
  source.addEventListener("error", (event) => {
    let message = "The answer stopped because the connection to the app was lost.";
    try { message = JSON.parse(event.data).message ?? message; } catch {}
    if (!out.textContent) out.textContent = message;
    finish();
  });
}

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("#prompt").value.trim();
  $("#prompt").value = "";
  autosize();
  send(text);
});
$("#prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
function autosize() {
  const area = $("#prompt");
  area.style.height = "auto";
  area.style.height = `${Math.min(area.scrollHeight, 200)}px`;
}
$("#prompt").addEventListener("input", autosize);
for (const starter of $$(".starter")) starter.addEventListener("click", () => send(starter.textContent));

$("#chat-clear").addEventListener("click", async () => {
  await api("/api/quick-chat/reset", { method: "POST", body: JSON.stringify({ session: state.session }) }).catch(() => {});
  state.session = crypto.randomUUID();
  $("#log").replaceChildren(el("div", { class: "empty", id: "chat-empty" }, el("p", { text: "New conversation. Ask anything." })));
});

$("#chat-engine").addEventListener("change", async (event) => {
  const engine = event.target.value;
  state.engine = engine;
  try {
    state.warmSource?.close();
    setHeat("warming", `Switching to ${engine}`);
    const { active } = await api("/api/activate", { method: "POST", body: JSON.stringify({ id: state.active.id, runtimeProvider: engine }) });
    state.active = active;
    warm();
  } catch (error) {
    setHeat("error", "Engine switch failed");
    addMessage("bot", `Could not switch to ${engine}: ${error.message}`);
  }
});

$("#unload").addEventListener("click", async () => {
  state.warmSource?.close();
  state.warmSource = null;
  try {
    await api("/api/deactivate", { method: "POST", body: "{}" });
  } catch (error) {
    addMessage("bot", `Unload failed: ${error.message}`);
    return;
  }
  state.active = null;
  setHeat("cold", "No model loaded");
  await loadLocal();
  show("model");
  loadModels();
});

// ---------- Pro note ----------
function renderPro(offer) {
  $("#pro-price").textContent = offer.price ?? "€5";
  $("#pro-cta-price").textContent = `${offer.price ?? "€5"} once`;
  $("#pro-list").replaceChildren(...(offer.includes ?? []).slice(0, 4).map((line) => el("li", { text: line })));
  if (offer.downloadUrl) $("#pro-link").href = offer.downloadUrl;
}

// ---------- Pro gem (three.js, loaded after the page is usable) ----------
function loadProGem() {
  const canvas = $("#pro-gem");
  if (!canvas) return;
  import("./pro-gem.js")
    .then(({ mountProGem }) => mountProGem(canvas, $("#pro-link")))
    .catch(() => {});
}
if ("requestIdleCallback" in window) requestIdleCallback(loadProGem, { timeout: 2000 });
else setTimeout(loadProGem, 800);

// ---------- start ----------
(async function start() {
  show("machine", { focus: false });
  const [active] = await Promise.all([loadLocal().catch(() => null), scan()]);
  if (active) {
    markDone("model", active.id);
    markDone("download", "Ready on disk");
    show("chat", { focus: false });
  } else if (location.hash === "#model") {
    show("model");
    loadModels();
  }
})();
