// Charts in chat, decided by the model rather than by a button.
//
// The user never asks Lizard for "a chart feature" - they ask a question that
// happens to want a picture ("compare the three quants", "show me how memory
// grows with context"). So there is no toggle anywhere in the UI. The turn
// pipeline runs a small chain, LangChain-shaped, over the SAME bound model the
// answer will come from:
//
//   gate      looksLikeChartRequest()   free, lexical, no model call
//   classify  buildClassifierPrompt()   one word out of the model: which chart
//   instruct  buildChartInstruction()   type-specific shape, injected as system
//   parse     extractChartSpec()        pull the fenced JSON back out
//   repair    normalizeChartSpec()      coerce what a 3B model actually emits
//
// The gate exists because the classify step costs a model call, and most
// messages are "hi". The gate is deliberately generous (it costs nothing to be
// wrong here) and the model makes the real decision - which is what "let the
// agent decide" has to mean if it is to mean anything.
//
// Everything below is pure: no I/O, no runtime, no config. That is what makes
// the awkward part - what a 3-billion-parameter model emits when asked for
// JSON - testable without a GPU.

/** The fence language. The client cuts this out of the live stream, so
 *  src/ui/chat.js carries the same literal; tests assert they agree. */
export const CHART_FENCE = "lizard-chart";

/** Chart kinds the renderer implements, and the only words the classifier may
 *  return. Kept short on purpose: every extra option costs a small model
 *  accuracy, and six shapes cover essentially every request a chat produces. */
export const CHART_TYPES = Object.freeze(["bar", "line", "area", "pie", "scatter", "flow"]);

// Node-link ("flow") carries a different payload from the five value charts.
const FLOW_TYPE = "flow";

/** Words a model reaches for instead of the canonical name. Mapping these is
 *  cheaper and far more reliable than trying to forbid them in the prompt. */
const TYPE_SYNONYMS = Object.freeze({
  column: "bar", columns: "bar", histogram: "bar", bars: "bar", barchart: "bar",
  linechart: "line", trend: "line", timeseries: "line", timeline: "line", lines: "line",
  areachart: "area", stacked: "area", filled: "area",
  donut: "pie", doughnut: "pie", piechart: "pie", proportion: "pie", share: "pie",
  bubble: "scatter", points: "scatter", scatterplot: "scatter", correlation: "scatter",
  flowchart: "flow", network: "flow", graph: "flow", tree: "flow", diagram: "flow",
  sequence: "flow", architecture: "flow", pipeline: "flow", nodes: "flow",
});

// The gate. Two families: the user names a visual ("bar chart", "diagram"), or
// they use a verb that implies one ("plot", "visualize", "compare ... over").
const CHART_NOUNS = /\b(chart|charts|graph|graphs|diagram|diagrams|plot|plots|histogram|piechart|flowchart|timeline|visuali[sz]ation|infographic)\b/i;
const CHART_VERBS = /\b(plot|graph|chart|visuali[sz]e|diagram|plot out|draw|plotted|sketch)\b/i;
// "show me" alone is far too common to trigger on; it needs a shape word near it.
const SHAPE_WORDS = /\b(bar|line|area|pie|donut|doughnut|scatter|bubble|flow|network|tree|column|stacked|trend)\b/i;
const SHOW_VERBS = /\b(show|display|illustrate|compare|breakdown|break down|distribution|over time|versus|vs\.?)\b/i;

/**
 * Free pre-filter: could this message plausibly want a picture?
 *
 * Wrong in the permissive direction on purpose - a false positive costs one
 * short model call that answers "none", while a false negative silently
 * removes the whole feature for that message. What it must not do is fire on
 * ordinary conversation, because then every "thanks" pays for a model call.
 */
export function looksLikeChartRequest(text) {
  const message = String(text ?? "");
  if (!message.trim()) {
    return { asked: false, reason: "empty" };
  }
  // A code request that merely mentions d3/chart libraries is asking for code,
  // not for a rendered chart. Checked first: it overrides everything below.
  if (/\b(write|generate|give me|show me)\b[^.?!]*\b(code|snippet|function|script|component)\b/i.test(message)) {
    return { asked: false, reason: "asks for code" };
  }
  if (CHART_NOUNS.test(message)) {
    return { asked: true, reason: "names a chart" };
  }
  if (CHART_VERBS.test(message)) {
    return { asked: true, reason: "asks to draw" };
  }
  if (SHAPE_WORDS.test(message) && SHOW_VERBS.test(message)) {
    return { asked: true, reason: "describes a shape to show" };
  }
  return { asked: false, reason: "no visual language" };
}

// Which chart the wording points at, used to pick the exemplar the model is
// shown. It is a SUGGESTION, not the decision: the model emits the "type" field
// and is told it may change it, so a request the words describe badly still
// gets the right chart.
//
// This exists because of measurement, not preference. Asking the model first -
// the obvious design, and one small extra call - costs 15-22 s on this machine:
// llama-3.2-3B runs at ~5 tok/s here with ~140 ms/token prefill, so the floor
// for ANY call is 4.3 s and the 563-character classifier prompt measured
// 21.6 s. That is a fifth of a minute of silence before the answer starts, on
// exactly the turns a user is most likely to be waiting on. Set
// LIZARD_CHART_PRECHECK=1 to spend it anyway (see decideChartType in
// src/ui/chat-routes.js); on a fast engine it is cheap and slightly sharper.
const TYPE_HINTS = Object.freeze([
  [/\b(pie|donut|doughnut)\b|\bproportion|\bshare of\b|\bsplit\b|%/i, "pie"],
  [/\b(flow|flowchart|diagram|architecture|pipeline|process|steps?|network|tree|sequence)\b/i, "flow"],
  [/\b(scatter|bubble|correlat\w*)\b|\bagainst\b|\bvs\.?\b|\bversus\b/i, "scatter"],
  [/\b(area|cumulative|stacked|total over)\b/i, "area"],
  [/\b(line|trend|timeline|time series)\b|\bover (?:the )?(?:last |past )?\w*\s*(?:time|days?|weeks?|months?|years?|quarters?)\b/i, "line"],
  [/\b(bar|column|histogram|compare|comparison|each|per)\b/i, "bar"],
]);

/**
 * The chart the wording suggests. Falls back to bar, which is the right answer
 * more often than any other single choice for "show me these numbers".
 */
export function suggestChartType(text) {
  const message = String(text ?? "");
  for (const [pattern, type] of TYPE_HINTS) {
    if (pattern.test(message)) {
      return type;
    }
  }
  return "bar";
}

/**
 * The optional pre-check: one model call, one word back. Off by default because
 * of the cost measured above; kept because on a fast engine it is the better
 * decision and it is the same model either way.
 *
 * Small models follow a completion cue ("Answer:") far more reliably than an
 * instruction to be terse, so the prompt ends in one. The list is repeated
 * inline because a 3B model does not reliably carry a list from the system
 * turn into a one-word answer.
 */
export function buildClassifierPrompt(question) {
  const asked = String(question ?? "").replace(/\s+/g, " ").trim().slice(0, 600);
  return {
    systemPrompt:
      "You classify requests. You reply with exactly one word and never explain.",
    prompt: [
      "Which single visual would best answer the request below?",
      "",
      `Reply with ONE word from this list: ${CHART_TYPES.join(", ")}, none`,
      "",
      "bar = compare amounts across categories",
      "line = a value changing over time or a sequence",
      "area = a total changing over time",
      "pie = parts of one whole",
      "scatter = relationship between two numbers",
      "flow = steps, boxes and arrows, an architecture or a process",
      "none = the request does not want a picture at all",
      "",
      `Request: ${asked}`,
      "Answer:",
    ].join("\n"),
  };
}

/**
 * Reads the classifier's reply. Accepts the word anywhere in the output and
 * maps the synonyms models reach for, because a model that answers
 * "A bar chart would work best" has classified correctly and refusing that
 * answer on formatting grounds would be pedantry with a user-visible cost.
 */
export function parseClassifierReply(text) {
  const reply = String(text ?? "").toLowerCase();
  if (!reply.trim()) {
    return null;
  }
  // Whole-word scan in reading order, so "not a pie, use a bar" resolves the
  // way the sentence does rather than by list order.
  const tokens = reply.match(/[a-z]+/g) ?? [];
  for (const token of tokens) {
    if (token === "none" || token === "no" || token === "nothing") {
      return null;
    }
    if (CHART_TYPES.includes(token)) {
      return token;
    }
    if (TYPE_SYNONYMS[token]) {
      return TYPE_SYNONYMS[token];
    }
  }
  return null;
}

// One worked example per type. A small model copies the shape it is shown far
// more faithfully than it follows a description of a shape, so each type gets
// its own exemplar rather than a generic schema with options.
const EXEMPLARS = Object.freeze({
  bar: '{"type":"bar","title":"Tokens per second by engine","x":"Engine","y":"tok/s","data":[{"label":"lizard-native","value":38.4},{"label":"caterpillar","value":12.9}]}',
  line: '{"type":"line","title":"Memory used as context grows","x":"Context tokens","y":"GB","data":[{"label":"1024","value":0.4},{"label":"2048","value":0.7},{"label":"4096","value":1.4}]}',
  area: '{"type":"area","title":"Cumulative downloads","x":"Month","y":"Downloads","data":[{"label":"Jan","value":120},{"label":"Feb","value":260},{"label":"Mar","value":430}]}',
  pie: '{"type":"pie","title":"Where the turn time goes","data":[{"label":"Prefill","value":62},{"label":"Decode","value":31},{"label":"Overhead","value":7}]}',
  scatter: '{"type":"scatter","title":"Size against speed","x":"Parameters (B)","y":"tok/s","data":[{"label":"1B","x":1,"y":61},{"label":"3B","x":3,"y":38},{"label":"14B","x":14,"y":9}]}',
  flow: '{"type":"flow","title":"How a turn is answered","nodes":["Prompt","Retrieval","Model","Answer"],"links":[{"from":"Prompt","to":"Retrieval"},{"from":"Retrieval","to":"Model"},{"from":"Model","to":"Answer"}]}',
});

/**
 * The instruct step: appended to the system block for this turn only.
 *
 * Deliberately asks for prose FIRST and the block LAST. Two reasons: the user
 * gets a readable answer even if the JSON is malformed, and the client can cut
 * a partial fence off the end of the live stream without ever hiding prose.
 */
export function buildChartInstruction(chartType) {
  const type = CHART_TYPES.includes(chartType) ? chartType : "bar";
  const example = EXEMPLARS[type];
  const shapeRule = type === FLOW_TYPE
    ? '"nodes" is a list of 2-10 short labels and "links" connects them by those exact labels.'
    : type === "scatter"
      ? '"data" holds 2-40 points, each with a plain number "x" and "y".'
      : '"data" holds 2-12 points, each with a short "label" and a plain number "value".';
  return [
    // "One or two sentences ABOUT THE DATA" is load-bearing. Asked only to
    // "answer in prose first", llama-3.2-3B pads with a paragraph about what
    // Lizard is before getting to the chart - it has nothing else to say about
    // a bare list of numbers, so it fills the space it was given.
    `Answer in one or two sentences about the data itself - what it shows, what stands out. Then, as the very last thing in your reply, include a ${type} chart in a fenced block exactly like this:`,
    "",
    "```" + CHART_FENCE,
    example,
    "```",
    "",
    "Chart rules:",
    `- ${shapeRule}`,
    "- Valid JSON on a single line. No comments, no trailing commas, no units inside numbers.",
    "- Use figures from this conversation when it has them; otherwise use your own knowledge.",
    "- If you do not have real figures, write the answer without the block rather than inventing data.",
    "- Do not describe yourself, and do not explain what a chart is.",
    // The model owns the decision, not the word-match that picked the exemplar.
    // Both escapes matter: the wrong shape suggested, and no chart wanted at all.
    `- ${type} is a suggestion. If another shape suits the question better, change "type" to one of: ${CHART_TYPES.join(", ")}.`,
    "- If the question does not actually want a picture, leave the block out entirely and just answer.",
  ].join("\n");
}

// ---- reading the answer back ----------------------------------------------

/**
 * Pulls the chart block out of an answer and returns the prose without it.
 *
 * Handles three things models actually do: the requested fence; a bare ```json
 * fence (it forgot the language); and no fence at all, just the object. The
 * last case is why this scans braces rather than running a regex - a regex for
 * a JSON object is wrong the moment a label contains a brace.
 */
export function extractChartSpec(answer) {
  const text = String(answer ?? "");
  if (!text.trim()) {
    return { text, raw: null };
  }

  // `source` tells the caller how sure we are that this text is ours.
  // "fence" is our own private syntax and is never prose, whatever it contains;
  // the other two are guesses about ordinary JSON and must be treated gently.
  // That distinction is what decides whether a block that failed to parse is
  // still removed from the answer - see the chart lift in chat-routes.js.
  const fence = findFencedBlock(text, CHART_FENCE);
  if (fence) {
    return { text: cut(text, fence.start, fence.end), raw: fence.body, source: "fence" };
  }

  // The model used a generic fence. Only treat it as a chart if it actually
  // looks like one, so an answer that legitimately shows JSON is left alone.
  const generic = findFencedBlock(text, "json");
  if (generic && looksLikeSpec(generic.body)) {
    return { text: cut(text, generic.start, generic.end), raw: generic.body, source: "json" };
  }

  // No fence at all. Take the last balanced object that looks like a spec.
  const bare = findLastSpecObject(text);
  if (bare) {
    return { text: cut(text, bare.start, bare.end), raw: bare.body, source: "bare" };
  }
  return { text, raw: null, source: null };
}

function cut(text, start, end) {
  return `${text.slice(0, start)}${text.slice(end)}`.replace(/\n{3,}/g, "\n\n").trim();
}

function looksLikeSpec(body) {
  return /"type"\s*:/.test(body) && /"(data|nodes|links|series|points|values|items)"\s*:/.test(body);
}

function findFencedBlock(text, language) {
  // The newline after the language is optional. Small models routinely open
  // the fence and keep writing on the same line ("```lizard-chart {...}"), and
  // insisting on the newline lost the chart and left the raw JSON in the
  // transcript - the one failure mode this whole module exists to avoid.
  const opener = new RegExp("```[ \\t]*" + language + "[ \\t]*(?:\\r?\\n)?", "i");
  const match = opener.exec(text);
  if (!match) {
    return null;
  }
  const bodyStart = match.index + match[0].length;
  const closer = text.indexOf("```", bodyStart);
  // An unterminated fence is the streaming case: the answer was cut off. Take
  // what is there - the repair step decides whether it is usable.
  const bodyEnd = closer === -1 ? text.length : closer;
  const end = closer === -1 ? text.length : closer + 3;
  return { start: match.index, end, body: text.slice(bodyStart, bodyEnd) };
}

function findLastSpecObject(text) {
  let found = null;
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "{") continue;
    const end = matchBrace(text, index);
    if (end === -1) continue;
    const body = text.slice(index, end + 1);
    if (looksLikeSpec(body)) {
      found = { start: index, end: end + 1, body };
    }
    index = end;
  }
  return found;
}

function matchBrace(text, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const character = text[index];
    if (inString) {
      if (escaped) { escaped = false; continue; }
      if (character === "\\") { escaped = true; continue; }
      if (character === '"') { inString = false; }
      continue;
    }
    if (character === '"') { inString = true; continue; }
    if (character === "{") { depth += 1; continue; }
    if (character === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

// ---- repairing what the model emitted -------------------------------------

const MAX_VALUE_POINTS = 40;
const MAX_FLOW_NODES = 14;
const MAX_LABEL_CHARS = 48;

// Keys models substitute for the ones asked for. Accepting them is the
// difference between a chart appearing and a 3B model "failing" for writing
// "name" instead of "label".
// "from"/"source" are last-resort label keys. When the model overrules the
// suggested type it often keeps the exemplar's vocabulary, producing a value
// chart whose points are keyed like links:
//   {"type":"line","data":[{"from":"Q1","value":12},{"from":"Q2","value":15}]}
// Every label and every number is right there; only the key name came from the
// wrong example. They go last so a real "label" always wins.
const LABEL_KEYS = ["label", "name", "category", "key", "x", "id", "title", "step", "from", "source"];
const VALUE_KEYS = ["value", "y", "count", "amount", "total", "score", "n", "size"];
const DATA_KEYS = ["data", "series", "points", "values", "items", "rows", "dataset"];

/**
 * Turns whatever came back into a spec the renderer can draw, or null.
 *
 * Returning null is a normal outcome, not an error: the user still has their
 * prose answer. The bar for drawing is that the chart would tell the truth -
 * two real points minimum, finite numbers, labels that exist.
 */
export function normalizeChartSpec(raw, { requestedType = null } = {}) {
  const parsed = typeof raw === "string" ? parseLenientJson(raw) : raw;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return null;
  }

  const type = resolveType(parsed.type ?? parsed.chart ?? parsed.kind, requestedType);
  if (!type) {
    return null;
  }
  const title = cleanText(parsed.title ?? parsed.name ?? "", 120);

  // Try what the model declared, then the other family. A model that overrules
  // the suggested type frequently keeps the exemplar's payload shape, so the
  // declared type and the data it supplied disagree. Whichever one is actually
  // complete is the one that can be drawn honestly; if neither is, nothing is.
  if (type === FLOW_TYPE) {
    return normalizeFlow(parsed, title) ?? normalizeValueChart(parsed, "bar", title);
  }
  return normalizeValueChart(parsed, type, title) ?? normalizeFlow(parsed, title);
}

function resolveType(value, requestedType) {
  const token = String(value ?? "").toLowerCase().replace(/[^a-z]/g, "");
  if (CHART_TYPES.includes(token)) return token;
  if (TYPE_SYNONYMS[token]) return TYPE_SYNONYMS[token];
  // The model omitted or mangled the type but we asked for a specific one.
  if (requestedType && CHART_TYPES.includes(requestedType)) return requestedType;
  return null;
}

function normalizeValueChart(parsed, type, title) {
  const rows = collectRows(parsed);
  if (rows.length === 0) {
    return null;
  }

  const points = [];
  for (const row of rows) {
    if (points.length >= MAX_VALUE_POINTS) break;
    if (type === "scatter") {
      const x = toNumber(pick(row, ["x", ...VALUE_KEYS]));
      const y = toNumber(pick(row, ["y", "value", "count", "amount", "total"]));
      if (x == null || y == null) continue;
      points.push({ label: cleanText(pick(row, ["label", "name", "id"]) ?? "", MAX_LABEL_CHARS), x, y });
      continue;
    }
    const label = cleanText(pick(row, LABEL_KEYS) ?? "", MAX_LABEL_CHARS);
    const value = toNumber(pick(row, VALUE_KEYS));
    if (!label || value == null) continue;
    points.push({ label, value });
  }

  // One point is not a chart, it is a number with extra steps.
  if (points.length < 2) {
    return null;
  }
  // A pie of negative or all-zero slices cannot be drawn honestly.
  if (type === "pie") {
    const usable = points.filter((point) => point.value > 0);
    if (usable.length < 2) return null;
    return { type, title, data: usable };
  }

  // Every value identical - a flat line, usually at zero. There is nothing to
  // see in it, and it is the fingerprint of a model that fell into a repetition
  // loop: llama-3.2-3B asked for a frequency chart produced fifty-odd points of
  // {"label":"N Hz","value":0} and ran out of token budget mid-object. Refusing
  // it costs a chart nobody could have read and keeps the prose answer intact.
  if (type !== "scatter" && points.every((point) => point.value === points[0].value)) {
    return null;
  }
  return {
    type,
    title,
    xLabel: cleanText(parsed.x ?? parsed.xLabel ?? parsed.xAxis ?? "", 60),
    yLabel: cleanText(parsed.y ?? parsed.yLabel ?? parsed.yAxis ?? "", 60),
    data: points,
  };
}

function collectRows(parsed) {
  let data = null;
  for (const key of DATA_KEYS) {
    if (parsed[key] != null) { data = parsed[key]; break; }
  }
  if (data == null) return [];
  if (Array.isArray(data)) {
    // Also accept [["A", 1], ["B", 2]] and a plain [1, 2, 3].
    return data.map((entry, index) => {
      if (Array.isArray(entry)) return { label: entry[0], value: entry[1] };
      if (entry != null && typeof entry === "object") return entry;
      return { label: `#${index + 1}`, value: entry };
    });
  }
  if (typeof data === "object") {
    // {"Jan": 12, "Feb": 19} - a shape small models produce constantly.
    return Object.entries(data).map(([label, value]) => ({ label, value }));
  }
  return [];
}

function normalizeFlow(parsed, title) {
  const rawNodes = Array.isArray(parsed.nodes) ? parsed.nodes : [];
  const rawLinks = Array.isArray(parsed.links)
    ? parsed.links
    : Array.isArray(parsed.edges) ? parsed.edges : [];

  const nodes = [];
  const seen = new Set();
  const addNode = (value) => {
    const label = cleanText(
      value != null && typeof value === "object" ? pick(value, ["label", "name", "id", "title"]) : value,
      MAX_LABEL_CHARS,
    );
    if (!label || seen.has(label) || nodes.length >= MAX_FLOW_NODES) return;
    seen.add(label);
    nodes.push({ id: label, label });
  };
  for (const node of rawNodes) addNode(node);

  const links = [];
  for (const link of rawLinks) {
    if (!link || typeof link !== "object") continue;
    const from = cleanText(pick(link, ["from", "source", "start", "a"]), MAX_LABEL_CHARS);
    const to = cleanText(pick(link, ["to", "target", "end", "b"]), MAX_LABEL_CHARS);
    if (!from || !to || from === to) continue;
    // A link naming a node the list forgot is a mistake worth fixing rather
    // than a reason to drop the diagram - models list edges more reliably
    // than they list every node.
    addNode(from);
    addNode(to);
    if (!seen.has(from) || !seen.has(to)) continue;
    links.push({ from, to, label: cleanText(pick(link, ["label", "text", "name"]) ?? "", 32) });
  }

  if (nodes.length < 2 || links.length === 0) {
    return null;
  }
  return { type: FLOW_TYPE, title, nodes, links };
}

function pick(row, keys) {
  if (!row || typeof row !== "object") return null;
  for (const key of keys) {
    if (row[key] != null && row[key] !== "") return row[key];
  }
  // Case-insensitive second pass: "Label" and "Value" are extremely common.
  const lowered = new Map(Object.keys(row).map((key) => [key.toLowerCase(), key]));
  for (const key of keys) {
    const actual = lowered.get(key);
    if (actual && row[actual] != null && row[actual] !== "") return row[actual];
  }
  return null;
}

function cleanText(value, limit) {
  return String(value ?? "").replace(/\s+/g, " ").trim().slice(0, limit);
}

/**
 * Numbers as models write them: "1,234", "45%", "$3.50", "12 GB", "3.4s".
 * Anything genuinely non-numeric returns null so the point is dropped rather
 * than silently charted as zero - a zero bar is a lie, a missing bar is not.
 */
export function toNumber(value) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value !== "string") {
    return null;
  }
  const text = value.trim();
  if (!text) return null;
  // Strip currency, thousands separators, percent and a trailing unit word.
  const stripped = text
    .replace(/[$€£¥]/g, "")
    .replace(/(\d),(?=\d{3}\b)/g, "$1")
    .replace(/%/g, "")
    .replace(/\s*[a-zA-Z/]{1,8}\.?$/, "")
    .trim();
  if (!/^-?\d*\.?\d+(?:[eE][-+]?\d+)?$/.test(stripped)) {
    return null;
  }
  const parsed = Number(stripped);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * JSON.parse first; then the specific malformations small models produce.
 * Deliberately NOT a general JSON5 implementation - each repair below is one
 * observed failure, and anything beyond them should fail so the chart is
 * dropped rather than drawn from a guess.
 */
export function parseLenientJson(text) {
  const source = String(text ?? "").trim();
  if (!source) return null;
  const direct = tryParse(source);
  if (direct) return direct;

  let repaired = source
    // Fences the extractor did not own (e.g. nested in a bare object).
    .replace(/^```[a-z-]*\s*/i, "")
    .replace(/```$/i, "")
    // Smart quotes from a model that was trained on prose.
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    // Trailing commas before a close.
    .replace(/,(\s*[}\]])/g, "$1")
    // // and /* */ comments.
    .replace(/\/\/[^\n\r]*/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .trim();
  const afterRepair = tryParse(repaired);
  if (afterRepair) return afterRepair;

  // Single-quoted keys and values. Applied last because it is the most
  // aggressive rewrite and would corrupt an apostrophe inside a label.
  if (!repaired.includes('"')) {
    const requoted = repaired.replace(/'/g, '"');
    const afterQuotes = tryParse(requoted);
    if (afterQuotes) return afterQuotes;
  }

  // Unquoted keys: {type: "bar"} -> {"type": "bar"}
  const quotedKeys = repaired.replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:/g, '$1"$2":');
  const afterKeys = tryParse(quotedKeys);
  if (afterKeys) return afterKeys;

  // Missing closers. Observed from llama-3.2-3B on the pie exemplar: it closed
  // the data array and stopped, one "}" short of valid. Every value was
  // present and correct, so refusing it would drop a good chart over a single
  // absent character. Safe because normalizeChartSpec still has to agree the
  // result is chartable - a wrongly-closed object simply fails that instead.
  return tryParse(closeUnbalanced(quotedKeys)) ?? tryParse(closeUnbalanced(repaired));
}

function closeUnbalanced(text) {
  const stack = [];
  let inString = false;
  let escaped = false;
  for (const character of text) {
    if (inString) {
      if (escaped) { escaped = false; continue; }
      if (character === "\\") { escaped = true; continue; }
      if (character === '"') { inString = false; }
      continue;
    }
    if (character === '"') { inString = true; continue; }
    if (character === "{" || character === "[") { stack.push(character); continue; }
    if (character === "}" || character === "]") { stack.pop(); }
  }
  if (inString || stack.length === 0) {
    // An unterminated string means the answer was cut mid-value; there is no
    // honest way to guess the rest of it.
    return inString ? "" : text;
  }
  let closed = text.replace(/,\s*$/, "");
  while (stack.length > 0) {
    closed += stack.pop() === "{" ? "}" : "]";
  }
  return closed;
}

function tryParse(text) {
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}
