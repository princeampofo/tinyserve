import { LineChart } from "./chart.js";
import { type Message, streamChat } from "./chat.js";
import { type Point, type Snapshot, parse, point } from "./metrics.js";

const POLL_MS = 2000;
const HISTORY = 90; // points kept: 3 minutes at one every 2 s
const WINDOW_MS = 10_000; // rates and TTFT are computed over the last 10 s

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------- live metrics

const charts = {
  throughput: new LineChart($("chart-throughput"), "Output tokens per second", (v) => v.toFixed(0)),
  ttft: new LineChart($("chart-ttft"), "Time to first token, p50", formatSeconds, undefined, formatSeconds),
  kv: new LineChart($("chart-kv"), "KV cache blocks used", percent, 1, percent),
};
// Recent scrapes per replica: with several pods behind one Service, each poll
// may reach a different one, and rates only make sense within one pod.
const snapshots = new Map<string, Snapshot[]>();
const points: Point[] = [];

function percent(v: number): string {
  return `${Math.round(v * 100)}%`;
}

function formatSeconds(s: number): string {
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`;
}

async function poll(): Promise<void> {
  try {
    const response = await fetch("/metrics");
    const instance = response.headers.get("X-Instance") ?? "server";
    const now = { time: Date.now(), instance, samples: parse(await response.text()) };
    const own = snapshots.get(instance) ?? [];
    own.push(now);
    while (own.length > 1 && now.time - own[1].time >= WINDOW_MS) own.shift();
    snapshots.set(instance, own);
    const p = point(own[0], now);
    if (p) {
      points.push(p);
      if (points.length > HISTORY) points.shift();
      render(p);
    }
    $("status").textContent = snapshots.size > 1 ? `live · ${snapshots.size} replicas seen` : "live";
  } catch {
    $("status").textContent = "server unreachable";
  }
}

function render(latest: Point): void {
  $("stat-throughput").textContent = latest.tokensPerSecond.toFixed(1);
  $("stat-ttft").textContent = latest.ttftP50 === null ? "–" : formatSeconds(latest.ttftP50);
  $("stat-running").textContent = `${latest.running} / ${latest.waiting}`;
  $("stat-kv").textContent = `${Math.round(latest.kvUsed * 100)}%`;
  charts.throughput.update(points.map((p) => ({ time: p.time, value: p.tokensPerSecond })));
  charts.ttft.update(points.filter((p) => p.ttftP50 !== null).map((p) => ({ time: p.time, value: p.ttftP50! })));
  charts.kv.update(points.map((p) => ({ time: p.time, value: p.kvUsed })));
}

// ---------------------------------------------------------------- chat

const history: Message[] = [];
let controller: AbortController | null = null;

function addBubble(role: "user" | "assistant", text: string): HTMLElement {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text; // model output is untrusted text, never HTML
  $("messages").append(bubble);
  bubble.scrollIntoView({ block: "end" });
  return bubble;
}

async function send(event: SubmitEvent): Promise<void> {
  event.preventDefault();
  const input = $<HTMLTextAreaElement>("prompt");
  const prompt = input.value.trim();
  if (!prompt || controller) return;
  input.value = "";
  history.push({ role: "user", content: prompt });
  addBubble("user", prompt);
  const bubble = addBubble("assistant", "");
  bubble.classList.add("streaming");

  controller = new AbortController();
  $<HTMLButtonElement>("send").hidden = true;
  $<HTMLButtonElement>("stop").hidden = false;
  const started = performance.now();
  let first: number | null = null;
  let tokens = 0;
  try {
    const reason = await streamChat(
      history,
      Number($<HTMLInputElement>("max-tokens").value),
      Number($<HTMLInputElement>("temperature").value),
      (delta) => {
        first ??= performance.now();
        tokens += 1;
        bubble.textContent += delta;
        bubble.scrollIntoView({ block: "end" });
      },
      controller.signal,
    );
    history.push({ role: "assistant", content: bubble.textContent ?? "" });
    const seconds = (performance.now() - (first ?? started)) / 1000;
    const ttft = first === null ? "–" : formatSeconds((first - started) / 1000);
    $("chat-meta").textContent = `first token ${ttft} · ${tokens} tokens · ${(tokens / Math.max(seconds, 1e-3)).toFixed(1)}/s · ${reason}`;
  } catch (error) {
    // Stopping closes the stream; the server notices and frees the request's KV cache.
    $("chat-meta").textContent = controller.signal.aborted ? "stopped" : `error: ${String(error)}`;
    history.pop();
  } finally {
    bubble.classList.remove("streaming");
    controller = null;
    $<HTMLButtonElement>("send").hidden = false;
    $<HTMLButtonElement>("stop").hidden = true;
  }
}

$<HTMLFormElement>("chat-form").addEventListener("submit", (e) => void send(e));
$<HTMLButtonElement>("stop").addEventListener("click", () => controller?.abort());
$<HTMLTextAreaElement>("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $<HTMLFormElement>("chat-form").requestSubmit();
  }
});

fetch("/v1/models")
  .then((r) => r.json())
  .then((m) => ($("model").textContent = m.data[0].id))
  .catch(() => undefined);
void poll();
setInterval(() => void poll(), POLL_MS);
