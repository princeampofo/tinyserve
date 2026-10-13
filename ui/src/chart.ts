// A small single-series line chart in SVG, with a crosshair tooltip.

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 320;
const HEIGHT = 120;
const PAD = { top: 8, right: 8, bottom: 18, left: 40 };

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

export class LineChart {
  private svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img" });
  private grid = el("g", { class: "grid" });
  private line = el("path", { class: "series", fill: "none" });
  private hair = el("line", { class: "crosshair", y1: PAD.top, y2: HEIGHT - PAD.bottom, visibility: "hidden" });
  private dot = el("circle", { class: "series-dot", r: 4, visibility: "hidden" });
  private tooltip = document.createElement("div");
  private points: { time: number; value: number }[] = [];
  private x = (_: number) => 0;
  private y = (_: number) => 0;

  constructor(
    container: HTMLElement,
    label: string,
    private format: (value: number) => string,
    private fixedMax?: number,
    // Axis labels can need more precision than readouts: a 0-1 axis has a 0.5 tick.
    private tickFormat: (value: number) => string = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1)),
  ) {
    this.svg.setAttribute("aria-label", label);
    this.svg.append(this.grid, this.line, this.hair, this.dot);
    this.tooltip.className = "tooltip";
    this.tooltip.hidden = true;
    container.append(this.svg, this.tooltip);
    this.svg.addEventListener("pointermove", (e) => this.hover(e));
    this.svg.addEventListener("pointerleave", () => this.hide());
  }

  update(points: { time: number; value: number }[]): void {
    this.points = points;
    this.grid.replaceChildren();
    if (points.length < 2) {
      this.line.setAttribute("d", "");
      return;
    }
    const t0 = points[0].time;
    const t1 = points.at(-1)!.time;
    const max = this.fixedMax ?? niceMax(Math.max(...points.map((p) => p.value)));
    this.x = (t) => PAD.left + ((t - t0) / Math.max(1, t1 - t0)) * (WIDTH - PAD.left - PAD.right);
    this.y = (v) => HEIGHT - PAD.bottom - (v / max) * (HEIGHT - PAD.top - PAD.bottom);

    for (const fraction of [0, 0.5, 1]) {
      const yy = this.y(max * fraction);
      this.grid.append(el("line", { x1: PAD.left, x2: WIDTH - PAD.right, y1: yy, y2: yy, class: fraction === 0 ? "baseline" : "gridline" }));
      const text = el("text", { x: PAD.left - 6, y: yy + 3, "text-anchor": "end", class: "tick" });
      text.textContent = this.tickFormat(max * fraction);
      this.grid.append(text);
    }
    const ago = el("text", { x: PAD.left, y: HEIGHT - 4, class: "tick" });
    ago.textContent = `${Math.round((t1 - t0) / 1000)} s ago`;
    const now = el("text", { x: WIDTH - PAD.right, y: HEIGHT - 4, "text-anchor": "end", class: "tick" });
    now.textContent = "now";
    this.grid.append(ago, now);

    this.line.setAttribute("d", points.map((p, i) => `${i ? "L" : "M"}${this.x(p.time)},${this.y(p.value)}`).join(""));
  }

  private hover(event: PointerEvent): void {
    if (this.points.length < 2) return;
    const box = this.svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * WIDTH;
    // Snap to the nearest point in time: the reader aims at a moment, not at a 2px line.
    const nearest = this.points.reduce((a, b) => (Math.abs(this.x(b.time) - px) < Math.abs(this.x(a.time) - px) ? b : a));
    const x = this.x(nearest.time);
    const y = this.y(nearest.value);
    this.hair.setAttribute("x1", String(x));
    this.hair.setAttribute("x2", String(x));
    this.dot.setAttribute("cx", String(x));
    this.dot.setAttribute("cy", String(y));
    this.hair.setAttribute("visibility", "visible");
    this.dot.setAttribute("visibility", "visible");
    const value = document.createElement("strong");
    value.textContent = this.format(nearest.value);
    const when = document.createElement("span");
    when.textContent = new Date(nearest.time).toLocaleTimeString();
    this.tooltip.replaceChildren(value, when);
    this.tooltip.hidden = false;
    this.tooltip.style.left = `${(x / WIDTH) * 100}%`;
  }

  private hide(): void {
    this.hair.setAttribute("visibility", "hidden");
    this.dot.setAttribute("visibility", "hidden");
    this.tooltip.hidden = true;
  }
}

function niceMax(value: number): number {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const steps = [1, 2, 2.5, 5, 10];
  return magnitude * steps.find((s) => s * magnitude >= value)!;
}
