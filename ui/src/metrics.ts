// Reads tinyserve's Prometheus /metrics text and turns two scrapes into rates.

export type Samples = Map<string, number>; // "name{labels}" -> value

export function parse(text: string): Samples {
  const samples: Samples = new Map();
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const space = line.lastIndexOf(" ");
    samples.set(line.slice(0, space), Number(line.slice(space + 1)));
  }
  return samples;
}

export interface Snapshot {
  time: number; // ms since epoch
  instance: string; // which server replica answered (the X-Instance header)
  samples: Samples;
}

export interface Point {
  time: number;
  tokensPerSecond: number;
  ttftP50: number | null; // seconds; null when no request started in the window
  kvUsed: number; // 0..1
  running: number;
  waiting: number;
}

// Counters only go up, so "how much happened between two scrapes" is a
// difference, which only means something between two scrapes of the same
// replica. A negative difference means that replica restarted: skip the point.
export function point(older: Snapshot, newer: Snapshot): Point | null {
  if (older.instance !== newer.instance) return null;
  const seconds = (newer.time - older.time) / 1000;
  const delta = (name: string) => (newer.samples.get(name) ?? 0) - (older.samples.get(name) ?? 0);
  const tokens = delta("generated_tokens_total");
  if (tokens < 0 || seconds <= 0) return null;
  return {
    time: newer.time,
    tokensPerSecond: tokens / seconds,
    ttftP50: histogramQuantile(0.5, "ttft_seconds", older.samples, newer.samples),
    kvUsed: newer.samples.get("kv_blocks_used_ratio") ?? 0,
    running: newer.samples.get("running_requests") ?? 0,
    waiting: newer.samples.get("waiting_requests") ?? 0,
  };
}

// The same estimate Prometheus's histogram_quantile makes: find the bucket the
// quantile falls in and interpolate linearly inside it.
function histogramQuantile(q: number, name: string, older: Samples, newer: Samples): number | null {
  const buckets: { le: number; count: number }[] = [];
  for (const [key, value] of newer) {
    const match = key.match(new RegExp(`^${name}_bucket\\{le="([^"]+)"\\}$`));
    if (!match) continue;
    buckets.push({ le: Number(match[1]), count: value - (older.get(key) ?? 0) });
  }
  buckets.sort((a, b) => a.le - b.le);
  const total = buckets.at(-1)?.count ?? 0;
  if (total <= 0) return null;
  const rank = q * total;
  let previous = { le: 0, count: 0 };
  for (const bucket of buckets) {
    if (bucket.count >= rank) {
      if (!Number.isFinite(bucket.le)) return previous.le; // can't see past the last finite bucket
      const inBucket = bucket.count - previous.count;
      const fraction = inBucket > 0 ? (rank - previous.count) / inBucket : 0;
      return previous.le + (bucket.le - previous.le) * fraction;
    }
    previous = bucket;
  }
  return null;
}
