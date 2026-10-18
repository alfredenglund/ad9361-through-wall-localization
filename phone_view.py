"""Tiny web page with a live graph, so rx.py can be watched from a phone on the same WiFi."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Through-wall</title>
<style>
:root { --bg:#fff; --fg:#111; --muted:#666; --grid:#e5e5e5; --line:#2563eb; --line2:#16a34a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111; --fg:#eee; --muted:#999; --grid:#2a2a2a; --line:#60a5fa; --line2:#4ade80; }
}
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:16px -apple-system, system-ui, sans-serif; }
.label { color:var(--muted); font-size:14px; margin:20px 0 4px; }
.big { font-size:56px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1; }
canvas { width:100%; height:200px; display:block; }
#status { color:var(--muted); font-size:13px; margin-top:12px; }
#badge { font-size:40px; font-weight:800; text-align:center; padding:20px; border-radius:14px; color:#fff; background:#888; }
#badge.MOVING { background:#dc2626; } #badge.BREATHING { background:#ea8c0c; } #badge.EMPTY { background:#16a34a; }
</style>
</head><body>
<div id="badge">WAIT</div>
<div class="label">3 s spread (dashed: breathing / moving thresholds)</div>
<div class="big" id="now">-</div>
<canvas id="presence"></canvas>
<div class="label">ch1-ch0 phase (breathing)</div>
<canvas id="phase"></canvas>
<div id="status">connecting...</div>
<script>
const WINDOW = 60;  // seconds shown
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function draw(canvas, t, y, color, log, marks = []) {
  const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  if (!t.length) return;

  const tStart = t[t.length - 1] - WINDOW;
  const pts = [];
  const tf = v => log ? Math.log10(Math.max(v, 1e-6)) : v;
  for (let i = 0; i < t.length; i++)
    if (t[i] >= tStart && y[i] !== null) pts.push([t[i], tf(y[i])]);
  if (!pts.length) return;
  const all = pts.map(p => p[1]).concat(marks.map(m => tf(m.v)));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
  const X = s => (s - tStart) / WINDOW * w, Y = v => h - (v - lo) / (hi - lo) * h;

  g.strokeStyle = css("--grid"); g.fillStyle = css("--muted"); g.font = "11px system-ui"; g.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const v = lo + (hi - lo) * k / 3, yy = Y(v);
    g.beginPath(); g.moveTo(0, yy); g.lineTo(w, yy); g.stroke();
    g.fillText(log ? (10 ** v).toPrecision(2) : v.toFixed(2), 4, Math.max(yy - 3, 12));
  }
  g.setLineDash([6, 4]);
  for (const m of marks) { g.strokeStyle = m.color; g.beginPath(); g.moveTo(0, Y(tf(m.v))); g.lineTo(w, Y(tf(m.v))); g.stroke(); }
  g.setLineDash([]);
  g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
  pts.forEach(([s, v], i) => i ? g.lineTo(X(s), Y(v)) : g.moveTo(X(s), Y(v)));
  g.stroke();
}

async function tick() {
  try {
    const d = await (await fetch("/data", { cache: "no-store" })).json();
    draw(document.getElementById("presence"), d.t, d.spread, css("--line"), true, d.marks);
    draw(document.getElementById("phase"), d.t, d.phase, css("--line2"), false);
    const last = d.spread[d.spread.length - 1];
    document.getElementById("now").textContent = last == null ? "-" : last.toFixed(4);
    const badge = document.getElementById("badge");
    badge.textContent = d.bpm ? `${d.status} ${d.bpm}/min` : d.status; badge.className = d.status;
    document.getElementById("status").textContent = "live, updated " + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById("status").textContent = "no connection to rx.py";
  }
}
setInterval(tick, 1000);
tick();
</script>
</body></html>
"""


def local_ip():
    # Connecting a UDP socket sends nothing; it just makes the OS pick the WiFi interface
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def start(get_data, port=8000, page=PAGE, actions=None):
    """Serve a page in a background thread.

    get_data() returns a JSON-able dict for /data. actions maps extra paths (e.g. "/start")
    to functions that are called when the page requests them.
    """
    actions = actions or {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/data":
                body, ctype = json.dumps(get_data()).encode(), "application/json"
            elif path in actions:
                actions[path]()
                body, ctype = b'{"ok": true}', "application/json"
            else:
                body, ctype = page.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # keep the terminal for rx.py's own prints

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Phone view: http://{local_ip()}:{port}")
