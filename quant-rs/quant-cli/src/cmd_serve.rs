//! `quant serve` — lightweight HTTP status server for k8s deployments.
//!
//! Serves health and status endpoints so the deployment can be probed
//! by Kubernetes liveness checks and accessed via Tailscale Ingress.

use std::io::{Read, Write};
use std::net::TcpListener;

use clap::Args;
use tracing::info;

use quant_data::MarketDataStore;

#[derive(Args)]
pub struct ServeArgs {
    /// Port to listen on.
    #[arg(long, default_value = "8000")]
    pub port: u16,

    /// Path to the DuckDB market data file (optional).
    #[arg(long)]
    pub db: Option<String>,

    /// Path to the Prometheus textfile written by `quant run once`.
    /// Contents are appended to the /metrics response on each scrape.
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    pub metrics_file: String,
}

pub fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = TcpListener::bind(&addr)?;
    info!("quant serve listening on {addr}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("accept error: {e}");
                continue;
            }
        };

        let mut buf = [0u8; 2048];
        let _ = stream.read(&mut buf);
        let request = String::from_utf8_lossy(&buf);

        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");

        let (status, content_type, body) = match path {
            "/health" | "/healthz" => (
                "200 OK",
                "application/json",
                r#"{"status":"ok"}"#.to_string(),
            ),
            "/metrics" => (
                "200 OK",
                "text/plain; version=0.0.4; charset=utf-8",
                build_metrics(&args.metrics_file),
            ),
            "/api/status" => ("200 OK", "application/json", build_status(&args.db)),
            _ => ("200 OK", "text/html; charset=utf-8", dashboard_html()),
        };

        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
    Ok(())
}

fn build_status(db_path: &Option<String>) -> String {
    let db_status = match db_path {
        Some(path) => match MarketDataStore::open(path) {
            Ok(_) => serde_json::json!("connected"),
            Err(e) => serde_json::json!(format!("error: {e}")),
        },
        None => serde_json::json!("no database configured"),
    };

    serde_json::json!({
        "service": "quant-rs",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "database": db_status,
    })
    .to_string()
}

fn dashboard_html() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant Infrastructure</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a0f;--surface:#12121a;--border:#1e1e2e;--text:#e0e0e8;
--dim:#6b6b80;--green:#34d399;--red:#f87171;--amber:#fbbf24;
--mono:'SF Mono',SFMono-Regular,Menlo,Consolas,monospace}
body{font-family:var(--mono);background:var(--bg);color:var(--text);
font-size:13px;line-height:1.6;min-height:100vh}
.container{max-width:960px;margin:0 auto;padding:32px 24px}
header{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:24px;
display:flex;justify-content:space-between;align-items:baseline}
h1{font-size:16px;font-weight:600;letter-spacing:0.5px}
h1 span{color:var(--dim);font-weight:400}
.ver{color:var(--dim);font-size:11px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:12px}
.stat{font-size:28px;font-weight:700;margin-bottom:4px}
.stat.ok{color:var(--green)}.stat.err{color:var(--red)}.stat.warn{color:var(--amber)}
.label{color:var(--dim);font-size:11px}
.metrics-table{width:100%;border-collapse:collapse}
.metrics-table td{padding:6px 8px;border-bottom:1px solid var(--border);font-size:12px}
.metrics-table td:first-child{color:var(--dim);width:50%}
.metrics-table td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.green{background:var(--green)}.dot.red{background:var(--red)}.dot.amber{background:var(--amber)}
.full{grid-column:1/-1}
footer{text-align:center;color:var(--dim);font-size:11px;margin-top:24px;
border-top:1px solid var(--border);padding-top:16px}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>QUANT <span>INFRASTRUCTURE</span></h1>
    <span class="ver" id="version"></span>
  </header>
  <div class="grid">
    <div class="card">
      <h2>Service Status</h2>
      <div class="stat" id="status"></div>
      <div class="label"><span class="dot" id="status-dot"></span><span id="status-label">loading</span></div>
    </div>
    <div class="card">
      <h2>Database</h2>
      <div class="stat" id="db-status"></div>
      <div class="label" id="db-detail"></div>
    </div>
    <div class="card full">
      <h2>Metrics</h2>
      <table class="metrics-table"><tbody id="metrics-body"></tbody></table>
    </div>
  </div>
  <footer>quant-rs · <span id="last-update"></span></footer>
</div>
<script>
function setText(id,v){document.getElementById(id).textContent=v}
function setClass(id,c){document.getElementById(id).className=c}
async function refresh(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    setText('version','v'+d.version);
    setText('status',d.status.toUpperCase());
    const ok=d.status==='running';
    setClass('status','stat '+(ok?'ok':'err'));
    setClass('status-dot','dot '+(ok?'green':'red'));
    setText('status-label',ok?'healthy':'degraded');
    const db=typeof d.database==='string'?d.database:'unknown';
    setText('db-status',db.toUpperCase());
    setClass('db-status','stat '+(db==='connected'?'ok':db.includes('error')?'err':'warn'));
    setText('db-detail',db==='connected'?'DuckDB':'');
  }catch(e){
    setText('status','UNREACHABLE');setClass('status','stat err');
  }
  try{
    const r=await fetch('/metrics');const t=await r.text();
    const tbody=document.getElementById('metrics-body');
    while(tbody.firstChild)tbody.removeChild(tbody.firstChild);
    t.split('\n').filter(l=>l&&!l.startsWith('#')).forEach(l=>{
      const parts=l.trim().split(/\s+/);
      const tr=document.createElement('tr');
      const k=document.createElement('td');k.textContent=parts[0]||'';
      const v=document.createElement('td');v.textContent=parts[1]||'';
      tr.appendChild(k);tr.appendChild(v);tbody.appendChild(tr);
    });
    if(!tbody.firstChild){
      const tr=document.createElement('tr');
      const td=document.createElement('td');td.colSpan=2;td.textContent='no metrics';
      tr.appendChild(td);tbody.appendChild(tr);
    }
  }catch(e){}
  setText('last-update','updated '+new Date().toLocaleTimeString());
}
refresh();setInterval(refresh,10000);
</script>
</body>
</html>"##.to_string()
}

fn build_metrics(metrics_file: &str) -> String {
    let mut out = String::from(
        "# HELP quant_up Whether the quant service is up\n\
         # TYPE quant_up gauge\n\
         quant_up 1\n",
    );
    if let Ok(extra) = std::fs::read_to_string(metrics_file) {
        out.push_str(&extra);
    }
    out
}
