"use strict";
const $ = (id) => document.getElementById(id);
const KEY = "pathsnap_api_base";

function lsGet() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
function lsSet(v) { try { localStorage.setItem(KEY, v); } catch (e) { /* ignore */ } }

const map = L.map("map").setView([0, 0], 2);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, attribution: "Tiles &copy; Esri"
}).addTo(map);
const gpsLayer = L.layerGroup().addTo(map);
const routeLayer = L.layerGroup().addTo(map);
L.control.layers(null, { "GPS points (blue)": gpsLayer, "Predicted route (red)": routeLayer }, { collapsed: false }).addTo(map);
const legend = L.control({ position: "bottomright" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = '<i style="background:#2563eb"></i>GPS points<br><i class="line" style="background:#dc2626"></i>Predicted route';
  return d;
};
legend.addTo(map);

const results = new Map();   // trip id -> {top1, cont, ms} for this browser session
let busy = false;

function base() { return $("apiBase").value.trim().replace(/\/+$/, ""); }
function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }
function esc(s) { const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }
const pct = (x) => (x == null ? "n/a" : (100 * x).toFixed(1) + "%");
const row = (k, v) => "<tr><td>" + k + "</td><td>" + v + "</td></tr>";

async function api(path, opts) {
  let r;
  try { r = await fetch(base() + path, opts); }
  catch (e) { throw new Error("Cannot reach API at " + base() + " (" + e.message + ")"); }
  if (!r.ok) {
    let d = "";
    try { d = (await r.json()).detail; } catch (e) { /* ignore */ }
    throw new Error("API error " + r.status + (d ? ": " + d : ""));
  }
  return r.json();
}

async function loadTrips() {
  showError(""); lsSet(base());
  $("conn").className = "dot";
  try {
    const h = await api("/health");
    $("conn").className = "dot ok";
    $("conn").title = "connected";
    $("modelInfo").textContent = h.model + " + " + h.decoder + " | " + h.n_trips + " trips";
    const t = await api("/trips");
    const box = $("trips"); box.innerHTML = "";
    if (!t.trips.length) box.innerHTML = '<p class="muted">API returned no trips.</p>';
    t.trips.forEach((id) => {
      const b = document.createElement("button");
      b.textContent = id; b.dataset.id = id;
      b.onclick = () => predict(id);
      box.appendChild(b);
    });
    $("runAll").disabled = !t.trips.length;
    refreshTripButtons();
  } catch (e) {
    $("conn").className = "dot bad"; $("conn").title = "not connected";
    $("modelInfo").textContent = "";
    showError(e.message);
  }
}

function refreshTripButtons(activeId) {
  document.querySelectorAll("#trips button").forEach((b) => {
    b.disabled = busy;
    b.classList.toggle("active", b.dataset.id === activeId);
    b.classList.toggle("done", results.has(b.dataset.id));
  });
  $("runAll").disabled = busy || !document.querySelector("#trips button");
}

function setBusy(on, text, activeId) {
  busy = on;
  $("loading").hidden = !on;
  if (on) $("loadingText").textContent = text;
  refreshTripButtons(activeId);
}

function renderSession() {
  const box = $("session");
  if (!results.size) { box.innerHTML = '<p class="muted">No trips run yet.</p>'; return; }
  let html = "<table><tr><th>Trip</th><th>Top-1</th><th>Continuity</th><th>Time</th></tr>";
  let n = 0, s1 = 0, s2 = 0, n1 = 0;
  [...results.entries()].sort().forEach(([id, r]) => {
    html += "<tr><td>" + esc(id) + "</td><td>" + pct(r.top1) + "</td><td>" + pct(r.cont) + "</td><td>" +
      (r.ms / 1000).toFixed(1) + " s</td></tr>";
    n += 1; s2 += r.cont; if (r.top1 != null) { s1 += r.top1; n1 += 1; }
  });
  if (n > 1) {
    html += '<tr class="mean"><td>Mean (' + n + ")</td><td>" + pct(n1 ? s1 / n1 : null) + "</td><td>" +
      pct(s2 / n) + "</td><td></td></tr>";
  }
  box.innerHTML = html + "</table>";
}

function render(d) {
  gpsLayer.clearLayers(); routeLayer.clearLayers();
  d.gps.forEach((p) => L.circleMarker(p, { radius: 4, color: "#1d4ed8", weight: 1, fillColor: "#2563eb", fillOpacity: .85 }).addTo(gpsLayer));
  d.route.forEach((a) => L.polyline(a.polyline, { color: "#dc2626", weight: 5, opacity: .85 }).addTo(routeLayer));
  const pts = d.gps.concat(...d.route.map((a) => a.polyline));
  if (pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [30, 30] });
  const m = d.metrics, t = d.timings_ms;
  $("metrics").innerHTML =
    '<div class="cards"><div class="card"><small>Top-1 accuracy</small><p class="big">' + pct(m.top1_accuracy) + "</p></div>" +
    '<div class="card"><small>Route continuity</small><p class="big">' + pct(m.route_continuity) + "</p></div></div>" +
    "<table>" + row("Trip", esc(d.trip_id)) + row("GPS observations", m.n_obs) + row("Route arcs", m.n_route_arcs) +
    row("Model", esc(d.model)) + row("Decoder", esc(d.decoder)) +
    row("Candidates + features", t.candidates_features_ms + " ms") + row("Model scoring", t.scoring_ms + " ms") +
    row("HMM decoding", t.hmm_ms + " ms") + row("Total", t.total_ms + " ms") + "</table>";
}

async function runTrip(id) {
  const d = await api("/predict/" + encodeURIComponent(id), { method: "POST" });
  results.set(id, { top1: d.metrics.top1_accuracy, cont: d.metrics.route_continuity, ms: d.timings_ms.total_ms });
  render(d); renderSession();
}

async function predict(id) {
  showError(""); setBusy(true, "Running pipeline for trip " + id + "...", id);
  try { await runTrip(id); }
  catch (e) { showError(e.message); }
  finally { setBusy(false, "", id); }
}

async function runAll() {
  showError("");
  const ids = [...document.querySelectorAll("#trips button")].map((b) => b.dataset.id);
  for (let i = 0; i < ids.length; i++) {
    setBusy(true, "Trip " + ids[i] + " (" + (i + 1) + " of " + ids.length + ")...", ids[i]);
    try { await runTrip(ids[i]); }
    catch (e) { showError("Trip " + ids[i] + ": " + e.message); break; }
  }
  setBusy(false, "", null);
}

// API address: ?api=http://host:8000 in the link > last used > default
const qp = new URLSearchParams(location.search).get("api");
$("apiBase").value = qp || lsGet() || $("apiBase").value;
$("reload").onclick = loadTrips;
$("runAll").onclick = runAll;
$("apiBase").addEventListener("keydown", (e) => { if (e.key === "Enter") loadTrips(); });
renderSession();
// ?run=<trip_id> or ?run=all starts a run as soon as the trip list is loaded
loadTrips().then(() => {
  const r = new URLSearchParams(location.search).get("run");
  if (r === "all") runAll(); else if (r) predict(r);
});
