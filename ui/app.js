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

function base() { return $("apiBase").value.trim().replace(/\/+$/, ""); }

function showError(msg) {
  const e = $("error");
  e.textContent = msg; e.hidden = !msg;
}

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
    $("conn").className = "dot ok"; $("conn").title = "connected: " + h.model + " / " + h.decoder;
    const t = await api("/trips");
    const box = $("trips"); box.innerHTML = "";
    if (!t.trips.length) box.innerHTML = '<p class="muted">API returned no trips.</p>';
    t.trips.forEach((id) => {
      const b = document.createElement("button");
      b.textContent = id; b.dataset.id = id;
      b.onclick = () => predict(id);
      box.appendChild(b);
    });
  } catch (e) {
    $("conn").className = "dot bad"; $("conn").title = "not connected";
    showError(e.message);
  }
}

function setBusy(on, id) {
  $("loading").hidden = !on;
  if (on) $("loadingText").textContent = "Running pipeline for trip " + id + "...";
  document.querySelectorAll("#trips button").forEach((b) => {
    b.disabled = on; b.classList.toggle("active", b.dataset.id === id);
  });
}

const pct = (x) => (x == null ? "n/a" : (100 * x).toFixed(1) + "%");
const row = (k, v) => "<tr><td>" + k + "</td><td>" + v + "</td></tr>";

function esc(s) { const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }

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

async function predict(id) {
  showError(""); setBusy(true, id);
  try { render(await api("/predict/" + encodeURIComponent(id), { method: "POST" })); }
  catch (e) { showError(e.message); }
  finally { setBusy(false, id); }
}

$("apiBase").value = lsGet() || $("apiBase").value;
$("reload").onclick = loadTrips;
$("apiBase").addEventListener("keydown", (e) => { if (e.key === "Enter") loadTrips(); });
loadTrips();
