"use strict";

const $ = (id) => document.getElementById(id);
const GESTURES = ["nod_yes", "shake_no", "tilt_head", "wiggle_antennas",
  "look_around", "happy", "excited", "sad", "surprised", "sleepy", "dance", "reset"];

let loaded = false;
let stateFails = 0;

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(res.status + " " + res.statusText);
  return res.json();
}

// ---- optimistic writes ----
// /api/state is polled every 3s, so a response already in flight when you drag
// the slider would land after your POST and rewrite the old value. Hold the
// value we just sent until the robot echoes it back (or we give up on it).
const pending = new Map(); // settings key -> {value, expires}
const PENDING_MS = 8000;   // a couple of poll rounds

function claim(key, value) {
  pending.set(key, { value, expires: Date.now() + PENDING_MS });
}

// True if the poll may write `serverValue` into this control.
function settled(key, serverValue) {
  const p = pending.get(key);
  if (!p) return true;
  const same = Math.abs(p.value - Number(serverValue)) < 1e-6;
  // Echoed back, or we've waited long enough that the robot clearly disagrees
  // (setting rejected, app restarted) — either way, stop holding it.
  if (same || Date.now() > p.expires) { pending.delete(key); return true; }
  return false;
}

// One chain per key so two fast changes to the same control can't land out of
// order — the last value you chose is the last one the robot sees.
const chains = new Map();

function postSetting(key, value) {
  claim(key, value);
  const body = {}; body[key] = value;
  const next = (chains.get(key) || Promise.resolve())
    .then(() => api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }))
    .catch(() => {
      pending.delete(key); // don't keep holding a value the robot never took
      toast("Failed to update");
    });
  chains.set(key, next);
  return next;
}

// ---- load current state ----
async function loadState() {
  let data;
  try {
    data = await api("/api/state");
  } catch (e) {
    stateFails++;
    setStatus(null);
    return;
  }
  stateFails = 0;
  const s = data.settings;

  // voices dropdown
  const vsel = $("voice");
  if (vsel.options.length === 0) {
    (data.voices || []).forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      vsel.appendChild(o);
    });
  }

  // Only overwrite fields the user isn't editing right now.
  setToggle("face_tracking", s.face_tracking);
  setToggle("ambient", s.ambient);
  setToggle("half_duplex", s.half_duplex);
  setSlider("speaker_volume", "volVal", s.speaker_volume);
  $("faceNote").textContent = s.face_available ? "" : "(camera/cascade unavailable)";

  if (!loaded) {
    $("voice").value = s.voice;
    // Model is a fixed dropdown; if the configured model isn't listed
    // (e.g. set via env), add it so the current value is always shown.
    const msel = $("model");
    if (![...msel.options].some((o) => o.value === s.model)) {
      const o = document.createElement("option");
      o.value = s.model; o.textContent = s.model;
      msel.appendChild(o);
    }
    msel.value = s.model;
    $("instructions").value = s.instructions;
    $("greeting").value = s.greeting;
    loaded = true;
  }

  const st = data.status;
  $("keyHint").textContent = st.key_set
    ? "✓ Key saved" + (st.key_hint ? " (" + st.key_hint + ")" : "") +
      " — stored on the robot. Paste a new key to replace it."
    : "No key yet — paste your OpenAI API key to start chatting. " +
      "It is stored only on the robot, never shared.";
  const keyEl = $("apikey");
  if (document.activeElement !== keyEl && !keyEl.value) {
    keyEl.placeholder = st.key_set ? "••••••••••••••••••••" : "sk-…";
  }

  setStatus(st.connected, st.key_set, st.error);
}

function setToggle(id, val) {
  const el = $(id);
  if (el && document.activeElement !== el) el.checked = !!val;
}

// ---- sliders ----
// The UI works in percent, the backend in a plain multiplier.
function showPct(valId, pct) {
  const out = $(valId);
  out.textContent = pct + "%";
  out.classList.toggle("boosted", pct > 100 && pct <= 200);
  out.classList.toggle("hot", pct > 200);
}

function setSlider(id, valId, val) {
  const el = $(id);
  if (!el || document.activeElement === el) return; // don't fight a live drag
  const raw = val == null ? 1 : val;
  if (!settled(id, raw)) return;                    // ours is newer
  const pct = Math.round(raw * 100);
  el.value = pct;
  showPct(valId, pct);
}

function wireSlider(id, key, valId) {
  const el = $(id);
  let debounce = null;
  const send = () => {
    clearTimeout(debounce); debounce = null;
    postSetting(key, Number(el.value) / 100);
  };
  // Apply while dragging (debounced) so you hear it change as you move, and
  // again on release so the resting value is always the one that sticks.
  el.addEventListener("input", () => {
    showPct(valId, Number(el.value));
    clearTimeout(debounce);
    debounce = setTimeout(send, 150);
  });
  el.addEventListener("change", send);
}

function setStatus(connected, keySet, error) {
  const dot = $("dot"), txt = $("statusText");
  if (connected === null) {
    // Don't flash the failure message on a single missed poll (e.g. while
    // the app is still starting) — keep the loading state for the first two.
    if (stateFails < 2) { dot.className = "dot wait"; txt.textContent = "connecting…"; return; }
    dot.className = "dot off"; txt.textContent = "settings server only — app not running"; return;
  }
  if (keySet === false) { dot.className = "dot off"; txt.textContent = "waiting for API key"; return; }
  if (connected) { dot.className = "dot on"; txt.textContent = "connected to OpenAI"; return; }
  dot.className = "dot off";
  txt.textContent = error ? "not connected — " + error : "reconnecting…";
}

// ---- instant toggles ----
function wireToggle(id) {
  $(id).addEventListener("change", async (e) => {
    const body = {}; body[id] = e.target.checked;
    try {
      await api("/api/settings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      toast("Updated: " + id.replace("_", " "));
    } catch (err) { toast("Failed to update"); }
  });
}

// ---- save voice/personality (reconnect) ----
async function save() {
  const body = {
    voice: $("voice").value,
    model: $("model").value,
    instructions: $("instructions").value,
    greeting: $("greeting").value,
  };
  try {
    const r = await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    toast(r.reconnect ? "Saved — reconnecting to apply…" : "Saved (no change)");
  } catch (e) { toast("Save failed"); }
}

// ---- API key ----
async function saveKey() {
  const key = $("apikey").value.trim();
  if (!key) { toast("Paste a key first"); return; }
  try {
    const r = await api("/api/key", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    $("apikey").value = "";
    toast(r.persisted ? "Key saved — connecting…"
                      : "Key active (couldn't persist — re-enter after restart)");
    loadState();
  } catch (e) { toast("Failed to save key"); }
}

// ---- gesture test buttons ----
function buildGestures() {
  const box = $("gestures");
  GESTURES.forEach((g) => {
    const b = document.createElement("button");
    b.textContent = g.replace(/_/g, " ");
    b.addEventListener("click", async () => {
      try {
        await api("/api/express", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: g }),
        });
      } catch (e) { toast("Gesture failed"); }
    });
    box.appendChild(b);
  });
}

// ---- transcript ----
let lastTranscript = "";

async function loadTranscript() {
  let data;
  try { data = await api("/api/transcript?n=30"); } catch (e) { return; }
  const box = $("transcript");
  const entries = data.entries || [];
  let html;
  if (entries.length === 0) {
    html = '<div class="t-empty">No messages yet — say hello to your robot.</div>';
  } else {
    html = entries.map((e) => {
      const txt = escapeHtml(e.text || "");
      if (e.role === "user") return `<div class="t-user"><b>You:</b> ${txt}</div>`;
      if (e.role === "assistant") return `<div class="t-assistant"><b>Reachy:</b> ${txt}</div>`;
      if (e.role === "tool") {
        let img = "";
        const f = e.meta && e.meta.image_file;
        if (f && /^[A-Za-z0-9_-]+\.jpg$/.test(f)) {
          const u = "/api/image/" + encodeURIComponent(f);
          img = `<a href="${u}" target="_blank" rel="noopener">` +
                `<img class="t-thumb" src="${u}" alt="camera capture" loading="lazy"></a>`;
        }
        return `<div class="t-tool">→ ${txt}${img}</div>`;
      }
      return "";
    }).join("");
  }
  if (html === lastTranscript) return; // nothing new — leave the DOM alone
  lastTranscript = html;
  // Keep the reader's place: only snap to the end if they were already there.
  const nearBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
  const prevTop = box.scrollTop;
  box.innerHTML = html;
  box.scrollTop = nearBottom ? box.scrollHeight : prevTop;
}

// ---- expand / collapse transcript ----
function setExpanded(on) {
  document.body.classList.toggle("transcript-full", on);
  const b = $("expand");
  b.textContent = on ? "⤡" : "⤢";
  b.title = on ? "Collapse transcript (Esc)" : "Expand transcript";
  const box = $("transcript");
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ---- boot ----
["face_tracking", "ambient", "half_duplex"].forEach(wireToggle);
wireSlider("speaker_volume", "speaker_volume", "volVal");
$("save").addEventListener("click", save);
$("saveKey").addEventListener("click", saveKey);
$("expand").addEventListener("click", () =>
  setExpanded(!document.body.classList.contains("transcript-full")));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setExpanded(false);
});
$("apikey").addEventListener("keydown", (e) => { if (e.key === "Enter") saveKey(); });
buildGestures();
loadState();
loadTranscript();
setInterval(loadState, 3000);
setInterval(loadTranscript, 4000);
