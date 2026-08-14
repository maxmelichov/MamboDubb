/* Dubbing editor — one page, no build step.
 *
 * State is deliberately thin: `view` is whatever the server last told us about a
 * run, and `dirty` holds only the fields the user actually changed, keyed by
 * segment id. Save sends exactly that, so an untouched form invalidates nothing.
 */

const $ = (sel) => document.querySelector(sel);
const api = async (url, opts) => {
  const res = await fetch(url, opts);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(body?.detail || res.statusText);
  return body;
};

let view = null;           // GET /api/runs/<run>
let dirty = new Map();     // seg id -> {field: value}
let jobTimer = null;

const toast = (msg, ms = 3000) => {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (el.hidden = true), ms);
};

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const fmt = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(2).padStart(5, "0")}`;

/* ---------------------------------------------------------------- routing */

async function route() {
  const hash = location.hash || "#/";
  const match = hash.match(/^#\/run\/(.+)$/);
  if (match) await showEditor(decodeURIComponent(match[1]));
  else await showImport();
}

async function showImport() {
  $("#editor-screen").hidden = true;
  $("#import-screen").hidden = false;
  $("#save").hidden = $("#rerun").hidden = true;
  $("#crumb").textContent = "";
  const { runs } = await api("/api/runs");
  $("#runs tbody").innerHTML = runs.map((r) => `
    <tr>
      <td><a href="#/run/${encodeURIComponent(r.run)}">${r.run}</a></td>
      <td>${esc(r.title)}</td>
      <td>${r.src_lang ?? "?"} → ${r.tgt_lang ?? "?"}</td>
      <td>${r.segments} segments</td>
      <td>${r.stages.length}/9 stages${r.has_preview ? " · preview" : ""}</td>
    </tr>`).join("") || "<tr><td>No runs in outputs/ yet.</td></tr>";
}

async function showEditor(run) {
  $("#import-screen").hidden = true;
  $("#editor-screen").hidden = false;
  $("#save").hidden = $("#rerun").hidden = false;
  dirty = new Map();
  view = await api(`/api/runs/${encodeURIComponent(run)}`);
  $("#crumb").textContent = `${view.run} · ${view.source.src_lang} → ${view.source.tgt_lang}`;
  renderHead();
  renderSegments();
}

/* --------------------------------------------------------------- rendering */

function renderHead() {
  const media = view.media || {};
  const video = media["preview.mp4"] || media["source_video.mp4"];
  $("#run-head").innerHTML = `
    ${video ? `<video controls src="/api/runs/${encodeURIComponent(view.run)}/file/${video}"></video>` : ""}
    <div>
      <h1>${esc(view.source.title || view.run)}</h1>
      <p>${view.segments.length} segments · speakers: ${Object.keys(view.speakers).join(", ") || "none"}</p>
      <p>stages done: ${view.stages.join(", ") || "none"}</p>
    </div>`;
}

function speakerOptions(current) {
  const ids = new Set([...Object.keys(view.speakers), ...view.segments.map((s) => s.speaker)]);
  ids.add(current);
  return [...ids].filter(Boolean).sort()
    .map((id) => `<option ${id === current ? "selected" : ""}>${esc(id)}</option>`).join("");
}

function renderSegments() {
  const run = encodeURIComponent(view.run);
  $("#segments").innerHTML = view.segments.map((s) => `
    <div class="seg${s.passthrough ? " passthrough" : ""}" data-id="${s.id}">
      <div class="meta">
        <span class="time">#${s.id} ${fmt(s.start)} → ${fmt(s.end)}</span>
        ${s.keep ? `<span class="badge keep">keep: ${esc(s.keep_reason ?? "—")}</span>` : ""}
        ${s.tts ? `<span class="badge">tts ${s.tts.verify}</span>` : `<span class="badge">no clip</span>`}
        <span class="inline">
          <button class="ghost" data-play="/api/runs/${run}/segments/${s.id}/original">▶ original</button>
          <button class="ghost" data-play="/api/runs/${run}/segments/${s.id}/dubbed">▶ dubbed</button>
        </span>
      </div>
      <div class="fields">
        <label>source text
          <textarea rows="2" data-field="text">${esc(s.text)}</textarea>
        </label>
        <div class="inline">
          <label>spoken lang <input size="4" data-field="lang" value="${esc(s.lang)}" placeholder="${view.source.src_lang}"></label>
          <label>speaker <select data-field="speaker">${speakerOptions(s.speaker)}</select></label>
          <label><input type="checkbox" data-field="passthrough" ${s.passthrough ? "checked" : ""}> passthrough (keep original)</label>
        </div>
      </div>
      <div class="fields">
        <label>translation
          <textarea rows="2" data-field="text_en">${esc(s.text_en)}</textarea>
        </label>
        <div class="inline">
          <label>target lang <input size="4" data-field="lang_override" value="${esc(s.lang_override)}" placeholder="${view.source.tgt_lang}"></label>
          <label style="flex:1">TTS instructions
            <input style="width:100%" data-field="tts_instructions" value="${esc(s.tts_instructions)}"
                   placeholder="e.g. calm, slightly amused">
          </label>
        </div>
      </div>
    </div>`).join("");
}

/* ------------------------------------------------------------ interaction */

$("#segments").addEventListener("input", (ev) => {
  const field = ev.target.dataset.field;
  if (!field) return;
  const row = ev.target.closest(".seg");
  const id = Number(row.dataset.id);
  const value = ev.target.type === "checkbox" ? ev.target.checked : ev.target.value;
  const fields = dirty.get(id) || {};
  fields[field] = value;
  dirty.set(id, fields);
  row.classList.add("dirty");
  if (field === "passthrough") row.classList.toggle("passthrough", value);
});
$("#segments").addEventListener("click", (ev) => {
  const url = ev.target.dataset.play;
  if (!url) return;
  const player = $("#player");
  player.src = url;
  player.play().catch(() => toast("nothing to play for this segment yet"));
});

$("#save").addEventListener("click", async () => {
  if (!dirty.size) return toast("nothing changed");
  const edits = [...dirty].map(([id, fields]) => ({ id, fields }));
  try {
    const res = await api(`/api/runs/${encodeURIComponent(view.run)}/segments`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ edits }),
    });
    dirty = new Map();
    await showEditor(view.run);
    if (res.force) {
      toast(`saved ${res.changed.length} segments — re-run from '${res.force}' to apply`, 6000);
      $("#rerun").dataset.force = res.force;
      $("#rerun").textContent = `Re-run from ${res.force}`;
    } else {
      toast("saved (nothing changed)");
    }
  } catch (err) {
    toast(`save failed: ${err.message}`, 6000);
  }
});

$("#rerun").addEventListener("click", async () => {
  const force = $("#rerun").dataset.force || null;
  const job = await api(`/api/runs/${encodeURIComponent(view.run)}/rerun`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ force }),
  });
  watchJob(job.id);
});

$("#import-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = new FormData(ev.target);
  if (!form.get("url") && !form.get("file")?.size) return toast("give a URL or a file");
  if (!form.get("file")?.size) form.delete("file");
  if (!form.get("url")) form.delete("url");
  if (!form.get("duration")) form.delete("duration");
  try {
    const job = await api("/api/import", { method: "POST", body: form });
    watchJob(job.id);
  } catch (err) {
    toast(`import failed: ${err.message}`, 6000);
  }
});

/* ------------------------------------------------------------------- jobs */

function watchJob(id) {
  $("#jobs").hidden = false;
  clearInterval(jobTimer);
  const poll = async () => {
    const job = await api(`/api/jobs/${id}`);
    $("#job-label").textContent = job.label;
    $("#job-status").textContent = job.status;
    $("#job-log").textContent = job.log.join("\n");
    $("#job-log").scrollTop = $("#job-log").scrollHeight;
    if (job.status !== "running") {
      clearInterval(jobTimer);
      toast(`job ${job.status}`);
      if (job.run) {
        if (location.hash === `#/run/${encodeURIComponent(job.run)}`) showEditor(job.run);
        else location.hash = `#/run/${encodeURIComponent(job.run)}`;
      }
    }
  };
  poll();
  jobTimer = setInterval(poll, 2000);
}

$("#job-close").addEventListener("click", () => {
  clearInterval(jobTimer);
  $("#jobs").hidden = true;
});

window.addEventListener("hashchange", route);
route();
