// AI Mark Checker: reads Content Credentials (c2pa-web), the TrustMark
// watermark (Adobe's decoder, vendor/trustmark) and China's AIGC field, all
// in the browser. Mirrors what InfinisX writes (src/lib/aiMarks.ts).
import { createC2pa, Reader } from "./vendor/c2pa-web/index.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const SOURCE_TYPES = {
  trainedAlgorithmicMedia: "made by AI (from a text description)",
  compositeWithTrainedAlgorithmicMedia: "AI-assisted (an artwork changed or finished with AI)",
  algorithmicMedia: "made by an algorithm",
  digitalCapture: "a camera picture",
  digitalArt: "digital art",
};

let c2pa = null;
const c2paReady = createC2pa({ wasmSrc: new URL("./vendor/c2pa-web/resources/c2pa_bg.wasm", import.meta.url).href })
  .then((c) => (c2pa = c))
  .catch((e) => console.error("c2pa-web", e));

/* global sessions, session_resize, runwmark */
const watermarkReady = () => typeof sessions !== "undefined" && sessions.session_wmarkQ && session_resize;
async function waitForWatermark(ms = 120000) {
  const t = Date.now();
  while (!watermarkReady() && Date.now() - t < ms) await new Promise((r) => setTimeout(r, 200));
  return !!watermarkReady();
}

(async () => {
  await c2paReady;
  $("status").textContent = watermarkReady() ? "Ready. Drop a picture." : "Ready for Content Credentials; the watermark reader is still loading (about 50 MB, once)…";
  if (await waitForWatermark()) $("status").textContent = "Ready. Drop a picture.";
})();

// The watermark carries the first 61 bits of the picture's ID (InfinisX: src/lib/aiMarks.ts watermarkBitsFor).
function idBits(id) {
  const hex = String(id).replace(/-/g, "");
  if (!/^[0-9a-f]{32}$/i.test(hex)) return null;
  return Array.from(hex, (h) => parseInt(h, 16).toString(2).padStart(4, "0")).join("").slice(0, 61);
}

function pngText(bytes, keyword) {
  if (bytes[0] !== 137 || bytes[1] !== 80) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 8;
  while (at + 12 <= bytes.length) {
    const len = view.getUint32(at);
    const type = String.fromCharCode(...bytes.subarray(at + 4, at + 8));
    const data = bytes.subarray(at + 8, at + 8 + len);
    if (type === "tEXt" || type === "iTXt") {
      const nul = data.indexOf(0);
      if (nul > 0 && new TextDecoder("latin1").decode(data.subarray(0, nul)) === keyword) {
        if (type === "tEXt") return new TextDecoder("latin1").decode(data.subarray(nul + 1));
        // iTXt: keyword\0 flag method lang\0 translated\0 text (uncompressed only)
        let p = nul + 3;
        p = data.indexOf(0, p) + 1;
        p = data.indexOf(0, p) + 1;
        return data[nul + 1] === 0 ? new TextDecoder().decode(data.subarray(p)) : null;
      }
    }
    if (type === "IEND") break;
    at += 12 + len;
  }
  return null;
}

async function readCredentials(file) {
  await c2paReady;
  if (!c2pa) return { error: "The Content Credentials reader did not load." };
  let reader;
  try {
    reader = await Reader.fromBlob(c2pa, file.type || "image/png", file);
  } catch (e) {
    return { none: true, detail: String(e?.message ?? e) };
  }
  if (!reader) return { none: true };
  try {
    const store = await reader.manifestStore();
    const id = store.active_manifest ?? store.activeManifest;
    const manifest = id ? store.manifests?.[id] : null;
    const vr = store.validation_results?.activeManifest ?? store.validationResults?.activeManifest ?? {};
    const failures = (vr.failure ?? store.validation_status ?? []).map((f) => f.code);
    return { manifest, failures };
  } finally {
    await reader.free?.();
  }
}

function credentialsRow(c) {
  if (c.error) return `<h2><span class="warn">?</span> Content Credentials</h2><p>${esc(c.error)}</p>`;
  if (c.none || !c.manifest) return `<h2><span class="none">–</span> Content Credentials: none</h2><p>This file has no C2PA record (or it was removed, e.g. by a screenshot or an app that re-saves pictures).</p>`;
  const m = c.manifest;
  const tampered = c.failures.some((f) => f !== "signingCredential.untrusted" && !/^signingCredential\.ocsp/.test(f));
  const untrusted = c.failures.includes("signingCredential.untrusted");
  const assertions = m.assertions ?? [];
  const actions = assertions.find((a) => /^c2pa\.actions/.test(a.label))?.data?.actions ?? [];
  const dst = actions.map((a) => a.digitalSourceType ?? a.digital_source_type).find(Boolean);
  const dstName = dst ? SOURCE_TYPES[dst.split("/").pop()] ?? dst.split("/").pop() : null;
  const disclosure = assertions.find((a) => a.label === "com.infinisx.ai-disclosure")?.data;
  const gen = (m.claim_generator_info ?? m.claimGeneratorInfo ?? [])[0];
  const head = tampered
    ? `<span class="bad">✗</span> Content Credentials: <span class="bad">the picture was changed after it was signed</span>`
    : `<span class="ok">✓</span> Content Credentials: intact`;
  const rows = [
    gen && ["Made with", `${gen.name ?? ""} ${gen.version ?? ""}`.trim()],
    dstName && ["Source", dstName],
    disclosure?.provider && ["Provider", disclosure.provider],
    disclosure?.time && ["When", new Date(disclosure.time).toLocaleString()],
    disclosure?.content && ["Content", disclosure.content === "created" ? "created by AI" : "altered by AI"],
    disclosure?.id && ["ID", disclosure.id],
    ["Signer", untrusted ? "not on the C2PA trust list (shown as unrecognised; the record is still valid)" : "recognised"],
  ].filter(Boolean);
  return `<h2>${head}</h2><dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>${tampered ? `<p>Problems: ${esc(c.failures.join(", "))}</p>` : ""}`;
}

async function readWatermark(dataUrl) {
  if (!(await waitForWatermark(60000))) return { error: "The watermark reader could not load (it needs to download its model from Adobe once)." };
  try {
    const r = await runwmark(dataUrl);
    return r?.watermark_present ? { bits: r.watermark, schema: r.schema } : { none: true };
  } catch (e) {
    return { error: String(e?.message ?? e) };
  }
}

function watermarkRow(w, credentialId) {
  if (w.error) return `<h2><span class="warn">?</span> Invisible watermark</h2><p>${esc(w.error)}</p>`;
  if (w.none) return `<h2><span class="none">–</span> Invisible watermark: none found</h2><p>No TrustMark watermark in the pixels.</p>`;
  const want = credentialId ? idBits(credentialId) : null;
  const match = want && w.bits === want;
  return `<h2><span class="ok">✓</span> Invisible watermark: found</h2><p>A TrustMark watermark is in the pixels${match ? ", and it matches this file's Content Credentials ID" : want ? " (it does not match this file's credentials — the picture may have been re-marked or edited)" : " — this picture was marked as AI-made, even if its credentials were removed"}.</p><dl><dt>Payload</dt><dd>${esc(w.bits)}</dd></dl>`;
}

function aigcRow(json) {
  if (!json) return `<h2><span class="none">–</span> China AIGC label: none</h2><p>No GB 45438 "AIGC" field in the file.</p>`;
  let d = null;
  try {
    d = JSON.parse(json);
  } catch {
    /* shown raw below */
  }
  const label = d?.Label === "1" ? "AI-generated" : d?.Label === "2" ? "may be AI-generated" : d?.Label === "3" ? "suspected AI-generated" : "present";
  return `<h2><span class="ok">✓</span> China AIGC label: ${esc(label)}</h2><dl>${d ? Object.entries(d).filter(([, v]) => v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("") : `<dt>Raw</dt><dd>${esc(json)}</dd>`}</dl>`;
}

const toDataUrl = (file) => new Promise((resolve, reject) => {
  const fr = new FileReader();
  fr.onload = () => resolve(fr.result);
  fr.onerror = () => reject(fr.error);
  fr.readAsDataURL(file);
});

async function check(file) {
  $("status").textContent = `Checking ${file.name}…`;
  const bytes = new Uint8Array(await file.arrayBuffer());
  const dataUrl = await toDataUrl(file);
  $("preview").src = dataUrl;
  const [cred, wm] = await Promise.all([readCredentials(file), readWatermark(dataUrl)]);
  const aigc = pngText(bytes, "AIGC");
  const disclosureId = cred.manifest?.assertions?.find((a) => a.label === "com.infinisx.ai-disclosure")?.data?.id ?? null;
  $("c2pa").innerHTML = credentialsRow(cred);
  $("wm").innerHTML = watermarkRow(wm, disclosureId);
  $("aigc").innerHTML = aigcRow(aigc);
  const anyMark = (cred.manifest && !cred.none) || wm.bits || aigc;
  $("verdict").innerHTML = anyMark ? `<span class="ok">This picture carries AI marks</span> — it says it was made or changed with AI.` : `<span class="none">No AI marks found.</span> That does not prove no AI was used.`;
  $("result").classList.add("show");
  $("status").textContent = `Checked ${file.name}.`;
  document.body.dataset.checked = JSON.stringify({ credentials: cred.none ? "none" : cred.manifest ? (cred.failures.some((f) => f !== "signingCredential.untrusted") ? "tampered" : "intact") : "error", watermark: wm.bits ?? (wm.none ? "none" : "error"), watermarkMatches: !!(disclosureId && wm.bits === idBits(disclosureId)), aigc: !!aigc });
}

$("urlform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = $("url").value.trim();
  if (!url) return;
  $("status").textContent = `Fetching ${url}…`;
  try {
    const res = await fetch(url, { mode: "cors" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const name = decodeURIComponent(new URL(url).pathname.split("/").pop() || "picture");
    await check(new File([blob], name, { type: blob.type || "image/png" }));
  } catch (err) {
    $("status").textContent = `Could not fetch that address from this page (${err?.message ?? err}). Many sites block it — download the picture and drop it here instead.`;
  }
});

const drop = $("drop");
const input = $("file");
$("pick").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
input.addEventListener("change", () => input.files[0] && check(input.files[0]));
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); });
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && check(e.dataTransfer.files[0]));
