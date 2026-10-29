#!/usr/bin/env node
/**
 * AI Mark Checker — programmatic interface (no website needed).
 *
 *   npm install            (once, in this folder)
 *   node check-ai-marks.mjs <file-or-URL> [more…]      → JSON, one result per picture
 *
 * Or from JavaScript:   import { checkAiMarks } from "./check-ai-marks.mjs";
 *                       const result = await checkAiMarks(bytes, "image/png");
 *
 * Reads, without uploading anything anywhere:
 *   - Content Credentials (C2PA) with InfinisX's disclosure (provider, system,
 *     version, time, ID, created/altered),
 *   - the invisible TrustMark watermark (Adobe; models download once, ~65 MB,
 *     SHA-256-checked, into ~/.cache/infinisx-trustmark),
 *   - China's GB 45438 "AIGC" label.
 * No personal data is read, logged or kept. MIT licence (see ../checker/LICENSE).
 * Feedback: https://github.com/crackvenom/infinisx-open/issues
 */
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Reader, Trustmark } from "@contentauth/c2pa-node";

const MODELS = path.join(os.homedir(), ".cache", "infinisx-trustmark");
const HOST = "https://cai-watermark.adobe.net/watermarking/trustmark-models/";
const FILES = [
  { name: "encoder_Q.onnx", sha256: "19b3d1b25836130ffd78775a8f61539f993375d1823ef0e59ba5b8dffb4f892d" },
  { name: "decoder_Q.onnx", sha256: "ee3268f057c9dabef680e169302f5973d0589feea86189ed229a896cc3aa88df" },
];

async function trustmark() {
  await fs.mkdir(MODELS, { recursive: true });
  for (const f of FILES) {
    const file = path.join(MODELS, f.name);
    const have = await fs.readFile(file).catch(() => null);
    if (have && createHash("sha256").update(have).digest("hex") === f.sha256) continue;
    const res = await fetch(HOST + f.name);
    if (!res.ok) throw new Error(`watermark model ${f.name}: HTTP ${res.status}`);
    const bytes = Buffer.from(await res.arrayBuffer());
    if (createHash("sha256").update(bytes).digest("hex") !== f.sha256) throw new Error(`watermark model ${f.name} failed its checksum`);
    await fs.writeFile(file, bytes);
  }
  return Trustmark.newTrustmark({ variant: "Q", version: "BCH_5", modelPath: MODELS });
}

/** The ID's first 61 bits — what InfinisX writes into the watermark. */
export function idBits(id) {
  const hex = String(id).replace(/-/g, "");
  if (!/^[0-9a-f]{32}$/i.test(hex)) return null;
  return Array.from(hex, (h) => parseInt(h, 16).toString(2).padStart(4, "0")).join("").slice(0, 61);
}

function pngText(buf, keyword) {
  if (buf[0] !== 137 || buf[1] !== 80) return null;
  let at = 8;
  while (at + 12 <= buf.length) {
    const len = buf.readUInt32BE(at);
    const type = buf.toString("latin1", at + 4, at + 8);
    if (type === "tEXt") {
      const data = buf.subarray(at + 8, at + 8 + len);
      const nul = data.indexOf(0);
      if (nul > 0 && data.toString("latin1", 0, nul) === keyword) return data.toString("latin1", nul + 1);
    }
    if (type === "IEND") break;
    at += 12 + len;
  }
  return null;
}

let tm = null;

/** Check one picture's bytes. */
export async function checkAiMarks(bytes, mimeType = "image/png") {
  const buf = Buffer.from(bytes);
  let credentials = null;
  try {
    const reader = await Reader.fromAsset({ buffer: buf, mimeType });
    const store = reader?.json();
    const m = store?.active_manifest ? store.manifests[store.active_manifest] : null;
    if (m) {
      const failures = (store.validation_results?.activeManifest?.failure ?? []).map((f) => f.code);
      const disclosure = m.assertions?.find((a) => a.label === "com.infinisx.ai-disclosure")?.data ?? null;
      const actions = m.assertions?.find((a) => /^c2pa\.actions/.test(a.label))?.data?.actions ?? [];
      credentials = {
        intact: !failures.some((f) => f !== "signingCredential.untrusted" && !f.startsWith("signingCredential.ocsp")),
        untrustedSigner: failures.includes("signingCredential.untrusted"),
        failures,
        generator: m.claim_generator_info?.[0] ?? null,
        digitalSourceType: actions.map((a) => a.digitalSourceType).find(Boolean) ?? null,
        disclosure,
      };
    }
  } catch {
    credentials = null; // no C2PA in this file
  }
  tm ??= await trustmark();
  let watermark = null;
  try {
    const found = await tm.decode(buf);
    watermark = typeof found === "string" && /^[01]+$/.test(found) ? found : null;
  } catch {
    watermark = null; // "watermark is corrupt or missing"
  }
  const aigcRaw = pngText(buf, "AIGC");
  let aigc = null;
  try {
    aigc = aigcRaw ? JSON.parse(aigcRaw) : null;
  } catch {
    aigc = { raw: aigcRaw };
  }
  const id = credentials?.disclosure?.id ?? null;
  return {
    aiMarked: !!(credentials || watermark || aigc),
    credentials,
    watermark: watermark ? { bits: watermark, matchesCredentialId: id ? watermark === idBits(id) : null } : null,
    aigc,
  };
}

async function load(arg) {
  if (/^https?:\/\//i.test(arg)) {
    const res = await fetch(arg);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return { bytes: Buffer.from(await res.arrayBuffer()), mime: (res.headers.get("content-type") ?? "image/png").split(";")[0] };
  }
  const ext = path.extname(arg).toLowerCase();
  return { bytes: await fs.readFile(arg), mime: ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : ext === ".webp" ? "image/webp" : "image/png" };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const args = process.argv.slice(2);
  if (!args.length) {
    console.error("usage: node check-ai-marks.mjs <file-or-URL> [more…]");
    process.exit(2);
  }
  const out = [];
  for (const a of args) {
    try {
      const { bytes, mime } = await load(a);
      out.push({ input: a, ...(await checkAiMarks(bytes, mime)) });
    } catch (e) {
      out.push({ input: a, error: e instanceof Error ? e.message : String(e) });
    }
  }
  console.log(JSON.stringify(out, null, 2));
}
