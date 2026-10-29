#!/usr/bin/env node
/**
 * Build the bundled render engine: `engine/` next to the app.
 *
 *   engine/python/   portable CPython 3.13 (python-build-standalone, install_only)
 *   engine/renderer/ Forge Neo source (Haoming02/sd-webui-forge-classic), pinned commit
 *   engine/renderer/extensions/adetailer/  ADetailer, pinned (Fix-ups)
 *   engine/engine.json  what was installed and when
 *
 * Forge's own `prepare_environment()` installs torch + gradio + requirements
 * into the portable Python the first time it runs, so this script only has
 * to put a Python and the source in place, then run launch.py once with
 * `--exit`. No Stability Matrix, no git, no system Python.
 *
 * Usage:  node scripts/build-engine.mjs            (full build, resumable)
 *         node scripts/build-engine.mjs --no-prepare   (skip the 5 GB pip step)
 *
 * Windows only for now (the installer target). Needs `tar` (ships with
 * Windows 10+) to unpack the downloads.
 */
import fs from "node:fs/promises";
import { createReadStream, createWriteStream } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// The desktop app puts the engine in LocalAppData (INFINISX_ENGINE_DIR), else
// in its writable home (INFINISX_HOME); dev uses the repo.
const ENGINE = process.env.INFINISX_ENGINE_DIR || path.join(process.env.INFINISX_HOME || ROOT, "engine");
const PY_DIR = path.join(ENGINE, "python");
const FORGE_DIR = path.join(ENGINE, "renderer");
const DL = path.join(ENGINE, "downloads");
const MANIFEST = path.join(ENGINE, "engine.json");

// Every install gets the engine InfinisX was tested on (audit 2026-10-14:
// the moving `neo` branch and "latest" Python gave two installs of one
// InfinisX build different engines). Upgrading = change these and re-test.
// Forge Neo: the commit in use since 2026-09-22 (read from that download's
// zip comment). GitHub's zip bytes are not guaranteed stable, so the check
// is the commit GitHub stamps in the zip, not a file hash.
const FORGE_SHA = "c21fa928cc08f2662fbcaedf3389858e7cf4a16f";
const FORGE_ZIP = `https://github.com/Haoming02/sd-webui-forge-classic/archive/${FORGE_SHA}.zip`;
// ADetailer (AGPL-3.0, Bing-su/adetailer) makes the Fix-ups real: it finds
// each face / hand after the render and repaints it. Pinned to a release tag
// so every install gets the same code (launch handoff: pin every revision).
const ADETAILER_TAG = "v26.2.0";
const ADETAILER_ZIP = `https://github.com/Bing-su/adetailer/archive/refs/tags/${ADETAILER_TAG}.zip`;
// Forge Couple (GPL-3.0, Haoming02 — the Forge Neo author): regional
// prompts, one prompt line per area of the picture. Pure Python, no
// installer. Pinned to a commit (launch handoff: pin every revision).
const COUPLE_SHA = "c7884e81623d7fcbf4c92e3aea14ced8b5b6aa74";
const COUPLE_ZIP = `https://github.com/Haoming02/sd-forge-couple/archive/${COUPLE_SHA}.zip`;
// Portable CPython 3.13 (Forge Neo checks for 3.13: modules/launch_utils.py),
// a fixed release asset with its SHA-256 — an uploaded file, so it never changes.
const PY_RELEASE = "20260901";
const PY_NAME = "cpython-3.13.15+20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz";
const PY_URL = `https://github.com/astral-sh/python-build-standalone/releases/download/${PY_RELEASE}/${encodeURIComponent(PY_NAME)}`;
const PY_SHA256 = "63d263ab0162f34a241a56dc5b283c22d6e131f5516117e6a921350c69ba7d4f";

const noPrepare = process.argv.includes("--no-prepare");
// Windows' own bsdtar: Git's GNU tar reads "F:\..." as a remote host.
const TAR = process.platform === "win32" ? path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe") : "tar";

const log = (...a) => console.log("[engine]", ...a);

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function download(url, dest) {
  if (await exists(dest)) {
    log("already downloaded", path.basename(dest));
    return;
  }
  log("downloading", url);
  const res = await fetch(url, { headers: { "User-Agent": "InfinisX-engine-builder" } });
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const tmp = dest + ".part";
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  await fs.rename(tmp, dest);
  const { size } = await fs.stat(dest);
  log(`saved ${path.basename(dest)} (${(size / 1e6).toFixed(1)} MB)`);
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    log("$", cmd, args.join(" "));
    const child = spawn(cmd, args, { stdio: "inherit", ...opts });
    child.on("error", reject);
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`))));
  });
}

async function sha256Of(file) {
  const h = createHash("sha256");
  await pipeline(createReadStream(file), h);
  return h.digest("hex");
}

/** A downloaded file that is not the pinned one is deleted, so a rerun fetches it again. */
async function verifySha256(file, want) {
  const got = await sha256Of(file);
  if (got !== want) {
    await fs.rm(file, { force: true });
    throw new Error(`${path.basename(file)}: SHA-256 ${got}, expected ${want} — download removed; run again`);
  }
}

/** The commit GitHub stamps in an archive zip's comment (end-of-central-directory record). */
async function zipCommit(file) {
  const fh = await fs.open(file, "r");
  try {
    const { size } = await fh.stat();
    const len = Math.min(size, 65_557);
    const buf = Buffer.alloc(len);
    await fh.read(buf, 0, len, size - len);
    for (let i = len - 22; i >= 0; i--) {
      if (buf.readUInt32LE(i) === 0x06054b50) return buf.toString("latin1", i + 22, i + 22 + buf.readUInt16LE(i + 20)).trim();
    }
    return "";
  } finally {
    await fh.close();
  }
}

async function installPython(manifest) {
  if (await exists(path.join(PY_DIR, "python.exe"))) {
    log("python already in place");
    return;
  }
  const tgz = path.join(DL, PY_NAME);
  await download(PY_URL, tgz);
  await verifySha256(tgz, PY_SHA256);
  // The tarball has a single top-level `python/` folder.
  await run(TAR, ["-xzf", tgz, "-C", ENGINE]);
  if (!(await exists(path.join(PY_DIR, "python.exe")))) throw new Error("python.exe missing after extract");
  manifest.python = { release: PY_RELEASE, asset: PY_NAME, sha256: PY_SHA256 };
}

async function installForge(manifest) {
  if (await exists(path.join(FORGE_DIR, "launch.py"))) {
    log("forge already in place");
    return;
  }
  const zip = path.join(DL, `forge-neo-${FORGE_SHA.slice(0, 7)}.zip`);
  await download(FORGE_ZIP, zip);
  const commit = await zipCommit(zip);
  if (commit !== FORGE_SHA) {
    await fs.rm(zip, { force: true });
    throw new Error(`Forge download is commit "${commit}", expected ${FORGE_SHA} — removed; run again`);
  }
  // Extracts to sd-webui-forge-classic-<sha>/; rename to renderer/.
  await run(TAR, ["-xf", zip, "-C", ENGINE]);
  await fs.rename(path.join(ENGINE, `sd-webui-forge-classic-${FORGE_SHA}`), FORGE_DIR);
  manifest.forge = { commit: FORGE_SHA, source: FORGE_ZIP, fetched: new Date().toISOString() };
}

/**
 * Forge Neo dropped the `--use-cpu` option that ADetailer reads when it
 * starts, so ADetailer failed to load ("Namespace has no attribute use_cpu").
 * One line, read it as "not set" instead. Recorded in engine.json as a
 * modification of the AGPL source (launch handoff §4: keep your patches).
 */
const ADETAILER_PATCHES = [
  {
    file: path.join("scripts", "!adetailer.py"),
    from: 'if "adetailer" in shared.cmd_opts.use_cpu:',
    to: 'if "adetailer" in getattr(shared.cmd_opts, "use_cpu", []):  # InfinisX: Forge Neo has no --use-cpu',
  },
  {
    // Same cause, at render time: the detector load patches an option Forge
    // Neo lacks, so every detailer pass failed silently (the render still
    // returned, and the infotext still named the ADetailer model).
    file: path.join("aaaaaa", "helper.py"),
    from: 'patch.object(cmd_opts, "disable_safe_unpickle", True),',
    to: 'patch.object(cmd_opts, "disable_safe_unpickle", True, create=True),  # InfinisX: absent in Forge Neo',
  },
];

async function patchADetailer(dest, manifest) {
  const applied = [];
  for (const p of ADETAILER_PATCHES) {
    const file = path.join(dest, p.file);
    const text = await fs.readFile(file, "utf8");
    if (text.includes(p.to)) {
      applied.push(p.file);
      continue;
    }
    if (!text.includes(p.from)) throw new Error(`adetailer patch no longer applies: ${p.file}`);
    await fs.writeFile(file, text.replace(p.from, p.to));
    applied.push(p.file);
    log("patched", p.file);
  }
  manifest.adetailer = { ...(manifest.adetailer ?? {}), patches: applied.map((f) => `${f}: tolerate a Forge option absent in Forge Neo (--use-cpu / --disable-safe-unpickle)`) };
  // AGPL-3.0 §5(a): a prominent notice that the files were modified, with the date.
  await fs.writeFile(
    path.join(dest, "INFINISX_MODIFICATIONS.txt"),
    [
      "ADetailer (https://github.com/Bing-su/adetailer, AGPL-3.0), tag " + ADETAILER_TAG + ", as installed by InfinisX.",
      "Modified by InfinisX on 2026-09-29 (each changed line ends with an \"# InfinisX:\" comment):",
      ...ADETAILER_PATCHES.map((p) => `  ${p.file}: ${p.from}  ->  ${p.to}`),
      "Complete corresponding source: https://github.com/crackvenom/infinisx-open (engine-sources).",
      "",
    ].join("\n")
  );
}

async function installADetailer(manifest) {
  const dest = path.join(FORGE_DIR, "extensions", "adetailer");
  if (await exists(path.join(dest, "install.py"))) {
    log("adetailer already in place");
    await patchADetailer(dest, manifest);
    return;
  }
  const zip = path.join(DL, `adetailer-${ADETAILER_TAG}.zip`);
  await download(ADETAILER_ZIP, zip);
  await run(TAR, ["-xf", zip, "-C", path.join(FORGE_DIR, "extensions")]);
  // Extracts to adetailer-<version>/ (the tag without its "v").
  await fs.rename(path.join(FORGE_DIR, "extensions", `adetailer-${ADETAILER_TAG.replace(/^v/, "")}`), dest);
  manifest.adetailer = { tag: ADETAILER_TAG, source: ADETAILER_ZIP, license: "AGPL-3.0", fetched: new Date().toISOString() };
  await patchADetailer(dest, manifest);
}

async function installForgeCouple(manifest) {
  const dest = path.join(FORGE_DIR, "extensions", "sd-forge-couple");
  if (await exists(path.join(dest, "scripts", "forge_couple.py"))) {
    log("forge couple already in place");
    return;
  }
  const zip = path.join(DL, `sd-forge-couple-${COUPLE_SHA.slice(0, 7)}.zip`);
  await download(COUPLE_ZIP, zip);
  await run(TAR, ["-xf", zip, "-C", path.join(FORGE_DIR, "extensions")]);
  // Extracts to sd-forge-couple-<sha>/.
  await fs.rename(path.join(FORGE_DIR, "extensions", `sd-forge-couple-${COUPLE_SHA}`), dest);
  manifest.forgeCouple = { commit: COUPLE_SHA, source: COUPLE_ZIP, license: "GPL-3.0", fetched: new Date().toISOString() };
}

/**
 * InfinisX's own engine extensions (engine-ext/, AGPL-3.0 like the engine):
 * the Flux ControlNet (2026-10-22). Copied on every build, tests left out, so
 * a newer one replaces the old; the app also tops an existing engine up at
 * start (lib/server/engine.ts syncInfinisxExtensions).
 */
async function installInfinisxExtensions(manifest) {
  const src = path.join(ROOT, "engine-ext");
  if (!(await exists(src))) return;
  for (const name of await fs.readdir(src)) {
    const from = path.join(src, name);
    if (!(await fs.stat(from)).isDirectory()) continue;
    const dest = path.join(FORGE_DIR, "extensions", name);
    await fs.rm(dest, { recursive: true, force: true });
    await fs.cp(from, dest, { recursive: true, filter: (p) => !/[\\/](tests|__pycache__)([\\/]|$)/.test(p.slice(from.length)) });
    const version = (await fs.readFile(path.join(from, "VERSION"), "utf8").catch(() => "")).trim();
    (manifest.infinisxExtensions ??= {})[name] = { version, license: "AGPL-3.0", installed: new Date().toISOString() };
    log("installed", name, version);
  }
}

/**
 * Let Forge install its own dependencies into the portable Python. `--exit`
 * stops it right after prepare_environment(). Gradio must be present for the
 * API even though we never show its UI. --skip-torch-cuda-test keeps the
 * build machine-independent (the real launch checks the GPU).
 */
async function prepare(manifest) {
  const python = path.join(PY_DIR, "python.exe");
  await run(python, ["-m", "pip", "install", "--upgrade", "pip", "--quiet"], { cwd: FORGE_DIR });
  await run(
    python,
    ["launch.py", "--skip-python-version-check", "--skip-torch-cuda-test", "--exit"],
    { cwd: FORGE_DIR, env: { ...process.env, PYTHONUNBUFFERED: "1", PIP_DISABLE_PIP_VERSION_CHECK: "1" } },
  );
  manifest.prepared = new Date().toISOString();
}

async function main() {
  await fs.mkdir(DL, { recursive: true });
  const manifest = (await exists(MANIFEST)) ? JSON.parse(await fs.readFile(MANIFEST, "utf8")) : {};
  await installPython(manifest);
  await installForge(manifest);
  await installADetailer(manifest);
  await installForgeCouple(manifest);
  await installInfinisxExtensions(manifest);
  await fs.writeFile(MANIFEST, JSON.stringify(manifest, null, 2));
  if (noPrepare) {
    log("skipping prepare (--no-prepare)");
  } else {
    await prepare(manifest);
    await fs.writeFile(MANIFEST, JSON.stringify(manifest, null, 2));
    // The app starts a prepared engine with --skip-prepare-environment only
    // when this marker exists (src/lib/server/engine.ts PREPARED_MARKER).
    await fs.writeFile(path.join(ENGINE, "prepared.ok"), `${manifest.prepared}\n`);
  }
  log("done →", ENGINE);
}

main().catch((e) => {
  console.error("[engine] FAILED:", e.message);
  process.exit(1);
});
