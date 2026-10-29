# Render-engine corresponding source

InfinisX installs the render engine on first start (`build-engine.mjs`) and
runs it as a separate program on the user's PC (local web API on
127.0.0.1). Below is the exact source of every free-software part of that
engine, for each InfinisX release. Copies of the upstream archives are
attached to each [release](https://github.com/crackvenom/infinisx-open/releases),
so the source stays available even if an upstream repository changes.

## InfinisX 1.1.0-beta.19 (2026-10-29)

| Component | Exact version | Licence | Changed by InfinisX? | Source |
| :--- | :--- | :--- | :--- | :--- |
| Forge Neo (Haoming02/sd-webui-forge-classic) | commit `c21fa928cc08f2662fbcaedf3389858e7cf4a16f` | GNU AGPL-3.0 | No (launch options and settings only) | release asset `forge-neo-c21fa92.zip`; upstream: https://github.com/Haoming02/sd-webui-forge-classic/tree/c21fa928cc08f2662fbcaedf3389858e7cf4a16f |
| ADetailer (Bing-su) | tag `v26.2.0` | GNU AGPL-3.0 | **Yes**: two one-line compatibility fixes (2026-09-29), `adetailer-infinisx.patch` | release assets `adetailer-v26.2.0.zip` (original) and `adetailer-v26.2.0-infinisx-patched.zip` |
| Forge Couple (Haoming02) | commit `c7884e81623d7fcbf4c92e3aea14ced8b5b6aa74` | GNU GPL-3.0 | No | release asset `sd-forge-couple-c7884e8.zip` |
| InfinisX Flux ControlNet | `infinisx-flux-controlnet/` (VERSION 2026-10-29.1) | GNU AGPL-3.0 | InfinisX's own | this folder |
| Install script | `build-engine.mjs` | — | InfinisX's own | this folder |

SHA-256 of the release assets:

```
a5c0ab6764cc01d4e7f84b8cc5e03ad973b55fa7b8d9e8015f6d408251c210d5  forge-neo-c21fa92.zip
b24ddf238ea31de24df5d36333cd8b2cd565109e785fe72f6cb7d5c36c3b1c25  adetailer-v26.2.0.zip
01a805e197aac8d32f19cc7f8ffea0104e5dfbd5d848431b74dae67c35cb5c63  adetailer-v26.2.0-infinisx-patched.zip
0beaff9d5c86c8bf5d0ea0fd759a0472b9ca65ab9337fc1e1f05effb76fe57f0  sd-forge-couple-c7884e8.zip
```

## Rebuilding the engine yourself

1. Download python-build-standalone CPython 3.13.15
   (`20260901`, Windows x86-64 install_only_stripped).
2. Unpack Forge Neo, ADetailer (patched) and Forge Couple as
   `build-engine.mjs` does: Forge Neo as `renderer/`, the extensions in
   `renderer/extensions/`, plus `infinisx-flux-controlnet`.
3. Run `python launch.py --api`. Forge installs PyTorch and its Python
   packages on first start (from `renderer/requirements.txt`; those
   packages carry their own licences).

The engine is fully usable on its own, without InfinisX.
