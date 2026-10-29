# InfinisX — open source

InfinisX (by Hawktagon LLC) is a Windows app that turns an artist's drawing
into a finished render on their own graphics card. This repository holds the parts of it
that are open source:

- **[AI Mark Checker](https://crackvenom.github.io/infinisx-open/checker/)**
  (`checker/`): a free page that shows whether a picture carries AI marks.
  It reads Content Credentials (C2PA), the invisible TrustMark watermark,
  and China's AIGC label. It runs entirely in your browser and never
  uploads the picture. Every picture InfinisX saves carries all three marks,
  as AI-transparency laws ask (EU AI Act Art. 50, California AI
  Transparency Act, China GB 45438, South Korea AI Basic Act).
- **AI Mark Checker for code** (`api/`): the same check as a command-line
  tool and JavaScript module (`node check-ai-marks.mjs picture.png` or a URL
  → JSON), no website needed. Feedback on the checker:
  [issues](https://github.com/crackvenom/infinisx-open/issues).
- **Render-engine source** (`engine-sources/`): the complete corresponding
  source for the free-software engine InfinisX installs and runs, as the
  GNU AGPL-3.0 and GPL-3.0 require. That includes InfinisX's own engine
  extension and its two compatibility fixes to ADetailer.

The InfinisX application itself (its interface) is separate, proprietary
software. It talks to the engine only through the engine's local web API.

## Licences

- `engine-sources/infinisx-flux-controlnet/`: GNU AGPL-3.0 (LICENSE in the folder).
- `engine-sources/adetailer-infinisx.patch`: changes to ADetailer, GNU AGPL-3.0.
- `engine-sources/build-engine.mjs`: the script InfinisX uses to install
  the engine. Published so the engine can be rebuilt exactly.
- `checker/index.html`, `checker/checker.js`, `api/`: MIT (Hawktagon LLC). `checker/vendor/*`: each
  library's own licence (MIT / ISC), in its folder.
- The upstream archives in each release keep their own licences.

No warranty. See each licence.
