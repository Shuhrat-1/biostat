# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are technically-literate biologists, agronomists, and ecologists — people comfortable with programming and statistics at a beginner-to-intermediate level. They arrive with their own experimental or field data (often messy CSV exports from Excel) and need to run standard statistical analyses (descriptive stats, hypothesis tests, ANOVA, regression) and get results they can trust and cite. Because the audience is technically prepared, the product does not simplify away complexity — it exposes method, formula, backend, and computation time rather than hiding them.

## Product Purpose

BioStat lets users upload their own data and run statistical calculations entirely in their own browser (via Pyodide/WebAssembly), with an explicit tiered-compute model for scaling beyond the browser (local GPU via WebGPU, then bring-your-own-key cloud compute) as data size demands. Success means: a user goes from a raw, possibly defective CSV to a correct, trustworthy statistical result — with any data-quality issues surfaced and attached to that result — without installing anything and without their data ever leaving their machine (except for the cloud tier, which runs through the user's own API key, not BioStat's infrastructure).

## Positioning

BioStat is scoped to one domain and one reference text — *Statistical Methods in Biology: Design and Analysis of Experiments and Regression* (Welham, Gezan, Clark, Mead; CRC Press, 2015) — rather than being a general-purpose statistics calculator. That scoping gives it a fixed method roadmap, a numerical reference to validate against (SciPy/statsmodels), and a defined audience (biologists, agronomists, ecologists). Its mechanism — real computation running client-side (or on infrastructure the user pays for directly with their own key) instead of on BioStat's servers — is what a generic hosted calculator could not truthfully copy: near-zero server cost, and data that never transits BioStat's backend for the browser and local-GPU tiers.

## Operating Context

- Core statistical functions are pure Python (`numpy`, `scipy`, `pandas`, `statsmodels`), developed and tested independently of the UI, then executed unmodified in-browser via Pyodide (WASM).
- Data enters as user-uploaded CSV; a two-tier validation model runs a fast sample scan (seconds, always) and an optional full per-cell scan (minutes, on demand or when the sample finds problems). Users may skip the full scan; the fast scan still runs unless explicitly disabled.
- Any data-quality warnings from validation travel with the computed result, including into exports — the result is never presented without its data-quality context.
- Compute backend (browser CPU / local WebGPU / user's own cloud API key such as Modal or RunPod) is chosen manually per calculation by the user, not auto-routed — this audience wants control, not automation deciding for them.
- UI is localized in RU/EN/PT from the start; number-locale parsing (decimal separators in uploaded data) is handled separately from UI language.
- Method coverage rolls out incrementally by book chapter (descriptive stats → hypothesis tests → ANOVA → regression → GLM/mixed models), each phase shipping a complete thin vertical slice (upload → compute → result) rather than building horizontal layers.

## Capabilities and Constraints

- MVP method scope (chapters 1–5): descriptive statistics and confidence intervals, hypothesis tests (t-tests + nonparametric equivalents), one-way ANOVA with multiple comparisons, simple linear regression with residual diagnostics.
- The diagnostics module (normality, homoscedasticity, residual independence, Q-Q plots) is treated as a priority feature, not an afterthought — it's what distinguishes a serious tool from a bare p-value calculator.
- ANOVA sum-of-squares types and multiple-comparison conventions differ between GenStat/R and Python; when ANOVA ships, the convention being followed must be stated explicitly rather than left implicit.
- Browser-tier (Pyodide) validation is capped by tab memory (~2–4GB); full per-cell validation on gigabyte-scale files is explicitly a local/cloud-tier capability, not a browser one.
- Cloud tier is bring-your-own-compute only: users supply their own API key (Modal, RunPod, etc.) and pay their own provider directly; BioStat does not run or bill for GPU compute. API keys must never be stored server-side in the clear — client-side/session storage only, passed through only for the duration of a request.
- No backend server, database, or accounts at this stage (later phase, not-yet-committed).

## Brand Commitments

- Name: BioStat (domain: biostat.app). Existing brand assets already shipped: favicon, OG/social preview image, meta description and keywords establishing the "statistics for biology and agronomy, in your browser" framing.
- Tagline framing already live: "Statistics for biology and agronomy, in your browser" / "your data never leaves your computer" — both are existing public claims, not proposals.

## Evidence on Hand

- `docs/PROJECT_SPEC.md` — full architecture and rationale document (source of most facts above).
- Core statistics modules are pytest-covered against SciPy/statsmodels as the numerical reference (`tests/`, `core/stats/`).
- No customer testimonials, case studies, or usage data exist yet — future work must not fabricate these.

## Product Principles

1. Vertical slices over horizontal layers — ship a complete thin path (upload → compute → export) before widening it.
2. Transparency over simplification — for this audience, showing method/formula/backend/timing is a feature, not clutter to hide.
3. Own the result — data-quality context is inseparable from the computed result, in-app and in every export.
4. Manual control over auto-routing — the user picks the compute backend per calculation; the system informs, it doesn't decide.
5. One model per concern — one report schema (result + data-quality), one JS↔Python bridge, one i18n system; no parallel logic for different outputs.

## Accessibility & Inclusion

WCAG AA is a required standard. The data-quality report already commits to colorblind-safe presentation: verdicts pair icon + text + color together (not color alone), so the design must hold up in black-and-white export as well as for colorblind users.
