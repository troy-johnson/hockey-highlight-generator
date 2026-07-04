# Workflow State — hockey-highlight-generator

> Resumability pointer only. Durable task state lives in Beads (`bd ready`,
> `bd show <id>`). If this file and Beads disagree, Beads wins.

Last updated: 2026-07-04
Current phase: V3 Phase 1 — complete, pending manual validation
Current checkout: local `main` — V3 Phase 1 plus arch-review follow-ups, ahead of `origin/main` (not pushed)
Current Gate: Phase 1 field validation (real footage + Resolve)
Blockers: hhg-38a.9 (real-footage smoke test) and hhg-38a.10 (Resolve verify) still open

## Active Artifacts

- spec: `docs/specs/001-v3-highlight-pipeline-2026-04-25.md`
- plan: `docs/plans/2026-04-25-v3-highlight-pipeline.md`
- reconciliation branch (pushed): `v3-phase1-reconcile` (== 70a711c)
- new scripts: `v3/scripts/discover.py`, `v3/scripts/gopro_meta.py`,
  `v3/resolve_scripts/compile_reel.py`

## Status

- V3 Phase 1 implementation merged onto local `main` at 70a711c (2026-07-01).
- 81-test pytest suite passes via `.venv/bin/python -m pytest -q`.
- `main` is NOT pushed; open a PR from `v3-phase1-reconcile` to merge (hhg-bz8).
- Arch-review follow-up fixes (deps, loud fallback, this doc) live on branch
  `fix/arch-review-followups`, also unpushed.

## Next atomic steps

1. hhg-38a.9 — run folder mode on a real GoPro game folder end to end.
2. hhg-38a.10 — verify `compile_reel.py` inside DaVinci Resolve (confirm the
   `CreateMultiCamClip` signature against the installed Resolve version).
3. hhg-bz8 — open PR from `v3-phase1-reconcile` and merge to `origin/main`.

## Not Started

- V3 Phase 2 (epic hhg-3hk) — gated behind the HockeyAI spike (hhg-3hk.1),
  which in turn depends on the real-footage smoke test (hhg-38a.9).
