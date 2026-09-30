# Resolve Edition Automation Architecture

**Status: Superseded in part (2026-09-29).** Spec 002 renders graphics with Remotion and assembles video with ffmpeg; Resolve is an optional finisher only.

We build editing automation in three layers, gated by Resolve edition, with a one-time Studio purchase unlocking the top layer.

**Context:** Resolve scripting (write access to timelines) is a Resolve Studio-only feature. The free edition can read timelines and export/interchange files but cannot be driven programmatically. Two MCP servers exist for Resolve (samuelgursky/davinci-resolve-mcp, 341 tools, live-tested; and an unofficial one with interchange engines for Free), both requiring Studio 18.5+. Studio is $295 one-time with historically free major upgrades, and adds Speed Warp (clean slo-mo <50%) and UltraNR (noisy-rink denoising) that are directly relevant to our output quality.

**Decision:** Run Free through Stage 1 (validation and ML development). At Stage 2, purchase Studio once, then operate in three tiers: (1) in-Resolve Python scripts for reel assembly and marker expansion (works now, lowest friction), (2) FCPXML/EDL/CSV interchange for any tool that needs to run outside Resolve, (3) live MCP control only for operations that genuinely need conversational/natural-language driving (e.g. "apply color grade to all keepers"). Do not build MCP-first — build script-first, wrap in MCP later.

**Consequences:** The verification flow and reel assembly are already implemented via in-Resolve scripts and work the day Studio is activated. The $295 is a single gate, not recurring. Free-tier users (including us pre-Stage-2) can still run detection and generate interchange files. Speed Warp becomes available for our 50%-speed replays, replacing Resolve's lower-quality optical-flow retiming.
