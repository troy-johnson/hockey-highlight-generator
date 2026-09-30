# Recall-Optimized Selection with Reserves Tier

**Status: Superseded by [ADR-0003](0003-game-sheet-first-selection.md) (2026-09-29).**

The ML pipeline selects ~5–6 min of footage for verification, biased to include borderline plays rather than risk missing a goal, with near-misses held on a second timeline for recovery.

**Context:** When ML only labeled events, missing one cost nothing — the human watched everything. Once ML culls, a false negative (missed goal) forces the user to scrub an hour of raw footage, destroying trust in the system. A false positive (junk in the reel) costs one Delete keystroke during verification. The cost asymmetry is ~10:1 against false negatives.

**Decision:** Train and tune the classifier for recall on goals first, precision second. Always include all detected goals regardless of confidence; include penalties and high-scoring non-scoring plays to fill the ~330–360s budget; drop low-scoring blue events first. Accept that ~10–20% of the Verify Timeline will be deleted during review — that deletion time is part of the ~3 min verification budget. Place the dropped candidates (the next ~5 min by score) on a Reserves timeline in the same Resolve project so recovery is a drag-and-drop, not a search.

**Considered:** Tight precision-first selection (~5% junk, risk ~5% missed goals, no reserves needed). Rejected: a single missed highlight per game is unacceptable for a highlight reel, and without reserves the recovery cost dominates any time saved by tighter selection.

**Consequences:** The selection target is 330–360s of base footage yielding ~7–9 min final after replays. The verification tooling must log every keep/drop/angle-correction decision — this is the training data that improves recall over successive games. The Reserves timeline adds ~5 min of dead weight to the Resolve project file, negligible relative to the source footage.
