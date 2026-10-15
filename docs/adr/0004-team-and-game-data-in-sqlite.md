# Team and Game Data in One SQLite Database

Team identity, people, season Rosters, and confirmed game events move from per-team JSON files to one SQLite database outside the repository. Each render reads a per-game snapshot written from the database.

**Context:** Team Config lives in `~/hockey/teams/<id>.json` with a single `roster` map from jersey number to name. The Buzz Light Beer game (2026-10-11) showed the limits. A roster has no season. Some players have no number, two players share #77, and one player wore #7 in the game but is listed as #77. A player's legal name (John David Mohr) differs from the display name (JD Mohr). Family links were added by hand. The game had no correct Scoresheet, so the scorers, assists, and penalties lived in an ad hoc file outside the pipeline. The roadmap needs questions across games: player and season reels, complete statistics, and the Scene Library.

**Decision:**
- **Store:** one SQLite file, `~/hockey/hockey.db`, outside the repository. Personal roster data stays out of git. The schema is versioned with forward migrations in the repository.
- **Tables:** `teams` (name, short name, colours, kits, logo and crest assets with their source), `people` (first, last, and display name), `family_links`, `roster_entries` (team, season, person, number or none, role), `games` (date, teams, focus team, game folder), `game_players` (number actually worn, with a flag when it differs from the roster), and `events` (goals with scorer and assists, penalties with infraction, classification, and minutes). Every name, number, and event records its source: Scoresheet read, override, or user.
- **Seasons:** each roster belongs to a season. A game uses the roster for its date.
- **Number conflicts:** the number worn in the game wins. The pipeline flags each difference from the season roster for review.
- **Renders:** before graphics, the pipeline writes `game_data.json` into the Game Folder from the database. Graphics and statistics read only that snapshot, so a past render can be repeated after a roster changes.
- **Edits:** the review page (hhg-3r5.9) and a small CLI write corrections to the database. Hand edits of JSON are no longer the source of truth.
- **Import:** the existing `teams/*.json` files and confirmed game data are imported once. The original files are kept.

**Considered:** Splitting the JSON files into teams, players, season rosters, and per-game events. Rejected: features across games would read and join many files, and JSON cannot enforce one number per player per game or that every goal has a team. Repository files. Rejected: personal data would enter git. A hosted database (Supabase or Postgres). Deferred: no server is needed for one user on one Mac, and the same schema can move to Postgres if the pipeline becomes a product for other teams.

**Consequences:** `recap_options.py`, the Scoresheet stage, and the graphics stage change to read from the database and the snapshot. Tests use a temporary database. Backups of `~/hockey/hockey.db` become the user's responsibility, like the NAS footage. The Black Ice graphics port (hhg-3r5.106) depends on this model for names, numbers, periods, and statistics.
