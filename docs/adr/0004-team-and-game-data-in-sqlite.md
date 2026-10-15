# Team and Game Data in One SQLite Database

Team identity, people, Season Rosters, and Game Sheet events move from per-team JSON files to one SQLite database outside the repository. Each render reads a per-game snapshot written from the database.

**Context:** Team Config lives in `~/hockey/teams/<id>.json` with a single `roster` map from jersey number to name. The Buzz Light Beer game (2026-10-11) showed the limits. A Roster has no Season. Some players have no number, two players share #77, and one player wore #7 in the game but is listed as #77. A player's legal name (John David Mohr) differs from the display name (JD Mohr). Family links were added by hand. The game had no correct Scoresheet, so the scorers, assists, and penalties lived in an ad hoc file outside the pipeline. The roadmap needs questions across games: player and season reels, complete statistics, and the Scene Library.

**Decision:**
- **Store:** one SQLite file, `~/hockey/hockey.db`, outside the repository. Personal roster data stays out of git. The schema is versioned with forward migrations in the repository.
- **Tables:** `teams` (name, short name, colors, kits, and logo and crest assets, each with its source), `people` (first, last, and display name), `family_links`, `roster_entries` (team, season, person, number or none, role), `games` (date, teams, focus team, game folder), `game_players` (number actually worn, with a flag when it differs from the roster), and `events` (goals with scorer and assists, penalties with infraction, classification, and minutes). Every name, number, and event records its source, which starts with sheet, override, user, or import.
- **Seasons:** each Roster belongs to a Season, which starts on 1 August. A game uses its Season's Roster, or the latest earlier Roster with a flag.
- **People:** one person across teams and Seasons. Two records with the same name on different teams are flagged; after review, `merge-people` makes them one.
- **Unknown names:** an import stops when a name is not on the Roster and shows the closest Roster names. A team-only penalty (bench minor) has no player and counts for team PIM.
- **Jerseys:** 08 and 8 are the same number; the printed label keeps 08.
- **Penalties:** each penalty has one class: minor, bench minor, double minor, major, misconduct, game misconduct, match, or penalty shot (USA Hockey 401; NHL Section 4). A game names its rulebook, USA Hockey by default. A penalty without minutes gets the minutes that rulebook puts in the records, and the rule is stored with it. A game misconduct records 10 minutes in both rulebooks; a match penalty records 10 (USA Hockey) or 15 (NHL). Given minutes that differ from the rulebook are flagged.
- **Number conflicts:** the number worn in the game wins. The pipeline flags each difference from the season roster for review.
- **Renders:** before graphics, the pipeline writes `game_data.json` into the Game Folder from the database. Graphics and statistics read only that snapshot, so a past render can be repeated after a roster changes.
- **Edits:** for now, a game's import file is the record for its events: corrections go into that file, then a reimport replaces the game's events in one transaction. `merge-people` joins two records of one person. The review page (hhg-3r5.9) will later write corrections to the database directly.
- **Import:** the existing `teams/*.json` files and confirmed game data are imported once. The original files are kept.

**Considered:** Splitting the JSON files into teams, players, season rosters, and per-game events. Rejected: features across games would read and join many files, and JSON cannot enforce one number per player per game or that every goal has a team. Repository files. Rejected: personal data would enter git. A hosted database (Supabase or Postgres). Deferred: no server is needed for one user on one Mac, and the same schema can move to Postgres if the pipeline becomes a product for other teams.

**Consequences:** `recap_options.py`, the Scoresheet stage, and the graphics stage change to read from the database and the snapshot. Tests use a temporary database. Backups of `~/hockey/hockey.db` become the user's responsibility, like the NAS footage. The Black Ice graphics port (hhg-3r5.106) depends on this model for names, numbers, periods, and statistics.
