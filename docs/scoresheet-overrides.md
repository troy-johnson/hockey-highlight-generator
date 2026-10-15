# Scoresheet read overrides

The Scoresheet stage keeps the reader output in `game_sheet_read.json`.
It applies `game_sheet_overrides.json` on top and writes `game_sheet.json`.
All downstream stages read `game_sheet.json`.

- The stage reads the sheet again only when the sheet photo or PDF changes.
- A change to the overrides file reruns the stage without a new read.
- An invalid overrides file stops the stage. The stage does not write `game_sheet.json`.

## File format

```json
{
  "schema_version": 1,
  "teams": {"home": "Salty Boys"},
  "rosters": {"home": {"73": "J. Tripper", "12": null}},
  "goals": {
    "away": {
      "2": {"was": {"type": "PP"}, "type": "SH", "time": "1:10"},
      "8": {"per": "3", "time": "4:00", "scorer": "00", "assists": ["47"], "type": "ES"},
      "5": null
    }
  },
  "penalties": {"home": {"1": {"per": "2", "player": "73", "infraction": "TRIP", "minutes": "2"}}},
  "shots_on_goal": {"home": [10, 4, 6], "away": [10, 13, 9]},
  "goalkeeping": {"home": [{"player": "59", "name": "Joe Shiflett", "saves": 24}]},
  "score_by_periods": {"home": [0, 0, 0], "away": [1, 3, 3]}
}
```

## Rules

- `rosters`: a number maps to a name. `null` removes the number.
- `goals` and `penalties`: the key is the 1-based row number in the read.
  - A key for an existing row changes only the given fields.
  - A key higher than the read row count adds a row.
  - `null` removes the row.
  - Changed and added rows get `status: ok`, no reasons, and `source: override`.
  - The stage recomputes `time_s` from `time`.
- `was` is optional. It lists read values that the override expects.
  If the read row differs, the stage skips the override and adds a flag.
  Use `was` so that a new read cannot get a wrong correction.
- `shots_on_goal`, `score_by_periods`, and `goalkeeping` replace the read values.
- Goalie rows accept `player`, `name`, `saves`, `ga`, and `shots`.
  The FINAL card derives missing values from goals and shots when it can.
- Unknown top-level keys and other schema versions stop the stage.

## More fields and rules

- `date` (text) and `final_score` (`{"home": 0, "away": 7}`) replace the read values. GameSheet PDFs write these; photo reads do not.
- `dismiss_flags` is a list of reader flag texts that you checked. Each matching flag is removed. A text that does not match a flag adds a new "not dismissed" flag.
- A new row (index above the read rows) must have `per`, `time` and `scorer` for a goal, or `per`, `player` and `infraction` for a penalty. Otherwise the stage stops.
- An edit does not set a goal type. An empty type stays unknown.
- An invalid overrides file stops the stage and leaves the previous `game_sheet.json` in place.
- The saved read records `READER_VERSION` in `scoresheet.py`. A new reader version reads the sheet again. To force a new read, delete `game_sheet_read.json`.
- A row edit without `was` applies to the new read too. Add `was` when the edit depends on the read value.
