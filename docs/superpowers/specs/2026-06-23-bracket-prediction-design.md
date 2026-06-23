# Tournament Bracket Prediction Tab — Design Spec

## Overview

Add a "Tournament Bracket" tab to the existing frontend that shows the model's predicted bracket for the entire 2026 FIFA World Cup — from group stage through the final. A single backend endpoint simulates the full tournament and returns results in one response.

## Format

2026 FIFA World Cup: 48 teams, 12 groups of 4.

- **Group stage**: 36 matches (3 per group). Top 2 per group (24) + 8 best third-place teams (32 total) advance.
- **Knockout**: R32 (16 matches) → R16 (8) → QF (4) → SF (2) → Final (1). 31 knockout matches total.

## Backend

### New file: `src/data/groups.py`

Hardcoded dict of the official 2026 World Cup group draw (12 groups, 4 teams each).

### New endpoint: `POST /simulate-bracket`

Located in `src/api/main.py`.

**Logic:**

1. **Group stage simulation**: For each of the 36 group matches, call the existing feature-building + prediction pipeline. Determine match outcome: highest-probability result wins (home_win, draw, or away_win). Assign 3 pts for win, 1 for draw. Estimate goal difference from probabilities (e.g., scale win probability to approximate scoreline).
2. **Group standings**: Rank each group by points → goal difference → goals scored → alphabetical tiebreak.
3. **Third-place ranking**: Rank all 12 third-place teams by the same criteria. Top 8 advance.
4. **Bracket seeding**: Place the 32 advancing teams into R32 slots following a standard bracket template: group winners face third-place qualifiers, runners-up face runners-up from other groups. Seeding defined as a static mapping in `bracket.py` (e.g., 1A vs 3C/D/E, 1B vs 3A/F/G, etc.). If the official FIFA bracket is available, use it; otherwise use a balanced seeding that avoids same-group rematches in R32.
5. **Knockout simulation**: For each matchup, predict using the existing pipeline. Team with higher win probability advances. If draw is the most likely outcome, the team with the higher win% advances (no draws in knockouts).
6. **Repeat** through R16, QF, SF, Final.

**Response schema:**

```json
{
  "groups": {
    "A": {
      "standings": [
        {"team": "...", "pts": 9, "gd": 5, "gf": 7, "pos": 1, "advanced": true}
      ],
      "matches": [
        {"home": "...", "away": "...", "home_win": 0.55, "draw": 0.25, "away_win": 0.20, "result": "home_win"}
      ]
    }
  },
  "knockout": {
    "r32": [
      {"home": "...", "away": "...", "home_win": 0.6, "draw": 0.15, "away_win": 0.25, "winner": "..."}
    ],
    "r16": [...],
    "qf": [...],
    "sf": [...],
    "final": {"home": "...", "away": "...", "home_win": 0.5, "draw": 0.2, "away_win": 0.3, "winner": "..."}
  },
  "champion": "..."
}
```

### Simulation module: `src/simulation/bracket.py`

Encapsulates all tournament simulation logic (group sim, standings calc, bracket seeding, knockout sim). Called by the API endpoint. Depends on existing feature engineering and model prediction code.

Key functions:
- `simulate_group_stage(groups, matches_df, elo_ratings, model, ...)` → group standings + match results
- `rank_third_place(group_standings)` → sorted list, top 8 flagged
- `seed_bracket(group_standings, third_place)` → ordered R32 matchups
- `simulate_knockout(r32_matchups, matches_df, elo_ratings, model, ...)` → full knockout tree
- `simulate_tournament(...)` → orchestrates everything, returns full response

### Goal estimation

To compute goal difference for standings, estimate scorelines from probabilities:
- If home_win: home gets ~2 goals, away gets ~0-1 (scaled by margin of win probability)
- If draw: both get ~1 goal
- If away_win: mirror of home_win

Simple heuristic — exact goals don't need to be precise, just consistent for ranking.

## Frontend

### Tab system

Add a tab bar at the top of `static/index.html`:
- **Match Predictor** — existing content (default active)
- **Tournament Bracket** — new bracket view

Vanilla JS tab switching. Hide/show content divs.

### Bracket tab content

1. **"Simulate Tournament" button** — triggers `POST /simulate-bracket`, shows loading spinner.
2. **Group stage section** — collapsible, shows 12 group tables in a 3×4 or 4×3 grid.
   - Each table: Team | W | D | L | GD | Pts
   - Advancing teams highlighted (green border or background tint)
   - Third-place qualifiers marked differently from top-2
3. **Bracket tree** — classic tournament bracket below groups.
   - Left side: top 16 R32 seeds → R16 → QF → SF (flowing right)
   - Right side: bottom 16 R32 seeds → R16 → QF → SF (flowing left)
   - Center: Final + Champion highlighted
   - Each matchup card: both team names, win probabilities, winner highlighted with accent color
   - CSS grid/flexbox layout with border-based connecting lines
   - Horizontally scrollable on smaller screens

### Styling

- Consistent with existing dark theme (slate-900 background, slate-700 cards)
- Tailwind CSS (loaded via CDN, same as current)
- Winner: green/emerald accent. Loser: faded/dimmed.
- Bracket connecting lines: slate-500 borders
- Champion: gold/amber highlight

## Testing

- **`tests/test_bracket.py`**: Unit tests for simulation logic (group ranking, third-place ranking, knockout advancement, goal estimation)
- **`tests/test_api.py`**: Add test for `/simulate-bracket` endpoint (mock model, verify response shape)

## Files to create/modify

**New files:**
- `src/data/groups.py` — 2026 World Cup group draw data
- `src/simulation/bracket.py` — tournament simulation logic
- `tests/test_bracket.py` — simulation tests

**Modified files:**
- `src/api/main.py` — add `/simulate-bracket` endpoint
- `static/index.html` — add tab system + bracket tab UI
- `tests/test_api.py` — add bracket endpoint test

## Out of scope

- Live score updates / real match results
- User-editable brackets
- Multiple simulation runs / Monte Carlo
- Third/fourth place match
- Detailed match page per fixture
