"""Actual 2026 FIFA World Cup match results."""

# Each match: home, away, home_goals, away_goals
# Team names match groups.py conventions.

ACTUAL_RESULTS = {
    "A": [
        {"home": "Mexico", "away": "South Africa", "home_goals": 2, "away_goals": 0},
        {"home": "South Korea", "away": "Czech Republic", "home_goals": 2, "away_goals": 1},
        {"home": "Czech Republic", "away": "South Africa", "home_goals": 1, "away_goals": 1},
        {"home": "Mexico", "away": "South Korea", "home_goals": 1, "away_goals": 0},
        {"home": "Mexico", "away": "Czech Republic", "home_goals": 3, "away_goals": 0},
        {"home": "South Africa", "away": "South Korea", "home_goals": 1, "away_goals": 0},
    ],
    "B": [
        {"home": "Canada", "away": "Bosnia and Herzegovina", "home_goals": 1, "away_goals": 1},
        {"home": "Switzerland", "away": "Qatar", "home_goals": 1, "away_goals": 1},
        {"home": "Switzerland", "away": "Bosnia and Herzegovina", "home_goals": 4, "away_goals": 1},
        {"home": "Canada", "away": "Qatar", "home_goals": 6, "away_goals": 0},
        {"home": "Switzerland", "away": "Canada", "home_goals": 2, "away_goals": 1},
        {"home": "Bosnia and Herzegovina", "away": "Qatar", "home_goals": 3, "away_goals": 1},
    ],
    "C": [
        {"home": "Brazil", "away": "Morocco", "home_goals": 1, "away_goals": 1},
        {"home": "Scotland", "away": "Haiti", "home_goals": 1, "away_goals": 0},
        {"home": "Morocco", "away": "Scotland", "home_goals": 1, "away_goals": 0},
        {"home": "Brazil", "away": "Haiti", "home_goals": 3, "away_goals": 0},
        {"home": "Brazil", "away": "Scotland", "home_goals": 3, "away_goals": 0},
        {"home": "Morocco", "away": "Haiti", "home_goals": 4, "away_goals": 2},
    ],
    "D": [
        {"home": "United States", "away": "Paraguay", "home_goals": 4, "away_goals": 1},
        {"home": "Australia", "away": "Turkey", "home_goals": 2, "away_goals": 0},
        {"home": "United States", "away": "Australia", "home_goals": 2, "away_goals": 0},
        {"home": "Turkey", "away": "Paraguay", "home_goals": 0, "away_goals": 1},
        {"home": "Turkey", "away": "United States", "home_goals": 3, "away_goals": 2},
        {"home": "Paraguay", "away": "Australia", "home_goals": 0, "away_goals": 0},
    ],
    "E": [
        {"home": "Germany", "away": "Curacao", "home_goals": 7, "away_goals": 1},
        {"home": "Ivory Coast", "away": "Ecuador", "home_goals": 1, "away_goals": 0},
        {"home": "Germany", "away": "Ivory Coast", "home_goals": 2, "away_goals": 1},
        {"home": "Ecuador", "away": "Curacao", "home_goals": 0, "away_goals": 0},
        {"home": "Curacao", "away": "Ivory Coast", "home_goals": 0, "away_goals": 2},
        {"home": "Ecuador", "away": "Germany", "home_goals": 2, "away_goals": 1},
    ],
    "F": [
        {"home": "Netherlands", "away": "Japan", "home_goals": 2, "away_goals": 2},
        {"home": "Sweden", "away": "Tunisia", "home_goals": 5, "away_goals": 1},
        {"home": "Netherlands", "away": "Sweden", "home_goals": 5, "away_goals": 1},
        {"home": "Japan", "away": "Tunisia", "home_goals": 4, "away_goals": 0},
        {"home": "Japan", "away": "Sweden", "home_goals": 1, "away_goals": 1},
        {"home": "Netherlands", "away": "Tunisia", "home_goals": 3, "away_goals": 1},
    ],
    "G": [
        {"home": "Belgium", "away": "Egypt", "home_goals": 1, "away_goals": 1},
        {"home": "Iran", "away": "New Zealand", "home_goals": 2, "away_goals": 2},
        {"home": "Belgium", "away": "Iran", "home_goals": 0, "away_goals": 0},
        {"home": "Egypt", "away": "New Zealand", "home_goals": 3, "away_goals": 1},
        {"home": "Egypt", "away": "Iran", "home_goals": 1, "away_goals": 1},
        {"home": "Belgium", "away": "New Zealand", "home_goals": 5, "away_goals": 1},
    ],
    "H": [
        {"home": "Spain", "away": "Cape Verde", "home_goals": 0, "away_goals": 0},
        {"home": "Saudi Arabia", "away": "Uruguay", "home_goals": 1, "away_goals": 1},
        {"home": "Spain", "away": "Saudi Arabia", "home_goals": 4, "away_goals": 0},
        {"home": "Uruguay", "away": "Cape Verde", "home_goals": 2, "away_goals": 2},
        {"home": "Cape Verde", "away": "Saudi Arabia", "home_goals": 0, "away_goals": 0},
        {"home": "Spain", "away": "Uruguay", "home_goals": 1, "away_goals": 0},
    ],
    "I": [
        {"home": "France", "away": "Senegal", "home_goals": 3, "away_goals": 1},
        {"home": "Norway", "away": "Iraq", "home_goals": 4, "away_goals": 1},
        {"home": "France", "away": "Iraq", "home_goals": 3, "away_goals": 0},
        {"home": "Norway", "away": "Senegal", "home_goals": 3, "away_goals": 2},
        {"home": "Norway", "away": "France", "home_goals": 4, "away_goals": 1},
        {"home": "Senegal", "away": "Iraq", "home_goals": 5, "away_goals": 0},
    ],
}

# Knockout results — add as rounds are played.
# Format: list of {"home": ..., "away": ..., "home_goals": ..., "away_goals": ...}
KNOCKOUT_RESULTS = {}
