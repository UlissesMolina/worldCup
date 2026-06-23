# src/data/groups.py
"""2026 FIFA World Cup group draw — 48 teams in 12 groups of 4."""

# Team names match the historical dataset (data/processed/matches.parquet).
# Mapping notes:
#   FIFA "Czechia" → dataset "Czech Republic"
#   FIFA "Türkiye" → dataset "Turkey"
#   FIFA "Curaçao" → dataset "Curacao" (may not exist; use as-is)
#   FIFA "Cabo Verde" → dataset "Cape Verde"
#   FIFA "Côte d'Ivoire" → dataset "Ivory Coast"
#   FIFA "Bosnia-Herzegovina" → dataset "Bosnia and Herzegovina"

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Fixed R32 pairings (runner-up vs runner-up, and winner vs runner-up).
# These 8 matches don't involve third-place teams.
FIXED_R32 = [
    ("2A", "2B"),  # Match 73
    ("1C", "2F"),  # Match 76
    ("1F", "2C"),  # Match 75
    ("2E", "2I"),  # Match 78
    ("1H", "2J"),  # Match 84
    ("1J", "2H"),  # Match 86 (Argentina side)
    ("2K", "2L"),  # Match 83
    ("2D", "2G"),  # Match 88
]

# Variable R32 pairings: group winners vs third-place teams.
# Each group winner has a pool of valid third-place opponents (no same-group clash).
# Key = group winner's group letter, Value = list of valid 3rd-place source groups.
WINNER_VS_THIRD_POOLS = {
    "A": ["C", "E", "F", "H", "I"],       # Match 79 (Mexico side)
    "B": ["E", "F", "G", "I", "J"],       # Match 85
    "D": ["B", "E", "F", "I", "J"],       # Match 81 (USA side)
    "E": ["A", "B", "C", "D", "F"],       # Match 74 (Germany side)
    "G": ["A", "E", "H", "I", "J"],       # Match 82
    "I": ["C", "D", "F", "G", "H"],       # Match 77
    "K": ["D", "E", "I", "J", "L"],       # Match 87
    "L": ["E", "H", "I", "J", "K"],       # Match 80
}

# R32 match ordering for bracket layout (left half then right half).
# Each entry is an index into the final R32 matchups list.
# Left bracket: matches 0-7, Right bracket: matches 8-15
# R16 pairings: winner of match 0 vs winner of match 1, etc.
# QF: winner of R16-0 vs winner of R16-1, etc.
# SF: winner of QF-0 vs winner of QF-1, etc.
# Final: winner of SF-0 vs winner of SF-1
BRACKET_STRUCTURE = {
    "r32_left": [0, 1, 2, 3, 4, 5, 6, 7],
    "r32_right": [8, 9, 10, 11, 12, 13, 14, 15],
    "r16": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    "qf": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "sf": [(0, 1), (2, 3)],
    "final": (0, 1),
}
