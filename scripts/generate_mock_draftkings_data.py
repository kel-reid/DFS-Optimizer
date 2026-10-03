#!/usr/bin/env python3
"""
Generates realistic mock DraftKings NFL Classic data for offline testing:
  - data/players/DKSalaries-mock.csv
  - data/templates/DKEntries-mock.csv (150 reserved entry rows)
"""

import csv
from pathlib import Path

PLAYERS_DIR = Path("data/players")
TEMPLATES_DIR = Path("data/templates")

MOCK_DK_PLAYERS = [
    # Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame
    ("QB", "Josh Allen (1001)", "Josh Allen", "1001", "QB", "8200", "NE@BUF 10/04/2026 01:00PM ET", "BUF", "28.5"),
    ("QB", "Patrick Mahomes (1002)", "Patrick Mahomes", "1002", "QB", "7800", "KC@DEN 10/04/2026 04:25PM ET", "KC", "24.2"),
    ("QB", "Lamar Jackson (1003)", "Lamar Jackson", "1003", "QB", "8000", "BAL@CIN 10/04/2026 01:00PM ET", "BAL", "26.1"),
    ("QB", "Brock Purdy (1004)", "Brock Purdy", "1004", "QB", "6600", "SF@LAR 10/04/2026 04:05PM ET", "SF", "21.4"),
    ("QB", "Jordan Love (1005)", "Jordan Love", "1005", "QB", "6400", "GB@CHI 10/04/2026 01:00PM ET", "GB", "20.1"),
    ("QB", "Dak Prescott (1006)", "Dak Prescott", "1006", "QB", "6700", "DAL@PHI 10/04/2026 08:20PM ET", "DAL", "22.3"),
    ("QB", "Jalen Hurts (1007)", "Jalen Hurts", "1007", "QB", "7500", "DAL@PHI 10/04/2026 08:20PM ET", "PHI", "23.9"),
    ("QB", "Joe Burrow (1008)", "Joe Burrow", "1008", "QB", "7000", "BAL@CIN 10/04/2026 01:00PM ET", "CIN", "21.8"),
    ("QB", "Kirk Cousins (1009)", "Kirk Cousins", "1009", "QB", "6000", "LV@LAC 10/04/2026 04:05PM ET", "LV", "18.5"),
    ("QB", "Geno Smith (1010)", "Geno Smith", "1010", "QB", "5800", "NYJ@MIA 10/04/2026 01:00PM ET", "NYJ", "17.9"),

    # RBs
    ("RB", "Christian McCaffrey (2001)", "Christian McCaffrey", "2001", "RB/FLEX", "9000", "SF@LAR 10/04/2026 04:05PM ET", "SF", "24.5"),
    ("RB", "Derrick Henry (2002)", "Derrick Henry", "2002", "RB/FLEX", "8200", "BAL@CIN 10/04/2026 01:00PM ET", "BAL", "21.3"),
    ("RB", "Breece Hall (2003)", "Breece Hall", "2003", "RB/FLEX", "7600", "NYJ@MIA 10/04/2026 01:00PM ET", "NYJ", "19.8"),
    ("RB", "Kyren Williams (2004)", "Kyren Williams", "2004", "RB/FLEX", "7400", "SF@LAR 10/04/2026 04:05PM ET", "LAR", "18.9"),
    ("RB", "James Cook (2005)", "James Cook", "2005", "RB/FLEX", "7100", "NE@BUF 10/04/2026 01:00PM ET", "BUF", "18.2"),
    ("RB", "Kenneth Walker III (2006)", "Kenneth Walker III", "2006", "RB/FLEX", "6800", "KC@DEN 10/04/2026 04:25PM ET", "KC", "17.4"),
    ("RB", "D'Andre Swift (2007)", "D'Andre Swift", "2007", "RB/FLEX", "6200", "GB@CHI 10/04/2026 01:00PM ET", "CHI", "15.8"),
    ("RB", "Aaron Jones (2008)", "Aaron Jones", "2008", "RB/FLEX", "6500", "MIN@DET 10/04/2026 01:00PM ET", "MIN", "16.1"),
    ("RB", "Josh Jacobs (2009)", "Josh Jacobs", "2009", "RB/FLEX", "6900", "GB@CHI 10/04/2026 01:00PM ET", "GB", "17.2"),
    ("RB", "Saquon Barkley (2010)", "Saquon Barkley", "2010", "RB/FLEX", "8500", "DAL@PHI 10/04/2026 08:20PM ET", "PHI", "22.5"),

    # WRs
    ("WR", "CeeDee Lamb (3001)", "CeeDee Lamb", "3001", "WR/FLEX", "8800", "DAL@PHI 10/04/2026 08:20PM ET", "DAL", "23.4"),
    ("WR", "Ja'Marr Chase (3002)", "Ja'Marr Chase", "3002", "WR/FLEX", "8600", "BAL@CIN 10/04/2026 01:00PM ET", "CIN", "22.8"),
    ("WR", "Justin Jefferson (3003)", "Justin Jefferson", "3003", "WR/FLEX", "8700", "MIN@DET 10/04/2026 01:00PM ET", "MIN", "23.0"),
    ("WR", "Amon-Ra St. Brown (3004)", "Amon-Ra St. Brown", "3004", "WR/FLEX", "8400", "MIN@DET 10/04/2026 01:00PM ET", "DET", "21.9"),
    ("WR", "Davante Adams (3005)", "Davante Adams", "3005", "WR/FLEX", "7400", "SF@LAR 10/04/2026 04:05PM ET", "LAR", "19.5"),
    ("WR", "Zay Flowers (3006)", "Zay Flowers", "3006", "WR/FLEX", "6700", "BAL@CIN 10/04/2026 01:00PM ET", "BAL", "17.6"),
    ("WR", "Jaxon Smith-Njigba (3007)", "Jaxon Smith-Njigba", "3007", "WR/FLEX", "6500", "SEA@ARI 10/04/2026 04:05PM ET", "SEA", "17.1"),
    ("WR", "Christian Watson (3008)", "Christian Watson", "3008", "WR/FLEX", "5500", "GB@CHI 10/04/2026 01:00PM ET", "GB", "15.2"),
    ("WR", "Kalif Raymond (3009)", "Kalif Raymond", "3009", "WR/FLEX", "4200", "GB@CHI 10/04/2026 01:00PM ET", "CHI", "11.8"),
    ("WR", "Khalil Shakir (3010)", "Khalil Shakir", "3010", "WR/FLEX", "5800", "NE@BUF 10/04/2026 01:00PM ET", "BUF", "15.4"),
    ("WR", "Deebo Samuel Sr. (3011)", "Deebo Samuel Sr.", "3011", "WR/FLEX", "6800", "SF@LAR 10/04/2026 04:05PM ET", "SF", "17.8"),
    ("WR", "Tee Higgins (3012)", "Tee Higgins", "3012", "WR/FLEX", "6300", "BAL@CIN 10/04/2026 01:00PM ET", "CIN", "16.4"),

    # TEs
    ("TE", "Brock Bowers (4001)", "Brock Bowers", "4001", "TE/FLEX", "6200", "LV@LAC 10/04/2026 04:05PM ET", "LV", "17.5"),
    ("TE", "Travis Kelce (4002)", "Travis Kelce", "4002", "TE/FLEX", "6400", "KC@DEN 10/04/2026 04:25PM ET", "KC", "17.9"),
    ("TE", "Dalton Kincaid (4003)", "Dalton Kincaid", "4003", "TE/FLEX", "5200", "NE@BUF 10/04/2026 01:00PM ET", "BUF", "14.2"),
    ("TE", "George Kittle (4004)", "George Kittle", "4004", "TE/FLEX", "6000", "SF@LAR 10/04/2026 04:05PM ET", "SF", "16.1"),
    ("TE", "Trey McBride (4005)", "Trey McBride", "4005", "TE/FLEX", "5900", "SEA@ARI 10/04/2026 04:05PM ET", "ARI", "15.8"),
    ("TE", "Mike Gesicki (4006)", "Mike Gesicki", "4006", "TE/FLEX", "4400", "BAL@CIN 10/04/2026 01:00PM ET", "CIN", "12.5"),

    # DSTs
    ("DST", "San Francisco 49ers (5001)", "49ers ", "5001", "DST", "3400", "SF@LAR 10/04/2026 04:05PM ET", "SF", "8.9"),
    ("DST", "Buffalo Bills (5002)", "Bills ", "5002", "DST", "3300", "NE@BUF 10/04/2026 01:00PM ET", "BUF", "8.6"),
    ("DST", "Baltimore Ravens (5003)", "Ravens ", "5003", "DST", "3200", "BAL@CIN 10/04/2026 01:00PM ET", "BAL", "8.4"),
    ("DST", "Philadelphia Eagles (5004)", "Eagles ", "5004", "DST", "3100", "DAL@PHI 10/04/2026 08:20PM ET", "PHI", "8.1"),
    ("DST", "Kansas City Chiefs (5005)", "Chiefs ", "5005", "DST", "3000", "KC@DEN 10/04/2026 04:25PM ET", "KC", "7.8"),
    ("DST", "Green Bay Packers (5006)", "Packers ", "5006", "DST", "2900", "GB@CHI 10/04/2026 01:00PM ET", "GB", "7.5"),
    ("DST", "Chicago Bears (5007)", "Bears ", "5007", "DST", "2800", "GB@CHI 10/04/2026 01:00PM ET", "CHI", "7.2"),
]


def generate_mock_draftkings_files():
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    salaries_path = PLAYERS_DIR / "DKSalaries-mock.csv"
    with open(salaries_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Position", "Name + ID", "Name", "ID", "Roster Position", "Salary", "Game Info", "TeamAbbrev", "AvgPointsPerGame"])
        for row in MOCK_DK_PLAYERS:
            writer.writerow(row)
    print(f"Generated mock DraftKings salaries: {salaries_path.resolve()}")

    entries_path = TEMPLATES_DIR / "DKEntries-mock.csv"
    with open(entries_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entry ID", "Contest ID", "Contest Name", "Entry Fee", "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"])
        for i in range(1, 151):
            writer.writerow([str(900000000 + i), "12345678", "$100K NFL Play-Action", "3.00", "", "", "", "", "", "", "", "", ""])
    print(f"Generated mock DraftKings entries: {entries_path.resolve()}")


if __name__ == "__main__":
    generate_mock_draftkings_files()
