"""
Script to generate realistic FanDuel NFL Main Slate player pool and 150-entry upload template.
Used for testing and demonstration of the MME lineup optimization pipeline.
"""

import csv
import random

def generate_mock_data():
    teams_matchups = [
        ("KC", "BUF"),
        ("PHI", "DAL"),
        ("BAL", "CIN"),
        ("SF", "LAR"),
        ("DET", "GB"),
        ("MIA", "NYJ"),
        ("HOU", "IND"),
        ("MIN", "CHI"),
        ("ATL", "NO"),
        ("TB", "CAR"),
        ("LAC", "DEN"),
        ("ARI", "SEA"),
    ]

    qb_names = {
        "KC": ("Patrick", "Mahomes"), "BUF": ("Josh", "Allen"),
        "PHI": ("Jalen", "Hurts"), "DAL": ("Dak", "Prescott"),
        "BAL": ("Lamar", "Jackson"), "CIN": ("Joe", "Burrow"),
        "SF": ("Brock", "Purdy"), "LAR": ("Matthew", "Stafford"),
        "DET": ("Jared", "Goff"), "GB": ("Jordan", "Love"),
        "MIA": ("Tua", "Tagovailoa"), "NYJ": ("Aaron", "Rodgers"),
        "HOU": ("C.J.", "Stroud"), "IND": ("Anthony", "Richardson"),
        "MIN": ("Sam", "Darnold"), "CHI": ("Caleb", "Williams"),
        "ATL": ("Kirk", "Cousins"), "NO": ("Derek", "Carr"),
        "TB": ("Baker", "Mayfield"), "CAR": ("Bryce", "Young"),
        "LAC": ("Justin", "Herbert"), "DEN": ("Bo", "Nix"),
        "ARI": ("Kyler", "Murray"), "SEA": ("Geno", "Smith"),
    }

    team_nicknames = {
        "KC": "Chiefs", "BUF": "Bills", "PHI": "Eagles", "DAL": "Cowboys",
        "BAL": "Ravens", "CIN": "Bengals", "SF": "49ers", "LAR": "Rams",
        "DET": "Lions", "GB": "Packers", "MIA": "Dolphins", "NYJ": "Jets",
        "HOU": "Texans", "IND": "Colts", "MIN": "Vikings", "CHI": "Bears",
        "ATL": "Falcons", "NO": "Saints", "TB": "Buccaneers", "CAR": "Panthers",
        "LAC": "Chargers", "DEN": "Broncos", "ARI": "Cardinals", "SEA": "Seahawks",
    }

    players = []
    pid_counter = 10000

    for away, home in teams_matchups:
        game_str = f"{away}@{home}"
        for team, opp in [(away, home), (home, away)]:
            # 1 QB
            first, last = qb_names[team]
            pid_counter += 1
            fd_id = f"115000-{pid_counter}"
            salary = random.randint(7000, 8800)
            fppg = round(random.uniform(17.5, 23.5), 2)
            players.append({
                "Id": fd_id, "Position": "QB", "First Name": first, "Nickname": f"{first} {last}",
                "Last Name": last, "FPPG": str(fppg), "Team": team, "Opponent": opp,
                "Game": game_str, "Injury Indicator": "", "Injury Details": "", "Tier": "",
                "Probable Pitcher": "", "Batting Order": "", "Roster Position": "QB", "Salary": str(salary)
            })

            # 3 RBs per team
            for rbi in range(1, 4):
                pid_counter += 1
                fd_id = f"115000-{pid_counter}"
                r_first = f"{team}"
                r_last = f"RB{rbi}"
                if rbi == 1:
                    sal = random.randint(6800, 8900)
                    fppg = round(random.uniform(14.0, 19.5), 2)
                elif rbi == 2:
                    sal = random.randint(5200, 6500)
                    fppg = round(random.uniform(9.0, 13.5), 2)
                else:
                    sal = random.randint(4600, 5100)
                    fppg = round(random.uniform(4.5, 8.5), 2)
                
                players.append({
                    "Id": fd_id, "Position": "RB", "First Name": r_first, "Nickname": f"{r_first} {r_last}",
                    "Last Name": r_last, "FPPG": str(fppg), "Team": team, "Opponent": opp,
                    "Game": game_str, "Injury Indicator": "", "Injury Details": "", "Tier": "",
                    "Probable Pitcher": "", "Batting Order": "", "Roster Position": "RB", "Salary": str(sal)
                })

            # 4 WRs per team
            for wri in range(1, 5):
                pid_counter += 1
                fd_id = f"115000-{pid_counter}"
                w_first = f"{team}"
                w_last = f"WR{wri}"
                if wri == 1:
                    sal = random.randint(7200, 9200)
                    fppg = round(random.uniform(15.0, 20.0), 2)
                elif wri == 2:
                    sal = random.randint(5800, 7100)
                    fppg = round(random.uniform(11.0, 14.8), 2)
                elif wri == 3:
                    sal = random.randint(4900, 5700)
                    fppg = round(random.uniform(7.5, 10.5), 2)
                else:
                    sal = random.randint(4500, 4800)
                    fppg = round(random.uniform(4.0, 7.0), 2)

                players.append({
                    "Id": fd_id, "Position": "WR", "First Name": w_first, "Nickname": f"{w_first} {w_last}",
                    "Last Name": w_last, "FPPG": str(fppg), "Team": team, "Opponent": opp,
                    "Game": game_str, "Injury Indicator": "", "Injury Details": "", "Tier": "",
                    "Probable Pitcher": "", "Batting Order": "", "Roster Position": "WR", "Salary": str(sal)
                })

            # 2 TEs per team
            for tei in range(1, 3):
                pid_counter += 1
                fd_id = f"115000-{pid_counter}"
                t_first = f"{team}"
                t_last = f"TE{tei}"
                if tei == 1:
                    sal = random.randint(5200, 6800)
                    fppg = round(random.uniform(9.0, 14.0), 2)
                else:
                    sal = random.randint(4200, 5000)
                    fppg = round(random.uniform(4.0, 7.5), 2)

                players.append({
                    "Id": fd_id, "Position": "TE", "First Name": t_first, "Nickname": f"{t_first} {t_last}",
                    "Last Name": t_last, "FPPG": str(fppg), "Team": team, "Opponent": opp,
                    "Game": game_str, "Injury Indicator": "", "Injury Details": "", "Tier": "",
                    "Probable Pitcher": "", "Batting Order": "", "Roster Position": "TE", "Salary": str(sal)
                })

            # 1 DEF per team (FanDuel official uses DEF or D)
            pid_counter += 1
            fd_id = f"115000-{pid_counter}"
            nick = team_nicknames[team]
            sal = random.randint(3400, 4800)
            fppg = round(random.uniform(5.5, 8.8), 2)
            players.append({
                "Id": fd_id, "Position": "DEF", "First Name": team, "Nickname": f"{team} {nick}",
                "Last Name": nick, "FPPG": str(fppg), "Team": team, "Opponent": opp,
                "Game": game_str, "Injury Indicator": "", "Injury Details": "", "Tier": "",
                "Probable Pitcher": "", "Batting Order": "", "Roster Position": "DEF", "Salary": str(sal)
            })

    from pathlib import Path
    out_dir_players = Path("data/players")
    out_dir_templates = Path("data/templates")
    out_dir_players.mkdir(parents=True, exist_ok=True)
    out_dir_templates.mkdir(parents=True, exist_ok=True)

    # Write mock player list
    player_headers = [
        "Id", "Position", "First Name", "Nickname", "Last Name", "FPPG", "Team", "Opponent",
        "Game", "Injury Indicator", "Injury Details", "Tier", "Probable Pitcher",
        "Batting Order", "Roster Position", "Salary"
    ]
    players_file = out_dir_players / "mock-FanDuel-NFL-players-list.csv"
    with open(players_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=player_headers)
        writer.writeheader()
        writer.writerows(players)
    print(f"Generated {players_file} with {len(players)} players.")

    # Write mock 150-entry template
    template_headers = ["entry_id", "contest_id", "contest_name", "entry_fee", "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"]
    template_file = out_dir_templates / "mock-FanDuel-NFL-entries-template.csv"
    with open(template_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(template_headers)
        for i in range(1, 151):
            entry_id = f"8814529-{100000 + i}"
            contest_id = "104928"
            contest_name = "NFL Sunday Million ($1.5M to 1st)"
            entry_fee = "$25"
            row = [entry_id, contest_id, contest_name, entry_fee, "", "", "", "", "", "", "", "", ""]
            writer.writerow(row)
    print(f"Generated {template_file} with 150 entries.")

if __name__ == "__main__":
    generate_mock_data()
