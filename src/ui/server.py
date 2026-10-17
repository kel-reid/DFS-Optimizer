"""
================================================================================
DFS Optimizer Web Dashboard Server (src/ui/server.py)
================================================================================
Zero-dependency, Python standard library HTTP server providing a modern,
drag-and-drop web interface for generating FanDuel and DraftKings DFS lineups.
================================================================================
"""

from __future__ import annotations

import csv
import errno
import io
import json
import logging
import mimetypes
import tempfile
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from src.config import DKOptimizerConfig, SimOptimizerConfig
from src.data.projections import normalize_name
from src.site_detector import detect_site_from_content

logger = logging.getLogger("DFSOptimizerUI")

STATIC_DIR = Path(__file__).parent / "static"

# In-memory thread-safe jobs store
_JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def _inspect_csv_data(
    players_content: str,
    template_content: str,
    projections_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspects uploaded CSV contents to auto-detect platform, format, entry fee, and player counts."""
    # 1. Platform auto-detection
    site = detect_site_from_content(content=players_content) or detect_site_from_content(content=template_content)
    if not site:
        site = "draftkings" if "DKSalaries" in players_content or "Roster Position" in players_content else "fanduel"

    # 2. Entry fee & multi-contest detection
    fees: list[float] = []
    contests: dict[str, dict[str, Any]] = {}
    total_entry_fee = 0.0

    if template_content:
        reader = csv.reader(io.StringIO(template_content))
        header = next(reader, None)
        if header:
            fee_idx = None
            contest_idx = None
            for idx, col in enumerate(header):
                clean_col = col.strip().lower()
                if clean_col in ("entry_fee", "entry fee", "entry ($)", "fee", "entryfee") or "fee" in clean_col:
                    if fee_idx is None:
                        fee_idx = idx
                if clean_col in ("contest_name", "contest name", "contest", "title") or ("contest" in clean_col and "id" not in clean_col):
                    if contest_idx is None:
                        contest_idx = idx

            for row in reader:
                if not row or not any(field.strip() for field in row):
                    continue
                # Row must have an entry ID in column 0
                if not row[0].strip().strip('"').isdigit():
                    continue

                row_fee: Optional[float] = None
                if fee_idx is not None and len(row) > fee_idx and row[fee_idx].strip():
                    val_str = row[fee_idx].strip().replace("$", "").replace(",", "")
                    try:
                        f_val = float(val_str)
                        if f_val >= 0:
                            row_fee = f_val
                    except ValueError:
                        pass

                c_name = "Contest"
                if contest_idx is not None and len(row) > contest_idx and row[contest_idx].strip():
                    c_name = row[contest_idx].strip()

                if row_fee is not None:
                    fees.append(row_fee)
                    total_entry_fee += row_fee

                if c_name not in contests:
                    contests[c_name] = {"name": c_name, "count": 0, "fee": row_fee}
                contests[c_name]["count"] += 1

    unique_fees = sorted(list(set(fees)))
    fee: Optional[float] = None
    fee_sub_display = ""
    if len(unique_fees) == 1:
        fee = unique_fees[0]
        fee_display = f"${fee:.2f}"
    elif len(unique_fees) > 1:
        fee = unique_fees[-1]
        fees_str = ", ".join(f"${f:.2f}" for f in unique_fees)
        fee_display = fees_str
        fee_sub_display = f"(${total_entry_fee:.2f} Total)"
    else:
        fee_display = "$1.00"

    # 3. Entries count (only count rows where entry_id in column 0 is present)
    num_entries = sum(c["count"] for c in contests.values()) if contests else 0
    if num_entries == 0 and template_content:
        reader = csv.reader(io.StringIO(template_content))
        _ = next(reader, None)  # header
        for row in reader:
            if row and row[0].strip().strip('"').isdigit():
                num_entries += 1

    # 4. Players count & format detection (Classic vs Single Game / Showdown)
    num_players = 0
    sample_players: list[dict[str, Any]] = []
    is_single_game = False

    if players_content:
        dict_reader = csv.DictReader(io.StringIO(players_content))
        positions = set()
        for p_row in dict_reader:
            if not p_row:
                continue
            num_players += 1
            pos = p_row.get("Position") or ""
            roster_pos = p_row.get("Roster Position") or ""
            if pos:
                positions.add(pos.strip().upper())
            if roster_pos:
                positions.add(roster_pos.strip().upper())
            if len(sample_players) < 5:
                name = (
                    p_row.get("Nickname")
                    or p_row.get("Name")
                    or p_row.get("Name + ID", "").split("(")[0].strip()
                    or "Unknown"
                )
                team = p_row.get("Team") or p_row.get("TeamAbbrev") or ""
                salary = p_row.get("Salary") or ""
                sample_players.append(
                    {
                        "name": name,
                        "team": team,
                        "position": pos,
                        "salary": salary,
                    }
                )

        # Detect Showdown / Single Game
        if site == "draftkings" and ("CPT" in positions or "CPT/FLEX" in str(positions)):
            is_single_game = True
        elif site == "fanduel" and ("MVP" in positions or "AnyFLEX" in str(positions)):
            is_single_game = True

    if not is_single_game and template_content:
        first_line = template_content.strip().split("\n")[0]
        if "MVP" in first_line or "AnyFLEX" in first_line or "CPT" in first_line:
            is_single_game = True

    return {
        "success": True,
        "platform": site,
        "is_single_game": is_single_game,
        "entry_fee": fee or 1.0,
        "entry_fee_display": fee_display,
        "entry_fee_sub": fee_sub_display,
        "unique_fees": unique_fees,
        "total_entry_fee": total_entry_fee,
        "contests": list(contests.values()),
        "num_entries": num_entries if num_entries > 0 else 150,
        "num_players": num_players,
        "has_projections": bool(projections_content and len(projections_content.strip()) > 0),
        "sample_players": sample_players,
    }


def _execute_optimization_job(job_id: str, temp_dir: Path, payload: Dict[str, Any]) -> None:
    """Worker thread that executes the full optimization pipeline and collects metrics."""
    try:

        def callback(pct: int, msg: str) -> None:
            with _JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = pct
                    JOBS[job_id]["stage"] = msg
                    JOBS[job_id]["logs"].append(f"[{pct}%] {msg}")

        callback(5, "Preparing environment and writing temporary contest files...")

        players_content = payload.get("players_content", "")
        template_content = payload.get("template_content", "")
        projections_content = payload.get("projections_content", "")
        raw_name = str(payload.get("template_filename") or "DKEntries.csv")
        template_filename = Path(raw_name).name or "DKEntries.csv"
        if not template_filename.lower().endswith(".csv"):
            template_filename += ".csv"
        options = payload.get("options", {})

        players_file = temp_dir / "players.csv"
        template_file = temp_dir / template_filename
        output_file = temp_dir / f"Completed-{template_filename}"

        if not template_file.resolve().is_relative_to(temp_dir.resolve()) or not output_file.resolve().is_relative_to(temp_dir.resolve()):
            raise ValueError("Invalid template filename.")

        players_file.write_text(players_content, encoding="utf-8")
        template_file.write_text(template_content, encoding="utf-8")

        if not projections_content or not str(projections_content).strip():
            raise ValueError("Player projections file is required to optimize lineups.")
        projections_file = temp_dir / "projections.csv"
        projections_file.write_text(projections_content, encoding="utf-8")

        # Detect site & format
        site = (
            payload.get("platform")
            or detect_site_from_content(path=players_file)
            or detect_site_from_content(path=template_file)
            or "draftkings"
        )
        is_single_game = bool(options.get("is_single_game", False))
        if not is_single_game:
            insp = _inspect_csv_data(players_content, template_content, projections_content)
            is_single_game = bool(insp.get("is_single_game", False))

        num_lineups = int(options.get("num_lineups", 150))
        num_candidates = int(options.get("num_candidates", 500))
        num_field = int(options.get("num_field", 10000))
        num_trials = int(options.get("num_trials", 5000))
        entry_fee = float(options.get("entry_fee", 1.0))
        stack_ratio = float(options.get("stack_ratio", 0.80))
        raw_game_stack_ratio = options.get("game_stack_ratio")
        if raw_game_stack_ratio is not None:
            game_stack_ratio = float(raw_game_stack_ratio)
        else:
            game_stack_ratio = min(0.40, stack_ratio)
        game_stack_ratio = min(game_stack_ratio, stack_ratio)
        max_qb_exposure = float(options.get("max_qb_exposure", 0.25))
        max_rb_exposure = float(options.get("max_rb_exposure", 0.25))
        max_wr_exposure = float(options.get("max_wr_exposure", 0.25))
        max_te_exposure = float(options.get("max_te_exposure", 0.25))
        max_def_exposure = float(options.get("max_def_exposure", 0.20))
        max_exposure = float(options.get("max_exposure", 0.25))
        max_repeating = int(options.get("max_repeating", 6))
        randomness = float(options.get("randomness", 0.25))
        zero_unprojected = bool(options.get("zero_unprojected", True))
        strict_caps = bool(options.get("strict_caps", False))

        selection_metric = str(options.get("selection_metric", "roi"))
        primary_stack_size = int(options.get("primary_stack_size", 1))
        bring_back_size = int(options.get("bring_back_size", 1))
        include_team_defense_stack = bool(
            options.get("team_defense_stack", False) or options.get("include_team_defense_stack", False)
        )
        raw_max_own = options.get("max_total_ownership")
        max_total_ownership = float(raw_max_own) if raw_max_own is not None and str(raw_max_own).strip() != "" else None

        if site == "draftkings":
            from src.build_draftkings_lineups import run_draftkings_pipeline

            cfg = DKOptimizerConfig.from_settings(
                players_csv=players_file,
                template_csv=template_file,
                output_csv=output_file,
                projections_csv=projections_file,
                num_candidates=num_candidates,
                num_field_lineups=num_field,
                num_sim_trials=num_trials,
                num_selected_lineups=num_lineups,
                entry_fee=entry_fee,
                stack_ratio=stack_ratio,
                game_stack_ratio=game_stack_ratio,
                primary_stack_size=primary_stack_size,
                bring_back_size=bring_back_size,
                include_team_defense_stack=include_team_defense_stack,
                max_total_ownership=max_total_ownership,
                selection_metric=selection_metric,
                max_qb_exposure=max_qb_exposure,
                max_rb_exposure=max_rb_exposure,
                max_wr_exposure=max_wr_exposure,
                max_te_exposure=max_te_exposure,
                max_def_exposure=max_def_exposure,
                max_exposure=max_exposure,
                max_repeating_players=max_repeating,
                randomness_deviation=randomness,
                zero_unprojected=zero_unprojected,
                strict_exposure_caps=strict_caps,
                is_single_game=is_single_game,
            )
            result_path = run_draftkings_pipeline(config=cfg, progress_callback=callback)
        else:
            from src.build_fanduel_lineups import run_fanduel_pipeline

            fd_cfg = SimOptimizerConfig.from_settings(
                players_csv=players_file,
                template_csv=template_file,
                output_csv=output_file,
                projections_csv=projections_file,
                num_candidates=num_candidates,
                num_field_lineups=num_field,
                num_sim_trials=num_trials,
                num_selected_lineups=num_lineups,
                entry_fee=entry_fee,
                stack_ratio=stack_ratio,
                game_stack_ratio=game_stack_ratio,
                primary_stack_size=primary_stack_size,
                bring_back_size=bring_back_size,
                include_team_defense_stack=include_team_defense_stack,
                max_total_ownership=max_total_ownership,
                selection_metric=selection_metric,
                max_qb_exposure=max_qb_exposure,
                max_rb_exposure=max_rb_exposure,
                max_wr_exposure=max_wr_exposure,
                max_te_exposure=max_te_exposure,
                max_def_exposure=max_def_exposure,
                max_exposure=max_exposure,
                max_repeating_players=max_repeating,
                randomness_deviation=randomness,
                zero_unprojected=zero_unprojected,
                strict_exposure_caps=strict_caps,
                is_single_game=is_single_game,
            )
            result_path = run_fanduel_pipeline(config=fd_cfg, progress_callback=callback)

        # Audit exported lineups for UI metrics
        summary = _compute_export_summary(
            result_path,
            num_lineups,
            salaries_path=Path(players_file) if players_file else None,
            projections_path=Path(projections_file) if projections_file else None,
        )

        with _JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["progress"] = 100
                JOBS[job_id]["stage"] = "Completed successfully!"
                JOBS[job_id]["output_path"] = str(result_path)
                JOBS[job_id]["output_filename"] = result_path.name
                JOBS[job_id]["summary"] = summary

    except Exception as exc:
        logger.exception("Optimization job %s failed: %s", job_id, exc)
        with _JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(exc)
                JOBS[job_id]["stage"] = f"Failed: {exc}"
                JOBS[job_id]["logs"].append(f"[ERROR] {exc}")


def _clean_player_name(val: str) -> str:
    """Extracts clean player name by removing ID prefixes (FanDuel) or suffixes (DraftKings)."""
    clean_name = val.strip()
    if ":" in clean_name:
        parts = clean_name.split(":", 1)
        if parts[1].strip():
            clean_name = parts[1].strip()
    if "(" in clean_name:
        clean_name = clean_name.split("(", 1)[0].strip()
    return clean_name


def _compute_export_summary(
    output_csv_path: Path,
    target_k: int,
    salaries_path: Optional[Path] = None,
    projections_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyzes the completed upload CSV to extract player exposure percentages and roster lineups for the UI."""
    player_counts: Dict[str, int] = {}
    player_positions: Dict[str, str] = {}
    total_lineups = 0
    lineups: list[Dict[str, Any]] = []

    # Build player metadata dictionary from salary and projection feeds if available
    player_db: Dict[str, Dict[str, Any]] = {}
    if salaries_path and Path(salaries_path).exists():
        try:
            with open(salaries_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "first_name" in r and "last_name" in r:
                        fn = (r.get("first_name") or "").strip()
                        ln = (r.get("last_name") or "").strip()
                        name = f"{fn} {ln}".strip() if ln else fn
                    elif "First Name" in r and "Last Name" in r:
                        fn = (r.get("First Name") or "").strip()
                        ln = (r.get("Last Name") or "").strip()
                        name = (r.get("Nickname") or f"{fn} {ln}").strip()
                    else:
                        name = (r.get("Nickname") or r.get("Name") or r.get("Player") or r.get("player") or "").strip()
                    if not name:
                        continue
                    sal_str = (r.get("Salary") or r.get("salary") or "0").replace("$", "").replace(",", "").strip()
                    try:
                        salary = int(float(sal_str))
                    except ValueError:
                        salary = 0
                    team = (r.get("Team") or r.get("team") or r.get("TeamAbbrev") or "").strip().upper()
                    opp = (r.get("Opponent") or r.get("opp") or r.get("Opp") or "").strip()
                    if not opp and "Game Info" in r:
                        game_info = r.get("Game Info", "")
                        if "@" in game_info:
                            matchup = game_info.split()[0]
                            teams = matchup.split("@")
                            if len(teams) == 2:
                                away, home = teams[0].strip().upper(), teams[1].strip().upper()
                                if team == away:
                                    opp = f"@{home}"
                                elif team == home:
                                    opp = f"vs {away}"
                    pos = (r.get("Position") or r.get("position") or r.get("Roster Position") or "").strip().upper()
                    # Prioritize forward-looking ppg_projection over historical szn_fppg_avg / FPPG
                    fppg_str = (
                        r.get("ppg_projection")
                        or r.get("PPG_Projection")
                        or r.get("ppg_proj")
                        or r.get("projection")
                        or r.get("Projection")
                        or r.get("Points")
                        or r.get("points")
                        or r.get("FPPG")
                        or r.get("AvgPointsPerGame")
                        or "0"
                    ).strip()
                    try:
                        fppg = round(float(fppg_str), 1)
                    except ValueError:
                        fppg = 0.0

                    c_name = _clean_player_name(name).lower()
                    entry = {
                        "name": name,
                        "salary": salary,
                        "team": team,
                        "opponent": opp,
                        "position": pos,
                        "fppg": fppg,
                    }
                    player_db[c_name] = entry
                    norm_n = normalize_name(name)
                    if norm_n:
                        player_db[norm_n] = entry
                    if pos in ("DEF", "DST", "D") or "defense" in name.lower() or "d/st" in name.lower():
                        if team:
                            player_db[normalize_name(team)] = entry
                        clean_t = (
                            name.lower()
                            .replace("defense", "")
                            .replace("d/st", "")
                            .replace("dst", "")
                            .replace("def", "")
                            .strip()
                        )
                        if clean_t:
                            player_db[normalize_name(clean_t)] = entry
        except Exception as e:
            logger.warning("Could not parse salaries_path for lineup inspector: %s", e)

    if projections_path and Path(projections_path).exists():
        try:
            with open(projections_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "first_name" in r and "last_name" in r:
                        fn = (r.get("first_name") or "").strip()
                        ln = (r.get("last_name") or "").strip()
                        p_name = f"{fn} {ln}".strip() if ln else fn
                    else:
                        p_name = (
                            r.get("Player")
                            or r.get("player")
                            or r.get("Name")
                            or r.get("name")
                            or r.get("player_name")
                            or r.get("Nickname")
                            or ""
                        ).strip()
                    if not p_name:
                        continue
                    c_name = _clean_player_name(p_name).lower()
                    norm_p = normalize_name(p_name)
                    pos = (r.get("position") or r.get("Position") or "").strip().upper()
                    proj_str = (
                        r.get("ppg_projection")
                        or r.get("PPG_Projection")
                        or r.get("ppg_proj")
                        or r.get("PPG_Proj")
                        or r.get("proj_fppg")
                        or r.get("Proj_FPPG")
                        or r.get("projection")
                        or r.get("Projection")
                        or r.get("Points")
                        or r.get("points")
                        or r.get("fantasy")
                        or r.get("FPPG")
                        or r.get("fppg")
                        or ""
                    ).strip()
                    fppg_val: Optional[float] = None
                    if proj_str:
                        try:
                            fppg_val = round(float(proj_str), 1)
                        except ValueError:
                            pass

                    # Match existing entry in player_db
                    matched = player_db.get(c_name) or player_db.get(norm_p)
                    team_val = (r.get("team") or r.get("Team") or "").strip().upper()
                    if not matched and (pos in ("DEF", "DST", "D") or "defense" in p_name.lower()):
                        if team_val and normalize_name(team_val) in player_db:
                            matched = player_db[normalize_name(team_val)]

                    if matched:
                        if fppg_val is not None:
                            matched["fppg"] = fppg_val
                        if pos and not matched.get("position"):
                            matched["position"] = pos
                    else:
                        opp = (r.get("opp") or r.get("Opp") or r.get("Opponent") or "").strip()
                        new_entry = {
                            "name": p_name,
                            "salary": 0,
                            "team": team_val,
                            "opponent": opp,
                            "position": pos,
                            "fppg": fppg_val or 0.0,
                        }
                        player_db[c_name] = new_entry
                        if norm_p:
                            player_db[norm_p] = new_entry
                        if pos in ("DEF", "DST", "D") or "defense" in p_name.lower():
                            if team_val:
                                player_db[normalize_name(team_val)] = new_entry
                            clean_t = (
                                p_name.lower()
                                .replace("defense", "")
                                .replace("d/st", "")
                                .replace("dst", "")
                                .replace("def", "")
                                .strip()
                            )
                            if clean_t:
                                player_db[normalize_name(clean_t)] = new_entry
        except Exception as e:
            logger.warning("Could not parse projections_path for lineup inspector: %s", e)

    salary_cap = 60000

    if output_csv_path.exists():
        with open(output_csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            if header:
                # Detect site salary cap
                header_str = " ".join(header).upper()
                if "DST" in header_str or "CPT" in header_str:
                    salary_cap = 50000
                elif "DEF" in header_str or "MVP" in header_str:
                    salary_cap = 60000

                # Find player columns
                slot_indices = []
                valid_slots = {
                    "QB", "RB", "WR", "TE", "FLEX", "DST", "DEF",
                    "MVP", "ANYFLEX", "CPT", "CAPTAIN", "UTIL", "SUPERFLEX",
                }
                for idx, col in enumerate(header):
                    col_u = col.strip().upper()
                    first_tok = col_u.replace("-", " ").replace("_", " ").split()[0] if col_u else ""
                    if col_u in valid_slots or first_tok in valid_slots:
                        slot_indices.append(idx)
                    elif idx >= 4 and not any(col_u.startswith(p) for p in ("ENTRY", "CONTEST", "INSTRUCTION", "LINEUP", "FEE", "DATE")) and col_u != "":
                        slot_indices.append(idx)

                for row in csv_reader:
                    if not row or not any(field.strip() for field in row) or not row[0].strip():
                        continue
                    total_lineups += 1
                    if target_k > 0 and total_lineups > target_k:
                        break

                    row_slots: list[Dict[str, Any]] = []
                    total_salary = 0
                    total_fppg = 0.0
                    qb_team = ""
                    opp_team = ""

                    # First pass: collect slot info and identify starting QB
                    for s_idx in slot_indices:
                        if s_idx < len(row):
                            val = row[s_idx].strip()
                            if not val:
                                continue
                            clean_name = _clean_player_name(val)
                            player_counts[clean_name] = player_counts.get(clean_name, 0) + 1

                            slot_label = header[s_idx].strip()
                            p_info = player_db.get(clean_name.lower())
                            if not p_info:
                                p_info = player_db.get(normalize_name(clean_name))
                            if not p_info and slot_label.upper() in ("DEF", "DST"):
                                for tok in clean_name.lower().split():
                                    norm_tok = normalize_name(tok)
                                    if norm_tok in player_db:
                                        p_info = player_db[norm_tok]
                                        break
                            if not p_info:
                                p_info = {}

                            p_salary = p_info.get("salary", 0)
                            p_team = p_info.get("team", "")
                            p_opp = p_info.get("opponent", "")
                            p_fppg = p_info.get("fppg", 0.0)

                            p_pos = p_info.get("position", "")
                            if slot_label.upper() == "QB":
                                player_positions[clean_name] = "QB"
                            elif p_pos:
                                player_positions[clean_name] = p_pos
                            elif clean_name not in player_positions:
                                player_positions[clean_name] = slot_label.upper()

                            total_salary += p_salary
                            total_fppg += p_fppg

                            if slot_label.upper() == "QB" or (not qb_team and p_info.get("position", "").upper() == "QB"):
                                qb_team = p_team
                                opp_clean = p_opp.replace("@", "").replace("vs", "").strip().upper()
                                if opp_clean:
                                    opp_team = opp_clean

                            row_slots.append({
                                "slot": slot_label,
                                "name": clean_name,
                                "team": p_team or "-",
                                "opponent": p_opp or "-",
                                "salary": p_salary,
                                "fppg": p_fppg,
                            })

                    # Second pass: tag stack correlation
                    qb_stack_count = 0
                    bring_back_count = 0
                    for slot in row_slots:
                        is_qb = (slot["slot"].upper() == "QB")
                        if is_qb:
                            slot["stack_type"] = "QB"
                        elif qb_team and slot["team"] == qb_team and slot["team"] != "-":
                            slot["stack_type"] = "Pass-Catcher"
                            qb_stack_count += 1
                        elif opp_team and slot["team"] == opp_team and slot["team"] != "-":
                            slot["stack_type"] = "Bring-Back"
                            bring_back_count += 1
                        else:
                            slot["stack_type"] = ""

                    stack_badge = ""
                    if qb_team and qb_stack_count > 0:
                        stack_badge = f"{qb_team} (QB+{qb_stack_count})"
                        if opp_team and bring_back_count > 0:
                            stack_badge += f" vs {opp_team} ({bring_back_count})"
                    elif qb_team:
                        stack_badge = f"{qb_team} Solo"

                    lineups.append({
                        "index": total_lineups,
                        "total_salary": total_salary,
                        "salary_cap": salary_cap,
                        "total_fppg": round(total_fppg, 1),
                        "stack_badge": stack_badge or "Balanced",
                        "slots": row_slots,
                    })

    # Partition exposures into Top 10 Quarterbacks and Top 10 Non-Quarterbacks
    denom = max(total_lineups, 1)
    qb_players: Dict[str, int] = {}
    non_qb_players: Dict[str, int] = {}

    for name, count in player_counts.items():
        pos = player_positions.get(name, "").upper()
        if not pos and name.lower() in player_db:
            pos = player_db[name.lower()].get("position", "").upper()

        if pos == "QB" or "QB" in pos.split("/"):
            qb_players[name] = count
        else:
            non_qb_players[name] = count

    sorted_qbs = sorted(qb_players.items(), key=lambda item: item[1], reverse=True)
    sorted_non_qbs = sorted(non_qb_players.items(), key=lambda item: item[1], reverse=True)
    sorted_all = sorted(player_counts.items(), key=lambda item: item[1], reverse=True)

    top_qb_exposures = [
        {
            "name": name,
            "position": "QB",
            "team": player_db.get(name.lower(), {}).get("team", "-"),
            "count": count,
            "exposure_pct": round((count / denom) * 100, 1),
        }
        for name, count in sorted_qbs[:10]
    ]

    top_non_qb_exposures = [
        {
            "name": name,
            "position": player_positions.get(name, player_db.get(name.lower(), {}).get("position", "-")),
            "team": player_db.get(name.lower(), {}).get("team", "-"),
            "count": count,
            "exposure_pct": round((count / denom) * 100, 1),
        }
        for name, count in sorted_non_qbs[:10]
    ]

    top_exposures = [
        {
            "name": name,
            "count": count,
            "exposure_pct": round((count / denom) * 100, 1),
        }
        for name, count in sorted_all[:15]
    ]

    return {
        "num_lineups": total_lineups,
        "unique_players_count": len(player_counts),
        "top_exposures": top_exposures,
        "top_qb_exposures": top_qb_exposures,
        "top_non_qb_exposures": top_non_qb_exposures,
        "lineups": lineups,
    }


class DFSOptimizerHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing REST endpoints and serving static UI assets."""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal output tidy
        if logger.isEnabledFor(logging.DEBUG):
            super().log_message(format, *args)

    def _send_json(self, data: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/") :]
            file_path = (STATIC_DIR / rel_path).resolve()
            if STATIC_DIR in file_path.parents and file_path.exists():
                mime, _ = mimetypes.guess_type(str(file_path))
                self._serve_file(file_path, mime or "application/octet-stream")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
        elif path == "/api/progress":
            params = parse_qs(parsed.query)
            job_id = params.get("job_id", [""])[0]
            with _JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
        elif path == "/api/download":
            params = parse_qs(parsed.query)
            job_id = params.get("job_id", [""])[0]
            with _JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job or job.get("status") != "completed" or not job.get("output_path"):
                self.send_error(HTTPStatus.NOT_FOUND, "Completed file not available for download")
                return

            out_path = Path(job["output_path"])
            if not out_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Output file does not exist on disk")
                return

            filename = job.get("output_filename", out_path.name)
            data = out_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/inspect":
            result = _inspect_csv_data(
                players_content=payload.get("players_content", ""),
                template_content=payload.get("template_content", ""),
                projections_content=payload.get("projections_content", ""),
            )
            self._send_json(result)

        elif path == "/api/optimize":
            players_content = payload.get("players_content", "")
            template_content = payload.get("template_content", "")
            projections_content = payload.get("projections_content", "")

            if not template_content or not str(template_content).strip():
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "Contest entries template file is required.",
                )
                return

            if not players_content or not str(players_content).strip():
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "Player salaries file is required.",
                )
                return

            if not projections_content or not str(projections_content).strip():
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "Player projections file is required.",
                )
                return

            job_id = str(uuid.uuid4())
            temp_dir = Path(tempfile.mkdtemp(prefix="dfs_opt_"))

            with _JOBS_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "running",
                    "progress": 0,
                    "stage": "Job queued...",
                    "logs": [],
                    "output_path": None,
                    "output_filename": None,
                    "summary": None,
                    "error": None,
                }

            thread = threading.Thread(
                target=_execute_optimization_job,
                args=(job_id, temp_dir, payload),
                daemon=True,
            )
            thread.start()

            self._send_json({"success": True, "job_id": job_id})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API route not found")

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(content)


def start_ui_server(port: int = 8500, auto_open: bool = False) -> None:
    """Initializes and runs the local DFS Optimizer Web Dashboard server."""
    server_address = ("127.0.0.1", port)
    try:
        httpd = ThreadingHTTPServer(server_address, DFSOptimizerHandler)
    except OSError as err:
        if err.errno == errno.EADDRINUSE:
            logger.warning("Port %d already in use, binding to port %d...", port, port + 1)
            port += 1
            server_address = ("127.0.0.1", port)
            httpd = ThreadingHTTPServer(server_address, DFSOptimizerHandler)
        else:
            raise

    url = f"http://localhost:{port}"
    print("\n" + "=" * 70)
    print("  🚀 DFS OPTIMIZER WEB DASHBOARD ACTIVE")
    print(f"  🌐 Local URL: {url}")
    print("  ⚡ Press Ctrl+C in terminal to stop server")
    print("=" * 70 + "\n")

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down DFS Optimizer Web Dashboard...")
    finally:
        httpd.server_close()
