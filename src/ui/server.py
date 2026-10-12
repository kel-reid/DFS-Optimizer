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

    # 2. Entry fee detection
    fee: Optional[float] = None
    if template_content:
        reader = csv.reader(io.StringIO(template_content))
        header = next(reader, None)
        if header:
            fee_idx = None
            for idx, col in enumerate(header):
                clean_col = col.strip().lower()
                if clean_col in ("entry_fee", "entry fee", "entry ($)", "fee", "entryfee") or "fee" in clean_col:
                    fee_idx = idx
                    break
            if fee_idx is not None:
                for row in reader:
                    if row and len(row) > fee_idx and row[fee_idx].strip():
                        val_str = row[fee_idx].strip().replace("$", "").replace(",", "")
                        try:
                            parsed_fee = float(val_str)
                            if parsed_fee >= 0:
                                fee = parsed_fee
                                break
                        except ValueError:
                            pass

    # 3. Entries count (only count rows where entry_id in column 0 is present)
    num_entries = 0
    if template_content:
        reader = csv.reader(io.StringIO(template_content))
        _ = next(reader, None)  # header
        for row in reader:
            if row and row[0].strip():
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
            pos = p_row.get("Position") or p_row.get("Roster Position") or ""
            if pos:
                positions.add(pos.strip().upper())
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

    return {
        "success": True,
        "platform": site,
        "is_single_game": is_single_game,
        "entry_fee": fee or 1.0,
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
        summary = _compute_export_summary(result_path, num_lineups)

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


def _compute_export_summary(output_csv_path: Path, target_k: int) -> Dict[str, Any]:
    """Analyzes the completed upload CSV to extract player exposure percentages for the UI."""
    player_counts: Dict[str, int] = {}
    total_lineups = 0

    if output_csv_path.exists():
        with open(output_csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                # Find player columns (typically slot columns starting from index 4 or named positions)
                slot_indices = []
                for idx, col in enumerate(header):
                    col_u = col.strip().upper()
                    if col_u in {"QB", "RB", "WR", "TE", "FLEX", "DST", "DEF", "MVP", "ANYFLEX", "CPT"} or (
                        idx >= 4 and not col_u.startswith("ENTRY") and not col_u.startswith("CONTEST")
                    ):
                        slot_indices.append(idx)

                for row in reader:
                    if not row or not any(field.strip() for field in row):
                        continue
                    total_lineups += 1
                    for s_idx in slot_indices:
                        if s_idx < len(row):
                            val = row[s_idx].strip()
                            if val:
                                # Extract clean player name (strips FanDuel "ID:Name" and DraftKings "Name (ID)")
                                clean_name = val
                                if ":" in clean_name:
                                    parts = clean_name.split(":", 1)
                                    if parts[1].strip():
                                        clean_name = parts[1].strip()
                                if "(" in clean_name:
                                    clean_name = clean_name.split("(", 1)[0].strip()
                                player_counts[clean_name] = player_counts.get(clean_name, 0) + 1

    # Sort exposures descending
    denom = max(total_lineups, 1)
    sorted_players = sorted(player_counts.items(), key=lambda item: item[1], reverse=True)
    top_exposures = [
        {
            "name": name,
            "count": count,
            "exposure_pct": round((count / denom) * 100, 1),
        }
        for name, count in sorted_players[:15]
    ]

    return {
        "num_lineups": total_lineups,
        "unique_players_count": len(player_counts),
        "top_exposures": top_exposures,
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
        self.end_headers()
        self.wfile.write(content)


def start_ui_server(port: int = 8500, auto_open: bool = True) -> None:
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
