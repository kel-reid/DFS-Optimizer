"""
================================================================================
Unit & Integration Tests for DFS Optimizer Web Dashboard (tests/test_ui_server.py)
================================================================================
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ui.server import (
    _JOBS_LOCK,
    JOBS,
    DFSOptimizerHandler,
    _compute_export_summary,
    _execute_optimization_job,
    _inspect_csv_data,
)


class DummyServer:
    server_address = ("127.0.0.1", 8500)


class DummySocket:
    def __init__(self, data: bytes):
        self.rfile = io.BytesIO(data)
        self.wfile = io.BytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile

    def sendall(self, data: bytes):
        self.wfile.write(data)


def simulate_request(
    method: str,
    path: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Simulates an HTTP request against DFSOptimizerHandler in-memory without network sockets."""
    hdr_lines = [f"{method} {path} HTTP/1.1", "Host: localhost"]
    if headers:
        for k, v in headers.items():
            hdr_lines.append(f"{k}: {v}")
    if body and "Content-Length" not in (headers or {}):
        hdr_lines.append(f"Content-Length: {len(body)}")
    raw_request = "\r\n".join(hdr_lines).encode("utf-8") + b"\r\n\r\n" + body

    sock = DummySocket(raw_request)
    _ = DFSOptimizerHandler(sock, ("127.0.0.1", 12345), DummyServer())  # type: ignore[arg-type]
    raw_resp = sock.wfile.getvalue()

    # Parse response
    parts = raw_resp.split(b"\r\n\r\n", 1)
    header_block = parts[0].decode("utf-8", errors="replace")
    body_data = parts[1] if len(parts) > 1 else b""

    lines = header_block.split("\r\n")
    status_line = lines[0]
    status_code = int(status_line.split(" ")[1])

    resp_headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            hk, hv = line.split(":", 1)
            resp_headers[hk.strip().lower()] = hv.strip()

    return status_code, resp_headers, body_data


@pytest.fixture
def mock_dk_csv_content() -> tuple[str, str]:
    players_csv = (
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
        "QB,Josh Allen (1001),Josh Allen,1001,QB,8000,BUF@KC,BUF,24.5\n"
        "RB,James Cook (1002),James Cook,1002,RB,6500,BUF@KC,BUF,15.2\n"
        "WR,Stefon Diggs (1003),Stefon Diggs,1003,WR,7000,BUF@KC,BUF,17.1\n"
        "TE,Dalton Kincaid (1004),Dalton Kincaid,1004,TE,4500,BUF@KC,BUF,10.0\n"
        "DST,Bills  (1005),Bills ,1005,DST,3200,BUF@KC,BUF,7.0\n"
    )
    template_csv = (
        "Entry ID,Contest ID,Contest Name,Entry Fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DST\n"
        "1234567,9876543,NFL $100K Flea Flicker [$15 Entry],$15.00,,,,,,,,,\n"
        "1234568,9876543,NFL $100K Flea Flicker [$15 Entry],$15.00,,,,,,,,,\n"
    )
    return players_csv, template_csv


def test_inspect_csv_data_draftkings_classic(mock_dk_csv_content):
    players_csv, template_csv = mock_dk_csv_content
    result = _inspect_csv_data(players_csv, template_csv)
    assert result["success"] is True
    assert result["platform"] == "draftkings"
    assert result["is_single_game"] is False
    assert result["entry_fee"] == 15.0
    assert result["num_entries"] == 2
    assert result["num_players"] == 5
    assert len(result["sample_players"]) == 5
    assert result["sample_players"][0]["name"] == "Josh Allen"


def test_inspect_csv_data_draftkings_showdown():
    showdown_players = (
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
        "CPT,Patrick Mahomes (2001),Patrick Mahomes,2001,CPT/FLEX,15000,KC@DEN,KC,30.0\n"
        "FLEX,Patrick Mahomes (2002),Patrick Mahomes,2002,CPT/FLEX,10000,KC@DEN,KC,20.0\n"
    )
    template = "Entry ID,Contest ID,Contest Name,CPT,FLEX,FLEX,FLEX,FLEX,FLEX\n1,100,Test Showdown,,,,,\n"
    result = _inspect_csv_data(showdown_players, template)
    assert result["platform"] == "draftkings"
    assert result["is_single_game"] is True
    assert result["num_entries"] == 1


def test_inspect_csv_data_fanduel():
    fd_players = (
        "Id,Position,First Name,Last Name,Nickname,Salary,FPPG,Team,Opponent,Injury Indicator\n"
        "101,QB,Patrick,Mahomes,Patrick Mahomes,8400,22.4,KC,BUF,\n"
        "102,RB,Isiah,Pacheco,Isiah Pacheco,6800,14.2,KC,BUF,\n"
    )
    # Template with entry_fee, 2 real entries, and trailing player list reference rows with empty entry_id
    fd_template = (
        'entry_id,contest_id,contest_name,entry_fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF,"","Instructions"\n'
        '"3743770838","135023-280945024","$35K Sun NFL Pooch Punt","0.05","","","","","","","","","","","1) Edit"\n'
        '"3743770840","135023-280945024","$35K Sun NFL Pooch Punt","0.05","","","","","","","","","","","2) Edit"\n'
        '"","","","","","","","","","","","","","",101:Patrick Mahomes\n'
    )
    result = _inspect_csv_data(fd_players, fd_template)
    assert result["platform"] == "fanduel"
    assert result["is_single_game"] is False
    assert result["entry_fee"] == 0.05
    assert result["num_entries"] == 2
    assert result["num_players"] == 2


def test_compute_export_summary(tmp_path: Path):
    out_file = tmp_path / "Completed-Template.csv"
    out_file.write_text(
        "Entry ID,Contest ID,Contest Name,Entry Fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DST\n"
        "1,100,Contest,$1.00,Josh Allen (1001),James Cook (1002),Breece Hall,Stefon Diggs,Garrett Wilson,Davante Adams,Travis Kelce,Kenneth Walker,Bills (1005)\n"
        "2,100,Contest,$1.00,Josh Allen (1001),Christian McCaffrey,Breece Hall,Stefon Diggs,CeeDee Lamb,A.J. Brown,George Kittle,Bijan Robinson,Jets (1006)\n"
    )
    summary = _compute_export_summary(out_file, target_k=2)
    assert summary["num_lineups"] == 2
    assert summary["unique_players_count"] > 5
    top_exposures = summary["top_exposures"]
    josh_allen = next((p for p in top_exposures if p["name"] == "Josh Allen"), None)
    assert josh_allen is not None
    assert josh_allen["count"] == 2
    assert josh_allen["exposure_pct"] == 100.0

    # Test FanDuel ID:Name format (e.g. 135023-169410:Quinshon Judkins)
    fd_out = tmp_path / "Completed-FD.csv"
    fd_out.write_text(
        "entry_id,contest_id,contest_name,entry_fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF\n"
        "1,100,Contest,$1.00,135023-1001:Jared Goff,135023-1002:Quinshon Judkins,135023-1003:Braelon Allen,135023-1004:Amon-Ra St. Brown,135023-1005:Nico Collins,135023-1006:Romeo Doubs,135023-1007:Tucker Kraft,135023-1008:Christian McCaffrey,135023-1009:Houston Texans\n"
    )
    fd_summary = _compute_export_summary(fd_out, target_k=1)
    judkins = next((p for p in fd_summary["top_exposures"] if p["name"] == "Quinshon Judkins"), None)
    assert judkins is not None
    assert judkins["name"] == "Quinshon Judkins"
    assert judkins["count"] == 1


def test_http_get_index_and_static_files():
    # 1. GET /
    status, headers, body = simulate_request("GET", "/")
    assert status == 200
    assert "text/html" in headers.get("content-type", "")
    content = body.decode("utf-8")
    assert "DFS Optimizer" in content
    assert "Contest Ingestion Hub" in content

    # 2. GET /static/style.css
    status_css, headers_css, body_css = simulate_request("GET", "/static/style.css")
    assert status_css == 200
    assert "text/css" in headers_css.get("content-type", "")
    assert len(body_css) > 0

    # 3. GET /static/app.js
    status_js, headers_js, body_js = simulate_request("GET", "/static/app.js")
    assert status_js == 200
    assert "javascript" in headers_js.get("content-type", "")
    assert len(body_js) > 0

    # 4. GET nonexistent static
    status_404, _, _ = simulate_request("GET", "/static/non_existent_file.xyz")
    assert status_404 == 404


def test_http_api_inspect(mock_dk_csv_content):
    players_csv, template_csv = mock_dk_csv_content
    payload = json.dumps(
        {
            "players_content": players_csv,
            "template_content": template_csv,
        }
    ).encode("utf-8")

    status, headers, body = simulate_request(
        "POST",
        "/api/inspect",
        body=payload,
        headers={"Content-Type": "application/json"},
    )
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["platform"] == "draftkings"
    assert data["entry_fee"] == 15.0
    assert data["num_entries"] == 2


def test_http_api_progress_and_download(tmp_path: Path):
    # 1. Missing job returns 404
    status_404, _, _ = simulate_request("GET", "/api/progress?job_id=missing_id")
    assert status_404 == 404

    # 2. Populate mock job in JOBS
    job_id = "test-job-progress-123"
    dummy_csv = tmp_path / "Completed-DKEntries.csv"
    dummy_csv.write_text("Entry ID,Contest ID,QB,RB\n1,10,QB1,RB1\n")

    with _JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "completed",
            "progress": 100,
            "stage": "Finished",
            "logs": ["Done"],
            "output_path": str(dummy_csv),
            "output_filename": "Completed-DKEntries.csv",
            "summary": {"num_lineups": 1, "unique_players_count": 2, "top_exposures": []},
            "error": None,
        }

    # 3. GET /api/progress for existing job
    status_prog, _, body_prog = simulate_request("GET", f"/api/progress?job_id={job_id}")
    assert status_prog == 200
    data = json.loads(body_prog.decode("utf-8"))
    assert data["status"] == "completed"
    assert data["progress"] == 100

    # 4. GET /api/download for existing job
    status_dl, headers_dl, body_dl = simulate_request("GET", f"/api/download?job_id={job_id}")
    assert status_dl == 200
    assert "attachment" in headers_dl.get("content-disposition", "")
    assert body_dl.decode("utf-8") == "Entry ID,Contest ID,QB,RB\n1,10,QB1,RB1\n"


def test_execute_optimization_job_mock(tmp_path: Path):
    """Tests the background worker execution flow with a mocked pipeline runner."""
    job_id = "unit-test-job-999"
    with _JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": 0,
            "stage": "Queued",
            "logs": [],
            "output_path": None,
            "output_filename": None,
            "summary": None,
            "error": None,
        }

    mock_out = tmp_path / "Completed-Entries.csv"
    mock_out.write_text("Entry ID,Contest ID,QB,RB\n1,10,Josh Allen,James Cook\n")

    payload = {
        "players_content": "Position,Name + ID,Salary\nQB,Josh Allen (1001),8000\n",
        "template_content": "Entry ID,Contest ID\n1,10\n",
        "projections_content": "Player,Projection\nJosh Allen,24.5\n",
        "template_filename": "Entries.csv",
        "platform": "draftkings",
        "options": {
            "num_lineups": 1,
        },
    }

    with patch("src.build_draftkings_lineups.run_draftkings_pipeline", return_value=mock_out):
        _execute_optimization_job(job_id, tmp_path, payload)

    with _JOBS_LOCK:
        job = JOBS[job_id]
        assert job["status"] == "completed"
        assert job["progress"] == 100
        assert job["output_path"] == str(mock_out)
        assert job["summary"]["num_lineups"] == 1


def test_http_api_optimize_endpoint_requires_all_files():
    """Verifies POST /api/optimize returns 400 when any required file is missing."""
    # Missing projections
    payload_no_proj = json.dumps(
        {
            "players_content": "Position,Name + ID,Salary\nQB,Josh Allen (1001),8000\n",
            "template_content": "Entry ID,Contest ID\n1,10\n",
            "projections_content": "",
            "template_filename": "DKEntries.csv",
            "platform": "draftkings",
            "options": {"num_lineups": 1},
        }
    ).encode("utf-8")

    status, _, _ = simulate_request(
        "POST",
        "/api/optimize",
        body=payload_no_proj,
        headers={"Content-Type": "application/json"},
    )
    assert status == 400

    # Missing template
    payload_no_temp = json.dumps(
        {
            "players_content": "Position,Name + ID,Salary\nQB,Josh Allen (1001),8000\n",
            "template_content": "",
            "projections_content": "Player,Projection\nJosh Allen,24.5\n",
        }
    ).encode("utf-8")

    status, _, _ = simulate_request(
        "POST",
        "/api/optimize",
        body=payload_no_temp,
        headers={"Content-Type": "application/json"},
    )
    assert status == 400


def test_http_api_optimize_endpoint_success():
    """Verifies POST /api/optimize creates a job and returns job_id when all files are present."""
    payload = json.dumps(
        {
            "players_content": "Position,Name + ID,Salary\nQB,Josh Allen (1001),8000\n",
            "template_content": "Entry ID,Contest ID\n1,10\n",
            "projections_content": "Player,Projection\nJosh Allen,24.5\n",
            "template_filename": "DKEntries.csv",
            "platform": "draftkings",
            "options": {"num_lineups": 1},
        }
    ).encode("utf-8")

    with patch("threading.Thread.start"):
        status, _, body = simulate_request(
            "POST",
            "/api/optimize",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        data = json.loads(body.decode("utf-8"))
        assert data["success"] is True
        assert "job_id" in data
