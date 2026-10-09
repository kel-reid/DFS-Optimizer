"""
================================================================================
FanDuel Template Exporter Module (src/data/exporter.py)
================================================================================
Reads target contest upload template, slices to reserved entries, formats
players into 'PlayerID:PlayerName', and exports ready-to-upload CSV.
================================================================================
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from pydfs_lineup_optimizer import Lineup

from src.config import SimOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


class FanDuelTemplateExporter:
    """
    Reads the target upload template with pandas, slices strictly to the 150
    reserved contest entries, maps the generated 150 lineups into the 9 roster
    columns ('QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF') using
    'PlayerID:PlayerName' formatting, preserves metadata, and exports CSV.
    """

    ROSTER_SLOTS: Tuple[str, ...] = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF")

    def __init__(self, config: SimOptimizerConfig) -> None:
        self.config = config

    def export_lineups(self, lineups: Sequence[Lineup]) -> Path:
        template_path = self.config.template_csv
        output_path = self.config.output_csv

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path.resolve()}")

        logger.info("Reading reserved entries template: %s", template_path)

        # Auto-detect whether Single Game (5 roster slots = 9 columns) or Classic (9 roster slots = 13 columns)
        is_single_game = (
            self.config.is_single_game
            or (lineups and any("MVP" in p.lineup_position for p in lineups[0].lineup))
        )
        num_cols = 9 if is_single_game else 13

        with open(template_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)[:num_cols]
            rows = []
            for row in reader:
                if row and row[0].strip():
                    r_slice = row[:num_cols]
                    while len(r_slice) < num_cols:
                        r_slice.append("")
                    rows.append(r_slice)
                if len(rows) == self.config.num_selected_lineups:
                    break

        df = pd.DataFrame(rows, columns=header)
        logger.info(
            "Loaded template with pandas (%s): %d reserved entries, %d columns.",
            "Single Game" if is_single_game else "Classic",
            len(df),
            len(df.columns),
        )

        if len(df) != len(lineups):
            raise ValueError(f"Row mismatch: Template has {len(df)} rows, but {len(lineups)} lineups selected.")

        # Map lineups into position columns (starting at index 4)
        for i, lineup in enumerate(lineups):
            formatted_players = [f"{p.id}:{p.full_name}" for p in lineup.lineup]
            for slot_idx, player_str in enumerate(formatted_players, start=4):
                if slot_idx < num_cols:
                    df.iat[i, slot_idx] = player_str

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Successfully populated and verified %d valid template entries.", len(df))
        return output_path
