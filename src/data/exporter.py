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

        lineup_size = len(list(lineups[0].lineup)) if lineups else 0

        with open(template_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            raw_header = next(reader)

            # Locate where position columns end (stop at instructions or blank trailing columns)
            pos_end = len(raw_header)
            for idx in range(4, len(raw_header)):
                col = raw_header[idx].strip()
                if not col or "instruction" in col.lower():
                    pos_end = idx
                    break

            # Exact number of columns matching contest format (4 metadata + lineup_size roster slots)
            num_cols = min(pos_end, 4 + lineup_size) if lineup_size > 0 else pos_end
            header = raw_header[:num_cols]
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
            "Loaded template with pandas: %d reserved entries, %d columns (%d roster positions).",
            len(df),
            len(df.columns),
            len(df.columns) - 4,
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
