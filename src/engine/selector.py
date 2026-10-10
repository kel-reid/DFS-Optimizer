"""
================================================================================
Portfolio Selector & Risk Audit Module (src/engine/selector.py)
================================================================================
Selects optimal K = 150 lineups maximizing Simulated ROI subject to exposure caps,
and provides post-solve risk and correlation auditing.
================================================================================
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Sequence, Set

import numpy as np
from pydfs_lineup_optimizer import Lineup, Player

from src.config import BaseOptimizerConfig

logger = logging.getLogger("FanDuelSimOptimizer")


class PortfolioSelector:
    """
    Selects the optimal K = 150 lineups from the simulated candidates
    maximizing expected Simulated ROI while strictly enforcing global exposure caps.
    """

    def __init__(self, config: BaseOptimizerConfig) -> None:
        self.config = config

    def select_portfolio(
        self,
        candidates: List[Lineup],
        sim_roi: np.ndarray,
        win_counts: np.ndarray,
        top1_rates: np.ndarray,
    ) -> List[Lineup]:
        """
        Greedily selects K lineups sorted by ROI descending, respecting position
        exposure limits:
          QB:  <= max_qb_exposure (e.g. 25% = 37 lineups)
          DEF: <= max_def_exposure (e.g. 20% = 30 lineups)
          Skill (RB, WR, TE): <= max_exposure (e.g. 25% = 37 lineups)
        """
        K = self.config.num_selected_lineups
        print()
        logger.info("=" * 70)
        logger.info("STAGE 4: PORTFOLIO SELECTION & EXPOSURE OPTIMIZATION (K = %d)", K)
        logger.info("=" * 70)

        # Sort candidate indices by ROI descending (secondary key: top1_rates)
        sort_order = np.lexsort((-top1_rates, -sim_roi))

        is_single_game = getattr(self.config, "is_single_game", False)
        sg_default_cap = getattr(self.config, "single_game_max_exposure", 0.65)
        classic_default = 0.25
        classic_def_default = 0.20

        def get_cap(player: Player) -> int:
            pos = set(player.positions)
            if "D" in pos or "DST" in pos:
                cap_rate = self.config.max_def_exposure
                if is_single_game and cap_rate == classic_def_default:
                    cap_rate = sg_default_cap
            elif "QB" in pos:
                cap_rate = self.config.max_qb_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            elif "RB" in pos:
                cap_rate = self.config.max_rb_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            elif "WR" in pos:
                cap_rate = self.config.max_wr_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            elif "TE" in pos:
                cap_rate = self.config.max_te_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            else:
                cap_rate = self.config.max_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap

            return math.floor(K * cap_rate)

        selected: List[Lineup] = []
        player_usage: Counter[str] = Counter()

        # Pass 1: Strict adherence to all exposure caps
        selected_indices: Set[int] = set()
        for idx in sort_order:
            lineup = candidates[idx]
            if all(player_usage[p.full_name] < get_cap(p) for p in lineup.lineup):
                selected.append(lineup)
                selected_indices.add(idx)
                for p in lineup.lineup:
                    player_usage[p.full_name] += 1
                if len(selected) == K:
                    break

        # Pass 2: Guaranteed fill to exactly K lineups minimizing exposure cap violations
        if len(selected) < K:
            if self.config.strict_exposure_caps:
                raise ValueError(
                    f"Strict exposure caps enforced: could only select {len(selected)} / {K} lineups "
                    f"strictly adhering to all position exposure caps. Increase candidate pool size "
                    f"(--num-candidates) or relax exposure caps."
                )
            needed = K - len(selected)
            logger.info(
                "Pass 1 selected %d / %d lineups under hard caps. Selecting remaining %d minimizing cap violations...",
                len(selected),
                K,
                needed,
            )
            remaining_indices = [idx for idx in sort_order if idx not in selected_indices]
            while len(selected) < K and remaining_indices:
                best_cand_idx = None
                best_penalty = float("inf")
                best_roi = -float("inf")
                best_pos = -1

                for pos, c_idx in enumerate(remaining_indices):
                    lineup = candidates[c_idx]
                    overage = sum(max(0, player_usage[p.full_name] + 1 - get_cap(p)) for p in lineup.lineup)
                    roi = sim_roi[c_idx]
                    if (overage < best_penalty) or (overage == best_penalty and roi > best_roi):
                        best_penalty = overage
                        best_roi = roi
                        best_cand_idx = c_idx
                        best_pos = pos

                if best_cand_idx is not None:
                    lineup = candidates[best_cand_idx]
                    selected.append(lineup)
                    selected_indices.add(best_cand_idx)
                    for p in lineup.lineup:
                        player_usage[p.full_name] += 1
                    remaining_indices.pop(best_pos)
                else:
                    break

        logger.info("Successfully assembled final portfolio of %d lineups.", len(selected))
        if len(selected) != K:
            raise ValueError(f"Could not assemble required {K} lineups from candidate pool (assembled {len(selected)}).")
        return selected


class SimAuditReporter:
    """Performs post-solve portfolio risk, correlation, and simulation audit."""

    @staticmethod
    def audit_and_report(
        lineups: Sequence[Lineup],
        config: BaseOptimizerConfig,
        players: List[Player],
        sim_roi: np.ndarray,
        win_counts: np.ndarray,
        top1_rates: np.ndarray,
    ) -> None:
        total = len(lineups)
        player_counts: Counter[str] = Counter()
        player_info: Dict[str, Player] = {p.full_name: p for p in players}

        for l in lineups:
            for p in l.lineup:
                player_counts[p.full_name] += 1

        print()
        logger.info("=" * 70)
        logger.info("PORTFOLIO AUDIT & SIMULATION ANALYSIS (%d LINEUPS)", total)
        logger.info("=" * 70)

        # 1. Salary and Stacking Audit
        stacked_count = 0
        unconstrained_count = 0
        salary_compliant = 0
        def_opp_violations = 0

        for l in lineups:
            p_list = l.lineup
            qb = next((p for p in p_list if "QB" in p.positions), None)
            defs = [p for p in p_list if ("D" in p.positions or "DST" in p.positions)]
            wr_te_teams = {p.team for p in p_list if ("WR" in p.positions or "TE" in p.positions)}
            off_teams = {p.team for p in p_list if ("D" not in p.positions and "DST" not in p.positions)}

            if sum(p.salary for p in p_list) <= config.salary_cap:
                salary_compliant += 1

            if qb and qb.team in wr_te_teams:
                stacked_count += 1
            else:
                unconstrained_count += 1

            for d in defs:
                if d.game_info and d.game_info.home_team and d.game_info.away_team:
                    opp_team = d.game_info.away_team if d.team == d.game_info.home_team else d.game_info.home_team
                    if opp_team in off_teams:
                        def_opp_violations += 1

        is_single_game = getattr(config, "is_single_game", False)
        sg_default_cap = getattr(config, "single_game_max_exposure", 0.65)
        classic_default = 0.25
        classic_def_default = 0.20

        # Evaluate exposure cap compliance across all rostered players
        def get_pos_cap(p: Player) -> int:
            pos = set(p.positions)
            if "D" in pos or "DST" in pos:
                cap_rate = config.max_def_exposure
                if is_single_game and cap_rate == classic_def_default:
                    cap_rate = sg_default_cap
            elif "QB" in pos:
                cap_rate = config.max_qb_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            elif "RB" in pos:
                cap_rate = config.max_rb_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            elif "WR" in pos:
                cap_rate = config.max_wr_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            elif "TE" in pos:
                cap_rate = config.max_te_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap
            else:
                cap_rate = config.max_exposure
                if is_single_game and cap_rate == classic_default:
                    cap_rate = sg_default_cap

            return math.floor(total * cap_rate)

        cap_violations = []
        for name, count in player_counts.items():
            p_obj = player_info.get(name)
            if p_obj:
                cap = get_pos_cap(p_obj)
                if count > cap:
                    cap_violations.append((name, count, cap))

        logger.info("✓ CONSTRAINTS AUDIT:")
        logger.info(
            "  - %d / %d (%.1f%%) of lineups comply with $%d salary cap.",
            salary_compliant,
            total,
            (salary_compliant / total) * 100,
            config.salary_cap,
        )
        logger.info(
            "  - Primary Stacks: %d / %d (%.1f%%) feature QB + WR/TE same-team stack.",
            stacked_count,
            total,
            (stacked_count / total) * 100,
        )
        logger.info(
            "  - Standalone Rushing QBs: %d / %d (%.1f%%) feature unconstrained rosters.",
            unconstrained_count,
            total,
            (unconstrained_count / total) * 100,
        )
        logger.info(
            "  - Opposing Defense Violations: %d / %d (%.1f%%).",
            def_opp_violations,
            total,
            (def_opp_violations / total) * 100,
        )
        if cap_violations:
            logger.warning(
                "  - Exposure Cap Overages: %d player(s) exceeded configured caps (via Pass 2 emergency fill):",
                len(cap_violations),
            )
            for name, count, cap in cap_violations:
                logger.warning("    * %s: %d lineups (cap: %d)", name, count, cap)
        else:
            logger.info("  - Exposure Cap Overages: 0 players exceeded position ceilings (100.0%% compliant).")

        # 2. QB Distribution Table
        qb_counts = Counter(p.full_name for l in lineups for p in l.lineup if "QB" in p.positions)
        print()
        logger.info("-" * 70)
        logger.info("STARTING QB EXPOSURE DISTRIBUTION:")
        for qb_name, count in qb_counts.most_common():
            pct = (count / total) * 100
            logger.info("  QB %-28s : %3d lineups (%5.1f%%)", qb_name, count, pct)

        # 3. Top Exposures Table
        print()
        logger.info("-" * 70)
        logger.info("POST-SIMULATION TOP 15 PLAYER EXPOSURES:")
        for name, count in player_counts.most_common(15):
            pct = (count / total) * 100
            p_obj = player_info.get(name)
            team = p_obj.team if p_obj else "NFL"
            pos = "/".join(p_obj.positions) if p_obj else "POS"
            logger.info("  %-25s (%s - %s) : %3d / %3d (%5.1f%%)", name, team, pos, count, total, pct)

        # 4. Simulation ROI & Equity Metrics
        print()
        logger.info("-" * 70)
        logger.info("PORTFOLIO SIMULATION METRICS (%s MONTE CARLO TRIALS):", f"{config.num_sim_trials:,}")
        logger.info("  Candidate Pool Mean Sim ROI : %+.1f%%", float(np.mean(sim_roi)))
        logger.info("  Top-Ranked Candidate Sim ROI: %+.1f%%", float(np.max(sim_roi)))
        total_wins = int(np.sum(win_counts))
        avg_wins = total_wins / config.num_sim_trials
        logger.info("  1st-Place Finishes          : %s total (avg %.2f candidate wins / trial)", f"{total_wins:,}", avg_wins)
        logger.info("=" * 70)
