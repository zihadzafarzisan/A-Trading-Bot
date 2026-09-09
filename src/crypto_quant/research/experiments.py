"""Experiment tracking.

Persists every research experiment so results are reproducible: an experiment
records its full configuration (strategy, data range, assets, market, risk,
execution) and every candidate evaluation. Rerunning the same configuration is
possible because nothing is stored that cannot be reconstructed.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db.connection import DatabaseManager
from ..db.models import Experiment, ExperimentResult, ResearchCandidate
from ..logging_config import get_logger
from ..utils.helpers import generate_experiment_id

logger = get_logger("research")


@dataclass
class CandidateRecord:
    """Record of one evaluated candidate."""

    strategy_type: str
    params: Dict[str, Any]
    symbol: str
    timeframe: str
    market_type: str
    metrics: Dict[str, Any]
    rank: Optional[int] = None
    passed_filters: bool = False


class ExperimentTracker:
    """Creates, updates, and queries research experiments."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ------------------------------------------------------------ creation
    def create_experiment(
        self,
        config: Dict[str, Any],
        start_date: str,
        end_date: str,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Create a new experiment row and return its ID."""
        exp_id = generate_experiment_id()
        session = self.db.get_session()
        try:
            exp = Experiment(
                id=exp_id,
                strategy_id=strategy_id or f"RESEARCH-{exp_id[-6:]}",
                config=json.dumps(config, default=str),
                start_date=start_date,
                end_date=end_date,
                status="running",
            )
            session.add(exp)
            session.commit()
            return exp_id
        finally:
            session.close()

    def update_status(self, experiment_id: str, status: str) -> None:
        """Set experiment status (running/completed/failed)."""
        session = self.db.get_session()
        try:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.status = status
                if status == "completed":
                    exp.completed_at = datetime.now(timezone.utc)
                session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------ candidates
    def save_candidates(self, experiment_id: str, candidates: List[CandidateRecord]) -> int:
        """Persist candidate evaluations. Returns rows written."""
        session = self.db.get_session()
        written = 0
        try:
            for c in candidates:
                session.add(ResearchCandidate(
                    experiment_id=experiment_id,
                    strategy_type=c.strategy_type,
                    params=json.dumps(c.params, default=str),
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    market_type=c.market_type,
                    metrics=json.dumps(c.metrics, default=str),
                    rank=c.rank,
                    passed_filters=c.passed_filters,
                ))
                written += 1
            session.commit()
            return written
        finally:
            session.close()

    def save_result_summary(self, experiment_id: str, metrics: Dict[str, Any]) -> None:
        """Store the aggregate ExperimentResult row for an experiment."""
        session = self.db.get_session()
        try:
            existing = session.query(ExperimentResult).filter(
                ExperimentResult.experiment_id == experiment_id
            ).first()
            if existing is None:
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    total_trades=int(metrics.get("total_trades", 0)),
                    winning_trades=int(metrics.get("winning_trades", 0)),
                    losing_trades=int(metrics.get("losing_trades", 0)),
                    win_rate=float(metrics.get("win_rate", 0)),
                    profit_factor=float(metrics.get("profit_factor", 0)),
                    expectancy=float(metrics.get("expectancy", 0)),
                    net_return=float(metrics.get("net_return", 0)),
                    max_drawdown=float(metrics.get("max_drawdown", 0)),
                    sharpe_ratio=float(metrics.get("sharpe_ratio", 0)),
                    sortino_ratio=float(metrics.get("sortino_ratio", 0)),
                    avg_win=float(metrics.get("avg_win", 0)),
                    avg_loss=float(metrics.get("avg_loss", 0)),
                    largest_win=float(metrics.get("largest_win", 0)),
                    largest_loss=float(metrics.get("largest_loss", 0)),
                    avg_holding_period=float(metrics.get("avg_holding_period", 0)),
                    total_fees=float(metrics.get("total_fees", 0)),
                    total_slippage=float(metrics.get("total_slippage", 0)),
                    total_funding=float(metrics.get("total_funding", 0)),
                    results_json=json.dumps(metrics, default=str),
                )
                session.add(result)
                session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------ queries
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Return an experiment and its result summary as dicts."""
        session = self.db.get_session()
        try:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp is None:
                return None
            data = {
                "id": exp.id,
                "strategy_id": exp.strategy_id,
                "config": json.loads(exp.config) if exp.config else {},
                "start_date": exp.start_date,
                "end_date": exp.end_date,
                "status": exp.status,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }
            result = session.query(ExperimentResult).filter(
                ExperimentResult.experiment_id == experiment_id
            ).first()
            data["result"] = result.to_dict() if result else None
            return data
        finally:
            session.close()

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent experiments (newest first)."""
        session = self.db.get_session()
        try:
            rows = (session.query(Experiment)
                    .order_by(Experiment.created_at.desc())
                    .limit(limit)
                    .all())
            return [{
                "id": r.id, "status": r.status, "strategy_id": r.strategy_id,
                "start_date": r.start_date, "end_date": r.end_date,
            } for r in rows]
        finally:
            session.close()

    def top_candidates(self, experiment_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the top-ranked candidates for an experiment."""
        session = self.db.get_session()
        try:
            rows = (session.query(ResearchCandidate)
                    .filter(ResearchCandidate.experiment_id == experiment_id)
                    .order_by(ResearchCandidate.rank)
                    .limit(limit)
                    .all())
            return [{
                "strategy_type": r.strategy_type,
                "params": json.loads(r.params) if r.params else {},
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "market_type": r.market_type,
                "metrics": json.loads(r.metrics) if r.metrics else {},
                "rank": r.rank,
                "passed_filters": r.passed_filters,
            } for r in rows]
        finally:
            session.close()