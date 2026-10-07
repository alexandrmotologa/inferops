"""SQLite token accounting, cost estimation, and API key management."""

import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class UsageRecord:
    id: int
    timestamp: float
    api_key_id: Optional[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_savings_usd: float


@dataclass
class ApiKeyInfo:
    key_id: str
    name: str
    rate_limit_rpm: int
    created_at: float
    revoked: bool


class TokenAccountingManager:
    """Manages SQLite storage for token usage analytics and API key access control."""

    # Baseline commercial benchmark rates per 1,000,000 tokens (e.g. GPT-4o equivalent)
    BENCHMARK_PROMPT_RATE_PER_M = 2.50
    BENCHMARK_COMPLETION_RATE_PER_M = 10.00

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
                    created_at REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    api_key_id TEXT,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    estimated_savings_usd REAL NOT NULL,
                    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON token_usage(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_model ON token_usage(model)")
            conn.commit()

    def calculate_savings(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost savings compared to commercial closed LLM pricing."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.BENCHMARK_PROMPT_RATE_PER_M
        completion_cost = (completion_tokens / 1_000_000.0) * self.BENCHMARK_COMPLETION_RATE_PER_M
        return round(prompt_cost + completion_cost, 6)

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        api_key_id: Optional[str] = None,
    ) -> UsageRecord:
        """Store a verified token usage record."""
        now = time.time()
        total_tokens = prompt_tokens + completion_tokens
        savings = self.calculate_savings(prompt_tokens, completion_tokens)

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO token_usage (timestamp, api_key_id, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, estimated_savings_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now, api_key_id, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, savings),
            )
            conn.commit()
            rec_id = cur.lastrowid or 0

        return UsageRecord(
            id=rec_id,
            timestamp=now,
            api_key_id=api_key_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_savings_usd=savings,
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Aggregate total tokens served, total requests, and total savings."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_requests,
                    COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) as grand_total_tokens,
                    COALESCE(SUM(estimated_savings_usd), 0.0) as total_savings_usd,
                    COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
                FROM token_usage
                """
            ).fetchone()

            model_rows = conn.execute(
                """
                SELECT
                    model,
                    COUNT(*) as requests,
                    COALESCE(SUM(total_tokens), 0) as tokens,
                    COALESCE(SUM(estimated_savings_usd), 0.0) as savings
                FROM token_usage
                GROUP BY model
                ORDER BY tokens DESC
                """
            ).fetchall()

        return {
            "total_requests": row["total_requests"] if row else 0,
            "total_prompt_tokens": row["total_prompt_tokens"] if row else 0,
            "total_completion_tokens": row["total_completion_tokens"] if row else 0,
            "grand_total_tokens": row["grand_total_tokens"] if row else 0,
            "total_savings_usd": round(row["total_savings_usd"], 4) if row else 0.0,
            "avg_latency_ms": round(row["avg_latency_ms"], 2) if row else 0.0,
            "by_model": [dict(r) for r in model_rows],
        }

    # API Key Authentication
    def _hash_key(self, raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def create_api_key(self, name: str, rate_limit_rpm: int = 60) -> Tuple[str, ApiKeyInfo]:
        """Generate a cryptographically secure API key and persist its hash."""
        raw_token = f"sk-inferops-{secrets.token_urlsafe(24)}"
        key_id = f"key_{secrets.token_hex(6)}"
        key_hash = self._hash_key(raw_token)
        now = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_id, key_hash, name, rate_limit_rpm, created_at, revoked)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (key_id, key_hash, name, rate_limit_rpm, now),
            )
            conn.commit()

        info = ApiKeyInfo(
            key_id=key_id,
            name=name,
            rate_limit_rpm=rate_limit_rpm,
            created_at=now,
            revoked=False,
        )
        return raw_token, info

    def verify_api_key(self, raw_key: str) -> Optional[ApiKeyInfo]:
        """Validate API key token against database."""
        key_hash = self._hash_key(raw_key)
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT key_id, name, rate_limit_rpm, created_at, revoked
                FROM api_keys
                WHERE key_hash = ? AND revoked = 0
                """,
                (key_hash,),
            ).fetchone()

            if not row:
                return None

            return ApiKeyInfo(
                key_id=row["key_id"],
                name=row["name"],
                rate_limit_rpm=row["rate_limit_rpm"],
                created_at=row["created_at"],
                revoked=bool(row["revoked"]),
            )

    def list_api_keys(self) -> List[ApiKeyInfo]:
        """Retrieve all active and revoked API keys."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT key_id, name, rate_limit_rpm, created_at, revoked FROM api_keys ORDER BY created_at DESC"
            ).fetchall()

            return [
                ApiKeyInfo(
                    key_id=r["key_id"],
                    name=r["name"],
                    rate_limit_rpm=r["rate_limit_rpm"],
                    created_at=r["created_at"],
                    revoked=bool(r["revoked"]),
                )
                for r in rows
            ]

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key by its key_id."""
        with self._get_connection() as conn:
            cur = conn.execute("UPDATE api_keys SET revoked = 1 WHERE key_id = ?", (key_id,))
            conn.commit()
            return cur.rowcount > 0
