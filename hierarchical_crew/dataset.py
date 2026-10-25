"""The market dataset, loaded from a sandboxed folder of three CSV files.

The researcher's tool reads only through :class:`MarketData`, which enforces the
sandbox rules once, at load time:

* only the three allowlisted file names are opened; the model can pick a dataset by
  name but never a path;
* the root is resolved first, symlinks are refused, and every file must resolve to a
  regular file inside the root;
* files over 1 MB and more than 5,000 rows are refused; cells are capped at 300 chars.
"""

from __future__ import annotations

import csv
import io
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

__all__ = [
    "DATASETS",
    "BUNDLED_DIR",
    "DatasetError",
    "DatasetName",
    "MarketData",
    "PriceBand",
]

DatasetName = Literal["products", "segments", "competitors"]
DATASETS: dict[DatasetName, tuple[str, ...]] = {
    "products": ("sku", "name", "category", "unit_cost_usd", "description"),
    "segments": ("category", "segment", "region", "households_k", "adoption_pct", "wtp_usd"),
    "competitors": ("category", "competitor", "product", "price_usd", "rating", "notes"),
}
BUNDLED_DIR = Path(str(resources.files("hierarchical_crew") / "data"))
MAX_FILE_BYTES = 1_000_000
MAX_ROWS = 5_000
MAX_CELL = 300

Row = dict[str, str]


class DatasetError(ValueError):
    """The dataset folder or one of its files breaks the sandbox rules."""


@dataclass(frozen=True, slots=True)
class PriceBand:
    """Competitor list prices for one category."""

    low: float
    median: float
    high: float
    count: int


def _read_inside(root: Path, filename: str) -> str:
    candidate = root / filename
    if candidate.is_symlink():
        raise DatasetError(f"{filename}: symlinks are not allowed")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise DatasetError(f"{filename}: not a regular file inside the data folder")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise DatasetError(f"{filename}: larger than {MAX_FILE_BYTES} bytes")
    return resolved.read_text(encoding="utf-8")


def _parse(name: DatasetName, text: str) -> list[Row]:
    reader = csv.DictReader(io.StringIO(text))
    expected = DATASETS[name]
    if tuple(reader.fieldnames or ()) != expected:
        raise DatasetError(f"{name}.csv: expected columns {expected}, got {reader.fieldnames}")
    rows: list[Row] = []
    for index, raw in enumerate(reader):
        if index >= MAX_ROWS:
            raise DatasetError(f"{name}.csv: more than {MAX_ROWS} rows")
        rows.append({k: (raw.get(k) or "").strip()[:MAX_CELL] for k in expected})
    return rows


class MarketData:
    """Read-only, in-memory view of the three market tables."""

    def __init__(self, tables: Mapping[DatasetName, list[Row]], root: Path) -> None:
        self._tables: dict[str, list[Row]] = {name: list(rows) for name, rows in tables.items()}
        self.root = root

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> MarketData:
        """Load the tables from ``data_dir`` (default: the bundled fictional dataset)."""
        root = Path(data_dir) if data_dir is not None else BUNDLED_DIR
        if root.is_symlink():
            raise DatasetError("the data folder itself must not be a symlink")
        root = root.resolve()
        if not root.is_dir():
            raise DatasetError(f"data folder not found: {root}")
        tables = {name: _parse(name, _read_inside(root, f"{name}.csv")) for name in DATASETS}
        return cls(tables, root)

    def rows(self, name: str) -> list[Row]:
        if name not in self._tables:
            raise DatasetError(f"unknown dataset {name!r}; use one of {sorted(DATASETS)}")
        return [dict(r) for r in self._tables[name]]

    def search(self, name: str, query: str = "", limit: int = 20) -> list[Row]:
        """Rows of ``name`` whose text contains every word of ``query`` (case-insensitive)."""
        words = [w for w in query.lower().split() if w]
        hits = [
            row
            for row in self.rows(name)
            if all(w in " ".join(row.values()).lower() for w in words)
        ]
        return hits[: max(1, min(limit, 50))]

    def product(self, name: str) -> Row | None:
        for row in self.rows("products"):
            if row["name"].lower() == name.strip().lower():
                return row
        return None

    def product_names(self) -> list[str]:
        return [row["name"] for row in self.rows("products")]

    def price_band(self, category: str) -> PriceBand | None:
        prices = [
            float(row["price_usd"])
            for row in self.rows("competitors")
            if row["category"].lower() == category.lower()
        ]
        if not prices:
            return None
        return PriceBand(min(prices), statistics.median(prices), max(prices), len(prices))
