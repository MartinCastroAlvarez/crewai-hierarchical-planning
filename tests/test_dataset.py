from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hierarchical_crew.dataset import BUNDLED_DIR
from hierarchical_crew.dataset import DatasetError
from hierarchical_crew.dataset import MarketData


def copy_bundled(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    for name in ("products.csv", "segments.csv", "competitors.csv"):
        shutil.copy(BUNDLED_DIR / name, root / name)
    return root


def test_bundled_dataset(market: MarketData) -> None:
    assert market.product_names() == ["Aurora", "Tidal", "Lumen"]
    assert len(market.rows("segments")) == 8
    band = market.price_band("air-quality monitor")
    assert band is not None
    assert (band.low, band.median, band.high, band.count) == (79.0, 139.0, 249.0, 5)
    assert market.price_band("toaster") is None
    assert market.product("aurora") is not None
    assert market.product("nope") is None


def test_search_is_case_insensitive_and_capped(market: MarketData) -> None:
    assert len(market.search("competitors", "QUILLON")) == 1
    assert len(market.search("segments", "", limit=3)) == 3
    with pytest.raises(DatasetError):
        market.rows("secrets")


def test_custom_folder_loads(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    assert MarketData.load(root).root == root.resolve()


def test_symlinked_file_is_refused(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    outside = tmp_path / "outside.csv"
    shutil.copy(root / "products.csv", outside)
    (root / "products.csv").unlink()
    (root / "products.csv").symlink_to(outside)
    with pytest.raises(DatasetError, match="symlink"):
        MarketData.load(root)


def test_symlinked_root_is_refused(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(DatasetError, match="symlink"):
        MarketData.load(link)


def test_directory_in_place_of_a_file_is_refused(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    (root / "segments.csv").unlink()
    (root / "segments.csv").mkdir()
    with pytest.raises(DatasetError, match="regular file"):
        MarketData.load(root)


def test_wrong_columns_and_size_limits(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    (root / "products.csv").write_text("name,secret\nx,y\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="expected columns"):
        MarketData.load(root)
    (root / "products.csv").write_text("x" * 1_000_001, encoding="utf-8")
    with pytest.raises(DatasetError, match="larger than"):
        MarketData.load(root)


def test_too_many_rows(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    rows = "".join(f"S{i},P{i},cat,1.00,d\n" for i in range(5_001))
    (root / "products.csv").write_text("sku,name,category,unit_cost_usd,description\n" + rows)
    with pytest.raises(DatasetError, match="rows"):
        MarketData.load(root)


def test_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        MarketData.load(tmp_path / "missing")


def test_long_cells_are_truncated(tmp_path: Path) -> None:
    root = copy_bundled(tmp_path)
    text = (root / "products.csv").read_text().replace("Indoor monitor", "I" * 1000, 1)
    (root / "products.csv").write_text(text)
    assert len(MarketData.load(root).rows("products")[0]["description"]) == 300
