import shutil
import numpy as np
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Point src.data_loader.DATA_DIR at a throwaway directory for the test,
    so tests never touch real trades_*.jsonl files."""
    from src import data_loader
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    yield tmp_path


@pytest.fixture
def sample_trades():
    return np.array([1.5, -1.0, 0.8, -0.5, 2.0, -0.3, 1.2])