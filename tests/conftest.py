import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture()
def eval_run_db_path():
    """Return an isolated SQLite database path for eval run storage tests."""
    base = Path(__file__).resolve().parent.parent / ".pytest-eval-run-db"
    test_dir = base / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    db_path = test_dir / "eval_runs.db"
    yield str(db_path)
    shutil.rmtree(test_dir, ignore_errors=True)
    try:
        base.rmdir()
    except OSError:
        pass
