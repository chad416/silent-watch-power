import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "req(*ids): DOSSIER requirement ids this test verifies")
    config.addinivalue_line("markers", "it(id): integration test id")
