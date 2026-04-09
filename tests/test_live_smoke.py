import os

import pytest

from test_live import run_smoke_test


@pytest.mark.integration
def test_live_smoke():
    if os.getenv("RUN_LIVE_SMOKE") != "1":
        pytest.skip("set RUN_LIVE_SMOKE=1 to execute the live HF Space smoke test")

    report = run_smoke_test()
    assert report["failed"] == 0
