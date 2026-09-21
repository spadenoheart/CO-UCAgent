"""Load baseline token telemetry only in the selected UCAgent entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path


raw_target = os.environ.get("UCAGENT_BASELINE_TOKEN_TARGET", "")
raw_log = os.environ.get("UCAGENT_BASELINE_TOKEN_LOG", "")
if raw_target and raw_log and sys.argv and sys.argv[0] not in {"", "-c", "-m"}:
    try:
        is_target = Path(sys.argv[0]).resolve() == Path(raw_target).resolve()
    except OSError:
        is_target = False
    if is_target:
        from baseline_token_meter import install

        install()
