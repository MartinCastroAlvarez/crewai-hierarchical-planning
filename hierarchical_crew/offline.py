"""Switch off every outbound call CrewAI can make on its own, before it is imported.

CrewAI ships anonymous usage telemetry (OpenTelemetry spans to its collector), an
opt-in execution tracing service and a version check. This project promises zero
outbound calls, so the package sets the documented opt-outs at import time:

* ``CREWAI_DISABLE_TELEMETRY=true`` and ``OTEL_SDK_DISABLED=true`` - CrewAI's own
  telemetry checks either of them (plus ``CREWAI_DISABLE_TRACKING``).
* ``CREWAI_TRACING_ENABLED=false`` - an explicit "no" to CrewAI's trace upload,
  which also suppresses the first-run "enable tracing?" prompt.
* ``CREWAI_DISABLE_VERSION_CHECK=true`` - no PyPI lookup for a newer release.

They are forced (not ``setdefault``): a stray ``CREWAI_TRACING_ENABLED=true`` in the
environment would otherwise ship prompts and outputs to a hosted service. The only
network peer left is the model endpoint you configure (loopback ollama by default).
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

__all__ = ["OFFLINE_ENV", "disable_telemetry"]

OFFLINE_ENV: dict[str, str] = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_DISABLE_VERSION_CHECK": "true",
}


def disable_telemetry(env: MutableMapping[str, str] | None = None) -> None:
    """Force the opt-out variables into ``env`` (default: ``os.environ``)."""
    target = os.environ if env is None else env
    for key, value in OFFLINE_ENV.items():
        target[key] = value
