"""CrewAI hierarchical process with planning: a product launch crew, wrapped in a Flow.

Importing this package first switches off CrewAI's telemetry, trace upload and
version check (see :mod:`hierarchical_crew.offline`), before CrewAI is imported.
"""

from __future__ import annotations

from hierarchical_crew.offline import disable_telemetry

disable_telemetry()

__all__ = ["__version__", "disable_telemetry"]
__version__ = "0.1.0"
