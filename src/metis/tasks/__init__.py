"""Built-in task decisions, independent of optional model dependencies.

Tensor objectives are available from ``metis.tasks.objectives`` and require
PyTorch. They are not imported by this package's lightweight entry point.
"""

from .decisions import decide

__all__ = ["decide"]
