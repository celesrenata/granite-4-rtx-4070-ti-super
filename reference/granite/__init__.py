"""Pre-kernel reference contract package for the Granite 4.2 8B checkpoint.

Stdlib-only. See :mod:`reference.granite.contract` for the validator and the
planning constants / verified-values separation.
"""

from reference.granite.contract import (
    PLANNING,
    SCHEMA,
    VERIFIED_CHECKPOINT,
    ContractResult,
    FieldResult,
    validate_checkpoint,
)

__all__ = [
    "PLANNING",
    "SCHEMA",
    "VERIFIED_CHECKPOINT",
    "ContractResult",
    "FieldResult",
    "validate_checkpoint",
]
