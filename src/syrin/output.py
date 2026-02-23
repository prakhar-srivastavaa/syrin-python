"""Output configuration for structured output validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

import builtins

from syrin.types.validation import OutputValidator


@dataclass
class Output:
    """Configuration for structured output validation.

    Groups all structured output options in one place for cleaner API.

    Example:
        agent = Agent(
            model=Model.OpenAI("gpt-4o"),
            output=Output(
                type=UserInfo,
                validation_retries=3,
                context={"allowed_domains": ["company.com"]},
            ),
        )

    Or use shorthand:
        agent = Agent(
            model=Model.OpenAI("gpt-4o"),
            output=Output(UserInfo),  # Just the type
        )
    """

    type: builtins.type[BaseModel] | None = None
    """Pydantic model to validate output against."""

    validation_retries: int = 3
    """Number of validation retry attempts (default: 3)."""

    context: dict[str, Any] = field(default_factory=dict)
    """Context passed to validators for dynamic validation."""

    validator: OutputValidator | None = None
    """Custom output validator for business logic."""

    strict: bool = False
    """Use strict validation mode."""

    def __init__(
        self,
        output_type: builtins.type[BaseModel] | None = None,
        *,
        validation_retries: int = 3,
        context: dict[str, Any] | None = None,
        validator: OutputValidator | None = None,
        strict: bool = False,
    ):
        self.type = output_type
        self.validation_retries = validation_retries
        self.context = context or {}
        self.validator = validator
        self.strict = strict
