"""Structured Output Example.

Demonstrates:
- @structured decorator for output schemas
- Output(output_type, validation_retries) config (use Output(MyModel) or Output(MyModel, ...))
- Custom OutputValidator with ValidationResult
- Accessing result.data, result.structured.parsed
- Validation hooks (OUTPUT_VALIDATION_START, etc.)

Run: python -m examples.05_tools.structured_output
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from examples.models.models import almock
from syrin import Agent, Output
from syrin.enums import Hook
from syrin.model import structured
from syrin.types.validation import (
    OutputValidator,
    ValidationAction,
    ValidationContext,
    ValidationResult,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Basic @structured
@structured
class UserInfo:
    name: str
    email: str
    age: int
    city: str


agent = Agent(model=almock, output=Output(UserInfo, validation_retries=3))
result = agent.response("Extract: John Doe, 35, john@example.com, San Francisco")
print(f"is_valid: {result.structured.is_valid}")
if result.structured.parsed:
    print(f"parsed.name: {result.structured.parsed.name}")


# 2. Pydantic model as output
class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool
    category: str


agent = Agent(model=almock, output=Output(ProductInfo, validation_retries=3))
result = agent.response("Product: Widget, $29.99, in stock, electronics")
print(f"is_valid: {result.structured.is_valid}")
if result.structured.parsed:
    print(f"parsed: {result.structured.parsed}")


# 3. Validation hooks
@structured
class SentimentResult:
    sentiment: str
    confidence: float
    explanation: str


agent = Agent(model=almock, output=Output(SentimentResult, validation_retries=3))


def on_start(ctx: object) -> None:
    print(f"  VALIDATION START: {ctx.output_type}")


def on_success(ctx: object) -> None:
    print(f"  VALIDATION SUCCESS at attempt {ctx.attempt}")


def on_failed(ctx: object) -> None:
    print(f"  VALIDATION FAILED: {ctx.reason}")


agent.events.on(Hook.OUTPUT_VALIDATION_START, on_start)
agent.events.on(Hook.OUTPUT_VALIDATION_SUCCESS, on_success)
agent.events.on(Hook.OUTPUT_VALIDATION_FAILED, on_failed)
result = agent.response("Analyze: 'This product is amazing!'")
print(f"is_valid: {result.structured.is_valid}")


# 4. Custom validator
class ReviewResult(BaseModel):
    rating: int
    sentiment: str
    summary: str


class RatingValidator(OutputValidator):
    max_retries = 3

    def validate(self, output: object, context: ValidationContext) -> ValidationResult:
        data = (
            output
            if isinstance(output, dict)
            else output.model_dump()
            if hasattr(output, "model_dump")
            else {}
        )
        rating = data.get("rating", 0)
        if rating < 1 or rating > 5:
            return ValidationResult.invalid(
                message=f"Rating {rating} out of range 1-5",
                action=ValidationAction.RETRY,
                hint="Rating must be between 1 and 5",
            )
        sentiment = data.get("sentiment", "").lower()
        if sentiment not in ["positive", "negative", "neutral"]:
            return ValidationResult.invalid(
                message=f"Invalid sentiment: {sentiment}",
                action=ValidationAction.RETRY,
            )
        return ValidationResult.valid(output)

    def on_retry(self, error: str, attempt: int) -> str:
        return f"Error: {error}. Please fix and retry."


agent = Agent(
    model=almock,
    output=Output(ReviewResult, validator=RatingValidator(), validation_retries=3),
)
result = agent.response("Review: 'Terrible product.' rating 1, negative")
print(f"is_valid: {result.structured.is_valid}")

# 5. Output with validation context
from pydantic import field_validator


class RestrictedUser(BaseModel):
    name: str
    email: str
    role: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        allowed = ["admin", "user", "guest"]
        if v.lower() not in allowed:
            raise ValueError(f"Role must be one of: {allowed}")
        return v.lower()


agent = Agent(
    model=almock,
    output=Output(
        RestrictedUser,
        validation_retries=3,
        context={"allowed_domains": ["company.com"]},
    ),
)
result = agent.response("Create user: John, john@company.com, admin")
print(f"is_valid: {result.structured.is_valid}")


class StructuredOutputAgent(Agent):
    _agent_name = "structured-output"
    _agent_description = "Agent with structured output (UserInfo extraction)"
    model = almock
    system_prompt = "You extract user information from text. Return valid UserInfo."
    output = Output(UserInfo, validation_retries=3)


if __name__ == "__main__":
    agent = StructuredOutputAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
