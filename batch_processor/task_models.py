from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class TaskCase:
    name: str
    prompt: str
    expected: str | None = None

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not isinstance(data, dict):
            raise ValueError("task must be a JSON object")

        for field_name in ("name", "prompt"):
            if field_name not in data:
                raise ValueError(
                    f"missing required field: {field_name!r}"
                )
            value = data[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"field content invalid {field_name!r}: {value!r}"
                )

        expected = data.get("expected")
        if expected is not None and not isinstance(expected, str):
            raise ValueError(
                f"field content invalid 'expected': {expected!r}"
            )

        return cls(
            name=data["name"],
            prompt=data["prompt"],
            expected=expected,
        )
