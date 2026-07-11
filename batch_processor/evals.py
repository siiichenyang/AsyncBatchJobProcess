from dataclasses import dataclass


@dataclass
class Summary:
    total: int = 0
    success: int = 0
    error: int = 0
    evaluated: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
