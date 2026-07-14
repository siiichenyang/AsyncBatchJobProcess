import pytest

from batch_processor.task_models import TaskCase


def test_task_case_from_valid_dict():
    task = TaskCase.from_dict({
        "name": "weather",
        "prompt": "What is the weather",
        "expected": None,
    })

    assert task.name == "weather"
    assert task.prompt == "What is the weather"
    assert task.expected == None


@pytest.mark.parametrize("data", ([], 123, "apple"))
def test_task_case_rejects_non_object(data):
    with pytest.raises(ValueError, match="JSON object"):
        TaskCase.from_dict(data)


def test_task_case_rejects_non_object_expected():
    with pytest.raises(ValueError, match="'expected'"):
        TaskCase.from_dict({
            "name": "weather",
            "prompt": "What is the weather",
            "expected": 123,
        })
