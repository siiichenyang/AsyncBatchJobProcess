from batch_processor.eval_run_store import EvalRunStore


def test_save_run_and_get_summary(eval_run_db_path):
    store = EvalRunStore(eval_run_db_path)

    store.save_run(
        "run-1",
        results=[{"name": "case", "passed": True}],
        summary={"total": 1, "passed": 1, "pass_rate": 1.0},
    )

    assert store.get_summary("run-1") == {
        "total": 1,
        "passed": 1,
        "pass_rate": 1.0,
    }


def test_summary_survives_new_store_instance(eval_run_db_path):
    first_store = EvalRunStore(eval_run_db_path)
    first_store.save_run(
        "run-1",
        results=[],
        summary={"total": 0, "passed": 0, "pass_rate": 0.0},
    )

    # A new store instance represents a process restart / another worker using
    # the same SQLite database.
    second_store = EvalRunStore(eval_run_db_path)

    assert second_store.get_summary("run-1") == {
        "total": 0,
        "passed": 0,
        "pass_rate": 0.0,
    }


def test_get_summary_returns_none_for_unknown_run(eval_run_db_path):
    store = EvalRunStore(eval_run_db_path)

    assert store.get_summary("missing") is None
