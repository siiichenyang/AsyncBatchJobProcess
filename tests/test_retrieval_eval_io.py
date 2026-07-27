from batch_processor.retrieval_eval_io import load_retrieval_eval_cases


def test_load_cases(tmp_path):
    input_path = tmp_path / "input_eval.jsonl"

    input_path.write_text(
        '{"name": "fruit-query","query": "Do we have apple or cherry?","relevant_spans": [{"document_id": "basket", "start_word": 0, "end_word": 1}]}\n'
        '{"name": "fruit-query","query": "Do we have apple or cherry?","relevant_spans": []}\n'
        'Invalid Json\n',
        encoding="utf-8",
    )

    records = load_retrieval_eval_cases(input_path)

    assert len(records) == 3

    assert records[0].line_number == 1
    assert records[0].case is not None and records[0].error is None

    assert records[1].line_number == 2
    assert records[1].case is None and records[1].error is not None
    assert "relevant_spans" in records[1].error

    assert records[2].line_number == 3
    assert records[2].case is None and records[2].error is not None
    assert "Invalid JSON line 3" in records[2].error
