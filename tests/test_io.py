from pathlib import Path

from afri_reasoning.io import read_json, read_jsonl


def test_json_readers_accept_utf8_bom(tmp_path: Path) -> None:
    object_path = tmp_path / "value.json"
    lines_path = tmp_path / "values.jsonl"
    object_path.write_text('{"ok": true}\n', encoding="utf-8-sig")
    lines_path.write_text('{"sample_id": "one"}\n', encoding="utf-8-sig")

    assert read_json(object_path) == {"ok": True}
    assert read_jsonl(lines_path) == [{"sample_id": "one"}]
