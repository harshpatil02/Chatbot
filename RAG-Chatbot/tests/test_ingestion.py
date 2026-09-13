from pathlib import Path

from openpyxl import Workbook

from app.ingestion import extract_text_from_file


def test_extract_text_from_xlsx(tmp_path: Path) -> None:
    workbook_path = tmp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["Name", "Role"])
    sheet.append(["Alice", "Engineer"])
    sheet.append(["Bob", "Analyst"])
    workbook.save(workbook_path)

    text = extract_text_from_file(str(workbook_path))

    assert "Alice" in text
    assert "Engineer" in text
    assert "Bob" in text
