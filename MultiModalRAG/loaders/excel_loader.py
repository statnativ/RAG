# loaders/excel_loader.py

from pathlib import Path
import pandas as pd


def load_excel_files(directory):
    excel_texts = {}
    for excel_path in Path(directory).glob("*.xls*"):  # both xls and xlsx
        try:
            text = ""
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                text += df.to_string(index=False, header=True) + "\n\n"
            excel_texts[excel_path.name] = text.strip()
        except Exception as e:
            print(f"Failed to load Excel file {excel_path}: {e}")
    return excel_texts
