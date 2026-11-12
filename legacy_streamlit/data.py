"""Bounded CSV/XLSX ingestion. No uploaded data are written to disk."""

import csv
import io
from pathlib import Path
from zipfile import ZipFile

import defusedxml  # noqa: F401 - openpyxl detects and uses this safe XML parser
import numpy as np
import pandas as pd
from openpyxl import load_workbook

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EXPANDED_BYTES = 25 * 1024 * 1024
MAX_ROWS = 5000
MAX_REPLICATES = 5
MAX_POINTS = 10000


class InputDataError(ValueError):
    """An invalid input with a translation key and non-sensitive coordinates."""

    def __init__(self, code, location=""):
        self.code, self.location = code, location
        super().__init__(f"{code}: {location}" if location else code)


def read_upload(content, filename, *, delimiter="auto", decimal="."):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise InputDataError("file_size")
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise InputDataError("encoding") from exc
            if delimiter == "auto":
                try:
                    delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t").delimiter
                except csv.Error as exc:
                    raise InputDataError("delimiter") from exc
            if delimiter not in (",", ";", "\t"):
                raise InputDataError("delimiter")
            reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(rows) >= MAX_ROWS + 1:
                    raise InputDataError("data_size")
                if len(row) > 2 * MAX_REPLICATES:
                    raise InputDataError("columns")
                if rows and len(row) != len(rows[0]):
                    raise InputDataError("columns", f"row {reader.line_num}")
                if all(not cell.strip() for cell in row):
                    continue
                rows.append(row)
        elif suffix == ".xlsx":
            with ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if len(members) > 200 or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                    raise InputDataError("file_size")
                if any(
                    m.flag_bits & 1
                    or "externallinks/" in m.filename.lower()
                    or m.filename.lower().endswith("vbaproject.bin")
                    for m in members
                ):
                    raise InputDataError("workbook")
            book = load_workbook(
                io.BytesIO(content), read_only=True, data_only=False, keep_links=False
            )
            try:
                sheet = book.worksheets[0]
                # Never trust XLSX dimension metadata; impose limits while reading.
                sheet.reset_dimensions()
                rows = []
                for index, values in enumerate(
                    sheet.iter_rows(values_only=True, max_col=2 * MAX_REPLICATES + 1), start=1
                ):
                    if index > MAX_ROWS + 1:
                        raise InputDataError("data_size")
                    row = list(values)
                    while row and row[-1] is None:
                        row.pop()
                    if not row:
                        continue
                    if len(row) > 2 * MAX_REPLICATES:
                        raise InputDataError("columns")
                    if any(isinstance(v, str) and v.startswith("=") for v in row):
                        raise InputDataError("formula", f"row {index}")
                    rows.append(row)
            finally:
                book.close()
        else:
            raise InputDataError("file_type")
    except InputDataError:
        raise
    except Exception as exc:
        # Parser exceptions must not disclose paths or imported private data.
        raise InputDataError("file_parse") from exc
    if len(rows) < 2:
        raise InputDataError("empty_data")
    header = rows[0]
    if any(v is None or not str(v).strip() for v in header) or all(_is_number(v) for v in header):
        raise InputDataError("headers")
    if any(len(row) > len(header) for row in rows[1:]):
        raise InputDataError("columns")
    frame = pd.DataFrame(
        [row + [None] * (len(header) - len(row)) for row in rows[1:]], columns=header
    )
    return process_data(frame, decimal=decimal)


def _is_number(value):
    try:
        float(str(value).strip().replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False


def process_data(df, decimal="."):
    """Paired blanks are missing observations; partial, nonnumeric and infinite pairs are errors."""
    df = df.reset_index(drop=True)
    if decimal not in (".", ","):
        raise InputDataError("decimal")
    if not 2 <= df.shape[1] <= 2 * MAX_REPLICATES or df.shape[1] % 2:
        raise InputDataError("columns")
    if len(df) > MAX_ROWS:
        raise InputDataError("data_size")
    replicates, all_t, all_y = [], [], []
    for i in range(df.shape[1] // 2):
        raw = df.iloc[:, 2 * i : 2 * i + 2].copy()
        blank = raw.isna() | raw.map(lambda v: isinstance(v, str) and not v.strip())
        partial = blank.iloc[:, 0] != blank.iloc[:, 1]
        if partial.any():
            raise InputDataError(
                "partial_pair", f"row {np.flatnonzero(partial)[0] + 2}, pair {i + 1}"
            )
        raw = raw.loc[~blank.all(axis=1)]
        if raw.empty:
            continue
        converted = raw.map(
            lambda v: v.strip().replace(",", ".") if isinstance(v, str) and decimal == "," else v
        )
        values = converted.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(values)
        if invalid.any():
            r, c = np.argwhere(invalid)[0]
            raise InputDataError("numeric", f"row {raw.index[r] + 2}, column {2 * i + c + 1}")
        if np.any(values[:, 0] < 0):
            raise InputDataError("time")
        order = np.argsort(values[:, 0], kind="stable")
        times, responses = values[order, 0], values[order, 1]
        replicates.append({"t": times, "y": responses, "name": f"Replica {i + 1}"})
        all_t.extend(times)
        all_y.extend(responses)
    if not all_t:
        raise InputDataError("empty_data")
    if len(all_t) > MAX_POINTS:
        raise InputDataError("data_size")
    t, y = np.array(all_t), np.array(all_y)
    if len(np.unique(t)) < 2:
        raise InputDataError("time")
    if np.ptp(y) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))):
        raise InputDataError("constant")
    order = np.argsort(t, kind="stable")
    return t[order], y[order], replicates
