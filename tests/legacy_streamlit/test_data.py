import io
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from legacy_streamlit.data import MAX_EXPANDED_BYTES, InputDataError, process_data, read_upload
from legacy_streamlit.plots_data import observation_frame


def test_decimal_comma_and_semicolon_utf8_bom():
    t, y, reps = read_upload("\ufefftempo;resposta\n0,0;1,5\n1,5;2,5\n".encode(), "Data.CSV")
    np.testing.assert_array_equal(t, [0.0, 1.5])
    np.testing.assert_array_equal(y, [1.5, 2.5])
    assert len(reps) == 1


@pytest.mark.parametrize("separator", [",", ";", "\t"])
@pytest.mark.parametrize("decimal", [".", ","])
def test_delimiter_and_decimal_are_both_automatic(separator, decimal):
    import csv

    content = io.StringIO()
    writer = csv.writer(content, delimiter=separator)
    writer.writerow(["time", "response"])
    writer.writerows(
        [[v.replace(".", decimal) for v in row] for row in [["0.0", "1.5e-3"], ["1.5", "2.5"]]]
    )
    t, y, _, detected = read_upload(content.getvalue().encode(), "data.csv", return_format=True)
    np.testing.assert_array_equal(t, [0, 1.5])
    np.testing.assert_array_equal(y, [0.0015, 2.5])
    assert detected == {"delimiter": separator, "decimal": decimal}


def test_auto_separator_handles_quoted_headers_blank_lines_and_paired_missing_values():
    content = b'"time, h";"response; mL";t2;y2\n0;1,5;0;1,6\n\n1;2,5;;\n2;3,5;2;3,6\n'
    t, y, reps = read_upload(content, "data.csv")
    assert len(reps) == 2
    np.testing.assert_array_equal(t, [0, 0, 1, 2, 2])
    np.testing.assert_allclose(y, [1.5, 1.6, 2.5, 3.5, 3.6])


def test_mixed_decimal_conventions_require_an_explicit_choice():
    content = b"t;y\n0;1,5\n1;2.5\n"
    with pytest.raises(InputDataError, match="mixed_decimal"):
        read_upload(content, "data.csv")
    # Explicit comma interpretation also accepts literal dot-decimal numbers.
    _, y, _ = read_upload(content, "data.csv", delimiter=";", decimal=",")
    np.testing.assert_array_equal(y, [1.5, 2.5])
    with pytest.raises(InputDataError, match="numeric"):
        read_upload(content, "data.csv", delimiter=";", decimal=".")


def test_ambiguous_column_separator_is_not_silently_selected():
    with pytest.raises(InputDataError, match="delimiter"):
        read_upload(b"time,response;other\n0,1;2\n1,2;3", "data.csv")


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"t,y,z\n0,1,2\n1,2,3", "columns"),
        (b"t,y\n0,1\n1,", "partial_pair"),
        (b"t,y\n0,1\n1,oops", "numeric"),
        (b"t,y\n0,1\n1,inf", "numeric"),
        (b"t,y\n0,1\n0,2", "time"),
        (b"t,y\n-1,1\n0,2", "time"),
        (b"t,y\n0,1\n1,1", "constant"),
        (b"0,1\n1,2\n2,3", "headers"),
        (b"t,y\n,,\n1,2", "columns"),
        (b"", "file_size"),
    ],
)
def test_invalid_data_rejected_with_reason(payload, code):
    with pytest.raises(InputDataError) as exc:
        read_upload(payload, "data.csv", delimiter=",")
    assert exc.value.code == code


def test_empty_replicate_and_duplicate_times_keep_observation_identity():
    df = pd.DataFrame(
        {
            "t1": [1, 0, 0],
            "y1": [4, 1, 2],
            "t2": [0, 1, None],
            "y2": [3, 5, None],
            "t3": [None] * 3,
            "y3": [None] * 3,
        }
    )
    t, y, reps = process_data(df)
    raw = observation_frame(reps)
    assert len(reps) == 2
    np.testing.assert_array_equal(raw.t, t)
    np.testing.assert_array_equal(raw.y, y)
    np.testing.assert_array_equal(y, [1, 2, 3, 4, 5])


def xlsx(rows):
    book = Workbook()
    ws = book.active
    for row in rows:
        ws.append(row)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def test_xlsx_roundtrip_and_no_formula_evaluation():
    _t, y, _ = read_upload(xlsx([["t", "y"], [0, 1], [1, 2]]), "data.xlsx")
    np.testing.assert_array_equal(y, [1, 2])
    with pytest.raises(InputDataError, match="formula"):
        read_upload(xlsx([["t", "y"], [0, 1], [1, "=1+1"]]), "data.xlsx")


def test_xlsx_numeric_cells_and_text_decimal_comma_are_automatic():
    t, y, _ = read_upload(xlsx([["t", "y"], [0.0, "1,5"], [1.25, 2.5]]), "data.xlsx")
    np.testing.assert_array_equal(t, [0, 1.25])
    np.testing.assert_array_equal(y, [1.5, 2.5])


def test_zip_bomb_rejected_before_workbook_parser():
    out = io.BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/fake.xml", b"0" * (MAX_EXPANDED_BYTES + 1))
    with pytest.raises(InputDataError, match="file_size"):
        read_upload(out.getvalue(), "data.xlsx")


def test_no_thousands_inference_or_silent_coercion():
    with pytest.raises(InputDataError, match="numeric"):
        read_upload(b"t;y\n0;1.000,1\n1;2.000,2", "data.csv", decimal=",")
