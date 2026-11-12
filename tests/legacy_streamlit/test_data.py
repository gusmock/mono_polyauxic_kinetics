import io
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from legacy_streamlit.data import MAX_EXPANDED_BYTES, InputDataError, process_data, read_upload
from legacy_streamlit.plots_data import observation_frame


def test_decimal_comma_and_semicolon_utf8_bom():
    t, y, reps = read_upload(
        "\ufefftempo;resposta\n0,0;1,5\n1,5;2,5\n".encode(), "Data.CSV", decimal=","
    )
    np.testing.assert_array_equal(t, [0.0, 1.5])
    np.testing.assert_array_equal(y, [1.5, 2.5])
    assert len(reps) == 1


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


def test_zip_bomb_rejected_before_workbook_parser():
    out = io.BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/fake.xml", b"0" * (MAX_EXPANDED_BYTES + 1))
    with pytest.raises(InputDataError, match="file_size"):
        read_upload(out.getvalue(), "data.xlsx")


def test_no_thousands_inference_or_silent_coercion():
    with pytest.raises(InputDataError, match="numeric"):
        read_upload(b"t;y\n0;1.000,1\n1;2.000,2", "data.csv", decimal=",")
