"""Stable observation identity shared by figures and exports."""

import pandas as pd


def observation_frame(replicates):
    rows = [
        {"replicate": rep["name"], "observation": j + 1, "t": float(t), "y": float(y)}
        for rep in replicates
        for j, (t, y) in enumerate(zip(rep["t"], rep["y"], strict=False))
    ]
    return (
        pd.DataFrame(rows, columns=["replicate", "observation", "t", "y"])
        .sort_values("t", kind="stable")
        .reset_index(drop=True)
    )
