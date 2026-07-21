"""Inspection-only Phase 1 dataset audit tooling.

This module inspects CSV metadata and quality signals without modifying source data.
It does not preprocess, split, train, or select a target automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _to_builtin(value: Any) -> Any:
    """Convert pandas and numpy scalar values to JSON-serializable Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def build_audit(csv_path: Path, target_column: str | None = None) -> dict[str, Any]:
    if not csv_path.exists():
        return _error(f"CSV file was not found: {csv_path}")

    if not csv_path.is_file():
        return _error(f"Provided path is not a file: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        return _error(f"Expected a .csv file, got: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)
    except PermissionError:
        return _error(
            f"CSV file is not readable due to permission restrictions: {csv_path}"
        )
    except Exception as exc:
        return _error(f"Failed to read CSV file '{csv_path}': {exc}")

    row_count = int(df.shape[0])
    column_count = int(df.shape[1])
    column_names = [str(c) for c in df.columns.tolist()]

    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.to_dict().items()}
    numeric_columns = [str(c) for c in df.select_dtypes(include=["number"]).columns]
    non_numeric_columns = [c for c in column_names if c not in numeric_columns]

    missing_by_column = {
        str(col): int(count) for col, count in df.isna().sum().to_dict().items()
    }
    total_missing_values = int(sum(missing_by_column.values()))
    duplicate_row_count = int(df.duplicated().sum())

    unique_values_by_column = {
        str(col): int(count)
        for col, count in df.nunique(dropna=False).to_dict().items()
    }

    candidate_low_cardinality_target_columns = [
        col
        for col, unique_count in unique_values_by_column.items()
        if 1 < unique_count <= 20
    ]

    target_report: dict[str, Any] = {
        "target_requested": bool(target_column),
        "target_column": target_column,
    }

    if target_column:
        if target_column not in df.columns:
            target_report.update(
                {
                    "target_exists": False,
                    "message": "Supplied target column is not present in this dataset.",
                }
            )
        else:
            target_series = df[target_column]
            class_counts = {
                str(idx): int(val)
                for idx, val in target_series.value_counts(dropna=False).to_dict().items()
            }
            class_proportions = {
                str(idx): float(val)
                for idx, val in target_series.value_counts(
                    dropna=False, normalize=True
                ).to_dict().items()
            }
            target_report.update(
                {
                    "target_exists": True,
                    "target_dtype": str(target_series.dtype),
                    "target_unique_values": [
                        _to_builtin(v)
                        for v in target_series.drop_duplicates().tolist()
                    ],
                    "target_class_counts": class_counts,
                    "target_class_proportions": class_proportions,
                }
            )

    inferred_feature_count = column_count - 1
    if target_column and target_column in column_names:
        inferred_feature_count = column_count - 1

    constraints = {
        "minimum_rows_1000": {
            "required": 1000,
            "actual": row_count,
            "passed": row_count >= 1000,
        },
        "minimum_feature_columns_8_excluding_target": {
            "required": 8,
            "actual": inferred_feature_count,
            "passed": inferred_feature_count >= 8,
            "note": (
                "Feature count assumes exactly one target column. "
                "Confirm after target column is finalized."
            ),
        },
        "has_numeric_columns": {
            "required": True,
            "actual": len(numeric_columns) > 0,
            "passed": len(numeric_columns) > 0,
        },
        "has_categorical_or_non_numeric_columns": {
            "required": True,
            "actual": len(non_numeric_columns) > 0,
            "passed": len(non_numeric_columns) > 0,
        },
        "missing_values_present": {
            "required": "Missing values preferred or simulation required",
            "actual_total_missing_values": total_missing_values,
            "passed": total_missing_values > 0,
            "note": (
                "No missing values detected; missingness simulation may be required "
                "for assignment validation."
                if total_missing_values == 0
                else "Missing values detected."
            ),
        },
    }

    return {
        "ok": True,
        "file_path": str(csv_path),
        "row_count": row_count,
        "column_count": column_count,
        "column_names": column_names,
        "dtypes": dtypes,
        "numeric_columns": numeric_columns,
        "categorical_or_non_numeric_columns": non_numeric_columns,
        "missing_values_by_column": missing_by_column,
        "total_missing_values": total_missing_values,
        "duplicate_row_count": duplicate_row_count,
        "unique_values_by_column": unique_values_by_column,
        "candidate_low_cardinality_target_columns": candidate_low_cardinality_target_columns,
        "target_report": target_report,
        "assignment_constraints": constraints,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a CSV dataset and produce a Phase 1 audit summary."
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the CSV file to inspect.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional target column name to inspect.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save the audit output to JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/dataset_audit.json",
        help="JSON output path used when --save-json is provided.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    audit = build_audit(csv_path=csv_path, target_column=args.target_column)
    print(json.dumps(audit, indent=2, ensure_ascii=True))

    if not audit.get("ok", False):
        return 1

    if args.save_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Saved audit JSON to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
