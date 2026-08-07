#!/usr/bin/env python3
"""Export independent Python expectations for the v8.4 browser rule."""

from __future__ import annotations

import json

from validate_context_consensus import (
    OUTPUT_DIR, fit_pipeline, load_metrics, predict_pipeline, raw_formula,
)


def export_cohort(cohort, regime):
    frame = load_metrics(cohort)
    y = frame["y"].to_numpy(int)
    model = fit_pipeline(frame, y, regime, "night_dynamic")
    intercept, coefficients = raw_formula(model["logistic"])
    probability = predict_pipeline(model, frame)
    return {
        "parameters": {
            "reference": {
                key: {"median": float(model["reference"][key][0]), "scale": float(model["reference"][key][1])}
                for key in ("lyapunov", "det", "entr")
            },
            "intercept": intercept,
            "nightMeanCoef": float(coefficients[0]),
            "dynamicCoef": float(coefficients[1]),
        },
        "probability_by_id": {
            str(subject_id): float(value)
            for subject_id, value in zip(frame["id"], probability)
        },
    }


def main():
    payload = {
        "hall": export_cohort("hall", "untreated"),
        "colas": export_cohort("colas", "treated"),
    }
    path = OUTPUT_DIR / "v84_expected.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
