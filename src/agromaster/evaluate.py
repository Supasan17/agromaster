"""Classification metrics for the risk label.

Follows the proposal's "Evaluation metrics" subsection. Classifiers are
assessed with accuracy, per-class and macro-averaged precision, recall and F1
score, and the confusion matrix. Because the Low class dominates, macro F1
and recall of the rare High risk class are the main comparison measures.

Every function takes the true and predicted labels as pandas Series (for
example the label column of a frame and the output of a ``predict_*``
function in :mod:`agromaster.models`) and aligns them on the true labels.
Rows whose true label is missing, such as rows without a seven-day target,
are left out. A missing prediction for a labelled row is an error, so a model
cannot look better by skipping hard rows. All risk labels in
``config.RISK_LABELS`` are always reported, in that order, even when a class
is absent from both series.

The grouped reports apply the same headline metrics within each commodity
and each period, because the proposal asks for results per commodity and per
period and not only as a single overall figure.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from agromaster import config

METRIC_COLUMNS = ["Precision", "Recall", "F1", "Support"]
SUMMARY_COLUMNS = [
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "High_Recall",
    "N",
]


def _aligned_labels(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Return the labelled rows of ``y_true`` and their predictions as strings.

    Raises ``ValueError`` when the two series have different indexes, when a
    labelled row has no prediction, or when a label is not one of
    ``config.RISK_LABELS``.
    """
    if not y_true.index.equals(y_pred.index):
        raise ValueError("True and predicted labels must share the same index")
    labelled = y_true.notna()
    true = y_true[labelled].astype("object")
    pred = y_pred[labelled].astype("object")
    if pred.isna().any():
        raise ValueError(
            f"{int(pred.isna().sum())} labelled rows have no prediction"
        )
    unknown = (set(true) | set(pred)) - set(config.RISK_LABELS)
    if unknown:
        raise ValueError(f"Unknown risk labels: {sorted(unknown)}")
    return true, pred


def confusion_matrix_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return the confusion matrix with true labels as rows.

    Rows and columns follow ``config.RISK_LABELS``; each cell counts the
    labelled rows with that true label (row) and predicted label (column).
    """
    true, pred = _aligned_labels(y_true, y_pred)
    counts = confusion_matrix(true, pred, labels=config.RISK_LABELS)
    return pd.DataFrame(
        counts,
        index=pd.Index(config.RISK_LABELS, name="True"),
        columns=pd.Index(config.RISK_LABELS, name="Predicted"),
    )


def per_class_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return precision, recall, F1 and support per risk class and macro-averaged.

    One row per label in ``config.RISK_LABELS`` followed by a
    ``config.MACRO_ROW`` row holding the unweighted mean of the per-class
    precision, recall and F1 and the total support.
    """
    true, pred = _aligned_labels(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=config.RISK_LABELS,
        average=None,
        zero_division=config.METRIC_ZERO_DIVISION,
    )
    table = pd.DataFrame(
        {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        },
        index=pd.Index(config.RISK_LABELS, name="Class"),
    )
    macro = table[["Precision", "Recall", "F1"]].mean()
    table.loc[config.MACRO_ROW] = [*macro, table["Support"].sum()]
    table["Support"] = table["Support"].astype("int64")
    return table[METRIC_COLUMNS]


def summary_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Return the headline metrics for one model as a Series.

    Accuracy, macro precision, macro recall, macro F1, recall of
    ``config.HIGH_RISK_LABEL`` and the number of labelled rows evaluated.
    """
    true, pred = _aligned_labels(y_true, y_pred)
    table = per_class_metrics(y_true, y_pred)
    macro = table.loc[config.MACRO_ROW]
    return pd.Series(
        {
            "Accuracy": accuracy_score(true, pred) if len(true) else float("nan"),
            "Macro_Precision": macro["Precision"],
            "Macro_Recall": macro["Recall"],
            "Macro_F1": macro["F1"],
            "High_Recall": table.loc[config.HIGH_RISK_LABEL, "Recall"],
            "N": len(true),
        },
        name="Metrics",
    )


def grouped_summary(
    y_true: pd.Series, y_pred: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Return :func:`summary_metrics` for each group, one row per group.

    ``groups`` gives the group of every row and must share the index of
    ``y_true``. Rows with a missing true label are left out before grouping,
    and groups without any labelled rows are dropped. Rows are sorted by
    group; the index is named after ``groups``.
    """
    if not y_true.index.equals(groups.index):
        raise ValueError("Groups must share the index of the true labels")
    true, pred = _aligned_labels(y_true, y_pred)
    keys = groups[true.index]
    if keys.isna().any():
        raise ValueError(f"{int(keys.isna().sum())} labelled rows have no group")
    rows = {
        key: summary_metrics(true[keys == key], pred[keys == key])
        for key in sorted(keys.unique())
    }
    table = pd.DataFrame.from_dict(rows, orient="index", columns=SUMMARY_COLUMNS)
    table.index.name = groups.name
    table["N"] = table["N"].astype("int64")
    return table


def per_commodity_report(
    frame: pd.DataFrame,
    y_pred: pd.Series,
    label_col: str = config.LABEL_COLUMN,
    commodity_col: str = config.COMMODITY_COLUMN,
) -> pd.DataFrame:
    """Return the headline metrics for each commodity in ``frame``.

    ``frame`` holds the true labels in ``label_col`` and the commodity of each
    row; ``y_pred`` holds the predictions on the same index.
    """
    groups = frame[commodity_col].astype("object").rename(commodity_col)
    return grouped_summary(frame[label_col], y_pred, groups)


def assign_period(
    dates: pd.Series, freq: str = config.PERIOD_FREQ
) -> pd.Series:
    """Return the period (as a string such as ``"2024"``) of each date."""
    period = pd.to_datetime(dates).dt.to_period(freq).astype("string")
    return period.astype("object").rename(config.PERIOD_COLUMN)


def per_period_report(
    frame: pd.DataFrame,
    y_pred: pd.Series,
    label_col: str = config.LABEL_COLUMN,
    date_col: str = config.DATE_COLUMN,
    freq: str = config.PERIOD_FREQ,
) -> pd.DataFrame:
    """Return the headline metrics for each period of the prediction date.

    Periods follow the pandas alias ``freq`` (calendar years by default) and
    are sorted chronologically.
    """
    groups = assign_period(frame[date_col], freq)
    return grouped_summary(frame[label_col], y_pred, groups)
