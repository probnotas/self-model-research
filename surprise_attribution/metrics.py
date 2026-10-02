"""Evaluation metrics: confusion matrix, error rates, ROC, boundaries.

Conventions (stated once, used everywhere):
  positive class = "body change"   (the classifier says "adapt your self-model")
  negative class = "world event"   (the classifier says "leave it alone")

  false positive  = a world event called "body change"
                    -> the robot rewrites its self-model to fit a push that is
                       already over. THIS IS THE SAFETY-CRITICAL ERROR: it
                       corrupts a correct model, and every later plan uses it.
  false negative  = a body change called "world event"
                    -> the robot keeps a stale model. Bad, but recoverable:
                       the error stays high, so the next check can catch it.

Every rate is reported with a 95% Wilson interval, because with 100 trials a
rate of 0/100 does not mean "never". It means "probably below ~4%".
"""
import numpy as np

from classifier import BODY, WORLD


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n.

    Chosen over the simple p +/- 1.96*sqrt(p(1-p)/n) because that one collapses
    to [0, 0] when k = 0, which would falsely claim certainty.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(true, pred):
    """2x2 counts: rows = true class, columns = predicted class."""
    true, pred = np.asarray(true), np.asarray(pred)
    return {
        "body->body":   int(np.sum((true == BODY) & (pred == BODY))),    # true positive
        "body->world":  int(np.sum((true == BODY) & (pred == WORLD))),   # false negative
        "world->body":  int(np.sum((true == WORLD) & (pred == BODY))),   # false positive
        "world->world": int(np.sum((true == WORLD) & (pred == WORLD))),  # true negative
    }


def rates(cm):
    """Accuracy, FPR, FNR, TPR from confusion counts, each with a Wilson CI."""
    tp, fn = cm["body->body"], cm["body->world"]
    fp, tn = cm["world->body"], cm["world->world"]
    n_body, n_world = tp + fn, fp + tn
    n = n_body + n_world
    return {
        "accuracy": ((tp + tn) / n, wilson(tp + tn, n)),
        "false_positive_rate": (fp / n_world, wilson(fp, n_world)),
        "false_negative_rate": (fn / n_body, wilson(fn, n_body)),
        "true_positive_rate": (tp / n_body, wilson(tp, n_body)),
    }


def roc(scores, true):
    """False-positive rate and true-positive rate at every useful threshold.

    The thresholds are the observed scores themselves (plus one above the
    maximum, giving the (0, 0) corner). Between two adjacent observed scores
    nothing changes, so these are all the distinct operating points there are.
    Rule as in the classifier: predict "body" when score > threshold.
    """
    scores, true = np.asarray(scores, float), np.asarray(true)
    is_body = true == BODY
    thresholds = np.concatenate([[np.inf], np.sort(np.unique(scores))[::-1]])
    rows = []
    for th in thresholds:
        pred_body = scores > th
        tpr = float(np.mean(pred_body[is_body]))
        fpr = float(np.mean(pred_body[~is_body]))
        rows.append((th, fpr, tpr))
    # the (1, 1) corner: threshold below every score
    rows.append((-np.inf, 1.0, 1.0))
    return rows


def auc(roc_rows):
    """Area under the ROC curve by the trapezoid rule.

    1.0 = some threshold separates the classes perfectly; 0.5 = chance.
    Equivalently: the probability that a random body-change trial scores
    higher than a random world-event trial.
    """
    f = np.array([r[1] for r in roc_rows]); t = np.array([r[2] for r in roc_rows])
    order = np.lexsort((t, f))
    f, t = f[order], t[order]
    return float(np.sum(np.diff(f) * (t[1:] + t[:-1]) / 2))


def boundary_table(results, key_fn, correct_label):
    """Per-setting rate of correct verdicts for the boundary grids.

    key_fn(row) -> the setting a trial belongs to (e.g. ("l", 1.1)).
    correct_label -> the right answer for every trial in this grid.
    """
    groups = {}
    for r in results:
        groups.setdefault(key_fn(r), []).append(r["prediction"] == correct_label)
    table = []
    for key in sorted(groups):
        hits = groups[key]
        k, n = int(np.sum(hits)), len(hits)
        table.append((key, k, n, k / n, wilson(k, n)))
    return table
