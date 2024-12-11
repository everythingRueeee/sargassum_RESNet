from sklearn.metrics import f1_score, jaccard_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score


def evaluate_metrics(y_true, y_pred):
    unique_true = set(y_true.flatten())
    unique_pred = set(y_pred.flatten())

    if len(unique_true) == 1 or len(unique_pred) == 1:
        raise ValueError(
            "Both y_true and y_pred should contain both binary classes (0 and 1) for valid metric computation.")

    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()

    f1 = f1_score(y_true.flatten(), y_pred.flatten())
    precision = precision_score(y_true.flatten(), y_pred.flatten())
    recall = recall_score(y_true.flatten(), y_pred.flatten())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
    iou = jaccard_score(y_true.flatten(), y_pred.flatten())
    dice = 2 * tp / (2 * tp + fp + fn)

    return {
        "F1-score": f1,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "FNR": fnr,
        "MCC": mcc,
        "IoU": iou,
        "Dice": dice,
        "Confusion Matrix": (tn, fp, fn, tp)
    }
