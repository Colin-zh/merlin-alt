"""
This file is directly copied from torchkeras licensed under Apache 2.0.
Original source: https://github.com/lyhue1991/torchkeras/blob/master/torchkeras/utils.py
Original copyright: Copyright (c) lyhue1991

See full Apache 2.0 license in the project root LICENSE-APACHE file.
"""

import sys

import torch
from torch import nn


class Accuracy(nn.Module):
    """Accuracy metric for binary classification tasks."""

    def __init__(self):
        super(Accuracy, self).__init__()
        # Counters for correct predictions and total samples
        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass to compute accuracy for binary classification.

        Args:
            preds (torch.Tensor): Predictions from the model (logits).
            labels (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Accuracy for the current batch.
        
        """
        assert preds.shape == labels.shape, "Predictions and labels must have the same shape."
        # Count correct predictions and total samples
        correct_batch = torch.sum((torch.sigmoid(preds) >= 0.5).float() == (labels > 0.5).float())
        total_batch = labels.numel()

        self.correct += correct_batch
        self.total += total_batch
        return correct_batch.float() / total_batch

    def compute(self):
        """
        Compute the overall accuracy.

        Returns:
            torch.Tensor: Overall accuracy.
        """
        if self.total.item() == 0:
            return torch.tensor(0.0)
        return self.correct / self.total

    def reset(self):
        """Reset the counters for correct predictions and total samples."""
        self.correct.data.zero_()
        self.total.data.zero_()


class Precision(nn.Module):
    """Precision metric for binary classification tasks."""

    def __init__(self):
        super(Precision, self).__init__()
        # Counters for true positives and false positives
        self.true_positives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.false_positives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    
    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass to compute precision for binary classification.

        Args:
            preds (torch.Tensor): Predictions from the model (logits).
            labels (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Precision for the current batch.
        
        """
        y_pred = torch.sigmoid(preds).reshape(-1)
        y_true = labels.reshape(-1)
        assert y_pred.shape == y_true.shape
        # Count true positive and false positive predictions
        tpi = torch.sum((y_pred >= 0.5) * (y_true >= 0.5))
        fpi = torch.sum((y_pred >= 0.5) * (y_true < 0.5))
        self.true_positive += tpi
        self.false_positive += fpi
        return torch.true_divide(tpi, tpi + fpi)

    def compute(self):
        """
        Compute precision.

        Returns:
            torch.Tensor: Precision.
        """
        # Calculate and return precision
        return torch.true_divide(self.true_positive, (self.true_positive + self.false_positive))

    def reset(self):
        """Reset counters for true positive and false positive to zero."""
        # Reset counters to zero for the next evaluation
        self.true_positive.data.zero_()
        self.false_positive.data.zero_()


class Recall(nn.Module):
    """Recall metric for binary classification tasks."""

    def __init__(self):
        super(Recall, self).__init__()
        # Counters for true positives and false negatives
        self.true_positives = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total_positives = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass to compute recall for binary classification.

        Args:
            preds (torch.Tensor): Predictions from the model (logits).
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Recall for the current batch.
        """
        y_pred = torch.sigmoid(preds).reshape(-1)
        y_true = labels.reshape(-1)
        assert y_pred.shape == y_true.shape

        true_positive_i = torch.sum((y_pred >= 0.5) * (y_true >= 0.5))
        total_positive_i = torch.sum(y_true >= 0.5)
        self.true_positive += true_positive_i
        self.total_positive += total_positive_i
        return torch.true_divide(true_positive_i, total_positive_i)

    def compute(self):
        """
        Compute recall.

        Returns:
            torch.Tensor: Recall.
        """
        # Calculate and return recall
        return torch.true_divide(self.true_positive, self.total_positive)

    def reset(self):
        """Reset counters for true positive and total positive to zero."""
        # Reset counters to zero for the next evaluation
        self.true_positive.data.zero_()
        self.total_positive.data.zero_()


class F1Score(nn.Module):
    """F1 Score metric for binary classification tasks."""

    def __init__(self):
        super(F1Score, self).__init__()
        self.precision_metric = Precision()
        self.recall_metric = Recall()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        pass


class AUC(nn.Module):
    """AUC metric for binary classification tasks."""

    def __init__(self):
        super(AUC, self).__init__()
        self.tp = nn.Parameter(torch.zeros(10001), requires_grad=False)
        self.fp = nn.Parameter(torch.zeros(10001), requires_grad=False)

    def eval_auc(self, tp, fp):
        tp_total = torch.sum(tp)
        fp_total = torch.sum(fp)
        length = len(tp)
        tp_reverse = tp[range(length-1, -1, -1)]
        tp_reverse_cum = torch.cumsum(tp_reverse, dim=0) - tp_reverse / 2.0
        fp_reverse = fp[range(length-1, -1, -1)]

        auc = torch.sum(torch.true_divide(tp_reverse_cum, tp_total)
                        * torch.true_divide(fp_reverse, fp_total))
        return auc
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        y_pred = (10000 * torch.sigmoid(preds)).reshape(-1).type(torch.int)
        y_true = targets.reshape(-1)

        tpi = self.tp - self.tp
        fpi = self.fp - self.fp
        assert y_pred.shape == y_true.shape
        for i, label in enumerate(y_true):
            if label >= 0.5:
                tpi[y_pred[i]] += 1.0
            else:
                fpi[y_pred[i]] += 1.0
        self.tp += tpi
        self.fp += fpi
        return self.eval_auc(tpi, fpi)

    def compute(self):
        return self.eval_auc(self.tp, self.fp)

    def reset(self):
        self.tp -= self.tp
        self.fp -= self.fp


class KS(nn.Module):
    """KS metric for binary classification tasks."""

    def __init__(self):
        super(KS, self).__init__()
        # Counters for true positive (tp) and false positive (fp) at various thresholds
        self.tp = nn.Parameter(torch.zeros(10001), requires_grad=False)
        self.fp = nn.Parameter(torch.zeros(10001), requires_grad=False)

    def eval_ks(self, tp, fp):
        """Evaluate KS given true positive (tp) and false positive (fp) counts."""
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        tp_curve = tp_cum / tp_cum[-1]
        fp_curve = fp_cum / fp_cum[-1]
        ks_value = torch.max(torch.abs(tp_curve - fp_curve))
        return ks_value

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Forward pass to compute KS for binary classification.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: KS for the current batch.
        """
        y_pred = (10000 * torch.sigmoid(preds)).reshape(-1).type(torch.int)
        y_true = targets.reshape(-1)

        tpi = self.tp - self.tp
        fpi = self.fp - self.fp
        assert y_pred.shape == y_true.shape
        for i, label in enumerate(y_true):
            if label >= 0.5:
                tpi[y_pred[i]] += 1.0
            else:
                fpi[y_pred[i]] += 1.0

        self.tp += tpi
        self.fp += fpi

        return self.eval_ks(tpi, fpi)

    def compute(self):
        """
        Compute KS.

        Returns:
            torch.Tensor: KS.
        """
        # Calculate and return KS
        return self.eval_ks(self.tp, self.fp)

    def reset(self):
        """Reset counters for true positive (tp) and false positive (fp) to zero."""
        # Reset counters to zero for the next evaluation
        self.tp.data.zero_()
        self.fp.data.zero_()


class IOU(nn.Module):
    """
    IOU calculation for segmentation task (both binary and multiclass).
    """

    def __init__(self, num_classes, if_print=False):
        """
        Initialize the IOU module.

        Args:
            num_classes (int): Number of classes.
            if_print (bool, optional): Whether to print IOU information. Defaults to False.
        """
        super().__init__()
        self.num_classes = num_classes
        n = num_classes if num_classes >= 2 else 2
        self.mat = nn.Parameter(torch.zeros((n, n), dtype=torch.int64), requires_grad=False)
        self.if_print = if_print

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass to compute IOU for segmentation.

        Args:
            preds (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Mean IOU for the current batch.
        """
        n = self.num_classes if self.num_classes >= 2 else 2
        with torch.no_grad():
            if self.num_classes >= 2:
                a, b = labels.flatten(), preds.argmax(1).flatten()
            else:
                a, b = labels.flatten(), (preds > 0).long().flatten()
            assert a.shape == b.shape
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            mati = torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            self.mat += mati
            acc_global, iou = self.eval_iou(mati)
            return iou.mean()

    def compute(self):
        """
        Compute IOU.

        Returns:
            torch.Tensor: Mean IOU.
        """
        acc_global, iou = self.eval_iou(self.mat)
        if self.if_print:
            print(self, file=sys.stderr)
        return iou.mean()

    def reset(self):
        """Reset the confusion matrix to zero."""
        self.mat.zero_()

    def eval_iou(self, mat):
        """
        Evaluate IOU given a confusion matrix.

        Args:
            mat (torch.Tensor): Confusion matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Global accuracy and IOU for each class.
        """
        h = mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, iou

    def __str__(self):
        """
        String representation of the IOU module.

        Returns:
            str: Formatted IOU information.
        """
        acc_global, iou = self.eval_iou(self.mat)
        return (
            'global correct: {:.4f}\n'
            'IoU: {}\n'
            'mean IoU: {:.4f}'
        ).format(
            acc_global.item(),
            ['{:.4f}'.format(i) for i in iou.tolist()],
            iou.mean().item()
        )
