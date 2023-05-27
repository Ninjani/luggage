from pytorch_lightning.callbacks import Callback
import torch
import pandas as pnd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics


class LogParameters(Callback):
    """
    Logs histograms of model parameters and gradients to check for vanishing/exploding gradients
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(f"Model/{name}", param, global_step=trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Model/{name}_grad", param.grad, global_step=trainer.global_step)

def get_metrics_and_curves(metric_type, y_pred, y_true, invert=False, threshold=0.5):
    """
    Calculate metrics and curves for a given metric type
    ROC: Receiver Operating Characteristic curve, metric = Area under the curve
    PR: Precision-Recall curve, metric = Area under the curve (Average precision)
    CM: Confusion Matrix, metric = F1 score

    Parameters
    ----------
    metric_type : str
        One of "ROC", "PR", "CM"
    y_pred : torch.Tensor
        Predicted labels
    y_true : torch.Tensor
        True labels
    invert : bool
        If True, do 1 - y_pred, use if y_pred is distance instead of probability

    Returns
    -------
    metric_value : float
        Value of the metric
    metric_disp : matplotlib.figure.Figure
        Figure of the curve/matrix
    """
    if invert:
        y_pred = 1 - y_pred
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    if metric_type == "ROC": 
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        roc_disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        return roc_auc, roc_disp.figure_
    elif metric_type == "PR":
        # Precision-Recall Curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
        pr_auc = metrics.auc(recall, precision)
        pr_disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot()
        return pr_auc, pr_disp.figure_
    elif metric_type == "CM":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred > threshold)
        df_cm = pnd.DataFrame(confusion_matrix)
        plt.figure(figsize = (10,7))
        cm_disp = sns.heatmap(df_cm, annot=True, cmap='Blues').get_figure()
        plt.close(cm_disp)
        f1 = metrics.f1_score(y_true, y_pred > threshold)
        return f1, cm_disp



class LogMetrics(Callback):
    """
    Log metrics and curves for validation and training

    Scalars: ROC/val_AUC, ROC/train_AUC, PR/val_AUC, PR/train_AUC, CM/val_F1, CM/train_F1 
    Images: ROC/val, ROC/train, PR/val, PR/train, CM/val, CM/train
    """
    def __init__(self, y_pred_key='y_pred', y_true_key='y_true', invert=False,
                 plot_every_n_epochs=10):
        """
        
        Parameters
        ----------
        y_pred_key : str, optional
            Key for predicted labels in the train_step_outputs and validation_step_outputs dictionaries, by default 'y_pred'
        y_true_key : str, optional
            Key for true labels in the train_step_outputs and validation_step_outputs dictionaries, by default 'y_true'
        invert : bool, optional
            If True, do 1 - y_pred, use if y_pred is distance instead of probability, by default False
        plot_every_n_epochs : int, optional
            Plot curves every n epochs, by default 10
        """
        super().__init__()
        self.y_pred_key = y_pred_key
        self.y_true_key = y_true_key
        self.invert = invert
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        y_pred = torch.cat([x[self.y_pred_key] for x in pl_module.validation_step_outputs], dim=0)
        y_true = torch.cat([x[self.y_true_key] for x in pl_module.validation_step_outputs], dim=0)
        for metric, value in zip(["ROC", "PR", "CM"], ["AUC", "AUC", "F1"]):
            metric_value, metric_disp = get_metrics_and_curves(metric, y_pred, y_true, invert=self.invert)
            pl_module.log(f"{metric}/val_{value}", metric_value)
            if trainer.current_epoch % self.plot_every_n_epochs == 0:
                trainer.logger.experiment.add_figure(f"{metric}/val", metric_disp, global_step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        y_pred = torch.cat([x[self.y_pred_key] for x in pl_module.train_step_outputs], dim=0)
        y_true = torch.cat([x[self.y_true_key] for x in pl_module.train_step_outputs], dim=0)
        for metric, value in zip(["ROC", "PR", "CM"], ["AUC", "AUC", "F1"]):
            metric_value, metric_disp = get_metrics_and_curves(metric, y_pred, y_true, invert=self.invert)
            pl_module.log(f"{metric}/train_{value}", metric_value)
            if trainer.current_epoch % self.plot_every_n_epochs == 0:
                trainer.logger.experiment.add_figure(f"{metric}/train", metric_disp, global_step=trainer.global_step)

class LogDistances(Callback):
    """
    Logs histograms of continuous y_true and y_pred and their difference
    """

    def __init__(self, name="Distances", y_pred_key='predicted_distances', y_true_key='distances'):
        super().__init__()
        self.name = name
        self.y_pred_key = y_pred_key
        self.y_true_key = y_true_key

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.y_true_key in pl_module.validation_step_outputs[0]:
            y_true = torch.cat([x[self.y_true_key] for x in pl_module.validation_step_outputs], dim=0)
            if trainer.current_epoch == 1:
                trainer.logger.experiment.add_histogram(f"{self.name}/true", 
                                                        y_true, 
                                                        global_step=trainer.global_step)
            y_pred = torch.cat([x['y_pred'] for x in pl_module.validation_step_outputs], dim=0)
            trainer.logger.experiment.add_histogram(f"{self.name}/predicted", 
                                                    y_pred, 
                                                    global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram(f"{self.name}/difference", 
                                                    torch.abs(y_pred - y_true), 
                                                    global_step=trainer.global_step)

class ClearOutputs(Callback):
    """
    Clears the outputs of the model after each epoch
    """
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.train_step_outputs.clear()
        
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.validation_step_outputs.clear()
