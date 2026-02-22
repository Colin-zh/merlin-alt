"""
Modified from torchkeras, licensed under Apache 2.0.
Original source: https://github.com/lyhue1991/torchkeras/blob/master/torchkeras/
Modifications made: Adapted for use in this project.

Original copyright: Copyright (c) lyhue1991
Modified work copyright: Copyright (c) Colin-zh

See LICENSE-APACHE for full license terms.
"""

import datetime
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from argparse import Namespace
from IPython.display import display

import numpy as np
import pandas as pd
from transformers import TrainerCallback

from ..utils import is_jupyter
from .vlog import VLog


def load_callback(callback):
    if isinstance(callback, str):
        if callback.lower() in ('vlog', 'lightgbm'):
            return VlogCallback
        elif callback.lower() == 'transformers':
            return TransformersCallback
        elif callback.lower() == 'tensorboard':
            return TensorBoardCallback
        elif callback.lower() == 'wandb':
            return WandbCallback
        else:
            raise ValueError(
                f"Unsupported callback string: {callback}, supported ones are " \
                "'vlog', 'transformers', 'tensorboard', 'wandb'.")
    else:
        return callback


class VlogCallback:
    def __init__(self, num_boost_round, 
                 monitor_metric='val_loss',
                 monitor_mode='min'):
        self.order = 20
        self.num_boost_round = num_boost_round
        self.vlog = VLog(epochs = num_boost_round, monitor_metric = monitor_metric, 
                         monitor_mode = monitor_mode)

    def __call__(self, env) -> None:
        metrics = {}
        for item in env.evaluation_result_list:
            print(item)
            if len(item) == 4:
                data_name, eval_name, result = item[:3]
                metrics[data_name+'_'+eval_name] = result
            else:
                data_name, eval_name = item[1].split()
                res_mean = item[2]
                res_stdv = item[4]
                metrics[data_name+'_'+eval_name] = res_mean
        self.vlog.log_epoch(metrics)


class TransformersCallback(TrainerCallback):
    def __init__(self, figsize=(6,4), update_freq=1, save_path='history.png'):
        self.figsize = figsize
        self.update_freq = update_freq
        self.save_path = save_path
        self.in_jupyter = is_jupyter()

    def on_fit_start(self, args, state, control, **kwargs):
        metric = args.metric_for_best_model
        self.greater_is_better = args.greater_is_better 
        self.prefix = 'val_' if metric.startswith('val_') else 'eval_'
        self.metric = metric

        dfhistory = pd.DataFrame()
        x_bounds = [0, args.logging_steps*10]
        self.update_graph(dfhistory, self.metric.replace(self.prefix,''), 
                             x_bounds = x_bounds, 
                             figsize = self.figsize)
            
    def on_fit_end(self, args, state, control, **kwargs):
        dfhistory = self.get_history(state)
        self.update_graph(dfhistory, self.metric.replace(self.prefix,''), 
                             figsize = self.figsize)
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        dfhistory = self.get_history(state)
        n = dfhistory['step'].max()
        if n%self.update_freq==0:
            x_bounds = [dfhistory['step'].min(), 
                        10*args.logging_steps+(n//(args.logging_steps*10))*args.logging_steps*10]
            self.update_graph(dfhistory, self.metric.replace(self.prefix,''), 
                              x_bounds = x_bounds, figsize = self.figsize)
        
    def get_history(self,state):
        log_history = state.log_history  
        train_history = [x for x in log_history if 'loss' in x.keys()]
        eval_history = [x for x in log_history if 'eval_loss'  in x.keys()]

        dfhistory_train = pd.DataFrame(train_history)
        dfhistory_eval = pd.DataFrame(eval_history)  
        dfhistory = dfhistory_train.merge(dfhistory_eval,on=['step','epoch'])
        if self.prefix=='val_':
            dfhistory.columns = [x.replace('eval_','val_') for x in dfhistory.columns]
        return dfhistory
    
    def get_best_score(self, dfhistory):
        arr_scores = dfhistory[self.metric]
        best_score = np.max(arr_scores) if self.greater_is_better==True else np.min(arr_scores)
        best_step = dfhistory.loc[arr_scores==best_score,'step'].tolist()[0]
        return (best_step, best_score)

    def update_graph(self, dfhistory, metric, x_bounds=None, 
                     y_bounds=None, figsize=(6,4)):
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)

        self.graph_ax.clear()
        steps = dfhistory['step'] if 'step' in dfhistory.columns else []

        m1 = metric
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(steps,train_metrics,'bo--',label= m1,clip_on=False)

        m2 = self.prefix+metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(steps,val_metrics,'co-',label =m2,clip_on=False)

        self.graph_ax.set_xlabel("step")
        self.graph_ax.set_ylabel(metric)  

        if m1 in dfhistory.columns or m2 in dfhistory.columns or metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')
            
        if len(steps)>0:
            best_step, best_score = self.get_best_score(dfhistory)
            self.graph_ax.plot(best_step,best_score,'r*',markersize=15,clip_on=False)
            title = f'best {self.metric} = {best_score:.4f} (@step {best_step})'
            self.graph_ax.set_title(title)
        else:
            title = f'best {self.metric} = ?'
            self.graph_ax.set_title(title)
            
        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        if self.in_jupyter:
            self.graph_out.update(self.graph_ax.figure)
        self.graph_fig.savefig(self.save_path)
        plt.close()


class TensorBoardCallback:
    def __init__(self, save_dir="runs", model_name="model",
                 log_weight=False, log_weight_freq=5):
        """
        TensorBoard callback for logging training progress

        Args:
        -  save_dir (str): Directory to save TensorBoard logs
        -  model_name (str): Name of the model
        -  log_weight (bool): Whether to log model weights
        -  log_weight_freq (int): Frequency of logging model weights during training
        """
        from torch.utils.tensorboard import SummaryWriter
        self.__dict__.update(locals())
        nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(save_dir, model_name, nowtime)
        self.writer = SummaryWriter(self.log_path)

    def on_fit_start(self, model):
        """
        Callback function called at the beginning of model fitting

        Args:
        - model (WandModel): The WandModel being trained
        """
        # Log model weights
        if self.log_weight:
            net = model.accelerator.unwrap_model(model.net)
            for name, param in net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
            self.writer.flush()

    def on_train_epoch_end(self, model):
        """
        Callback function called at the end of each training epoch

        Args:
        - model (WandModel): The WandModel being trained.
        """
        epoch = max(model.history['epoch'])

        # Log model weights
        net = model.accelerator.unwrap_model(model.net)
        if self.log_weight and epoch % self.log_weight_freq == 0:
            for name, param in net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()

    def on_validation_epoch_end(self, model):
        """
        Callback function called at the end of each validation epoch

        Args:
        - model (WandModel): The WandModel being trained
        """
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        epoch = max(model.history['epoch'])

        # Log metrics
        dic = deepcopy(dfhistory.iloc[n - 1])
        dic.pop("epoch")

        metrics_group = {}
        for key, value in dic.items():
            g = key.replace("train_", '').replace("val_", '')
            metrics_group[g] = dict(metrics_group.get(g, {}), **{key: value})
        for group, metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()

    def on_fit_end(self, model):
        """
        Callback function called at the end of model fitting

        Args:
        - model (WandModel): The WandModel being trained
        """
        # Log model weights
        epoch = max(model.history['epoch'])
        if self.log_weight:
            net = model.accelerator.unwrap_model(model.net)
            for name, param in net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()
        self.writer.close()

        # Save history
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(self.log_path, 'dfhistory.csv'), index=None)


class WandbCallback:
    def __init__(self, project=None, config=None, name=None, save_ckpt=True, save_code=True):
        """
        WandbCallback for logging training progress using Weights & Biases

        Args:
        - project (str): Name of the project in W&B
        - config (dict or Namespace): Configuration parameters
        - name (str): Name of the run.
        - save_ckpt (bool): Whether to save model checkpoints
        - save_code (bool): Whether to save code artifacts
        """
        self.__dict__.update(locals())
        if isinstance(config, Namespace):
            self.config = config.__dict__
        if name is None:
            self.name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        import wandb
        self.wb = wandb

    def on_fit_start(self, model):
        """
        Callback function called at the beginning of model fitting

        Args:
        - model (WandModel): The WandModel being trained
        """
        if self.wb.run is None:
            self.wb.init(project=self.project, config=self.config, name=self.name, save_code=self.save_code)
        model.run_id = self.wb.run.id

    def on_train_epoch_end(self, model):
        """
        Callback function called at the end of each training epoch

        Args:
        - model (WandModel): The WandModel being trained
        """
        pass

    def on_validation_epoch_end(self, model):
        """
        Callback function called at the end of each validation epoch

        Args:
        - model (WandModel): The WandModel being trained
        """
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        if n == 1:
            for m in dfhistory.columns:
                self.wb.define_metric(name=m, step_metric='epoch', hidden=False if m != 'epoch' else True)
            self.wb.define_metric(name='best_' + model.monitor, step_metric='epoch')

        dic = dict(dfhistory.iloc[n - 1])
        monitor_arr = dfhistory[model.monitor]
        best_monitor_score = monitor_arr.max() if model.mode == 'max' else monitor_arr.min()
        dic.update({'best_' + model.monitor: best_monitor_score})
        self.wb.run.summary["best_score"] = best_monitor_score
        self.wb.log(dic)

    def on_fit_end(self, model):
        """
        Callback function called at the end of model fitting

        Args:
        - model (WandModel): The WandModel being trained
        """
        # Save dfhistory
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(self.wb.run.dir, 'dfhistory.csv'), index=None)

        # Save ckpt
        if self.save_ckpt:
            arti_model = self.wb.Artifact('checkpoint', type='model')
            if os.path.isdir(model.ckpt_path):
                arti_model.add_dir(model.ckpt_path)
            else:
                arti_model.add_file(model.ckpt_path)
            self.wb.log_artifact(arti_model)

        run_dir = self.wb.run.dir
        self.wb.finish()

        # Local save
        try:
            import shutil
            copy_fn = shutil.copytree if os.path.isdir(model.ckpt_path) else shutil.copy
            copy_fn(model.ckpt_path, os.path.join(run_dir, os.path.basename(model.ckpt_path)))
        except Exception as err:
            print(err)


class VisProgress:
    def __init__(self):
        pass

    def on_fit_start(self, model):
        """Callback at the beginning of the training

        Args:
            model (WandModel): The WandModel instance
        """
        from .pbar import ProgressBar
        self.progress = ProgressBar(range(model.epochs))
        model.EpochRunner.progress = self.progress

    def on_train_epoch_end(self, model):
        """Callback at the end of each training epoch.

        Args:
            model (WandModel): The WandModel instance
        """
        pass

    def on_validation_epoch_end(self, model):
        """Callback at the end of each validation epoch

        Args:
            model (WandModel): The WandModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        self.progress.update(dfhistory['epoch'].iloc[-1])

    def on_fit_end(self, model):
        """Callback at the end of the entire training process

        Args:
            model (WandModel): The WandModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        if dfhistory['epoch'].max() < model.epochs:
            self.progress.on_interrupt(msg='')
        self.progress.display = False


class VisMetric:
    def __init__(self, figsize=(6, 4), save_path='history.png'):
        """Visualization callback for monitoring metrics

        Args:
            figsize (tuple, optional): Figure size. Defaults to (6, 4)
            save_path (str, optional): Path to save the history plot. Defaults to 'history.png'
        """
        self.figsize = figsize
        self.save_path = save_path
        self.in_jupyter = is_jupyter()

    def on_fit_start(self, model):
        """Callback at the beginning of the training

        Args:
            model (WandModel): The WandModel instance.
        """
        if not self.in_jupyter:
            print('\nView dynamic loss/metric plot: \n' + os.path.abspath(self.save_path))
        self.metric = model.monitor.replace('val_', '')
        dfhistory = pd.DataFrame(model.history)
        x_bounds = [0, min(10, model.epochs)]
        title = f'best {model.monitor} = ?'
        self.update_graph(model, title=title, x_bounds=x_bounds)

    def on_train_epoch_end(self, model):
        """Callback at the end of each training epoch

        Args:
            model (WandModel): The WandModel instance
        """
        pass

    def on_validation_epoch_end(self, model):
        """Callback at the end of each validation epoch

        Args:
            model (WandModel): The WandModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10 + (n // 10) * 10, model.epochs)]
        title = self.get_title(model)
        self.update_graph(model, title=title, x_bounds=x_bounds)

    def on_fit_end(self, model):
        """Callback at the end of the entire training process

        Args:
            model (WandModel): The WandModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        title = self.get_title(model)
        self.update_graph(model, title=title)

    def get_best_score(self, model):
        """Get the best score and epoch.

        Args:
            model (WandModel): The WandModel instance

        Returns:
            tuple: Best epoch and best score
        """
        dfhistory = pd.DataFrame(model.history)
        arr_scores = dfhistory[model.monitor]
        best_score = np.max(arr_scores) if model.mode == "max" else np.min(arr_scores)
        best_epoch = dfhistory.loc[arr_scores == best_score, 'epoch'].tolist()[0]
        return (best_epoch, best_score)

    def get_title(self, model):
        """Get the title for the plot

        Args:
            model (WandModel): The WandModel instance

        Returns:
            str: The title.
        """
        best_epoch, best_score = self.get_best_score(model)
        title = f'best {model.monitor}={best_score:.4f} (@epoch {best_epoch})'
        return title

    def update_graph(self, model, title=None, x_bounds=None, y_bounds=None):
        """Update the metric plot.

        Args:
            model (WandModel): The WandModel instance
            title (str, optional): Plot title. Defaults to None
            x_bounds (list, optional): x-axis bounds. Defaults to None
            y_bounds (list, optional): y-axis bounds. Defaults to None
        """
        import matplotlib.pyplot as plt
        self.plt = plt
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
            if self.in_jupyter:
                self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()

        dfhistory = pd.DataFrame(model.history)
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []

        m1 = "train_" + self.metric
        if m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(epochs, train_metrics, 'bo--', label=m1, clip_on=False)

        m2 = 'val_' + self.metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(epochs, val_metrics, 'co-', label=m2, clip_on=False)

        if self.metric in dfhistory.columns:
            metric_values = dfhistory[self.metric]
            self.graph_ax.plot(epochs, metric_values, 'co-', label=self.metric, clip_on=False)

        self.graph_ax.set_xlabel("epoch")
        self.graph_ax.set_ylabel(self.metric)
        if title:
            self.graph_ax.set_title(title)
            if not self.in_jupyter and hasattr(model.EpochRunner, 'progress'):
                model.EpochRunner.progress.comment_tail = title
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')

        if len(epochs) > 0:
            best_epoch, best_score = self.get_best_score(model)
            self.graph_ax.plot(best_epoch, best_score, 'r*', markersize=15, clip_on=False)

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        if self.in_jupyter:
            self.graph_out.update(self.graph_ax.figure)
        self.graph_fig.savefig(self.save_path)
        self.plt.close()


class VisDisplay:
    def __init__(self, display_fn, model=None, init_display=True, dis_period=1):
        """Visualization callback for displaying custom information during training

        Args:
            display_fn (callable): Function to display information. Should accept a WandModel instance
            model (WandModel, optional): The WandModel instance. Defaults to None
            init_display (bool, optional): Whether to display information initially. Defaults to True
            dis_period (int, optional): Display period (in epochs). Defaults to 1
        """
        from ipywidgets import Output
        self.display_fn = display_fn
        self.init_display = init_display
        self.dis_period = dis_period
        self.out = Output()

        if self.init_display:
            display(self.out)
            with self.out:
                self.display_fn(model)

    def on_fit_start(self, model):
        """Callback at the beginning of the training

        Args:
            model (WandModel): The WandModel instance
        """
        if not self.init_display:
            display(self.out)

    def on_train_epoch_end(self, model):
        """Callback at the end of each training epoch

        Args:
            model (WandModel): The WandModel instance
        """
        pass

    def on_validation_epoch_end(self, model):
        """Callback at the end of each validation epoch

        Args:
            model (WandModel): The WandModel instance
        """
        if len(model.history['epoch']) % self.dis_period == 0:
            self.out.clear_output()
            with self.out:
                self.display_fn(model)

    def on_fit_end(self, model):
        """Callback at the end of the entire training process

        Args:
            model (WandModel): The WandModel instance
        """
        pass


class EpochCheckpoint:
    def __init__(self, ckpt_dir="weights", save_freq=1, max_ckpt=10):
        """Callback for saving model checkpoints during training

        Args:
            ckpt_dir (str, optional): Directory to save checkpoints. Defaults to "weights"
            save_freq (int, optional): Save frequency (in epochs). Defaults to 1
            max_ckpt (int, optional): Maximum number of checkpoints to keep. Defaults to 10
        """
        self.__dict__.update(locals())
        self.ckpt_idx = 0

    def on_fit_start(self, model):
        """Callback at the beginning of the training

        Args:
            model (WandModel): The WandModel instance
        """
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.ckpt_list = ['' for i in range(self.max_ckpt)]

    def on_train_epoch_end(self, model):
        """Callback at the end of each training epoch

        Args:
            model (WandModel): The WandModel instance
        """
        pass

    def on_validation_epoch_end(self, model):
        """Callback at the end of each validation epoch

        Args:
            model (WandModel): The WandModel instance
        """
        dfhistory = pd.DataFrame(model.history)
        epoch = dfhistory['epoch'].iloc[-1]
        if epoch > 0 and epoch % self.save_freq == 0:
            ckpt_path = os.path.join(self.ckpt_dir, f'checkpoint_epoch{epoch}.pt')
            net_dict = model.accelerator.get_state_dict(model.net)
            model.accelerator.save(net_dict, ckpt_path)

            if self.ckpt_list[self.ckpt_idx] != '':
                os.remove(self.ckpt_list[self.ckpt_idx])
            self.ckpt_list[self.ckpt_idx] = ckpt_path
            self.ckpt_idx = (self.ckpt_idx + 1) % self.max_ckpt

    def on_fit_end(self, model):
        """Callback at the end of the entire training process

        Args:
            model (WandModel): The WandModel instance
        """
        pass
