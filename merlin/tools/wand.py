"""
Modified from torchkeras, licensed under Apache 2.0.
Original source: https://github.com/lyhue1991/torchkeras/blob/master/torchkeras/
Modifications made: Adapted for use in this project.

Original copyright: Copyright (c) lyhue1991, zhangyu
Modified work copyright: Copyright (c) Colin-zh

See LICENSE-APACHE for full license terms.
"""

import datetime
import os
import sys
from argparse import Namespace
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..utils import is_jupyter
from .callbacks import WandbCallback, VisMetric, VisProgress


class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None, **kwargs):
        """Initialize the StepRunner object
        
        Args:
            net: Neural network model to be trained.
            loss_fn: Loss function to be used during training.
            accelerator: Accelerator object for distributed training.
            stage: Training stage (e.g., "train" or "val").
            metrics_dict: Dictionary of metrics to be evaluated.
            optimizer: Optimizer for training.
            lr_scheduler: Learning rate scheduler.
            **kwargs: Additional keyword arguments.
        """
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.kwargs = kwargs
        self.accelerator = accelerator

        # Set the network to training mode during the training stage, and evaluation mode during the validation stage
        if stage == "train":
            self.net.train()
        else:
            self.net.eval()
    
    def __call__(self, batch):
        """Perform a training or evaluation step.
        
        Args:
            batch: A batch of input data.
        
        Returns:
            Tuple of dictionaries containing the step loss and metrics.
        """
        features, labels = batch

        # Compute loss
        with self.accelerator.autocast():
            preds = self.net(features)
            loss = self.loss_fn(preds, labels)
        
        # Backward pass and optimization (only during training)
        if self.stage == "train" and self.optimizer is not None:
            self.accelerator.backward(loss)

            # Clip gradients if synchronization is enabled
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            
            self.optimizer.step()

            # Adjust learning rate if scheduler is provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Zero gradients after each optimization step
            self.optimizer.zero_grad()

        # Gather loss, predictions, and labels for metrics computation
        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)

        # Compute and gather additional metrics
        step_losses = {self.stage + "_loss": all_loss.item()}
        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
                        for name, metric_fn in self.metrics_dict.items()}
        
        # Include learning rate in metrics if available
        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics["lr"] = self.optimizer.state_dict()["param_groups"][0]["lr"]
            else:
                step_metrics["lr"] = 0.0
        
        return step_losses, step_metrics


class EpochRunner:
    def __init__(self, step_runner, quiet=False):
        """Initialize the EpochRunner. object

        Args:
            step_runner (StepRunner): The step runner to use for training and evaluation.
            quiet (bool, optional): If True, suppress output during training.
        """
        self.step_runner = step_runner
        self.quiet = quiet

        self.stage = step_runner.stage
        self.accelerator = step_runner.accelerator
        self.net = step_runner.net
    
    def __call__(self, dataloader):
        """Perform an epoch of training or evaluation.
        
        Args:
            dataloader: DataLoader providing batches of data.
        
        Returns:
            Dictionary containing epoch loss and metrics.
        """
        # Determin the size of the dataloader
        n = dataloader.size if hasattr(dataloader, 'size') else len(dataloader)

        # Initialize tqdm progress bar
        loop = tqdm(enumerate(dataloader, start=1),
                        total=n,
                        file=sys.stdout,
                        disable=self.quiet or not self.accelerator.is_local_main_process,
                        ncols=100
                    )
        epoch_losses = {}

        for step, batch in loop:
            # Perform a step with the provided StepRunner
            step_losses, step_metrics = self.step_runner(batch)
            step_log = dict(step_losses, **step_metrics)

            # Accumulate losses for the epoch
            for key, value in step_losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + value
            
            # Update progress bar with current metrics
            if step < n:
                loop.set_postfix(**step_log)
                if hasattr(self, "progress") and self.accelerator.is_local_main_process:
                    post_log = dict(**{"i": step, "n": n}, **step_log)
                    self.progress.set_postfix(**post_log)
            
            # Compute and diplay epoch-level metrics at the end of the epoch
            elif step == n:
                epoch_metrics = step_metrics
                epoch_metrics.update({self.stage + "_" + name: metric_fn.compute().item()
                                      for name, metric_fn in self.step_runner.metrics_dict.items()})
                epoch_losses = {k: v / step for k, v in epoch_losses.items()}
                epoch_log = dict(epoch_losses, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                # Update progress bar if available
                if hasattr(self, "progress") and self.accelerator.is_local_main_process:
                    post_log = dict(**{"i": step, "n": n}, **epoch_log)
                    self.progress.set_postfix(**post_log)
                
                # Reset metrics for the next epoch
                for metric in self.step_runner.metrics_dict.values():
                    metric.reset()
            
            else:
                break

        return epoch_log


class WandModel(nn.Module):
    StepRunner, EpochRunner = StepRunner, EpochRunner

    def __init__(self, net, loss_fn, metrics_dict=None, optimizer=None, lr_scheduler=None, **kwargs):
        """Initialize the WandModel. Wrapper for PyTorch models to add training and validation functionality.
        
        Args:
            net (nn.Module): The neural network model to be trained.
            loss_fn (callable): The loss function to be used during training.
            metrics_dict (dict, optional): A dictionary of metric functions to evaluate during training and validation.
            optimizer (torch.optim.Optimizer, optional): The optimizer to use for training the model.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for adjusting the 
                    learning rate during training.
            **kwargs: Additional keyword arguments for customization.
        """
        super().__init__()
        self.net, self.loss_fn, self.metrics_dict = net, loss_fn, nn.ModuleDict(metrics_dict)
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            self.net.parameters(), lr=3e-4
        )
        self.lr_scheduler = lr_scheduler
        self.kwargs = kwargs
        self.from_scratch = True
    
    def save_ckpt(self, ckpt_path, accelerator=None):
        """Save the model checkpoint to the specified path."""
        accelerator = accelerator if accelerator is not None else self.accelerator
        net_dict = accelerator.get_state_dict(self.net)
        accelerator.save(net_dict, ckpt_path)
    
    def load_ckpt(self, ckpt_path):
        """Load the model checkpoint from the specified path."""
        map_location = {'cuda:%d' % 0: 'cpu'} if not torch.cuda.is_available() else None
        net_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        self.net.load_state_dict(net_dict)
        self.from_scratch = False

    def forward(self, x):
        """Forward pass through the network."""
        return self.net.forward(x)
    
    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint', 
            patience=5, monitor="val_loss", mode="min", callbacks=None, plot=True,
            wandb=False, mixed_precision="no", cpu=False, gradient_accumulation_steps=1):
        """Train the model using the provided training and validation data.
        
        Args:
            train_data (DataLoader): DataLoader for training data.
            val_data (DataLoader, optional): DataLoader for validation data.
            epochs (int, optional): Number of epochs to train the model.
            ckpt_path (str, optional): Path to save the best model checkpoint.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped.
            monitor (str, optional): Metric to monitor for early stopping and checkpointing.
            mode (str, optional): One of {'min', 'max'}. In 'min' mode, training will stop when the monitored metric 
                    stops decreasing; in 'max' mode it will stop when the metric stops increasing.
            callbacks (list, optional): List of callback functions to be called during training.
            plot (bool, optional): Whether to plot training and validation metrics after training.
            wandb (bool, optional): Whether to log metrics to Weights & Biases (WandB).
            mixed_precision (str, optional): Mixed precision training mode. Options are 'no', 'fp16', 'bf16'.
            cpu (bool, optional): Whether to force training on CPU.
            gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before updating model 
                    weights.
        """
        self.__dict__.update(locals())
        from accelerate import Accelerator
        from merlin.utils import colorful, is_jupyter

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, cpu=cpu, gradient_accumulation_steps=gradient_accumulation_steps)
        device = str(self.accelerator.device)
        device_type = '🐌' if ('cpu' in device or 'mps' in device) else ('⚡️' if 'cuda' in device else '🚀')
        self.accelerator.print(
            colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>")
        )

        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler)
        
        for key in self.kwargs:
            self.kwargs[key] = self.accelerator.prepare(self.kwargs[key])
        
        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
        train_dataloader.size = train_data.size if hasattr(train_data, 'size') else len(train_data)
        train_dataloader.size = min(train_dataloader.size, len(train_dataloader))

        if val_data:
            val_dataloader.size = val_data.size if hasattr(val_data, 'size') else len(val_data)
            val_dataloader.size = min(val_dataloader.size, len(val_dataloader))
        
        self.history = {}
        callbacks = callbacks if callbacks is not None else []

        if bool(plot) & is_jupyter():
            callbacks += [VisProgress(), VisMetric()] + callbacks
        
        if wandb != False:
            project = wandb if isinstance(wandb, str) else "wandmodel"
            callbacks.append(WandbCallback(project=project))
        
        self.callbacks = [self.accelerator.prepare(cb) for cb in callbacks]

        if self.accelerator.is_local_main_process:
            [cb.on_fit_start(model=self) for cb in self.callbacks if hasattr(cb, 'on_fit_start')]
        
        start_epoch = 1 if self.from_scratch else 0
        quiet = bool(plot) & is_jupyter()

        for epoch in range(start_epoch, epochs + 1):
            if not quiet:
                now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.accelerator.print("\n" + "==========" * 8 + "%s" % now_time)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs) + "\n")
            
            # 1. Training phase
            train_step_runner = self.StepRunner(
                net=self.net,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),
                optimizer=self.optimizer if epoch > 0 else None,
                lr_scheduler=self.lr_scheduler if epoch > 0 else None,
                **self.kwargs
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, quiet)
            train_metrics = {"epoch": epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history.setdefault(name, []).append(metric)
            
            if self.accelerator.is_local_main_process:
                [cb.on_train_epoch_end(model=self) for cb in self.callbacks if hasattr(cb, 'on_train_epoch_end')]
            
            # 2. Validation phase
            if val_dataloader is not None:
                val_step_runner = self.StepRunner(
                    net=self.net,
                    loss_fn=self.loss_fn,
                    accelerator=self.accelerator,
                    stage="val",
                    metrics_dict=deepcopy(self.metrics_dict),
                    **self.kwargs
                )

                val_epoch_runner = self.EpochRunner(val_step_runner, quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history.setdefault(name, []).append(metric)
                
                if self.accelerator.is_local_main_process:
                    [cb.on_val_epoch_end(model=self) for cb in self.callbacks if hasattr(cb, 'on_val_epoch_end')]
                
            # 3. early stopping
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmin(arr_scores) if mode == "min" else np.argmax(arr_scores)

            if best_score_idx == len(arr_scores) - 1 and self.accelerator.is_local_main_process:
                self.save_ckpt(ckpt_path, accelerator = self.accelerator)
                if not quiet:
                    self.accelerator.print(
                        colorful(f"🎉🎉🎉 reach best {monitor} : {arr_scores[best_score_idx]}. "
                                 f"The best model has been saved to {ckpt_path} 🎉🎉🎉" )
                    )
            
            if len(arr_scores) - best_score_idx > patience:
                break
        
        if self.accelerator.is_local_main_process:
            dfhistory = pd.DataFrame(self.history)
            [cb.on_fit_end(model=self) for cb in self.callbacks if hasattr(cb, 'on_fit_end')]
            if epoch < epochs:
                self.accelerator.print(
                    colorful(f"Early stopping at epoch {epoch}. Best {monitor}: {arr_scores[best_score_idx]} at"
                             f" epoch {best_score_idx + 1}")
                )
            self.net = self.accelerator.unwrap_model(self.net)
            self.net.cpu()
            self.load_ckpt(ckpt_path)
            return dfhistory
    
    def evaluate(self, val_data, quiet=False):
        """Evaluate the model on the validation data.
        
        Args:
            val_data (DataLoader): DataLoader for validation data.
            quiet (bool, optional): Whether to suppress output during evaluation.
        
        Returns:
            dict: A dictionary of evaluation metrics.
        """
        # Ensure accelerator is available or create a new one
        from accelerate import Accelerator
        accelerator = Accelerator() if not hasattr(self, 'accelerator') else self.accelerator

        # Prepare model, loss function, and metrics for evaluation
        self.net, self.loss_fn, self.metrics_dict = accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict)
        
        val_dataloader = accelerator.prepare(val_data)

        # Initialize the step and epoch runners for validation
        val_step_runner = self.StepRunner(
            net=self.net,
            loss_fn=self.loss_fn,
            accelerator=accelerator,
            stage="val",
            metrics_dict=deepcopy(self.metrics_dict),
            **self.kwargs
        )
        val_epoch_runner = self.EpochRunner(val_step_runner, quiet)

        # Evaluate the model without computing gradients
        with torch.no_grad():
            val_metrics = val_epoch_runner(val_dataloader)
        
        return val_metrics
    
    def fit_ddp(self, num_processes, train_data,
                val_data=None, epochs=10, ckpt_path='checkpoint',
                patience=5, monitor="val_loss", mode="min", callbacks=None,
                plot=True, wandb=False, mixed_precision='no', 
                cpu=False, gradient_accumulation_steps=1):
        """
        Distributed Data Parallel (DDP) training for the model.

        Args:
            num_processes: Number of processes for DDP
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            ckpt_path: Path to save model checkpoints
            patience: Number of epochs with no improvement after which training will be stopped
            monitor: Metric to monitor for early stopping
            mode: 'min' for minimizing the monitor metric, 'max' for maximizing
            callbacks: List of callback functions
            plot: Whether to plot training progress
            wandb: Whether to use WandB for logging
            mixed_precision: Mixed precision training ('no', 'O1', 'O2', 'O3')
            cpu: Use CPU for training
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
        """
        # Import notebook_launcher from accelerate
        from accelerate import notebook_launcher

        # Create a tuple of arguments for the fit method
        args = (train_data, val_data, epochs, ckpt_path, patience, monitor, mode,
                callbacks, plot, wandb, mixed_precision, cpu, gradient_accumulation_steps)

        # Launch the fit method using notebook_launcher
        notebook_launcher(self.fit, args, num_processes=num_processes)
    
    def evaluate_ddp(self, num_processes, val_data, quiet=False):
        """
        Distributed Data Parallel (DDP) evaluation for the model

        Args:
            num_processes: Number of processes for DDP
            val_data: Validation data.
            quiet: Whether to suppress evaluation progress logs

        Returns:
            Dictionary of evaluation metrics
        """
        # Import notebook_launcher from accelerate
        from accelerate import notebook_launcher

        # Create a tuple of arguments for the evaluate method
        args = (val_data, quiet)

        # Launch the evaluate method using notebook_launcher
        notebook_launcher(self.evaluate, args, num_processes=num_processes)
