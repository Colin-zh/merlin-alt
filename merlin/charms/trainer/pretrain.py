# MIT License
# Copyright (c) 2026 Colin-zh
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..models.bert import BERTBackbone
from .optim_schedule import ScheduledOptim


class BERTTrainer:
    """
    BERTTrainer makes the pretrained BERT model with two LM training tasks:
        Masked Language Model and Next Sentence Prediction
    
    Args:
        bert: the BERT model
        vocab_size: the vocabulary size of the BERT model
        train_dataloader: the dataloader for training data
        test_dataloader: the dataloader for testing data
        lr: learning rate
        betas: Adam optimizer betas
        weight_decay: Adam optimizer weight decay
        warmup_steps: number of steps for the warmup phase in learning rate scheduling
        with_cuda: use CUDA or not
        cuda_devices: a list of CUDA device ids for multi-GPU training
        log_freq: the frequency to print the loss (in batches)
    """

    def __init__(
        self,
        bert: BERTBackbone,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        with_cuda: bool = True,
        cuda_devices: list = None,
        log_freq: int = 10,
    ):

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Initialize the BERT Language Model, with BERT model
        self.model = bert.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)
    
    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
    
    def iteration(self, epoch, data_loader, train=True):
        """
        Loop over the data_loader for training or testing.
        If on train status, backward operation is activated and also auto save the model every epoch.

        Args:
            epoch: epoch number
            data_loader: the DataLoader for the training or testing data
            train: train or test status
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct, total_element = 0, 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device (GPU or CPU)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the mlm and nsp model
            _, nsp_output, mlm_output = self.model(data["bert_input"], data["segment_label"])

            # 2-1. NLL loss for nsp task
            nsp_loss = self.criterion(nsp_output, data["is_next"])

            # 2-2. NLL loss for mlm task, calculated only on masked tokens
            mlm_loss = self.criterion(
                mlm_output.transpose(1, 2),  # [batch size, vocab size, seq len]
                data["bert_label"]  # [batch size, seq len]
            )

            # 2-3. combined loss
            loss = nsp_loss + mlm_loss

            # 3. backward and optimization only in train status
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            # next sentence prediction accuracy
            correct = nsp_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "nsp_acc": total_correct * 100.0 / total_element,
                "mlm_loss": mlm_loss.item(),
                "nsp_loss": nsp_loss.item(),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        
        print("EP%d_%s, avg_loss=%.4f, nsp_acc=%.4f" %
              (epoch, str_code, avg_loss / len(data_iter), total_correct * 100.0 / total_element))

    
    def save(self, epoch, file_path="outputs/bert_trained.model"):
        """ Saving the current BERT model to the file_path """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
