# Inspired by: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
from typing import Union, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from logger import Logger


class ModelWrapper(object):
    """
    This class implements a model wrapper for training a classification model.
    """

    def __init__(self,
                 model: Union[nn.Module, nn.DataParallel],
                 optimizer: torch.optim.Optimizer,
                 loss_function: nn.Module,
                 loss_function_test: nn.Module,
                 training_dataset: DataLoader,
                 test_dataset: DataLoader,
                 lr_schedule: Any,
                 # augmentation: Any,
                 validation_metric: nn.Module,
                 logger: Logger,
                 device: str = "cuda") -> None:
        """
        Constructor method
        :param model: (Union[nn.Module, nn.DataParallel]) Model to be trained
        :param optimizer: (Optimizer) Optimizer module
        :param loss_function: (nn.Module) Loss function
        :param training_dataset: (DataLoader) Training dataset
        :param test_dataset: (DataLoader) Test dataset
        :param validation_metric: (nn.Module) Validation metric
        :param device: (str) Device to be utilized
        """
        # Save parameters
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_function_test = loss_function_test
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.lr_schedule = lr_schedule
        # self.augmentation = augmentation
        self.validation_metric = validation_metric
        self.logger = logger
        self.device = device
        self.best_metric = 0.

    def train(self,
              epochs: int = 250) -> None:
        """
        Training function
        :param epoch: (int) Number of the current epoch
        """
        # Model to device
        self.model.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=(epochs * len(self.training_dataset)))
        # Training loop
        for epoch in range(epochs):
            # Model into train mode
            self.model.train()
            for index, (inputs, labels) in enumerate(self.training_dataset):
                # Update progress bar
                self.progress_bar.update(n=1)
                # Data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Perform augmentation
                # inputs, labels = self.augmentation(inputs, labels)
                # Reset gradients
                self.optimizer.zero_grad()
                # Make prediction
                predictions = self.model(inputs)
                # Calc loss
                loss = self.loss_function(predictions, labels)
                # Compute gradients
                loss.backward()
                # Perform optimization
                self.optimizer.step()
                # Print info in progress bar
                self.progress_bar.set_description(
                    "Epoch: {} | Loss: {:.4f}".format(epoch + 1, loss.item()))
                # Log loss and metric
                self.logger.log_metric(metric_name="training_loss", value=loss.item())
                # Learning rate schedule step
                self.lr_schedule.step_update(epoch * len(self.training_dataset) + index)
            # Perform testing
            self.test(epoch=epoch)
            # Save metrics
            self.logger.save()
        # Close progress bar
        self.progress_bar.close()
        # Final testing
        print("Training")
        self.test(train=True, print_results=True)
        print("Validation")
        self.test(print_results=True)

    @torch.no_grad()
    def test(self, epoch: int = -1, train: bool = False, print_results: bool = False) -> None:
        """
        Test function
        :param epoch: (int) Current epoch
        """
        # Model to device
        self.model.to(self.device)
        # Init list to store accuracies and losses
        metrics: List[float] = []
        losses: List[float] = []
        # Model into eval mode
        self.model.eval()
        # Training loop
        for index, (inputs, labels) in enumerate(self.test_dataset if not train else self.training_dataset):
            # Data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # Make prediction
            predictions = self.model(inputs)
            # Compute loss
            loss = self.loss_function_test(predictions, labels)
            # Get metric
            metric = self.validation_metric(predictions, labels)
            # Print progress bar
            self.progress_bar.set_description(
                "Testing | Loss: {:.4f} | Acc: {:.4f}".format(loss.item(), metric.item()))
            # Save metric and loss
            metrics.append(metric.item())
            losses.append(loss.item())
        # Print results if utilized
        if print_results:
            print("Accuracy:", np.mean(metrics))
            print("Loss:", np.mean(losses))
        # Save model
        if not train:
            # Log loss and metric
            self.logger.log_metric(metric_name="test_loss", value=np.mean(losses))
            self.logger.log_metric(metric_name="test_metric", value=np.mean(metrics))
            if np.mean(metrics) > self.best_metric:
                # Set new best accuracy
                self.best_metric = np.mean(metrics)
                # Print info
                print("Save best model with accuracy", self.best_metric)
                # Save model
                self.logger.save_model(
                    model_sate_dict={
                        "model": self.model.module.state_dict()
                        if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                        "acc": self.best_metric,
                        "epoch": epoch + 1,
                        "optimizer": self.optimizer.state_dict()},
                    name="best_model")
                # Save only the backbone
                self.logger.save_model(
                    model_sate_dict=self.model.module.model.state_dict()
                    if isinstance(self.model, nn.DataParallel) else self.model.model.state_dict(),
                    name="best_model_backbone")