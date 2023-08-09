from typing import List

import torch

from model.model import swin_transformer_v2_t, SwinTransformerV2
from model.model_wrapper import ModelWrapper
from torch.optim import Adam
from torch.nn import MSELoss
import conf
from dataloader import getDataLoader
from logger import Logger

def train() -> None:
    # Check for cuda and set device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Make input tensor and init Swin Transformer V2, for the custom deformable version set use_deformable_block=True
    # input = torch.rand(2, 3, 256, 256, device=device)
    swin_transformer = swin_transformer_v2_t(in_channels=3,
                                                                window_size=8,
                                                                input_resolution=(256, 256),
                                                                sequential_self_attention=False,
                                                                use_checkpoint=False)
    optimizer = torch.optim.AdamW(swin_transformer.parameters(), lr=conf.lr)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    loss_function = MSELoss()
    training_dataset, test_dataset = getDataLoader(conf.data_path[1])
    logger = Logger()
    modelWrapper = ModelWrapper(
        model=swin_transformer,
        optimizer = optimizer,
        loss_function = loss_function,
        loss_function_test = loss_function,
        lr_schedule= lr_schedule,
        training_dataset = training_dataset,
        test_dataset = test_dataset,
        # augmentation: Any,
        validation_metric = loss_function,
        logger = logger,
        device = device
    )

    print(1)
    modelWrapper.train()
    print(2)

    # # Model to device
    # swin_transformer.to(device=device)
    # # Perform forward pass
    # features = swin_transformer(input)
    # # Print shape of features
    # for feature in features:
    #     print(feature.shape)



if __name__ == '__main__':
    train()