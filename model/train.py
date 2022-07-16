import random
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from sklearn.model_selection import ParameterGrid
from pytorch_lightning import loggers as pl_loggers, callbacks
from torch.utils.data import DataLoader

from model import PhytoModel
from model.dataset import Dataset
from model.utils import get_preprocessing, get_training_augmentation

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if __name__ == '__main__':
    DATA_DIR = Path('photoshop_dataset/final_dataset').expanduser()
    TRAIN_DIR = DATA_DIR / "new_train"
    VALID_DIR = DATA_DIR / "new_valid"
    x_train_dir = TRAIN_DIR / 'original'
    y_train_dir = TRAIN_DIR / 'masks'

    x_valid_dir = VALID_DIR / 'original'
    y_valid_dir = VALID_DIR / 'masks'

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['phyto']
    ACTIVATION = 'sigmoid'
    params = {
        "encoder": [
            #"resnext50_32x4d",
            # "resnet50",
            "se_resnext50_32x4d",
            #"resnet18",
            #"resnet34",
            "resnet101",
            #"resnet152",
            #"timm-resnest14d",
            # "timm-resnest26d",
            # "timm-resnest50d",
            # "timm-resnest101e",
            # "timm-res2net50_26w_4s",
            # "timm-res2net101_26w_4s",
            # "timm-res2net50_26w_6s",
            # "timm-res2net50_26w_8s",
            # "timm-res2net50_48w_2s",
            # "timm-regnety_120",
            # "timm-regnety_080",
            # "timm-regnety_064",
            # "timm-regnety_040",
            # "timm-regnetx_120",
            # "timm-regnetx_160",
            # "timm-gernet_l",
            "se_resnext101_32x4d",
            # "se_resnext50_32x4d",
            # "se_resnet152",
            # "timm-skresnext50_32x4d",
            # "timm-skresnet34",
            # "densenet161",
            # "densenet201",
            "xception",
            "inceptionv4",
            "inceptionresnetv2",
            "efficientnet-b6",
            # "timm-mobilenetv3_large_100",
            # "timm-mobilenetv3_small_100",
            # "dpn98",
            # "dpn92",
            # "vgg19_bn",
            # "vgg19",
            # "vgg16_bn",
            # "vgg16"
        ],
        "lr": [0.00009, 0.00005, 0.0008, 0.0005, 0.0001]
    }
    grid_params = list(ParameterGrid(params))
    for iter_params in grid_params:
        encoder_name = iter_params.get('encoder')
        lr = iter_params.get('lr')
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name,
                                                             ENCODER_WEIGHTS)
        train_dataset = Dataset(
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        valid_dataset = Dataset(
            x_valid_dir,
            y_valid_dir,
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=4)
        monitor_metric = 'val_iou'

        model = PhytoModel("FPN", encoder_name, encoder_weights=ENCODER_WEIGHTS,
                           in_channels=3, out_classes=1,
                           target_metric=monitor_metric,
                           learning_rate=lr)
        model_name = f"{encoder_name}-{ENCODER_WEIGHTS}-{lr}"
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs-photoshop",
                                                 name=model_name)

        ckpt_callback = callbacks.ModelCheckpoint(
            monitor=monitor_metric,
            filename='model-{epoch:02d}-{val_iou:.5f}',
            save_top_k=2,
            mode='max',
            save_last=True,
            verbose=True,
        )
        early_callback = callbacks.EarlyStopping(
            monitor=monitor_metric,
            min_delta=0.001,
            patience=12,
            verbose=True,
            mode='max',
            check_finite=True,
            stopping_threshold=0.99999
        )

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=60,
            callbacks=[ckpt_callback, early_callback],
            logger=tb_logger
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        del trainer, model
        del train_loader, valid_loader
