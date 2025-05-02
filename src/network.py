# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy
from src.metric import MyF1Score
import src.config as cfg
from src.util import show_setting
from einops import rearrange
import math


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self, num_classes: int = 200, dropout: float = 0.5):
        super().__init__(num_classes=num_classes)

        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # attention
        feature_dim = 256
        feature_map_size = 8 * 8
        num_heads = 8
        transformer_ff_dim = feature_dim * 4
        transformer_dropout = 0.1

        self.positional_encoding = nn.Parameter(torch.zeros(1, feature_map_size, feature_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # self.pre_ln = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)

        # spatial attention
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = self.pre_ln(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# original AlexNet
# class MyNetwork(AlexNet):
#     def __init__(self, num_classes: int = 200):
#         super().__init__(num_classes=num_classes)

#         # [TODO] Modify feature extractor part in AlexNet

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # [TODO: Optional] Modify this as well if you want
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork(num_classes=num_classes)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1_score = MyF1Score(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        # Optimizer
        optim_cfg = copy.deepcopy(self.hparams.optimizer_params)
        optimizer = getattr(torch.optim, optim_cfg.pop("type"))(
            self.parameters(), **optim_cfg
        )

        # Warm-up
        steps_per_epoch = math.ceil(100_000 / (cfg.BATCH_SIZE * cfg.NUM_GPUS))
        warmup_steps = 5 * steps_per_epoch
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )

        # RLROP
        pl_cfg = copy.deepcopy(self.hparams.scheduler_params)
        pl_cfg.pop("type", None)
        monitor_key = pl_cfg.pop("monitor", "loss/val")
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **pl_cfg
        )

        # Lightning
        return (
            [optimizer],
            [
                {  # warm-up
                    "scheduler": warmup,
                    "interval": "step",
                    "frequency": 1,
                    "name": "warmup_lr",
                },
                {  # plateau
                    "scheduler": plateau,
                    "interval": "epoch",
                    "monitor": monitor_key,
                    "reduce_on_plateau": True,
                    "name": "plateau_lr",
                },
            ],
        )


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'F1/train': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'F1/val': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
