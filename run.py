import os
import torch
import numpy as np
import random
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.utils.visualization import ImageVisualizer
from anomalib.data import Folder
from anomalib.models import Patchcore, EfficientAd, Padim
from anomalib.engine import Engine
from anomalib import TaskType

SEED = 1000
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)


def initialize(seed=SEED):
    print(f"seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    initialize(SEED)

    datamodule = Folder(
        root="datasets/BGA",
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        task=TaskType.CLASSIFICATION,
        image_size=IMAGE_SIZE,
        eval_batch_size=BATCH_SIZE,
        train_batch_size=BATCH_SIZE,
        seed=SEED,
    )

    model = EfficientAd(input_size=IMAGE_SIZE)

    engine = Engine(
        task=TaskType.CLASSIFICATION,
        image_metrics=['F1Score', 'AUROC'],
        pixel_metrics=['F1Score', 'AUROC'],
        accelerator="gpu",
        devices=1,
        visualizers=ImageVisualizer(),
        show_image=True,
        callbacks=[EarlyStopping(monitor="train_loss_epoch", mode="min", min_delta=0.01, patience=5)]
    )

    engine.fit(
        datamodule=datamodule,
        model=model,
    )

    result = engine.test(
        datamodule=datamodule,
        model=model,
    )