import os
import torch
import numpy as np
import random
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.data import Folder
from anomalib.models import Patchcore, EfficientAd, Padim
from anomalib.engine import Engine
from anomalib import TaskType

SEED = 1234
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)


def initialize(seed=SEED):
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
    f1_score_list = []
    auroc_list = []

    for i in range(100):
        seed = 1000 + i
        print(f"seed = {seed}")
        initialize(seed)

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

        model = Padim(input_size=IMAGE_SIZE)

        engine = Engine(
            task=TaskType.CLASSIFICATION,
            image_metrics=['F1Score', 'AUROC'],
            pixel_metrics=['F1Score', 'AUROC'],
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="image_AUROC", mode="min", min_delta=0.0001, patience=10)],
        )

        engine.fit(
            datamodule=datamodule,
            model=model,
        )

        result = engine.test(
            datamodule=datamodule,
            model=model,
        )

        f1_score_list.append(result[0]['image_F1Score'])
        auroc_list.append(result[0]['image_AUROC'])

    print("image_F1Score")
    print("mean :", np.mean(f1_score_list))
    print("variance :", np.var(f1_score_list))
    print(f1_score_list)
    print()
    print("image_AUROC")
    print("mean :", np.mean(auroc_list))
    print("variance :", np.var(auroc_list))
    print(auroc_list)
