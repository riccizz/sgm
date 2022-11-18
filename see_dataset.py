import os
import torch
import datasets
import tensorflow as tf
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from torchvision.utils import make_grid, save_image
import numpy as np

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)

def main(argv):
    config = FLAGS.config
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)
    for batch in train_iter:
        sample = torch.from_numpy(batch['image']._numpy())
        print(sample.shape)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample.permute(0, 3, 1, 2), nrow, padding=2)
        with tf.io.gfile.GFile(os.path.join("sample.png"), "wb") as fout:
            save_image(image_grid, fout)
        break

if __name__ == "__main__":
    app.run(main)
