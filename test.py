import gc
import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from brats_dataset import Brats
import evaluation
import tqdm

def test(config):
    np.random.seed(101)
    dataset = Brats(transform=transforms.ToTensor())
    # dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [90000, 2897], generator=torch.Generator().manual_seed(99))
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(99))
    train_iter = DataLoader(dataset=train_dataset, batch_size = config.training.batch_size, shuffle = True, pin_memory = True)
    eval_iter = DataLoader(dataset=test_dataset, batch_size = config.eval.batch_size, shuffle = False, pin_memory = True)
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    all_pools = []
    all_pools_shuffle = []

    for batch in eval_iter:
        # batch = batch[0]
        if batch.shape[0] != config.eval.batch_size: continue
        real_img = batch.to(config.device)
        real_img = torch.cat([real_img, real_img, real_img], dim=1)
        real_img = np.clip(real_img.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        real_img_shuffle = np.copy(real_img)
        np.random.shuffle(real_img)
        
        gc.collect()
        latents = evaluation.run_inception_distributed(real_img, inception_model, inceptionv3=inceptionv3)
        all_pools.append(latents["pool_3"])
        gc.collect()
        latents = evaluation.run_inception_distributed(real_img_shuffle, inception_model, inceptionv3=inceptionv3)
        all_pools_shuffle.append(latents["pool_3"])

    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]
    all_pools_shuffle = np.concatenate(all_pools_shuffle, axis=0)[:config.eval.num_samples]

    fid = evaluation.cal_fid(all_pools,all_pools_shuffle)
    print(fid)

if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image
    import tensorflow as tf
    recon = np.load("./bratswork/recon/reconstructions.npz")['recon'][:64]
    recon = torch.from_numpy(recon.astype(np.float32) / 255.).permute(0,3,1,2).numpy()
    print(recon.shape)

    # test_imgs = np.load("./test_data/BraTS.npz")['all_imgs']
    # current_batch = torch.from_numpy(np.asarray(test_imgs[:64][...,None], dtype=np.float32) / 255.).permute(0,3,1,2).numpy()

    dataset = Brats(transform=transforms.ToTensor())
    _, test_dataset = torch.utils.data.random_split(dataset, [90000, 2897])
    eval_iter = DataLoader(dataset=test_dataset, batch_size = 64, shuffle = False, pin_memory = True)
    for x in eval_iter:
        current_batch = x.numpy()
        break
    print(current_batch.shape)

    nrow = int(np.sqrt(recon.shape[0]))
    image_grid_recon = make_grid(torch.from_numpy(recon), nrow, padding=2)
    with tf.io.gfile.GFile("recon.png", "wb") as fout:
        save_image(image_grid_recon, fout)
    image_grid = make_grid(torch.from_numpy(current_batch), nrow, padding=2)
    with tf.io.gfile.GFile("test_img.png", "wb") as fout:
        save_image(image_grid, fout)
    