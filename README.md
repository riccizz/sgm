# Score-based Generative Model (SGM) Pytorch version

## Some Links
Paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

Official Repo [Jax version](https://github.com/yang-song/score_sde), [Pytorch version](https://github.com/yang-song/score_sde_pytorch), [Inverse problem](https://github.com/yang-song/score_inverse_problems)

## Dependencies and CUDA Setting
`requirements.txt` contains the dependencies song use. Some of the packages with specifed version is not available.  
`requirements114.txt` contains the dependencies I use.  

After install the dependencies in `requirements114.txt`, run 
```sh 
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
to install the cuda version of the jax and pytorch.  

Make sure to add 
```sh
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```
in `.bashrc` with corresponding cuda version for different az machine so that nvcc is working. Or if you use conda, load the conda lib path. 

## Dataset Brats
Link [Brats](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)  
The raw data (13G) is in `/data01/yichi5/brats_data`. You can also download them from the link above. To register and preprocess this dataset to be a tensorflow dataset, run `./brats/brats_test.py`. 

## Run
```sh
$ CUDA_VISIBLE_DEVICES=X python main.py [--config str] [--workdir str] [--mode str] [--eval_folder str]

--config: the configuration file
--workdir: the working directory
--mode: <train|eval|test|recon> running mode
--eval_folder: the folder name for storing evaluation results
```

## Current Issues & Working Status
### CUDA Out of Memory (OOM) (Solved)  
To avoid this, I changed the batch size to 16 and ran the training process for brats dataset on 4 RTX2080. The generated images lose detailed information. Might be caused by the changing of the batch size. The training process uses about 9GB for each GPU so at most 40GB in total. However, when I trained this model on one RTX A6000 which has 48GB, it failed by OOM. Also, I tried to train cifar10 on one RTX A6000 and it also failed by OOM.  
Solved by using PyTorch DataLoader. 

### Unable to Do Distributed Training  
After I changed to PyTorch DataLoader, whenever I try distributed training, the program crashes. 

### Working on the CT Parts
Finished tests on MRI. Reached the same level of PSNR and SSIM as the paper. Now working on the CT parts. 



