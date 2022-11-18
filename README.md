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
in `.bashrc` with corresponding cuda version for different az machine so that nvcc is working. 

## Current Problems
CUDA Out of Memory (OOM)  
To avoid this, I changed the batch size to 16 and ran the training process for brats dataset on 4 RTX2080. The generated images lose detailed information. Might be caused by the changing of the batch size. The training process uses about 9GB for each GPU so at most 40GB in total. However, when I trained this model on one RTX A6000 which has 48GB, it failed by OOM. Also, I tried to train cifar10 on one RTX A6000 and it also failed by OOM. 

