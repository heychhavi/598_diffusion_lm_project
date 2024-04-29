# EECS 498 Foundations of Large Language Models 
### Final Project: Diffusion-LM with Bigger Models and Smaller Steps
##### Group members: Zesen Zhao, Yaoxin (Selina) Li, Keqian Wang, Kevin Chen


## Reference

This code is forked from [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM), which is the repo for [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/pdf/2205.14217.pdf) NeurIPS 2022.


## Code contributions

We cleaned up a lot of files in the code, either to accomodate our testing or to make tweaks and fixes.

Some of the files we created or made key changes to:

 - [`diffusion-models/`](https://github.com/selina-lii/598_diffusion_lm_project/tree/main/diffusion-models): training models

- [`gaussian_diffusion.py`](https://github.com/selina-lii/598_diffusion_lm_project/blob/main/improved-diffusion/improved_diffusion/gaussian_diffusion.py): streamlined core algorithms

- [`infill.py`](https://github.com/selina-lii/598_diffusion_lm_project/blob/main/improved-diffusion/scripts/infill.py) and [`infill_parrot.py`](https://github.com/selina-lii/598_diffusion_lm_project/blob/main/improved-diffusion/scripts/infill_parrot.py): infilling task and a cleaned up version.

- [`bertscore.py`](https://github.com/selina-lii/598_diffusion_lm_project/blob/main/improved-diffusion/scripts/bertscore.py): diffusion experiments


Among many others.

-----------------------------------------------------
## Conda Setup:
```python
# setup on U-M GreatLakes
conda create -n env python=3.9.7
conda activate env
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ transformers/
pip install spacy==3.2.4 datasets==1.8.0 huggingface_hub==0.4.0 wandb
# magic
pip install --upgrade stanza stanza-spacy pydantic
# solves the mpi4py issue!!
module load gcc/8.2.0
module load openmpi/3.1.6
which mpicc # put your mpicc path here
env MPICC=/home/<username>/.conda/envs/498/bin/mpicc pip install --no-cache-dir mpi4py
```

```python
# original version
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb
```
-----------------------------------------------------
## Train Diffusion-LM:

```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000  --seed 101 --noise_schedule sqrt  --in_channel 128 --modality roc --submit no --padding_mode pad  --app "--predict_xstart True --training_mode e2e  --vocab_size 11043  --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64```


-------------------
## Decode Diffusion-LM:
mkdir generation_outputs 

``python scripts/batch_decode.py {path-to-diffusion-lm} -1.0 ema``


------------------- 
## Controllable Text Generation 
First, train the classsifier used to guide the generation (e.g. a syntactic parser) 

``  
python train_run.py --experiment e2e-tgt-tree  --app "--init_emb {path-to-diffusion-lm} --n_embd {16} --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10
``

Then, we can use the trained classifier to guide generation. 
(currently, need to update the classifier directory in scripts/infill.py. I will clean this up in the next release.)

``
python scripts/infill.py --model_path {path-to-diffusion-lm} --eval_task_ 'control_tree' --use_ddim True  --notes "tree_adagrad" --eta 1. --verbose pipe``



-----------------------------------------------------
```
Real Command feed into train.py:

OPENAI_LOGDIR=diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e  TOKENIZERS_PARALLELISM=false python scripts/train.py   --checkpoint_path diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e --model_arch transformer --modality e2e-tgt --save_interval 50000 --lr 0.0001 --batch_size 64  --diffusion_steps 2000 --noise_schedule sqrt  --use_kl False --learn_sigma False  --image_size 8 --num_channels 128 --seed 102 --dropout 0.1 --in_channel 16 --out_channel 16 --padding_mode block --experiment random  --lr_anneal_steps 200000 --weight_decay 0.0 --num_res_blocks 2  --predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data
```

presentation link for the paper: [link](https://slideslive.com/38990777/diffusionlm-improves-controllable-text-generation?ref=speaker-34175)
