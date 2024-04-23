This is a fork of ForecastTKGQA. [ForecastTKGQA HTML Paper](https://link.springer.com/chapter/10.1007/978-3-031-47240-4_29 "Link to the original paper"). [ForecastTKGQA repo](https://github.com/ZifengDing/ForecastTKGQA "Link to the original repo").

# Introduction

We have updated the code to support `torch-ddp` and ensured its compatibility with Linux running `Python 3.8`.

## Environment setup:

```markdown
Package            Version
------------------ ------------
numpy              1.19.0
pandas             1.2.4
scipy              1.9.3
torch              2.2.2+cu121
torchaudio         2.2.2+cu121
torchvision        0.17.2+cu121
tqdm               4.66.2
transformers       4.2.2
typing_extensions  4.10.0
```

Run this command to create the virtual env and install the required packages:

```bash
conda create -n fdp python=3.8
conda activate fdp
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy=1.19
conda install pandas=1.2.4
conda install transformers
```

While the version of `PyTorch` isn't critical, the versions of `numpy` and `pandas` are. Specifically, `numpy` should be version `1.19.0` or earlier, and `pandas` should be version `1.2.4` or earlier.

If you're having difficulty connecting to [hugging.co](hugging.co), you can download the `distilbert-base-uncased` model and save it in a folder. Subsequently, replace the string '/data/qing/distilbert-base-uncased' in the code with the path to your folder. If you can connect to, simply replace the string with 'distilbert-base-uncased'.

# Run code:

After doing other instructions in the original repo, you can run the code with the following command:

```bash
source activate && conda activate fdp
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 trainer.py --tkg_model_file tango_submission.pkl --model forecasttkgqa --lm_model distilbert --batch_size 512 --max_epochs 200 --valid_freq 1 --save_to forecasttkgqa --question_type entity_prediction
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 trainer.py --tkg_model_file tango_submission.pkl --model forecasttkgqa --lm_model distilbert --batch_size 256 --max_epochs 200 --valid_freq 1 --save_to forecasttkgqa --question_type yes_unknown
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 trainer.py --tkg_model_file tango_submission.pkl --model forecasttkgqa --lm_model distilbert --batch_size 256 --max_epochs 200 --valid_freq 1 --save_to forecasttkgqa --question_type fact_reasoning
```
