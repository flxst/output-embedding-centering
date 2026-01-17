# Output Embedding Centering for Stable LLM Pretraining

This repository contains code to reproduce results from the paper https://arxiv.org/abs/2601.02031. 

## 0. TL;DR

- Implementation of mu-centering: [here](https://github.com/flxst/output-embedding-centering/blob/26f4196836ea83f676387acaa953e24bb3cc5f4d/nanoGPT/train.py#L628-L633)
- Implementation of mu-loss: [here](https://github.com/flxst/output-embedding-centering/blob/26f4196836ea83f676387acaa953e24bb3cc5f4d/nanoGPT/model.py#L247-L253)

## 1. Structure

### Experiments

Our model training code can be found in the folder `nanoGPT`. It is based on [nanoGPT](https://github.com/karpathy/nanoGPT) (commit [7a1614e](https://github.com/EleutherAI/lm-evaluation-harness/commit/7a1614eb90d29b2983ffa027a7974b7ef53fba19)).

Our changes include (but are not limited to) the use of 
- FineWeb
- RoPE, SwiGLU
- qk-layernorm & independent weight decay
- sequence length 2048
- customizable batch size & learning rate + automatic choice of micro batch size
- optional Xavier weight initialization
- optional weight tying

### Results

Our analysis code can be found in the folder `results`.


## 2. Preparation

### Create and Activate a Virtual Environment

```
# e.g. using conda
conda create -n venv26 python=3.11
conda activate venv26
```

### Install Dependencies

```
pip install torch==2.6 numpy transformers datasets tiktoken wandb tqdm rotary-embedding-torch scipy matplotlib seaborn jupyter scienceplots
```

### Download FineWeb 
~100B tokens, ~300GB disk space

```
cd nanoGPT/data
python prepare_fineweb.py   
```

Note that the above python script contains a `TARGET_DIRECTORY` variable that should be adjusted beforehand.


## 3. Run Experiments 

The experiments can be run step by step with the bash scripts listed in the following table.

| Script Name             | Purpose                                          |
| ------------------------| ------------------------------------------------ |
| nanoGPT/config/*.sh     | Create training config files                     |
| 0_run_training.sh       | Run training                                     |
| 1_prepare_validation.sh | Create validation config files                   |
| 2_run_validation.sh     | Run validation                                   |
| 3_aggregate.sh          | Aggregate results                                |

Note:

- Each bash script contains the commands for all main experiments (not the hyperparameter sensitivity experiments)

- The "Scale" variable in the bash scripts corresponds to the model size as follows:

    | Scale | Model Size |
    | ----- | ---------- |
    | 4     | 16M        |
    | 6     | 29M        |
    | 8     | 57M        |
    | A     | 109M       |
    | C     | 221M       |

- The "Method" variable in the bash scripts corresponds to the mitigation strategy as follows:

    | Method | Mitigation Strategy |
    | ----- | ------------ |
    | A     | baseline     |
    | E     | mu-loss      | 
    | R     | mu-centering |
    | Z     | z-loss       |

- W&B logging is turned off by default. To turn it on, change `wandb_log = False` to `wandb_log = True` in the config files and log in to W&B. 

- The output checkpoints from each experiment can be found in the subfolders of `nanoGPT/output`.

- The aggregated results can be found in `nanoGPT/output/loss_overview.csv`

- The actual experiments were conducted in parallel using slurm scripts


## 4. Analyze Results

The actual results,

- `loss_overview.csv` (main experiments)
- `loss_overview_all.csv` (main + hyperparameter sensitivity experiments)
- `checkpoints`

are analyzed using the jupyter notebooks in the `results` folder:

```
cd results
jupyter notebook
```

They produce figures and tables that can be found in 
- `results/figs` and 
- `results/tables`

respectively.
