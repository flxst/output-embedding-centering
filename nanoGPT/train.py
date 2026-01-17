"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from config.settings import SINGLE_CHECKPOINT

os.environ['PYTHONHASHSEED']=str(0)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
init_directory = ''
init_checkpoint = ''  # only for init_from = 'restart', needs to be be overwritten by string
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'fineweb'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
independent_weight_decay = True
weight_decay = 1e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
final_hidden_states_iterations = 4

seed = 1
activation = 'swiglu'
positional = 'rope'
weight_tying = False
z_loss = False
qk_layernorm = True
final_ln = True
final_ln_affine = True
muloss = False
mucentering = False
gamma = 1.  # a.k.a. lambda

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


if eval_only is True:
    wandb_log = False
    always_save_checkpoint = False

if len(init_directory) == 0:
    init_directory = out_dir

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

random.seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
torch.manual_seed(seed + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
if dataset == 'openwebtext':
    number_of_training_chunks = 1
elif dataset == 'fineweb':
    number_of_training_chunks = 1027
else:
    raise ValueError(f'ERROR! dataset = {dataset} unknown.')
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        # randomly sample training data chunk
        random_chunk = np.random.randint(1, 1+number_of_training_chunks)
        data = np.memmap(os.path.join(data_dir, f'{dataset}_train_{random_chunk:06d}.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f'{dataset}_val_000000.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(low=256*4 if dataset=='fineweb' else 0, high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, 
                  activation=activation, positional=positional, weight_tying=weight_tying,
                  z_loss=z_loss, qk_layernorm=qk_layernorm,
                  final_ln=final_ln, final_ln_affine=final_ln_affine,
                  muloss=muloss, gamma=gamma,
                  ) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    iter_num_skip_checkpointing = -1234  # will never happen
elif init_from == 'resume':
    print(f"Resuming training from {init_directory}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(init_directory, init_checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    iter_num_skip_checkpointing = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    iter_num_skip_checkpointing = -1234  # will never happen
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    splits = ['train', 'val'] if eval_only is False else ['val']
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, _, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def analyze_final_hidden_states(iterations: int, only_logits_mean: bool = False):
    DEBUG = False

    out = {}
    model.eval()

    # embedding
    if only_logits_mean is False:
        emean_input = torch.mean(raw_model.transformer.wte.weight, dim=0)  # [V, H] -> [H] 
        emean_output = torch.mean(raw_model.lm_head.embedding.weight, dim=0)  # [V, H] -> [H] 

    # final hidden state
    def compute_mean_logsqZ(_logits):
        # shapes: 
        #   logits = [k*B*T, V]
        #   => logsumexp = [k*B*T] 
        #   => square = [k*B*T] 
        #   => mean = [] 
        return torch.square(torch.logsumexp(_logits, dim=-1)).mean()  # []

    split = 'val'

    # aggregate batches -> fhs (=final hidden states), lgt (=logits)
    for k in range(iterations):
        X, Y = get_batch(split)
        with ctx:
            h = raw_model.get_final_hidden_states(X)         # [B, T, H]
            h = h.view(-1, h.size(-1))                       # [B*T, H]
            Eh = h @ raw_model.lm_head.embedding.weight.T    # [B, T, V]
            Eh = Eh.view(-1, Eh.size(-1))                    # [B*T, V]
        if k == 0:
            fhs = h   # [B*T, H]
            lgt = Eh  # [B*T, V]
        else:
            fhs = torch.cat((fhs, h), axis=0)   # [k*B*T, H]
            lgt = torch.cat((lgt, Eh), axis=0)  # [k*B*T, V]

    if DEBUG is True:
        print(f'fhs.shape={fhs.shape}')
        print(f'lgt.shape={lgt.shape}')

    # get statistics
    logits_mean_mean = torch.mean(torch.mean(lgt, dim=-1), dim=0)                 # []
    logits_mean_std = torch.mean(torch.std(lgt, dim=-1), dim=0)                   # []
    logits_std_mean = torch.std(torch.mean(lgt, dim=-1), dim=0)                   # []
    logits_std_std = torch.std(torch.std(lgt, dim=-1), dim=0)                     # []
    logits_mean_absmean = torch.mean(torch.mean(torch.abs(lgt), dim=-1), dim=0)   # []
    logits_mean_absmax = torch.mean(torch.max(torch.abs(lgt), dim=-1)[0], dim=0)  # []

    if only_logits_mean is False:
        hmean = torch.mean(fhs, dim=0)          # [H]
        mean_logsqZ = compute_mean_logsqZ(lgt)  # []

    if DEBUG is True:
        print(f'logits_mean_mean={logits_mean_mean}')
        print(f'logits_mean_std={logits_mean_std}')
        print(f'logits_std_mean={logits_std_mean}')
        print(f'logits_std_std={logits_std_std}')
        if only_logits_mean is False:
            print(f'hmean.shape={hmean.shape}')
            print(f'mean_logsqZ={mean_logsqZ}')

    # output
    if only_logits_mean is False:
        out['dot_input'] = torch.dot(emean_input, hmean)     # []
        out['dot_output'] = torch.dot(emean_output, hmean)   # []
        out['mean_logsqZ'] = mean_logsqZ                     # []
        emean_input = torch.unsqueeze(emean_input, dim=0)    # [1, H]
        emean_output = torch.unsqueeze(emean_output, dim=0)  # [1, H]
        hmean = torch.unsqueeze(hmean, dim=0)                # [1, H]
        out['cos_input'] = torch.squeeze(F.cosine_similarity(emean_input, hmean))    # []
        out['cos_output'] = torch.squeeze(F.cosine_similarity(emean_output, hmean))  # []

    out['logits_mean_mean'] = logits_mean_mean               # []
    out['logits_mean_std'] = logits_mean_std                 # []
    out['logits_std_mean'] = logits_std_mean                 # []
    out['logits_std_std'] = logits_std_std                   # []
    out['logits_mean_absmean'] = logits_mean_absmean         # []
    out['logits_mean_absmax'] = logits_mean_absmax           # []

    if DEBUG is True:
        print("output:")
        for k, v in out.items():
            print(f'{k}:{v.shape},{v}')

    out = {k: v.double().detach().cpu().numpy() for k, v in out.items()}
    model.train()
    return out

@torch.no_grad()
def get_mu_norms():
    out = {}
    # input
    emean_input = torch.mean(raw_model.transformer.wte.weight, dim=0)  # [V, H] -> [H] 
    emean_input_norm = torch.sqrt(torch.dot(emean_input, emean_input))
    out['e_input'] = emean_input_norm
    # output
    emean_output = torch.mean(raw_model.lm_head.embedding.weight, dim=0)  # [V, H] -> [H] 
    emean_output_norm = torch.sqrt(torch.dot(emean_output, emean_output))
    out['e_output'] = emean_output_norm
    # return
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr_decayed(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# learning rate constant scheduler (constant with warmup)
def get_lr_constant(it):
    if it < warmup_iters:
        # 1) linear warmup for warmup_iters steps
        return learning_rate * (it + 1) / (warmup_iters + 1)
    else:
        # 2) constant
        return learning_rate

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
t0_total = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
nr_norms = len(raw_model.transformer.h) + 1


while True:

    # determine and set the learning rate for this iteration
    lr = get_lr_decayed(iter_num) if decay_lr else get_lr_constant(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if independent_weight_decay is True:
            if param_group['weight_decay'] > 0:
                param_group['weight_decay'] = weight_decay / param_group['lr'] 

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 or iter_num == max_iters) and (iter_num > iter_num_skip_checkpointing) and master_process:
        if not eval_only:
            losses = estimate_loss()
            mu_norms = get_mu_norms()
            fhs = analyze_final_hidden_states(iterations=final_hidden_states_iterations, only_logits_mean=True)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "train/mu_norm_e_input": mu_norms['e_input'],
                "train/mu_norm_e_output": mu_norms['e_output'],
                "train/logits_mean_mean": fhs['logits_mean_mean'],
                "train/logits_mean_std": fhs['logits_mean_std'],
            }
            wandb.log(wandb_dict)

        if eval_only is False and (losses['val'] < best_val_loss or always_save_checkpoint):
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                _ckpt_name = init_checkpoint if SINGLE_CHECKPOINT is True else f'ckpt_{iter_num}.pt'
                torch.save(checkpoint, os.path.join(out_dir, _ckpt_name))
                if hasattr(raw_model, 'lm_head') and hasattr(raw_model.lm_head, 'scale'):
                    scale = raw_model.lm_head.scale
                    if isinstance(scale, torch.Tensor):
                        print(f"scale: min = {torch.min(scale)}, mean = {torch.mean(scale)}, max = {torch.max(scale)}")

    if eval_only is True:
        # validation loss
        losses = estimate_loss()
        val_loss = losses['val'].detach().numpy()
        if master_process:
            print("\n============================================================")
            print(f"> VALIDATION LOSS ON {eval_iters} BATCHES: {val_loss:.3f}")
            print("============================================================")
            val_loss_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.valloss-{dataset.replace('/', '-')}.npy"
            )
            with open(val_loss_path, 'wb') as f:
                np.save(f, val_loss)
            print(f"> wrote val_loss to file {val_loss_path}")

        # final hidden states: cosine similarity, dot product & z-loss
        fhs = analyze_final_hidden_states(iterations=final_hidden_states_iterations)
        if master_process:
            for position in ['input', 'output']:
                key = f'cos_{position}'
                print("\n============================================================")
                print(f"> COSINE SIMILARITY MU(e_{position}, h) ON {final_hidden_states_iterations} BATCHES: {fhs[key]:.4f}")
                print("============================================================")
                cos_path = os.path.join(
                    init_directory, 
                    f"{init_checkpoint}.{key}.npy"
                )
                with open(cos_path, 'wb') as f:
                    np.save(f, fhs[key])
                print(f"> wrote cos ({position}) to file {cos_path}")

            for position in ['input', 'output']:
                key = f'dot_{position}'
                print("\n============================================================")
                print(f"> DOT PRODUCT MU(e_{position}, h) ON {final_hidden_states_iterations} BATCHES: {fhs[key]:.4f}")
                print("============================================================")
                dot_path = os.path.join(
                    init_directory, 
                    f"{init_checkpoint}.{key}.npy"
                )
                with open(dot_path, 'wb') as f:
                    np.save(f, fhs[key])
                print(f"> wrote dot ({position}) to file {dot_path}")

            print("\n============================================================")
            print(f"> logits_mean_mean ON {final_hidden_states_iterations} BATCHES: {fhs['logits_mean_mean']:.4f}")
            print("============================================================")
            logits_mean_mean_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.logits_mean_mean.npy"
            )
            with open(logits_mean_mean_path, 'wb') as f:
                np.save(f, fhs['logits_mean_mean'])
            print(f"> wrote logits_mean_mean to file {logits_mean_mean_path}")

            print("\n============================================================")
            print(f"> logits_mean_std ON {final_hidden_states_iterations} BATCHES: {fhs['logits_mean_std']:.4f}")
            print("============================================================")
            logits_mean_std_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.logits_mean_std.npy"
            )
            with open(logits_mean_std_path, 'wb') as f:
                np.save(f, fhs['logits_mean_std'])
            print(f"> wrote logits_mean_std to file {logits_mean_std_path}")

            print("\n============================================================")
            print(f"> logits_std_mean ON {final_hidden_states_iterations} BATCHES: {fhs['logits_std_mean']:.4f}")
            print("============================================================")
            logits_std_mean_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.logits_std_mean.npy"
            )
            with open(logits_std_mean_path, 'wb') as f:
                np.save(f, fhs['logits_std_mean'])
            print(f"> wrote logits_std_mean to file {logits_std_mean_path}")

            print("\n============================================================")
            print(f"> logits_std_std ON {final_hidden_states_iterations} BATCHES: {fhs['logits_std_std']:.4f}")
            print("============================================================")
            logits_std_std_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.logits_std_std.npy"
            )
            with open(logits_std_std_path, 'wb') as f:
                np.save(f, fhs['logits_std_std'])
            print(f"> wrote logits_std_std to file {logits_std_std_path}")

            print("\n============================================================")
            print(f"> logits_mean_absmean ON {final_hidden_states_iterations} BATCHES: {fhs['logits_mean_absmean']:.4f}")
            print("============================================================")
            logits_mean_absmean_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.logits_mean_absmean.npy"
            )
            with open(logits_mean_absmean_path, 'wb') as f:
                np.save(f, fhs['logits_mean_absmean'])
            print(f"> wrote logits_mean_absmean to file {logits_mean_absmean_path}")

            print("\n============================================================")
            print(f"> logits_mean_absmax ON {final_hidden_states_iterations} BATCHES: {fhs['logits_mean_absmax']:.4f}")
            print("============================================================")
            logits_mean_absmax_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.logits_mean_absmax.npy"
            )
            with open(logits_mean_absmax_path, 'wb') as f:
                np.save(f, fhs['logits_mean_absmax'])
            print(f"> wrote logits_mean_absmax to file {logits_mean_absmax_path}")

            print("\n============================================================")
            print(f"> mean_logsqZ ON {final_hidden_states_iterations} BATCHES: {fhs['mean_logsqZ']:.4f}")
            print("============================================================")
            mean_logsqZ_path = os.path.join(
                init_directory, 
                f"{init_checkpoint}.mean_logsqZ.npy"
            )
            with open(mean_logsqZ_path, 'wb') as f:
                np.save(f, fhs['mean_logsqZ'])
            print(f"> wrote mean_logsqZ to file {mean_logsqZ_path}")

        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, loss_with_z, loss_with_regularization = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            loss_with_z = loss_with_z / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            loss_with_regularization = loss_with_regularization / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss_with_regularization).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    if mucentering:
        # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/4
        with torch.no_grad():
            emean_output = torch.mean(raw_model.lm_head.embedding.weight, dim=0)  # [V, H] -> [H] 
            new_embeddings = raw_model.lm_head.embedding.weight - emean_output
            raw_model.lm_head.embedding.weight.copy_(new_embeddings)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        lossz = loss_with_z.item() * gradient_accumulation_steps
        lossr = loss_with_regularization.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f} ({lossz:.4f}) [{lossr:.4f}], time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        # end loop
        break

if ddp:
    destroy_process_group()

t1_total = time.time()
dt_total = t1_total - t0_total
print()
print(f"> total time = {dt_total:.1f}s | steps = {max_iters} | total time / steps = {dt_total / max_iters:.4f}s")
if master_process:
    if wandb_log:
        wandb.log({
            "time/total": dt_total,
            "time/per_step": dt_total / max_iters,
        })

    if eval_only is False:
        time_path = os.path.join(
            init_directory, 
            "time.npy"
        )
        with open(time_path, 'wb') as f:
            np.save(f, np.array(dt_total))
        print(f"> wrote time to file {time_path}")

print("> DONE.")
