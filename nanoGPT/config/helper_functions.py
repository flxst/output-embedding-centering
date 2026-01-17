
from typing import Tuple
from settings import MICRO_BATCH_SIZE, LEARNING_RATE, WARMUP, WARMUP_WORTSMAN, BLOCK_SIZE, GPUS


def get_prefix(porian, wortsman):
    if porian:
        return 'por' 
    elif wortsman:
        return 'wor' 
    else:
        return 'exp'


def get_mbs_and_gas(batch_size: int, model_size_str: str, gpus: int, verbose: bool = False) -> tuple[int, int]:
    """
    batch_size = 4 * micro_batch_size * gradient_accumulation_steps
    => need to find combination of micro_batch_size & gradient_accumulation_steps that fulfills equation,
    with micro_batch_size as close as possible to its predefined value but not larger.
    """
    assert batch_size % 4 == 0, f'ERROR! batch_size = {batch_size} must be divisible by 4'

    batch_size_per_gpu = int(batch_size / gpus)
    micro_batch_size = MICRO_BATCH_SIZE[model_size_str]
    if verbose:
        print('\nINPUT:')
        print(f'> batch_size = {batch_size}')
        print(f'> gpus = {gpus}')
        print(f'> batch_size / gpu = {batch_size_per_gpu}')
        print(f'> max. micro_batch_size = {micro_batch_size}')
    success = False
    for candidate_mbs in range(micro_batch_size, 0, -1):
        if batch_size_per_gpu % candidate_mbs == 0:
            micro_batch_size = candidate_mbs
            gradient_accumulation_steps = int(batch_size_per_gpu / micro_batch_size)
            success = True
            break
    if success is False: 
        raise ValueError(f'ERROR! could not determine combination of micro_batch_size & gradient_accumulation_steps for batch_size = {batch_size}')
    if verbose:
        print('OUTPUT:')
        print(f'> micro_batch_size = {micro_batch_size}')
        print(f'> gradient_accumulation_steps = {gradient_accumulation_steps}')
        print()
    return micro_batch_size, gradient_accumulation_steps
    

def get_global_batch_size(micro_batch_size, block_size, gradient_accumulation_steps, gpus):
    return micro_batch_size * block_size * gradient_accumulation_steps * gpus


def get_beta2(_exp: str, porian: bool) -> float:
    if porian:
        if _exp in ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B']:
            return 0.99
        elif _exp in ['C', 'D', 'E', 'F']:
            return 0.95
        else:
            raise ValueError(f'unknown _exp = {_exp} when parsing beta2.')
    else:
        return 0.95


def get_model_size(_exp: str, porian: bool) -> str:
    if porian:
        if _exp == '1':
            return '5M'
        elif _exp == '2':
            return '8M'
        elif _exp == '3':
            return '10M'
        elif _exp == '4':
            return '16M'
        elif _exp == '5':
            return '23M'
        elif _exp == '6':
            return '29M'
        elif _exp == '7':
            return '37M'
        elif _exp == '8':
            return '57M'
        elif _exp == '9':
            return '85M'
        elif _exp == 'A':
            return '109M'
        elif _exp == 'B':
            return '149M'
        elif _exp == 'C':
            return '221M'
        elif _exp == 'D':
            return '347M'
        elif _exp == 'E':
            return '455M'
        elif _exp == 'F':
            return '612M'
        elif _exp == 'G':
            return '902M'

    else:
        if _exp == '1':
            return '125M'
        elif _exp == '2':
            return '355M'
        elif _exp == '3':
            return '760M'
        elif _exp == '4':
            return '1300M'

    raise Exception(f'ERROR! model size not defined for input {_exp} and porian = {porian}')

def get_dataset_size_and_steps(_exp: str, batch_size: int, porian: bool, wortsman: bool, scaling: bool, model_size_str: str = '') -> Tuple[str, str, str]:
    if porian:
        model_size = int(model_size_str[:-1]) * 10**6
        if scaling:
            if model_size_str == '5M':
                dataset_size = 32051200000
            elif model_size_str == '8M':
                dataset_size = 22221946880
            elif model_size_str == '10M':
                dataset_size = 17006592000
            elif model_size_str == '16M':
                dataset_size = 10683678720
            elif model_size_str == '23M':
                dataset_size =  7407697920
            elif model_size_str == '29M':
                dataset_size =  5807800320
            elif model_size_str == '37M':
                dataset_size =  4492492800
            elif model_size_str == '57M':
                dataset_size =  2903080960
            elif model_size_str == '85M':
                dataset_size =  1966080000
            elif model_size_str == '109M':
                dataset_size =  1536819200
            elif model_size_str == '149M':
                dataset_size =  1116733440
            elif model_size_str == '221M':
                dataset_size =   754974720
            elif model_size_str == '347M':
                dataset_size =   478412800
            elif model_size_str == '455M':
                dataset_size =   367001600
            elif model_size_str == '612M':
                dataset_size =   272629760
            elif model_size_str == '902M':
                dataset_size =   183500800
            else:
                raise ValueError(f'ERROR! model_size_str = {model_size_str} unknown in combination with --porian --scaling')
            
        else:
            dataset_size = 20 * model_size
        dataset_size_repr = f'{dataset_size/10**9:.1f}'.replace('.', 'p') + 'B'
    elif wortsman:
        dataset_size = 13107200000
        dataset_size_repr = f'{dataset_size/10**9:.1f}'.replace('.', 'p') + 'B'
    else:
        if _exp == '0':
            dataset_size = 0.1 * 10**9
            dataset_size_repr = '0p1B'
        elif _exp == '1':
            dataset_size = 1.0 * 10**9
            dataset_size_repr = '1B'
        elif _exp == '2':
            dataset_size = 5.0 * 10**9
            dataset_size_repr = '5B'
        elif _exp == '3':
            dataset_size = 10.0 * 10**9
            dataset_size_repr = '10B'
        elif _exp == '4':
            dataset_size = 15.0 * 10**9
            dataset_size_repr = '15B'
        elif _exp == '5':
            dataset_size = 20.0 * 10**9
            dataset_size_repr = '20B'
        elif _exp == '6':
            dataset_size = 25.0 * 10**9
            dataset_size_repr = '25B'
        else:
            raise Exception(f'ERROR! dataset size not defined for input {_exp}')

    steps = int(dataset_size / batch_size)
    # steps = int(steps / 1000) * 1000  # round such that it is divisible by 1000
    steps_str = f'{steps}'
    steps_repr = f'{int(steps/1000)}k'
    return dataset_size_repr, steps_repr, steps_str

def get_learning_rate(args_lr: int, model_size_str: str, wortsman: bool) -> Tuple[str, str, str]:
    lr = f'{args_lr}e-4' if args_lr > 0 else LEARNING_RATE[model_size_str]
    if wortsman:
        lr_min = '1e-5'
    else:
        lr_min = lr[:-1] + '6'
    lr_str = lr.split('e-4')[0].replace('.', 'p')
    return lr, lr_min, lr_str

def get_gpus(porian: bool, model_size_str: str) -> int:
    if porian:
        return GPUS[model_size_str]
    else:
        return GPUS["node"]

def get_model_architecture(model_size_str: str, porian: bool) -> Tuple[int, int, int]:
    if porian:
        n_head = 4
        if model_size_str == '5M':
            n_layer = 3
            n_embd = 96
        elif model_size_str == '8M':
            n_layer = 4
            n_embd = 128
        elif model_size_str ==  '10M':
            n_layer = 5
            n_embd = 160
        elif model_size_str == '16M':
            n_layer = 6
            n_embd = 224
        elif model_size_str == '23M':
            n_layer = 8
            n_embd = 288
        elif model_size_str == '29M':
            n_layer = 9 
            n_embd = 320
        elif model_size_str == '37M':
            n_layer = 10
            n_embd = 384
        elif model_size_str == '57M':
            n_layer = 12
            n_embd = 480
        elif model_size_str == '85M':
            n_layer = 14
            n_embd = 576
        elif model_size_str == '109M':
            n_layer = 15
            n_embd = 640
        elif model_size_str == '149M':
            n_layer = 18
            n_embd = 704
        elif model_size_str == '221M':
            n_layer = 21
            n_embd = 832
        elif model_size_str == '347M':
            n_layer = 23
            n_embd = 1024
        elif model_size_str == '455M':
            n_layer = 26
            n_embd = 1120
        elif model_size_str == '612M':
            n_layer = 26
            n_embd = 1312
        elif model_size_str == '902M':
            n_layer = 30
            n_embd = 1504
        else:
            raise ValueError(f'ERROR! model_size = {model_size_str} unknown.')
    else:
        if model_size_str == '125M':
            n_layer = 12
            n_head = 12
            n_embd = 768
        elif model_size_str == '355M':
            n_layer = 24
            n_head = 16
            n_embd = 1024
        elif model_size_str == '760M':
            n_layer = 24
            n_head = 16
            n_embd = 1536
        elif model_size_str == '1300M':
            n_layer = 24
            n_head = 16
            n_embd = 2048
        else:
            raise ValueError(f'ERROR! model_size = {model_size_str} unknown.')

    return n_layer, n_head, n_embd

def get_warmup(porian: bool, wortsman: bool, model_size_str: str = '', bs: str = '') -> int:
    if porian:
        # set warmup tokens equal to model size
        #
        # warmup_steps 
        # = warmup_tokens / batch_size_in_tokens 
        # = warmup_tokens / (batch_size_in_samples * sequence_length)
        model_size = int(model_size_str[:-1]) * 10**6
        return int(model_size / (bs * BLOCK_SIZE))
    elif wortsman:
        return WARMUP_WORTSMAN
    else:
        return WARMUP

def get_weight_decay(wortsman: bool) -> tuple[bool, float]:
    if wortsman:
        independent_weight_decay = False
        weight_decay = 0.
    else:
        independent_weight_decay = True
        weight_decay = 1e-4
    return independent_weight_decay, weight_decay

def get_log_interval(steps_str: str) -> int:
    steps_int = int(steps_str)
    if steps_int <= 20000:
        return 200
    else:
        return 1000

def get_variant(_exp: str) -> str:

    # no weight tying
    if _exp[0] == 'A':
        variant = 'baseline'
    elif _exp[0] == 'E':
        variant = f'muloss'
    elif _exp[0] == 'R':
        variant = f'mucentering'
    elif _exp[0] == 'Z':
        variant = f'zloss'

    # weight tying
    elif _exp[0] == 'a':
        variant = 'baselinewt'
    elif _exp[0] == 'e':
        variant = f'mulosswt'
    elif _exp[0] == 'r':
        variant = f'mucenteringwt'
    elif _exp[0] == 'z':
        variant = f'zlosswt'
    else:
        raise Exception(f'ERROR! variant = {_exp} unknown.')

    return variant
