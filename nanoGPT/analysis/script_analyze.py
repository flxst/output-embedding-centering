"""
Example: python analysis/script_analyze.py output
"""

import argparse
from typing import Dict, List, Optional, Tuple
import os
from os.path import abspath, dirname, join, isdir, isfile
import csv
import numpy as np
import scipy

import sys
ROOT_DIR = abspath(dirname(dirname(__file__)))
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)
from analysis.embeddings import get_input_embeddings_torch
from analysis.isotropy import compute_isotropy
from analysis.norms import compute_norm_stats, compute_align, compute_mu_norm, compute_all_norms, compute_ratio_norms, compute_Emu, compute_average_acc, extract_cos_sim, extract_dot_sim, load_isotropy, load_results_from_npy
from analysis.spectrum import compute_spectrum
from analysis.embeddings import Embeddings

USE_CENTERED_ISOTROPY = False
DATASET = 'fineweb'

def get_hardcoded_loss(_directory) -> float:
    # not used
    return -1.

def _get_directories(output_directory: str) -> List[str]:
    directories = [elem for elem in os.listdir(output_directory) if isdir(join(output_directory, elem))]
    directories = sorted(directories)
    return directories

def _get_last_file_in_directory(directory_path: str) -> Optional[str]:
    modalities = "mn5" in directory_path
    suffix = ".bin" if modalities else ".pt"
    files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem)) and elem.endswith(suffix)]
    if modalities:
        files = [file for file in files if "model" in file]
        files = sorted(files, key=lambda x: int(x.split("target_steps_")[-1].split("-target_tokens")[0]))
    else:
        files = sorted(files, key=lambda x: int(x.split("ckpt_")[-1].split(".pt")[0]))
    if len(files) == 0:
        files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem))]
        try:
            last_file = [elem for elem in files if elem.endswith('embeddings.npy')][0]
        except IndexError:
            return None
        last_file = last_file.split(".embeddings.npy")[0]
        return last_file
    try:
        last_file = files[-1]
    except IndexError:
        files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem)) and elem.endswith(".pt.embeddings.npy")]
        last_file = files[0].rstrip(".embeddings.npy")
    return last_file

def get_losses(output_directory: str) -> Dict[str, float]:
    modalities = "mn5" in output_directory
    losses = {}
    directories = _get_directories(output_directory)
    if modalities:
        for directory in directories:
            directory_path = join(output_directory, directory)
            losses[directory] = float(f"{get_hardcoded_loss(directory):.2f}")
    else:
        for directory in directories:
            directory_path = join(output_directory, directory)
            last_file = _get_last_file_in_directory(directory_path)
            if last_file is None:
                continue
            loss = float(last_file.split("_valloss=")[1].split(".pt")[0])
            losses[directory] = loss
    return losses

def get_test_losses_and_perplexities(output_directory: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    modalities = "mn5" in output_directory
    directories = _get_directories(output_directory)

    test_losses = {}
    perplexities = {}
    if modalities:
        for directory in directories:
            directory_path = join(output_directory, directory)
            test_losses[directory] = get_hardcoded_loss(directory)
            perplexities[directory] = np.exp(test_losses[directory])
    else:
        for directory in directories:
            directory_path = join(output_directory, directory)
            last_file = _get_last_file_in_directory(directory_path)
            if last_file is None:
                continue
            test_loss_file = join(directory_path, last_file + f'.valloss-{DATASET}.npy')
            if isfile(test_loss_file):
                test_losses[directory] = np.load(test_loss_file)
                perplexities[directory] = np.exp(test_losses[directory])
            else:
                print(f"> WARNING! could not find file {test_loss_file}")
    return test_losses, perplexities

def _change_suffix(_path, place):
    if place == 'input':
        _path = _path.replace('.npy', '-input.npy')
    elif place == 'output_centered':
        _path = _path.replace('.npy', '-output_centered.npy')
    return _path

def get_embeddings(output_directory: str, place: str):
    directories = _get_directories(output_directory)
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if last_file is None:
            continue
        npy_path = f"{directory_path}/{last_file}.embeddings.npy"
        npy_path = _change_suffix(npy_path, place)
        if isfile(npy_path):
            pass
        else:
            _ = get_input_embeddings_torch(path=npy_path, place=place)

def get_isotropy(output_directory: str, place: str) -> Dict[str, float]:
    directories = _get_directories(output_directory)
    isotropy = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if last_file is None:
            continue
        try:
            npy_path = f"{directory_path}/{last_file}.isotropy.npy"
            npy_path = _change_suffix(npy_path, place)
            iso = compute_isotropy(path=npy_path, place=place)
        except Exception as e:
            print(f"> path = {npy_path}")
            raise Exception(e)
        isotropy[directory] = iso
    return isotropy

def get_norms(output_directory: str, place: str) -> Tuple[Dict[str, np.array], Dict[str, np.array], Dict[str, np.array]]:
    centered = place.endswith('centered')

    directories = _get_directories(output_directory)
    max_norms, avg_norms, min_norms = {}, {}, {}
    mu_norms, all_norms, ratio_norms = {}, {}, {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if last_file is None:
            continue
        npy_path = f"{directory_path}/{last_file}.maxnorm.npy"
        npy_path = _change_suffix(npy_path, place)
        max_norm = compute_norm_stats(path=npy_path, place=place, which='max', centered=centered)
        max_norms[directory] = max_norm
        npy_path = f"{directory_path}/{last_file}.avgnorm.npy"
        npy_path = _change_suffix(npy_path, place)
        avg_norm = compute_norm_stats(path=npy_path, place=place, which='avg', centered=centered)
        avg_norms[directory] = avg_norm
        npy_path = f"{directory_path}/{last_file}.minnorm.npy"
        npy_path = _change_suffix(npy_path, place)
        min_norm = compute_norm_stats(path=npy_path, place=place, which='min', centered=centered)
        min_norms[directory] = min_norm
        npy_path = f"{directory_path}/{last_file}.munorm.npy"
        npy_path = _change_suffix(npy_path, place)
        mu_norm = compute_mu_norm(path=npy_path, place=place, centered=centered)
        mu_norms[directory] = mu_norm
        npy_path = f"{directory_path}/{last_file}.allnorm.npy"
        npy_path = _change_suffix(npy_path, place)
        all_norm = compute_all_norms(path=npy_path, place=place, centered=centered)
        all_norms[directory] = all_norm
        npy_path = f"{directory_path}/{last_file}.rationorm.npy"
        npy_path = _change_suffix(npy_path, place)
        ratio_norm = compute_ratio_norms(path=npy_path, mu_norm=mu_norm, avg_norm=avg_norm)
        ratio_norms[directory] = ratio_norm
    return max_norms, avg_norms, min_norms, mu_norms, all_norms, ratio_norms

def get_align(output_directory: str, place: str) -> Tuple[Dict[str, np.array], Dict[str, np.array], Dict[str, np.array]]:
    centered = place.endswith('centered')

    directories = _get_directories(output_directory)
    max_aligndots, avg_aligndots, min_aligndots = {}, {}, {}
    max_aligncoss, avg_aligncoss, min_aligncoss = {}, {}, {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if last_file is None:
            continue

        # dot
        npy_path = f"{directory_path}/{last_file}.maxaligndot.npy"
        npy_path = _change_suffix(npy_path, place)
        max_aligndot = compute_align(path=npy_path, place=place, which='max', kind='dot', centered=centered)
        max_aligndots[directory] = max_aligndot
        npy_path = f"{directory_path}/{last_file}.avgaligndot.npy"
        npy_path = _change_suffix(npy_path, place)
        avg_aligndot = compute_align(path=npy_path, place=place, which='avg', kind='dot', centered=centered)
        avg_aligndots[directory] = avg_aligndot
        npy_path = f"{directory_path}/{last_file}.minaligndot.npy"
        npy_path = _change_suffix(npy_path, place)
        min_aligndot = compute_align(path=npy_path, place=place, which='min', kind='dot', centered=centered)
        min_aligndots[directory] = min_aligndot

        # cos
        npy_path = f"{directory_path}/{last_file}.maxaligncos.npy"
        npy_path = _change_suffix(npy_path, place)
        max_aligncos = compute_align(path=npy_path, place=place, which='max', kind='cos', centered=centered)
        max_aligncoss[directory] = max_aligncos
        npy_path = f"{directory_path}/{last_file}.avgaligncos.npy"
        npy_path = _change_suffix(npy_path, place)
        avg_aligncos = compute_align(path=npy_path, place=place, which='avg', kind='cos', centered=centered)
        avg_aligncoss[directory] = avg_aligncos
        npy_path = f"{directory_path}/{last_file}.minaligncos.npy"
        npy_path = _change_suffix(npy_path, place)
        min_aligncos = compute_align(path=npy_path, place=place, which='min', kind='cos', centered=centered)
        min_aligncoss[directory] = min_aligncos
    return max_aligndots, avg_aligndots, min_aligndots, max_aligncoss, avg_aligncoss, min_aligncoss

def get_spectrum(output_directory: str) -> Dict[str, float]:
    directories = _get_directories(output_directory)
    spectrum = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if last_file is None:
            continue
        spec = compute_spectrum(path=f"{directory_path}/{last_file}.spectrum.npy")
        spectrum[directory] = spec
    return spectrum

def get_Emu(output_directory: str, place: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    Emu = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if last_file is None:
            continue
        npy_path = f"{directory_path}/{last_file}.Emu.npy"
        npy_path = _change_suffix(npy_path, place)
        emu = compute_Emu(path=npy_path, place=place)
        Emu[directory] = emu
    return Emu

def get_lm_eval_results(output_directory: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    lm_eval_average_acc, lm_eval_average_acc_std = {}, {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        json_files = [
            elem for elem in os.listdir(directory_path) 
            if elem.startswith('results') and elem.endswith('.json')
        ]
        if len(json_files) == 1:
            results_path = join(directory_path, json_files[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                average_acc, average_acc_std = compute_average_acc(path=f"{directory_path}/{last_file}.lm-eval-avg-acc.npy",
                                                                   results_path=results_path)
                lm_eval_average_acc[directory] = average_acc
                lm_eval_average_acc_std[directory] = average_acc_std
    return lm_eval_average_acc, lm_eval_average_acc_std

def get_tmic_results(output_directory: str, place: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    tmic_isotropy = {}
    tmic_ebenchmark_original_cos_sim = {}
    tmic_ebenchmark_original_dot_sim = {}
    for directory in directories:
        directory_path = join(output_directory, directory)

        # isotropy
        json_files_isotropy = [
            elem for elem in os.listdir(directory_path) 
            if elem.startswith('isotropy') and elem.endswith('.jsonl')
        ]
        if place == 'output':
            json_files_isotropy = [
                elem for elem in json_files_isotropy
                if not elem.endswith('---input.jsonl')
            ]
        elif place == 'input':
            json_files_isotropy = [
                elem for elem in json_files_isotropy
                if elem.endswith('---input.jsonl')
            ]
        if len(json_files_isotropy) == 1:
            results_path = join(directory_path, json_files_isotropy[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                isotropy = load_isotropy(results_path)
                tmic_isotropy[directory] = isotropy

        # ebenchmark
        json_files_ebenchmark = [
            elem for elem in os.listdir(directory_path) 
            if elem.startswith('ebenchmark') and elem.endswith('.jsonl')
        ]
        if place == 'output':
            json_files_ebenchmark = [
                elem for elem in json_files_ebenchmark
                if not elem.endswith('---input.jsonl')
            ]
        elif place == 'input':
            json_files_ebenchmark = [
                elem for elem in json_files_ebenchmark
                if elem.endswith('---input.jsonl')
            ]
        if len(json_files_ebenchmark) == 1:
            results_path = join(directory_path, json_files_ebenchmark[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                cos_sim_path = f"{directory_path}/{last_file}.tmic-ebenchmark-original-cos-sim.npy"
                cos_sim_path = _change_suffix(cos_sim_path, place)
                cos_sim = extract_cos_sim(
                    path=cos_sim_path,
                    results_path=results_path
                )
                tmic_ebenchmark_original_cos_sim[directory] = cos_sim
                dot_sim_path = f"{directory_path}/{last_file}.tmic-ebenchmark-original-dot-sim.npy"
                dot_sim_path = _change_suffix(dot_sim_path, place)
                dot_sim = extract_dot_sim(
                    path=dot_sim_path,
                    results_path=results_path
                )
                tmic_ebenchmark_original_dot_sim[directory] = dot_sim
    return tmic_ebenchmark_original_cos_sim, tmic_ebenchmark_original_dot_sim, tmic_isotropy

def get_fhs_cos_sim(output_directory: str, place: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    fhs_cos_sim = {}
    for directory in directories:
        directory_path = join(output_directory, directory)

        # isotropy
        json_files_fhs_cos_sim = [
            elem for elem in os.listdir(directory_path) 
            if '.pt.cos_' in elem and elem.endswith('.npy')
        ]
        if place == 'output':
            json_files_fhs_cos_sim = [
                elem for elem in json_files_fhs_cos_sim
                if '.pt.cos_input' in elem
            ]
        elif place == 'input':
            json_files_fhs_cos_sim = [
                elem for elem in json_files_fhs_cos_sim
                if '.pt.cos_output' in elem
            ]
        if len(json_files_fhs_cos_sim) == 1:
            results_path = join(directory_path, json_files_fhs_cos_sim[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                _cos_sim = load_results_from_npy(results_path, 'fhs_cos_sim')
                fhs_cos_sim[directory] = _cos_sim

    return fhs_cos_sim

def get_fhs_dot_prod(output_directory: str, place: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    fhs_dot_prod = {}
    for directory in directories:
        directory_path = join(output_directory, directory)

        # isotropy
        json_files_fhs_dot_prod = [
            elem for elem in os.listdir(directory_path) 
            if '.pt.dot_' in elem and elem.endswith('.npy')
        ]
        if place == 'output':
            json_files_fhs_dot_prod = [
                elem for elem in json_files_fhs_dot_prod
                if '.pt.dot_input' in elem
            ]
        elif place == 'input':
            json_files_fhs_dot_prod = [
                elem for elem in json_files_fhs_dot_prod
                if '.pt.dot_output' in elem
            ]
        if len(json_files_fhs_dot_prod) == 1:
            results_path = join(directory_path, json_files_fhs_dot_prod[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                _dot_prod = load_results_from_npy(results_path, 'fhs_dot_prod')
                fhs_dot_prod[directory] = _dot_prod

    return fhs_dot_prod

def get_logits_stats(output_directory: str, which: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    logits_stats = {}
    for directory in directories:
        directory_path = join(output_directory, directory)

        # logits_stats
        json_files_logits_stats = [
            elem for elem in os.listdir(directory_path) 
            if elem.endswith(f'.pt.logits_{which}.npy')
        ]
        if len(json_files_logits_stats) == 1:
            results_path = join(directory_path, json_files_logits_stats[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                _logits_stats = load_results_from_npy(results_path, f'logits_{which}')
                logits_stats[directory] = _logits_stats

    return logits_stats

def get_npy(output_directory: str, which: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    results = {}
    for directory in directories:
        directory_path = join(output_directory, directory)

        # results
        if which == 'time':
            json_files_results = [
                elem for elem in os.listdir(directory_path) 
                if elem.endswith('time.npy')
            ]
        else:
            json_files_results = [
                elem for elem in os.listdir(directory_path) 
                if elem.endswith(f'.pt.{which}.npy')
            ]
        if len(json_files_results) == 1:
            results_path = join(directory_path, json_files_results[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                if last_file is None:
                    continue
                _results = load_results_from_npy(results_path, which)
                results[directory] = _results

    return results

def compute_centering_ratio(_max_aligndot, _min_aligndot, _mu_norms):
    centering_ratios = {}
    for directory in _max_aligndot.keys():
        esq = _mu_norms[directory]**2
        dot_min = _min_aligndot[directory]
        dot_max = _max_aligndot[directory]

        Bminus = esq - dot_min
        Bplus = dot_max - esq

        # bound_before = np.max((-dot_min, dot_max))
        # bound_after = np.max((esq - dot_min, dot_max - esq))
        bound_before = np.max((Bminus, Bplus))
        bound_after = np.max((Bminus - esq, Bplus + esq))

        centering_ratios[directory] = bound_before / bound_after
        # centering_ratios[directory] = Rminus / (esq + Rplus)

    return centering_ratios

def main(output_directory: str):
    Emu, iso, max_norms, avg_norms, min_norms, mu_norms, all_norms, ratio_norms = {}, {}, {}, {}, {}, {}, {}, {}
    tmic_ebenchmark_original_cos_sim, tmic_ebenchmark_original_dot_sim, tmic_isotropy = {}, {}, {}
    fhs_cos_sim, fhs_dot_prod = {}, {}
    for place in ['input', 'output']:
        get_embeddings(output_directory, place)
        Emu[place] = get_Emu(output_directory, place)
        iso[place] = get_isotropy(output_directory, place)
        max_norms[place], avg_norms[place], min_norms[place], mu_norms[place], all_norms[place], ratio_norms[place] = get_norms(output_directory, place)
        tmic_ebenchmark_original_cos_sim[place], tmic_ebenchmark_original_dot_sim[place], tmic_isotropy[place] = get_tmic_results(output_directory, place)
        fhs_cos_sim[place] = get_fhs_cos_sim(output_directory, place)
        fhs_dot_prod[place] = get_fhs_dot_prod(output_directory, place)

    # NEW
    for place in ['output_centered']:
        max_norms[place], avg_norms[place], min_norms[place], mu_norms[place], all_norms[place], ratio_norms[place] = get_norms(output_directory, place)

    # NEW
    max_aligndot, avg_aligndot, min_aligndot, max_aligncos, avg_aligncos, min_aligncos = {}, {}, {}, {}, {}, {}
    centering_ratio = {}
    for place in ['output', 'output_centered']:
        max_aligndot[place], avg_aligndot[place], min_aligndot[place], max_aligncos[place], avg_aligncos[place], min_aligncos[place] = get_align(output_directory, place)
        centering_ratio[place] = compute_centering_ratio(max_aligndot[place], min_aligndot[place], mu_norms[place])

    lbp = {}  
    # lbp = get_lbp(output_directory)
    # losses = get_losses(output_directory)
    test_losses, perplexities = get_test_losses_and_perplexities(output_directory)
    lm_eval_average_acc, lm_eval_average_acc_std = get_lm_eval_results(output_directory)
    if USE_CENTERED_ISOTROPY is True:
        iso = {k: np.array(tmic_isotropy[k]['centered']) for k in iso}
        print("-> use centered isotropy!")
    corr_norm_pearson = {
        directory: scipy.stats.pearsonr(all_norms['output'][directory], lbp[directory]).statistic
        for directory in test_losses.keys()
        if directory in all_norms['output'].keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    corr_norm_spearman = {
        directory: scipy.stats.spearmanr(all_norms['output'][directory], lbp[directory]).statistic
        for directory in test_losses.keys()
        if directory in all_norms['output'].keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    corr_emu_pearson = {
        directory: scipy.stats.pearsonr(Emu['output'][directory], lbp[directory]).statistic
        for directory in test_losses.keys()
        if directory in all_norms['output'].keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    corr_emu_spearman = {
        directory: scipy.stats.spearmanr(Emu['output'][directory], lbp[directory]).statistic
        for directory in test_losses.keys()
        if directory in all_norms['output'].keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    spec = get_spectrum(output_directory)
    logits_mean_mean = get_logits_stats(output_directory, 'mean_mean')
    logits_mean_std = get_logits_stats(output_directory, 'mean_std')
    logits_std_mean = get_logits_stats(output_directory, 'std_mean')
    logits_std_std = get_logits_stats(output_directory, 'std_std')
    logits_mean_absmean = get_logits_stats(output_directory, 'mean_absmean')
    logits_mean_absmax = get_logits_stats(output_directory, 'mean_absmax')
    mean_logsqZ = get_npy(output_directory, 'mean_logsqZ')
    time = get_npy(output_directory, 'time')

    # print
    print("\n=====================================================================")

    # csv file
    rows = [
        [
            'name', 'test_loss (~)', 'test_loss', 'test_perplexity', 'lm-eval-avg-acc', 'lm-eval-avg-acc-std', 
            '-', '-', '-', '-', '-',
            'I.isotropy', 
            'I.max norm', 'I.avg norm', 'I.min norm', 'I.mu norm', 'I.ratio norm', 
            'I.fhs_cos_sim', 'I.fhs_dot_prod', 'I.eb_cos_sim',
            '-',
            'O.isotropy', 
            'O.max norm', 'O.avg norm', 'O.min norm', 'O.mu norm', 'O.ratio norm', 
            'O.max aligndot', 'O.avg aligndot', 'O.min aligndot',  # NEW
            'O.max aligncos', 'O.avg aligncos', 'O.min aligncos',  # NEW
            'O.centering_ratio',  # NEW
            'OC.max norm', 'OC.avg norm', 'OC.min norm', 'OC.mu norm', 'OC.ratio norm',  # NEW
            'OC.max aligndot', 'OC.avg aligndot', 'OC.min aligndot',  # NEW
            'OC.max aligncos', 'OC.avg aligncos', 'OC.min aligncos',  # NEW
            'OC.centering_ratio',  # NEW
            'O.fhs_cos_sim', 'O.fhs_dot_prod', 'O.eb_cos_sim',
            'O.eb_dot_sim', 'O.corr_p(norm, lbp)', 'O.corr_s(norm, lbp)', 
            'O.corr_p(Emu, lbp)', 'O.corr_s(Emu, lbp)', 
            'O.Smin', 'O.Smax', 'O.Smin/Smax',
            'O.logits_mean_mean', 'O.logits_mean_std', 'O.logits_std_mean', 'O.logits_std_std', 'O.logits_mean_absmean', 'O.logits_mean_absmax', 'H.mean_logsqZ', 'time',
        ]
    ]
    def correct_directory_name(_directory: str) -> str:

        if 'e-0' in _directory or 'e+0' in _directory:
            _corrected_directory_name = _directory
        else:
            if 1:
                parts = _directory.split('-')
                assert parts[-2].startswith('g'), f'ERROR! could not correct directory name = {_directory}, second last part = {parts[-2]} expected to start with letter g.'
                gamma = float(parts[-2][1:].replace('p', '.'))
                parts[-2] = f'g{gamma:.0e}'
                _corrected_directory_name = '-'.join(parts)
            else:
                _corrected_directory_name = _directory

        if 'baseline' in _directory:
            _corrected_directory_name = _corrected_directory_name.replace('g1e+00', 'g0e+00')
        elif 'zloss' in _directory:
            _corrected_directory_name = _corrected_directory_name.replace('g1e+00', 'g1e-04')

        return _corrected_directory_name

    for directory, loss in test_losses.items():
        directory_corrected_name = correct_directory_name(directory)
        _loss = f"{test_losses[directory]:.3f}" if directory in test_losses.keys() else ""
        _test_loss = f"{test_losses[directory]:.3f}" if directory in test_losses.keys() else ""
        _test_perplexity = f"{perplexities[directory]:.3f}" if directory in perplexities.keys() else ""
        _lm_eval_avg_acc = f"{lm_eval_average_acc[directory]:.3f}" if directory in lm_eval_average_acc.keys() else ""
        _lm_eval_avg_acc_std = f"{lm_eval_average_acc_std[directory]:.3f}" if directory in lm_eval_average_acc_std.keys() else ""

        _Iiso = f"{iso['input'][directory]:.3f}" if directory in iso['input'].keys() else ""
        _Imax_norms = f"{max_norms['input'][directory]:.3f}" if directory in max_norms['input'].keys() else ""
        _Iavg_norms = f"{avg_norms['input'][directory]:.3f}" if directory in avg_norms['input'].keys() else ""
        _Imin_norms = f"{min_norms['input'][directory]:.3f}" if directory in min_norms['input'].keys() else ""
        _Imu_norms = f"{mu_norms['input'][directory]:.3f}" if directory in mu_norms['input'].keys() else ""
        _Iratio_norms = f"{ratio_norms['input'][directory]:.3f}" if directory in ratio_norms['input'].keys() else ""
        _Ifhs_cos_sim = f"{fhs_cos_sim['input'][directory]:.4f}" if directory in fhs_cos_sim['input'].keys() else ""
        _Ifhs_dot_prod = f"{fhs_dot_prod['input'][directory]:.4f}" if directory in fhs_dot_prod['input'].keys() else ""
        _Ieb_cos_sim = f"{tmic_ebenchmark_original_cos_sim['input'][directory]:.1f}" if directory in tmic_ebenchmark_original_cos_sim['input'].keys() else ""
        _gap = "-"
        _iso = f"{iso['output'][directory]:.3f}" if directory in iso['output'].keys() else ""
        _max_norms = f"{max_norms['output'][directory]:.3f}" if directory in max_norms['output'].keys() else ""
        _avg_norms = f"{avg_norms['output'][directory]:.3f}" if directory in avg_norms['output'].keys() else ""
        _min_norms = f"{min_norms['output'][directory]:.3f}" if directory in min_norms['output'].keys() else ""
        _mu_norms = f"{mu_norms['output'][directory]:.3f}" if directory in mu_norms['output'].keys() else ""
        _ratio_norms = f"{ratio_norms['output'][directory]:.3f}" if directory in ratio_norms['output'].keys() else ""
        _max_aligndot = f"{max_aligndot['output'][directory]:.3f}" if directory in max_aligndot['output'].keys() else ""
        _avg_aligndot = f"{avg_aligndot['output'][directory]:.3f}" if directory in avg_aligndot['output'].keys() else ""
        _min_aligndot = f"{min_aligndot['output'][directory]:.3f}" if directory in min_aligndot['output'].keys() else ""
        _max_aligncos = f"{max_aligncos['output'][directory]:.3f}" if directory in max_aligncos['output'].keys() else ""
        _avg_aligncos = f"{avg_aligncos['output'][directory]:.3f}" if directory in avg_aligncos['output'].keys() else ""
        _min_aligncos = f"{min_aligncos['output'][directory]:.3f}" if directory in min_aligncos['output'].keys() else ""
        _centering_ratio = f"{centering_ratio['output'][directory]:.3f}" if directory in centering_ratio['output'].keys() else ""
        _Cmax_norms = f"{max_norms['output_centered'][directory]:.3f}" if directory in max_norms['output_centered'].keys() else ""        # NEW
        _Cavg_norms = f"{avg_norms['output_centered'][directory]:.3f}" if directory in avg_norms['output_centered'].keys() else ""        # NEW
        _Cmin_norms = f"{min_norms['output_centered'][directory]:.3f}" if directory in min_norms['output_centered'].keys() else ""        # NEW
        _Cmu_norms = f"{mu_norms['output_centered'][directory]:.3f}" if directory in mu_norms['output_centered'].keys() else ""           # NEW
        _Cratio_norms = f"{ratio_norms['output_centered'][directory]:.3f}" if directory in ratio_norms['output_centered'].keys() else ""  # NEW
        _Cmax_aligndot = f"{max_aligndot['output_centered'][directory]:.3f}" if directory in max_aligndot['output_centered'].keys() else ""
        _Cavg_aligndot = f"{avg_aligndot['output_centered'][directory]:.3f}" if directory in avg_aligndot['output_centered'].keys() else ""
        _Cmin_aligndot = f"{min_aligndot['output_centered'][directory]:.3f}" if directory in min_aligndot['output_centered'].keys() else ""
        _Cmax_aligncos = f"{max_aligncos['output_centered'][directory]:.3f}" if directory in max_aligncos['output_centered'].keys() else ""
        _Cavg_aligncos = f"{avg_aligncos['output_centered'][directory]:.3f}" if directory in avg_aligncos['output_centered'].keys() else ""
        _Cmin_aligncos = f"{min_aligncos['output_centered'][directory]:.3f}" if directory in min_aligncos['output_centered'].keys() else ""
        _Ccentering_ratio = f"{centering_ratio['output_centered'][directory]:.3f}" if directory in centering_ratio['output_centered'].keys() else ""
        _eb_dot_sim = f"{tmic_ebenchmark_original_dot_sim['output'][directory]:.1f}" if directory in tmic_ebenchmark_original_dot_sim['output'].keys() else ""
        _fhs_cos_sim = f"{fhs_cos_sim['output'][directory]:.4f}" if directory in fhs_cos_sim['output'].keys() else ""
        _fhs_dot_prod = f"{fhs_dot_prod['output'][directory]:.4f}" if directory in fhs_dot_prod['output'].keys() else ""
        _eb_cos_sim = f"{tmic_ebenchmark_original_cos_sim['output'][directory]:.1f}" if directory in tmic_ebenchmark_original_cos_sim['output'].keys() else ""

        _corr_norm_pearson = f"{corr_norm_pearson[directory]:.3f}" if directory in corr_norm_pearson.keys() else ""
        _corr_norm_spearman = f"{corr_norm_spearman[directory]:.3f}" if directory in corr_norm_spearman.keys() else ""
        _corr_emu_pearson = f"{corr_emu_pearson[directory]:.3f}" if directory in corr_emu_pearson.keys() else ""
        _corr_emu_spearman = f"{corr_emu_spearman[directory]:.3f}" if directory in corr_emu_spearman.keys() else ""
        if _corr_norm_pearson == "nan":
            _corr_norm_pearson = ""
        if _corr_norm_spearman == "nan":
            _corr_norm_spearman = ""
        if _corr_emu_pearson == "nan":
            _corr_emu_pearson = ""
        if _corr_emu_spearman == "nan":
            _corr_emu_spearman = ""
        _Smin = f"{np.min(spec[directory]):.2f}" if directory in spec.keys() else ""
        _Smax = f"{np.max(spec[directory]):.2f}" if directory in spec.keys() else ""
        _Sratio = f"{100*np.min(spec[directory])/np.max(spec[directory]):.2f}%" if directory in spec.keys() else ""
        _logits_mean_mean = f"{logits_mean_mean[directory]:.4f}" if directory in logits_mean_mean.keys() else ""
        _logits_mean_std = f"{logits_mean_std[directory]:.4f}" if directory in logits_mean_std.keys() else ""
        _logits_std_mean = f"{logits_std_mean[directory]:.4f}" if directory in logits_std_mean.keys() else ""
        _logits_std_std = f"{logits_std_std[directory]:.4f}" if directory in logits_std_std.keys() else ""
        _logits_mean_absmean = f"{logits_mean_absmean[directory]:.4f}" if directory in logits_mean_absmean.keys() else ""
        _logits_mean_absmax = f"{logits_mean_absmax[directory]:.4f}" if directory in logits_mean_absmax.keys() else ""
        _mean_logsqZ = f"{mean_logsqZ[directory]:.1f}" if directory in mean_logsqZ.keys() else ""
        _time = f"{time[directory]:.1f}" if directory in time.keys() else ""

        row = [
            directory_corrected_name, _loss, _test_loss, _test_perplexity, _lm_eval_avg_acc, _lm_eval_avg_acc_std, 
            _gap, _gap, _gap, _gap, _gap,
            _Iiso, 
            _Imax_norms, _Iavg_norms, _Imin_norms, _Imu_norms, _Iratio_norms, 
            _Ifhs_cos_sim, _Ifhs_dot_prod, _Ieb_cos_sim,
            _gap,
            _iso, 
            _max_norms, _avg_norms, _min_norms, _mu_norms, _ratio_norms, 
            _max_aligndot, _avg_aligndot, _min_aligndot, _max_aligncos, _avg_aligncos, _min_aligncos, # NEW
            _centering_ratio,  # NEW
            _Cmax_norms, _Cavg_norms, _Cmin_norms, _Cmu_norms, _Cratio_norms,  # NEW 
            _Cmax_aligndot, _Cavg_aligndot, _Cmin_aligndot, _Cmax_aligncos, _Cavg_aligncos, _Cmin_aligncos, # NEW
            _Ccentering_ratio,  # NEW
            _fhs_cos_sim, _fhs_dot_prod, _eb_cos_sim,
            _eb_dot_sim, _corr_norm_pearson, _corr_norm_spearman,
            _corr_emu_pearson, _corr_emu_spearman, 
            _Smin, _Smax, _Sratio,
            _logits_mean_mean, _logits_mean_std, _logits_std_mean, _logits_std_std, _logits_mean_absmean, _logits_mean_absmax, _mean_logsqZ, _time,
        ]
        rows.append(row)

        # print
        print(
            directory_corrected_name.ljust(40), 
            _test_loss,
            _test_loss,
            _test_perplexity,
            _lm_eval_avg_acc,
            _lm_eval_avg_acc_std,
            _gap, 
            _gap, 
            _gap, 
            _gap, 
            _gap, 
            _Iiso,
            _Imax_norms,
            _Iavg_norms,
            _Imin_norms,
            _Imu_norms,
            _Iratio_norms,
            _Ifhs_cos_sim,
            _Ifhs_dot_prod,
            _Ieb_cos_sim, 
            _gap, 
            _iso,
            _max_norms,
            _avg_norms,
            _min_norms,
            _mu_norms,
            _ratio_norms,
            _max_aligndot, 
            _avg_aligndot, 
            _min_aligndot, 
            _max_aligncos, 
            _avg_aligncos, 
            _min_aligncos,
            _centering_ratio,
            _Cmax_norms,
            _Cavg_norms,
            _Cmin_norms,
            _Cmu_norms,
            _Cratio_norms,
            _Cmax_aligndot, 
            _Cavg_aligndot, 
            _Cmin_aligndot, 
            _Cmax_aligncos, 
            _Cavg_aligncos, 
            _Cmin_aligncos,
            _Ccentering_ratio,
            _fhs_cos_sim,
            _fhs_dot_prod,
            _eb_cos_sim,
            _eb_dot_sim,
            _corr_norm_pearson,
            _corr_norm_spearman,
            _corr_emu_pearson,
            _corr_emu_spearman,
            _Smin,
            _Smax,
            _Sratio,
            _logits_mean_mean,
            _logits_mean_std,
            _logits_std_mean,
            _logits_std_std,
            _logits_mean_absmean,
            _logits_mean_absmax,
            _mean_logsqZ,
            _time,
        )

    filename = join(output_directory, "loss_overview.csv")
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
    print(f"> wrote table to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='output directory to analyze, e.g. "output"')
    parser.add_argument('directory')
    args = parser.parse_args()
    output_directory = join(ROOT_DIR, args.directory)
    main(output_directory)
