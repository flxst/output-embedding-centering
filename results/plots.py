import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# FONTS
import sys
from os.path import abspath
source_path = abspath('..')
if not source_path in sys.path:
    sys.path.append(source_path)
from font import SETTINGS
for key in SETTINGS:
    plt.rc(key, **SETTINGS[key])

COLOR = plt.rcParams['axes.prop_cycle']
COLOR = [v for elem in list(COLOR) for _, v in elem.items()]

##############
# PARAMETERS #
##############
CLR = {
    'A': COLOR[3],
    'E': COLOR[2],
    'R': COLOR[1],
    'Z': COLOR[0],
    'S': COLOR[6],
    'W': COLOR[4],
}

LABEL = {
    'A': 'baseline',
    'E': r'$\mu$-loss',
    'R': r'$\mu$-centering',
    'Z': 'z-loss',
    'S': 'soft-capping',
    'W': 'weight decay',
}

MARKER = {
    '4': 's',
    '6': 'd',
    '8': 'o',
    'A': 'P',
    'C': 'X',
}

NLABEL = {
    '4': '16M',
    '6': '29M',
    '8': '57M',
    'A': '109M',
    'C': '221M',
}

LINESTYLE = {
    '4': ':',
    '6': '--',
    '8': '-.',
    'A': '-',
    'C': '-',
}

ALPHA = {
    1e-7: 0.25,
    0.0001: 0.5,
    0.1: 0.75,
    100.0: 1.00,
    #
    30.0: 0.75,
}

YLABEL = {
    'test_loss': r'$\mathcal{L}$',
    'logits_mean_mean': r'$\overline{l}$',
    'logits_mean_std': r'$\sigma_l$',
    'logits_mean_absmean': r'$\overline{|l|}$',
    'logits_mean_absmax': r'$\max_j | l_j |$',
    'mu_norm': r'$\| \mu \|$',
    'max_norm': r'$\max_j \| e_j \|$',
    'avg_norm': r'$\overline{\| e_j \|}$',
    'min_norm': r'$\min_j \| e_j \|$',
    'isotropy': r'Iso',
    'fhs_dot_prod': r'$\mu \cdot \overline{h}$',
    'fhs_cos_sim': r'$\cos(\mu, \overline{h})$',
    'mean_norm_h': r'$\overline{\| h \|}$',
    'mean_norm_hperp': r'$\overline{\| h^\perp \|}$',
    'mean_norm_hpara': r'$\hat \mu \cdot \overline{ h }$',
    'mean_norm_hratio': r'$\overline{\| h^r \|}$',
    'mean_logsqZ': r'$\log^2 (Z)$',
    'time': r'$t [s]$',
}

##############
# FUNCTIONS  #
##############
def get_label(variant, _lambda):
    label = LABEL[variant]
    if variant in ['E', 'S', 'Z', 'W']:
        label = f'{label} ({_lambda:.0e})'.replace("-0", "-").replace("+0", "+")
    return label


def plot_wortsman(_lrs, _loss, quantity, ns, variants_lambdas, ylim=None, legend=False, alpha=False, details=False, save_as=''):
    """
    Args:
        _lrs: e.g. [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        _loss: e.g. '4': {
                        'A': {0.0: [3.961, 3.881, 3.85, 3.841, 3.857, 4.269, 5.065]},
                        'S': {30.0: [3.96, 3.88, 3.849, 3.837, 3.84, 3.842, 3.845]},
                        'W': {0.0001: [3.961, 3.881, 3.851, 3.843, 3.846, 3.944, 4.842]},
                        'Z': {1e-07: [3.961, 3.881, 3.85, 3.844, 3.866, 3.858, 3.871],
                              0.0001: [3.961, 3.878, 3.848, 3.836, 3.841, 3.854, 3.955],
                              0.1: [3.936, 3.856, 3.844, 3.835, 3.849, 3.847, 3.871],
                              100.0: [4.267, 4.188, 4.245, 4.6, 4.942, 6.358, 7.652]},
                        'E': {1e-07: [3.961, 3.881, 3.85, 3.84, 3.863, 3.977, 4.597],
                              0.0001: [3.961, 3.881, 3.849, 3.839, 3.843, 3.839, 3.845],
                              0.1: [3.961, 3.881, 3.848, 3.838, 3.838, 3.837, 3.845],
                              100.0: [3.964, 3.883, 3.85, 3.839, 3.836, 3.811, 3.816]},
                        'R': {0.0: [3.962, 3.881, 3.85, 3.842, 3.843, 3.844, 3.842]}
                        },
                     '6': {..}
    
        }
        quantity: e.g. 'test_loss'
        ns: e.g. ['4', '6', '8', 'A', 'C']
        variants_lambda: e.g. VARIANTS_LAMBDAS = {'A': [0.0], 'W': [0.0001], 'S': [30.0], 'Z': [0.0001], 'E': [0.0001], 'R': [0.0]}
        ylim: e.g. (2.9, 5.2)
    """
    if details is True:
        NR_PLOTS = 6
        fix, ax = plt.subplots(1, NR_PLOTS, figsize=(14,4), width_ratios=[4, 1, 1, 1, 1, 1])
    else:
        NR_PLOTS = 1
        fix, ax = plt.subplots(1, NR_PLOTS, figsize=(6,4))
        ax = [ax, None]

    for i in range(NR_PLOTS):
        ax[i].set_xscale('log')
    for counter, n in enumerate(ns):
        for variant in variants_lambdas:
            for _lambda in variants_lambdas[variant]:
                if _lambda in _loss[n][variant]:
                    ax[0].plot(
                        _lrs, 
                        _loss[n][variant][_lambda], 
                        marker=MARKER[n], 
                        linestyle=LINESTYLE[n],
                        color=CLR[variant], 
                        label=get_label(variant, _lambda) if n == ns[0] else None,
                        alpha=ALPHA[_lambda] if alpha is True else 1,
                    )
                    if details is True:
                        ax[1+counter].plot(
                            _lrs, 
                            _loss[n][variant][_lambda], 
                            marker=MARKER[n], 
                            linestyle=LINESTYLE[n],
                            color=CLR[variant], 
                            label=get_label(variant, _lambda) if n == ns[0] else None,
                            alpha=ALPHA[_lambda] if alpha is True else 1,
                        )
                else:
                    print(f'ERROR! could not plot lambda = {_lambda} for n = {n} & variant = {variant}.')
    if legend is True:
        ax[0].legend()
    if ylim is not None:
        for i in range(NR_PLOTS):
            ax[i].set_ylim(ylim)
    xlim = ax[0].get_xlim()
    for i in range(1, NR_PLOTS):
        ax[i].set_yticks([])
        ax[i].set_xlim([0.06, xlim[1]])
        ax[i].set_xticks([0.1], minor=True)
        ax[i].get_xaxis().get_major_formatter().labelOnlyBase = False
        ax[i].set_title(fr'{NLABEL[ns[i-1]]}')
    ax[0].set_xlabel(r'$\eta$')
    ax[0].set_ylabel(YLABEL[quantity])
    if len(save_as):
        assert save_as.endswith('.pdf') or save_as.endswith('.png')
        fig_path = f'figs/{save_as}'
        plt.savefig(fig_path, format=save_as[-3:], bbox_inches='tight')
        print(f'> saved as {fig_path}')


def plot_lr_sensitivity(_lr_sensitivity, model_size, ns, variants_lambdas, ylim = None, legend = False, alpha = False, save_as = ''):
    """
    Args:
        _lr_sensitivity: e.g. {
        'A': {0.0: {'4': 0.30616666666666653, '6': .., ..}}
        'W': {0.0001: {'4': 0.21116666666666659, '6': .., ..}}
        'S': {30.0: {'4': 0.03233333333333318, '6': .., ..}}
        'Z': {0.0001: {'4': 0.05350000000000018, '6': .., ..}}
        'E': {0.0001: {'4': 0.030666666666666693, '6': .., ..}}
        'R': {0.0: {'4': 0.02833333333333325, '6': .., ..}}
        }
        model_size: e.g. MODEL_SIZE = {'4': 16e6, '6': 29e6, '8': 57e6, 'A': 109e6, 'C': 221e6}
        ns: e.g. ['4', '6', '8', 'A', 'C']
        variants_lambda: e.g. VARIANTS_LAMBDAS = {'A': [0.0], 'W': [0.0001], 'S': [30.0], 'Z': [0.0001], 'E': [0.0001], 'R': [0.0]}
        ylim: e.g. (2.9, 5.2)
    """
    _, ax = plt.subplots(1, 1, figsize=(6,3))
    ax = [ax, None]
    for variant in variants_lambdas:
        for _lambda in variants_lambdas[variant]:
            x = [model_size[n] for n in ns]
            if _lambda in _lr_sensitivity[variant]:
                y = [elem for elem in _lr_sensitivity[variant][_lambda].values() if elem is not None]
                x = x[:len(y)]
                ax[0].loglog(
                    x, 
                    y, 
                    marker='.', 
                    linestyle='--', 
                    color=CLR[variant], 
                    label=get_label(variant, _lambda),
                    alpha=ALPHA[_lambda] if alpha is True else 1,
                )
                for i, (n, elem_x, elem_y) in enumerate(zip(ns, x, y)):
                   ax[0].loglog(
                    elem_x, 
                    elem_y, 
                    marker=MARKER[n], 
                    linestyle='', 
                    color=CLR[variant], 
                    alpha=ALPHA[_lambda] if alpha is True else 1,
                ) 
            else:
                print(f'ERROR! could not plot lambda = {_lambda} for variant = {variant}.')
    if legend is True:
        ax[0].legend()
    
    ax[0].set_xlabel(r'$N$')
    ax[0].set_ylabel('LRS')

    if ylim is not None:
        ax[0].set_ylim(ylim)
    else:
        ylim = ax[0].get_ylim()
    
    if len(save_as):
        assert save_as.endswith('.pdf') or save_as.endswith('.png')
        fig_path = f'figs/{save_as}'
        plt.savefig(fig_path, format=save_as[-3:], bbox_inches='tight')
        print(f'> saved as {fig_path}')

    return ylim


def plot_box(_lrs, quantity, percentage, model_sizes, titles, ylabel, ylabel2, _df_all, diverged=None, save_as=''):

    clr = CLR['A']

    if quantity in percentage:
        percentage_quantity = {
            model_size: [percentage[quantity][(model_size, lr)] for lr in _lrs]
            for model_size in model_sizes
        }

    if quantity == 'B_ratio':
        # ylim = [0, np.ceil(max(_df_all[quantity])*10)/10 + 0.1]
        ylim = [0, 4]
    elif quantity == 'lgt_mean':
        ylim_extreme = max(-np.floor(min(_df_all[quantity])*10)/10 - 0.1, np.ceil(max(_df_all[quantity])*10)/10 + 0.1)
        ylim = [-ylim_extreme, ylim_extreme]
    elif quantity == 'product':
        ylim = [-500, 500]
    elif quantity == 'c_ratio':
        ylim = [0, 4]
    elif quantity in ['c', 'c_star']:
        ylim = [0, 1]
    elif quantity == "Estar/E":
        # ylim = [0, np.ceil(max(_df_all[quantity])*10)/10 + 0.1]
        ylim = [0, 4]
    else:
        ylim = [-0.4, 0.4]
    
    _, ax = plt.subplots(5, 1, figsize=(6, 15))

    for n, (model_size, title) in enumerate(zip(model_sizes, titles)):
        ax[n].set_xscale("log")
        _ = ax[n].set_xticks(_lrs)
        _ = ax[n].set_xticklabels(['', r'$10^{-3}$', '', r'$10^{-2}$', '', r'$10^{-1}$', ''])
        
        _ = sns.boxplot(
            data=_df_all[_df_all['model_size'] == model_size], 
            x="lr", 
            y=quantity,
            color='k', 
            fill=True, 
            whis=(0, 100), 
            boxprops=dict(alpha=.3),
            native_scale=True,
            ax=ax[n],
        )
        if n == len(model_sizes) - 1:
            _ = ax[n].set_xlabel(r'$\eta$')
        else:
            _ = ax[n].set_xlabel('')
        _ = ax[n].set_ylim(ylim)
        _ = ax[n].set_ylabel(ylabel)
        _ = ax[n].set_title(title)

        if quantity in percentage:
            ax2 = ax[n].twinx()
            y = [i*ylim[-1]+(1-i)*ylim[0] for i in percentage_quantity[model_size]]
            _ = ax2.plot(_lrs, y, linestyle='--', marker='o', markerfacecolor='w', color=clr)
            if diverged is not None:
                _lrs_diverged = [elem for div, elem in zip(diverged[model_size], _lrs) if div is True]
                y_diverged = [elem for div, elem in zip(diverged[model_size], y) if div is True]
                _ = ax2.plot(_lrs_diverged, y_diverged, linestyle='--', marker='o', markerfacecolor=clr, color=clr)
    
            if quantity == 'B_ratio':
                _ = ax2.plot(_lrs, [1.0]*len(_lrs), linestyle=':', marker='', color=clr)
                ytext = 1.4
            elif quantity == 'lgt_mean':
                pass
            elif quantity == 'c_ratio':
                _ = ax2.plot(_lrs, [1.0]*len(_lrs), linestyle=':', marker='', color=clr)
            elif quantity in ['c', 'c_star']:
                _ = ax2.plot(_lrs, [0.1]*len(_lrs), linestyle=':', marker='', color=clr)
                ytext = 1.4
            elif quantity == 'Estar/E':
                _ = ax2.plot(_lrs, [1.0]*len(_lrs), linestyle=':', marker='', color=clr)
                ytext = 1.4
            else:
                _ = ax2.plot(_lrs, [0.0]*len(_lrs), linestyle=':', marker='', color=clr)
                ytext = 0.3 if n > 1 else -0.35
            if 0:
                _ = ax2.text(_lrs[-1]-0.1, ytext, fr'${percentage_quantity[model_size][-1]*100:.2f}\%$', color=clr)
            _ = ax2.set_ylim(ylim)
            _ = ax2.set_ylabel(ylabel2)
            _ = ax2.set_yticks([i*ylim[-1]+(1-i)*ylim[0] for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]])
            _ = ax2.set_yticklabels([r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$', r'$80\%$', r'$100\%$'])
            if n != len(model_sizes) - 1:
                _ = ax2.set_xticklabels(['']*7)
    
            ax2.yaxis.label.set_color(clr)
            ax2.spines["right"].set_edgecolor(clr)
            ax2.tick_params(axis='y', colors=clr)
        
    if len(save_as):
        assert save_as.endswith('.pdf') or save_as.endswith('.png')
        fig_path = f'figs/{save_as}'
        plt.savefig(fig_path, format=save_as[-3:], bbox_inches='tight')
        print(f'> saved as {fig_path}')
