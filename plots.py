import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import argparse

#####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    default='.',
    help="path"
)
parser.add_argument(
    '--base',
    default=False,
    help="Pretrained tasks (base or not-base)",
    action='store_true'
)
parser.add_argument(
    '--task',
    type=int,
    default=0,
    help="Test task"
)
args = parser.parse_args()
#####################################################################################

def plotdata(data, name):
    s = 20
    rc_ = {'figure.figsize':(10,6),'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 20}
    sns.set(rc=rc_, style="darkgrid")
    # rc('text', usetex=True)
    
    fig, ax = plt.subplots()
    
    lw = 2.0
    task = np.arange(data[0][0].shape[0])
    for (mean, std, label) in data:
        ax.plot(task, mean,  label=label, lw = lw)
        ax.fill_between(task, mean - std, mean + std, alpha=0.4)
    
    ax.legend()
    plt.xlabel("Episodes")
    plt.ylabel('Returns')
    plt.ylim(top=10)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    fig.savefig("{}/{}.pdf".format(args.path,name), bbox_inches='tight')
    plt.show()

def process_data(alldata):
    pdata = []
    m = 100
    s = 0.5
    o = alldata[0][0].shape[0]
    for (data, label) in alldata:
        # for i in range(o):
        #     data[i] = np.convolve(data[i], np.ones(m)/m, mode='same')

        mean = data.mean(axis=0)
        std = data.std(axis=0)*s

        mean = np.convolve(mean, np.ones(m)/m, mode='valid')
        std = np.convolve(std, np.ones(m)/m, mode='valid')

        pdata.append([mean, std, label])

    return pdata

#####################################################################################

if __name__ == '__main__': 
    name = "returns"
    returns = []

    algos = ["Ql","Ql_WVF"]
    for algo in algos:
        data = []
        for run in range(30): 
            data_path = "data/algo_{}.run_{}.npy".format(algo,run)
            if os.path.exists(data_path):
                log = np.load(data_path, allow_pickle=True).tolist()
                data.append(log)
            else:
                break
        data = np.array(data)
        if algo == "Ql":
            algo = "Value function (Q-learning)"
        if algo == "Dyna_Ql":
            algo = "Value function (Dyna)"
        if algo == "Ql_WVF":
            algo = "WVF (Q-learning)"
        if algo == "Dyna_Ql_WVF":
            algo = "WVF (Dyna)"
        returns.append((data, algo))
        print("LOADED: ", data.shape)
    
    plotdata(process_data(returns), name)
