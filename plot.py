#import numpy as np
import matplotlib.pyplot as plt
from monty.serialization import loadfn, MontyDecoder,MontyEncoder
import os

plt.style.use("bmh")

def plot_results(json_data, lists=[0], figname='result.png', title=None):
# plot the results
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    ax1 = plt.subplot(211)
    for i in lists:
        data = json_data[i]
        eig_str = "{:d} {:6.2f}".format(data['eigenvalue'][0], data['eigenvalue'][1])
        ax1.plot(data['potential'][:, 0], data['wavefunction'], '--', label=eig_str)
    ax1.set_ylabel('$\Psi(x)$')
    ax1.legend(loc=2)
    plt.setp(ax1.get_xticklabels(), visible=False)


    ax2 = plt.subplot(212, sharex=ax1)
    for i in lists:
        data = json_data[i]
        ax2.hlines(y=data['eigenvalue'][1], xmin=data['potential'][0,0], xmax=data['potential'][-1,0], linewidth=1)
        ax2.plot(data['potential'][:, 0], data['potential'][:, 1], 'b-')
    ax2.set_ylabel('$V(x)$')
    ax2.set_xlabel('$x$')
    #ax2.set_ylim([0, 2*max(eigs)])
    plt.tight_layout()
    if title:
        plt.title(title)
    plt.savefig(figname)
    plt.close()

json_data = loadfn('trainingdata.json', cls=MontyDecoder)

folder = 'figs'
if not os.path.isdir(folder):
    os.mkdir(folder)
    
for i in range(len(json_data)):
    if i < 10:
        string = '00'+str(i)
    elif i<100:
        string = '0'+str(i)
    else:
        string = str(i)

    plot_results(json_data, [i], figname=folder+'/'+string+'.png', title=folder+'/'+string+'.png')
os.system('convert -loop 0 -delay 20 ' + folder + '/*.png out.gif')
