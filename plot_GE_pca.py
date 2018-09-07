import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from test_GE_theory_calculate_average_erros import calculate_average_errors

#Set Latex Usage
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#Load Empirical Values

layer300 = np.load('cifar_10_with_pca/results/pca_layer0_ncom300_acc.npy')
layer310 = np.load('cifar_10_with_pca/results/pca_layer0_ncom310_acc.npy')
layer320 = np.load('cifar_10_with_pca/results/pca_layer0_ncom320_acc.npy')
layer330 = np.load('cifar_10_with_pca/results/pca_layer0_ncom330_acc.npy')
layer340 = np.load('cifar_10_with_pca/results/pca_layer0_ncom340_acc.npy')
layer350 = np.load('cifar_10_with_pca/results/pca_layer0_ncom350_acc.npy')

spLevel = np.load('cifar_10_with_pca/results/pca_layer0_ncom350_sp.npy')
spLevel = spLevel*100

#Plot First Figure
size_font = 12

fig2, ax2 = plt.subplots()
plt.ylabel('GE(g)',fontsize=size_font)
plt.xlabel('Sparsity \%',fontsize=size_font)

ax2.set_ylim(ymin=0, ymax=1.05)
ax2.set_xlim(xmin=0, xmax=100)

ax2.plot(spLevel,1 - layer300,label = "layers $\geq$ 0",linestyle = '-',color = 'xkcd:violet')
ax2.plot(spLevel,1 - layer310,label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:red')
ax2.plot(spLevel,1 - layer320,label = "layers $\geq$ 2",linestyle = '-',color = 'xkcd:orange')
ax2.plot(spLevel,1 - layer330,label = "layers $\geq$ 3",linestyle = '-',color = 'xkcd:yellow')
ax2.plot(spLevel,1 - layer340,label = "layers $\geq$ 4",linestyle = '-',color = 'xkcd:bright blue')
ax2.plot(spLevel,1 - layer350,label = "layers $\geq$ 4",linestyle = '-',color = 'xkcd:green')


patch0 = mpatches.Patch(color = 'xkcd:violet', label = "Layers $=$ 0")

plt.legend(loc = 2,handles=[patch0],fontsize=size_font)

plt.grid(linestyle=':')

end = 1