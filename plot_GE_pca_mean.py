import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from test_GE_theory_calculate_average_erros import calculate_average_errors

#Set Latex Usage
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#Load Empirical Values

layer350mean = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom350_mean_acc.npy')
layer350var = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom350_var_acc.npy')
layer340mean = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom340_mean_acc.npy')
layer340var = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom340_var_acc.npy')
layer300mean = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom300_mean_acc.npy')
layer300var = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom300_var_acc.npy')
layer250mean = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom250_mean_acc.npy')
layer250var = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom250_var_acc.npy')
layer200mean = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom200_mean_acc.npy')
layer200var = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom200_var_acc.npy')
layer50mean = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom50_mean_acc.npy')
layer50var = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom50_var_acc.npy')

spLevel = np.load('cifar_10_with_pca/results_with_averaging/pca_layer0_ncom350_sp.npy')
spLevel = spLevel*100

#Preprocessing
layer350 = 1-layer350mean
layer340 = 1 -layer340mean
layer300 = 1 -layer300mean
layer250 = 1 -layer250mean
layer200 = 1 -layer200mean
layer50 = 1 -layer50mean

#Centering?
centering = 1
if centering ==1:
    layer350 = layer350 - layer350[0]
    layer340 = layer340 - layer340[0]
    layer300 = layer300 - layer300[0]
    layer250 = layer250 - layer250[0]
    layer200 = layer200 - layer200[0]
    layer50  = layer50  - layer50[0]

#Plot First Figure
size_font = 12

fig2, ax2 = plt.subplots()
plt.ylabel('GE(g)',fontsize=size_font)
plt.xlabel('Sparsity \%',fontsize=size_font)

ax2.set_ylim(ymin=0, ymax=0.60)
ax2.set_xlim(xmin=40, xmax=100)

ax2.plot(spLevel,layer350,label = "layers $\geq$ 0",linestyle = '-',color = 'xkcd:violet')
ax2.fill_between(spLevel,layer350-layer350var,layer350+layer350var,interpolate=True,facecolor='xkcd:magenta',alpha=0.3)

#ax2.plot(spLevel,layer340,label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:red')
#ax2.fill_between(spLevel,layer340-layer340var,layer340+layer340var,interpolate=True,facecolor='xkcd:brick red',alpha=0.3)

ax2.plot(spLevel,layer300,label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:red')
ax2.fill_between(spLevel,layer300-layer300var,layer300+layer300var,interpolate=True,facecolor='xkcd:brick red',alpha=0.3)

ax2.plot(spLevel,layer250,label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:yellow')
ax2.fill_between(spLevel,layer250-layer250var,layer250+layer250var,interpolate=True,facecolor='xkcd:orange',alpha=0.3)

ax2.plot(spLevel,layer200,label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:blue')
ax2.fill_between(spLevel,layer200-layer200var,layer200+layer200var,interpolate=True,facecolor='xkcd:light blue',alpha=0.3)

ax2.plot(spLevel,layer50,label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:black')
ax2.fill_between(spLevel,layer50-layer50var,layer50+layer50var,interpolate=True,facecolor='xkcd:grey',alpha=0.3)



patch0 = mpatches.Patch(color = 'xkcd:violet', label = "350 PCA components")
patch1 = mpatches.Patch(color = 'xkcd:red', label = "300 PCA components")
patch2 = mpatches.Patch(color = 'xkcd:yellow', label = "250 PCA components")
patch3 = mpatches.Patch(color = 'xkcd:blue', label = "200 PCA components")
patch4 = mpatches.Patch(color = 'xkcd:black', label = " 50 PCA components")


plt.legend(loc = 2,handles=[patch0,patch1,patch2,patch3,patch4],fontsize=size_font)

plt.grid(linestyle=':')

end = 1