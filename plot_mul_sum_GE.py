import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from test_GE_theory_calculate_average_erros import calculate_average_errors

#Set Latex Usage
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#Load Empirical Values

layers = np.zeros((20,8))

layers[:,0] = np.reshape(np.load('cifar10/results/layer1_ac.npy'),(-1))
layers[:,1] = np.reshape(np.load('cifar10/results/layer2_ac.npy'),(-1))
layers[:,2] = np.reshape(np.load('cifar10/results/layer3_ac.npy'),(-1))
layers[:,3] = np.reshape(np.load('cifar10/results/layer4_ac.npy'),(-1))
layers[:,4] = np.reshape(np.load('cifar10/results/layer5_ac.npy'),(-1))
layers[:,5] = np.reshape(np.load('cifar10/results/layer6_ac.npy'),(-1))
layers[:,6] = np.reshape(np.load('cifar10/results/layer7_ac.npy'),(-1))
layers[:,7] = np.reshape(np.load('cifar10/results/layer8_ac.npy'),(-1))

layers = layers/100
layers = 1-layers
layers = layers-0.2

for i in range(0,8):
    layers[:,i] = np.sum(layers[:,i:],axis=1)
layers = layers+0.2

spLevel = np.load('cifar10/results/layer1_sp.npy')

#Plot First Figure
size_font = 12

fig2, ax2 = plt.subplots()
plt.ylabel('GE(g)',fontsize=size_font)
plt.xlabel('Sparsity \%',fontsize=size_font)

ax2.set_ylim(ymin=0, ymax=1.05)
ax2.set_xlim(xmin=0, xmax=100)

ax2.plot(spLevel,layers[:,0],label = "layers $\geq$ 0",linestyle = '-',color = 'xkcd:violet')
ax2.plot(spLevel,layers[:,1],label = "layers $\geq$ 1",linestyle = '-',color = 'xkcd:bright blue')
ax2.plot(spLevel,layers[:,2],label = "layers $\geq$ 2",linestyle = '-',color = 'xkcd:orange')
ax2.plot(spLevel,layers[:,3],label = "layers $\geq$ 3",linestyle = '-',color = 'xkcd:red')

ax2.plot(spLevel,layers[:,4],label = "layers $\geq$ 4",linestyle = '-',color = 'xkcd:blue')
ax2.plot(spLevel,layers[:,5],label = "layers $\geq$ 5",linestyle = '-',color = 'xkcd:yellow')
ax2.plot(spLevel,layers[:,6],label = "layers $\geq$ 6",linestyle = '-',color = 'xkcd:green')
ax2.plot(spLevel,layers[:,7],label = "layers $\geq$ 7",linestyle = '-',color = 'xkcd:purple')

patch0 = mpatches.Patch(color = 'xkcd:violet', label = "Layers $=$ 0")
patch1 = mpatches.Patch(color = 'xkcd:bright blue', label = "Layers $=$ 1")
patch2 = mpatches.Patch(color = 'xkcd:orange', label = "Layers $=$ 2")
patch3 = mpatches.Patch(color = 'xkcd:red', label = "Layers $=$ 3")

patch4 = mpatches.Patch(color = 'xkcd:blue', label = "Layers $=$ 4")
patch5 = mpatches.Patch(color = 'xkcd:yellow', label = "Layers $=$ 5")
patch6 = mpatches.Patch(color = 'xkcd:green', label = "Layers $=$ 6")
patch7 = mpatches.Patch(color = 'xkcd:purple', label = "Layers $=$ 7")

plt.legend(loc = 2,handles=[patch0,patch1,patch2,patch3,patch4,patch5,patch6,patch7],fontsize=size_font)

plt.grid(linestyle=':')

end = 1