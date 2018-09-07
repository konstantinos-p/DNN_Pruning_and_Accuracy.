# DNN_Pruning_and_Accuracy.
We conduct experiments on pruning DNN layers and it's effect on network accuracy. We also explore the relationship betweeen the intrinsic dimensionality of the data and the network robustness to pruning.

create_model_cifar: trains a DNN on the cifar10 dataset and saves the corresponding saved model.

create_model_cifar_with_pca: trains a DNN on the cifar10 dataset after performing dimensionality reduction and saves the corresponding saved model.

plot_GE_pca: Plots the accuracy of the DNN for different levels of sparsity on the first convolutional layer. The plot is calculated using a single DNN instance.

plot_GE_pca_mean: Plots the accuracy of the DNN for different levels of sparsity on the first convolutional layer. The 

plot is calculated using multiple DNN instances and taking an average.

plot_single_GE: Plots accuracy for different levels of sparsity for individual DNN layers.

plot_mul_GE: Plots accuracy for different levels of sparsity across all DNN layers.

plot_mul_sum_GE: For different levels of sparsity across all DNN layers, plots the sum of the induced degradation for i>i_\* where i_\* \in \{1,..d\} where d is the number of DNN layers.

thresholding_cifar10: The main function used to take measurements.
utils: Includes the implementation of Hard Thresholding and other auxiliary functions.




