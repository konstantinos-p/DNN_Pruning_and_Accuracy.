# DNN_Pruning_and_Accuracy.
We conduct experiments on pruning DNN layers and it's effect on network accuracy. We also explore the relationship betweeen the intrinsic dimensionality of the data and the network robustness to pruning.

<B>create_model_cifa</B>r: trains a DNN on the cifar10 dataset and saves the corresponding saved model.

<B>create_model_cifar_with_pca</B>: trains a DNN on the cifar10 dataset after performing dimensionality reduction and saves the corresponding saved model.

<B>plot_GE_pca</B>: Plots the accuracy of the DNN for different levels of sparsity on the first convolutional layer. The plot is calculated using a single DNN instance.

<B>plot_GE_pca_mean</B>: Plots the accuracy of the DNN for different levels of sparsity on the first convolutional layer. The 

plot is calculated using multiple DNN instances and taking an average.

<B>plot_single_GE</B>: Plots accuracy for different levels of sparsity for individual DNN layers.

<B>plot_mul_GE</B>: Plots accuracy for different levels of sparsity across all DNN layers.

<B>plot_mul_sum_GE</B>: For different levels of sparsity across all DNN layers, plots the sum of the induced degradation for i>i\* where i\* in \{1,..d\} where d is the number of DNN layers.

<B>thresholding_cifar10</B>: The main function used to take measurements.

<B>utils</B>: Includes the implementation of Hard Thresholding and other auxiliary functions.




