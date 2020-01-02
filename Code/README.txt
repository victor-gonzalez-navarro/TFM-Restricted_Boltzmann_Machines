---------- MAIN CODE ------------
This folder contains the main code of the project. Inside this folder we have 8 different Mains, although the code is the same. The only difference between these Mains is that some lines are commented in each Main to be able to parallelize the code. That is,

- Main1.jl --> Contains the code when training the RBM with Contrastive Divergence with 1 Gibb step.
- Main2.jl --> Contains the code when training the RBM with Contrastive Divergence with NG (hyerparameter) Gibb steps.
- Main3.jl --> Contains the code when training the RBM with Weighted Contrastive Divergence with NG (hyerparameter) Gibb steps.
- Main4.jl --> Contains the code when training the RBM with Parallel Tempering with NG (hyerparameter) Gibb steps.
- Main5.jl --> Contains the code when training the RBM with Parallel Tempering with NG*K (hyerparameters) Gibb steps.
- Main6.jl --> Contains the code when training the RBM with Contrastive Divergence with NG*K (hyerparameters) Gibb steps.
- Main7.jl --> Contains the code when training the RBM with Weighted Contrastive Divergence with NG*K (hyerparameters) Gibb steps.
- Main8.jl --> Contains the code when training the RBM with all the instances of the Gibbs chain.

When running these Main files, we generate the Log-Likelihood for each epoch over small problems and the probabilities assigned to the testing instances after training the Restricted Boltzmann Machine.

-------- OTHER EXPERIMENTS --------
This folder contains the code to compare the performance of each algorithm described above for high-dimensional problems. Two sub-folders are included, one to test the sampling performance between Gibbs and Parallel Tempering, and the other one to test the learning performance between Gibbs and Parallel Tempering.

** NOTE ** There is a variable to set the path of the Main files. It is important to change this variable to the local path of each user.


