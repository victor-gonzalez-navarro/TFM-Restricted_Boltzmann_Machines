clearconsole();

using JLD
using Random
using ProgressMeter
# Random.seed!(0);

## =================== Initialize parameters of the software ===================
abs_path    = "/Users/gonav/Documents/GitHub/TFM-Restricted_Boltzmann_Machines/Code/OtherExperiments/Learning/";

learningR   = 0.1;                                    # Learning rate
momentum    = 0.9;                                    # Momentum
Nneurons    = 700;                                    # Number of neurons in the hidden layer
Nepochs     = 5000000;                                # Number of epochs
NG          = 5;                                      # Number of Gibbs steps
batchsize   = 256;                                    # Batch size for Stochastic Gradient Ascent

NinstOut    = 10000;                                  # Number of instances outside the training set (for evaluation purposes)

K           = 6;                                      # Number of RBMs for Paralle Tempering
NG_par      = 1;                                      # Number of Gibbs steps for each RBM before swapping

dataset     = "MNIST";                                # Dataset used in the experiment: "BS16", "BS09", "BS02-3", "LSE11", "LSE15", "P08", "P10"
ReducedData = 2000;                                   # Number of instances for trainining (validation set are 1/3 and test is fixed)

plotLL      = 1;                                      # Plot LogLikelihood (just for small datasets)

## ======================== Import auxiliary functions =========================
include(abs_path * "functions.jl");

## ============================== Create dataset ===============================
I, I_real, full_targets = createDataset(dataset, ReducedData, abs_path);  # Input data: (#attr X #examples)

## ================= Create instances outside the training set =================
I_outside = zeros(size(I,1), NinstOut);
for i=1:size(I_outside,2)
    enter1 = true;
    vec = -ones(size(I_outside,1),1);
    while enter1
        enter1 = false
        vec = round.(rand(size(I_outside,1),1));
        j = 1;
        enter2 = true;
        while enter2 && j <= size(I,2)
            if vec == I[:,j]
                enter2 = false;
                enter1 = true;
            end
            j = j + 1;
        end
    end
    I_outside[:,i] = vec;
end

## ======================== Initialize parameters of RBM =======================
Nexamples = size(I,2);
println("Training RBM with the following hyperparameters:")
println("Epsilon = " * string(eps) * " | Number of hidden neurons = " * string(Nneurons) * " | Number of epochs = " * string(Nepochs) * " | Number of Gibbs steps = " * string(NG) * " | Batchsize = " * string(batchsize))

W = (2*rand(Nneurons,size(I,1)).-1)*4*sqrt(3.0/16.0);    # Initial weights (#h_units X #v_units)
b = zeros(size(I,1),1);                                  # Initial bias input neurons
c = zeros(Nneurons,1);                                   # Initial bias hidden neurons

## =============================== Train the RBM ===============================
# Compute target distribution
targdis = ones(size(I,2), 1)./size(I,2);

# Compute RBM parameters
param, plotparams = computeRBMparamAll(W, b, c, I, I_outside, Nexamples, batchsize, NG, Nepochs, momentum, targdis, plotLL);

## =============================== Print Results ===============================
println("ratio_CD = "   * string(plotparams["ratio_CD"])   * ";");
println("ratio_PTCD = " * string(plotparams["ratio_PTCD"]) * ";");
