clearconsole();

using Random
using ProgressMeter
Random.seed!(0);

## =================== Initialize parameters of the software ===================
abs_path    = "/Users/gonav/Documents/GitHub/TFM-Restricted_Boltzmann_Machines/Code/MainCode/";

learningR   = 0.1;                                     # Learning rate
momentum    = 0.9;                                     # Momentum
Nneurons    = 64;                                      # Number of neurons in the hidden layer
Nepochs     = 1000000;                                 # Number of epochs
NG          = 5;                                       # Number of Gibbs steps
batchsize   = 30;                                      # Batch size for Stochastic Gradient Ascent

K           = 6;                                       # Number of RBMs for Paralle Tempering
NG_par      = 1;                                       # Number of Gibbs steps for each RBM before swapping

M           = 100;                                     # Number of samples to estimate Za/Zb
numInt      = 1000;                                    # Number of intermediate distributions for estimating Za/Zb counting first and last
NG2         = 10;                                      # Number of steps for running the Markov Chain for each intermediate distribution

dataset     = "BS16";                                  # Dataset used in the experiment: "BS16", "BS09", "BS02-3", "LSE11", "LSE15", "P08", "P10"
distr       = "empirical";                             # Type of distribution of the dataset
plotLL      = 1;                                       # Plot LogLikelihood (just for small datasets)

## ======================== Import auxiliary functions =========================
include(abs_path * "functions.jl");

## ============================== Create dataset ===============================
I, I_real, full_targets = createDataset(dataset, abs_path);  # Input data: (#attr X #examples)

## ======================== Initialize parameters of RBM =======================
Nexamples = size(I,2);
println("Training RBM with the following hyperparameters:")
println("Epsilon = " * string(eps) * " | Number of hidden neurons = " * string(Nneurons) * " | Number of epochs = " * string(Nepochs) * " | Number of Gibbs steps = " * string(NG) * " | Batchsize = " * string(batchsize))

W = (2*rand(Nneurons,size(I,1)).-1)*4*sqrt(3.0/16.0);  # Initial weights (#h_units X #v_units)
b = zeros(size(I,1),1);                                # Initial bias input neurons
c = zeros(Nneurons,1);                                 # Initial bias hidden neurons

## =============================== Train the RBM ===============================
if plotLL==1
  # Compute LL with initial parameters
  LLvec = zeros(1,Nepochs+1);
  LL,Z = computeLLNP(I,Nneurons,W,b,c);
  LLvec[1,1] = LL;
end

## Compute target distribution
targdis = ones(size(I,2), 1)./size(I,2);

# Compute RBM parameters
param, plotparams = computeRBMparamAll(W, b, c, LLvec, Z, I, Nexamples, batchsize, NG, Nepochs, momentum, targdis, plotLL);

## =============================== Print Results ===============================
println("expdis_CD1 = "   * string(plotparams["expdis_CD1"])   * ";");
println("expdis_CD = "    * string(plotparams["expdis_CD"])    * ";");
println("expdis_WCD = "   * string(plotparams["expdis_WCD"])   * ";");
println("expdis_PTCD = "  * string(plotparams["expdis_PTCD"])  * ";");
println("expdis_PTWCD = " * string(plotparams["expdis_PTWCD"]) * ";");
println("expdis_CDP = "   * string(plotparams["expdis_CDP"])   * ";");
println("expdis_WCDP = "  * string(plotparams["expdis_WCDP"])  * ";");
println("expdis_FIN = "   * string(plotparams["expdis_FIN"])   * ";");

println("LLvec_CD1 = "    * string(plotparams["LLvec_CD1"])    * ";");
println("LLvec_CD = "     * string(plotparams["LLvec_CD"])     * ";");
println("LLvec_WCD = "    * string(plotparams["LLvec_WCD"])    * ";");
println("LLvec_PTCD = "   * string(plotparams["LLvec_PTCD"])   * ";");
println("LLvec_PTWCD = "  * string(plotparams["LLvec_PTWCD"])  * ";");
println("LLvec_CDP = "    * string(plotparams["LLvec_CDP"])    * ";");
println("LLvec_WCDP = "   * string(plotparams["LLvec_WCDP"])   * ";");
println("LLvec_FIN = "    * string(plotparams["LLvec_FIN"])    * ";");
