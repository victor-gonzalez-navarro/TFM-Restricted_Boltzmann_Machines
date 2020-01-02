
# using JLD
using Random
using ProgressMeter
Random.seed!(0);

## =================== Initialize parameters of the software ===================
abs_path    = "/scratch/nas/4/victorg/JuliaUp/ExpFinal/Code/";

learningR   = 0.1;                                         # Learning rate
momentum    = 0.9;                                         # Momentum
Nneurons    = 64;                                          # Number of neurons in the hidden layer
Nepochs     = 1000000;                                     # Number of epochs
NG          = 5;                                           # Number of Gibbs steps
batchsize   = 30;                                          # Batch size for Stochastic Gradient Ascent

K           = 6;                                           # Number of RBMs for Paralle Tempering
NG_par      = 1;                                           # Number of Gibbs steps for each RBM before swapping

M           = 100;      # 100;                             # Number of samples to estimate Za/Zb
numInt      = 1000;     # 14500;                           # Number of intermediate distributions for estimating Za/Zb counting first and last
NG2         = 10;       # 10000;                           # Number of steps for running the Markov Chain for each intermediate distribution

dataset     = "BS16";  # "MNIST";                          # Dataset used in the experiment: "BS16", "BS09", "BS02-3", "LSE11", "LSE15", "P08", "P10"
ReducedData = 1000;                                        # Number of instances for trainining (validation set are 1/3 and test is fixed)

distr       = "empirical";                                 # Type of distribution of the dataset

plotLL      = 1;                                           # Plot LogLikelihood (just for small datasets)
estimateZ   = 0;                                           # Estimate partition function of the RBM by Annealed Importance Sampling

## ======================== Import auxiliary functions =========================
include(abs_path * "functions3.jl");

## ============================== Create dataset ===============================
I, I_real, full_targets = createDataset(dataset, ReducedData, abs_path);  # Input data: (#attr X #examples)

## ======================== Initialize parameters of RBM =======================
Nexamples = size(I,2);
println("Training RBM with the following hyperparameters:")
println("Epsilon = " * string(eps) * " | Number of hidden neurons = " * string(Nneurons) * " | Number of epochs = " * string(Nepochs) * " | Number of Gibbs steps = " * string(NG) * " | Batchsize = " * string(batchsize))

W = (2*rand(Nneurons,size(I,1)).-1)*4*sqrt(3.0/16.0);    # Initial weights (#h_units X #v_units)
b = zeros(size(I,1),1);                                  # Initial bias input neurons
c = zeros(Nneurons,1);                                   # Initial bias hidden neurons

## =============================== Train the RBM ===============================
# We have implemented W NG-step Constrastive Divergence

if plotLL==1
  # Compute LL with initial parameters
  LLvec = zeros(1,Nepochs+1);
  LL,Z = computeLLNP(I,Nneurons,W,b,c);
  LLvec[1,1] = LL;
end

## Compute target distribution
if distr == "empirical"
    targdis = ones(size(I,2), 1)./size(I,2);
elseif distr == "gaussian"
elseif distr == "descrete"
end

# Compute RBM parameters
param, plotparams = computeRBMparamAll(W, b, c, LLvec, Z, I, Nexamples, batchsize, NG, Nepochs, momentum, targdis, plotLL);
# param, plotparams = computeRBMparamAllDel(W, b, c, LLvec, Z, I, Nexamples, batchsize, NG, Nepochs, momentum, targdis, plotLL);

## ========= Estimation of the partition function for the obtained RBM =========
if estimateZ==1
  W = params["W"]; b = params["b"]; c = params["c"]; Z = params["Z"];
  println("The ground truth (log) partition function of this RBM is: " * string(log(Z)));

  # Initialize base rate RBM for the estimation of the partition function
  WA = zeros(size(W,1),size(W,2));
  bA = b;
  cA = c;

  # Ensure that p(vA)>0 for all v in V
  probvec = sigmoid(b);  # Equation below 18 paper: On the Quantitative Analysis of Deep Belief Networks
  c_assert(sum(probvec.==0) == 0, "Error, we have not ensured that p(vA)>0 for all v in V");

  # Computation of the estimation of the partition function
  Zb = estimate_Zb(M,numInt,NG2,WA,bA,cA,W,b,c);
  println("The estimated (log) partition function of this RBM is " * string(log(Zb)));
end


# Save variables to perform the Plotting
save_string = "info = LR"*string(learningR)*"MO"*string(momentum)*"NN"*string(Nneurons)*"EP"*string(Nepochs)*"NG"*string(NG)*"BA"*string(batchsize)*"K"*string(K)*"NGP"*string(NG_par)*"M"*string(M)*"NI"*string(numInt)*"NG2"*string(NG2)*"DA"*string(dataset)*"RD"*string(ReducedData)*"DS"*string(distr);
println(save_string * ";");

println("targdis = " * string(targdis) * ";");
# save("/scratch/nas/4/victorg/JuliaUp/Output/"*save_string*".jld", "plotparams", plotparams, "targdis", targdis);

# println("expdis_CD1 = "    * string(plotparams["expdis_CD1"])    * ";");
# println("expdis_CD = "    * string(plotparams["expdis_CD"])    * ";");
println("expdis_WCD = "   * string(plotparams["expdis_WCD"])   * ";");
# println("expdis_PTCD = "  * string(plotparams["expdis_PTCD"])  * ";");
# println("expdis_PTWCD = " * string(plotparams["expdis_PTWCD"]) * ";");
# println("expdis_CDP = "   * string(plotparams["expdis_CDP"])   * ";");
# println("expdis_WCDP = "  * string(plotparams["expdis_WCDP"])  * ";");
# println("expdis_FIN = "   * string(plotparams["expdis_FIN"])   * ";");

# println("LLvec_CD1 = "     * string(plotparams["LLvec_CD1"])     * ";");
# println("LLvec_CD = "     * string(plotparams["LLvec_CD"])     * ";");
println("LLvec_WCD = "    * string(plotparams["LLvec_WCD"])    * ";");
# println("LLvec_PTCD = "   * string(plotparams["LLvec_PTCD"])   * ";");
# println("LLvec_PTWCD = "  * string(plotparams["LLvec_PTWCD"])  * ";");
# println("LLvec_CDP = "    * string(plotparams["LLvec_CDP"])    * ";");
# println("LLvec_WCDP = "   * string(plotparams["LLvec_WCDP"])   * ";");
# println("LLvec_FIN = "    * string(plotparams["LLvec_FIN"])    * ";");
