clearconsole();

using Random
using Plots
using ProgressMeter
Random.seed!(0);

## =================== Initialize parameters of the code =======================
abs_path    = "/Users/gonav/Documents/GitHub/TFM-Restricted_Boltzmann_Machines/Code/EstimationPartitionFunction/";

Ninput      = 9;                                      # Number of neurons in the input layer
Nneurons    = 20;                                     # Number of neurons in the hidden layer

K           = 6;                                      # Number of RBMs for Paralle Tempering
NG_par      = 1;                                      # Number of Gibbs steps for each RBM before swapping

M           = 100;                                    # Number of samples to estimate Za/Zb
numInt      = 1000;                                   # Number of intermediate distributions for estimating Za/Zb counting first and last
NG2         = 10;                                     # Number of steps for running the Markov Chain for each intermediate distribution

estimateZ   = 1;                                      # Estimate partition function of the RBM by Annealed Importance Sampling

## ======================== Import auxiliary functions =========================
include(abs_path * "functions.jl");

## ======================== Initialize parameters of RBM =======================
W = (2*rand(Nneurons,Ninput).-1)*4*sqrt(3.0/16.0);    # Initial weights (#h_units X #v_units)
b = rand(Ninput,1);                                   # Initial bias input neurons
c = rand(Nneurons,1);                                 # Initial bias hidden neurons

## =============================== Compute Real Z ==============================
Z = computeRealZ(Nneurons,W,b,c,Ninput);
println("The ground truth (log) partition function of this RBM is: " * string(log(Z)));

## ======== Estimation of the partition function for the generated RBM =========
if estimateZ==1
  # Initialize base rate RBM for the estimation of the partition function
  WA = zeros(size(W,1),size(W,2));
  bA = b;
  cA = c;

  # Ensure that p(vA)>0 for all v in V
  probvec = sigmoid(b);
  c_assert(sum(probvec.==0) == 0, "Error, we have not ensured that p(vA)>0 for all v in V");

  # Computation of the estimation of the partition function
  Zb = estimate_Zb(M,numInt,NG2,WA,bA,cA,W,b,c);
  println("The estimated (log) partition function of this RBM is " * string(log(Zb)));
end
