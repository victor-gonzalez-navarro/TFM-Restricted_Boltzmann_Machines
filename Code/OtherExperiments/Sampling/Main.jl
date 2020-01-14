clearconsole();

using Random
using Plots
using ProgressMeter
using Distances
using JLD

## =================== Initialize parameters of the software ===================
abs_path    = "/Users/gonav/Documents/GitHub/TFM-Restricted_Boltzmann_Machines/Code/OtherExperiments/Sampling/";

NG          = 5000000;                                # Number of Gibbs steps
Nneurons    = 150;                                    # Number of hidden units
Nvisible    = 36;                                     # Number of visible units
Per         = 0.01;                                   # Percentage of states with high probability to compare (out of 2^positive_elements)

K           = 10;                                     # Number of RBMs for Paralle Tempering
NG_par      = 5;                                      # Number of Gibbs steps for each RBM before swapping

## ======================== Import auxiliary functions =========================
include(abs_path*"functions.jl");

## ======================== Initialize parameters of RBM =======================
typeUnits = 0;
Nv = Nvisible;
Nh = Nneurons;

Ak = ones(Nv,1);      # Vector of alphas (each row of W will have this value)
b  = 0.0*randn(Nv);   # Bias of the visible units (before 20*)
c  = 1.0;             # Bias of the hidden units  *** MUST BE CONSTANT ***

first = true;
endpos = 0;
for kv in Nv:-1:1
    global first;
    global endpos;
    if kv < Nv/2
        if first
            Ak[kv] = 0.0001;
            first = false;
            endpos = kv;
        else
            Ak[kv] = sum(Ak[kv+1:endpos])+1e-12;
        end
    else
        Ak[kv] = -abs(0.2*randn())*Ak[kv];
    end
end;
Ak = copy(Ak[randperm(length(Ak))]);

Wtranspose, logZ = generateWeightsRows(typeUnits,Nv,Nh,Ak,b,c);
W = copy(Wtranspose[2:end,2:end]');
b = copy(reshape(b,length(b),1));
c = copy(c*ones(Nh,1));

## ============================= Generate Samples ==============================
init_random = round.(rand(Nvisible,1));
samples_gibbsRBM = gibbsRBM(init_random, W, b, c, NG);  # Generate samples with Gibbs samppling
samples_gibbsPT  = gibbsPT(init_random, W, b, c, NG, K, NG_par);  # Generate samples with Parallel Tempering

## ================ Computation states with highest probability ================
perm = reverse(sortperm(vec(Ak)));    # Argsort of Ak
poselements = sum(Ak[perm] .> 0);     # Number of positive elements in Ak
T = round(Int, 2^(poselements)*Per);  # Number of states with highest probability
statesHighProb = zeros(T,Nv);
for i=1:T
    vec = de2bi(2^(poselements)-i, poselements)';
    statesHighProb[i,1:poselements] = copy(vec);
    statesHighProb[i,:] = copy(statesHighProb[i,sortperm(perm)])
end

## ============================= Compute Distances =============================
# Compare probability distributions (model with experimental) using empirical samples
targetprobRBM, experiprobRBM = analysis(samples_gibbsRBM, W, b, c, logZ);
targetprobPT, experiprobPT = analysis(samples_gibbsPT, W, b, c, logZ);
distancesKLRBM = kl_divergence(targetprobRBM, experiprobRBM);
distancesKLPT  = kl_divergence(targetprobPT, experiprobPT);

# Compute the LogLikelihood and sum of probabilities of obtained samples using model distribution
logLikelihRBM,sumprobabiRBM = analysis2(samples_gibbsRBM, W, b, c, logZ);
logLikelihPT,sumprobabiPT   = analysis2(samples_gibbsPT, W, b, c, logZ);

# Compare probability distributions (model with experimental) using top probability samples
targetprob, experiprobRBM2, experiprobPT2 = analysis3(statesHighProb', samples_gibbsRBM, samples_gibbsPT, W, b, c, logZ);
distancesKL2RBM = kl_divergence(targetprob, experiprobRBM2);
distancesKL2PT  = kl_divergence(targetprob, experiprobPT2);

## =========================== Print Results ===================================
# Distance corresponding to the KL divergence using empirical samples
println("distancesKLRBM = " * string(distancesKLRBM));
println("distancesKLPT = " * string(distancesKLPT));

# LogLikelihood of obtained samples using the model distribution
println("logLikelihRBM = " * string(logLikelihRBM));
println("logLikelihPT = " * string(logLikelihPT));

# Sum of probabilities of obtained samples using the model distribution
println("sumprobabiRBM = " * string(sumprobabiRBM));
println("sumprobabiPT = " * string(sumprobabiPT));

# Distance corresponding to the KL divergence using top probability samples
println("distancesKL2RBM = " * string(distancesKL2RBM));
println("distancesKL2PT = " * string(distancesKL2PT));
