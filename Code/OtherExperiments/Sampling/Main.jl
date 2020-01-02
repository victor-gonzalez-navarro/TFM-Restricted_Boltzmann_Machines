# My goal is to determine which is the best for sampling: Gibbs or PT
# What I do in this experiment is the following:
# 1) I initialize randomly the parameyters (using Rows to estimate afterwards the partition function)
# 2) I generate sample by Gibbs and by PT using an infinite chain (length NG) and keeping 1 sample every 5
# 3) I compute the probability distribution of the samples (for both Gibbs and PT)
# 4) I compute the real probability distribution when only using the samples obtained in the previous step


using Random
using Plots
using ProgressMeter
using Distances
# using StatsPlots
using JLD
Random.seed!(0);

## =================== Initialize parameters of the software ===================
abs_path    = "/scratch/nas/4/victorg/JuliaUp/Exp7/A/Code/";

NG          = 500# 5000000;                                # Number of Gibbs steps
Npoints     = 10# 1000;                                   # Number of points for the plot
Nneurons    = 150;                                    # Number of hidden units
Nvisible    = 36;                                     # Number of visible units
Per         = 0.01;                                   # Percentage of states with high probability to compare (out of 2^positive_elements)

K           = 10;                                     # Number of RBMs for Paralle Tempering
NG_par      = 1;                                      # Number of Gibbs steps for each RBM before swapping

## ======================== Import auxiliary functions =========================
include(abs_path*"functions.jl");

## ============================== Main Experiment ==============================
NGvec = range(5,stop=NG/5,length=Npoints); # Number of samples used to compute the distances
NGvec = round.(Int, NGvec);

Nexp = length(NGvec);
typeUnits = 0;
Nv = Nvisible;
Nh = Nneurons;

c_assert(maximum(NGvec) <= NG/5, "The maximum of NGvec cannot be greater then NG/5")

## ======================== Initialize parameters of RBM =======================
Ak = ones(Nv,1);      ### Vector of alphas (each row of W will have this value)
b  = 0.0*randn(Nv);  ### Bias of the visible units (before 20*)
c  = 1.0;    ;        ### Bias of the hidden units  *** MUST BE CONSTANT ***

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
            Ak[kv] = sum(Ak[kv+1:endpos])+0.000000000001;
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
# Hinton RBM
samples_gibbsRBM = gibbsRBM(init_random, W, b, c, NG);
# Hinton PT
samples_gibbsPT  = gibbsPT(init_random, W, b, c, NG, K, NG_par);
# Save samples of both RBM-Gibbs and PT
save(abs_path * "Samples.jld", "RBM", samples_gibbsRBM, "PT", samples_gibbsPT)

## ================ Computation states with highest probability ================
perm = reverse(sortperm(vec(Ak)));  # Argsort of Ak
poselements = sum(Ak[perm] .> 0);  # Number of positive elements in Ak
T = round(Int, 2^(poselements)*Per);  # Number of states with highest probability
statesHighProb = zeros(T,Nv);
for i=1:T
    vec = de2bi(2^(poselements)-i, poselements)';
    statesHighProb[i,1:poselements] = copy(vec);
    statesHighProb[i,:] = copy(statesHighProb[i,sortperm(perm)])
end

println(Ak);
println("--------------------");

## ============================= Compute Distances =============================
distancesKLRBM  = zeros(1, Nexp);  # Distances corresponding to the KL divergence using empirical samples
distancesKLPT   = zeros(1, Nexp);  # Distances corresponding to the KL divergence using empirical samples
logLikelihRBM   = zeros(1, Nexp);  # LogLikelihood of obtained samples using target distribution
logLikelihPT    = zeros(1, Nexp);  # LogLikelihood of obtained samples using target distribution
sumprobabiRBM   = zeros(1, Nexp);  # Sum of probabilities of obtained samples using the target distribution
sumprobabiPT    = zeros(1, Nexp);  # Sum of probabilities of obtained samples using the target distribution
distancesKL2RBM = zeros(1, Nexp);  # Distances corresponding to the KL divergence using top probability samples
distancesKL2PT  = zeros(1, Nexp);  # Distances corresponding to the KL divergence using top probability samples

@showprogress 1 "Computing measures ..." for nexp=1:Nexp
    # Compare probability distributions (target with experimental) using empirical samples
    targetprobRBM, experiprobRBM = analysis(samples_gibbsRBM[:,1:NGvec[nexp]], W, b, c, logZ);
    targetprobPT, experiprobPT = analysis(samples_gibbsPT[:,1:NGvec[nexp]], W, b, c, logZ);
    distancesKLRBM[1,nexp] = kl_divergence(targetprobRBM, experiprobRBM);
    distancesKLPT[1,nexp] = kl_divergence(targetprobPT, experiprobPT);

    # Compute the LogLikelihood and sum of probabilities of obtained samples using target distribution
    logLikelihRBM[1,nexp],sumprobabiRBM[1,nexp] = analysis2(samples_gibbsRBM[:,1:NGvec[nexp]], W, b, c, logZ);
    logLikelihPT[1,nexp],sumprobabiPT[1,nexp] = analysis2(samples_gibbsPT[:,1:NGvec[nexp]], W, b, c, logZ);

    # Compare probability distributions (target with experimental) using top probability samples
    targetprob, experipro2bRBM, experiprob2PT = analysis3(statesHighProb', samples_gibbsRBM[:,1:NGvec[nexp]], samples_gibbsPT[:,1:NGvec[nexp]], W, b, c, logZ);
    distancesKL2RBM[1,nexp] = kl_divergence(targetprob, experipro2bRBM);
    distancesKL2PT[1,nexp] = kl_divergence(targetprob, experiprob2PT);
end

## ================================ Plotting ===================================
p1 = plot([NGvec, NGvec],[distancesKLRBM', distancesKLPT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="KL Divergence",leg=false);
p2 = plot([NGvec, NGvec],[logLikelihRBM', logLikelihPT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="Log-Likelihood",leg=true, label=["Gibbs", "PT"]);
p3 = plot([NGvec, NGvec],[sumprobabiRBM', sumprobabiPT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="Sum of probabilities",leg=false);
p4 = plot([NGvec, NGvec],[distancesKL2RBM', distancesKL2PT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="KL Divergence v2",leg=false);
p = plot(p1,p2,p3,p4, layout=(2,2), dpi=500);
savefig(p, abs_path*"Plots/plot6.png");

# print("distancesKLRBM = ");
# println(distancesKLRBM);
# print("distancesKLPT = ");
# println(distancesKLPT);

# print("logLikelihRBM = ");
# println(logLikelihRBM);
# print("logLikelihPT = ");
# println(logLikelihPT);

# print("sumprobabiRBM = ");
# println(sumprobabiRBM);
# print("sumprobabiPT = ");
# println(sumprobabiPT);

# print("distancesKL2RBM = ");
# println(distancesKL2RBM);
# print("distancesKL2PT = ");
# println(distancesKL2PT);
