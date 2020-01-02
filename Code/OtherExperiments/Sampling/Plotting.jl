abs_path    = "/Users/gonav/Desktop/Julia/CodeForCloud/Exp7/A/Plots/";

p1 = plot([NGvec, NGvec],[distancesKLRBM', distancesKLPT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="KL Divergence",leg=false);
p2 = plot([NGvec, NGvec],[logLikelihRBM', logLikelihPT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="Log-Likelihood",leg=true, label=["Gibbs", "PT"]);
p3 = plot([NGvec, NGvec],[sumprobabiRBM', sumprobabiPT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="Sum of probabilities",leg=false);
p4 = plot([NGvec, NGvec],[distancesKL2RBM', distancesKL2PT'], linewidth=1.5,lc=[:blue :orange],xlabel="Number of samples",ylabel="KL Divergence v2",leg=false);
p = plot(p1,p2,p3,p4, layout=(2,2), dpi=500);
savefig(p, abs_path*"plot.png");
