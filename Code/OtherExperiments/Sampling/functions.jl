using Match
using ProgressMeter

function sigmoid(z)
    out = 1.0 ./ (1.0 .+ exp.(-z));
end;

function gibbsRBM(x, W, b, c, NG)
    samples_return = zeros(size(x,1),trunc(Int, NG/5));
    count = 1;
    for i = 1:NG
        # Find hidden units by sampling the visible layer
        lgst_c_xi = sigmoid(c .+ W* x);                  # lgst_c_x = (#h_units X batch_size)
        rand_vector = rand(size(lgst_c_xi,1),size(lgst_c_xi,2));
        h = rand_vector .<= lgst_c_xi;

        # Find visible units by sampling the hidden layer
        lgst_b_hi = sigmoid(b .+ W'*h);                 # lgst_b_h1 = (#v_units X batch_size)
        rand_vector = rand(size(lgst_b_hi,1),size(lgst_b_hi,2));
        x = rand_vector .<= lgst_b_hi;

        # Save samples
        if i % 5 == 0
            samples_return[:,count] = copy(x);
            count = count + 1;
        end
    end
    return samples_return
end;

function gibbsRBMv2(x, W, b, c, NG)
    # Modified version of GibbsRBM to save all the samples from the chain
    samples_return = zeros(size(x,1),1);
    count = 1;
    for i = 1:NG
        # Find hidden units by sampling the visible layer
        lgst_c_xi = sigmoid(c .+ W* x);                  # lgst_c_x = (#h_units X batch_size)
        rand_vector = rand(size(lgst_c_xi,1),size(lgst_c_xi,2));
        h = rand_vector .<= lgst_c_xi;

        # Find visible units by sampling the hidden layer
        lgst_b_hi = sigmoid(b .+ W'*h);                 # lgst_b_h1 = (#v_units X batch_size)
        rand_vector = rand(size(lgst_b_hi,1),size(lgst_b_hi,2));
        x = rand_vector .<= lgst_b_hi;

        # Save sample
        samples_return = copy(x);
    end
    return samples_return
end;

function compute_Pxbinary_PTcondition(x,W,b,c)
    # Computation of the unnormalized probability but a term is canceled since it is not needed in PT
    Px = prod(1 .+ exp.(W*x .+ c), dims=1);  # Only useful for binary inputs
end;

function gibbsPT(x, W, b, c, NG, K, NG_par)
    samples_return = zeros(size(x,1),trunc(Int, NG/5));
    count = 1;
    candidates_T = range(0,stop=1,length=K);  # K different RBMs for PT

    # Initialize samples as a matrix of [x_1,x_1,...,x_1]
    samples = repeat(x,1,K);
    # Start Gibbs sampling with multiple RBMS (Paralle Tempering)
    for i=1:NG
        for idt=1:K  # This can be done in parallel
            t = candidates_T[idt];
            samples[:,idt] = gibbsRBMv2(samples[:,idt], t*W, b, c, NG_par);
        end

        for idt=2:K
            WT1 = copy(candidates_T[idt-1].*W);
            WT2 = copy(candidates_T[idt].*W);
            XT1 = copy(samples[:,idt-1]);
            XT2 = copy(samples[:,idt]);
            pswapnum = compute_Pxbinary_PTcondition(XT2,WT1,b,c).*compute_Pxbinary_PTcondition(XT1,WT2,b,c);
            pswapden = compute_Pxbinary_PTcondition(XT1,WT1,b,c).*compute_Pxbinary_PTcondition(XT2,WT2,b,c);
            pswap = min(1, pswapnum[1]/pswapden[1]);
            if rand() < pswap
                # Swap samples of two consecutive (with respect to temperature) RBM
                aux = copy(samples[:,idt-1]);
                samples[:,idt-1] = copy(samples[:,idt]);
                samples[:,idt] = copy(aux);
            end
        end
        # Save samples
        if i % 5 == 0
            samples_return[:,count] = copy(samples[:,K]);
            count = count + 1;
        end
    end
    return samples_return
end;

function tobin(num)
  # Convert decimal to binary numbers
  @match num begin
    0 => "0"
    1 => "1"
    _ => string(tobin(div(num,2)), mod(num, 2))
  end
end;

function de2bi(num, len)
    # Convert decimal to binary numbers and add zeros at the beginning
    second = tobin(num);
    first = "0"^(len-length(second))
    bin = first * second
    arrbin = zeros(Int8,length(bin),1)
    for i=1:length(bin)
        arrbin[i] = parse(Int8,bin[i]);
    end
    return arrbin
end;

function myLogSumExp(logX)
  max_logX = maximum(logX);
  return (max_logX + log(sum(exp.(logX .- max_logX))));

end;

function generateWeightsRows(typeUnits,Nv,Nh,Ak,b,c)
  # Generate parameters of W with equal rows
  W = ones(Nv+1,Nh+1);
  for kv in 1:Nv
    W[kv+1,:] .= Ak[kv];
  end
  W[1,2:end] .= c;
  W[2:end,1]  = b;
  W[1,1] = 0;

  # Computation of logZ
  logZv = Float64[];
  n_kh = 0;
  for kh in 0:Nh
    kh == 0  ?  n_kh = 1.0  :  n_kh = n_kh * (Nh+1-kh) / kh;
    sumf = 0;
    for kv in 1:Nv
      if typeUnits == 0
        p = Ak[kv]*kh + b[kv];
        p > 30       ?  sumf += p  :  sumf += log(1+exp(p));
      else
        p = Ak[kv]*(Nh-2*kh) + b[kv];
        abs(p) > 30  ?  sumf += p  :  sumf += log(exp(-p)+exp(+p));
      end;
    end;
    if typeUnits == 0
      push!(logZv,log(n_kh) + sumf + c*kh);
    else
      push!(logZv,log(n_kh) + sumf + c*(Nh-2*kh));
    end;
  end;
  # Sanity check - exact computation
  if Nv <= 10 && Nh <= 10   ### it can be improved...
    bb = b;
    cc = c * ones(Nh);
    Z = 0.0;
    if typeUnits == 0
      for ist in 0:2^Nv-1
        x = digits(ist,2,Nv)
        prob = exp(dot(bb,x))
        for j in 1:Nh
          prob *= 1.0 + exp(cc[j] + dot(x,W[2:end,j+1]))
        end
        Z += prob
      end
    else
      for ist in 0:2^Nv-1
        x = 2*digits(ist,2,Nv)-1
        prob = exp(dot(bb,x))
        for j in 1:Nh
          prob *= cosh(cc[j] + dot(x,W[2:end,j]))
        end
        Z += prob
      end
      Z *= 2.0^Nh
    end
    println("(Sanity check logZ) = ",log(Z))
  end
  logZ = myLogSumExp(logZv)
  return W, logZ;
end;

function kl_divergence(p, q)
    return sum(p .* log.(p ./ q));
end;

function uniqueind(mat)
    indices = [];
    occurre = [];
    seen = [];
    for i = 1:size(mat,2)
        new = mat[:,i];
        findings = findall(x->x==new, seen);
        if findings == []
            append!(seen, [new]);
            append!(indices, i);
            append!(occurre, 1);
        else
            occurre[findings[1]] += 1;
        end
    end
    return indices,occurre
end;

function analysis(samples, W, b, c, logZ)
    # Function to compute the probability distribution of the RBM and the Sampling algorithm using only the generated samples
    indices,occurre = uniqueind(samples);  # Find indices of unique columns (samples)
    # Compute model probabilities
    targetprob = zeros(1,length(indices));
    for id = 1:length(indices)
        x = samples[:,indices[id]];
        targetprob[1,id] = MathConstants.e.^(log(exp.(b'*x)[1]) + sum(log.(1 .+ exp.(c' .+ x'*W'))) .- logZ);
    end
    targetprob = targetprob ./ sum(targetprob);
    # Compute experimental probabilities
    experiprob = occurre ./ sum(occurre);
    return targetprob, experiprob
end;

function analysis2(samples, W, b, c, logZ)
    # Function to compute the Log-Likelihood and the Sum of the Probabilities
    indices,occurre = uniqueind(samples);  # Find indices of unique columns (samples)
    # Compute target probabilities
    targetprob = zeros(1,length(indices));
    targetproblog = zeros(1,length(indices));
    for id = 1:length(indices)
        x = samples[:,indices[id]];
        problog = log(exp.(b'*x)[1]) + sum(log.(1 .+ exp.(c' .+ x'*W'))) .- logZ;
        prob = MathConstants.e.^(problog);
        targetproblog[1,id] = problog * occurre[id];
        targetprob[1,id] = prob * occurre[id];
    end
    # Compute LogLikelihood and sum of probabilities
    logLikelihood = sum(targetproblog);
    sumProb = sum(targetprob)/sum(occurre);
    return logLikelihood, sumProb;
end;

function analysis3(topPrbSamples, empiSamplesRBM, empiSamplesPT, W, b, c, logZ)
    # Function to compute the model probability distribution of the RBM and the empirical
    # using the generated samples with higher probability

    # Compute target probabilities
    targetprob = zeros(1,size(topPrbSamples,2));
    experiprobRBM = zeros(1,size(topPrbSamples,2));
    experiprobPT = zeros(1,size(topPrbSamples,2));

    for id = 1:size(topPrbSamples,2)
        x = topPrbSamples[:,id];
        targetprob[1,id] = MathConstants.e.^(log(exp.(b'*x)[1]) + sum(log.(1 .+ exp.(c' .+ x'*W'))) .- logZ);

        countRBM = 0;
        countPT  = 0;
        for j=1:size(empiSamplesRBM,2)
            if empiSamplesRBM[:,j] == x
                countRBM = countRBM + 1;
            end
            if empiSamplesPT[:,j] == x
                countPT = countPT + 1;
            end
        end
        experiprobRBM[1,id] = countRBM + 0.0001;
        experiprobPT[1,id] = countPT + 0.0001;
    end
    targetprob = targetprob ./ sum(targetprob);
    experiprobRBM = experiprobRBM ./ sum(experiprobRBM);
    experiprobPT = experiprobPT ./ sum(experiprobPT);
    return targetprob, experiprobRBM, experiprobPT
end;
