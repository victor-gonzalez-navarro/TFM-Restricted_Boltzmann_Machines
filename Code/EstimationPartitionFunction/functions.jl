using MAT
using Distributed
using Match
using ProgressMeter
using Statistics

function sigmoid(z)
    out = 1.0 ./ (1.0 .+ exp.(-z));
end

function c_assert(boolean, phrase)
    if boolean == false
        println(phrase)
        error(phrase)
    end
end

function compute_Pxbinary_PTcondition(x,W,b,c)
    # Computation of the unnormalized probability but a term is canceled since it is not needed in PT
    Px = prod(1 .+ exp.(W*x .+ c), dims=1);  # Only useful for binary inputs
end

function estimate_Zb(M,numInt,NG2,WA,bA,cA,WB,bB,cB)
    # Estimate partition function using Annealed Importance Sampling
    # M:        Number of samples (and weights) to estimate partition fucntion partition = Zb/Za = mean(weights)
    # numInt:   Number of intermediate distributions for estimating Za/Zb
    # Wx,bx,cx: Parameters of RBM number x

    weights = zeros(M,1);
    betas = range(0,stop=1,length=numInt);
    @showprogress 1 "Computing estimation of Z... " for m=1:M
        # Sample from p_0 (Base rate RBM W=0)
        rand_vector = rand(size(bA,1),size(bA,2));
        vk = rand_vector .<= sigmoid(bA);
        weights[m,1] = rationProbUnnorm([betas[1] betas[2]],vk,bA,cA,WB,bB,cB);
        # Sample from v_1 to v_K
        for it = 2:(numInt-1)
            vk = gibbsRBMais(betas[it],vk,WA,bA,cA,WB,bB,cB,NG2);
            weights[m,1] = weights[m,1]*(rationProbUnnorm([betas[it],betas[it+1]],vk,bA,cA,WB,bB,cB));
        end
    end
    ratioZaZb = Statistics.mean(weights);

    # Compute partition function of base RBM (Wij = 0)
    Za = prod(1.0 .+ exp.(bA))*prod(1.0 .+ exp.(cA));
    # Compute partition function of new RBM
    Zb = Za * ratioZaZb;
    return Zb
end

function rationProbUnnorm(betas,x,bA,cA,WB,bB,cB)
    # Compute p_k(v_{k+1}) where v_{k+1} does not mean the feature {k+1}
    info = zeros(1,2);
    for it=1:2
        first  = exp.((1-betas[it])*bA'*x) * prod(1 .+ exp.((1-betas[it])*cA));
        second = exp.(betas[it]*bB'*x) * prod(1 .+ exp.(betas[it]*(WB*x + cB)));
        info[1,it] = first[1] * second[1];
    end
    ratio = info[1,2]/info[1,1];
    return ratio
end

function tobin(num)
  # Convert decimal to binary numbers
  @match num begin
    0 => "0"
    1 => "1"
    _ => string(tobin(div(num,2)), mod(num, 2))
  end
end

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
end

function gibbsRBMais(betak,x,WA,bA,cA,WB,bB,cB,NG)
    for i = 1:NG
        # Find hidden units by sampling the visible layer
        lgst_c_xiA = sigmoid((1-betak)*(cA + WA*x));                # lgst_c_x = (#h_unitsX1)
        lgst_c_xiB = sigmoid((betak)*(cB + WB*x));                  # lgst_c_x = (#h_unitsX1)
        rand_vectorA = rand(size(lgst_c_xiA,1),size(lgst_c_xiA,2));
        rand_vectorB = rand(size(lgst_c_xiB,1),size(lgst_c_xiB,2));
        hA = rand_vectorA .<= lgst_c_xiA;
        hB = rand_vectorB .<= lgst_c_xiB;

        # Find visible units by sampling the hidden layer
        lgst_b_hi = sigmoid((1-betak)*(bA + WA'*hA) + betak*(bB + WB'*hB));  # lgst_b_h1 = (#v_unitsX1)
        rand_vector = rand(size(lgst_b_hi,1),size(lgst_b_hi,2));
        x = rand_vector .<= lgst_b_hi;
    end
    return x
end

function compute_Zb(W,b,c,bin)
    # Computation of Zb (reduced version, computationally feasible)
    realZb = 0;
    if bin == 1
        K = 2^(length(c))-1;
        @showprogress 1 "Computing Z... " for k=0:K
            h = de2bi(k,length(c));
            realZb = realZb + exp(c'*h)[1] * prod(1 .+ exp.(b' .+ h'*W));
        end
    else
        K = 2^(length(b))-1;
        @showprogress 1 "Computing Z... " for k=0:K
            x = de2bi(k,length(b));
            realZb = realZb + exp(b'*x)[1] * prod(1 .+ exp.(c' .+ x'*W'));
        end
    end
    return realZb
end

function computeRealZ(Nneurons,W,b,c,Ninput)
    # Computation of the ground truth partition function Z
    if Nneurons <= 25 && Nneurons <= Ninput
        Z = compute_Zb(W,b,c,1);
    elseif Ninput <= 25
        Z = compute_Zb(W,b,c,2);
    end
    return Z;
end
