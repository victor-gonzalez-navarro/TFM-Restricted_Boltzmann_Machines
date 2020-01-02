using MAT
using Distributed
using Match
using ProgressMeter
using Statistics

function sigmoid(z)
    out = 1.0 ./ (1.0 .+ exp.(-z));
end

function read_variable(absolute_path, var)
    println("Reading variable " * var[1:end-4])
    file = matopen(absolute_path * "Data/" * var)
    variable = read(file, var[1:end-4])
    close(file)
    return variable
end

function LoadData_MNIST(absolute_path);
    println("Loading data set... ");
    dataTrain_x = read_variable(absolute_path,"Train_x.mat")
    dataTrain_y = read_variable(absolute_path,"Train_y.mat")
    dataTest_x = read_variable(absolute_path,"Test_x.mat")
    dataTest_y = read_variable(absolute_path,"Test_y.mat")

    dataTrain_x = dataTrain_x/255.0;
    dataTrain_y = dataTrain_y/255.0;

    return dataTrain_x, dataTrain_y, dataTest_x, dataTest_y
end

function c_assert(boolean, phrase)
    if boolean == false
        println(phrase)
        error(phrase)
    end
end

function gibbsRBM(x, W, b, c, NG)
    lgst_c_x1 = zeros(size(W,1),size(x,2));
    for i = 1:NG
        # Find hidden units by sampling the visible layer
        lgst_c_xi = sigmoid(c .+ W* x);                  # lgst_c_x = (#h_units X batch_size)
        if i == 1
            lgst_c_x1 = copy(lgst_c_xi);
        end
        rand_vector = rand(size(lgst_c_xi,1),size(lgst_c_xi,2));
        h = rand_vector .<= lgst_c_xi;

        # Find visible units by sampling the hidden layer
        lgst_b_hi = sigmoid(b .+ W'*h);                 # lgst_b_h1 = (#v_units X batch_size)
        rand_vector = rand(size(lgst_b_hi,1),size(lgst_b_hi,2));
        x = rand_vector .<= lgst_b_hi;
    end
    return lgst_c_x1,x
end

function gibbsRBMkeep(x, W, b, c, NG)
    lgst_c_x1 = zeros(size(W,1),size(x,2));
    xsamplestrack = zeros(size(x,1),size(x,2)*NG);
    for i = 1:NG
        # Find hidden units by sampling the visible layer
        lgst_c_xi = sigmoid(c .+ W* x);                  # lgst_c_x = (#h_units X batch_size)
        if i == 1
            lgst_c_x1 = copy(lgst_c_xi);
        end
        rand_vector = rand(size(lgst_c_xi,1),size(lgst_c_xi,2));
        h = rand_vector .<= lgst_c_xi;

        # Find visible units by sampling the hidden layer
        lgst_b_hi = sigmoid(b .+ W'*h);                 # lgst_b_h1 = (#v_units X batch_size)
        rand_vector = rand(size(lgst_b_hi,1),size(lgst_b_hi,2));
        x = rand_vector .<= lgst_b_hi;
        init  = (i-1)*size(x,2)+1;
        finit = init + size(x,2) - 1;
        xsamplestrack[:,init:finit] = copy(x);
    end
    return lgst_c_x1,xsamplestrack
end

function compute_Pxbinary(x,W,b,c)
    # Computation of the unnormalized probability
    Px = exp.(b'*x) .* prod(1 .+ exp.(W*x .+ c), dims=1);  # Only useful for binary inputs
end

function compute_Pxbinary_PTcondition(x,W,b,c)
    # Computation of the unnormalized probability but a term is canceled since it is not needed in PT
    Px = prod(1 .+ exp.(W*x .+ c), dims=1);  # Only useful for binary inputs
end

function gibbsPT(x_1, W, b, c, NG, K, NG_par)
    candidates_T = range(0,stop=1,length=K);  # K different RBMs for PT
    batchsize = size(x_1, 2);
    # Initialize samples as a matrix of [x_1,x_2,...,x_100|...|x_1,x_2,...,x_100] where 100 corresponds to the batchsize
    samples = repeat(x_1,1,1,K);
    # Start Gibbs sampling with multiple RBMS (Paralle Tempering)
    for i = 1:NG
        for idt=1:K  # This can be done in parallel
            t = candidates_T[idt];
            _, samples[:,:,idt] = gibbsRBM(samples[:,:,idt], t*W, b, c, NG_par);
        end
        for idt=2:K
            WT1 = copy(candidates_T[idt-1].*W);
            WT2 = copy(candidates_T[idt].*W);
            XT1 = copy(samples[:,:,idt-1]);
            XT2 = copy(samples[:,:,idt]);
            pswapnum = compute_Pxbinary_PTcondition(XT2,WT1,b,c).*compute_Pxbinary_PTcondition(XT1,WT2,b,c);
            pswapden = compute_Pxbinary_PTcondition(XT1,WT1,b,c).*compute_Pxbinary_PTcondition(XT2,WT2,b,c);
            pswap = min.(1, pswapnum./pswapden);
            for index=1:batchsize
                if rand(1)[1] < pswap[1,index]
                    # Swap samples of two consecutive (with respect to temperature) RBM
                    aux = copy(samples[:,index,idt-1]);
                    samples[:,index,idt-1] = copy(samples[:,index,idt]);
                    samples[:,index,idt] = copy(aux);
                end
            end
        end
    end
    x = copy(samples[:,:,end]);
    lgst_c_x1 = sigmoid(c .+ W*x_1);
    return lgst_c_x1,x
end

function estimate_Zb(M,numInt,NG2,WA,bA,cA,WB,bB,cB)
    # Estimate partition function using Annealed Importance Sampling
    # M:        Number of samples (and weights) to estimate partition fucntion partition = Zb/Za = mean(weights)
    # numInt:   Number of intermediate distributions for estimating Za/Zb
    # Wx,bx,cx: Parameters of RBM number x

    weights = zeros(M,1);
    betas = range(0,stop=1,length=numInt);
    for m=1:M
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
    # Using equation below 17
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
        # Obtained from Equations 15, 16, 17 of the paper: On the Quantitative Analysis of Deep Belief Networks
        # Find hidden units by sampling the visible layer
        lgst_c_xiA = sigmoid((1-betak)*(cA + WA*x));                # lgst_c_x = (#h_unitsX1)
        lgst_c_xiB = sigmoid((betak)*(cB + WB*x));                  # lgst_c_x = (#h_unitsX1)
        rand_vectorA = rand(size(lgst_c_xiA,1),size(lgst_c_xiA,2));
        rand_vectorB = rand(size(lgst_c_xiB,1),size(lgst_c_xiB,2));
        hA = rand_vectorA .<= lgst_c_xiA;
        hB = rand_vectorB .<= lgst_c_xiB;

        # Find visible units by sampling the hidden layer
        lgst_b_hi = sigmoid((1-betak)*(bA + WA'*hA) + betak*(bB + WB'*hB));      # lgst_b_h1 = (#v_unitsX1)
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

function compute_ZbNP(W,b,c,bin)
    # Computation of Zb (reduced version, computationally feasible) without showing the progress on the screen
    realZb = 0;
    if bin == 1
        K = 2^(length(c))-1;
        for k=0:K
            h = de2bi(k,length(c));
            realZb = realZb + exp(c'*h)[1] * prod(1 .+ exp.(b' .+ h'*W));
        end
    else
        K = 2^(length(b))-1;
        for k=0:K
            x = de2bi(k,length(b));
            realZb = realZb + exp(b'*x)[1] * prod(1 .+ exp.(c' .+ x'*W'));
        end
    end
    return realZb
end

function createDataset(datatype, ReducedData, abs_path)
    if datatype == "MNIST"
        # Each line in matrix is a sample
        include(abs_path * "functions.jl");
        dataTrain_x, dataTrain_y, dataTest_x, dataTest_y = LoadData_MNIST(abs_path);

        if ReducedData > 0
          c_assert(ReducedData + round(ReducedData/3) < 60001, "Please reduce the value of ReducedData");
          r            = randperm(size(dataTrain_x,1));
          full_data    = [dataTrain_x[r[1:ReducedData + trunc(Int,round(ReducedData/3))],:]; dataTest_x];
          full_targets = [dataTrain_y[r[1:ReducedData + trunc(Int,round(ReducedData/3))],:]; dataTest_y];
        end;

        x      = full_data';
        x      = x .- minimum(x);
        I_real = (x/maximum(x));
        I      = round.(I_real[:,1:ReducedData]);                # Transform input into binary values

    elseif datatype[1:2] == "BS"
        if datatype == "BS02-3"
            rows = 2;
            cols = 3;
        elseif datatype == "BS09"
            rows = 3;
            cols = 3;
        elseif datatype == "BS16"
            rows = 4;
            cols = 4;
        end
        I = zeros(rows*cols, (2^rows)+(2^cols) - 2);
        # If aux = [0 0 0 1] --> [0 0 0 1; ...; 0 0 0 1]
        N = (2^cols) - 2;
        for i=1:N
            aux = de2bi(i,cols);
            I[:,i] = repeat(aux, rows,1)  ;
        end
        # If aux = [0 0 0 1] --> [0 ... 0; 0 ... 0; 0 ... 0; 1 ... 1]
        for i=1:((2^rows) - 2)
            aux = de2bi(i,rows);
            for j=1:rows
                ini = cols*(j-1)+1;
                I[ini:(ini+cols-1) ,N + i] =  aux[j]*ones(cols,1);
            end
        end
        # Add 0000 and 1111
        I[:,end-1] = zeros(rows*cols, 1);
        I[:,end] = ones(rows*cols, 1);
        I_real = 0;
        full_targets = 0;
    elseif datatype[1:3] == "LSE"
        if datatype == "LSE11"
            N = 4;
        elseif datatype == "LSE15"
            N = 6;
        end
        I = zeros(2*N + 3, (2^N)*3);                            # Attr X Samples
        for i=1:(2^N)
            aux = de2bi(i-1, N);
            I[1:N, i]       = copy(aux);
            I[1:N, (2^N+i)]   = copy(aux);
            I[1:N, (2*(2^N)+i)] = copy(aux);
            for j in [1, 2, 4]
                bits = de2bi(j,3);
                if j == 1      # 001
                    I[(N+1):(N+3), i]             = copy(bits);
                    I[(end-N+1):(end-1), i]       = I[2:N, i];
                    I[end, i]                     = I[1, i];
                elseif j == 2  # 010
                    I[(N+1):(N+3), (2^N+i)]       = copy(bits);
                    I[(end-N+1):end, (2^N+i)]     = I[1:N, (2^N+i)];
                else           # 100 --> shift to the right
                    I[(N+1):(N+3), (2*(2^N)+i)]   = copy(bits);
                    I[(end-N+2):end, (2*(2^N)+i)] = I[1:(N-1), (2*(2^N)+i)];
                    I[(end-N+1), (2*(2^N)+i)]     = I[N, (2*(2^N)+i)];
                end
            end
        end
        I_real = 0;
        full_targets = 0;

    elseif datatype[1] == 'P'
        if datatype == "P08"
            N = 7;
        elseif datatype == "P10"
            N = 9;
        end
        I = zeros(N+1, 2^N);
        it = 2^N - 1;
        for i=0:it
            aux = de2bi(i, N);
            I[1:(end-1), (i+1)] = copy(aux);
            if sum(aux)%2 == 0
                I[end, (i+1)]   = 1;
            else
                I[end, (i+1)]   = 0;
            end
        end
        I_real = 0;
        full_targets = 0;
    end
    return I, I_real, full_targets
end

function computeLL(I,Nneurons,W,b,c)
    # Computation of the LogLikelihood
    if Nneurons <= 25 && Nneurons <= size(I,1)
        Z = compute_Zb(W,b,c,1);
    elseif size(I,1) <= 25
        Z = compute_Zb(W,b,c,2);
    end
    LL = sum(log.(compute_Pxbinary(I,W,b,c)))-size(I,2)*log(Z);
    return LL,Z
end

function computeLLNP(I,Nneurons,W,b,c)
    # Computation of the LogLikelihood without showing the progress on the screen
    if Nneurons <= 25 && Nneurons <= size(I,1)
        Z = compute_ZbNP(W,b,c,1);
    elseif size(I,1) <= 25
        Z = compute_ZbNP(W,b,c,2);
    end
    LL = sum(log.(compute_Pxbinary(I,W,b,c)))-size(I,2)*log(Z);
    return LL,Z
end

function computeRBMparamAll(W, b, c, LLvec, Z, I, Nexamples, batchsize, NG, Nepochs, momentum, targdis, plotLL)
    # Compute RBM parameters either for CD, WCD and PT
    W_CD1    = copy(W); b_CD1   = copy(b); c_CD1   = copy(c); LLvec_CD1   = copy(LLvec); Z_CD1   = copy(Z); VdW_CD1   = zeros(size(W,1),size(W,2)); Vdb_CD1 = zeros(size(b,1),size(b,2)); Vdc_CD1 = zeros(size(c,1),size(c,2));

    # W_CD     = copy(W); b_CD    = copy(b); c_CD    = copy(c); LLvec_CD    = copy(LLvec); Z_CD    = copy(Z); VdW_CD    = copy(VdW_CD1); Vdb_CD    = copy(Vdb_CD1); Vdc_CD    = copy(Vdc_CD1);
    # W_WCD    = copy(W); b_WCD   = copy(b); c_WCD   = copy(c); LLvec_WCD   = copy(LLvec); Z_WCD   = copy(Z); VdW_WCD   = copy(VdW_CD1); Vdb_WCD   = copy(Vdb_CD1); Vdc_WCD   = copy(Vdc_CD1);
    W_PTCD   = copy(W); b_PTCD  = copy(b); c_PTCD  = copy(c); LLvec_PTCD  = copy(LLvec); Z_PTCD  = copy(Z); VdW_PTCD  = copy(VdW_CD1); Vdb_PTCD  = copy(Vdb_CD1); Vdc_PTCD  = copy(Vdc_CD1);
    # W_PTWCD  = copy(W); b_PTWCD = copy(b); c_PTWCD = copy(c); LLvec_PTWCD = copy(LLvec); Z_PTWCD = copy(Z); VdW_PTWCD = copy(VdW_CD1); Vdb_PTWCD = copy(Vdb_CD1); Vdc_PTWCD = copy(Vdc_CD1);
    # W_CDP    = copy(W); b_CDP   = copy(b); c_CDP   = copy(c); LLvec_CDP   = copy(LLvec); Z_CDP   = copy(Z); VdW_CDP   = copy(VdW_CD1); Vdb_CDP   = copy(Vdb_CD1); Vdc_CDP   = copy(Vdc_CD1);
    # W_WCDP   = copy(W); b_WCDP  = copy(b); c_WCDP  = copy(c); LLvec_WCDP  = copy(LLvec); Z_WCDP  = copy(Z); VdW_WCDP  = copy(VdW_CD1); Vdb_WCDP  = copy(Vdb_CD1); Vdc_WCDP  = copy(Vdc_CD1);
    # W_FIN   = copy(W) ; b_FIN   = copy(b); c_FIN   = copy(c); LLvec_FIN   = copy(LLvec); Z_FIN   = copy(Z); VdW_FIN   = copy(VdW_CD1); Vdb_FIN   = copy(Vdb_CD1); Vdc_FIN   = copy(Vdc_CD1);

    for i=1:Nepochs
      ordering = randperm(Nexamples);
      Iepoch = copy(I[:, ordering]);

      ini_sample = 1;
      end_sample = min(Nexamples, ini_sample+batchsize-1);

      while ini_sample <= Nexamples
        x_1 = copy(Iepoch[:,ini_sample:end_sample]);
        truebatchsize = size(x_1,2);

        # Find hidden units by sampling the visible layer and viceversa
        # lgst_c_x1_CD1  , x_2_CD1   = gibbsRBM(x_1, W_CD1  , b_CD1  , c_CD1  , 1);
        # lgst_c_x1_CD   , x_2_CD    = gibbsRBM(x_1, W_CD   , b_CD   , c_CD   , NG);
        # lgst_c_x1_WCD  , x_2_WCD   = gibbsRBM(x_1, W_WCD  , b_WCD  , c_WCD  , NG);
        lgst_c_x1_PTCD , x_2_PTCD  = gibbsPT(x_1 , W_PTCD , b_PTCD , c_PTCD , NG, K, NG_par);
        # lgst_c_x1_PTWCD, x_2_PTWCD = gibbsPT(x_1 , W_PTWCD, b_PTWCD, c_PTWCD, NG, K, NG_par);
        # lgst_c_x1_CDP  , x_2_CDP   = gibbsRBM(x_1, W_CDP  , b_CDP  , c_CDP  , NG*K);
        # lgst_c_x1_WCDP , x_2_WCDP  = gibbsRBM(x_1, W_WCDP , b_WCDP , c_WCDP , NG*K);
        # lgst_c_x1_FIN  , x_2_FIN   = gibbsRBMkeep(x_1, W_FIN  , b_FIN  , c_FIN  , NG*K);

        # Compute weights of WCD
        # probweights_WCD   = compute_Pxbinary(x_2_WCD  , W_WCD  , b_WCD  , c_WCD);
        # probweights_PTWCD = compute_Pxbinary(x_2_PTWCD, W_PTWCD, b_PTWCD, c_PTWCD);
        # probweights_WCDP  = compute_Pxbinary(x_2_WCDP , W_WCDP , b_WCDP , c_WCDP);
        # c_assert(sum(probweights_WCD)   != Inf, "There is at least one Inf in probweights_WCD");
        # c_assert(sum(probweights_PTWCD) != Inf, "There is at least one Inf in probweights_PTWCD");
        # c_assert(sum(probweights_WCDP)  != Inf, "There is at least one Inf in probweights_WCDP");
        # probweights_WCD   = probweights_WCD ./ sum(probweights_WCD);
        # probweights_PTWCD = probweights_PTWCD ./ sum(probweights_PTWCD);
        # probweights_WCDP  = probweights_WCDP ./ sum(probweights_WCDP);

        # Compute weights of any algorithm except WCD
        probweights       = (1/truebatchsize) * ones(1,truebatchsize);

        # Computation of the CD-1 terms for x_2
        # lgst_c_x2_CD1   = sigmoid(c_CD1   .+ W_CD1   * x_2_CD1);                # lgst_c_x2 = (#h_units X batchsize)
        # lgst_c_x2_CD    = sigmoid(c_CD    .+ W_CD    * x_2_CD);                 # lgst_c_x2 = (#h_units X batchsize)
        # lgst_c_x2_WCD   = sigmoid(c_WCD   .+ W_WCD   * x_2_WCD);                # lgst_c_x2 = (#h_units X batchsize)
        lgst_c_x2_PTCD  = sigmoid(c_PTCD  .+ W_PTCD  * x_2_PTCD);               # lgst_c_x2 = (#h_units X batchsize)
        # lgst_c_x2_PTWCD = sigmoid(c_PTWCD .+ W_PTWCD * x_2_PTWCD);              # lgst_c_x2 = (#h_units X batchsize)
        # lgst_c_x2_CDP   = sigmoid(c_CDP   .+ W_CDP   * x_2_CDP);                # lgst_c_x2 = (#h_units X batchsize)
        # lgst_c_x2_WCDP  = sigmoid(c_WCDP  .+ W_WCDP  * x_2_WCDP);               # lgst_c_x2 = (#h_units X batchsize)
        # lgst_c_x2_FIN   = sigmoid(c_FIN   .+ W_FIN   * x_2_FIN);                # lgst_c_x2 = (#h_units X batchsize*NG)

        # Update parameters
        # VdW_CD1  , Vdb_CD1  , Vdc_CD1  , W_CD1  , b_CD1  , c_CD1   = update_parameters(probweights, probweights      , x_1, lgst_c_x1_CD1  , x_2_CD1  , lgst_c_x2_CD1  , momentum, VdW_CD1  , Vdb_CD1  , Vdc_CD1  , W_CD1  , b_CD1  , c_CD1);
        # VdW_CD   , Vdb_CD   , Vdc_CD   , W_CD   , b_CD   , c_CD    = update_parameters(probweights, probweights      , x_1, lgst_c_x1_CD   , x_2_CD   , lgst_c_x2_CD   , momentum, VdW_CD   , Vdb_CD   , Vdc_CD   , W_CD   , b_CD   , c_CD);
        # VdW_WCD  , Vdb_WCD  , Vdc_WCD  , W_WCD  , b_WCD  , c_WCD   = update_parameters(probweights, probweights_WCD  , x_1, lgst_c_x1_WCD  , x_2_WCD  , lgst_c_x2_WCD  , momentum, VdW_WCD  , Vdb_WCD  , Vdc_WCD  , W_WCD  , b_WCD  , c_WCD);
        VdW_PTCD , Vdb_PTCD , Vdc_PTCD , W_PTCD , b_PTCD , c_PTCD  = update_parameters(probweights, probweights      , x_1, lgst_c_x1_PTCD , x_2_PTCD , lgst_c_x2_PTCD , momentum, VdW_PTCD , Vdb_PTCD , Vdc_PTCD , W_PTCD , b_PTCD , c_PTCD);
        # VdW_PTWCD, Vdb_PTWCD, Vdc_PTWCD, W_PTWCD, b_PTWCD, c_PTWCD = update_parameters(probweights, probweights_PTWCD, x_1, lgst_c_x1_PTWCD, x_2_PTWCD, lgst_c_x2_PTWCD, momentum, VdW_PTWCD, Vdb_PTWCD, Vdc_PTWCD, W_PTWCD, b_PTWCD, c_PTWCD);
        # VdW_CDP  , Vdb_CDP  , Vdc_CDP  , W_CDP  , b_CDP  , c_CDP   = update_parameters(probweights, probweights      , x_1, lgst_c_x1_CDP  , x_2_CDP  , lgst_c_x2_CDP  , momentum, VdW_CDP  , Vdb_CDP  , Vdc_CDP  , W_CDP  , b_CDP  , c_CDP);
        # VdW_WCDP , Vdb_WCDP , Vdc_WCDP , W_WCDP , b_WCDP , c_WCDP  = update_parameters(probweights, probweights_WCDP , x_1, lgst_c_x1_WCDP , x_2_WCDP , lgst_c_x2_WCDP , momentum, VdW_WCDP , Vdb_WCDP , Vdc_WCDP , W_WCDP , b_WCDP , c_WCDP);
        # VdW_FIN  , Vdb_FIN  , Vdc_FIN  , W_FIN  , b_FIN  , c_FIN   = update_parameterskeep(probweights, probweights  , x_1, lgst_c_x1_FIN  , x_2_FIN  , lgst_c_x2_FIN  , momentum, VdW_FIN  , Vdb_FIN  , Vdc_FIN  , W_FIN  , b_FIN  , c_FIN, NG*K);

        ini_sample = end_sample + 1;
        end_sample = min(Nexamples, ini_sample+batchsize-1);
      end
      if plotLL==1
        # Compute LL to proove maximization
        # LL_CD1  , Z_CD1    = computeLLNP(I, Nneurons, W_CD1  , b_CD1  , c_CD1);
        # LL_CD   , Z_CD     = computeLLNP(I, Nneurons, W_CD   , b_CD   , c_CD);
        # LL_WCD  , Z_WCD    = computeLLNP(I, Nneurons, W_WCD  , b_WCD  , c_WCD);
        LL_PTCD , Z_PTCD   = computeLLNP(I, Nneurons, W_PTCD , b_PTCD , c_PTCD);
        # LL_PTWCD, Z_PTWCD  = computeLLNP(I, Nneurons, W_PTWCD, b_PTWCD, c_PTWCD);
        # LL_CDP  , Z_CDP    = computeLLNP(I, Nneurons, W_CDP  , b_CDP  , c_CDP);
        # LL_WCDP , Z_WCDP   = computeLLNP(I, Nneurons, W_WCDP , b_WCDP , c_WCDP);
        # LL_FIN  , Z_FIN    = computeLLNP(I, Nneurons, W_FIN , b_FIN , c_FIN);
        # LLvec_CD1[1,i+1]   = LL_CD1;
        # LLvec_CD[1,i+1]    = LL_CD;
        # LLvec_WCD[1,i+1]   = LL_WCD;
        LLvec_PTCD[1,i+1]  = LL_PTCD;
        # LLvec_PTWCD[1,i+1] = LL_PTWCD;
        # LLvec_CDP[1,i+1]   = LL_CDP;
        # LLvec_WCDP[1,i+1]  = LL_WCDP;
        # LLvec_FIN[1,i+1]   = LL_FIN;
      end
    end
    # expdis_CD1   = compute_Pxbinary(I, W_CD1  , b_CD1  , c_CD1  ) ./ Z_CD1;
    # expdis_CD    = compute_Pxbinary(I, W_CD   , b_CD   , c_CD   ) ./ Z_CD;
    # expdis_WCD   = compute_Pxbinary(I, W_WCD  , b_WCD  , c_WCD  ) ./ Z_WCD;
    expdis_PTCD  = compute_Pxbinary(I, W_PTCD , b_PTCD , c_PTCD ) ./ Z_PTCD;
    # expdis_PTWCD = compute_Pxbinary(I, W_PTWCD, b_PTWCD, c_PTWCD) ./ Z_PTWCD;
    # expdis_CDP   = compute_Pxbinary(I, W_CDP  , b_CDP  , c_CDP  ) ./ Z_CDP;
    # expdis_WCDP  = compute_Pxbinary(I, W_WCDP , b_WCDP , c_WCDP ) ./ Z_WCDP;
    # expdis_FIN   = compute_Pxbinary(I, W_FIN  , b_FIN  , c_FIN  ) ./ Z_FIN;
    params = Dict(
                  # "W_CD1"   => W_CD1  , "b_CD1"   => b_CD1  , "c_CD1"   => c_CD1,
                  # "W_CD"    => W_CD   , "b_CD"    => b_CD   , "c_CD"    => c_CD,
                  # "W_WCD"   => W_WCD  , "b_WCD"   => b_WCD  , "c_WCD"   => c_WCD,
                  "W_PTCD"  => W_PTCD , "b_PTCD"  => b_PTCD , "c_PTCD"  => c_PTCD,
                  # "W_PTWCD" => W_PTWCD, "b_PTWCD" => b_PTWCD, "c_PTWCD" => c_PTWCD,
                  # "W_CDP"   => W_CDP  , "b_CDP"   => b_CDP  , "c_CDP"   => c_CDP,
                  # "W_WCDP"  => W_WCDP , "b_WCDP"  => b_WCDP , "c_WCDP"  => c_WCDP,
                  # "W_FIN"   => W_FIN  , "b_FIN"   => b_FIN  , "c_FIN"   => c_FIN
                  );
    plotparams = Dict(
                      # "LLvec_CD1"   => LLvec_CD1  , "Z_CD1"   => Z_CD1  , "expdis_CD1"   => expdis_CD1,
                      # "LLvec_CD"    => LLvec_CD   , "Z_CD"    => Z_CD   , "expdis_CD"    => expdis_CD,
                      # "LLvec_WCD"   => LLvec_WCD  , "Z_WCD"   => Z_WCD  , "expdis_WCD"   => expdis_WCD,
                      "LLvec_PTCD"  => LLvec_PTCD , "Z_PTCD"  => Z_PTCD , "expdis_PTCD"  => expdis_PTCD,
                      # "LLvec_PTWCD" => LLvec_PTWCD, "Z_PTWCD" => Z_PTWCD, "expdis_PTWCD" => expdis_PTWCD,
                      # "LLvec_CDP"   => LLvec_CDP  , "Z_CDP"   => Z_CDP  , "expdis_CDP"   => expdis_CDP,
                      # "LLvec_WCDP"  => LLvec_WCDP , "Z_WCDP"  => Z_WCDP , "expdis_WCDP"  => expdis_WCDP,
                      # "LLvec_FIN"   => LLvec_FIN  , "Z_FIN"   => Z_FIN  , "expdis_FIN"   => expdis_FIN
                      );
    return params, plotparams
end

function update_parameters(probweightsP,probweightsN,x_1,lgst_c_x1,x_2,lgst_c_x2,momentum,VdW,Vdb,Vdc,W,b,c)
    probweightsPmat1 = repeat(probweightsP, size(lgst_c_x1,1), 1);
    probweightsPmat2 = repeat(probweightsP, size(x_1,1), 1);
    probweightsNmat1 = repeat(probweightsN, size(lgst_c_x1,1), 1);
    probweightsNmat2 = repeat(probweightsN, size(x_1,1), 1);
    # Updating formulas
    dW = (lgst_c_x1 .* probweightsPmat1)*x_1' - (lgst_c_x2 .* probweightsNmat1)*x_2';
    db = sum((x_1 .* probweightsPmat2 - x_2 .* probweightsNmat2), dims=2);
    dc = sum((lgst_c_x1 .* probweightsPmat1 - lgst_c_x2 .* probweightsNmat1), dims=2);
    VdW = momentum .* VdW + (1-momentum) .* dW;
    Vdb = momentum .* Vdb + (1-momentum) .* db;
    Vdc = momentum .* Vdc + (1-momentum) .* dc;
    W = W + learningR * VdW;
    b = b + learningR * Vdb;
    c = c + learningR * Vdc;
    return VdW,Vdb,Vdc,W,b,c
end

function update_parameterskeep(probweightsP,probweightsN,x_1,lgst_c_x1,x_2,lgst_c_x2,momentum,VdW,Vdb,Vdc,W,b,c,NG)
    probweightsPmat1 = repeat(probweightsP, size(lgst_c_x1,1), 1);
    probweightsPmat2 = repeat(probweightsP, size(x_1,1), 1);
    probweightsNmat1 = repeat(probweightsN, size(lgst_c_x1,1), 1);
    probweightsNmat2 = repeat(probweightsN, size(x_1,1), 1);
    x_2mod       = zeros(size(x_1,1),size(x_1,2));
    lgst_c_x2mod = zeros(size(lgst_c_x1,1),size(lgst_c_x1,2));
    positions    = range(1, stop=((NG-1)*size(x_1,2)+1), length=NG) |> collect;
    positions    = floor.(Int, positions)
    for i=1:size(x_1,2)
        mat1              = copy(x_2[:,positions]);
        mat2              = copy(lgst_c_x2[:,positions]);
        x_2mod[:,i]       = sum(mat1, dims=2) ./ NG;
        lgst_c_x2mod[:,i] = sum(mat2, dims=2) ./ NG;
        positions = positions .+ 1;
    end
    # Updating formulas
    dW = (lgst_c_x1 .* probweightsPmat1)*x_1' - (lgst_c_x2mod .* probweightsNmat1)*x_2mod';   # (#h_units X #visible_units)
    db = sum((x_1 .* probweightsPmat2 - x_2mod .* probweightsNmat2), dims=2);
    dc = sum((lgst_c_x1 .* probweightsPmat1 - lgst_c_x2mod .* probweightsNmat1), dims=2);
    VdW = momentum .* VdW + (1-momentum) .* dW;
    Vdb = momentum .* Vdb + (1-momentum) .* db;
    Vdc = momentum .* Vdc + (1-momentum) .* dc;
    W = W + learningR * VdW;
    b = b + learningR * Vdb;
    c = c + learningR * Vdc;
    return VdW,Vdb,Vdc,W,b,c
end

function  update_paramDEL(x_1,Z_DEL,momentum,VdW,Vdb,Vdc,W,b,c)
    negativePhaseW = zeros(size(W,1),size(W,2));
    negativePhaseB = zeros(size(b,1),size(b,2));
    negativePhaseC = zeros(size(c,1),size(c,2));
    batchsize = size(x_1,2);
    K = 2^(length(b))-1;
    for k=0:K
        x = de2bi(k,length(b));
        px = (exp(b'*x)[1] * prod(1 .+ exp.(c' .+ x'*W'))) / Z_DEL;
        lgst_c_x = sigmoid(c .+ W * x);
        negativePhaseW = negativePhaseW + ((lgst_c_x * x') .* px);
        negativePhaseB = negativePhaseB + (x .* px);
        negativePhaseC = negativePhaseC + (lgst_c_x .* px);
    end
    lgst_c_x1 = sigmoid(c .+ W * x_1);
    db = sum((x_1 - repeat(negativePhaseB,1,size(x_1,2))), dims=2) ./ batchsize;
    dc = sum(lgst_c_x1 - repeat(negativePhaseC,1,size(x_1,2)), dims=2) ./ batchsize;
    dW = (lgst_c_x1*x_1' - negativePhaseW .* batchsize) ./ batchsize;         # The multiplication is due to the fact that I have a batch
    VdW = momentum .* VdW + (1-momentum) .* dW;
    Vdb = momentum .* Vdb + (1-momentum) .* db;
    Vdc = momentum .* Vdc + (1-momentum) .* dc;
    W = W + learningR * VdW;
    b = b + learningR * Vdb;
    c = c + learningR * Vdc;
    return VdW,Vdb,Vdc,W,b,c
end
