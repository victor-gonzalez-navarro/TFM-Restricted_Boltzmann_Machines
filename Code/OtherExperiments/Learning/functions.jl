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

function createDataset(datatype, ReducedData, abs_path)
    if datatype == "MNIST"
        # Each line in matrix is a sample
        dataTrain_x, dataTrain_y, dataTest_x, dataTest_y = LoadData_MNIST(abs_path);

        if ReducedData > 0
          r            = randperm(size(dataTrain_x,1));
          full_data    = dataTrain_x[r[1:ReducedData],:];
          full_targets = dataTrain_y[r[1:ReducedData],:];
        end;

        x      = full_data';
        I_real = 0;
        I      = round.(x);                # Transform input into binary values
    end
    return I, I_real, full_targets
end

function computeRBMparamAll(W, b, c, I, I_outside, Nexamples, batchsize, NG, Nepochs, momentum, targdis, plotLL)
    # Compute RBM parameters either for CD and PT
    VdW_CD1   = zeros(size(W,1),size(W,2)); Vdb_CD1 = zeros(size(b,1),size(b,2)); Vdc_CD1 = zeros(size(c,1),size(c,2));

    W_CD     = copy(W); b_CD    = copy(b); c_CD    = copy(c); ratio_CD   = [0, 0]; VdW_CD    = copy(VdW_CD1); Vdb_CD    = copy(Vdb_CD1); Vdc_CD    = copy(Vdc_CD1);
    W_PTCD   = copy(W); b_PTCD  = copy(b); c_PTCD  = copy(c); ratio_PTCD  = [0, 0]; VdW_PTCD  = copy(VdW_CD1); Vdb_PTCD  = copy(Vdb_CD1); Vdc_PTCD  = copy(Vdc_CD1);

    @showprogress 1 "Epochs..." for i=1:Nepochs
      ordering = randperm(Nexamples);
      Iepoch = copy(I[:, ordering]);

      ini_sample = 1;
      end_sample = min(Nexamples, ini_sample+batchsize-1);

      while ini_sample <= Nexamples
        x_1 = copy(Iepoch[:,ini_sample:end_sample]);
        truebatchsize = size(x_1,2);

        # Find hidden units by sampling the visible layer and viceversa
        lgst_c_x1_CD   , x_2_CD    = gibbsRBM(x_1, W_CD   , b_CD   , c_CD   , NG);
        lgst_c_x1_PTCD , x_2_PTCD  = gibbsPT(x_1 , W_PTCD , b_PTCD , c_PTCD , NG, K, NG_par);

        # Compute weights of any algorithm except WCD
        probweights       = (1/truebatchsize) * ones(1,truebatchsize);

        # Computation of the CD-1 terms for x_2
        lgst_c_x2_CD    = sigmoid(c_CD    .+ W_CD    * x_2_CD);                   # lgst_c_x2 = (#h_units X batchsize)
        lgst_c_x2_PTCD  = sigmoid(c_PTCD  .+ W_PTCD  * x_2_PTCD);               # lgst_c_x2 = (#h_units X batchsize)

        # Update parameters
        VdW_CD   , Vdb_CD   , Vdc_CD   , W_CD   , b_CD   , c_CD    = update_parameters(probweights, probweights, x_1, lgst_c_x1_CD  , x_2_CD  , lgst_c_x2_CD  , momentum, VdW_CD  , Vdb_CD  , Vdc_CD  , W_CD  , b_CD  , c_CD);
        VdW_PTCD , Vdb_PTCD , Vdc_PTCD , W_PTCD , b_PTCD , c_PTCD  = update_parameters(probweights, probweights, x_1, lgst_c_x1_PTCD, x_2_PTCD, lgst_c_x2_PTCD, momentum, VdW_PTCD, Vdb_PTCD, Vdc_PTCD, W_PTCD, b_PTCD, c_PTCD);

        ini_sample = end_sample + 1;
        end_sample = min(Nexamples, ini_sample+batchsize-1);
      end
    end
    max_expdis_CD    = compute_Pxbinarydiff(I, W_CD   , b_CD   , c_CD   );
    max_dis_CDout    = compute_Pxbinarydiff(I_outside, W_CD   , b_CD   , c_CD   );
    ratio_CD         = [max_expdis_CD, max_dis_CDout];

    max_expdis_PTCD  = compute_Pxbinarydiff(I        , W_PTCD, b_PTCD, c_PTCD);
    max_dis_PTCDout  = compute_Pxbinarydiff(I_outside, W_PTCD, b_PTCD, c_PTCD);
    ratio_PTCD       = [max_expdis_PTCD, max_dis_PTCDout];
    params = Dict(
                  "W_CD"    => W_CD   , "b_CD"    => b_CD   , "c_CD"    => c_CD,
                  "W_PTCD"  => W_PTCD , "b_PTCD"  => b_PTCD , "c_PTCD"  => c_PTCD
                  );
    plotparams = Dict(
                      "ratio_CD"    => ratio_CD  ,
                      "ratio_PTCD"  => LLvec_PTCD
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

function num_power10(val)
    if val < 10
        res = 0;
    else
        res = ceil(log10(val)) - 1;
    end
    return res
end;

function list_digits(x)
    digits = Int[]  # Empty array of Ints
    while x != 0
        rem = x % 10  # Modulo division - get last digit
        push!(digits, rem)
        x = div(x,10)  # Integer division - drop last digit
    end
    return digits
end;

function num_power10_insideexp(a)
    # Computes the power of the following number: exp(a). For example, if exp(a) is 2.423e200 the output will be 200.
    if a >= 0
        if a > 700
            normalizing = a/700;
        elseif a > 0 && a < 10
            normalizing = 1;
        else
            normalizing = 10;
        end
        digits_list = list_digits(round(exp(a/normalizing)));
        if length(digits_list) >= 3
            ret = num_power10(exp(a/normalizing))*normalizing + num_power10((digits_list[end] + digits_list[end-1]/10 + digits_list[end-2]/100)^normalizing);
        elseif length(digits_list) == 2
            ret = num_power10(exp(a/normalizing))*normalizing + num_power10((digits_list[end] + digits_list[end-1]/10)^normalizing);
        elseif length(digits_list) == 1
            ret = num_power10(exp(a/normalizing))*normalizing + num_power10((digits_list[end])^normalizing);
        else
            ret = num_power10(exp(a/normalizing))*normalizing;
        end
    else
        if a < -700
            normalizing = abs(a/700);
        else
            normalizing = 1;
        end
        if occursin("e", string(exp(a/normalizing)))
            st = string(exp(a/normalizing));
            position = findfirst("e", st)[1];
            digits_list = st[1:position-1];
            ret = parse(Float64, st[position+1:end])*normalizing + num_power10(parse(Float64,digits_list[1:position-1])^normalizing);
        else
            st = string(exp(a/normalizing))
            i =3;
            while st[i] == '0'
                i = i + 1
            end
            ret = -(i-2);
        end
    end
    return round(ret)
end;

function compute_Pxbinarydiff(I, W , b , c)
    max_val = -Inf;
    for i=1:size(I,2)
        x = I[:,i];
        new_val = 0;
        for j=1:size(c,1)
            inside = W[j,:]'*x + c[j,1];
            new_val = new_val + num_power10_insideexp(W[j,:]'*x + c[j,1]);
        end
        inside = b'*x;
        new_val = new_val + num_power10_insideexp(inside[1]);
        max_val = max(max_val, new_val);
    end
    return max_val;
end;
