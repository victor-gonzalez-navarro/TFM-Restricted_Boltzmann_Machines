clearconsole();

using MAT
using Distributed
using Match
using ProgressMeter
using Statistics

function myLogSumExp(logX)

  max_logX = maximum(logX);
  return (max_logX + log(sum(exp.(logX .- max_logX))));

end;

function generateWeightsRows(typeUnits,Nv,Nh,Ak,b,c)
  ### Computation of W
  W = ones(Nv+1,Nh+1);
  for kv in 1:Nv
    W[kv+1,:] .= Ak[kv];
  end
  W[1,2:end] .= c;
  W[2:end,1]  = b;
  W[1,1] = 0;

  ###
  ### Computation of logZ
  ###
  ### Neurons in {0,1}
  ###
  ###   Z = sum Nh  ( Nh ) exp(c*kh) prod Nv  (1 + exp(alpha_i * kh + b))
  ###          kh=0 ( kh )                i=1
  ###
  ### Neurons in {-1,+1}
  ###
  ###   Z = sum Nh  ( Nh ) exp(c*kh) prod Nv  ( exp(-alpha_i * (Nh-2*kh) - b) + exp(+alpha_i * (Nh-2*kh) + b) )
  ###          kh=0 ( kh )                i=1
  ###
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

  ### Sanity check - exact computation
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

  ##########################
  logZ = myLogSumExp(logZv)
  println("Enrique: logZ = ",logZ)
  #println(log(Z))

  return W;
end;

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

function tobin(num)
  # Convert decimal to binary numbers
  @match num begin
    0 => "0"
    1 => "1"
    _ => string(tobin(div(num,2)), mod(num, 2))
  end
end
