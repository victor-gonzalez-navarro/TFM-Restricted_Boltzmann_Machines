clearconsole()
function myLogSumExp(logX)

  max_logX = maximum(logX);
  return (max_logX + log(sum(exp.(logX .- max_logX))));

end;

function generateWeightsColumns(typeUnits,Nv,Nh,Ak,b,c)
  ### Computation of W
  W = ones(Nv+1,Nh+1);

  for kh in 1:Nh
    W[:,kh+1] .= Ak[kh];
  end

  W[1,2:end]  = c;
  W[2:end,1] .= b;
  W[1,1] = 0;

  ###
  ### Computation of logZ
  ###
  ### Neurons in {0,1}
  ###
  ###   Z = sum Nv  ( Nv ) exp(b*kv) prod Nh  (1 + exp(alpha_j * kv + c))
  ###          kv=0 ( kv )                j=1
  ###
  ### Neurons in {-1,+1}
  ###
  ###   Z = sum Nv  ( Nv ) exp(b*kv) prod Nh  ( exp(-alpha_j * (Nv-2*kv) - c) + exp(+alpha_j * (Nv-2*kv) + c) )
  ###          kv=0 ( kv )                j=1
  ###
  logZv = Float64[];
  n_kv = 0;
  for kv in 0:Nv
    kv == 0  ?  n_kv = 1.0  :  n_kv = n_kv * (Nv+1-kv) / kv;
    sumf = 0;
    for kh in 1:Nh
      if typeUnits == 0
        p = Ak[kh]*kv + c[kh];
        p > 30       ?  sumf += p  :  sumf += log(1+exp(p));
      else
        p = Ak[kh]*(Nv-2*kv) + c[kh];
        abs(p) > 30  ?  sumf += p  :  sumf += log(exp(-p)+exp(+p));
      end;
    end;
    if typeUnits == 0
      push!(logZv,log(n_kv) + sumf + b*kv);
    else
      push!(logZv,log(n_kv) + sumf + b*(Nv-2*kv));
    end;
  end;


  ### Sanity check - exact computation
  if Nv <= 10 && Nh <= 10   ### it can be improved...
    bb = b * ones(Nv);
    cc = c;
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
  println("logZ = ",logZ)
  #println(log(Z))

  return W;
end;
