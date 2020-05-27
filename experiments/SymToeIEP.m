function Y = SymToeIEP(eigs,h,maxiter,tol)
% This function attempts to compute a Symmetric Toeplitz matrix with
% eigenvalues given by eigs. The algorithm is based on integrating Chu's
% ODE on the isospectral manifold. 
% - h is a stepsize parameter
% - maxiter is the maximum number of timesteps 
% - tol is the tolerance for the Toeplitz error, as measured by 
% norm(PZ(Y),'fro')/n^2, where PZ is the orthogonal projection onto 
% orthogonal complement of Toeplitz matrices
%
% Written by Marcus Webb. Last edited 22 May 2020

n = length(eigs);
p = ceil(n/2);
q = n-p;
seigs = sort(eigs);
if p > q
    eveneigs = seigs(1:2:n);
    oddeigs = seigs(2:2:n-1);
else
    eveneigs = seigs(2:2:n);
    oddeigs = seigs(1:2:n-1);
end

K = makeK(n);
Y = K*[diag(eveneigs) zeros(p,q); zeros(q,p) diag(flip(oddeigs))]*K;


for j = 1:maxiter
    Q = (eye(n) - h*TA(Y)) \ ( eye(n) + h*TA(Y));
    Y = Q'*Y*Q;
    ToeErr = norm(PZ(Y),'fro')/n^2; 
    if ToeErr < tol
        break
    end
end

if ToeErr >= tol
    fprintf('WARNING: not converged. Final Toeplitz Error = %f \n', ToeErr);
end
end

% Used to make the initial matrix
function K = makeK(N)
  p = ceil(N/2);
  q = N-p;
  E = hankel([zeros(1,q-1),1]);
  if p==q
      K = [eye(q),E; E, -eye(q)]/sqrt(2);
  else
      K = [eye(q),zeros(q,1),E;zeros(1,q),sqrt(2),zeros(1,q);E,zeros(q,1),-eye(q)]/sqrt(2);
  end
end

% Toepltiz annihilator
function B = TA(Y)
    n = size(Y,1);
    for l = (1:n)
        for k = (1:l-1)
            B(k,l) = Y(k,l-1) - Y(k+1,l);
            B(l,k) = Y(l,k+1) - Y(l-1,k);
        end
        B(l,l) = 0;
    end
end

% Orthogonal projection onto orthogonal complement of Toeplitz matrices
function Z = PZ(X)
    N = size(X,1);
    t = X(1,:)*0;
    for i = 1:N
        for j = 1:N-i+1
            t(i) = t(i) + X(j,j+i-1);
        end
        t(i) = t(i)/(N-i+1);
    end
    Z = X - toeplitz(t);
end