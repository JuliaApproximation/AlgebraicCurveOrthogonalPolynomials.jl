using ToeplitzMatrices, FillArrays

# This function attempts to compute a Symmetric Toeplitz matrix with
# eigenvalues given by eigs. The algorithm is based on integrating Chu's
# ODE on the isospectral manifold. 
# - h is a stepsize parameter
# - maxiter is the maximum number of timesteps 
# - tol is the tolerance for the Toeplitz error, as measured by 
# norm(PZ(Y),'fro')/n^2, where PZ is the orthogonal projection onto 
# orthogonal complement of Toeplitz matrices
# 
# Written by Marcus Webb. Last edited 22 May 2020
# Ported to Julia by Sheehan Olver

# Used to make the initial matrix
function _makeK(N)
    p = (N+1) ÷ 2
    q = N-p
    E = Hankel([zeros(q-1); 1], [1; zeros(q-1)])
    if p == q
        [Matrix(I,q,q) E; E -Matrix(I,q,q)]/sqrt(2)
    else
        [Matrix(I,q,q) zeros(q,1) E; 
             zeros(1,q) sqrt(2) zeros(1,q);
             E zeros(q,1) -Matrix(I,q,q)]/sqrt(2)
    end
end

# Toepltiz annihilator
function _toep_anni(Y)
    n = size(Y,1)
    B = similar(Y, n, n)
    for l = 1:n
        for k = 1:l-1
            B[k,l] = Y[k,l-1] - Y[k+1,l]
            B[l,k] = Y[l,k+1] - Y[l-1,k]
        end
        B[l,l] = 0
    end
    B
end


# Orthogonal projection onto orthogonal complement of Toeplitz matrices
function _ortho_toep(X)
    N = size(X,1)
    t = X[1,:]*0;
    for i = 1:N
        for j = 1:N-i+1
            t[i] = t[i] + X[j,j+i-1];
        end
        t[i] = t[i]/(N-i+1)
    end
    X - SymmetricToeplitz(t)
end

function invtoepeigen(eigs; h=0.1, maxiter=100_000, tol=1E-14)
    n = length(eigs)
    p = (n+1) ÷ 2
    q = n-p
    seigs = sort(eigs)
    if p > q
        eveneigs = seigs[1:2:n]
        oddeigs = seigs[2:2:n-1]
    else
        eveneigs = seigs[2:2:n]
        oddeigs = seigs[1:2:n-1]
    end

    K = _makeK(n)
    _invtoepeigen(K*[diagm(eveneigs) zeros(p,q); zeros(q,p) diagm(reverse(oddeigs))]*K, h, maxiter, tol)
end

function _invtoepeigen(Y, h, maxiter, tol)
    n = size(Y,1)
    ToeErr = norm(_ortho_toep(Y))/n^2
    for j = 1:maxiter
        Q = (Matrix(I,n,n) - h*_toep_anni(Y)) \ ( Matrix(I,n,n) + h*_toep_anni(Y))
        Y = Q'*Y*Q
        ToeErr = norm(_ortho_toep(Y))/n^2
        if ToeErr < tol
            return Y
        end
    end
    @warn "Not converged. Final Toeplitz Error = $ToeErr"
    Y
end