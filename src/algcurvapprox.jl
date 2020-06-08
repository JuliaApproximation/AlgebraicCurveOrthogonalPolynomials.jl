"""
parameterise A and Bʸ
"""
function comunroll(V, λa)
    n = (-3 + isqrt(9 + 8length(λa))) ÷ 2
    @assert n + sum(1:n) == length(λa)
    λ = λa[1:n]
    symunroll(λa[n+1:end]), V * Diagonal(λ) * inv(V)
end

cm(A,B) = A*B-B*A

"""
Create a spectral curve from parameters
"""
function speccurvemat(Aˣ::Symmetric, (λˣ, V), κʸ)
    Bˣ = V*Diagonal(λˣ)*inv(V)
    N = length(λˣ)
    @assert length(κʸ) == 3
    cond1 = (Aʸ,Bʸ) -> cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ)
    cond0 = (Aʸ,Bʸ) -> cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ)
    
    conds = function(λa)
        Aʸ, Bʸ = comunroll(V,λa)
        vec([cond1(Aʸ,Bʸ); cond0(Aʸ,Bʸ)])
    end
    K = nullspace(jacobian(conds, zeros(N+sum(1:N))))
    (Aˣ,Bˣ),comunroll(V, K * κʸ)
end
function speccurve(Aˣ, (λˣ, V), κʸ)
    (Aˣ,Bˣ),(Aʸ,Bʸ) = speccurvemat(Aˣ, (λˣ, V), κʸ)
    X = z -> Aˣ + Bˣ/z + z*Bˣ'
    Y = z -> Aʸ + Bʸ/z + z*Bʸ'
    X,Y
end

"""
Create random spectral curve
"""
randspeccurve(N) = speccurve(randn(N,N), randn(N,N), randn(3))


"""
evaluate spectral curve at grid
"""
function specgrid(X, Y)
    NN = 20
    n = size(X(1),1)
    Z = Matrix{ComplexF64}(undef,n,NN)
    for (j,θ) in enumerate(range(0,2π; length=NN))
        z = exp(θ*im)
        λ,Q = eigen(Hermitian(X(z)))
        Z[:,j] = λ  .+ im*real(diag(Q'*Y(z)*Q))
    end
    Z
end



# using ApproxFun, LinearAlgebra


# function evalmat(s, p, n)
#     M = Array{ApproxFun.prectype(s)}(length(p), n)
#     for j = 1:n
#         M[:,j] .= Fun(s,[zeros(j-1);1]).(p)
#     end
#     M
# end

# function algebraiccurve(s, p)
#     N_max = 10000
#     tol = 1E-5
#     for N = 1:N_max
#         M = evalmat(s, p, N)
#         U, σ, V = svd(M)
#         if last(σ) ≤ tol
#             return Fun(s, V[:,end])
#         end
#         @show last(σ)
#     end
#     error("Unresolved")
# end

# function algebraiccurve(p)
#     x, y = first.(p), last.(p)
#     s = Chebyshev(minimum(x)-0.1..maximum(x)+0.1) *  Chebyshev(minimum(y)-0.1..maximum(y)+0.1)
#     algebraiccurve(s, p)
# end


evalmonbasis(N, x, y) = mortar([[x^k * y^(n-k) for k=0:n] for n=0:N])
evalmonbasis(N, z) = evalmonbasis(N, reim(z)...)

function vandermonde(N, x, y)
    @assert length(x) == length(y)
    ret = Matrix{Float64}(undef, length(x), sum(1:N+1))
    for k in axes(ret,1)
        ret[k,:] .= evalmonbasis(N, x[k], y[k])
    end
    ret
end

vandermonde(N, z) = vandermonde(N, real(z), imag(z))

"""
Gives coefficients in monomial basis of polynomial vanishing
on curve
"""
function spec2alg(X, Y)
    N = size(X(1),1)
    Z = vec(specgrid(X,Y))
    K = nullspace(vandermonde(N, Z))
    @assert size(K,2) == 1
    vec(K)
end
