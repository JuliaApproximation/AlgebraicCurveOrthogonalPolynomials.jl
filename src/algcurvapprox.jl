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
Jacobian of A*X
"""
ljac(A) = mortar(Diagonal(Fill(A,size(A,1))))

"""
Jacobian of X*A
"""
function rjac(A::AbstractMatrix{T}) where T
    N = size(A,1)
    ret = zeros(T, N^2, N^2)
    for k = 1:N
        ret[k:N:end,k:N:end] = A'
    end
    ret
end
"""
Jacobian of cm(A,X)
"""
cmjac(A) = ljac(A) - rjac(A)
cmjac(A::Symmetric) = cmjac(Matrix(A))
cmjac(A::Adjoint) = cmjac(Matrix(A))

"""
Jacobian of Diagonal(λ)
"""
function diagjac(::Type{T}, N) where T
    ret = BlockArray{T}(undef, Fill(N,N), [N])
    fill!(ret, zero(T))
    for k = 1:N
        ret[Block(k)[k],k] = 1
    end
    ret
end
diagjac(N) = diagjac(Int, N)

"""
Jacobian of symunroll(λ)
"""
function symjac(::Type{T}, N) where T
    ret = BlockArray{T}(undef, Fill(N,N), 1:N)
    fill!(ret, zero(T))
    for J = 1:N
        for k = 1:J
            ret[Block(J)[k] , Block(J)[k]] = one(T)
            ret[Block(k)[J], Block(J)[k]] = one(T)
        end
    end
    ret
end
symjac(N) = symjac(Int, N)

"""
Jacobian of transpose(A)
"""
function trjac(::Type{T}, N) where T
    ret = BlockArray{T}(undef, Fill(N,N), Fill(N,N))
    fill!(ret, zero(T))
    for J = 1:N, k = 1:N
        ret[Block(k)[J] , Block(J)[k]] = one(T)
    end
    ret
end
trjac(N) = trjac(Int, N)


function cond1jac(Aˣ, Bˣ, V)
    N = size(Aˣ, 1)
    Bʸjac = rjac(inv(V)) * ljac(V) * diagjac(N) 
    Aʸjac = symjac(N)
    AˣBʸjac = cmjac(Aˣ) * Bʸjac
    BˣAʸjac = cmjac(Bˣ) * Aʸjac
    [AˣBʸjac BˣAʸjac]
end

function cond0jac(Aˣ, Bˣ, V)
    N = size(Aˣ, 1)
    Bʸjac = rjac(inv(V)) * ljac(V) * diagjac(N) 
    Aʸjac = symjac(N)
    [(cmjac(Bˣ) * trjac(N) + cmjac(Bˣ'))*Bʸjac  cmjac(Aˣ)*Aʸjac]
end

condsjac(Aˣ, Bˣ, V) = [cond1jac(Aˣ, Bˣ, V); cond0jac(Aˣ, Bˣ, V)]


"""
special implementation for autodiff
"""
function qr_nullspace(N, A)
    Q, R = qr(A',Val(true))
    # @assert norm(R[end-N+1:end,:]) ≤ 1E-12
    Q[:,end-N+1:end]
end

function eigen_nullspace(N, A)
    λ,V = eigen(Symmetric(A'A))
    # @assert norm(λ[1:N]) ≤ tol
    V[:,1:N]
end

"""
Create a spectral curve from parameters
"""
function speccurvemat(Aˣ::Symmetric, (λˣ, V), κʸ)
    Bˣ = V*Diagonal(λˣ)*inv(V)
    N = length(λˣ)
    # @assert length(κʸ) == 3
    cond1 = (Aʸ,Bʸ) -> cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ)
    cond0 = (Aʸ,Bʸ) -> cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ)
    J = condsjac(Aˣ, Bˣ, V)
    K = nullspace(J)
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


function jointeigen(A, B)
    _,Q = eigen(A + 1.23409304233B)
    x,y = real.(diag(Q'A*Q)), real.(diag(Q'B*Q))
    x,y,Q
end

jointeigvals(A, B) = jointeigen(A, B)[1:2]



"""
evaluate spectral curve at grid
"""
function specgrid(X, Y)
    NN = 40
    n = size(X(1),1)
    Z = Matrix{ComplexF64}(undef,n,NN)
    for (j,θ) in enumerate(range(0,2π; length=NN))
        z = exp(θ*im)
        x,y = jointeigvals(X(z), Y(z))
        Z[:,j] .= complex.(x,y)
    end
    Z
end

function specgrid(X::AbstractQuasiVector, Y::AbstractQuasiVector)
    NN = 40
    n = size(X[1],1)
    Z = Matrix{ComplexF64}(undef,n,NN)
    for (j,θ) in enumerate(range(0,2π; length=NN))
        z = exp(θ*im)
        x,y = jointeigvals(X[z], Y[z])
        Z[:,j] .= complex.(x,y)
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
