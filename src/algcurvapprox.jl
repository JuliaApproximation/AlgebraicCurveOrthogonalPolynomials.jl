"""
Create random spectral curve
"""
function randspeccurve(n)
    V = randn(n,n)
    Λˣ = Diagonal(randn(n))
    Bˣ = V * Λˣ * inv(V)

    if n == 2 # commuting is enough
        Bʸ = V * Diagonal(randn(n)) * inv(V)
    else
        function f(λ)
            Bʸ = V * Diagonal(λ) * inv(V)
            Bˣ * Bʸ' + Bˣ' * Bʸ - Bʸ * Bˣ' - Bʸ' * Bˣ
        end
        
        J = jacobian(f,zeros(n))
        K = nullspace(J)
        Bʸ = V * Diagonal(K*randn(2)) * inv(V)
    end

    X = z -> Bˣ/z + z*Bˣ'
    Y = z -> Bʸ/z + z*Bʸ'

    X,Y
end


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