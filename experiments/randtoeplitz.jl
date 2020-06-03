using BlockArrays, ForwardDiff, LinearAlgebra, Plots, Test
import ForwardDiff: jacobian

###
# explore 2 x 2 commuting Laurent polynomials,
# begining with A_x = A_y = 0, that is,
#
# X(z) = B_x/z + z*B_x'
# Y(z) = B_y/z + z*B_y'
# 
# These satisfy
# 
# X(z) Y(z) = Y(z) X(z)
# 
# i.e.
#
# 0 = (B_x * B_y - B_y * B_x)/z^2 + B_x*B_y' + B_x'*B_y - B_y*B_x' - B_y'*B_x + (B_x' * B_y' - B_y' * B_x')*z^2
#
# i.e.
#
# B_x*B_y = B_y*B_x
# B_x*B_y' + B_x'*B_y = B_y*B_x' + B_y'*B_x 
#
# Note Given B_x this is a linear constraint in B_y, hence it must lie in the kernel. 
###


##
# First: is commuting enough?
# Yes for 2 x2
##

V = randn(2,2)
Λˣ = Diagonal(randn(2))
Λʸ = Diagonal(randn(2))
Bˣ = V * Λˣ * inv(V)
Bʸ = V * Λʸ * inv(V)

@test Bˣ * Bʸ ≈ Bʸ * Bˣ
@test Bˣ * Bʸ' + Bˣ' * Bʸ ≈ Bʸ * Bˣ' + Bʸ' * Bˣ

# No for 3 x 3:
n = 3
V = randn(n,n)
Λˣ = Diagonal(randn(n))
Λʸ = Diagonal(randn(n))
Bˣ = V * Λˣ * inv(V)
Bʸ = V * Λʸ * inv(V)

@test Bˣ * Bʸ ≈ Bʸ * Bˣ
@test !(Bˣ * Bʸ' + Bˣ' * Bʸ ≈ Bʸ * Bˣ' + Bʸ' * Bˣ)

# No for 4 x 4:
n = 4
V = randn(n,n)
Λˣ = Diagonal(randn(n))
Λʸ = Diagonal(randn(n))
Bˣ = V * Λˣ * inv(V)
Bʸ = V * Λʸ * inv(V)

@test Bˣ * Bʸ ≈ Bʸ * Bˣ
@test !(Bˣ * Bʸ' + Bˣ' * Bʸ ≈ Bʸ * Bˣ' + Bʸ' * Bˣ)

# Interestingly if Bˣ is symmetric it works:
Q = qr(randn(n,n)).Q
Bˣ = Q * Λˣ * Q'
Bʸ = Q * Λʸ * Q'

@test Bˣ * Bʸ ≈ Bʸ * Bˣ
@test Bˣ * Bʸ' + Bˣ' * Bʸ ≈ Bʸ * Bˣ' + Bʸ' * Bˣ


# The question becomes: Given  Bˣ find Bʸ so that 
# Bʸ = V * Λʸ * inv(V) satisfies the second equation. 
# This is a linear constraint
# so can be found via nullspace:






@test Bˣ * Bʸ' + Bˣ' * Bʸ ≈ Bʸ * Bˣ' + Bʸ' * Bˣ


X,Y = randspeccurve(3)
z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

scatter(vec(specgrid(X, Y)))

X,Y = randspeccurve(2)

@test X(z)Y(z) ≈ Y(z)X(z)

N = 2

evalmonbasis(N, x, y) = mortar([[x^k * y^(n-k) for k=0:n] for n=0:N])

function vandermonde(N, x, y)
    @assert length(x) == length(y)
    ret = Matrix{Float64}(undef, length(x), sum(1:N+1))
    for k in axes(ret,1)
        ret[k,:] .= evalmonbasis(N, x[k], y[k])
    end
    ret
end

vandermonde(N, z) = vandermonde(N, real(z), imag(z))

N = 3; 
X,Y = randspeccurve(N)
nullspace(vandermonde(N, vec(specgrid(X,Y))))

N = 4; 
X,Y = randspeccurve(N)
nullspace(vandermonde(N, vec(specgrid(X,Y))))






λ,Q = eigen(Hermitian(X(z)))


λ .+ im*real(diag(Q'Y(z)*Q))


scatter(vec(Z))





X,Y = randspeccurve(6)
z = exp(0.1im)
@test x(z)y(z) ≈ y(z)x(z)


n
NN = 60; Z = Matrix{ComplexF64}(undef,n,NN)
for (j,θ) in enumerate(range(0,2π; length=NN))
    z = exp(θ*im)
    λ,Q = eigen(x(z))
    Z[:,j] = λ  .+ im*real(diag(Q'*y(z)*Q))
end
scatter(vec(Z))



##
# x^4 + y^4 = 1 ?
##

V = randn(n,n)
Λˣ = Diagonal(randn(n))
Bˣ = V * Λˣ * inv(V)

z = exp(0.1*im)

b = randn(n^2)
z
g = 

g = z -> function(b)
    n = isqrt(length(b))
    B = reshape(b, n, n)
    [vec(real(B/z + z*B')); vec(imag(B/z + z*B'))]
end

X = z -> Matrix(0.5I, n, n) * (z + 1/z)
g = z -> function(b)
    n = isqrt(length(b))
    B = reshape(b, n, n)
    ret = (B/z + z*B')^4 + Matrix(real(z)^4*I,n,n) - I
    [real(vec(ret)); imag(vec(ret))]
end
g(z)(randn(n^2))

Jz = jacobian(g(z), randn(n^2))


for (j,θ) in enumerate(range(0,2π; length=NN))[
    z = exp(θ*im)
    Z[:,j] = real(λ)  .+ im*real(diag(Q'*y(z)*Q))
end