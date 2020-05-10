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



n = 3
V = randn(n,n)
Λˣ = Diagonal(randn(n))
Bˣ = V * Λˣ * inv(V)

function f(λ)
    Bʸ = V * Diagonal(λ) * inv(V)
    Bˣ * Bʸ' + Bˣ' * Bʸ - Bʸ * Bˣ' - Bʸ' * Bˣ
end

J = jacobian(f,[0.1,0.2,0.3])
K = nullspace(J)
Bʸ = V * Diagonal(K*randn(2)) * inv(V)


x = z -> Bˣ/z + z*Bˣ'
y = z -> Bʸ/z + z*Bʸ'

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

λ,Q = eigen(Hermitian(X(z)))
λ .+ im*real(diag(Q'Y(z)*Q))

NN = 20; Z = Matrix{ComplexF64}(undef,n,NN)
for (j,θ) in enumerate(range(0,2π; length=NN))
    z = exp(θ*im)
    λ,Q = eigen(x(z))
    Z[:,j] = λ  .+ im*real(diag(Q'*y(z)*Q))
end
scatter(vec(Z))



n = 6
V = randn(n,n)
Λˣ = Diagonal(randn(n))
Bˣ = V * Λˣ * inv(V)

function f(λ)
    Bʸ = V * Diagonal(λ) * inv(V)
    Bˣ * Bʸ' + Bˣ' * Bʸ - Bʸ * Bˣ' - Bʸ' * Bˣ
end

J = jacobian(f,zeros(n))
K = nullspace(J)
Bʸ = V * Diagonal(K*randn(2)) * inv(V)


x = z -> Bˣ/z + z*Bˣ'
y = z -> Bʸ/z + z*Bʸ'

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

λ,Q = eigen(Hermitian(X(z)))
λ .+ im*real(diag(Q'Y(z)*Q))

NN = 20; Z = Matrix{ComplexF64}(undef,n,NN)
for (j,θ) in enumerate(range(0,2π; length=NN))
    z = exp(θ*im)
    λ,Q = eigen(x(z))
    Z[:,j] = λ  .+ im*real(diag(Q'*y(z)*Q))
end
scatter(vec(Z))