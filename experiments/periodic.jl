using BandedMatrices, Plots

##
# Periodic Toeplitz
###

n = 100
J = Symmetric(BandedMatrix(0 => (-1) .^ (0:n-1), 1 => ones(n-1)))

L = (m,z) -> [[Matrix(I,2,2) zeros(2,m-2)]; (J-z*I)[2:m-1, 1:m]]
M = z -> inv(L(4,z))[end-1:end,1:2]
z = 1.5;
let m = 4
    
end
z
Δ = z -> tr(M(z))/2

1 - Δ(1.5)^2

m = 10; inv()

det(M)
eigvals(M)
eigvals(M)

BandedMatrix(J - 1.5I) \ [1; zeros(n-1)]
scatter(eigvals(J), zeros(size(J,1)))


a = vcat(fill(randn(3),n)...)
b = vcat(fill(randn(3),n)...)

J = Symmetric(BandedMatrix(0 => a, 1 => b[1:end-1]))

scatter(eigvals(J), zeros(size(J,1)))

# symbol is 
# cos(θ) + z + inv(z)

a = (θ,z) -> cos(θ) + z + 1/z

θ = range(0,2π; length=100)
z = @. exp(im*θ)
scatter(real(a.(θ, z)))
