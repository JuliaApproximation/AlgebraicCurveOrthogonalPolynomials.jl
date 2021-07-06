using AlgebraicCurveOrthogonalPolynomials, MultivariateOrthogonalPolynomials, LinearAlgebra, ForwardDiff, Test
import AlgebraicCurveOrthogonalPolynomials: zernikeannulusr


@testset "Annulus" begin
    A = ZernikeAnnulus(0.5)
    B = ZernikeAnnulus(0.5,0,1)

    R = B \ A
    @test A[SVector(0.5,0.1), Block.(1:3)]' ≈ B[SVector(0.5,0.1), Block.(1:3)]' * R[Block.(1:3),Block.(1:3)]

    ρ, ℓ, m, a, b = 0.5, 2, 0, 0, 0
    
    Z_r, Z_θ = complex.(ForwardDiff.gradient((rθ) -> ((r,θ) = rθ; real(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), SVector(0.5,0.1)),
                ForwardDiff.gradient((rθ) -> ((r,θ) = rθ; imag(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), SVector(0.5,0.1)))



end