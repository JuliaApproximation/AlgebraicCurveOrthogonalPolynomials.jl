using AlgebraicCurveOrthogonalPolynomials, LinearAlgebra, Test


@testset "Annulus" begin
    A = ZernikeAnnulus(0.5)
    B = ZernikeAnnulus(0.5,0,1)

    R = B \ A
    @test A[SVector(0.5,0.1), Block.(1:3)]' ≈ B[SVector(0.5,0.1), Block.(1:3)]' * R[Block.(1:3),Block.(1:3)]
end