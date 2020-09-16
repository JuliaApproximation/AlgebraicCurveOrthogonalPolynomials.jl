using OrthogonalPolynomialsAlgebraicCurves, StaticArrays, Test

@testset "HermLaurent" begin
    @testset "2x2" begin
        Ax = Matrix(0.5I,2,2)
        Bx = Matrix(0.25I,2,2)
        a₁₂ = (1 + sqrt(2))/4
        a₂₁ = (1 - sqrt(2))/4
        Ay = [0 -0.5; -0.5 0]
        By = [0 a₁₂; a₂₁ 0]

        X = HermLaurent(SVector(Ax,Bx))
        Y = HermLaurent(SVector(Ay,By))
        z = exp(0.1im)
        @test X[z]*Y[z] ≈ Y[z]*X[z]
        @test X[z]^2 + Y[z]^2 ≈ I

        X² = X.^2
        X³ = X.^3
        X⁴ = X.^4
        @test X²[z] ≈ X[z]^2
        @test X³[z] ≈ X[z]^3
        @test X⁴[z] ≈ X[z]^4

        XY = X .* Y
        @test all(XY.A .≈ (Y .* X).A)
        @test XY[z] ≈ X[z]Y[z]

        @test X.^2 .+ Y.^2 ≈ I
    end
end