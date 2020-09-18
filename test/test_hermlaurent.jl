using OrthogonalPolynomialsAlgebraicCurves, StaticArrays, Test

@testset "HermLaurent" begin
    @testset "UniformScaling" begin
        X = HermLaurent((A = randn(2,2); A+A'),randn(2,2))
        z = exp(0.1im)
        @test (X + I)[z] ≈ X[z] + I
        @test (I + X)[z] ≈ I + X[z]
        @test (X - I)[z] ≈ X[z] - I
        @test (I - X)[z] ≈ I - X[z]
    end

    @testset "quadratic" begin
        Aˣ = Matrix(0.5I,2,2)
        Bˣ = Matrix(0.25I,2,2)
        a₁₂ = (1 + sqrt(2))/4
        a₂₁ = (1 - sqrt(2))/4
        Aʸ = [0 -0.5; -0.5 0]
        Bʸ = [0 a₁₂; a₂₁ 0]

        X = HermLaurent(Aˣ,Bˣ)
        Y = HermLaurent(Aʸ,Bʸ)
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
    @testset "cubic" begin
        v = 1/(4*sqrt(2))
        Aˣ = [0 0.5 0; 0.5 0 0; 0 0 0]
        Bˣ = [0 0 -0.5; 0 0 0; 0 -0.5 0]
        Aʸ = [0 0 -v; 0 0 -v; -v -v 0]
        Bʸ = [-v v 0; 0 -v 0; 0 0 -v]
        X = HermLaurent(Aˣ,Bˣ)
        Y = HermLaurent(Aʸ,Bʸ)
    
        X² = X.^2
        X³ = X.^3
        X⁴ = X.^4
        @test X²[z] ≈ X[z]^2
        @test X³[z] ≈ X[z]^3
        @test X⁴[z] ≈ X[z]^4

        @test X .* Y ≈ Y .* X
        @test Y.^2 ≈ ((I - X).^2 .* (I + X))/4
    end

    @testset "square" begin
        Bˣ = [-1/4 1/4 1/2 1/2; 1/4 -1/4 1/2 1/2; 0 0 -1/4 1/4; 0 0 1/4 -1/4]
        Bʸ = [1/4 1/4 -1/2 1/2; 1/4 1/4 1/2 -1/2; 0 0 1/4 1/4; 0 0 1/4 1/4]
        X = HermLaurent(zero.(Bˣ), Bˣ)
        Y = HermLaurent(zero.(Bʸ), Bʸ)
        @test X .* Y ≈ Y .* X
        @test (I - X.^2) .* (I - Y.^2) ≈ 0I
    end
end