using OrthogonalPolynomialsAlgebraicCurves, BlockBandedMatrices, BlockArrays, FillArrays, Test

@testset "Quartic" begin
    @testset "Commutating properties" begin
        X,Y = quarticjacobi(10)
        @test (X*Y)[Block.(1:11), Block.(1:11)] ≈ (Y*X)[Block.(1:11), Block.(1:11)]
        @test (X^4 + Y^4)[Block.(1:10), Block.(1:10)] ≈ Eye(34)
    end

    @testset "Toeplitz SVD" begin
        X,Y = quarticjacobi(30)
        K = 25; σ1 = svdvals(Float64.(X[Block(K,K+1)]))
        K = 26; σ2 = svdvals(Float64.(X[Block(K,K+1)]))
        @test σ1 ≈ σ2 rtol=1E-2

        K = 25; σ1 = svdvals(Float64.(Y[Block(K,K+1)]))
        K = 26; σ2 = svdvals(Float64.(Y[Block(K,K+1)]))
        @test σ1 ≈ σ2 rtol=1E-2
    end
end
