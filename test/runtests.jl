using OrthogonalPolynomialsAlgebraicCurves, LinearAlgebra, BlockBandedMatrices, BlockArrays, FillArrays, Test

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
    
    @testset "32-by-32 symbols" begin
        periods = 5
        n=8*periods+7
        X,Y=quarticjacobi(n)
        Ax = BlockBandedMatrix(zeros(BigFloat,32,32), fill(4,8),fill(4,8), (1,1))
        for b = 0:6
            Ax[Block(b+1,b+2)]=X[Block(n-6+b,n-5+b)]
            Ax[Block(b+2,b+1)]=X[Block(n-5+b,n-6+b)]
        end
        Bx=BlockArray{BigFloat}(zeros(BigFloat,32,32), fill(4,8), fill(4,8))
        Bx[Block(8,1)]=X[Block(n+1,n+2)]
        Ay = BlockBandedMatrix(zeros(BigFloat,32,32), fill(4,8),fill(4,8), (1,1))
        for b = 0:6
            Ay[Block(b+1,b+2)]=Y[Block(n-6+b,n-5+b)]
            Ay[Block(b+2,b+1)]=Y[Block(n-5+b,n-6+b)]
        end
        By=BlockArray{BigFloat}(zeros(BigFloat,32,32), fill(4,8), fill(4,8))
        By[Block(8,1)]=Y[Block(n+1,n+2)]
        θ=pi/2;
        z=exp(im*θ)
        xz = Bx'/z+Ax+Bx*z
        yz = By'/z+Ay+By*z
        @test xz*yz ≈ yz*xz rtol = 1E-2
        @test xz^4+yz^4≈I rtol=1E-2
    end
end
