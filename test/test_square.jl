using OrthogonalPolynomialsAlgebraicCurves, OrthogonalPolynomialsQuasi, FastGaussQuadrature, StaticArrays, BlockArrays, Test

@testset "Square" begin
    @testset "quad" begin
        x,y, w = gausssquare(2)
        @test dot(w,ones(length(x))) ≈ 8
        @test dot(w,x) ≈ 0
        @test dot(w,y) ≈ 0
        @test dot(w,x.^2) ≈ 16/3
        @test dot(w,x.*y) ≈ 0
        @test dot(w,y.^2) ≈ 16/3
        @test abs(dot(w,x.^3)) ≤ 10eps()
        @test abs(dot(w,x.^2 .* y)) ≤ 10eps()
        @test abs(dot(w,x .* y.^2)) ≤ 10eps()
        @test abs(dot(w,y.^3)) ≤ 10eps()

        @test dot(w,wedgep.(1,-1/2,-1/2,0,x.^2,y.^2)) ≈ 0
        @test dot(w,wedgeq.(1,-1/2,-1/2,0,x.^2,y.^2)) ≈ 0

        x,y, w = gausssquare(3)
        @test dot(w,ones(length(x))) ≈ 8
        @test abs(dot(w,x)) ≤ 10eps()
        @test abs(dot(w,y)) ≤ 10eps()
        @test dot(w,x.^2) ≈ 16/3
        @test abs(dot(w,x.*y)) ≤ 10eps()
        @test dot(w,y.^2) ≈ 16/3
        @test abs(dot(w,x.^3)) ≤ 10eps()
        @test abs(dot(w,x.^2 .* y)) ≤ 10eps()
        @test abs(dot(w,x .* y.^2)) ≤ 10eps()
        @test abs(dot(w,y.^3)) ≤ 10eps()
        @test dot(w,x.^4) ≈ 24/5
        @test abs(dot(w,x.^3 .* y)) ≤ 10eps()
        @test dot(w,x.^2 .* y.^2) ≈ 8/3
        @test abs(dot(w,x .* y .^ 3)) ≤ 10eps()
        @test dot(w,y.^4) ≈ 24/5

        x,y, w = gausssquare(10)
        v = x .* wedgep.(1,1/2,-1/2,0,x.^2,y.^2)
        u = x .* wedger.(1,1/2,-1/2,0,x.^2,y.^2,3)
        @test abs(dot(u,Diagonal(w),v)) ≤ 10eps()
        v = y .* wedgep.(1,-1/2,1/2,0,x.^2,y.^2)
        u = y .* wedger.(1,-1/2,1/2,0,x.^2,y.^2,1/3)
        @test abs(dot(u,Diagonal(w),v)) ≤ 10eps()


        x̃, ỹ, w̃ = gausswedge(5,-1/2,-1/2,0)

        @test sum(w) ≈ 4sum(w̃)
        @test dot(w,x.^2) ≈ dot(w̃,x̃) + 2sum(w̃)
        x̃, ỹ, w̃ = gausswedge(5,1/2,-1/2,0)

        v = wedgep.(1,1/2,-1/2,0,x̃,ỹ)
        u = wedger.(1,1/2,-1/2,0,x̃,ỹ)
        dot(u,Diagonal(w̃),v)

        @test abs(u' * Diagonal(w) * v) ≤ eps()

    end

    @testset "LegendreSquare" begin
        P = LegendreSquare()

        @testset "Indexing" begin
            @test P[SVector(0.1,1),Block(1)] == [1.0]
            @test P[SVector(0.1,1),Block(2)] == [0.1,1.0]
            @test length(P[SVector(0.1,1),Block(3)]) == 3
            @test length(P[SVector(0.1,1),Block(4)]) == 4
            @test length(P[SVector(0.1,1),Block(5)]) == 4
            @test P[SVector(0.1,1),1] == P[SVector(0.1,1),Block(1)[1]] == 1.0
        end

        @testset "transform" begin
            x,y, w = gausssquare(3)
            @test dot(w,P[SVector.(x,y),1]) ≈ 8
        
            P̃ = P[SVector.(x,y),Block.(1:3)]
            M = P̃' * Diagonal(w) * P̃
            @test M ≈ Diagonal(M)

            x,y, w = gausssquare(6)
            P̃ = P[SVector.(x,y),Block.(1:4)]
            M = P̃' * Diagonal(w) * P̃
            @test M ≈ Diagonal(M)
        end
    end
end

