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
        @test abs(u' * Diagonal(w̃) * v) ≤ 100eps()
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

            x,y, w = gausssquare(4)
            P̃ = P[SVector.(x,y),Block.(1:4)]
            M = P̃' * Diagonal(w) * P̃
            @test M ≈ Diagonal(M)

            F = plan_squaretransform(5)
            x,y,w = gausssquare(6)
            @test F * P[SVector.(x,y),Block.(1:5)] ≈ I
        end

        @testset "jacobi" begin
            n = 10
            F = plan_squaretransform(n)
            x,y,w = gausssquare(n+1)
            P̃ = P[SVector.(x,y),Block.(1:n)]
            X = F * (x .* P̃)
            Y = F * (y .* P̃)
            M = P̃' * Diagonal(w) * P̃
            Q = P̃ * inv(sqrt(M))
            @test Q' * Diagonal(w) * Q ≈ I
            X = sqrt(M)*X*inv(sqrt(M))
            Y = sqrt(M)*Y*inv(sqrt(M))
            @test X ≈ X'
            @test Y ≈ Y'            
        end
    end

    @testset "symbols" begin
        @testset "8x8" begin
            Z = zeros(4,4)
            B₁ˣ = [3/4 1/4 0 0; 1/4 3/4 0 0; 0 0 1/4 -1/4; 0 0 -1/4 1/4]
            B₂ˣ = [1/4 -1/4 0 0; -1/4 1/4 0 0; 0 0 3/4 1/4; 0 0 1/4 3/4]
            Bˣ  = [Z Z;   B₁ˣ  Z]
            Aˣ  = [Z B₂ˣ; B₂ˣ' Z]
                
            B₁ʸ = [0 0 3/4 -1/4; 0 0 -1/4 3/4; 1/4 1/4 0 0; 1/4 1/4 0 0]
            B₂ʸ = B₁ʸ
            Bʸ  = [Z Z;   B₁ʸ  Z]
            Aʸ  = [Z B₂ʸ; B₂ʸ' Z]

            x = z -> Bˣ/z + Aˣ + Bˣ'*z
            y = z -> Bʸ/z + Aʸ + Bʸ'*z

            z = exp(0.1im)
            @test x(z)y(z) ≈ y(z)x(z)
            @test norm((I-x(z)^2)*(I-y(z)^2)) ≤ 10eps()

            P = Matrix(I,8,8)[:,[1:4; 7; 8; 5; 6]]
            P*Aˣ*P'
            P*Bˣ*P'
            P*Aʸ*P'
            P*Bʸ*P'
        end
        # @testset "4x4" begin
        #     Bˣ = [3/4 1/4 0 0; 1/4 3/4 0 0; 0 0 1/4 -1/4; 0 0 -1/4 1/4]
        #     Bʸ = [0 0 3/4 -1/4; 0 0 -1/4 3/4; 1/4 1/4 0 0; 1/4 1/4 0 0]
        #     x = z -> Bˣ/z + Bˣ'*z
        #     y = z -> Bʸ/z + Bʸ'*z

        #     z = exp(0.1im)
        #     @test x(z)y(z) ≈ y(z)x(z)
        #     @test norm((I-x(z)^2)*(I-y(z)^2)) ≤ 10eps()
        # end
    end
end

