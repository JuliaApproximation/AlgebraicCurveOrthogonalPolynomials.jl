using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, QuasiArrays, FillArrays, Test
import ClassicalOrthogonalPolynomials: SetindexInterlace

@testset "HermLaurent" begin
    @testset "basics" begin
        F = HermLaurent{2}()
        z = exp(0.1im)
        @test F[z,Block(1)] ≈ [[1 0; 0 0],[0 1; 1 0],[0 0; 0 1]]
        @test F[z,Block(2)] ≈ [[z+inv(z) 0; 0 0],[0 z; inv(z) 0],[0 inv(z); z 0],[0 0; 0 z+inv(z)]]
        @test F[z,Block(3)] ≈ [[z^2+z^(-2) 0; 0 0],[0 z^2; z^(-2) 0],[0 z^(-2); z^2 0],[0 0; 0 z^2+z^(-2)]]

        f = F * [1:7; zeros(∞)]
        A = [1 2; 2 3]
        B = [4 6; 5 7]
        @test f[z] ≈ A + B/z + B'*z
    end

    @testset "Expansion" begin
        A = Hermitian(randn(2,2)); B = randn(2,2)
        X = z -> SHermitianCompact{2}(B/z + A + z*B')
        

        F = Fourier{ComplexF64}()
        M = SetindexInterlace(SMatrix{2,2,ComplexF64},fill(F,4)...)
        θ = axes(M,1)
        c = M \ BroadcastQuasiVector{SMatrix{2,2,ComplexF64,4}}(θ -> X(exp(im*θ)), θ)
        @test reshape(c[Block(1)],2,2) ≈ A
        @test reshape(c[Block(2)],2,2) ≈ im*(B'-B)
        @test reshape(c[Block(3)],2,2) ≈ B'+B
        @test B ≈ (im*reshape(c[Block(2)],2,2) + reshape(c[Block(3)],2,2))/2

        L = HermLaurent{2}()
        z = axes(L,1)
        c = L \ BroadcastQuasiVector{SHermitianCompact{2,ComplexF64,3}}(X,z)
        @test (L * c)[exp(0.1im)] ≈ X(exp(0.1im))
    end

    @testset "UniformScaling" begin
        X = HermLaurent{2}() * [randn(7); zeros(∞)]
        z = exp(0.1im)
        @test (X .+ I)[z] ≈ X[z] + I
        @test (I .+ X)[z] ≈ I + X[z]
        @test (X .- I)[z] ≈ X[z] - I
        @test (I .- X)[z] ≈ I - X[z]
    end

    @testset "quadratic" begin
        Aˣ = Matrix(0.5I,2,2)
        Bˣ = Matrix(0.25I,2,2)
        a₁₂ = (1 + sqrt(2))/4
        a₂₁ = (1 - sqrt(2))/4
        Aʸ = [0 -0.5; -0.5 0]
        Bʸ = [0 a₁₂; a₂₁ 0]

        X = hermlaurent(Aˣ,Bˣ)
        Y = hermlaurent(Aʸ,Bʸ)
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
        Y .* X - XY
        @test XY[z] ≈ X[z]Y[z]

        @test norm(X.^2 .+ Y.^2 .- I) ≤ 10eps()
    end
    @testset "cubic" begin
        v = 1/(4*sqrt(2))
        Aˣ = [0 0.5 0; 0.5 0 0; 0 0 0]
        Bˣ = [0 0 -0.5; 0 0 0; 0 -0.5 0]
        Aʸ = [0 0 -v; 0 0 -v; -v -v 0]
        Bʸ = [-v v 0; 0 -v 0; 0 0 -v]
        X = hermlaurent(Aˣ,Bˣ)
        Y = hermlaurent(Aʸ,Bʸ)
    
        X² = X.^2
        X³ = X.^3
        X⁴ = X.^4
        z = exp(0.1im)
        @test X²[z] ≈ X[z]^2
        @test X³[z] ≈ X[z]^3
        @test X⁴[z] ≈ X[z]^4

        @test X .* Y ≈ Y .* X
        @test Y.^2 ≈ ((I .- X).^2 .* (I .+ X))/4
    end

    @testset "square" begin
        Bˣ = [-1/4 1/4 1/2 1/2; 1/4 -1/4 1/2 1/2; 0 0 -1/4 1/4; 0 0 1/4 -1/4]
        Bʸ = [1/4 1/4 -1/2 1/2; 1/4 1/4 1/2 -1/2; 0 0 1/4 1/4; 0 0 1/4 1/4]
        X = hermlaurent(zero.(Bˣ), Bˣ)
        Y = hermlaurent(zero.(Bʸ), Bʸ)
        @test X .* Y ≈ Y .* X
        @test norm((I .- X.^2)[exp(0.1im)] * (I .- Y.^2)[exp(0.1im)]) ≤ 10eps()
        @test_skip norm((I .- X.^2) .* (I .- Y.^2)) ≤ 10eps()
    

        # simplify
        _,Q = eigen(Bˣ[1:2,1:2]); Q=[Q zeros(2,2); zeros(2,2) Q]; 
        Bˣ = Q'Bˣ*Q; Bʸ = Q'Bʸ*Q;
        X = hermlaurent(zeros(4,4), Bˣ)
        Y = hermlaurent(zeros(4,4), Bʸ)
        @test_skip checkcommutes(X, Y)
        @test_skip norm((I .- X.^2) .* (I .- Y.^2)) ≤ 10eps()

        # explicit simple
        Bˣ = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
        Bʸ = [0 0 -1 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0.5]
        X = hermlaurent(zeros(4,4), Bˣ)
        Y = hermlaurent(zeros(4,4), Bʸ)
        @test_skip checkcommutes(X, Y)
        @test_skip norm((I .- X.^2) .* (I .- Y.^2)) ≤ 10eps()
    end

    @testset "mean curvature flow" begin
        X = hermlaurent(Zeros(2,2), Eye(2)/2) # z -> Eye(2) * (z/2 + 1/2z)
        Y = hermlaurent(Zeros(2,2), [0 -1/2; 1/2 0]) #z -> [0 z/2-1/2z; 1/2z-z/2 0]
        
        @testset "diff" begin
            z = exp(0.1im)
            Ẋ,Ẏ = diff(X),diff(Y)
            @test Ẋ[z] ≈ Eye(2)/2 * im*(z - 1/z)
            @test Ẏ[z] ≈ im/2*[0 z+1/z; -(z+1/z) 0]

            @test (Ẋ .^ 2)[z] ≈ Ẋ[z]^2
            @test (Ẏ .^ 2)[z] ≈ Ẏ[z]^2

            T = function(z)
                Xd = Ẋ[z]
                Yd = Ẏ[z]
                N = sqrt(Xd^2 + Yd^2)
                (Xd/N, Yd/N)
            end
            ##
            # the joint eigenvectors of Q tell us how to relate T to the 2 tangent vectors
            ##
            
            x,y,Q = jointeigen(X[z],Y[z])

            @test (Q'T(z)[1]*Q) ≈ real(Diagonal(Q'T(z)[1]*Q))
            @test (Q'T(z)[2]*Q) ≈ real(Diagonal(Q'T(z)[2]*Q))

            Ẍ,Ÿ = diff(Ẋ),diff(Ẏ)
            @test Ẍ[z] ≈ -Eye(2)/2 * (z + 1/z)
            @test Ÿ[z] ≈ -1/2*[0 z-1/z; -(z-1/z) 0]

            ##
            # The curvature is defined as `norm(Ṫ)`. 
            # or equivalentally
            #
            # (ẋ*ÿ - ẏ*ẍ)/(ẋ^2 + ẏ^2)^(3/2)
            κ = function(z)
                Xd = Ẋ[z]; X2 = Ẍ[z]
                Yd = Ẏ[z]; Y2 = Ÿ[z]
                @assert Xd^2 + Yd^2 ≈ real(Xd^2 + Yd^2)
                N = Symmetric(real(Xd^2 + Yd^2))^(3/2)
                (Xd * Y2 - Yd * X2)/N
            end

            # note the direction of movement of the two points is different
            # so the curvature is opposite sign. This is balenced by the direction
            # of the normal
            @test Q'*κ(exp(0.1im))*Q ≈ Q'*κ(exp(0.5im))*Q ≈ Diagonal([-1,1])
        end
    end
end