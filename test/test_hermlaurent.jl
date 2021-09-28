using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, QuasiArrays, Test
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
    

        # simplify
        _,Q = eigen(Bˣ[1:2,1:2]); Q=[Q zeros(2,2); zeros(2,2) Q]; 
        Bˣ = Q'Bˣ*Q; Bʸ = Q'Bʸ*Q;
        X = HermLaurent(zeros(4,4), Bˣ)
        Y = HermLaurent(zeros(4,4), Bʸ)
        @test checkcommutes(X, Y)
        @test norm((I - X.^2) .* (I - Y.^2)) ≤ 10eps()

        # explicit simple
        Bˣ = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
        Bʸ = [0 0 -1 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0.5]
        X = HermLaurent(zeros(4,4), Bˣ)
        Y = HermLaurent(zeros(4,4), Bʸ)
        @test checkcommutes(X, Y)
        @test norm((I - X.^2) .* (I - Y.^2)) ≤ 10eps()
    end
end