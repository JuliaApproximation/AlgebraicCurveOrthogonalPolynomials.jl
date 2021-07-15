using AlgebraicCurveOrthogonalPolynomials, MultivariateOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials, ContinuumArrays, LinearAlgebra, ForwardDiff, InfiniteArrays, Test
import AlgebraicCurveOrthogonalPolynomials: zernikeannulusr, complexzernikeannulusz, UnitInterval, ModalInterlace
using LazyArrays
import ForwardDiff: derivative, hessian, gradient
import SemiclassicalOrthogonalPolynomials: HalfWeighted, divmul


@testset "Annulus" begin
    @testset "Complex calculus" begin
        ρ = 0.5; t = inv(1-ρ^2)
        g = τ -> exp(τ*cos(τ))
        m = 2
        f = s -> g((1-s)/(1-ρ^2))
        v = r -> f(r^2)
        h = r -> r^m * v(r)


        F = (r,θ) -> exp(im*m*θ) * h(r)
        xy = SVector(0.5,0.1)
        (x,y) = xy; (r,θ) = (norm(xy),atan(y,x));


        Δ = tr(hessian(xy -> ((x,y) = xy; (r,θ) = (norm(xy),atan(y,x)); real(F(r,θ))), xy)) +
                im * tr(hessian(xy -> ((x,y) = xy; (r,θ) = (norm(xy),atan(y,x)); imag(F(r,θ))), xy))


        s = r^2
        τ = (1-s)/(1-ρ^2)

        @test s ≈ 1-τ/t

        h_r = derivative(h,r)
        h_rr = derivative(r -> derivative(h,r), r)
        v_r = derivative(v,r)
        v_rr = derivative(r -> derivative(v,r), r)
        f_s = derivative(f, s)
        f_ss = derivative(s -> derivative(f,s), s)
        g_τ = derivative(g, τ)
        g_ττ = derivative(τ -> derivative(g,τ), τ)

        @test h_r ≈ m*r^(m-1)*v(r) + r^m * v_r
        @test h_rr ≈ m*(m-1)*r^(m-2)*v(r) + 2m*r^(m-1)*v_r + r^m * v_rr

        @test v_r ≈ 2r*f_s
        @test v_rr ≈ 2f_s + 4r^2*f_ss

        @test f_s ≈ -t*g_τ
        @test f_ss ≈ t^2*g_ττ

        @test Δ ≈ exp(im*m*θ) * (h_rr + r \ h_r - m^2 * (r^2 \ h(r))) ≈
                    exp(im*m*θ) * r^m * ((2m+1)*r^(-1)*v_r + v_rr) ≈
                    4exp(im*m*θ) * r^m * (s*f_ss + (1+m)*f_s) ≈
                    4t*exp(im*m*θ) * r^m * ((t-τ)*g_ττ - (1+m)*g_τ) ≈
                    4t*exp(im*m*θ) * r^m * (t-τ)^(-m) * derivative(τ -> (t-τ)^(m+1) * derivative(g, τ), τ)
    end

    @testset "calculus" begin
        ρ = 0.5; t = inv(1-ρ^2)
        g = τ -> exp(τ*cos(τ))
        m = 2
        f = s -> g((1-s)/(1-ρ^2))
        v = r -> f(r^2)
        h = r -> r^m * v(r)


        F = (r,θ) -> cos(m*θ) * h(r)
        xy = SVector(0.5,0.1)
        (x,y) = xy; (r,θ) = (norm(xy),atan(y,x));


        Δ = tr(hessian(xy -> ((x,y) = xy; (r,θ) = (norm(xy),atan(y,x)); F(r,θ)), xy))

        s = r^2
        τ = (1-s)/(1-ρ^2)

        @test s ≈ 1-τ/t

        h_r = derivative(h,r)
        h_rr = derivative(r -> derivative(h,r), r)
        v_r = derivative(v,r)
        v_rr = derivative(r -> derivative(v,r), r)
        f_s = derivative(f, s)
        f_ss = derivative(s -> derivative(f,s), s)
        g_τ = derivative(g, τ)
        g_ττ = derivative(τ -> derivative(g,τ), τ)

        @test h_r ≈ m*r^(m-1)*v(r) + r^m * v_r
        @test h_rr ≈ m*(m-1)*r^(m-2)*v(r) + 2m*r^(m-1)*v_r + r^m * v_rr

        @test v_r ≈ 2r*f_s
        @test v_rr ≈ 2f_s + 4r^2*f_ss

        @test f_s ≈ -t*g_τ
        @test f_ss ≈ t^2*g_ττ

        @test Δ ≈ cos(m*θ) * (h_rr + r \ h_r - m^2 * (r^2 \ h(r))) ≈
                    cos(m*θ) * r^m * ((2m+1)*r^(-1)*v_r + v_rr) ≈
                    4cos(m*θ) * r^m * (s*f_ss + (1+m)*f_s) ≈
                    4t*cos(m*θ) * r^m * ((t-τ)*g_ττ - (1+m)*g_τ) ≈
                    4t*cos(m*θ) * r^m * (t-τ)^(-m) * derivative(τ -> (t-τ)^(m+1) * derivative(g, τ), τ)
    end

    @testset "Real" begin
        ρ  = 0.5; t = inv(1-ρ^2)

        @testset "Unweighted" begin
            A = ZernikeAnnulus(ρ)
            C = ZernikeAnnulus(ρ,2,2)
            Δ  = C \ (Laplacian(axes(A,1)) * A)

            xy = SVector(0.5,0.1)
            # @test tr(hessian(xy -> ZernikeAnnulus{eltype(xy)}(ρ)[xy,4], xy)) ≈ C[xy,1] * Δ[1,4]
            # @test tr(hessian(xy -> ZernikeAnnulus{eltype(xy)}(ρ)[xy,7], xy)) ≈ C[xy,2] * Δ[2,7]
            # @test tr(hessian(xy -> ZernikeAnnulus{eltype(xy)}(ρ)[xy,8], xy)) ≈ C[xy,3] * Δ[3,8]
            # @test tr(hessian(xy -> ZernikeAnnulus{eltype(xy)}(ρ)[xy,11], xy)) ≈ C[xy,1:4]' * Δ[1:4,11]
            c = randn(20)
            @test tr(hessian(xy -> ZernikeAnnulus{eltype(xy)}(ρ)[xy,1:20]'*c, xy)) ≈ (C * (Δ * [c; zeros(∞)]))[xy]
        end

        @testset "Weighted" begin
            P = ZernikeAnnulus(ρ,1,1)
            W = Weighted(P)
            Δ  = P \ (Laplacian(axes(P,1)) * W)

            x = Inclusion(UnitInterval())
            D = Derivative(x)
            Δ = P \ (Laplacian(axes(P,1)) * W)
            xy = SVector(0.5,0.1); r = norm(xy); τ = (1-r^2)/(1-ρ^2)
            @test Weighted(ZernikeAnnulus{eltype(xy)}(ρ,1,1))[xy,1] ≈ (1-r^2) * (r^2-ρ^2) * ZernikeAnnulus{eltype(xy)}(ρ,1,1)[xy,1] ≈ (1-ρ^2)^2 * HalfWeighted{:ab}(SemiclassicalJacobi.(t,1,1,0:∞)[1])[τ,1]
            @test tr(hessian(xy -> Weighted(ZernikeAnnulus{eltype(xy)}(ρ,1,1))[xy,1], xy)) ≈ P[xy,1:4]'* Δ[1:4,1]

            c = [randn(100); zeros(∞)]
            @test tr(hessian(xy -> (Weighted(ZernikeAnnulus{eltype(xy)}(ρ,1,1))*c)[xy], xy)) ≈ (P*(Δ*c))[xy]
        end
    end


    @testset "Complex" begin
        ρ  = 0.5
        A = ComplexZernikeAnnulus(ρ)
        B = ComplexZernikeAnnulus(ρ,0,1)

        R = B \ A
        @test A[SVector(0.5,0.1), Block.(1:3)]' ≈ B[SVector(0.5,0.1), Block.(1:3)]' * R[Block.(1:3),Block.(1:3)]

        D = Derivative(Inclusion(UnitInterval()))
        t = inv(1-ρ^2)
        a,b = 0,0
        D₊ = ModalInterlace(BroadcastVector{AbstractMatrix{Float64}}((B,A) -> B\(D*A), SemiclassicalJacobi.(t,b+1,a+1,1:∞), SemiclassicalJacobi.(t,b,a,0:∞)), (ℵ₀,ℵ₀), (-2,4))

        ℓ, m = 2, 0, 0, 0
        r,θ = 0.6,0.1; x,y = r*cos(θ),r*sin(θ); rθ = SVector(r,θ); xy = SVector(x,y)
        Z_x, Z_y = ForwardDiff.jacobian(xy -> SVector{2}(reim(ComplexZernikeAnnulus{eltype(xy)}(0.5)[xy,4])...), xy) |> Z_xy -> complex.(Z_xy[1,:], Z_xy[2,:])
        # Z_x, Z_y = ForwardDiff.jacobian(xy -> SVector{2}(reim(complexzernikeannulusz(ρ, ℓ, m, a, b, xy))...), xy) |> Z_xy -> complex.(Z_xy[1,:], Z_xy[2,:])

        Z_x + im*Z_y
        B[xy,1] * D₊[1,4]





        Z_r, Z_θ = ForwardDiff.jacobian(rθ -> SVector{2}(reim(complexzernikeannulusz(ρ, ℓ, m, a, b, RadialCoordinate(rθ...)))...), rθ) |> Z_rθ -> complex.(Z_rθ[1,:], Z_rθ[2,:])

        Z_r, Z_θ = complex.(ForwardDiff.gradient((rθ) -> ((r,θ) = rθ; real(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), rθ),
                    ForwardDiff.gradient((rθ) -> ((r,θ) = rθ; imag(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), rθ))

        Z_x, Z_y = complex.(ForwardDiff.gradient((xy) -> ((x,y) = xy; (r,θ) = (norm(xy),atan(y,x)); real(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), SVector(x,y)),
                    ForwardDiff.gradient((xy) -> ((x,y) = xy; (r,θ) = (norm(xy),atan(y,x)); imag(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), SVector(x,y)))

        @test exp(im*θ) * (Z_r + im/r * Z_θ) ≈ Z_x + im*Z_y


        T = Float64
        P = SemiclassicalJacobi(inv(1-ρ^2),b,a,m)
        P¹ = SemiclassicalJacobi(inv(1-ρ^2),b+1,a+1,m+1)
        x = axes(P,1)
        D = Derivative(x)

        P¹ \ (D * P)

        jr = isone((ℓ-m) ÷ 2) ? ((ℓ-m) ÷ 2) .+ (0:0) : ((ℓ-m) ÷ 2) .+ (-1:0)
        @test iszero(exp(im*θ) * (Z_r + im/r * Z_θ))

        B[xy,]
        @time D₊[1]
        -t/2 * D₊[1][1,2]


        ρ, ℓ, m, a, b = 0.5, 2, 0, 0, 0
        ModalInterlace

        T = Float64
        P = SemiclassicalJacobi(inv(1-ρ^2),b,a,m)
        P¹ = SemiclassicalJacobi(inv(1-ρ^2),b+1,a+1,m+1)
        x = axes(P,1)
        D = Derivative(x)
        D₊ = P¹ \ (D * P)

        jr = isone((ℓ-m) ÷ 2) ? ((ℓ-m) ÷ 2) .+ (0:0) : ((ℓ-m) ÷ 2) .+ (-1:0)
        @test Z_r + im/r * Z_θ ≈ -2t * exp(im*m*θ) * sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^(m+1) * (D*P)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1] ≈
                -2t * exp(-im*θ) * exp(im*(m+1)*θ) * sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^(m+1) * P¹[(r^2 - 1)/(ρ^2 - 1), jr]' * D₊[jr,(ℓ-m) ÷ 2 + 1] ≈
                -t/2 * exp(-im*θ) * exp(im*(m+1)*θ) * zernikeannulusr(ρ, ℓ-1, m+1, a+1, b+1, r) * D₊[jr[1],(ℓ-m) ÷ 2 + 1]

        ρ, ℓ, m, a, b = 0.5, 7, 3, 0, 0
        ρ, ℓ, m, a, b = 0.5, 10, 6, 0, 0

        r,θ = 0.6,0.1
        Z_r, Z_θ = complex.(ForwardDiff.gradient((rθ) -> ((r,θ) = rθ; real(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), SVector(r,θ)),
                            ForwardDiff.gradient((rθ) -> ((r,θ) = rθ; imag(zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ))), SVector(r,θ)))

        T = Float64
        t = inv(1-ρ^2)
        P = SemiclassicalJacobi(t,b,a,m)
        P¹ = SemiclassicalJacobi(t,b+1,a+1,m+1)
        x = axes(P,1)
        D = Derivative(x)
        D₊ = P¹ \ (D * P)
        c = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π)

        @test exp(im*θ) *(Z_r + im/r * Z_θ) ≈ c * exp(im*(m+1)*θ) * r^m * ForwardDiff.derivative(r -> SemiclassicalJacobi{typeof(r)}(t,b,a,m)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1] , r) ≈
                                -2t*c * exp(im*(m+1)*θ) * r^(m+1) * ForwardDiff.derivative(τ -> SemiclassicalJacobi{typeof(τ)}(t,b,a,m)[τ, (ℓ-m) ÷ 2 + 1] , (r^2 - 1)/(ρ^2 - 1))


        jr = isone((ℓ-m) ÷ 2) ? ((ℓ-m) ÷ 2) .+ (0:0) : ((ℓ-m) ÷ 2) .+ (-1:0)
        @test exp(im*θ) * (Z_r + im/r * Z_θ) ≈ -2t * exp(im*(m+1)*θ) * sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^(m+1) * (D*P)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1] ≈
                -2t * exp(im*(m+1)*θ) * sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^(m+1) * P¹[(r^2 - 1)/(ρ^2 - 1), jr]' * D₊[jr,(ℓ-m) ÷ 2 + 1] ≈
                -2/sqrt(2^3)*t * exp(im*(m+1)*θ) * sqrt(convert(T,2)^(m+1+a+1+b+1+2-iszero(m+1))/π) * r^(m+1) * P¹[(r^2 - 1)/(ρ^2 - 1), jr]' * D₊[jr,(ℓ-m) ÷ 2 + 1] ≈
                -t/sqrt(2) * exp(im*(m+1)*θ) * (zernikeannulusr(ρ, ℓ-2, m+1, a+1, b+1, r) * D₊[jr[1],(ℓ-m) ÷ 2 + 1] + zernikeannulusr(ρ, ℓ-1, m+1, a+1, b+1, r) * D₊[jr[2],(ℓ-m) ÷ 2 + 1])

    end

end