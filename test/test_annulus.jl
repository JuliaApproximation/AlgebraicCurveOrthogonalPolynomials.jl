using AlgebraicCurveOrthogonalPolynomials, MultivariateOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials, ContinuumArrays, LinearAlgebra, ForwardDiff, InfiniteArrays, Test
import AlgebraicCurveOrthogonalPolynomials: zernikeannulusr, complexzernikeannulusz, UnitInterval, ModalInterlace
using LazyArrays


@testset "Annulus" begin
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