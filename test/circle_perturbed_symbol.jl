using LinearAlgebra, Test

@testset "perturbed circle 6×6 symbols" begin

    # Using P = Normalized(Legendre()) and Pϕ = LanczosPolynomial(ϕ), where
    #  y² = ϕ = (1 - x²)*(1 + ε*x), computed approximate 6×6 symbols for OPs on y² = ϕ,
    # then found the following exact 6×6 symbols with symbolic computing:

    Ax = Symmetric(diagm(2 => 0.5*[1,1,1,1]))
    Bx = diagm(-4 => 0.5*[1,1])
    ε = 0.1
    ϕ = asin(ε)/2
    c0 = cos(ϕ)/2
    c1 = sin(ϕ)/2
    c2 = -c0
    c3 = -c1
    Ay = Symmetric(diagm(1 => [c1,c2,c1,c2,c1],3 => [c0,c3,c0]))
    By = diagm(-3 => [c3,c0,c3])
    By[6,1] = c2

    θ = π/4
    z = exp(im*θ)
    X = Bx'/z + Ax + Bx*z
    Y = By'/z + Ay + By*z

    @test X*Y ≈ Y*X
    @test Y^2 ≈ (I - X^2)*(I + ε*X)
end;
