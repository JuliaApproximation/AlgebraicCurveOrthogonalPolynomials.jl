using OrthogonalPolynomialsAlgebraicCurves, OrthogonalPolynomialsQuasi, Test

@testset "Circle" begin
    @testset "LegendreCircle" begin
        P = LegendreCircle()

        T,U = ChebyshevT(),ChebyshevU()
        x = 0.1; y = sqrt(1-x^2)
        @test P[SVector(x,y),Block(1)] == [1.0]
        @test P[SVector(x,y),Block(2)] == [T[0.1,2], y*U[0.1,1]]
        @test P[SVector(x,y),Block(3)] == [T[0.1,3], y*U[0.1,2]]

        xy = axes(P,1)
        @test xy[SVector(x,y)] == SVector(x,y)
        @test xy[SVector(x,y)] isa CircleCoordinate
        @test xy[CircleCoordinate(0.1)] ≡ CircleCoordinate(0.1)
        @test CircleCoordinate(0.1) in xy
        @test first.(xy)[SVector(x,y)] == 0.1

        @test (P * [1; zeros(∞)])[CircleCoordinate(0.1)] ≈ 1
        @test (P * [0; 1; zeros(∞)])[CircleCoordinate(0.1)] ≈ sin(0.1)
        @test (P * [0; 0; 1; zeros(∞)])[CircleCoordinate(0.1)] ≈ cos(0.1)

        x,y = first.(xy),last.(xy)
        @test x[CircleCoordinate(0.1)] == cos(0.1) 
        @test P[:,Base.OneTo(5)] \ x ≈ [0,0,1,0,0]
        @test (P \ y)[Block(2)] ≈ [1,0]
        @test (P \ x)[Block(2)] ≈ [0,1]
        u = P * (P \ cos.(x .+ exp.(y)))
        @test u[CircleCoordinate(0.1)] ≈ cos(cos(0.1) + exp(sin(0.1)))
    end
end