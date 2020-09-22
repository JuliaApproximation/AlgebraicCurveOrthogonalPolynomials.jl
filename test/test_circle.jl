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
        @test first.(xy)[SVector(x,y)] == 0.1
        x,y = first.(xy),last.(xy)
        P \ first.(xy)
    end
end