using OrthogonalPolynomialsAlgebraicCurves, ClassicalOrthogonalPolynomials, Test

@testset "cubic" begin
    P = LegendreCubic(2)
    xy = axes(P,1)
    x = 0.1
    xy = SVector(x, sqrt(x*(1-x)*(P.t-x)))
    
    @testset "jacobi" begin
        
    end
end

