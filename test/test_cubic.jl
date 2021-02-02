using OrthogonalPolynomialsAlgebraicCurves, ClassicalOrthogonalPolynomials, FillArrays, BlockArrays, LazyBandedMatrices, LazyArrays, Test
import OrthogonalPolynomialsAlgebraicCurves: LegendreCubicJacobiX

@testset "cubic" begin
    P = LegendreCubic(2)
    xy = axes(P,1)
    x = 0.1
    y = sqrt(x*(1-x)*(P.t-x))
    xy = SVector(x, y)
    
    @testset "evaluation" begin
        @test P[xy,1] == 1
        @test P[xy,2] == P.P[x,2]
        @test P[xy,3] == y*P.Q[x,1]
        @test P[xy,6] == P.P[x,4]
    end

    @testset "jacobi" begin
        X_P = jacobimatrix(P.P);
        X_Q = jacobimatrix(P.Q);
        R = P.Q \ P.P;
        R[1:10,1:10]

        X = jacobimatrix(Val(1), P)
        Y = jacobimatrix(Val(2), P)

        N = 4
        @test x*P[xy,Block.(1:N)]' ≈ P[xy,Block.(1:N+1)]' * X[Block.(1:N+1), Block.(1:N)]
        @test y*P[xy,Block.(1:N)]' ≈ P[xy,Block.(1:N+1)]' * Y[Block.(1:N+1), Block.(1:N)]

        # @testset "blocked" begin 
        #     # make 1-blocks
        #     X_P_bl = PseudoBlockArray(view(X_P,2:∞,2:∞), Ones{Int}(∞), Ones{Int}(∞))
        #     X_Q_bl = PseudoBlockArray(X_Q, Ones{Int}(∞), Ones{Int}(∞))
        #     Z = Zeros(axes(X_P_bl))
        #     X_tail = PseudoBlockArray(BlockBroadcastArray{Float64}(hvcat, 2, X_P_bl, Z, Z, X_Q_bl),
        #                     Vcat(2,Fill(3,∞)), Vcat(2,Fill(3,∞)));

        #     r = PseudoBlockArray([X_P[2,1]; Zeros(∞)], (axes(X_tail,1),))

        #     X = BlockHvcat(2, X_P[1,1], r', r, X_tail);
        # end
    end
end

