using OrthogonalPolynomialsAlgebraicCurves, BandedMatrices, BlockBandedMatrices, BlockArrays

function blocksymtricirculant(A, B, N)
    M = size(A,1)
    ret = BlockMatrix{Float64}(0I, Fill(M,N), Fill(M,N))
    for K = 1:N 
        ret[Block(K,K)] = A 
    end
    for K = 1:N-1 
        ret[Block(K,K+1)] = B
        ret[Block(K+1,K)] = B' 
    end
    ret[Block(1,N)] = B'
    ret[Block(N,1)] = B
    ret
end


@testset "Circulant" begin    
    Ax = Matrix(0.5I,2,2)
    Bx = Matrix(0.25I,2,2)
    a₁₂ = (1 + sqrt(2))/4
    a₂₁ = (1 - sqrt(2))/4
    Ay = [0 -0.5; -0.5 0]
    By = [0 a₁₂; a₂₁ 0]

    N = 10
    X = blocksymtricirculant(Ax, Bx, N)
    Y = blocksymtricirculant(Ay, By, N)
    @test X == X'
    @test Y == Y'
    @test X*Y ≈ Y*X
    @test X^2 + Y^2 ≈ I(2N)

    x = z -> Ax + (Bx/z + Bx'*z)
    y = z -> Ay + (By/z + By'*z)
    @test x(1) ≈ I(2)
    @test norm(y(1)) ≤ 10eps()
    @test x(1) ≈ I(2)
end

@testset "Jacobi" begin
    x₂ = -(1/4)*sqrt(4+2*sqrt(2))
    x₀ = -1/sqrt(2)
    N = 100;
    X = BandedMatrix{Float64}(undef, (N,N), (2,2))
    X[band(0)] = [x₀ 1/2*ones(1,N-2) x₀]
    X[band(1)] .= X[band(-1)] .= 0;
    X[band(2)] = X[band(-2)] = [x₂ 1/4*ones(1,N-4) x₂]

    a₁₂ = (1 + sqrt(2))/4
    a₂₁ = (1 - sqrt(2))/4
    y₁=(1/4)*sqrt(4-2*sqrt(2))
    Y = BandedMatrix{Float64}(undef, (N,N), (3,3))
    Y[band(0)].=0
    Y[band(2)].=Y[band(-2)].=0
    d1 = [y₁ repeat([-1/2 a₂₁],1,Int64(round(N/2))-1)]
    d1[N-1] = y₁
    d3 = repeat([0 a₁₂],1,Int64(round(N/2)))
    Y[band(1)] = Y[band(-1)] = d1[1:N-1]
    Y[band(3)] = Y[band(-3)] = d3[1:N-3]

    @test X*Y ≈ Y*X
    @test X^2 + Y^2 ≈ Eye(N)
end