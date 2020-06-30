using LinearAlgebra, BlockBandedMatrices, OrthogonalPolynomialsAlgebraicCurves, Test

@testset "commuting truncations cubic teardrop upper branch" begin
    γ(x) = 1/2*(1 - x)*sqrt(1 + x)
    blocks = 10
    v = sqrt(2)/64
    Bˣ = [0 0 0;1/8 0 0;1/2 -1/8 0]
    Bʸ = [v 0 0;-6v v 0;v 6v v]
    Aˣ = [-1/4 -1/2 -1/8;-1/2 -1/4 1/2;-1/8 1/2 -1/4]
    Aʸ = [12v -v 6v;-v 12v v;6v v 12v];
    x1 = 1/4; y1 = γ(x1)
    x4 = -1/4; y4 = γ(x4)
    v = sqrt(2)/32
    A0x = [x1]
    B0x = [0 0]
    A1x = [x4 0;0 -3/8]
    B1x = [0 0 0;1/2 -1/8 0]
    A0y = [y1]
    B0y = [0 0]
    A1y = [y4 0;0 9v]
    B1y = [0 0 0;v 3v v/2]
    u1 = 0; v1 = γ(u1)
    u4 = -1; v4 = γ(u4)
    a0x = [u1]
    b0x = [0 0]
    a1x = [-3/8 0;0 u4]
    b1x = [0 -1/8 -1/2;0 0 0]
    a0y = [v1]
    b0y = [0 0]
    a1y = [9v 0;0 v4]
    b1y = [-v/2 3v -v;0 0 0]
    X = BlockBandedMatrix(zeros(3(blocks+2),3(blocks+2)),[1;2;fill(3,blocks);2;1],[1;2;fill(3,blocks);2;1],(1,1))
    X[Block(1,1)] = A0x
    X[Block(1,2)] = B0x; X[Block(2,1)] = B0x'
    X[Block(2,2)] = A1x
    X[Block(2,3)] = B1x; X[Block(3,2)] = B1x'
    X[Block(2+blocks,3+blocks)] = b1x'; X[Block(3+blocks,2+blocks)] = b1x
    X[Block(3+blocks,3+blocks)] = a1x
    X[Block(3+blocks,4+blocks)] = b0x'; X[Block(4+blocks,3+blocks)] = b0x
    X[Block(4+blocks,4+blocks)] = a0x
    for k = 1:blocks-1
        X[Block(k+2,k+2)] = Aˣ
        X[Block(k+2,k+3)] = Bˣ; X[Block(k+3,k+2)] = Bˣ'
    end
    X[Block(blocks+2,blocks+2)] = Aˣ
    Y = BlockBandedMatrix(zeros(3(blocks+2),3(blocks+2)),[1;2;fill(3,blocks);2;1],[1;2;fill(3,blocks);2;1],(1,1))
    Y[Block(1,1)] = A0y
    Y[Block(1,2)] = B0y; Y[Block(2,1)] = B0y'
    Y[Block(2,2)] = A1y
    Y[Block(2,3)] = B1y; Y[Block(3,2)] = B1y'
    Y[Block(2+blocks,3+blocks)] = b1y'; Y[Block(3+blocks,2+blocks)] = b1y
    Y[Block(3+blocks,3+blocks)] = a1y
    Y[Block(3+blocks,4+blocks)] = b0y'; Y[Block(4+blocks,3+blocks)] = b0y
    Y[Block(4+blocks,4+blocks)] = a0y
    for k = 1:blocks-1
        Y[Block(k+2,k+2)] = Aʸ
        Y[Block(k+2,k+3)] = Bʸ; Y[Block(k+3,k+2)] = Bʸ'
    end
    Y[Block(blocks+2,blocks+2)] = Aʸ
   @test X*Y ≈ Y*X
   @test Y^2 ≈ 1/4*(I - X)^2*(I + X)
end

@testset "commuting truncations full cubic teardrop" begin
    γ(x) = 1/2*(1 - x)*sqrt(1 + x)
    blocks = 10
    v = sqrt(2)/8
    Bˣ = [0 0 -1/2;0 0 0;0 -1/2 0]
    Bʸ = [-v v 0;0 -v 0;0 0 -v]
    Aˣ = [0 1/2 0;1/2 0 0;0 0 0]
    Aʸ = [0 0 -v;0 0 -v;-v -v 0];
    x1 = 1/4; y1 = γ(x1)
    x6 = -1/4; y6 = -γ(x6)
    A0x = [x1]
    B0x = [0 0]
    A1x = [-1/2 0;0 x6]
    B1x = [0 0 1/2;0 0 0]
    A0y = [y1]
    B0y = [0 0]
    A1y = [0 0;0 y6]
    B1y = [v -2v 0;0 0 0]
    u1 = 0; v1 = -γ(u1)
    u4 = -1; v4 = γ(u4)
    a0x = [u1]
    b0x = [0 0]
    a1x = [-1/2 0;0 u4]
    b1x = [0 0 1/2;0 0 0]
    a0y = [v1]
    b0y = [0 0]
    a1y = [0 0;0 v4]
    b1y = [-2v v 0;0 0 0]
    X = BlockBandedMatrix(zeros(3(blocks+2),3(blocks+2)),[1;2;fill(3,blocks);2;1],[1;2;fill(3,blocks);2;1],(1,1))
    X[Block(1,1)] = A0x
    X[Block(1,2)] = B0x; X[Block(2,1)] = B0x'
    X[Block(2,2)] = A1x
    X[Block(2,3)] = B1x; X[Block(3,2)] = B1x'
    X[Block(2+blocks,3+blocks)] = b1x'; X[Block(3+blocks,2+blocks)] = b1x
    X[Block(3+blocks,3+blocks)] = a1x
    X[Block(3+blocks,4+blocks)] = b0x'; X[Block(4+blocks,3+blocks)] = b0x
    X[Block(4+blocks,4+blocks)] = a0x
    for k = 1:blocks-1
        X[Block(k+2,k+2)] = Aˣ
        X[Block(k+2,k+3)] = Bˣ; X[Block(k+3,k+2)] = Bˣ'
    end
    X[Block(blocks+2,blocks+2)] = Aˣ
    Y = BlockBandedMatrix(zeros(3(blocks+2),3(blocks+2)),[1;2;fill(3,blocks);2;1],[1;2;fill(3,blocks);2;1],(1,1))
    Y[Block(1,1)] = A0y
    Y[Block(1,2)] = B0y; Y[Block(2,1)] = B0y'
    Y[Block(2,2)] = A1y
    Y[Block(2,3)] = B1y; Y[Block(3,2)] = B1y'
    Y[Block(2+blocks,3+blocks)] = b1y'; Y[Block(3+blocks,2+blocks)] = b1y
    Y[Block(3+blocks,3+blocks)] = a1y
    Y[Block(3+blocks,4+blocks)] = b0y'; Y[Block(4+blocks,3+blocks)] = b0y
    Y[Block(4+blocks,4+blocks)] = a0y
    for k = 1:blocks-1
        Y[Block(k+2,k+2)] = Aʸ
        Y[Block(k+2,k+3)] = Bʸ; Y[Block(k+3,k+2)] = Bʸ'
    end
    Y[Block(blocks+2,blocks+2)] = Aʸ
   @test X*Y ≈ Y*X
   @test Y^2 ≈ 1/4*(I - X)^2*(I + X)
end
