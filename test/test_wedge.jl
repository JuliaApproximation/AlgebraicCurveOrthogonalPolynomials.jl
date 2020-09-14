using OrthogonalPolynomialsAlgebraicCurves, OrthogonalPolynomialsQuasi, Test


@testset "transform" begin
    n = 2
    V = plan_wedgetransform(n)
    x,y,w = gausswedge(n)
    @test V * wedgep.(0,x,y) ≈ [1,0,0]
    @test V * wedgep.(1,x,y) ≈ [0,1,0]
    @test V * wedgeq.(1,x,y) ≈ [0,0,1]

    n = 3
    V = plan_wedgetransform(n)
    x,y,w = gausswedge(n)
    @test V * wedgep.(0,x,y) ≈ [1,0,0,0,0]
    @test V * wedgep.(1,x,y) ≈ [0,1,0,0,0]
    @test V * wedgeq.(1,x,y) ≈ [0,0,1,0,0]
    @test V * wedgep.(2,x,y) ≈ [0,0,0,1,0]
    @test V * wedgeq.(2,x,y) ≈ [0,0,0,0,1]
end

@testset "symbols" begin
    X = z -> [ 1/8   -1/8; -1/8    1/8 ]/z + [ 3/4    1/4;   1/4    3/4] + [ 1/8  -1/8; -1/8     1/8]*z
    Y = z -> [ 1/8    1/8;  1/8    1/8 ]/z + [ 3/4    -1/4; -1/4    3/4] + [ 1/8   1/8;  1/8     1/8]*z

    # (1-x) * (1-y) == 0
    z = exp(0.1im)
    @test norm((I - X(z)) * (I - Y(z))) ≤ eps()
    @test X(z)Y(z) ≈ Y(z)X(z)
end



# n = 200
# x,y,w = gausswedge(n); V= plan_wedgetransform(n); M = wedgemassmatrix(n);

# m = 150;
# p0 = p.(m, x, y)/sqrt(M[Block(m+1,m+1)[1,1]]); (sqrt.(M) * V * (x .* p0))[Block.(m:m+2)]
# q0 = q.(m, x, y)/sqrt(M[Block(m+1,m+1)[2,2]]); (sqrt.(M) * V * (x .* q0))[Block.(m:m+2)]

# p0 = p.(m, x, y)/sqrt(M[Block(m+1,m+1)[1,1]]); (sqrt.(M) * V * (y .* p0))[Block.(m:m+2)]
# q0 = q.(m, x, y)/sqrt(M[Block(m+1,m+1)[2,2]]); (sqrt.(M) * V * (y .* q0))[Block.(m:m+2)]


# [0.125 



# n = 3

# m = 2
# p0 = p.(m,x,y); dot(p0, Diagonal(w), p0)
# q0 = q.(m,x,y); dot(q0, Diagonal(w), q0)

# (p0 = p.(m,x,y); dot(p0, Diagonal(w), p0))

# M = 

# m = 100; V[Block.(m:m+2), :] * (x .* p.(m, x, y))
# m = 70; V[Block.(m:m+2), :] * (x .* q.(m, x, y))

# m = 45; V[Block.(m:m+2), :] * (y .* p.(m, x, y))
# m = 45; V[Block.(m:m+2), :] * (y .* q.(m, x, y))



# x,w = gausswedge(2); x,y = first.(x),last.(x);

# dot(P[x[1:10],4],w[1:10])

# m,n = 0,1; dot(p.(m, x, y) .* p.(n, x, y), w)
# m,n = 1,1; dot(p.(m, x, y) .* q.(n, x, y), w)
# m,n = 1,1; dot(q.(m, x, y) .* q.(n, x, y), w)

# jacobiweight(a,b,d) = JacobiWeight(a,b)[affine(d,ChebyshevInterval())]
# L = P¹⁰ \ (jacobiweight(1,0,0..1) .* P²⁰)
# R = P¹⁰ \ P

# @test jacobiweight(1,0,0..1)[0.1] ≈  2*(1-0.1)

# X = jacobimatrix(P)
# n,x,y = 5,0.2,1.0
# @test x*p(n,x,y) ≈ x*P[x,n+1] ≈ P[x,n:n+2]' * X[n:n+2,n+1] ≈ p(n-1,x,y)*X[n,n+1] + p(n,x,y)*X[n+1,n+1] + p(n+1,x,y)*X[n+2,n+1]

# x,y = 1.0,0.2
# P¹⁰ = jacobi(1,0,0..1)
# @test q(n,x,y) ≈ -n*(1-y)*P²⁰[y,n] ≈ -n*(dot(P¹⁰[y,n:n+1],L[n:n+1,n]))/2 ≈ -n/2 * (P¹⁰[y,n]*L[n,n]+P¹⁰[y,n+1]*L[n+1,n])
# @test x*p(n,x,y) ≈ P[y,n+1] ≈ dot(P¹⁰[y,n:n+1],R[n:n+1,n+1]) ≈ P¹⁰[y,n]*R[n,n+1] + P¹⁰[y,n+1]*R[n+1,n+1]

# @test p(n,x,y)/R[n+1,n+1] - q(n,x,y) * (-2/(n*L[n+1,n])) ≈ P¹⁰[y,n]*(R[n,n+1]/R[n+1,n+1] - L[n,n]/L[n+1,n])


# P[y,n+1] ≈ p(n-1,x,y)*X[n,n+1] + p(n,x,y)*X[n+1,n+1] + p(n+1,x,y)*X[n+2,n+1]

