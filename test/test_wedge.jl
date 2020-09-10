using OrthogonalPolynomialsQuasi, FastGaussQuadrature, StaticArrays, BlockArrays, Test

function gausswedge(n)
    x,w = gaussradau(n)
    reverse!(x); reverse!(w)
    x .= (1 .- x)./2; ldiv!(2, w);
    [x; ones(length(x)-1)], [ones(length(x)); reverse!(x[1:end-1])], [w[1:end-1]; 2w[end]; reverse!(w[1:end-1])]
end

P = legendre(0..1)
p = (n,x,y) -> P[x,n+1] + P[y,n+1] - 1
P²⁰ = jacobi(2,0,0..1)
q = (n,x,y) -> sqrt(n) * ((1-x) * P²⁰[x,n] - (1-y)*P²⁰[y,n])

function wedgemassmatrix(n)
    x,y,w = gausswedge(n)
    N = 2n-1
    ret = Vector{Float64}(undef, N)
    ret[1] = dot(p0, Diagonal(w), p0)
    for m = 1:n-1
        p0 = p.(m, x, y)
        q0 = q.(m, x, y)
        ret[2m] = dot(p0, Diagonal(w), p0)
        ret[2m+1] = dot(q0, Diagonal(w), q0)
    end
    PseudoBlockArray(Diagonal(ret), [1; fill(2,n-1)], [1; fill(2,n-1)])
end

function plan_wedgetransform(n)
    N = 2n-1
    x,y,w = gausswedge(n)
    ret = Array{Float64}(undef, N, N)
    p0 = p.(0, x, y)
    σ = dot(p0, Diagonal(w), p0)
    ret[1,:] .= p.(0, x, y) .* w ./ σ
    for m = 1:n-1
        p0 = p.(m, x, y)
        q0 = q.(m, x, y)
        σ = dot(p0, Diagonal(w), p0)
        μ = dot(q0, Diagonal(w), q0)
        ret[2m,:] .= p.(m, x, y) .* w ./ σ
        ret[2m+1,:] .= q.(m, x, y) .* w ./ μ
    end
    PseudoBlockArray(ret, [1; fill(2,n-1)], [N])
end

wedgetransform(v::AbstractVector) = plan_wedgetransform(length(v) ÷ 2 + 1) * v

@testset "transform" begin
    n = 2
    V = plan_wedgetransform(n)
    x,y,w = gausswedge(n)
    @test V * p.(0,x,y) ≈ [1,0,0]
    @test V * p.(1,x,y) ≈ [0,1,0]
    @test V * q.(1,x,y) ≈ [0,0,1]

    n = 3
    V = plan_wedgetransform(n)
    x,y,w = gausswedge(n)
    @test V * p.(0,x,y) ≈ [1,0,0,0,0]
    @test V * p.(1,x,y) ≈ [0,1,0,0,0]
    @test V * q.(1,x,y) ≈ [0,0,1,0,0]
    @test V * p.(2,x,y) ≈ [0,0,0,1,0]
    @test V * q.(2,x,y) ≈ [0,0,0,0,1]
end

@testset "sparse Legendre -> (1-y)*P²⁰" begin
    P = Legendre()
    Q = JacobiWeight(1,0) .* Jacobi(2,0)
    P \ Q[:,4]
end


n = 200
x,y,w = gausswedge(n); V= plan_wedgetransform(n); M = wedgemassmatrix(n);

m = 150;
p0 = p.(m, x, y)/sqrt(M[Block(m+1,m+1)[1,1]]); (sqrt.(M) * V * (x .* p0))[Block.(m:m+2)]
q0 = q.(m, x, y)/sqrt(M[Block(m+1,m+1)[2,2]]); (sqrt.(M) * V * (x .* q0))[Block.(m:m+2)]

p0 = p.(m, x, y)/sqrt(M[Block(m+1,m+1)[1,1]]); (sqrt.(M) * V * (y .* p0))[Block.(m:m+2)]
q0 = q.(m, x, y)/sqrt(M[Block(m+1,m+1)[2,2]]); (sqrt.(M) * V * (y .* q0))[Block.(m:m+2)]


X = z -> [ 1/8   -1/8; -1/8    1/8 ]/z + [ 3/4    1/4;   1/4    3/4] + [ 1/8  -1/8; -1/8     1/8]*z
Y = z -> [ 1/8    1/8;  1/8    1/8 ]/z + [ 3/4    -1/4; -1/4    3/4] + [ 1/8   1/8;  1/8     1/8]*z

# (1-x) * (1-y) == 0
z = exp(0.1im)
@test norm((I - X(z)) * (I - Y(z))) ≤ eps()
@test X(z)Y(z) ≈ Y(z)X(z)


[0.125 



n = 3

m = 2
p0 = p.(m,x,y); dot(p0, Diagonal(w), p0)
q0 = q.(m,x,y); dot(q0, Diagonal(w), q0)

(p0 = p.(m,x,y); dot(p0, Diagonal(w), p0))

M = 

m = 100; V[Block.(m:m+2), :] * (x .* p.(m, x, y))
m = 70; V[Block.(m:m+2), :] * (x .* q.(m, x, y))

m = 45; V[Block.(m:m+2), :] * (y .* p.(m, x, y))
m = 45; V[Block.(m:m+2), :] * (y .* q.(m, x, y))


[0.125


x,w = gausswedge(2); x,y = first.(x),last.(x);

dot(P[x[1:10],4],w[1:10])

m,n = 0,1; dot(p.(m, x, y) .* p.(n, x, y), w)
m,n = 1,1; dot(p.(m, x, y) .* q.(n, x, y), w)
m,n = 1,1; dot(q.(m, x, y) .* q.(n, x, y), w)

jacobiweight(a,b,d) = JacobiWeight(a,b)[affine(d,ChebyshevInterval())]
L = P¹⁰ \ (jacobiweight(1,0,0..1) .* P²⁰)
R = P¹⁰ \ P

@test jacobiweight(1,0,0..1)[0.1] ≈  2*(1-0.1)

X = jacobimatrix(P)
n,x,y = 5,0.2,1.0
@test x*p(n,x,y) ≈ x*P[x,n+1] ≈ P[x,n:n+2]' * X[n:n+2,n+1] ≈ p(n-1,x,y)*X[n,n+1] + p(n,x,y)*X[n+1,n+1] + p(n+1,x,y)*X[n+2,n+1]

x,y = 1.0,0.2
P¹⁰ = jacobi(1,0,0..1)
@test q(n,x,y) ≈ -n*(1-y)*P²⁰[y,n] ≈ -n*(dot(P¹⁰[y,n:n+1],L[n:n+1,n]))/2 ≈ -n/2 * (P¹⁰[y,n]*L[n,n]+P¹⁰[y,n+1]*L[n+1,n])
@test x*p(n,x,y) ≈ P[y,n+1] ≈ dot(P¹⁰[y,n:n+1],R[n:n+1,n+1]) ≈ P¹⁰[y,n]*R[n,n+1] + P¹⁰[y,n+1]*R[n+1,n+1]

@test p(n,x,y)/R[n+1,n+1] - q(n,x,y) * (-2/(n*L[n+1,n])) ≈ P¹⁰[y,n]*(R[n,n+1]/R[n+1,n+1] - L[n,n]/L[n+1,n])


P[y,n+1] ≈ p(n-1,x,y)*X[n,n+1] + p(n,x,y)*X[n+1,n+1] + p(n+1,x,y)*X[n+2,n+1]

