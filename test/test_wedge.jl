using OrthogonalPolynomialsQuasi, Test

@testset "sparse Legendre -> (1-y)*P²⁰" begin
    P = Legendre()
    Q = JacobiWeight(1,0) .* Jacobi(2,0)
    P \ Q[:,4]
end


P = legendre(0..1)
p = (n,x,y) -> P[x,n+1] + P[y,n+1] - 1
P²⁰ = jacobi(2,0,0..1)
q = (n,x,y) -> n * ((1-x) * P²⁰[x,n] - (1-y)*P²⁰[y,n])


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

