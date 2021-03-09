using AlgebraicCurveOrthogonalPolynomials, ForwardDiff, Test
import AlgebraicCurveOrthogonalPolynomials: checkcommutes
import ForwardDiff: gradient, jacobian


###
# Square
###

Bˣ = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
Bʸ = [0 0 -1 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0.5]
X = HermLaurent(zeros(4,4), Bˣ)
Y = HermLaurent(zeros(4,4), Bʸ)

ε = 1
@test ε .* X.^2 .* Y.^2 .- X.^2 .- Y.^2 ≈ -I
@test eigvals(X[1]) ≈ [-1,-1,-1,1]
@test eigvals(Y[1]) ≈ [-1,1,1,1]


##
# Old
##



comroll(A, B) = [symroll(A); vec(B)]


Bˣ = [-1/4 1/4 1/2 1/2; 1/4 -1/4 1/2 1/2; 0 0 -1/4 1/4; 0 0 1/4 -1/4]
Bʸ = [1/4 1/4 -1/2 1/2; 1/4 1/4 1/2 -1/2; 0 0 1/4 1/4; 0 0 1/4 1/4]
X = HermLaurent(zeros(4,4), Bˣ)
Y = HermLaurent(zeros(4,4), Bʸ)
ε = 1
@test ε .* X.^2 .* Y.^2 .- X.^2 .- Y.^2 ≈ -I


# (B/z + B'z)*(C/z + C'z) == B*C' + C*B' + B*C/z
function hermcmjac(B)
    N = size(Bˣ,1)
    Tr = Matrix(trjac(N))
    CmB = Matrix(cmjac(B))
    J = [CmB;
         (CmB*Tr + Matrix(cmjac(B')))]
end

Bˣ = randn(4,4)
K = nullspace(hermcmjac(Bˣ))
Bʸ = reshape(K*randn(size(K,2)),4,4)
X = HermLaurent(zeros(4,4), Bˣ); Y = HermLaurent(zeros(4,4), Bʸ)
@test checkcommutes(X, Y)






K = nullspace(hermcmjac(Bˣ))
p₀ = [vec(Bˣ); K \ vec(Bʸ)]

conds(0.9)(p₀)

G = randn(2,2)
Bˣ = [G randn(2,2); zeros(2,2) G]
nullspace(Matrix(cmjac(Bˣ)))

conds = ε -> function(p)
    Bˣ = reshape(p[1:16],4,4)
    M = 2
    J = hermcmjac(Bˣ)
    K = qr_nullspace(M,J)
    K = K*(K')[:,1:M] # normalise to enforce continuity
    Bʸ = reshape(K * p[17:end],4,4)
    X = HermLaurent(zeros(4,4), Bˣ)
    Y = HermLaurent(zeros(4,4), Bʸ)
    vec((ε .* X.^2 .* Y.^2 .- X.^2 .- Y.^2 + I).A[1])
end


()


# random commuting
Bˣ = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
Bʸ = [0 0 -1 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0.5]
X = HermLaurent(zeros(4,4), Bˣ); Y = HermLaurent(zeros(4,4), Bʸ)
p₀ = [Bʸ[6]; Bʸ[1]; Bʸ[3]; Bʸ[9]; Bʸ[11]; Bʸ[14]]


conds = function(p)
    α = p[1]; Bʸ = reshape([p[2]; 0; p[3]; 0; 0; α; 0; 0; p[4]; 0; p[5]; 0; 0; p[6]; 0; α], 4, 4)
    X = HermLaurent(zeros(4,4), Bˣ)
    Y = HermLaurent(zeros(4,4), Bʸ)
    R = ε .* X.^2 .* Y.^2 .- X.^2 .- Y.^2 + I
    λ1 = eigvals(Y[1])
    λ2 = eigvals(Y[-1])
    λ3 = eigvals(Y[-im])
    λ4 = eigvals(Y[im])
    [vec(R.A[1]); vec(R.A[3]); vec(R.A[5])] #; λ1[1]+1; λ1[end]-1;  λ2[1]+1; λ2[end]-1; λ3[1]+1; λ3[end]-1; λ4[1]+1; λ4[end]-1]
end


p = p₀
p .+= 0.000001
p = p - jacobian(conds,p) \ conds(p)


p = randn(6)
p = p - jacobian(conds,p) \ conds(p)
α = p[1]; Bʸ = reshape([p[2]; 0; p[3]; 0; 0; α; 0; 0; p[4]; 0; p[5]; 0; 0; p[6]; 0; α], 4, 4)
X = HermLaurent(zeros(4,4), Bˣ)
Y = HermLaurent(zeros(4,4), Bʸ)
scatter(vec(specgrid(X, Y)))



# impose upper triangular
Bˣ = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
Bʸ = [0 0 -1 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0.5]
p₀ = [Bʸ[6]; Bʸ[1]; Bʸ[9]; Bʸ[11]; Bʸ[14]]

conds = function(p)
    α = p[1]; Bʸ = reshape([p[2]; 0; 0; 0; 0; α; 0; 0; p[3]; 0; p[4]; 0; 0; p[5]; 0; α], 4, 4)
    X = HermLaurent(zeros(4,4), Bˣ)
    Y = HermLaurent(zeros(4,4), Bʸ)
    R = ε .* X.^2 .* Y.^2 .- X.^2 .- Y.^2 + I
    λ1 = eigvals(Y[1])
    λ2 = eigvals(Y[-1])
    λ3 = eigvals(Y[-im])
    λ4 = eigvals(Y[im])
    [vec(R.A[1]); vec(R.A[3]); vec(R.A[5])] #; λ1[1]+1; λ1[end]-1;  λ2[1]+1; λ2[end]-1; λ3[1]+1; λ3[end]-1; λ4[1]+1; λ4[end]-1]
end

p = p₀
p .+= 0.1 .* randn()
p = p - jacobian(conds,p) \ conds(p)