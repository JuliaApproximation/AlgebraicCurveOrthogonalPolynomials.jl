using OrthogonalPolynomialsAlgebraicCurves, ForwardDiff, Test, Plots
import OrthogonalPolynomialsAlgebraicCurves: symunroll, comunroll, cm, speccurvemat, spec2alg, evalmonbasis
import ForwardDiff: jacobian

# explore 2 x 2 commuting Laurent polynomials,
# begining with A_x = A_y = 0, that is,
#
# X(z) = A_x + B_x/z + z*B_x'
# Y(z) = A_y + B_y/z + z*B_y'
# 
# These satisfy
# 
# X(z) Y(z) = Y(z) X(z)
# 
# i.e.
#
# 0 = [B_x,B_y]/z^2 + ([A_x,B_y] + [B_x,A_y])/z + [B_x,B_y'] + [B_x',B_y] + [A_x,A_y]  + c.c.
#
# First we need B_x and B_y to commute. Is that enough? No.

N = 3
Aˣ, (λˣ, V), κʸ = Symmetric(randn(N,N)), (randn(N),randn(N,N)), randn(3)
X, Y = speccurve(Aˣ, (λˣ, V), κʸ)

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

c = spec2alg(X,Y)
p = (x,y) -> evalmonbasis(N, x+im*y)'*c

x = y = range(-5,5; length=50)
contour(x,y, p.(x',y); nlevels=500)
scatter!(vec(specgrid(X,Y)))

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

@test norm(p.(real.(vec(specgrid(X,Y))), imag.(vec(specgrid(X,Y))))) ≤ 1E-12


# try half-arc
# X^2 = B_x^2/z^2 + (A_x*B_x + B_x*A_x)/z + A_x^2 + B_x*B_x' + B_x'*B_x + cc
# Y^2 = B_y^2/z^2 + (A_y*B_y + B_y*A_y)/z + A_y^2 + B_y*B_y' + B_y'*B_y + cc
# So we have
#
# 1) B_x^2 + B_y^2 = 0
# 2) A_x*B_x + B_x*A_x + A_y*B_y + B_y*A_y = 0
# 3) A_x^2 + B_x*B_x' + B_x'*B_x + A_y^2 + B_y*B_y' + B_y'*B_y = I
#

conds = function(p)
    N = round(Int,(-3 + sqrt(3)sqrt(-21 + 8length(p))) / 6)
    @assert sum(1:N) + N + N^2 + 3 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    λˣ = p[sum(1:N)+1:sum(1:N)+N]
    V  = reshape(p[sum(1:N)+N+1:sum(1:N)+N+N^2],N,N)
    κʸ = p[sum(1:N)+N+N^2+1:end]
    (Aˣ,Bˣ),(Aʸ,Bʸ) = speccurvemat(Aˣ, (λˣ, V), κʸ)
    [vec(Bˣ^2 + Bʸ^2); 
    vec(Aˣ*Bˣ + Bˣ*Aˣ + Aʸ*Bʸ + Bʸ*Aʸ); 
    vec(Aˣ^2 + Bˣ*Bˣ' + Bˣ'*Bˣ + Aʸ^2 + Bʸ*Bʸ' + Bʸ'*Bʸ);
    eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - [1; 1];
    eigvals(Symmetric(Aʸ - Bʸ - Bʸ')) - [-1; 1]
    ]
end

p = [[0.5; 0; 0.5]; [0.25; 0.25]; ]


N = 2
p = randn(sum(1:N) + N + N^2 + 3)
J = jacobian(conds,p); p = p - (J \ conds(p)); conds(p)

jacobian(conds,p)
h = 0.01; p̃ = copy(p); p̃[1] += h;  (conds(p̃) - conds(p))/h



