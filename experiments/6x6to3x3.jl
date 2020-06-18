using OrthogonalPolynomialsAlgebraicCurves, Plots, ForwardDiff
import ForwardDiff: jacobian
import OrthogonalPolynomialsAlgebraicCurves: cm

Z = zeros(6,6)
Q1 = Matrix(I,6,6)[[1,2,3,6,5,4],:]
Q = [Q1 Z; Z Q1]
X̃ = PseudoBlockArray(Q*[A66x B66x; B66x' A66x]*Q', fill(3,4), fill(3,4))
Ỹ = PseudoBlockArray(Q*[A66y B66y; B66y' A66y]*Q', fill(3,4), fill(3,4))

Ax = X̃[Block(1,1)]
Bx = X̃[Block(1,2)]

Ay = Ỹ[Block(1,1)]
By = Ỹ[Block(1,2)]


X = z -> B66x/z + A66x + B66x'*z
Y = z -> B66y/z + A66y + B66y'*z

scatter(vec(specgrid(X,Y)))


comroll(A, B) = [symroll(A); vec(B)]

function comunroll(p) 
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 24m))/6)
    symunroll(p[1:sum(1:N)]),reshape(p[sum(1:N)+1:end],N,N)
end
Aˣ = A66x; Bˣ = B66x;

Aˣ = Ay; Bˣ = By;
conds = function(p)
    Aʸ, Bʸ = comunroll(p)
    [vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ))]
end

X = z -> Bˣ/z + Aˣ + Bˣ'*z
Y = z -> Bʸ/z + Aʸ + Bʸ'*z


p = comroll(Aˣ, Bˣ)
K = nullspace(jacobian(conds,p))
Aʸ, Bʸ = comunroll(K*randn(size(K,2))); scatter(vec(specgrid(X,Y)))

###
# X = B/z + A + B'*z
# X^2 = B^2/z^2 + 
#    (A*B + B*A)/z +
#    A^2 + B*B' + B'*B + …
# X^3 = B^3/z^3 +
#           (B*A*B + B^2*A + A*B^2)/z^2 +
#           (B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2)/z +
#           B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A +
#           …
# 
###

"""
 Squares B/z + A + B'*z 
"""
lrntsquare(A,B) = (B*B'+A^2 + B'B, B*A+A*B, B^2)

lrntcube(A, B) = (B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A, 
                    B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2,
                    B*A*B + B^2*A + A*B^2, B^3)


function conds2(c)
    Aʸ, Bʸ = comunroll(K*c)
    Y²₀,Y²₁,Y²₂ = lrntsquare(Aʸ, Bʸ)
    X²₀,X²₁,X²₂ = lrntsquare(Aˣ, Bˣ)
    X³₀,X³₁,X³₂,X³₃ = lrntcube(Aˣ, Bˣ)
    vcat(vec(4Y²₀ - (I - Aˣ - X²₀ + X³₀)),
         vec(4Y²₁ - (Bˣ - X²₁ + X³₁)),
         vec(4Y²₂ - (- X²₂ + X³₂)),
         vec(X³₃))
end

c = randn(3)
c = c - jacobian(conds2,c) \ conds2(c); norm(conds2(c)) # does not converge




# Y(z)^2 ≈ 1/4*(I - X(z))^2*(I + X(z))
# Y(z)^2 ≈ 1/4*(I - 2X(z) + X(z)^2)*(I + X(z))
# 4Y(z)^2 ≈ (I - X(z) - X(z)^2 + X(z)^3)
X⁴
\^4 

eigvals(X(1))

speccurve
Ax
Bx

Ay
By