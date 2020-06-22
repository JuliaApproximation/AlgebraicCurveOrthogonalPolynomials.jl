using BandedMatrices, InfiniteLinearAlgebra, Test, BlockArrays, BlockBandedMatrices

function polar(A)
    U,σ,V = svd(A)
    U*V',V*Diagonal(σ)*V'
end
Q,S = polar(B)
@test Q'Q ≈ I
@test S ≈ S'
@test Q*S ≈ B

##
# Scalar
##

b = randn()
J = BandedMatrix(SymTridiagonal(Fill(0.0,∞), Fill(b,∞)))

z = 1+im
g = inv(((J - z*I) \ [1; zeros(∞)])[1])
@test -z - b^2 / g ≈ g
@test g ≈ (-z - sqrt(z-2b)sqrt(z+2b))/2

@test g * (-z + sqrt(z-2b)sqrt(z+2b))/2 ≈ b^2

@test inv(g) ≈ (-z + sqrt(z-2b)sqrt(z+2b))/2b^2

##
# Tri Matrix
##
d = 2
B = tril(randn(d,d))
N = 1000
J = BlockBandedMatrix(mortar(Tridiagonal(Fill(Matrix(B'),N-1), Fill(zeros(d,d),N), Fill(B,N-1))))

z=10
G = inv(((J - z*I) \ [Matrix(I,d,d); zeros(d*N-d,d)])[Block(1,1)])

@test -z*I - B*inv(G)*B' ≈ G

@test -z*B^(-1/2)*(B^(-1/2))' - sqrt(B)*inv(G)*sqrt(B)' ≈ B^(-1/2)*G*(B^(-1/2))'
@test -z*B^(-1/2)*(B^(-1/2))' - inv((B^(-1/2))'*G*B^(-1/2)) ≈ B^(-1/2)*G*(B^(-1/2))'

# b^2/g = g
# g = b*y
# b/y = b*y
Y = (B^(1/2))*inv(G)*(B^(1/2))'
@test -z*I - sqrt(B)*Y*sqrt(B)' ≈ G
@test -z*I - sqrt(B)*Y*sqrt(B)' ≈ sqrt(B)'*inv(Y)*sqrt(B)


λ,Q = eigen(Symmetric(G))
Λ = Diagonal(λ)
@test -z* I - Q'B*Q * inv(Λ) * Q'*B'*Q ≈ Λ


G̃ = (-z*I - sqrt(z^2 * I - 4B*B'))/2
@test G ≈ G̃

z = 100
G̃ = (-z*I - sqrt(z^2 * I - 4B*B'))/2
norm(-z*I - B*inv(G̃)*B' - G̃)


z=50
G = inv(((J - z*I) \ [Matrix(I,d,d); zeros(d*N-d,d)])[Block(1,1)])
(2G + z*I)^2 - z^2*I


##
# Matrix
##

# B = randn(2,2)
B = [1.0 0; 2 3]
N = 100
J = BlockBandedMatrix(mortar(Tridiagonal(Fill(Matrix(B'),N-1), Fill(zeros(2,2),N), Fill(B,N-1))))

z = 10
G = inv(((J - z*I) \ [Matrix(I,2,2); zeros(2N-2,2)])[Block(1,1)])
@test -z*I - B*inv(G)*B' ≈ G

z = 4
Z = z*I
@test -Z - B*inv(G)*B' ≈ G
Q,S = polar(B')
@test -Z - S*Q'*inv(G)*Q*S ≈ G
@test -Z - S*Q'*inv(G)*Q*S ≈ G
# Z = sqrt(S)*T*sqrt(S)
# G = sqrt(S)*X*sqrt(S)
T = S^(-1/2) * Z * S^(-1/2)
X = S^(-1/2) * G * S^(-1/2)
@test -sqrt(S)T*sqrt(S) - S*Q'*inv(G)*Q*S ≈ sqrt(S)X*sqrt(S)
@test -T - sqrt(S)*Q'*S^(-1/2)*inv(X)*S^(-1/2)*Q*sqrt(S) ≈ X


Q,S = polar(B')
ZZ = zeros(2,2)
II = Matrix(I,2,2)
Q1 = mortar(Diagonal([Matrix(I,2,2),Matrix(Q'),Matrix(Q')^2,Matrix(Q')^3]));
M = 4;  Q1*J[Block.(1:M),Block.(1:M)]*Q1'


# -z - b^2/g == g
# z = |b|*t, g = |b|*x
# -|b|*t - |b|/x == |b|*x
# -t - 1/x == x



polar(B)


Y = inv(G)
@test -z*Y - Y*B*Y*B' ≈ I
Z = Y*B
@test -z*Z*inv(B) - Z^2*inv(B)*B' ≈ I
@test norm(Z^2 + z*Z*inv(B'B)*B  + inv(B')*B) ≤ 1E-13

# Q*B*V' == R1
# Q*B'*V' == R2
# -z*Q*V' - Q*B*Z'**inv(Q'*Y*V')*Q*B'*V' == Q*Y*V'
Q = schur(B,Matrix(B')).Q
Z = schur(B,Matrix(B')).Z
Q'*B*Z
Q'*B'*Z

@test -z*I - B*inv(G)*B' ≈ G
@test -z*Q'*Z - Q'*B*Z * Z'*inv(G)Q*Q'*B'*Z ≈ Q'*G*Z

fit = (a,b) -> vec((-z*I - sqrt(z^2*I - a*B^2 - b*B'B - (4-2a-b)*B*B' - a*(B')^2))/2)
fit2 = abc -> fit(abc...) - vec(G)
using ForwardDiff
import ForwardDiff: jacobian
abc = randn(2)
abc = abc - jacobian(fit2,abc) \ fit2(abc)

G̃ = 
@test -z*I - B*inv(G̃)*B' ≈ G̃

G^2 + z * G
G'G + z * G

G1^2 + D*G1 + E == 0
G2^2 + D*G1 + E == 0


G̃ = (-z*I - sqrt(z^2 * I - 4B'*B))/2
@test G̃ * (-z*I + sqrt(z^2 * I - 4B'B))/2 ≈ B'B
@test (-z*I + sqrt(z^2 * I - 4B'B))/2 * inv(B'B) ≈ inv(G̃)
@test B*inv(G̃)*B' ≈ (-z*I + sqrt(z^2 * I - 4B*B'))/2

@test -z*I - B*inv(G̃)*B' ≈ (-z*I - sqrt(z^2 * I - 4B*B'))/2

(2G + z*I)^2 - z^2*I
(2G + z*I)'*(2G + z*I)

 + C == 0

B'B
B^2


(-z*I + sqrt(z^2 * I - 4B*B'))/2





B*inv(G̃)*B'


(-z*I - sqrt(z^2 * I - 4B'*B))/2