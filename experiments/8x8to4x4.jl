using OrthogonalPolynomialsAlgebraicCurves, Plots, ForwardDiff
import ForwardDiff: jacobian
import OrthogonalPolynomialsAlgebraicCurves: cm

Z = zeros(4,4)
B₁ˣ = [3/4 1/4 0 0; 1/4 3/4 0 0; 0 0 1/4 -1/4; 0 0 -1/4 1/4]
B₂ˣ = [1/4 -1/4 0 0; -1/4 1/4 0 0; 0 0 3/4 1/4; 0 0 1/4 3/4]
Bˣ  = [Z Z;   B₁ˣ  Z]
Aˣ  = [Z B₂ˣ; B₂ˣ' Z]
    
B₁ʸ = [0 0 3/4 -1/4; 0 0 -1/4 3/4; 1/4 1/4 0 0; 1/4 1/4 0 0]
B₂ʸ = B₁ʸ
Bʸ  = [Z Z;   B₁ʸ  Z]
Aʸ  = [Z B₂ʸ; B₂ʸ' Z]

# We want to conjugate by Q = Diagonal([…,I,Q,I,Q,…])
# so that B₁ˣ*Q' == Q*B₂ˣ
B₂ˣ
Q = [-1 1 1 1; 1 -1 1 1; 1 1 -1 1; 1 1 1 -1]/2
@test Q'Q ≈ I

@test B₁ˣ*Q' ≈ Q*B₂ˣ ≈ [-1/4 1/4 1/2 1/2; 1/4 -1/4 1/2 1/2; 0 0 -1/4 1/4; 0 0 1/4 -1/4]
@test B₁ʸ*Q' ≈ Q*B₂ʸ ≈ [1/4 1/4 -1/2 1/2; 1/4 1/4 1/2 -1/2; 0 0 1/4 1/4; 0 0 1/4 1/4]



B₂ˣ

[0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]


conds = function(p)
    Q = qr(reshape(p,4,4)).Q
    [vec(B₁ˣ*Q' - Q*B₂ˣ); vec(B₁ʸ*Q' - Q*B₂ʸ)]
end

p = randn(16)
p = p - jacobian(conds,p) \ conds(p); norm(conds(p))

@test B₁ˣ*Q' ≈ Q*B₂ˣ
@test B₁ʸ*Q' ≈ Q*B₂ʸ
B₁ʸ*Q'
B₂ʸ




cm(