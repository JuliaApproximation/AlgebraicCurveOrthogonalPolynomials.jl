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
