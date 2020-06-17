using LinearAlgebra, Test
Ae1 = [0 1/2 0;1/2 0 0;0 0 0]
v = 1/(4*sqrt(2))
Ae2 = [0 0 -v;0 0 -v;-v -v 0]
Be1 = [1/2 0 0;0 0 0;0 0 1/2]
Be2 = [0 v -v;0 0 v;v 0 0]
Ao1 = [0 0 0;0 0 1/2;0 1/2 0]
Ao2 = [0 -v -v;-v 0 0;-v 0 0]
Bo1 = [0 1/2 0;0 0 1/2;0 0 0]
Bo2 = [0 0 v;v -v 0;0 v 0]
A66x = [Ae1 Be1;Be1' Ao1]
B66x = zeros(6,6)
B66x[4:6,1:3] = Bo1
A66y = [Ae2 Be2;Be2' Ao2]
B66y = zeros(6,6)
B66y[4:6,1:3] = Bo2
X(z) = B66x'/z + A66x + B66x*z
Y(z) = B66y'/z + A66y + B66y*z
z = exp(im*pi/2)
@testset "cubic teardrop symbol" begin
    @test X(z)*Y(z) ≈ Y(z)*X(z)
    @test Y(z)^2 ≈ 1/4*(I - X(z))^2*(I + X(z))
end
