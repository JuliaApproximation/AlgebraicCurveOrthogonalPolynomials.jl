using OrthogonalPolynomialsQuasi, ContinuumArrays, Plots

# y^2 = x^3 - ε

ε = 0.1
x = Inclusion(ε^(2/3) .. (1+ε^2)^(1/3))
y = sqrt.( x.^3 .- ε^2)

P = Legendre()
m = affine(x,axes(P,1))
P = P[m,:]

OrthogonalPolynomialsQuasi.orthogonalityweight(P)
import OrthogonalPolynomialsQuasi: LegendreWeight
LegendreWeight()[m][0.3]

ones(
P'P

xx = range(first(x), last(x); length=1000)
plot(xx, y[xx])
plot!(xx, -y[xx])

