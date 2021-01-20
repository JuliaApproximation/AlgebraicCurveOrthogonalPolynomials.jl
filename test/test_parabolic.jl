##
# Instead Sum Basis
#
#  u(x) == sqrt(x) * P^(0,1/2)(x) * c + P(x) * d
# where c, d are vectors of coefficients
#
# Let y = sqrt(x) so that y^2 = x
# 
#  u(x) == y * P^(0,1/2)(x) * c + P(x) * d
#
# idea: use OPs in x and y to get an orthogonal expansion
#
# u(x) == [P_0(x,y) | P_10(x,y) P_11(x,y) | P_20(x,y) P_21(x,y) | … ] c
#
# where P_nk(x,y) for k = 0,1 are polynomials in (x,y) of degree n, orthogonal on some inner product
# define on 0 ≤ x ≤ 1, y = sqrt(x)
# Note there are only 2 degree d polynomials since they are spanned by
#
# span(x^{d-1}*y, x^d)
#
# since x^{d-2}*y^2 = x^{d-1} is of lower degree.
#
# To explicitely construct we want
#
# \int_0^1 w(x) * P_nk(x,y) f_m(x,y) dx = 0
# 
# for all polynomials in (x,y) of degree m < n. Do a change of variables y^2 = x, 2*y*dy = dx we get
#
# \int_0^1 2y*w(y^2) * P_nk(y^2,y) f_m(y^2,y) dy = 0
#
# CLAIM: P_nk(y^2,y) can be written in terms of Q_j(y) orthogonal w.r.t. y*w(y^2).
# since they have the same span:
#
# 1, y, y^2 = x, y^3 =x*y, y^4 = x^2, …
#
# in other words
#
# P_00(x,y) = Q_0(y), P_10(x,y) = Q_1(y), P_11(x,y) = Q_2(y), P_20(x,y) = Q_3(y), …
#
# For example if w(x) = 1, then Q_j(y) are orthogonal w.r.t. y, i.e., are equal to 
# Q_j(y) = P_j^(0,1)(2y-1)
#
# Q: What does the fractional integral Q^(1/2) look like when acting on expansions in P_nk(x,y)?
# To answer the question, probably first compute the (block-upper triangular) conversion from Sum Space to
# P_nk(x,y). 

