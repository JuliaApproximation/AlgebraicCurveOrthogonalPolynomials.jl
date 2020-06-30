
# Want Q = […,Q_1,I,Q_1,I…] so that
# Q*Y*Q' is 3x3
# This requires
# Be2 * Q_1' == Q_1*Bo2

function eqns(p)
    Q_1 = qr(reshape(p,3,3)).Q
    vec(Be2 * Q_1' - Q_1*Bo2)
end

p = randn(9)
p = p - svd(jacobian(eqns,p)) \ eqns(p); eqns(p)