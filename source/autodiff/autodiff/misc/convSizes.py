# Interestingly these size computations work for fractional numbers to.
def c(i, k, s, p):
    return (i + 2*p - k) // s + 1


def t(i, k, s, p, a):
    pp = (k - 1) - p
    ss = 1

    # An expression for what size we actually output
    # a = (I + 2 * p - k) % s
    # return s * (i - 1) + a + k - 2 * p

    # The inputs to conv we need to get the desired output
    # a = (I + 2 * p - k) % s
    stretchedSize = s * (i - 1) + 1
    ii = stretchedSize + a
    return c(ii, k, ss, pp)


# # Same sizes for any n :)
# n = 2
# K = 2*n+1
# S = 1
# P = K//2


K = 4
S = 2
P = 1

# a bit of algebra shows that t(c(I,K,S,P),K,S,P,a) == I
for I in range(2, 15):
    # I = 2 ** I
    a = (I + 2 * P - K) % S
    print('I:', I)
    print('c:', c(I, K, S, P))
    print('tc:', t(c(I, K, S, P), K, S, P, a))
    print()


print('(N*P*Q, C*R*S)   | (H,C),   (P,K)')


def getSizes(i):
    N = 1
    C = 2**(10-i)
    K = 2**(9-i)
    H = W = 2 ** (2 + i)
    P = Q = 2 * H
    R = S = 4
    if i == 3:
        K = 1
    s = str((N * P * Q, C * R * S)) + '\t   '
    return s + str(((H, C), (P, K)))


for i in range(4):
    print(getSizes(i))
