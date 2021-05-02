def cmp(f, g):
    return lambda x: f(g(x))

def mul(f, g):
    return lambda x: f(x) * g(x)

def mulk(f, k):
    return mul(f, const(k))

def add(f, g):
    return lambda x: f(x) + g(x)

def addk(f, k):
    return add(f, const(k))

def negate(f):
    return lambda x: -f(x)

def const(k):
    return lambda _: k
