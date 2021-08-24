def logistic_map(x,r):

    return r*x*(1-x)

def iterate_f(it, x, r):

    out = [logistic_map(x,r)]

    for i in range(it-1):
        out.append(logistic_map(out[-1],r))

    return out