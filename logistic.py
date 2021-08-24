def logistic_map(r,x):

    return r*x*(1-x)

def iterate_f(it, r, x):

    out = [logistic_map(r,x)]

    for i in range(it-1):
        out.append(logistic_map(r,out[-1]))

    return out