from functools import reduce

def smul(c,l): return [c*x for x in l]
def alleq(l): return len(set(l)) <= 1
def col(M,i): return [l[i] for l in M]
def lneg(l): return [-i for i in l]
def ladd(l1,l2): return [l1[idx]+l2[idx] for idx in range(len(l1))]
def lmul(l1,l2): return [l1[idx]*l2[idx] for idx in range(len(l1))]
def zeros(pad): return [type(pad[0]).zero for _ in range(len(pad))]
def lincombi(M,ais):
    l = zeros(M[0])
    for i in range(len(M)):
        l = ladd(l,smul(ais[i],M[i]))
    return l
def map2d(f,M): return [list(map(f,l)) for l in M]
def insertcol(M,idx,x): return [l[:idx]+[x]+l[idx:] for l in M]
def pad(l1, l2, val):
    a,b = len(l1),len(l2)
    if a>=b: return l1, l2+[val for _ in range(a-b)]
    else: return l1+[val for _ in range(b-a)], l2
