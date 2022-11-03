from abc import ABC, abstractmethod
from util import *

class EuclideanDomain(ABC):
    @abstractmethod
    def __add__(self, other): pass
    @abstractmethod
    def __neg__(self): pass
    def __sub__(self, other): return self+(-other)
    @abstractmethod
    def __mul__(self, other): pass
    @abstractmethod
    def norm(self): pass
    @abstractmethod
    def div(self, other):
        assert(self.norm() >= other.norm())
        if other.norm()==0: return zero, zero
    def __truediv__(self, other):
        q, r = self.div(other)
        assert(r.norm()==0)
        return q
    @abstractmethod
    def __repr__(self): pass
    @abstractmethod
    def __eq__(self): pass

class Int(EuclideanDomain):
    def __init__(self, n):
        assert(type(n)==int)
        self.n = n
    def __add__(self, other): return Int(self.n + other.n)
    def __neg__(self): return Int(-self.n)
    def __mul__(self, other): return Int(self.n * other.n)
    def norm(self): return abs(self.n)
    def div(self, other):
        super().div(other)
        return Int(self.n // other.n), Int(self.n % other.n)
    def __repr__(self): return str(self.n)
    def __eq__(self, other): return self.n == other.n
Int.zero = Int(0)
Int.one = Int(1)

def trim(l):
    while True:
        if len(l)==1: return l
        if l[-1]==0: del l[-1]
        else: return l

class QPoly(EuclideanDomain):
    def __init__(self, coeffs):
        assert(all(type(x) in (int,float) for x in coeffs))
        self.coeffs = list(coeffs)
    def __add__(self, other):
        l1, l2 = pad(self.coeffs, other.coeffs, 0)
        return QPoly(trim(ladd(l1,l2)))
    def __neg__(self): return QPoly(lneg(self.coeffs))
    def __mul__(self, other):
        P = QPoly([0])
        for idx1, i in enumerate(self.coeffs):
            for idx2, j in enumerate(other.coeffs):
                P += QPoly.mono(idx1+idx2,i*j)
        return P
    def norm(self): return (0 if self.coeffs==[0] else len(self.coeffs))
    def lead(self): return self.coeffs[-1]
    def divaux(_self, other, P):
        m,n = _self.norm(),other.norm()
        if m<n: return P, _self
        super().div(other)
        M = QPoly.mono(m-n, _self.lead()/other.lead())
        return QPoly.divaux(_self-M*other, other, P+M)
    def div(self, other):
        return QPoly.divaux(self, other, QPoly([]))
    def __repr__(self): return '0' if self.norm()==0 else ' + '.join([f'{i}x^{self.norm()-idx-1}' for idx, i in enumerate(reversed(self.coeffs))])
    def __eq__(self, other): return self.coeffs == other.coeffs
QPoly.zero = QPoly([0])
QPoly.one = QPoly([1])
QPoly.mono = lambda n,x: QPoly([0 for _ in range(n)]+[x])