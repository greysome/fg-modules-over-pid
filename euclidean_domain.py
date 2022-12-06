from abc import ABC, abstractmethod
from functools import reduce
from copy import deepcopy
from typing import TypeAlias, Optional, Sequence, Union
from matrix import Matrix

Number: TypeAlias = Union[int, float]
def isnum(x):
    return isinstance(x, int) or isinstance(x, float)

EDType: TypeAlias = 'EuclideanDomain'
class EuclideanDomain(ABC):
    zero = NotImplemented
    one = NotImplemented

    @abstractmethod
    def __add__(self): pass

    @abstractmethod
    def __neg__(self): pass

    def __sub__(self, oth): return self + (-oth)

    @abstractmethod
    def __mul__(self, oth): pass

    @abstractmethod
    def __repr__(self): pass

    @abstractmethod
    def __eq__(self, oth): pass

    @property
    @abstractmethod
    def norm(self): pass

    @abstractmethod
    def div(self, oth) -> tuple[EDType, EDType]: pass

    def __floordiv__(self, oth):
        q, r = self.div(oth)
        return q

    def __mod__(self, oth):
        q, r = self.div(oth)
        return r

    def __truediv__(self, oth):
        assert oth.norm != 0, 'cannot divide by 0'
        q, r = self.div(oth)
        assert r.norm == 0
        return q

    '''
    Returns a, b such that ax+by = gcd(x,y).
    '''
    @staticmethod
    def _egcdaux(x, y) -> tuple[EDType,EDType]:
        typ = type(x)
        if x.norm == 0: return (typ.zero, typ.one)
        if y.norm == 0: return (typ.one, typ.zero)
        a, b = EuclideanDomain._egcdaux(y, x%y)
        return (b, a-b*(x//y))

    '''
    Returns gcd(x,y), as well as a, b such that ax+by = gcd(x,y).
    '''
    @staticmethod
    def egcd(x, y) -> tuple[EDType,EDType,EDType]:
        a, b = EuclideanDomain._egcdaux(x, y)
        return (a*x+b*y, a, b)

    @staticmethod
    def lcm(x: EDType, y: EDType) -> EDType:
        d, _, _ = egcd(x, y)
        return x*y/d

    '''
    egcd but for lists.
    '''
    @staticmethod
    def egcdl(l: Sequence[EDType]) -> tuple[EDType, Sequence[EDType]]:
        typ = type(l[0])
        if len(l) == 0: return (typ.zero, [])
        elif len(l) == 1: return (l[0], [typ.one])
        d, ais = egcdl(l[1:])
        d1, a1, b1 = egcd(l[0], d)
        return (d1, [a1] + [b1*ai for ai in ais])

egcd = EuclideanDomain.egcd
egcdl = EuclideanDomain.egcdl
lcm = EuclideanDomain.lcm


IntType: TypeAlias = 'Int'
class Int(EuclideanDomain):
    def __init__(self, n): self.n = n
    @property
    def norm(self): return abs(self.n)
    def __add__(self, oth): return Int(self.n + oth.n)
    def __neg__(self): return Int(-self.n)
    def __mul__(self, oth: Union[IntType, Matrix]):
        if isinstance(oth, Matrix):
            return oth.__rmul__(self)
        return Int(self.n * oth.n)
    def __repr__(self): return str(self.n)
    def __eq__(self, oth): return type(self) is type(oth) and self.n == oth.n
    def div(self, oth):
        if oth.norm == 0: return Int(0), Int(0)
        return Int(self.n // oth.n), Int(self.n % oth.n)
Int.zero = Int(0)
Int.one = Int(1)


PolyType: TypeAlias = 'Poly'
class Poly(EuclideanDomain):
    def __init__(self, coeffs: Optional[Sequence[Number]] = None) -> None:
        if coeffs is None: coeffs = [0]
        self.coeffs = coeffs

    @property
    def norm(self) -> int: return 0 if self.coeffs==[0] else len(self.coeffs)

    @property
    def lead(self) -> Number: return self.coeffs[-1]

    '''
    Argument can be a Number (int or float) or another Poly.
    '''
    def __add__(self, oth: Union[Number, PolyType]):
        if isnum(oth): oth = Poly([oth])
        # make lengths of self.coeffs and oth.coeffs equal by padding
        # the shorter list with 0s
        a, b = len(self.coeffs), len(oth.coeffs)
        L1, L2 = deepcopy(self.coeffs), deepcopy(oth.coeffs)
        if a >= b:
            L2 += [0 for i in range(a-b)]
        else:
            L1 += [0 for i in range(b-a)]
        # do the addition
        L = [L1[i]+L2[i] for i in range(max(a,b))]
        # trim excess 0s
        while True:
            if len(L) == 1 or L[-1] != 0: break
            if L[-1] == 0: del L[-1]
        return Poly(L)

    def __radd__(self, oth: Number):
        return self + oth

    '''
    Argument can be a Number (int or float) or another Poly.
    '''
    def __sub__(self, oth: Union[Number, PolyType]):
        return self + (-oth)

    def __rsub__(self, oth: Number):
        return self + (-oth)

    def __neg__(self): return Poly([-x for x in self.coeffs])

    '''
    Implements both scalar multiplication and polynomial multiplication.
    '''
    def __mul__(self, oth: Union[Number, PolyType, Matrix]):
        if isinstance(oth, Matrix):
            return oth.__rmul__(self)
        if isnum(oth):
            return Poly([oth*x for x in self.coeffs])
        P = Poly([0])
        for idx1, i in enumerate(self.coeffs):
            for idx2, j in enumerate(oth.coeffs):
                P += Poly([0]*(idx1+idx2) + [i*j])
        return P

    def __rmul__(self, oth: Number):
        return self.__mul__(oth)

    def _div(_self: PolyType, oth: PolyType, P: PolyType) -> tuple[PolyType,PolyType]:
        # greedily subtract away monomial multiples of oth,
        # accumulating the sum of multiples in P
        m, n = _self.norm, oth.norm
        if m < n: return P, _self
        super().div(oth)
        M = (_self.lead/oth.lead) * Poly([0,1])**(m-n)
        return Poly._div(_self-M*oth, oth, P+M)

    def div(self, oth: PolyType) -> tuple[PolyType,PolyType]:
        if oth.norm == 0: return Poly(), Poly()
        return Poly._div(self, oth, Poly())

    def __pow__(self, oth: int):
        assert oth >= 0, 'index must be non-negative'
        if oth == 0: return Poly([1])
        elif oth == 1: return deepcopy(self)
        return self * self.__pow__(oth-1)

    def __truediv__(self, oth: Union[Number, PolyType]):
        assert oth != 0 and oth != 0*C, 'cannot divide by 0'
        if isnum(oth):
            return 1/oth * self
        q, r = self.div(oth)
        assert r.norm == 0
        return q

    def normalise(self):
        if self == 0*C: return self
        return self / self.lead

    @staticmethod
    def _monomial_repr(idx: int, coeff: float) -> str:
        if idx == 0: return str(coeff)
        xpow_str = f'x' if idx == 1 else f'x^{idx}'
        return xpow_str if coeff == 1 else f'{coeff}{xpow_str}'

    def __repr__(self):
        if (n := self.norm) == 0: return '0'
        coeffplaces: list[tuple[int,float]] = []
        for idx, coeff in enumerate(reversed(self.coeffs)):
            if coeff == 0: continue
            coeffplaces.append((n-1-idx, coeff))

        res = ''
        for idx, p in enumerate(coeffplaces):
            s = Poly._monomial_repr(p[0], p[1])
            if idx == 0: res += s
            elif p[1] < 0: res += s
            else: res += '+' + s
        return res

    def __eq__(self, oth):
        return type(self) is type(oth) and self.coeffs == oth.coeffs

Poly.zero = Poly([0])
Poly.one = Poly([1])
X = Poly([0,1])
C = Poly([1])