from math import isqrt
from enum import Enum
from copy import deepcopy
from typing import Union, Sequence, Optional, TypeAlias, Any
from dataclasses import dataclass
from util import *

'''
End-of-column/row markers, to pass into the constructor `Matrix()`
'''
Marker = Enum('Marker', ('eoc', 'eor'))
eoc = Marker.eoc
eor = Marker.eor

MatrixType: TypeAlias = 'Matrix'
# ExtensionTyp because I have an another enum ExtensionType defined below.
ExtensionTyp: TypeAlias = 'Extension'

'''
A 2D array of elements of the same type. The public API provides two ways
to construct matrices: the raw `__init__` constructor, as well as a special
fancy constructor `Matrix()`.
'''
class Matrix:
    #==============#
    # CONSTRUCTORS #
    #==============#

    '''
    Constructs a matrix directly from a 2D array of entries, or an
    empty matrix if no arguments are passed.
    '''
    def __init__(self, data: Optional[Sequence[Sequence]] = None) -> None:
        if data is None: data = []
        assert alleq(len(row) for row in data), 'all rows must have the same length'
        assert alleq(flatten(data), type), 'all entries must have the same type'
        self.data = data

    @staticmethod
    def _build_rows(rowlen: int, entries: Sequence) -> MatrixType:
        assert len(entries) % rowlen == 0, f'number of entries must be a multiple of {rowlen}'
        mat = Matrix()
        i = 0
        while i < len(entries):
            curslice = entries[i:i+rowlen]
            mat.data.append(curslice)
            i += rowlen
        return mat

    @staticmethod
    def _build_cols(collen: int, entries: Sequence) -> MatrixType:
        assert len(entries) % collen == 0, f'number of entries must be a multiple of {collen}'
        return Matrix._build_rows(collen, entries).transpose

    '''
    `Matrix()` provides a convenient way to define matrices. There are three ways it can be used:
    - `M(*firstrow, eor, *remaining)` to list out entries row-wise
    - `M(*firstcol, eoc, *remaining)` to list out entries column-wise
    - `M(*entries)` if the number of entries is a perfect square, in which case a square matrix
    is created
    '''
    @staticmethod
    def M(*entries: Any) -> MatrixType:
        entries = list(entries)
        match (eoc in entries, eor in entries):
            case (True, True):
                raise AssertionError('argument cannot have both end-of-column and end-of-row marker')
            case (True, False):
                collen = entries.index(eoc)
                assert collen > 0, 'first column must be nonempty'
                del entries[collen]
                assert alleq(entries, type), 'type of all entries must be equal'
                return Matrix._build_cols(collen, entries)
            case (False, True):
                rowlen = entries.index(eor)
                assert rowlen > 0, 'first row must be nonempty'
                del entries[rowlen]
                assert alleq(entries, type), 'type of all entries must be equal'
                return Matrix._build_rows(rowlen, entries)
            case (False, False):
                n = len(entries)
                r = isqrt(n)
                assert r**2 == n, 'number of entries must be a perfect square'
                assert alleq(entries, type), 'type of all entries must be equal'
                if len(entries) == 0: return Matrix()
                else: return Matrix._build_rows(r, entries)

    '''
    A small wrapper around M. Used primarily to construct matrices over
    EuclideanDomain.Int and EuclideanDomain.GaussInt.
    '''
    @staticmethod
    def mapM(fn: Callable, *entries: Any) -> MatrixType:
        entries = list(entries)
        for idx, x in enumerate(entries):
            if type(x) != Marker:
                entries[idx] = fn(x)
        return M(*entries)

    #==================#
    # BASIC PROPERTIES #
    #==================#

    '''
    m and n are the number of rows and columns respectively.
    '''
    @property
    def m(self) -> int: return len(self.data)

    @property
    def n(self) -> int: return 0 if self.m == 0 else len(self.data[0])

    '''
    The typ of a matrix is the type of its elements (which are asserted to be equal),
    or None if the matrix is empty.
    '''
    @property
    def typ(self) -> type: return None if self.m == 0 else type(self.data[0][0])

    '''
    Default value associated with the matrix's type T. If T extends
    EuclideanDomain, then the default value is T.zero. Otherwise it is
    None.
    An example where `default` is used: given matrices M and `N = row() | M`,
    the first row of N is populated with M.default.
    '''
    @property
    def default(self) -> Any:
        if len(self.data) >= 1:
            try: return type(self.data[0][0]).zero
            except AttributeError: return 0.0
        else: return None

    '''
    Parse the argument `key` for __get/setitem__.

    If `key` is a 2-tuple and both components are non-slice indices, then return
    both non-slice indices, as well as an extra boolean True (indicating that `key`
    will be picking out a single element from the matrix rather than a submatrix).

    Otherwise, return a pair of slices and False.
    '''
    @staticmethod
    def _parse_key(key: Any) -> tuple[slice,slice,bool]:
        try: key[0]
        except TypeError:
            # `key` is a single index
            # In this case return the slice for the corresponding row(s)
            if isinstance(key, slice):
                return key, slice(None,None,None), False
            else:
                return slice(key,key+1,None), slice(None,None,None), False
        else:
            match len(key):
                case 2:
                    rows, cols = key
                    match (isinstance(rows, slice), isinstance(cols, slice)):
                        case (True, True): return rows, cols, False
                        case (True, False): return rows, slice(cols, cols+1), False
                        case (False, True): return slice(rows, rows+1), cols, False
                        case (False, False): return rows, cols, True
                case other:
                    raise AssertionError('key must have either 1 or 2 arguments')

    '''
    The functions __get/set/delitem__ accepts two types of arguments for the
    argument `key`.
    1. A single index (integer/slice), in which case the index acts on the rows.
    2. A pair of indices, in which case the first index acts on the rows and the
    second one on the indices.
    '''
    def __getitem__(self, key: Any) -> MatrixType:
        rows, cols, singleflag = Matrix._parse_key(key)
        if singleflag:
            return self.data[rows][cols]
        # If the column slice is empty, then return empty matrix rather than
        # a matrix with data [[]].
        if self.data[0][cols] == []:
            return Matrix()
        L = [row[cols] for row in self.data[rows]]
        return Matrix([row[cols] for row in self.data[rows]])

    '''
    Convert matrices with a single row/column to a list of entries.
    Otherwise convert it to a list of lists.
    '''
    @property
    def lst(self) -> list:
        if self.m == 1: return self.data[0]
        if self.n == 1: return [row[0] for row in self.data]
        return self.data

    @property
    def transpose(self) -> MatrixType:
        return Matrix([[self.data[i][j] for i in range(self.m)]
                       for j in range(self.n)])

    #==================#
    # BASIC OPERATIONS #
    #==================#

    def __setitem__(self, key: Any, value: Any):
        rows, cols, singleflag = Matrix._parse_key(key)
        if singleflag:
            assert isinstance(value, self.typ), 'types of LHS and RHS do not match'
            self.data[rows][cols] = value
            return
        assert isinstance(value, Matrix), 'RHS must be a matrix'
        assert value.typ is self.typ, 'types of LHS and RHS do not match'
        assert value.m == self[key].m and value.n == self[key].n, 'dimensions of LHS and RHS do not match'
        for idx1, i in enumerate(range(self.m)[rows]):
            for idx2, j in enumerate(range(self.n)[cols]):
                self.data[i][j] = value.data[idx1][idx2]

    '''
    __delitem__ does not use Matrix._parse_key as it handles 'single indices' in the same
    way as it does slices.
    '''
    def __delitem__(self, key: Any):
        try: key[0]
        except TypeError:  # deleting a row
            if not isinstance(key, slice):
                key = slice(key, key+1)
            del self.data[key]
        else:
            match len(key):
                case 2:
                    rows, cols = key[0], key[1]
                    if rows is None:
                        rows = slice(0,0,1)  # empty slice
                    elif not isinstance(rows, slice):
                        rows = slice(rows, rows+1)

                    if cols is None:
                        cols = slice(0,0,1)
                    elif not isinstance(cols, slice):
                        cols = slice(cols, cols+1)

                    del self.data[rows]
                    for i in range(self.m):
                        del self.data[i][cols]
                case other:
                    raise AssertionError('__delitem__ must have either 1 or 2 arguments')

    '''
    Element-wise operations. If `oth` is None, then nothing is done to
    the matrix.
    '''
    def __add__(self, oth: Optional[MatrixType]) -> MatrixType:
        if oth is None: return deepcopy(self)
        assert (self.n, self.m) == (oth.n, oth.m), 'dimensions do not match'
        return Matrix([[self[i,j]+oth[i,j] for j in range(self.n)] for i in range(self.m)])

    def __radd__(self, oth: Optional[MatrixType]) -> MatrixType:
        assert oth is None
        return self

    def __neg__(self) -> MatrixType:
        return Matrix([[-self[i,j] for j in range(self.n)] for i in range(self.m)])

    def __sub__(self, oth: Optional[MatrixType]) -> MatrixType:
        if oth is None: return deepcopy(self)
        return self + (-oth)

    def __rsub__(self, oth: Optional[MatrixType]) -> MatrixType:
        assert oth is None
        return self

    '''
    Matrix multiplication.
    '''
    def __mul__(self, oth: MatrixType) -> MatrixType:
        assert self.n == oth.m, 'matrix dimensions do not match'
        A = [[None] * oth.n for _ in range(self.m)]
        for i in range(self.m):
            for j in range(oth.n):
                A[i][j] = sum((self[i,k]*oth[k,j] for k in range(self.n)),
                               self.default)
        return Matrix(A)

    '''
    Matrix exponentiation.
    '''
    def __pow__(self, n: int) -> MatrixType:
        assert n >= 0, 'exponent must be nonnegative'
        assert self.n == self.m, 'matrix must be square'
        if n == 0: return I(self.typ, self.n)
        elif n == 1: return self
        return self * self.__pow__(n-1)

    '''
    Scalar multiplication.
    '''
    def __rmul__(self, oth: Any) -> MatrixType:
        assert isinstance(oth, self.typ)
        return Matrix([[oth*self[i,j] for j in range(self.n)]
                       for i in range(self.m)])

    '''
    There is an API around *extensions* to (nonempty) matrices, namely
    adding a new column / row.
    This is facilitated by the class Extension which removes the
    burden of having to specify the exact matrix data of the
    aforementioned column / row.
    Rather, this is done automatically by _inferext().
    Also, see docstring above Extension for examples.
    '''
    def _inferext(self, ext: ExtensionTyp) -> MatrixType:
        match ext.typ:
            case ExtensionType.row:
                match ext.datatyp:
                    case ExtensionDataType.none:
                        if self == Matrix():
                            return Matrix()
                        return M(*[self.default for _ in range(self.m)], eor)
                    case ExtensionDataType.single:
                        if self != Matrix():
                            assert isinstance(ext.data, self.typ), 'types do not match'
                        return M(*[ext.data for _ in range(self.m)], eor)
                    case ExtensionDataType.exact:
                        if self != Matrix():
                            assert isinstance(ext.data[0], self.typ), 'types do not match'
                            assert len(ext.data) == self.n, 'dimensions do not match'
                        return M(*ext.data, eor)
            case ExtensionType.col:
                match ext.datatyp:
                    case ExtensionDataType.none:
                        if self == Matrix():
                            return Matrix()
                        return M(*[self.default for _ in range(self.m)], eoc)
                    case ExtensionDataType.single:
                        if self != Matrix():
                            assert isinstance(ext.data, self.typ), 'types do not match'
                        return M(*[ext.data for _ in range(self.m)], eoc)
                    case ExtensionDataType.exact:
                        if self != Matrix():
                            assert isinstance(ext.data[0], self.typ), 'types do not match'
                            assert len(ext.data) == self.m, 'dimensions do not match'
                        return M(*ext.data, eoc)

    @staticmethod
    def _concat_horizontal(A: MatrixType, B: MatrixType) -> MatrixType:
        if A == Matrix(): return B
        elif B == Matrix(): return A
        assert A.m == B.m, 'dimensions do not match'
        assert A.typ == B.typ, 'types do not match'
        return Matrix([A.data[i] + B.data[i] for i in range(A.m)])

    @staticmethod
    def _concat_vertical(A: MatrixType, B: MatrixType) -> MatrixType:
        if A == Matrix(): return B
        elif B == Matrix(): return A
        assert A.n == B.n, 'dimensions do not match'
        assert A.typ == B.typ, 'types do not match'
        return Matrix(A.data + B.data)

    @staticmethod
    def _concat_diagonal(A: MatrixType, B: MatrixType) -> MatrixType:
        if A == Matrix(): return B
        elif B == Matrix(): return A
        assert A.typ == B.typ, 'types do not match'
        data = []
        for i in range(A.m): data.append(A.data[i] + [A.default for _ in range(B.n)])
        for i in range(B.m): data.append([B.default for _ in range(A.n)] + B.data[i])
        return Matrix(data)

    '''
    Extending a matrix to the right or bottom, or horizontally concatenating
    another matrix.
    Just as with __add__, if the other argument is None then nothing is done.
    '''
    def __or__(self, oth: Optional[Union[MatrixType, ExtensionTyp]]) -> MatrixType:
        if oth is None: return deepcopy(self)
        if isinstance(oth, Matrix):
            return Matrix._concat_horizontal(self, oth)
        elif isinstance(oth, Extension):
            match oth.typ:
                case ExtensionType.row:
                    return Matrix._concat_vertical(self, self._inferext(oth))
                case ExtensionType.col:
                    return Matrix._concat_horizontal(self, self._inferext(oth))

    '''
    Extending a matrix to the left or top. Note that concatenation of
    another matrix to the left is already handled in __or__.
    '''
    def __ror__(self, oth: Optional[ExtensionTyp]) -> MatrixType:
        if oth is None: return deepcopy(self)
        match oth.typ:
            case ExtensionType.row:
                return Matrix._concat_vertical(self._inferext(oth), self)
            case ExtensionType.col:
                return Matrix._concat_horizontal(self._inferext(oth), self)
      
    '''
    Vertically concatenating another matrix.
    Just as with __add__, if the other argument is None then nothing is done.
    '''
    def __and__(self, oth: Optional[MatrixType]) -> MatrixType:
        if oth is None: return deepcopy(self)
        return Matrix._concat_vertical(self, oth)

    '''
    This is to handle the case `None & M`.
    '''
    def __rand__(self, oth: Optional[MatrixType]) -> MatrixType:
        return self & oth

    '''
    Diagonally concatenating another matrix.
    '''
    def __xor__(self, oth: Optional[MatrixType]) -> MatrixType:
        if oth is None: return deepcopy(self)
        return Matrix._concat_diagonal(self, oth)

    def __rxor__(self, oth: Optional[MatrixType]) -> MatrixType:
        return self ^ oth

    def __eq__(self, oth: MatrixType) -> bool:
        return type(self) is type(oth) and self.data == oth.data

    #===============#
    # MISCELLANEOUS #
    #===============#

    def __repr__(self) -> str:
        if self.data == []: return '[]'
        s = ''
        max_lens = [max(len(repr(entry)) for entry in self[:,j].lst) for j in range(self.n)]
        for rowidx, row in enumerate(self.data):
            if self.m == 1: s += '[ '
            else:
                if rowidx == 0: s += '\u23a1 '
                elif rowidx == len(self.data)-1: s += '\u23a3 '
                else: s += '\u23a2 '

            for colidx, entry in enumerate(row):
                l = len(repr(entry))
                s += repr(entry) + ' ' * (max_lens[colidx]-l+1)

            if self.m == 1: s += ']'
            else:
                if rowidx == 0: s += '\u23a4\n'
                elif rowidx == len(self.data)-1: s += '\u23a6'
                else: s += '\u23a5\n'
        return s

    def find(self, f: Callable) -> Optional[tuple[int,int]]:
        for i in range(self.m):
            for j in range(self.n):
                if f(self.data[i][j]): return i,j
        else: return None

    #===============#
    # MATH ROUTINES #
    #===============#

    '''
    These are meant for matrices over a EuclideanDomain.
    Unfortunately importing the class (to add it as a type annotation)
    leads to an import loop.
    '''
    @property
    def det(self):
        assert self.m == self.n, 'det only defined for square matrices'
        if self.m == 1: return self[0,0]
        # apply the cofactor formula along the first column
        res = self.typ.zero
        for i in range(self.m):
            A = deepcopy(self)
            del A[i,0]
            x = self.data[i][0]
            res += x*A.det if i%2==0 else x*(-A.det)
        return res

    def cof(self, i: int, j: int):
        assert self.m == self.n, 'cof only defined for square matrices'
        A = deepcopy(self)
        del A[i,j]
        return A.det if (i+j)%2==0 else -A.det

    @property
    def adj(self) -> MatrixType:
        assert self.m == self.n, 'adj only defined for square matrices'
        return Matrix([[self.cof(i,j) for i in range(self.m)] for j in range(self.n)])

    @property
    def inv(self) -> MatrixType:
        assert self.m == self.n, 'inv only defined for square matrices'
        assert self.det.isunit, 'determinant must be a unit'
        if self.m == 1: return M(self.typ.one/self.det)
        return (self.typ.one/self.det) * self.adj

    '''
    The identity matrix.
    '''
    def I(typ: type, n: int) -> MatrixType:
        assert n >= 0
        if typ is None:
            return Matrix()
        elif typ in (float, int):
            one, zero = 1.0, 0.0
        else:
            one, zero = typ.one, typ.zero
        return Matrix([[zero for _ in range(i)] + [one] + [zero for _ in range(n-i-1)]
                       for i in range(n)])

    '''
    The next few functions deal with elementary row/column operations.
    In addition to mutating the original matrix, they return an
    elementary matrix corresponding to the operation.
    '''
    def swaprow(self, i: int, j: int) -> MatrixType:
        self[i], self[j] = self[j], self[i]
        A = I(self.typ, self.m)
        A[i,i], A[j,j], A[i,j], A[j,i] = \
            self.typ.zero, self.typ.zero, self.typ.one, self.typ.one
        return A

    def swapcol(self, i: int, j: int) -> MatrixType:
        self[:,i], self[:,j] = self[:,j], self[:,i]
        A = I(self.typ, self.n)
        A[i,i], A[j,j], A[i,j], A[j,i] = \
            self.typ.zero, self.typ.zero, self.typ.one, self.typ.one
        return A

    def scalerow(self, i: int, c: Any) -> MatrixType:
        assert isinstance(c, self.typ)
        self[i] *= c
        A = I(self.typ, self.m)
        A[i,i] = c
        return A

    def scalecol(self, j: int, c: Any) -> MatrixType:
        assert isinstance(c, self.typ)
        self[:,j] *= c
        A = I(self.typ, self.n)
        A[j,j] = c
        return A

    def addrow(self, i: int, j: int, c: Any) -> MatrixType:
        assert isinstance(c, self.typ)
        self[i] += c * self[j]
        A = I(self.typ, self.m)
        A[i,j] = c
        return A

    def addcol(self, i: int, j: int, c: Any) -> MatrixType:
        assert isinstance(c, self.typ)
        self[:,i] += c * self[:,j]
        A = I(self.typ, self.n)
        A[j,i] = c
        return A

M = Matrix.M
mapM = Matrix.mapM
I = Matrix.I


'''
There are two types of extensions to a matrix M: `row` and `col`. There
are three ways to specify an extension, corresponding to a different
ExtensionDataType.
- `none` means that the new row/col will be filled with `M.default`
- `single` means that the new row/col will be filled with a single specified entry
- `exact` means that a specified list of entries will be inserted

Examples:
> from euclidean_domain import Int
> A = mapM(Int,1,2,3,4)
> A | row()
⎡ 1 2 ⎤
⎢ 3 4 ⎥
⎣ 0 0 ⎦
> col(Int(5)) | A
⎡ 5 1 2 ⎤
⎣ 5 3 4 ⎦
> A | row(Int(5),Int(6))
⎡ 1 2 ⎤
⎢ 3 4 ⎥
⎣ 5 6 ⎦
'''
ExtensionType = Enum('ExtensionType', ('row', 'col'))
ExtensionDataType = Enum('ExtensionDataType', ('none', 'single', 'exact'))
class Extension:
    @staticmethod
    def _aux(*args: Any) -> ExtensionTyp:
        ext = Extension()
        if len(args) == 0:
            ext.data = None
            ext.datatyp = ExtensionDataType.none
        elif len(args) == 1:
            ext.data = args[0]
            ext.datatyp = ExtensionDataType.single
        else:
            assert alleq(args, type), 'types of all entries must be equal'
            ext.data = args
            ext.datatyp = ExtensionDataType.exact
        return ext
    
    @staticmethod
    def row(*args: Any) -> ExtensionTyp:
        ext = Extension._aux(*args)
        ext.typ = ExtensionType.row
        return ext

    @staticmethod
    def col(*args: Any) -> ExtensionTyp:
        ext = Extension._aux(*args)
        ext.typ = ExtensionType.col
        return ext

row = Extension.row
col = Extension.col
