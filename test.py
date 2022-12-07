from matrix import *
from euclidean_domain import *
import unittest

class MatrixTest(unittest.TestCase):
    def test_init(self):
        A = Matrix()
        self.assertEqual(A.data, [])

    def test_M(self):
        A = M()
        self.assertEqual(A.data, [])
        A = M(1,2,3,4)
        self.assertEqual(A.data, [[1,2],[3,4]])
        self.assertRaises(AssertionError, M, 1,2,3,4,5,6)
        A = M(1,2,3,eoc,4,5,6)
        self.assertEqual(A.data, [[1,4],[2,5],[3,6]])
        self.assertRaises(AssertionError, M, 1,2,3,eoc,4,5,6,7)
        A = M(1,2,3,eor,4,5,6)
        self.assertEqual(A.data, [[1,2,3],[4,5,6]])
        self.assertRaises(AssertionError, M, 1,2,3,eor,4,5,6,7)
        A = mapM(Int,1,2,3,4)
        self.assertEqual(A.data, [[Int(1),Int(2)],[Int(3),Int(4)]])

    def test_props(self):
        self.assertEqual(M().m, 0)
        self.assertEqual(M().n, 0)
        self.assertEqual(M(1,2,3,eor,4,5,6).m, 2)
        self.assertEqual(M(1,2,3,eor,4,5,6).n, 3)
        self.assertEqual(M().typ, None)
        self.assertEqual(mapM(Int,1,2,3,4).typ, Int)
        self.assertEqual(M().default, None)
        self.assertEqual(M(1,2,3,4).default, None)
        self.assertEqual(mapM(Int,1,2,3,4).default, Int.zero)

    def test_getitem(self):
        A = M(1,2,3,4,5,6,7,8,9)
        self.assertEqual(A[0], M(1,2,3,eor))
        self.assertEqual(A[1:], M(4,5,6,eor,7,8,9))
        self.assertEqual(A[:,0], M(1,4,7,eoc))
        self.assertEqual(A[0,0], 1)
        self.assertEqual(A[0,:2], M(1,2,eor))
        self.assertEqual(A[1:,1:], M(5,6,8,9))

    def test_setitem(self):
        A = M(1,2,3,4,5,6,7,8,9)
        A[0] = M(0,0,0,eor)
        self.assertEqual(A, M(0,0,0,4,5,6,7,8,9))
        A = M(1,2,3,4,5,6,7,8,9)
        A[1:] = M(0,0,0,eor,0,0,0)
        self.assertEqual(A, M(1,2,3,0,0,0,0,0,0))
        A = M(1,2,3,4,5,6,7,8,9)
        A[0,0] = 0
        self.assertEqual(A, M(0,2,3,4,5,6,7,8,9))
        A = M(1,2,3,4,5,6,7,8,9)
        A[1:,1:] = M(0,0,0,0)
        self.assertEqual(A, M(1,2,3,4,0,0,7,0,0))
        A = M(1,2,3,4,5,6,7,8,9)
        A[:,0] = M(0,0,0,eoc)
        self.assertEqual(A, M(0,2,3,0,5,6,0,8,9))

    def test_lst(self):
        self.assertEqual(M().lst, [])
        self.assertEqual(M(1,2,3,eor).lst, [1,2,3])
        self.assertEqual(M(1,2,3,eoc).lst, [1,2,3])
        self.assertEqual(M(1,2,3,4).lst, [[1,2],[3,4]])

    def test_ops(self):
        self.assertEqual(M(1,2,3,eor,4,5,6).transpose, M(1,2,3,eoc,4,5,6))
        self.assertEqual(M(1,1,1,1) + M(1,1,1,1), M(2,2,2,2))
        self.assertEqual(None + M(1,1,1,1), M(1,1,1,1))
        self.assertEqual(M(1,1,1,1) + None, M(1,1,1,1))
        self.assertEqual(-M(1,1,1,1), M(-1,-1,-1,-1))
        self.assertEqual(2 * M(1,1,1,1), M(2,2,2,2))

    def test_ext(self):
        self.assertEqual(M(1,2,3,4)._inferext(row()), M(None,None,eor))
        self.assertEqual(M(1,2,3,4)._inferext(col()), M(None,None,eoc))
        self.assertEqual(mapM(Int,1,2,3,4)._inferext(row()), M(Int.zero,Int.zero,eor))
        self.assertEqual(M(1,2,3,4)._inferext(col(5)), M(5,5,eoc))
        self.assertRaises(AssertionError, M(1,2,3,eor,4,5,6)._inferext, col(7,8,9))
        self.assertEqual(M(1,2,3,eor,4,5,6)._inferext(col(5,6)), M(5,6,eoc))
        self.assertEqual(Matrix._concat_horizontal(M(1,2,3,4),M(5,6,7,8)),
                            M(1,2,5,6,eor,3,4,7,8))
        self.assertEqual(Matrix._concat_vertical(M(1,2,3,4),M(5,6,7,8)),
                            M(1,2,eor,3,4,5,6,7,8))
        A = mapM(Int,1,2,3,4)
        self.assertEqual(A | row(), mapM(Int,1,2,eor,3,4,0,0))
        self.assertEqual(row() | A, mapM(Int,0,0,eor,1,2,3,4))
        self.assertEqual(A | col(), mapM(Int,1,2,0,eor,3,4,0))
        self.assertEqual(col() | A, mapM(Int,0,1,2,eor,0,3,4))
        self.assertEqual(A | row(Int(5)), mapM(Int,1,2,eor,3,4,5,5))
        self.assertEqual(A | row(Int(5),Int(6)), mapM(Int,1,2,eor,3,4,5,6))
        self.assertEqual(A | col(Int(5)), mapM(Int,1,2,5,eor,3,4,5))
        self.assertEqual(A | col(Int(5),Int(6)), mapM(Int,1,2,5,eor,3,4,6))
        self.assertEqual(A | None, A)
        self.assertEqual(None | A, A)
        self.assertEqual(A & mapM(Int,5,6,7,8), mapM(Int,1,2,eor,3,4,5,6,7,8))
        self.assertEqual(mapM(Int,5,6,7,8) & A, mapM(Int,5,6,eor,7,8,1,2,3,4))
        self.assertEqual(A & None, A)
        self.assertEqual(None & A, A)

class PolyTest(unittest.TestCase):
    def test_C_and_X(self):
        self.assertEqual(C.coeffs, [1])
        self.assertEqual((2*C).coeffs, [2])
        self.assertEqual((C*2).coeffs, [2])
        self.assertEqual(X.coeffs, [0,1])
        self.assertEqual((2*X).coeffs, [0,2])
        self.assertEqual((X*2).coeffs, [0,2])
        self.assertEqual((X**2).coeffs, [0,0,1])
        self.assertEqual((2*X**2).coeffs, [0,0,2])
        self.assertEqual((3*X**2+2*X+1).coeffs, [1,2,3])
        self.assertEqual((1+2*X+3*X**2).coeffs, [1,2,3])

    def test_norm(self):
        self.assertEqual((0*C).norm, 0)
        self.assertEqual(C.norm, 1)
        self.assertEqual(X.norm, 2)
        self.assertEqual((X**2).norm, 3)
        self.assertEqual((X**2+X+1).norm, 3)

    def test_lead(self):
        self.assertEqual((0*C).lead, 0)
        self.assertEqual((1*C).lead, 1)
        self.assertEqual((2*X+1).lead, 2)

    def test_add_sub(self):
        self.assertEqual(C+C, 2*C)
        self.assertEqual(X+X, 2*X)
        self.assertEqual(X-X, 0*C)
        self.assertEqual((X**2+X)+(X**3+X**2), X**3+2*X**2+X)

    def test_mul_pow(self):
        self.assertEqual(X*X, X**2)
        self.assertEqual((X+1)**3, X**3+3*X**2+3*X+1)
        self.assertEqual(X*(X+2), X**2+2*X)

    def test_div(self):
        self.assertEqual((X**2).div(X), (X,0*C))
        self.assertEqual((X**2+1).div(X), (X,1*C))
        self.assertEqual((1*C).div(X), (0*C,1*C))
        self.assertRaises(AssertionError, X.__truediv__, 0)
        self.assertRaises(AssertionError, X.__truediv__, 0*C)
        self.assertEqual((2*X)/2, X)

class MathTest(unittest.TestCase):
    def test_egcd(self):
        o = Int.one
        z = Int.zero
        self.assertEqual(egcd(o,z), (o,o,z))
        self.assertEqual(egcd(z,o), (o,z,o))
        d,a,b = egcd(Int(6),Int(9))
        self.assertEqual(d, Int(3))
        self.assertEqual(a*Int(6)+b*Int(9), d)
        d,a,b = egcd(Int(2),Int(3))
        self.assertEqual(d, Int(1))
        self.assertEqual(a*Int(2)+b*Int(3), d)

        d,a,b = egcd(X,X+1)
        self.assertEqual(d.normalise(), C)
        self.assertEqual(a*X+b*(X+1), d)

    def test_egcdl(self):
        self.assertEqual(egcdl([Int(1)]), (Int(1),[Int(1)]))
        d,(a,b,c) = egcdl([Int(0), Int(2), Int(3)])
        self.assertEqual(d, Int(1))
        self.assertEqual(a*Int(0)+b*Int(2)+c*Int(3), d)
        d,(a,b,c) = egcdl([Int(6), Int(8), Int(10)])
        self.assertEqual(d, Int(2))
        self.assertEqual(a*Int(6)+b*Int(8)+c*Int(10), d)

    def test_lcm(self):
        self.assertEqual(lcm(Int(1),Int(0)), Int(0))
        self.assertEqual(lcm(Int(2),Int(3)), Int(6))
        self.assertEqual(lcm(Int(2),Int(4)), Int(4))
        self.assertEqual(lcm(Int(6),Int(9)), Int(18))

if __name__ == '__main__':
    unittest.main()