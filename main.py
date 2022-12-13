from algos import *

#print(bosgb(mapM(Int,1,0,-1,eor,2,-3,1,0,3,1,3,1,5)))
#print(bosgb(M(2*X-1,X,X**2+3,X,X,X**2,X+1,2*X,2*X**2-3)))
## The solutions (a,b,c) that satisfy a+2b+3c=0 and a+4b+9c=0
## form the set {(-2b-3c, b, c) | 2b+6c = 0}.
#A = solve([Int(2),Int(6)])
#A = col(*[-Int(2)*b-Int(3)*c for b,c in A.data]) | A
## Actually the solution set has rank 1 so we don't even need to
## call bosgb. But I'm just demonstrating the principle.
#print(bosgb(A))

A1 = mapM(Int,6,2,3,0,2,3,-4,1,-3,3,1,2,-1,2,-3,5)
A2 = M(X-17,8*C,12*C,-14*C,-46*C,X+22,35*C,-41*C,2*C,-C,X-4,4*C,-4*C,2*C,2*C,X-3)
A3 = M(X+1,2*C,-6*C,C,X,-3*C,C,C,X-4)
A4 = mapM(Int,1,2,3,eor,4,5,6)
B1, P1, Q1 = smith(A1)
B2, P2, Q2 = smith(A2)
B3, P3, Q3 = smith(A3)
B4, P4, Q4 = smith(A4)

decompose(mapM(Int,2,1,-3,eor,1,-1,2))
A = mapM(GaussInt,1,3,6,(2,3),(0,-3),(12,-18),(2,-3),(6,-9),(0,-18))
decompose(mapM(GaussInt,1,3,6,(2,3),(0,-3),(12,-18),(2,-3),(6,-9),(0,-18)))
