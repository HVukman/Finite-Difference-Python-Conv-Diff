#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


# setting boundary points
def boundary(M,rhs,n,c_0,h,q):
    # changing Matrix M for zero values upper and lower boundary
    for i in range(0,n):     
            M[i,:]=0
            M[i,i]=1.0
            rhs[i]=0
    
    
    for i in range(n**2-(n-1),n**2):       
            M[i,:]=0
            M[i,i]=1.0
            rhs[i]=0
    
    
    # changing for initial values on left side
    for i in range(n+1,(n**2)-n,n):
        M[i,:]=0
        M[i,i]= 1.0
        rhs[i]=c_0
    
    
    # open boundary on right side 
    for i in range(2*n,n**2-n,n):
        M[i,:]=0
        M[i,i]= (1/h)*(1.5)-q
        M[i,i-1]=(-2.0/h)
        M[i,i-2]=0.5*h
        rhs[i]=c_0


    return [c,M]

# convection is stiff since q>> diffusion, reaction
Cycle=1
Max_Cycle=100
tau=0.01
L=0.02
q=1
diffusion=2.0*10e-6
reaction_coefficient=0.5*10e-6
source_coefficient=10e-6

# length between grid points
h=L/50

x=np.linspace(0,L, endpoint=True, retstep=h) 
x=x[0]
y=np.linspace(0,L, endpoint=True, retstep=h) 
y=y[0]
n=np.size(x)


# initial condition 
c_0=1;

#Matrix of result vector c and reshaping it
c=np.zeros(n**2,)
c=c.reshape(n,n,order='F').copy() 
c[0]=c_0
c=c.reshape(n**2,1,order='F').copy() 

# building identity matrices 
Ident_n=sparse.eye(n).toarray()
Ident=sparse.eye(n**2).toarray()

# discrete diffusion
T= diags([1, -2, 1], [-1, 0, 1], shape=(n,n)).toarray()
T= T*(1/h**2)
A= sparse.kron(T,Ident_n).toarray()+ sparse.kron(Ident_n,T).toarray()

# discrete convection
A1=diags([-1, 1, 0], [-1, 0, 1], shape=(n**2,n**2)).toarray()
A1=A1/h

# matrix algebra (IMEX-Scheme)
M=  Ident + tau*q/2*A1 
N = Ident + tau*diffusion*A  - (q*tau/2)*A1 - (tau/2)*reaction_coefficient*Ident + (tau/2)*source_coefficient*Ident


# right hand side
rhs=N.dot(c)

# Loop
while Cycle<Max_Cycle:
    # setting as lil_matrix, indexing is supposed to be faster
    M=lil_matrix(M)
    [c,M]=boundary(M,rhs,n,c_0,h,q)
    # setting as csr_matrix for solving
    M=csc_matrix(M)
    c_neu = spsolve(M,rhs)
    c_neu[c_neu < 0] = 0
    c=c_neu
    # new cycle right hand side
    rhs=N.dot(c_neu)
    Cycle = Cycle+1


# reshaping as matrix and plotting
Matrix_C=c.reshape(n,n,order='F').copy() 
[XX,YY] = np.meshgrid(x,y) 
fig, ax = plt.subplots()
CS = ax.contourf(XX, YY, np.transpose(Matrix_C))
cbar = plt.colorbar(CS)
plt.savefig("Countourf.png")



