#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:45:48 2019

"""
import numpy as np
import random
import math

def opt_space(matrix, mask, rank, niter=500, tol=1e-6):
    m0 = 10000
    rho= 0
    masked_matrix = matrix * mask
    n, m = np.shape(matrix)
    eps = np.sum(mask) / math.sqrt(n*m)
    
    # trimming
    rescal_param = math.sqrt(np.sum(mask)*rank / np.linalg.norm(masked_matrix, ord='fro')**2)
    masked_matrix = masked_matrix * rescal_param
    
#    trim_matrix = np.copy(masked_matrix)
#    
#    d = np.sum(mask, axis=0)
#    d_ = np.mean(d)
#    print(d, d_)
#    for col in range(m):
#        if np.sum(mask[:,col]) > 2 * d_:
#            l = np.where(mask[:,col]>0)
#            print(col, l)
#            np.random.shuffle(l)
    
    
#for col=1:m
#    if ( sum(E(:,col))>2*d_ )
#        list = find( E(:,col) > 0 );
#        p = randperm(length(list));
#        M_Et( list( p(ceil(2*d_):end) ) , col ) = 0;
#    end
#end
#
#d = sum(E');
#d_= mean(full(d));
#for row=1:n
#    if ( sum(E(row,:))>2*d_ )
#        list = find( E(row,:) > 0 );
#        p = randperm(length(list));
#        M_Et(row,list( p(ceil(2*d_):end) ) ) = 0;
#    end
#end
    X0, S0, Y0H = np.linalg.svd(masked_matrix, full_matrices=False)
    Y0 = Y0H.T
    
    X0 = X0[:,:rank] * math.sqrt(n)
    Y0 = Y0[:,:rank] * math.sqrt(m)
    S0 = S0[:rank] / eps
    
#    print(X0, Y0, S0)
    
    X = X0
    Y = Y0
    S = get_opt_s(X, Y, masked_matrix, mask)
#    print(S)
    
    # gradient descent 
    dist = []
    nnz = np.sum(mask)
    dist.append(np.linalg.norm( mask * (masked_matrix - X@S@Y.T) , ord='fro') / math.sqrt(nnz))
    
    for i in range(niter):
        W, Z = gradF_t(X, Y, S, masked_matrix, mask, m0, rho) #right
        t = get_opt_t(X, W, Y, Z, S, masked_matrix, mask, m0, rho)
        X = X + t * W
        Y = Y + t * Z
        S = get_opt_s(X, Y, masked_matrix, mask)
#        print("X: ", X.T)
#        print("Y: ", Y.T)
#        print("S: ", S)
#        print(np.linalg.norm(mask * (masked_matrix - X@S@Y.T) , ord='fro'))
        dist.append(np.linalg.norm(mask * (masked_matrix - X@S@Y.T) , ord='fro') /  math.sqrt(nnz))
        if dist[-1] < tol or (dist[-2] - dist[-1]) < tol:
            break
        
    S = S /rescal_param
    
    return X, S, Y, dist

def gradF_t(X, Y, S, M_E, E, m0, rho):
    n, r = np.shape(X)
    m, r = np.shape(Y)
    
    XS = X @ S
    YS = Y @ S.T
    XSY = XS @ Y.T
    
    QX = X.T @ ((M_E - XSY) * E) @ YS / n
    QY = Y.T @ ((M_E - XSY) * E).T @ XS /m
    W = ((XSY - M_E) * E) @ YS + X @ QX + rho * Gp(X, m0, r)
    Z = ((XSY - M_E) * E).T @ XS + Y @ QY + rho * Gp(Y, m0, r)
    
    return W, Z

def Gp(X, m0, r):
    z = np.sum(np.power(X, 2), axis=1) / (2 * m0 * r)
    z = 2 * np.exp(np.power(z-1, 2)) * (z-1)
    z[np.where(z<0)] = 0
    
    return X * np.tile(z, (r,1)).T / (m0 * r)
        

def get_opt_s(X, Y, M_E, E):
    n, r = np.shape(X)
    C = X.T @ ( M_E ) @ Y
    C = C.reshape(r**2, 1)
    A = np.zeros((r**2, r**2))
    for j in range(r):
        for i in range(r):
            ind = j * r + i 
            temp = X.T @ ( np.outer(X[:,i], Y[:,j]) * E) @ Y
            A[:,ind] = temp.reshape(r**2, )
#    print(A)
#    print(np.shape(A))
    S = C.T @ np.linalg.pinv(A)
#    print(S)
    
    return S.reshape(r, r)


def get_opt_t(X, W, Y, Z, S, M_E, E, m0, rho):
    norm2WZ = np.linalg.norm(W, ord='fro')**2 + np.linalg.norm(Z, ord='fro')**2
    f = []
    f.append(F_t(X, Y, S, M_E, E, m0, rho))
    t = -1e-1
    for i in range(20):
        f.append(F_t(X+t*W, Y+t*Z, S, M_E, E, m0, rho))
        if f[-1] - f[0] <= 0.5 * t * norm2WZ:
            return t
        t = t / 2
    return t
    
def F_t(X, Y, S, M_E, E, m0, rho):
    n, r = np.shape(X)
    out1 = np.sum( np.sum( np.power(E * (X@S@Y.T - M_E), 2), axis=1) ) / 2
    out2 = rho * G(Y, m0, r)
    out3 = rho * G(X, m0, r)
    out = out1 + out2 + out3
    return out

def G(X, m0, r):
    z = np.sum(np.power(X, 2), axis=1) / (2 * m0 * r)
    y = np.exp(np.power(z-1, 2)) - 1
    y[np.where(z<1)] = 0
    return np.sum(y)

def const_mask(m, n, eps):
    mask = np.ones((m,n))
    for i in range(m):
        for j in range(n):
            if random.random() < eps:
                mask[i][j] = 0
    return mask

def max_diff(M1, M2):
    return np.max(np.abs(M1-M2))

def fro_diff(M1, M2):
    return np.linalg.norm((M1-M2), ord='fro')

if __name__ == "__main__":
    n = 20
    m = 10
    r = 2
    
    diffs = []
    for i in range(10):
        print(i)
        u = np.random.rand(n, r)
        v = np.random.rand(m, r)
        s = np.random.rand(r)
        M = np.dot(u*s, v.T)
        noise = np.random.randn(n, m) * 0.05
#        print("M: ", M)
#        print("noi: ", noise)
    #    print("u:", u)
    #    print("v:", v)
    #    print("s:", s)
        
    #    M = np.array([[.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1]])
    #    print(M)
    #    n = 5
    #    m = 6
    #    r = 1
        
    #    mask = np.array([[0. ,0., 1. ,1. ,1. ,1.],
    #                     [1. ,1. ,1. ,0. ,1. ,0.],
    #                     [0. ,1. ,0. ,1. ,0. ,1.],
    #                     [1. ,1. ,0. ,1. ,1. ,0.],
    #                     [1. ,1. ,1., 1. ,1. ,1.]])
        mask = const_mask(n, m, 0.3)
#        print(mask)
        X, S, Y, dist = opt_space(M+noise, mask, r, niter=500)
#        print(dist)
        print("dist: ", len(dist), dist[-1])
        res = X @ S @ Y.T
        print(max_diff(res, M))
        diffs.append(max_diff(res, M))
    print("mean max error: ", np.mean(np.array(diffs)))