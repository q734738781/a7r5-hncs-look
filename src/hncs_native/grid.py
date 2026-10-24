"""Declared BT.601-limited 5x9x9 grid operator, in display RGB.

Adapted from QAQLeonard/a7c2-fl23-research, looktool/native.py (MIT).
Copyright (c) 2026 A7C II FL2/FL3 preview contributors.
See licenses/a7c2-fl23-research.txt and THIRD_PARTY_NOTICES.md.
This is the fixed software model used by this project, not an ISP proof.
"""
import itertools
import numpy as np

def coordinates(rgb):
    kr, kb = .299, .114
    kg = 1-kr-kb
    matrix = np.array([[kr,kg,kb],[-kr/(2*(1-kb)),-kg/(2*(1-kb)),.5],
                       [.5,-kg/(2*(1-kr)),-kb/(2*(1-kr))]])
    return (np.asarray(rgb).reshape(-1,3)@matrix.T-[16/255,0,0])*255/np.array([219,224,224])

def interpolation(rgb):
    at = np.clip(coordinates(rgb)*[4,8,8]+[0,4,4], [0,0,0], [4,8,8])
    lo = np.minimum(at.astype(int), [3,7,7]); f = at-lo
    ids, weights = [], []
    for dy,dc1,dc2 in itertools.product((0,1),repeat=3):
        ids.append((lo[:,0]+dy)*81+(lo[:,1]+dc1)*9+lo[:,2]+dc2)
        weights.append(np.prod(np.where([dy,dc1,dc2],f,1-f),axis=1))
    return np.stack(ids,1),np.stack(weights,1)

def predict(source,coefficients,response=None):
    rgb=np.asarray(source).reshape(-1,3)
    response=np.eye(3) if response is None else np.asarray(response,float)
    if response.shape!=(3,3) or not np.isfinite(response).all() or np.linalg.cond(response)>30:
        raise ValueError('Invalid grid response matrix')
    ids,weights=interpolation(rgb)
    delta=np.sum(np.asarray(coefficients)[ids]*weights[:,:,None],axis=1)/256
    return np.clip(rgb+delta@response,0,1)

def smoothness():
    laplacian=np.zeros((405,405))
    for y,c1,c2 in itertools.product(range(5),range(9),range(9)):
        i=y*81+c1*9+c2
        for condition,j in ((y<4,i+81),(c1<8,i+9),(c2<8,i+1)):
            if condition:
                laplacian[i,i]+=1;laplacian[j,j]+=1
                laplacian[i,j]-=1;laplacian[j,i]-=1
    return laplacian
