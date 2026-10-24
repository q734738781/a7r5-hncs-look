"""Exact sparse updates for discrete grid polishing with the same objective.

A one-integer coordinate move touches only eight-cell interpolation supports.
Regional means/variances update algebraically; no approximate loss is used.
"""
import numpy as np
import itertools
from .fit import lab
from .unified import grid_forward,residual_values

def robust_squared(q):
    return 32*(np.sqrt(1+q/16)-1)

class IntegerGridProblem:
    def __init__(self,src,loss,L,grid,*,grid_smooth=.3,grid_ridge=.005):
        self.src=src;self.loss=loss;self.L=L.toarray() if hasattr(L,'toarray') else np.asarray(L)
        self.grid_smooth=grid_smooth;self.grid_ridge=grid_ridge
        self.g=np.array(grid,float)/256
        self.A=grid_forward(src,np.zeros((405,3)))[2].tocsc()
        self.A.eliminate_zeros()
        self.raw=src+self.A@self.g;self.lv=lab(np.clip(self.raw,0,1))
        self.mean=loss.P@self.lv;self.second=loss.P@(self.lv*self.lv)
        self.pcost=self.point_cost(np.arange(len(src)),self.lv)
        self.rcost=self.region_cost(np.arange(len(loss.pw)),self.mean,self.second)
        self.reg=float(self.grid_smooth*np.sum(self.g*(self.L@self.g))+self.grid_ridge*np.sum(self.g*self.g))
        self.value=float(self.pcost.sum()+self.rcost.sum()+self.reg)
        self.support=[]
        for node in range(405):
            lo,hi=self.A.indptr[node:node+2];idx=self.A.indices[lo:hi];weights=self.A.data[lo:hi]
            pp=loss.P[:,idx];rows=np.unique(pp.nonzero()[0]);pp=pp[rows].tocsr()
            self.support.append((idx,weights,rows,pp))

    def point_cost(self,idx,lv):
        residual=residual_values(np.ascontiguousarray(lv),np.ascontiguousarray(self.loss.target[idx]))
        return self.loss.w[idx]*(.2*robust_squared(np.sum(residual**2,1))+
                    .3*robust_squared(np.sum(((lv-self.loss.target[idx])/self.loss.scale)**2,1)))

    def region_cost(self,rows,mean,second):
        pr=(mean-self.loss.mean[rows])/self.loss.scale
        std=np.sqrt(np.maximum(second-mean*mean,0)+.25)
        dr=(std-self.loss.std[rows])/self.loss.scale
        dist=np.linalg.norm(pr,axis=1)
        excess=np.zeros_like(dist) if self.loss.reference_distance is None else np.maximum(dist-self.loss.reference_distance[rows]-.3,0)
        return self.loss.pw[rows]*(.35*robust_squared(np.sum(pr*pr,1))+
                        .15*robust_squared(np.sum(dr*dr,1))+.5*excess*excess)

    def trial(self,node,channel,step,accept=False):
        steps=np.zeros(3);steps[channel]=step
        return self.trial_vector(node,steps,accept)

    def trial_vector(self,node,steps,accept=False):
        change=np.asarray(steps)/256
        if np.any(self.g[node]+change < -64/256) or np.any(self.g[node]+change>63/256):return np.inf
        idx,w,rows,pp=self.support[node]
        raw=self.raw[idx]+w[:,None]*change
        lv=lab(np.clip(raw,0,1));pc=self.point_cost(idx,lv)
        mean=self.mean[rows]+pp@(lv-self.lv[idx])
        second=self.second[rows]+pp@(lv*lv-self.lv[idx]*self.lv[idx])
        rc=self.region_cost(rows,mean,second)
        regdelta=self.grid_smooth*(2*change@(self.L[node]@self.g)+(change@change)*self.L[node,node])
        regdelta+=self.grid_ridge*(2*change@self.g[node]+change@change)
        delta=float(np.sum(pc-self.pcost[idx])+np.sum(rc-self.rcost[rows])+regdelta)
        if accept:
            self.raw[idx]=raw;self.lv[idx]=lv;self.pcost[idx]=pc
            self.mean[rows]=mean;self.second[rows]=second;self.rcost[rows]=rc
            self.reg+=regdelta;self.g[node]+=change;self.value+=delta
        return self.value if accept else self.value+delta

    def full_value(self):
        g=self.g
        return self.loss(np.clip(self.src+self.A@g,0,1),False)+self.grid_smooth*np.sum(g*(self.L@g))+self.grid_ridge*np.sum(g*g)

    def block_support(self,nodes):
        nodes=np.asarray(nodes,dtype=int)
        idx=np.unique(np.concatenate([self.support[n][0] for n in nodes]))
        weights=self.A[idx][:,nodes].toarray()
        pp=self.loss.P[:,idx];rows=np.unique(pp.nonzero()[0])
        return idx,weights,rows,pp[rows].tocsr()

    def trial_block(self,nodes,steps,accept=False,support=None):
        """Exact simultaneous moves, including interpolation and graph cross terms."""
        nodes=np.asarray(nodes,dtype=int);change=np.asarray(steps,dtype=float)/256
        if len(np.unique(nodes))!=len(nodes):raise ValueError('Duplicate block node')
        if change.shape!=(len(nodes),3):raise ValueError('One RGB vector per node required')
        if np.any(self.g[nodes]+change < -64/256) or np.any(self.g[nodes]+change>63/256):return np.inf
        idx,w,rows,pp=self.block_support(nodes) if support is None else support
        raw=self.raw[idx]+w@change;lv=lab(np.clip(raw,0,1));pc=self.point_cost(idx,lv)
        mean=self.mean[rows]+pp@(lv-self.lv[idx])
        second=self.second[rows]+pp@(lv*lv-self.lv[idx]*self.lv[idx])
        rc=self.region_cost(rows,mean,second)
        regdelta=self.grid_smooth*(2*np.sum(change*(self.L[nodes]@self.g))+
                        np.sum(change*(self.L[np.ix_(nodes,nodes)]@change)))
        regdelta+=self.grid_ridge*(2*np.sum(change*self.g[nodes])+np.sum(change*change))
        delta=float(np.sum(pc-self.pcost[idx])+np.sum(rc-self.rcost[rows])+regdelta)
        if accept:
            self.raw[idx]=raw;self.lv[idx]=lv;self.pcost[idx]=pc
            self.mean[rows]=mean;self.second[rows]=second;self.rcost[rows]=rc
            self.reg+=regdelta;self.g[nodes]+=change;self.value+=delta
        return self.value if accept else self.value+delta

    def sweep_pairs(self,tolerance=1e-11):
        """For each grid graph edge, jointly change each channel in four sign pairs."""
        edges=np.argwhere(np.triu(self.L<0,1));accepted=0;start=self.value
        directions=[]
        for c in range(3):
            for signs in itertools.product((-1,1),repeat=2):
                step=np.zeros((2,3),dtype=int);step[:,c]=signs;directions.append(step)
        for nodes in edges:
            support=self.block_support(nodes)
            values=[self.trial_block(nodes,d,support=support) for d in directions]
            best=int(np.argmin(values))
            if values[best]<self.value-tolerance:
                self.trial_block(nodes,directions[best],True,support);accepted+=1
        exact=float(self.full_value())
        if abs(exact-self.value)>1e-8:raise AssertionError(('incremental block loss drift',exact,self.value))
        self.value=exact
        return {'edges':len(edges),'moves_per_edge':len(directions),'accepted':accepted,
                'before':start,'after':exact,'improvement':start-exact}

    def sweep(self,step=1,tolerance=1e-11):
        accepted=0;start=self.value
        # Fixed order ensures repeatability and permits a complete no-move certificate.
        for node in range(405):
            for channel in range(3):
                a=self.trial(node,channel,-step);b=self.trial(node,channel,step)
                if min(a,b)<self.value-tolerance:
                    self.trial(node,channel,-step if a<=b else step,True);accepted+=1
        exact=float(self.full_value())
        if abs(exact-self.value)>1e-8:raise AssertionError(('incremental loss drift',exact,self.value))
        self.value=exact
        return {'step':step,'accepted':accepted,'before':start,'after':exact,'improvement':start-exact}

    def sweep_vectors(self,tolerance=1e-11):
        directions=[np.array(s) for s in itertools.product((-1,0,1),repeat=3) if s!=(0,0,0)]
        accepted=0;start=self.value
        for node in range(405):
            values=[self.trial_vector(node,d) for d in directions]
            best=int(np.argmin(values))
            if values[best]<self.value-tolerance:
                self.trial_vector(node,directions[best],True);accepted+=1
        exact=float(self.full_value())
        if abs(exact-self.value)>1e-8:raise AssertionError(('incremental vector loss drift',exact,self.value))
        self.value=exact
        return {'step':1,'joint_rgb_neighbors_per_node':26,'accepted':accepted,'before':start,'after':exact,'improvement':start-exact}
