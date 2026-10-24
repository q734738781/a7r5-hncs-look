"""Analytic guard against amplifying unreliable highlight chroma.

The mask selects constraints during fitting only. The deployed software Look
has one fixed native parameter set and one fixed grid; no runtime defringe mask.
"""
import numpy as np

PAIRS=np.array([[1.,-1.,0.],[0.,-1.,1.],[1.,0.,-1.]])
LUMA=np.array([.2126,.7152,.0722])
CENTRE=np.arange(5)*81+40

def project_neutral_grid(grid):
    g=np.array(grid,copy=True).reshape(405,3)
    g[CENTRE]=g[CENTRE].mean(1)[:,None]
    g[CENTRE[[0,4]]]=0
    return g


class NeutralProjectedObjective:
    def __init__(self,objective):self.objective=objective
    def project(self,z):
        o=self.objective;q=z.copy()
        g=(o.initial[45:]+q[45:]*o.scale[45:]).reshape(405,3)
        q[45:]=(project_neutral_grid(g).ravel()-o.initial[45:])/o.scale[45:]
        return q
    def __call__(self,z):
        o=self.objective;c,g=o(self.project(z));g=g.copy()
        g[45:]=project_neutral_grid(g[45:].reshape(405,3)/o.scale[45:].reshape(405,3)).ravel()*o.scale[45:]
        return c,g
