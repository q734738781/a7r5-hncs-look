"""Portable construction of the released HNCS-only appearance objective."""
import colorsys
import numpy as np
from .bootstrap import ROOT
from .dataset import load_manifest
from .pooling import build_pool
from .optimizer.unified import AppearanceLoss,JointObjective
from .optimizer.hncs_highlight import OperatorStabilityLoss
from .optimizer.fit import lab

OBJECTIVE_ID='hncs-only-appearance-self-stability-v2'
GRID_SMOOTH=.3
ENDPOINT_SMOOTH=.01

def quadrature(nh=18,ns=4,ne=7,hue_offset=.31):
    sn,sw=np.polynomial.legendre.leggauss(ns);en,ew=np.polynomial.legendre.leggauss(ne)
    values=[];weights=[]
    for hue in (np.arange(nh)+hue_offset)/nh:
        for saturation,ws in zip((sn+1)*.49,sw/2):
            colour=np.array(colorsys.hsv_to_rgb(hue,float(saturation),1))
            for exposure,we in zip((en+1)*(13.7-5)/2+5,ew/2):
                values.append(colour*2**exposure);weights.append(ws*we/nh)
    return np.asarray(values),np.asarray(weights)

def regularization(theta,grid,L):
    cost=GRID_SMOOTH*np.sum(grid*(L@grid))
    for start,width in ((3,6),(21,8)):
        cost+=ENDPOINT_SMOOTH*np.sum(np.diff(theta[start:start+3*width].reshape(3,width),axis=0)**2)
    return float(cost)

def joint_objective(batches,loss):
    return JointObjective(batches,loss,np.zeros(1260),prior_strength=0.,grid_ridge=0.,
        grid_smooth=GRID_SMOOTH,endpoint_smooth=ENDPOINT_SMOOTH)

def make_problem(dataset,stride=2,strength=.05):
    from .sony.physical import SonyPhysicalLookRenderer,BASELINE_ID
    from .teacher.reference import HNCSReference,TEACHER_ID
    doc,frames=load_manifest(dataset)
    training=[row for row in frames if row['split']=='construction']
    batches=[];targets=[];weights=[];groups=[];rays=[];ray_weights=[];neutrals=[];offset=0
    def append(renderer,x,y=None,weight=None):
        nonlocal offset
        start=offset;offset+=len(x);batches.append((renderer,x,renderer.prepare(x)))
        targets.append(np.zeros((len(x),3)) if y is None else y)
        weights.append(np.zeros(len(x)) if weight is None else weight)
        return np.arange(start,offset)
    def add_rays(renderer,x,weight):
        if not len(x):return
        factors=2**(np.arange(-2,3)/2)
        source=np.rint(np.clip(x[:,None,:]*factors[None,:,None],0,65535)).astype(np.uint16)
        ids=append(renderer,source.reshape(-1,3)).reshape(-1,5)
        rays.extend(ids);ray_weights.extend(np.full(len(ids),weight/len(ids)))
    for row in training:
        m,z=row['metadata'],row['data'];renderer=SonyPhysicalLookRenderer(m['main'],m['st'])
        x=z['rgb'][::stride];y=z['target'][::stride];valid=z['valid'][::stride];patch=z['patch'][::stride]
        if not valid.any():raise ValueError('Construction frame has no valid target pixels')
        ids=append(renderer,x,y,.82*valid/(valid.sum()*len(training)))
        groups.append((ids[valid],.25/len(training)))
        for region in range(24):
            take=valid&((patch//12//2)*6+patch%12//2==region)
            if take.any():groups.append((ids[take],.5/(24*len(training))))
        lightness=lab(y)[:,0];cuts=np.quantile(lightness[valid],np.linspace(0,1,6))
        for j,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            take=valid&(lightness>=lo)&((lightness<=hi) if j==4 else lightness<hi)
            if take.any():groups.append((ids[take],.25/(5*len(training))))
        sx=z['stress_rgb'];teacher=HNCSReference(m['main'],m['st'])
        bright=teacher.render(sx).max(1)>.6 if len(sx) else np.zeros(0,bool)
        selected=np.flatnonzero((~z['stress_valid'])&bright)
        if len(selected)>32:selected=selected[np.linspace(0,len(selected)-1,32).astype(int)]
        add_rays(renderer,sx[selected],.8/len(training))
    # Explicit state IDs allow reproducing a known calibration set. Without
    # them, use up to seven construction states; never use validation pixels.
    state_ids=doc.get('calibration_state_ids',[row['id'] for row in training[:7]])
    state_rows=[]
    for key in state_ids:
        match=next((row for row in frames if row['id']==key),None)
        if match is None:raise ValueError('Unknown calibration state ID')
        state_rows.append(match)
    if not state_rows:raise ValueError('At least one explicit calibration state is required')
    sx,sw=quadrature();sx=np.rint(sx).astype(np.uint16)
    for state in state_rows:
        m=state['metadata'];r=SonyPhysicalLookRenderer(m['main'],m['st']);teacher=HNCSReference(m['main'],m['st'])
        y,valid=teacher.fitting_target(sx,np.ones(len(sx),bool))
        if not valid.any():raise ValueError('Calibration state has no valid quadrature support')
        append(r,sx,y,.18*sw*valid/(sw@valid)/len(state_rows))
        directions=np.array([colorsys.hsv_to_rgb(h/12,s,1.) for s in (.04,.2) for h in range(12)])
        probe=np.rint(np.concatenate([directions*level for level in (2048,8192,16384)])).astype(np.uint16)
        probe=probe[teacher.render(probe).max(1)>.6]
        add_rays(r,probe,.2/len(state_rows))
        gray=np.repeat(np.rint(np.geomspace(64,65535,80)).astype(np.uint16)[:,None],3,1)
        neutrals.append(append(r,gray))
    target=np.concatenate(targets);weight=np.concatenate(weights);P,pw=build_pool(groups,len(target))
    appearance=AppearanceLoss(target,weight,P,pw,reference=None)
    loss=OperatorStabilityLoss(appearance,rays,ray_weights,neutrals,strength)
    if np.any(appearance.w[loss.ids]):raise AssertionError('Stability samples must not receive colour-target weight')
    contract=dict(objective_id=OBJECTIVE_ID,teacher=TEACHER_ID,baseline=BASELINE_ID,
        construction_photos=len(training),validation_photos=sum(r['split']=='validation' for r in frames),
        calibration_states=len(state_rows),validation_is_blind=False,sampling_stride=stride,
        ST_reference_loss=False,old_look_reference_loss=False,old_parameter_distance_penalty=0,
        appearance_weights=dict(de00=.2,lab=.3,pooled_mean=.35,pooled_spread=.15),
        photograph_weight=.82,operator_quadrature_weight=.18,grid_smooth=GRID_SMOOTH,
        endpoint_smooth=ENDPOINT_SMOOTH,stability_strength=strength,nodes=len(target),
        stability_nodes=len(loss.ids),highlight_rays=len(rays),dro_applications=1,
        per_scene_routing=False,runtime_defringe=False,firmware=False)
    return batches,loss,contract
