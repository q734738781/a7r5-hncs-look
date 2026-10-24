"""Analytic joint fit and exact integer polish, HNCS-only colour objective."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from hncs_native.problem import make_problem,joint_objective,regularization,GRID_SMOOTH,ROOT
from hncs_native.optimizer.operator_math import LOW,HIGH
from hncs_native.optimizer.highlight import NeutralProjectedObjective,project_neutral_grid
from hncs_native.optimizer.hncs_highlight import HncsIntegerGridProblem
from hncs_native.optimizer.unified import grid_forward
from hncs_native.optimizer.reverse import forward_pullback
from hncs_native.optimizer.fit import vector_of,from_vector
from hncs_native.runtime import render_fast
from hncs_native.format import read_look,write_look as export_look
from hncs_native.io import dump


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--start',type=Path,default=ROOT/'looks/HNCS_Unified_Only.a7look')
    p.add_argument('--strength',type=float,default=.05);p.add_argument('--maxiter',type=int,default=600)
    p.add_argument('--seconds',type=float,default=1200);p.add_argument('--integer-sweeps',type=int,default=3)
    p.add_argument('--resume',type=Path);p.add_argument('--grid-iterations',type=int,default=600);a=p.parse_args(argv);folder=a.output;a.name=folder.name
    if folder.exists():raise ValueError('Use a new result directory')
    folder.mkdir(parents=True);(folder/'looks').mkdir();start=time.perf_counter()
    batches,loss,contract=make_problem(dataset=a.dataset,strength=a.strength)
    contract.update(initializer=a.start.name,initializer_sha256=hashlib.sha256(a.start.read_bytes()).hexdigest(),
        solver='analytic joint L-BFGS-B, fixed-native sparse grid polish, exact integer coordinate descent')
    if a.resume:
        contract.update(continuous_initializer=a.resume.name,continuous_initializer_sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest())
    dump(folder/'contract.json',contract)
    fn=joint_objective(batches,loss);objective=NeutralProjectedObjective(fn)
    initial_look=read_look(a.start);initial=np.r_[vector_of(initial_look,True),project_neutral_grid(initial_look.grid/256).ravel()]
    if a.resume:
        with np.load(a.resume) as z:initial=np.r_[z['theta'],z['grid'].ravel()]
    lower=np.r_[LOW,np.full(1215,-.25)]/fn.scale;upper=np.r_[HIGH,np.full(1215,63/256)]/fn.scale
    z=np.clip(initial/fn.scale,lower,upper);first=objective(z)[0];history=[]
    print('START',a.name,'nodes',contract['nodes'],'objective',first,flush=True)
    def pg(z,g):
        q=g.copy();q[(z<=lower+1e-9)&(g>0)]=0;q[(z>=upper-1e-9)&(g<0)]=0;return float(np.max(abs(q)))
    def cb(z):
        c,g=objective(z);history.append(dict(iteration=len(history)+1,cost=c,projected_gradient=pg(z,g),seconds=time.perf_counter()-start))
        if len(history)%100==0:
            th,gr=fn.unpack(objective.project(z));np.savez(folder/'checkpoint.npz',theta=th,grid=gr)
            dump(folder/'history.json',history);print(a.name,history[-1],flush=True)
        if time.perf_counter()-start>a.seconds:raise StopIteration
    res=minimize(objective,z,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),callback=cb,
        options=dict(maxiter=a.maxiter,maxcor=50,maxls=60,ftol=1e-13,gtol=2e-6))
    z=objective.project(res.x);theta,grid=fn.unpack(z)
    np.savez(folder/'checkpoint.npz',theta=theta,grid=grid);dump(folder/'history.json',history)
    continuous=float(objective(z)[0]);cgrad=pg(z,objective(z)[1])
    candidate=from_vector(theta,'HNCS_UserFit');candidate.grid=np.rint(project_neutral_grid(grid)*256)
    candidate.grid_model='rea-bt601-limited-reference-v1';candidate=candidate.quantized();theta=vector_of(candidate,True)
    def core_source(th):
        core=from_vector(th);return np.concatenate([render_fast(r,x,core,prepared=prep) for r,x,prep in batches])
    src=core_source(theta)
    reference=np.concatenate([r.render(x,from_vector(theta),prepared=prep) for r,x,prep in batches])
    np.testing.assert_array_equal(src,reference)
    A=grid_forward(src,np.zeros((405,3)))[2]
    def polish(flat):
        g=project_neutral_grid(flat.reshape(405,3));raw=src+A@g;c,d=loss(np.clip(raw,0,1));lg=fn.L@g
        c+=GRID_SMOOTH*np.sum(g*lg);deriv=A.T@(d*((raw>0)&(raw<1)))+2*GRID_SMOOTH*lg
        return float(c),project_neutral_grid(deriv).ravel()
    g0=candidate.grid.ravel()/256
    rr=minimize(polish,g0,jac=True,method='L-BFGS-B',bounds=[(-.25,63/256)]*1215,
        options=dict(maxiter=a.grid_iterations,maxcor=40,maxls=60,ftol=1e-12,gtol=1e-6))
    q=np.rint(project_neutral_grid(rr.x.reshape(405,3))*256)/256
    if polish(q.ravel())[0]<polish(g0)[0]:candidate.grid=q*256
    problem=HncsIntegerGridProblem(src,loss,fn.L,candidate.grid);zero=np.zeros((405,3))
    native_penalty=lambda th:regularization(th,zero,fn.L)
    integer_first=problem.value+native_penalty(theta);int_history=[];certified=False
    qscale=np.r_[np.full(21,1024),np.full(24,512)]
    for sweep in range(a.integer_sweeps):
        grow=problem.sweep();g=problem.g.copy();reg=GRID_SMOOTH*np.sum(g*(fn.L@g))
        current=problem.value+native_penalty(theta);before=current;accepted=0;trials=0
        for c in range(45):
            best=None
            for direction in (-4,-1,1,4):
                trial=theta.copy();trial[c]+=direction/qscale[c]
                if not LOW[c]-1e-12<=trial[c]<=HIGH[c]+1e-12:continue
                try:
                    trial_src=core_source(trial);pred=grid_forward(trial_src,g)[0]
                    val=loss(pred,False)+reg+native_penalty(trial);trials+=1
                except ValueError:continue
                if val<current-1e-11 and (best is None or val<best[0]):best=val,trial,trial_src
            if best is not None:current,theta,src=best;accepted+=1
        if accepted:problem=HncsIntegerGridProblem(src,loss,fn.L,g*256)
        total=problem.value+native_penalty(theta);np.testing.assert_allclose(total,current,atol=1e-8,rtol=1e-10)
        int_history.append(dict(sweep=sweep+1,grid=grow,native_accepted=accepted,native_trials=trials,
            native_improvement=before-total,objective=total,seconds=time.perf_counter()-start))
        print(a.name,'INTEGER',json.dumps(int_history[-1]),flush=True)
        certified=grow['accepted']==0 and accepted==0
        if certified:break
    candidate=from_vector(theta,'HNCS_UserFit');candidate.grid=np.rint(problem.g*256)
    candidate.grid_model='rea-bt601-limited-reference-v1';candidate=candidate.quantized();pred=np.clip(problem.raw,0,1)
    exact=np.concatenate([r.render(x,candidate,prepared=prep) for r,x,prep in batches])
    np.testing.assert_allclose(exact,pred,rtol=0,atol=2e-14)
    report=dict(initial_objective=first,continuous_objective=continuous,continuous_iterations=res.nit,
        continuous_status=str(res.message),continuous_projected_gradient=cgrad,grid_polish_iterations=rr.nit,
        integer_initial_objective=integer_first,integer_objective=problem.value+native_penalty(theta),
        hncs_appearance_loss=loss.appearance(pred,False),operator_stability_loss=loss.guard(pred,False),
        structural_regularization=regularization(theta,problem.g,fn.L),integer_history=int_history,
        local_integer_certificate=certified,certificate_scope='native +/-1,+/-4; non-neutral grid channel +/-1; neutral RGB tied; endpoints fixed',
        no_global_optimality_claim=True,seconds=time.perf_counter()-start,integer_reference_agreement=True)
    report['receipt']=export_look(candidate,folder/'looks/HNCS_UserFit.a7look',{
        'teacher_id':contract['teacher'],'physical_baseline':contract['baseline'],'objective_id':contract['objective_id'],
        'colour_target':'published composite teacher; HNCS-inspired approximation','st_reference_constraint':False,'old_look_prior':False,'initializer_only':True,
        'extra_dro_layer':False,'per_scene_routing':False,'runtime_defringe':False,'camera_verified':False,
        'fit':'analytic HNCS-only joint fit and exact integer refinement with target-free operator stability'})
    dump(folder/'solution.json',report);print('DONE',a.name,json.dumps({k:v for k,v in report.items() if k!='integer_history'}),flush=True)

if __name__=='__main__':main()
