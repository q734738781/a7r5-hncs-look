import json
from pathlib import Path
import numpy as np
import pytest
from hncs_native.format import read_look,document,decode,pack_native,write_look
from hncs_native.grid import predict,smoothness
from hncs_native.optimizer.fit import lab,vector_of
from hncs_native.optimizer.operator_math import lab_jacobian,forward_jacobian
from hncs_native.optimizer.reverse import lab_pullback,forward_pullback
from hncs_native.optimizer.unified import AppearanceLoss,JointObjective,grid_forward,grid_forward_reference
from hncs_native.optimizer.hncs_highlight import OperatorStabilityLoss,HncsIntegerGridProblem
from hncs_native.optimizer.highlight import project_neutral_grid
from hncs_native.pooling import build_pool
from hncs_native.sony.rea_adapter import NativeLook,expand_matrix,tone_curve

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    rng=np.random.default_rng(741);source=rng.uniform(.1,.9,(190,3));target=rng.uniform(.15,.85,(190,3))
    P,pw=build_pool([(np.arange(20),1.)],190);weights=np.zeros(190);weights[:20]=1
    app=AppearanceLoss(target,weights,P,pw,reference=None)
    loss=OperatorStabilityLoss(app,np.arange(20,30).reshape(2,5),[.5,.5],np.arange(30,190).reshape(2,80),.05)
    return source,loss,rng

def toy_renderer():
    # Synthetic calibration only: no camera tags or proprietary tables bundled.
    class R:
        wb_weights=np.array([.2,.5,.3]);cc1=np.array([0,0,0,0,512,512,512,512]);luma=np.array([19,38,7])
        gamma_x=np.linspace(0,2097120,128);gamma_y=np.linspace(0,262128,128)
        cs=np.array([0,16383,0,0]);ylut=np.r_[np.zeros(32768,dtype='i2'),np.arange(32768,dtype='u2').view('i2')]
        yoff=0;ygain=1
        def prepare(self,x):
            x=np.asarray(x,float);return x,x.copy(),np.zeros(len(x)),(0,0,0)
    return R()

def test_export_roundtrip_and_original_payload(tmp_path):
    path=ROOT/'looks/HNCS_Unified_Only.a7look';look=read_look(path)
    other=decode(document(look));np.testing.assert_array_equal(other.grid,look.grid)
    np.testing.assert_array_equal(other.matrix_planes,look.matrix_planes)
    assert len(pack_native(look))==108
    assert np.all(expand_matrix(look.matrix_planes).sum(-1)==0)
    out=tmp_path/'test.a7look';write_look(look,out);np.testing.assert_array_equal(read_look(out).grid,look.grid)
    with pytest.raises(FileExistsError):write_look(look,out)

def test_format_rejects_bad_integer_and_legacy_target():
    doc=document(NativeLook());doc['native_photo']['runtime_basis']['tone_q10'][0]=True
    with pytest.raises(ValueError):decode(doc)
    doc=document(NativeLook());doc['target']['model']='ILCE-7RM5'
    with pytest.raises(ValueError):decode(doc)

def test_zero_and_quantized_grid_match_declared_renderer():
    rng=np.random.default_rng(93);rgb=rng.uniform(0,1,(1000,3));grid=rng.integers(-20,21,(405,3))
    np.testing.assert_array_equal(predict(rgb,np.zeros((405,3))),rgb)
    np.testing.assert_allclose(predict(rgb,grid),grid_forward_reference(rgb,grid/256)[0],atol=2e-15)
    a,j,A,active=grid_forward_reference(rgb,grid/256);b,k,B,other=grid_forward(rgb,grid/256)
    np.testing.assert_allclose(a,b,atol=3e-15);np.testing.assert_allclose(j,k,atol=1e-13)
    np.testing.assert_allclose(A@grid,B@grid,atol=1e-12);np.testing.assert_array_equal(active,other)

def test_grid_input_jacobian_away_from_boundaries():
    rng=np.random.default_rng(94);rgb=rng.uniform(.1,.9,(37,3));grid=rng.uniform(-.02,.02,(405,3))
    value,J,_,_=grid_forward_reference(rgb,grid);direction=rng.normal(size=rgb.shape);h=1e-7
    numerical=(grid_forward_reference(rgb+h*direction,grid)[0]-grid_forward_reference(rgb-h*direction,grid)[0])/(2*h)
    np.testing.assert_allclose(numerical,np.einsum('noi,ni->no',J,direction),rtol=2e-6,atol=2e-8)

def test_lab_forward_and_reverse_derivatives():
    rng=np.random.default_rng(95);x=rng.uniform(.05,.95,(101,3));gradient=rng.normal(size=x.shape)
    a,J=lab_jacobian(x);b,back=lab_pullback(x)
    np.testing.assert_allclose(a,lab(x),atol=1e-12);np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(back(gradient),np.einsum('no,noi->ni',gradient,J),atol=2e-12)

def test_native_reverse_matches_forward_derivative():
    rng=np.random.default_rng(96);r=toy_renderer();x=rng.uniform(200,9000,(93,3));theta=np.zeros(45)
    a,J=forward_jacobian(r,x,theta);b,back=forward_pullback(r,x,theta);g=rng.normal(size=a.shape)
    np.testing.assert_allclose(a,b,atol=3e-15)
    np.testing.assert_allclose(back(g),np.einsum('no,nop->p',g,J),atol=2e-10)
    d=rng.normal(size=45);d/=np.linalg.norm(d);h=1e-7
    num=(forward_pullback(r,x,theta+h*d)[0]-forward_pullback(r,x,theta-h*d)[0])/(2*h)
    np.testing.assert_allclose(num,np.einsum('nop,p->no',J,d),rtol=2e-5,atol=2e-7)

def test_target_free_guard_gradient_and_zero_target_weight():
    rgb,loss,rng=fixture();assert not np.any(loss.appearance.w[loss.ids])
    value,g=loss(rgb);np.testing.assert_allclose(value,loss(rgb,False),atol=1e-10)
    d=rng.normal(size=rgb.shape);h=1e-6
    num=(loss(rgb+h*d,False)-loss(rgb-h*d,False))/(2*h)
    np.testing.assert_allclose(num,np.sum(g*d),rtol=2e-6,atol=1e-6)

def test_integer_local_update_equals_full_objective():
    src,loss,rng=fixture();L=smoothness();grid=np.rint(project_neutral_grid(rng.integers(-4,5,(405,3))))
    problem=HncsIntegerGridProblem(src,loss,L,grid)
    for node in (61,121,202,245,300):
        step=np.ones(3);g=problem.g.copy();g[node]+=step/256
        expected=loss(grid_forward(src,g)[0],False)+.3*np.sum(g*(L@g))
        np.testing.assert_allclose(problem.trial_vector(node,step),expected,rtol=1e-10,atol=1e-8)
        problem.trial_vector(node,step,True);np.testing.assert_allclose(problem.value,expected,rtol=1e-10,atol=1e-8)

def test_absolute_objective_has_no_initial_look_prior():
    r=toy_renderer();rng=np.random.default_rng(97);x=rng.uniform(300,7000,(30,3))
    P,pw=build_pool([(np.arange(30),1.)],30);loss=AppearanceLoss(np.full((30,3),.5),np.ones(30),P,pw)
    a=np.zeros(1260);b=rng.uniform(-.01,.01,1260);p=rng.uniform(-.005,.005,1260)
    one=JointObjective([(r,x,r.prepare(x))],loss,a,prior_strength=0.,grid_ridge=0.)
    two=JointObjective([(r,x,r.prepare(x))],loss,b,prior_strength=0.,grid_ridge=0.)
    ca,ga=one((p-a)/one.scale);cb,gb=two((p-b)/two.scale)
    np.testing.assert_allclose(ca,cb,atol=1e-10);np.testing.assert_allclose(ga,gb,atol=1e-10)

def test_released_gamma_monotone_in_integer_arithmetic():
    look=read_look(ROOT/'looks/HNCS_Unified_Only.a7look');source=np.arange(0,262129,2048)
    modified=tone_curve(source,look.tone_q10,True)
    assert np.all(np.diff(modified)>=0)
