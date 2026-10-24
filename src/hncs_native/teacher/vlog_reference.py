"""Vectorized V-Log-Alchemy daylight reference, verified against its generator.

This is a fixed daylight colour operator and published highlight shoulder.
It is not a multi-temperature Phocus recovery, nor an A7R V input calibration.
"""
import importlib.util
import numpy as np
from hncs_native.bootstrap import VENDOR
from hncs_core.color_correct import apply_color_correct

class VLogReference:
    def __init__(self):
        path=VENDOR/'V-Log-Alchemy/Tools/generate_hasselblad_vlog.py'
        spec=importlib.util.spec_from_file_location('vlog_alchemy_reference',path)
        self.upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.upstream)
        u=self.upstream;ap,self.artifact=u.load_artifact(u.DEFAULT_ARTIFACT)
        fp,fs=u.resolve_asset(ap,self.artifact,'standard_film_curve')
        tp,ts=u.resolve_asset(ap,self.artifact,'daylight_color_correct_lut')
        self.film=np.asarray(u.load_film_curve(fp,int(fs['entries'])))/65535.
        self.texture=np.asarray(u.load_float2_texture(tp,int(ts['width']),int(ts['height']))).reshape(int(ts['height']),int(ts['width']),2)
        self.params=self.artifact['color_correct']['params'];self.working=self.artifact['working_space']
        self.to_display=np.asarray(u.build_hasselblad_to_display_matrix(self.working))
        self.from_vgamut=np.asarray(u.build_vgamut_to_hasselblad_rgb_matrix(self.working))

    def color(self,rgb):
        x=np.asarray(rgb,float);return apply_color_correct(x.reshape(-1,3),self.params,self.texture).reshape(x.shape)

    def tail(self,rgb,*,rolloff=True,knee=.5):
        x=np.maximum(np.asarray(rgb,float),0)
        if rolloff:
            x=x.copy();mask=x>knee;span=1-knee;excess=x[mask]-knee
            x[mask]=knee+span*excess/(excess+span)
        values=self.film[np.floor(np.clip(x,0,.9999)*65536).astype(int)]
        linear=values**self.working['icc_trc_gamma']@self.to_display.T
        positive=np.maximum(linear,0)
        encoded=np.where(linear<=.0031308,12.92*linear,1.055*positive**(1/2.4)-.055)
        return np.clip(encoded,0,1)

    def render_hasselblad(self,rgb,*,rolloff=True):
        return self.tail(self.color(np.maximum(rgb,0)),rolloff=rolloff)

    def render_vlog(self,rgb):
        x=np.asarray(rgb,float)
        linear=np.where(x<.181,(x-.125)/5.6,10**((x-.598206)/.241514)-.00873)
        return self.render_hasselblad(np.maximum(linear@self.from_vgamut.T,0))
