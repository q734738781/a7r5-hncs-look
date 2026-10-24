"""Historical composite target, retained for reproducing previous experiments.

This is the public-profile teacher, not unavailable original Phocus records.
Profile matrices/HSM/Adobe LookTable/XMP HNCS table are read from actual assets.
Film Tone uses V-Log-Alchemy's recovered 65536-entry Phocus Film Curve in
Hasselblad RGB, then decodes its ICC TRC and converts to sRGB. This is an explicit
standalone composite teacher, not an emulation of Lightroom's PV rendering.
New fitting must use teacher.reference.HNCSReference with physical RAW inputs
and clipping-support masks. Do not promote old cached targets to ground truth.
"""
import struct, json, hashlib, xml.etree.ElementTree as ET
import numpy as np
import tifffile
from scipy.interpolate import PchipInterpolator
from hncs_native.bootstrap import VENDOR
import crs_table
from hncs_core.adobe_sdk_color import TripleColorSpec
from hncs_core.adobe_triple_illuminant import IlluminantDescriptor, xy_to_temperature_tint, temperature_tint_to_xy
from hncs_core.colour import PROPHOTO_PRIMARIES,D50_WHITE_XY,D65_WHITE_XY,rgb_to_xyz_matrix,prophoto_to_xyz_d65,chromatic_adaptation_bradford
from hncs_core.dng_huesat import GridSpec,apply_payload_rgb
from tools.hncs_full_probe import apply_map_hdr

def encode_srgb(x):
    x=np.maximum(x,0)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

class HNCSTeacher:
    def __init__(self):
        folder=VENDOR/'HNCS-Lightroom/profiles/sony-ilce-7rm5'
        with tifffile.TiffFile(folder/'HNCS Base - ILCE-7RM5.dcp') as tf:
            t=tf.pages[0].tags
            def matrix(k):
                a=np.asarray(t[k].value,dtype=float).reshape(-1,2)
                return (a[:,0]/a[:,1]).reshape(3,3)
            descriptors=[]
            for k in (52533,52534,52535):
                _,nx,dx,ny,dy=struct.unpack('<Hiiii',bytes(t[k].value))
                descriptors.append(IlluminantDescriptor(*xy_to_temperature_tint(nx/dx,ny/dy)))
            self.spec=TripleColorSpec(tuple(descriptors),np.stack([matrix(k) for k in (50721,50722,52531)]),
                                     np.stack([matrix(k) for k in (50964,50965,52532)]))
            self.hsm_spec=GridSpec(*t[50937].value)
            self.bases=np.stack([np.asarray(t[k].value,dtype=np.float32) for k in (50938,50939,52537)])
            self.adobe_spec=GridSpec(*t[50981].value); self.adobe=np.asarray(t[50982].value,dtype=np.float32)
            self.hdr=52551 in t
        root=ET.parse(folder/'HNCS.xmp').getroot()
        desc=next(e for e in root.iter() if e.tag.endswith('Description'))
        attrs={k.split('}')[-1]:v for k,v in desc.attrib.items()}
        binary=crs_table.decode(attrs['Table_'+attrs['LookTable']])
        version,kind,h,s,v=struct.unpack_from('<5I',binary)
        if (version,kind)!=(0,1): raise ValueError('Unexpected XMP HSM table')
        self.fixed_spec=GridSpec(h,s,v)
        self.fixed=np.frombuffer(binary,dtype='<f4',count=h*s*v*3,offset=20).copy()
        self.fixed_encoding=struct.unpack_from('<I',binary,20+h*s*v*12)[0]
        if self.fixed_encoding != 0: raise ValueError('Unexpected fixed-HSM encoding')
        curve=next(e for e in root.iter() if e.tag.endswith('ToneCurvePV2012'))
        knots=np.array([[float(x) for x in e.text.split(',')] for e in curve.iter() if e.tag.endswith('li')])/255
        self.film=PchipInterpolator(knots[:,0],knots[:,1])
        # Historical XMP metadata only. render() uses film64k below, deliberately
        # not both tone curves. This class never reproduced Lightroom PV.
        self.pro_from_pcs=np.linalg.inv(rgb_to_xyz_matrix(PROPHOTO_PRIMARIES,D50_WHITE_XY))
        self.xyz_to_srgb=np.array([[3.2404542,-1.5371385,-.4985314],[-.969266,1.8760108,.041556],[.0556434,-.2040259,1.0572252]])
        folder=VENDOR/'V-Log-Alchemy/Tools'
        artifact=json.loads((folder/'hasselblad_x2d100c_standard_cc.json').read_text())
        asset=artifact['assets']['standard_film_curve']; raw=(folder/asset['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=asset['sha256']: raise ValueError('Film curve asset hash mismatch')
        self.film64k=np.frombuffer(raw,dtype='<f4')/65535
        hs=artifact['working_space']; hm=np.array(hs['rgb_to_xyz'])
        self.h_from_pro=np.linalg.inv(hm) @ rgb_to_xyz_matrix(PROPHOTO_PRIMARIES,D50_WHITE_XY)
        self.h_to_srgb=self.xyz_to_srgb @ chromatic_adaptation_bradford(D50_WHITE_XY,D65_WHITE_XY) @ hm
        self.h_gamma=hs['icc_trc_gamma']

    def color_stages(self,camera_rgb,kelvin=5550,*,white_scale=8192,neutral=None):
        """Expose the camera bridge and table stages, before any Film Tone.

        The final HSM already includes HNCS colour correction. Applying V-Log's
        ColorCorrect to its output a second time would duplicate that operation.
        """
        shape=np.shape(camera_rgb); x=np.asarray(camera_rgb).reshape(-1,3)/white_scale
        white=temperature_tint_to_xy(float(kelvin),0) if neutral is None else self.spec.neutral_to_xy(neutral)
        state=self.spec.set_white_xy(white)
        # Input already white balanced; undo that normalization before camera_to_pcs.
        pro=(x*state.camera_white) @ state.camera_to_pcs.T @ self.pro_from_pcs.T
        stages={'camera_to_prophoto':pro.reshape(shape)}
        w=state.weights.astype(np.float32)
        payload=(w[0]*self.bases[0]+w[1]*self.bases[1]+w[2]*self.bases[2]).astype(np.float32)
        pro=apply_map_hdr(self.hsm_spec,payload,np.clip(pro,0,1)) if self.hdr else apply_payload_rgb(self.hsm_spec,payload,np.clip(pro,0,1))
        stages['wb_carrier']=pro.reshape(shape)
        scale=np.maximum(pro.max(axis=1),1)
        pro=apply_payload_rgb(self.adobe_spec,self.adobe,pro/scale[:,None])*scale[:,None]
        stages['adobe_look']=pro.reshape(shape)
        # Fixed HSM asset is an SDR table. Preserve exposure by homogeneous extension.
        scale=np.maximum(pro.max(axis=1),1)
        pro=apply_payload_rgb(self.fixed_spec,self.fixed,pro/scale[:,None])*scale[:,None]
        stages['hncs_color']=pro.reshape(shape)
        stages['hasselblad_linear']=(pro @ self.h_from_pro.T).reshape(shape)
        return stages

    def render_reference_tail(self,camera_rgb,kelvin=5550,*,white_scale=8192,neutral=None):
        """Explicit candidate: WB-aware HNCS colour + published V-Log shoulder.

        This is still an A7R V composite, not the complete V-Log input pipeline.
        The shoulder is V-Log-Alchemy's extension, not a recovered Phocus stage.
        Legacy render() remains reproducible for existing fitted Looks/caches.
        """
        from hncs_native.teacher.vlog_reference import VLogReference
        if not hasattr(self,'_vlog_reference'):self._vlog_reference=VLogReference()
        stages=self.color_stages(camera_rgb,kelvin,white_scale=white_scale,neutral=neutral)
        return self._vlog_reference.tail(stages['hasselblad_linear'])

    def render(self,camera_rgb,kelvin=5550,*,film=True,white_scale=8192,neutral=None):
        shape=np.shape(camera_rgb)
        stages=self.color_stages(camera_rgb,kelvin,white_scale=white_scale,neutral=neutral)
        pro=stages['hncs_color'].reshape(-1,3)
        if film:
            hb=np.clip(stages['hasselblad_linear'].reshape(-1,3),0,.9999)
            rendered=self.film64k[np.floor(hb*65536).astype(int)]
            rgb=encode_srgb(np.clip(rendered**self.h_gamma @ self.h_to_srgb.T,0,1))
        else:
            rgb=encode_srgb(np.clip(prophoto_to_xyz_d65(pro) @ self.xyz_to_srgb.T,0,1))
        return np.clip(rgb,0,1).reshape(shape)
