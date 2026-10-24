"""Declared software camera baseline; original CP execution is optional."""

BASELINE_ID = 'sony-edit-a7rv-raw-v1'

def baseline_metadata():
    return {
        'id': BASELINE_ID,
        'pipeline': '751K/sony-a7m5-looks Sony Edit reconstruction',
        'calibration_model': 'ILCE-7RM5',
        'camera_equivalence': 'accepted engineering premise',
        'requires_original_cp_execution': False,
        'scope': 'colour pipeline; no spatial ISP',
    }
