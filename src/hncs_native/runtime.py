"""Optional exact-stage acceleration; no alternative colour model."""
try:
    from .sony.fast_native import render_fast,prepare_fast
except ImportError:
    def prepare_fast(renderer,source):return renderer.prepare(source)
    def render_fast(renderer,source,look=None,*,prepared=None):
        return renderer.render(source,look,prepared=prepared,integer=True)

def teacher_renderer(main,st):
    try:
        from .teacher.fast_reference import FastHNCSReference
        return FastHNCSReference(main,st)
    except ImportError:
        from .teacher.reference import HNCSReference
        return HNCSReference(main,st)
