# code to give rsvg.render_cairo(ctx) ability
# requires libgobject and librsvg from https://github.com/jmcb/python-rsvg-dependencies/tree/master/bin
# to be included in the program's directory
import os

try:
    import rsvg
    WINDOWS = False
except ImportError:
    print("Warning, could not import 'rsvg'\n")

    if os.name == 'nt':
        print('Detected windows, creating rsvg.')

        from ctypes import *

        l = CDLL('librsvg-2-2.dll')
        g = CDLL('libgobject-2.0-0.dll')
        g.g_type_init()

        class rsvgHandle():

            class RsvgDimensionData(Structure):
                __fields__ = [("width", c_int),
                              ("height", c_int),
                              ("em", c_double),
                              ("ex", c_void_p)]

            def __init__(self, path):
                self.path = path
                error = ''
                z = self.PycairoContext.from_address(id(ctx))
                self.handle = l.rsvg_handle_new_from_file(self.path, error)

            def get_dimension_data(self):
                svgDim = self.RsvgDimensionData()
                l.rsvg_handle_get_dimensions(self.handle, byref(svgDim))
                return (svgDim.width, svgDim.height)

            def render_cairo(self, ctx):
                ctx.save()
                z = self
                l.rsvg_handle_render_cairo(self.handle, z.ctx)
                ctx.restore()

            class rsvgClass():
                def Handle(self, file):
                    return rsvgHandle(file)

rC = rsvg.rsvgClass()
h = rC.Handle("activity.svg")
s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
ctx = cairo.Context(s)
h.render_cairo(ctx)