
import scipy.interpolate as interp


class field_interpolator:
    def __init__(self,x,y,field):        
        self.f =  interp.RectBivariateSpline(x, y, field)
        
