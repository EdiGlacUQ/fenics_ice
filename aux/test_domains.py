import numpy as np
from IPython import embed

class gldbg2013:

    def __init__(self, nx=151, ny=151):
        self.x = np.linspace(0,150e3, num=nx)
        self.y = np.linspace(0,150e3, num=ny)

        self.init_bed(self.x,self.y)
        self.init_surf(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)

    def Rx(self,x):
        '''Return x component of bedrock topography function
        Input:
        x -- x coordinates in metres as a numpy array
        '''
        #metres -> km
        x = x/1000.0;

        #Eq 10, x dimension of topography
        rx = 1.0 + (5.0/6.0) * (150.0 - x) / 150.0

        return rx

    def Ry(self,y):
        '''Return y component of bedrock topography function
        Input:
        y -- y coordinates in metres as a numpy array
        '''
        #metres -> km
        y = y/1000.0;

        #Partition into three parts: p1, p2, p3
        p1 = np.logical_and(50 <= y, y < 100)

        p2a = np.logical_and(25 <= y, y< 50)
        p2b = np.logical_and(100 <= y, y< 125)
        p2 = np.logical_or(p2a,p2b)

        p3a = np.logical_or(p1,p2)
        p3 = np.logical_not(p3a)

        #Initiatalize empty array
        ry = np.empty(y.shape)
        ry[:] = np.nan

        #Bedrock height in metres
        ry[p1] = -100.0 - 600.0*np.sin( np.pi*(y[p1]-50)/50.0 )
        ry[p2] = -100.0 - 100.0*np.sin( np.pi*(y[p2]-50)/50.0 )
        ry[p3] = 0

        return ry

    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        rx = self.Rx(x)
        ry = self.Ry(y)
        self.bed = np.array([ry]).T * rx



    def init_surf(self,x,y):

        self.surf = self.bed + 2000.0*np.ones([y.size, x.size])
        self.surf[:,-1] = self.bed[:,-1];


    def init_B2(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        x = x/1000.0
        y = y/1000.0

        #Determine area of enhanced basal drag
        yy = np.meshgrid(x,y)[1]
        p1 = np.logical_and(50 <= yy, yy <= 100)

        #Assign basal drag
        self.B = 9*np.sqrt(30)*np.ones(yy.shape)
        self.B[p1] = np.sqrt(30)
        self.B2 = self.B**2


    def init_bmelt(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 1.0*np.ones(yy.shape)


class grnld_margin:

    def __init__(self, nx=151, ny=151):
        self.x = np.linspace(0,150e3, num=nx)
        self.y = np.linspace(0,150e3, num=ny)

        self.init_bed(self.x,self.y)
        self.init_surf(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)


    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = np.zeros([y.size,x.size])


    def init_surf(self,x,y):
        H0 = 2000.0
        li = -4
        xx = np.meshgrid(x, y)[0]
        self.surf = np.zeros(xx.shape)
        self.surf[:,0:li] = H0*np.sqrt(1.0-(xx[:,0:li])/max(x[0:li]))



    def init_B2(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        x = x/1000.0
        y = y/1000.0

        #Determine area of enhanced basal drag
        yy = np.meshgrid(x,y)[1]
        p1 = np.logical_and(50 <= yy, yy <= 100)

        #Assign basal drag
        #self.B2 = 9*np.sqrt(30)*np.ones(yy.shape)
        #self.B2[p1] = np.sqrt(30)
        self.B2 = (2000)*np.ones(yy.shape)
        self.B2[p1] = (1500)



    def init_bmelt(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 1.0*np.ones(yy.shape)


class ismipC:

    def __init__(self, L, nx=151, ny=151):
        self.L = L
        self.x = np.linspace(0, L, num=nx)
        self.y = np.linspace(0, L, num=ny)

        self.init_surf(self.x,self.y)
        self.init_bed(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)


    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = self.surf - 1000.0



    def init_surf(self,x,y):
        xx = np.meshgrid(x, y)[0]
        self.surf = 1e4 -xx*np.tan(0.1*np.pi/180.0)

    def init_B2(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        w = 2.0*np.pi/self.L
        xx,yy = np.meshgrid(x,y)
        self.B2 = (1000.0 + 1000.0*np.sin(3.0*w*xx)*np.sin(3.0*w*yy))



    def init_bmelt(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 1.0*np.ones(yy.shape)



class analytical1:

    def __init__(self, L, nx=151, ny=151):
        self.L = L
        self.x = np.linspace(0, L, num=nx)
        self.y = np.linspace(0, L, num=ny)

        self.init_surf(self.x,self.y)
        self.init_bed(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)


    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = self.surf - 1000.0



    def init_surf(self,x,y):
        xx = np.meshgrid(x, y)[0]
        self.surf = 1e4 -xx*np.tan(0.1*np.pi/180.0)

    def init_B2(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        w = 2.0*np.pi/self.L
        xx,yy = np.meshgrid(x,y)
        self.B2 = 1500.0*(np.ones([y.size, x.size]))



    def init_bmelt(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 1.0*np.ones(yy.shape)



class analytical2:

    def __init__(self, Lx, Ly, nx=151, ny=151):

        self.x = np.linspace(0, Lx, num=nx)
        self.y = np.linspace(0, Ly, num=ny)

        self.init_bed(self.x,self.y)
        self.init_thick(self.x,self.y)
        self.init_surf(self.x,self.y)

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)


    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = -1000.0*(np.ones([x.size, y.size]))

    def init_thick(self,x,y):
        self.thick = 1000.0*(np.ones([x.size, y.size]))
        self.thick[-5:,] = 0.0

    def init_surf(self,x,y):
        rhoi = 917.0
        rhow = 1000.0
        self.surf = (1-rhoi/rhow) * self.thick

    def init_B2(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        self.B2 = 0.0*(np.ones([x.size, y.size]))

    def init_bmelt(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 0.0*(np.ones([x.size, y.size]))
