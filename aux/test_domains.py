import numpy as np
from IPython import embed

class gldbg2013:

    def __init__(self, nx=151, ny=151):
        ''' Create the test domain detailed in Goldberg (2013)
        nx -- number of grid cells in x direction
        ny -- number of grid cell in y direction
        '''

        self.x = np.linspace(0,150e3, num=nx)
        self.y = np.linspace(0,150e3, num=ny)

        self.init_bed(self.x,self.y)
        self.init_surf(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)
        self.init_smb(self.x,self.y)


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
        '''Return ice sheet surface elevation in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        self.surf = self.bed + 2000.0*np.ones([y.size, x.size])
        self.surf[:,-1] = self.bed[:,-1];


    def init_B2(self,x,y):
        '''Return basal drag coefficient B^2 for linear sliding law
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
        '''Return basal melt rate
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 1.0*np.ones(yy.shape)

    def init_smb(self,x,y):
        '''Return surface mass balance (m/yr)
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.smb = 0.0*np.ones(yy.shape)

class ismipC:

    def __init__(self, L, nx=151, ny=151, tiles=1.0):
        ''' Create the ISMIP-Hom Experiment C domain
        nx -- number of grid cells in x direction
        ny -- number of grid cell in y direction
        tiles -- tile the domain to allow the use of no flow boundary conditions
        '''
        self.L = L
        self.tiles = tiles
        self.x = np.linspace(0, L, num=nx)
        self.y = np.linspace(0, L, num=ny)

        self.ty = 31556926 #seconds in year from ismip document

        self.init_surf(self.x,self.y)
        self.init_bed(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)
        self.init_smb(self.x,self.y)
        self.init_Bglen(self.x,self.y)

        self.init_mask(self.x,self.y)

    def init_surf(self,x,y):
        '''Return ice sheet surface elevation in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        xx = np.meshgrid(x, y)[0]
        self.surf = 1e4 -xx*np.tan(0.1*np.pi/180.0)

    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = self.surf - 1000.0

    def init_B2(self,x,y):
        '''Return basal drag coefficient B^2 for linear sliding law
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        w = 2.0*np.pi/self.L
        xx,yy = np.meshgrid(x,y)
        self.B2 = (1000.0 + 1000.0*np.sin(self.tiles*w*xx)*np.sin(self.tiles*w*yy))


    def init_bmelt(self,x,y):
        '''Return basal melt rate beneath floating ice (m/yr)
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 0.0*np.ones(yy.shape)

    def init_smb(self,x,y):
        '''Return surface mass balance (m/yr)
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.smb = 0.0*np.ones(yy.shape)

    def init_Bglen(self,x,y):
        '''Return Bglen for Nye's flow law
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]
        A = 10**(-16)
        self.Bglen = A**(-1.0/3.0)*np.ones(yy.shape)

    def init_mask(self,x,y):
        '''Return ice mask
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]
        self.mask = 1.0*(np.ones([x.size, y.size]))



class analytical1:

    def __init__(self, L, nx=151, ny=151):
        ''' Incline slab test domain
        L -- length of domain in metres
        nx -- number of grid cells in x direction
        ny -- number of grid cell in y direction
        '''
        self.L = L
        self.x = np.linspace(0, L, num=nx)
        self.y = np.linspace(0, L, num=ny)

        self.init_surf(self.x,self.y)
        self.init_bed(self.x,self.y)
        self.thick = self.surf - self.bed;

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)
        self.init_smb(self.x,self.y)



    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = self.surf - 1000.0



    def init_surf(self,x,y):
        '''Return ice sheet surface elevation in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        xx = np.meshgrid(x, y)[0]
        self.surf = 1e4 -xx*np.tan(0.1*np.pi/180.0)

    def init_B2(self,x,y):
        '''Return basal drag coefficient B^2 for linear sliding law
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

    def init_smb(self,x,y):
        '''Return surface mass balance (m/yr)
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.smb = 0.0*np.ones(yy.shape)



class analytical2:

    def __init__(self, Lx, Ly, nx=151, ny=151, li = -4):
        ''' Floating test domain
        Lx -- length of domain (metres) in x direction
        Ly -- length of domain (metres) in y direction
        nx -- number of grid cells in x direction
        ny -- number of grid cell in y direction
        li -- number of buffer cells at leading margin
        '''

        self.x = np.linspace(0, Lx, num=nx)
        self.y = np.linspace(0, Ly, num=ny)
        self.li = li

        self.init_bed(self.x,self.y)
        self.init_thick(self.x,self.y)
        self.init_surf(self.x,self.y)
        self.init_mask(self.x,self.y)

        self.init_B2(self.x,self.y)
        self.init_bmelt(self.x,self.y)
        self.init_smb(self.x,self.y)



    def init_bed(self,x,y):
        '''Return bedrock topography in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        #Create bed topography in metres
        self.bed = -1000.0*(np.ones([x.size, y.size]))

    def init_thick(self,x,y):
        '''Return ice thickness in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        li = self.li
        self.thick = 1000.0*(np.ones([x.size, y.size]))
        self.thick[li:,] = 0.0

    def init_surf(self,x,y):
        '''Return ice sheet surface elevation in metres
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        rhoi = 917.0
        rhow = 1000.0
        self.surf = (1-rhoi/rhow) * self.thick


    def init_mask(self,x,y):
        '''Return ice mask
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        li = self.li
        self.mask = 1.0*(np.ones([x.size, y.size]))
        self.mask[li:,] = 0.0

    def init_B2(self,x,y):
        '''Return basal drag coefficient B^2 for linear sliding law
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        self.B2 = 1e3*(np.ones([x.size, y.size]))

    def init_bmelt(self,x,y):
        '''Return basal melt rate beneath floating ice (m/yr)
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.bmelt = 0.0*(np.ones([x.size, y.size]))

    def init_smb(self,x,y):
        '''Return surface mass balance (m/yr)
        Input:
        x -- x coordinates in metres as a numpy array
        y -- y coordinates in metres as a numpy array
        '''
        yy = np.meshgrid(x,y)[1]

        self.smb = 0.0*np.ones(yy.shape)
