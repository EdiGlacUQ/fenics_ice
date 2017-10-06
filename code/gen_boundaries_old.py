def gen_boundaries(self):

    #Cell labels
    self.OMEGA_X    = 0     #exterior to ice sheet
    self.OMEGA_GND   = 1   # internal cells over bedrock
    self.OMEGA_FLT   = 2   # internal cells over water

    #Face labels
    self.GAMMA_X = 0    #facets not on boundary
    self.GAMMA_DMN = 1   # domain boundary
    self.GAMMA_GND = 2   # terminus
    self.GAMMA_FLT = 3   # terminus

    #Cell and Facet markers
    self.cf      = CellFunction('size_t',  self.mesh)
    self.ff      = FacetFunction('size_t', self.mesh)

    #Initialize Values
    self.cf.set_all(self.OMEGA_X)
    self.ff.set_all(self.GAMMA_X)

    # Build connectivity between facets and cells
    D = self.mesh.topology().dim()
    self.mesh.init(D-1,D)


    #Label cells
    rhow = self.rhow
    rhoi = self.rhoi

    for c in cells(self.mesh):
        x_m       = c.midpoint().x()
        y_m       = c.midpoint().y()
        mask_xy   = self.mask(x_m, y_m)
        h_xy = self.thick(x_m, y_m)
        bed_xy = self.bed(x_m,y_m)

        #Determine whether the cell is floating, grounded, or not of interest
        if near(mask_xy, 1, self.tol):
            if h_xy >= rhow*(0-bed_xy)/rhoi:
                self.cf[c] = self.OMEGA_GND
            else:
                self.cf[c] = self.OMEGA_FLT


    #Label facets
    cntr = 0
    cntr_ext = 0
    for f in facets(self.mesh):
        x_m      = f.midpoint().x()
        y_m      = f.midpoint().y()
        mask_xy = self.mask(x_m, y_m)
        height_xy = self.thick(x_m, y_m)

        if f.exterior():
            if near(mask_xy,1,self.tol):
                cntr_ext += 1
                self.ff[f] = self.GAMMA_DMN

        else:
            #Identify the 2 neighboring cells
            [n1_num,n2_num] = f.entities(D)

            #Properties of neighbor 1
            n1 = Cell(self.mesh,n1_num)
            n1_x = n1.midpoint().x()
            n1_y = n1.midpoint().y()
            n1_mask = self.mask(n1_x,n1_y)
            n1_bool = near(n1_mask,1, self.tol)

            #Properties of neighbor 2
            n2 = Cell(self.mesh,n2_num)
            n2_x = n2.midpoint().x()
            n2_y = n2.midpoint().y()
            n2_mask = self.mask(n2_x,n2_y)
            n2_bool = near(n2_mask,1, self.tol)


            #Identify if terminus cell
            if n1_bool + n2_bool == 1: #XOR
                cntr += 1

                #Grounded or Floating
                bed_xy = self.bed(x_m, y_m)
                if bed_xy >= -self.tol:
                    self.ff[f] = self.GAMMA_GND
                else:
                    self.ff[f] = self.GAMMA_FLT

                #Set unit vector to point outwards
                n = f.normal()
                print n.str(True)
