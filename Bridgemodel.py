from openseespy.opensees import *
import numpy as np
import Bridge_member
# --------------------------------------------------------------------------------------------------------------------------------
# Class definition - Black box!!! do not change - -- - --
# --------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================

# Coordinate system: y: vertical axis, x: longitudinal direction, z: transverse direction
class Bridge:
    @classmethod
    #FUNCTION INPUT ORDER: 1.Transverse length, 2.Beam spacing, 3.Longitudinal (beam) length, 4.Slab spacing, 5.skew, 6.beamtype)
    def __init__(cls, Transverse_length, Beam_spacing, Beam_length, Slab_spacing, skew, beamtype):
        cls.Lz = Transverse_length
        cls.spacingz = Beam_spacing
        cls.Lx = Beam_length
        cls.spacingx = Slab_spacing
        cls.skewz = skew           # Skew angle for future versions incorporating skew bridge deck
        cls.numbeam = int(cls.Lz/cls.spacingz)
        cls.numxele = int((cls.Lx / cls.spacingx))
        cls.ele_z = [0]  # initialize Z-coordinate vector (representing main and edge beam)
        cls.ele_x = np.linspace(0, cls.Lx, cls.numxele+1) # generate uniform nodes in X dir (each representing slabs)
        b = (cls.Lz - ((cls.numbeam - 1) * cls.spacingz + cls.spacingz)) / 4  # distance from edge beam to outermost main beam
        if b == 0:
            print("Edge beams coordinate not located at member centroid")
        # Generate vectors for nodes in X and Z direction
        for z in range(0, cls.numbeam+1): # loop for each Z coord to generate node vector in Z dir
            if z == 0 or z == cls.numbeam:  # for first and last node point, spacing is b distance from regular beam spacing
                # Node coordinate for first Edge beam and all main beams
                cls.ele_z.append(cls.ele_z[-1]+ (b + cls.spacingz/2))# first node spacing b
            else:
                # Node coordinate for last edge beam
                cls.ele_z.append((z * cls.spacingz + b + cls.spacingz/2)) #last node spacing = (numbeam+1)*Beam_spacing+b

        # future versions consider skew - assemble different vector into matrix list
        cls.xv, cls.zv = np.meshgrid(cls.ele_x, cls.ele_z) # used for element_assemble()
        #    Inputs:(default) ndm, ndof

        #print("X coord = ", cls.ele_x)
        #print("Z coord = ", cls.ele_z)

        cls.beameletype = beamtype
    @classmethod
    # bridge member objects (composition class)
    def assign_beam_member_prop(cls,longbeam,LRbeam,edgebeam,slab,diaphragm):
        # arguments are in a list returned by class method of Bridge_member class() - get_beam_prop()
        cls.longbeam = longbeam
        cls.LRbeam = LRbeam
        cls.edgebeam = edgebeam
        cls.slab = slab
        cls.diaphragm = diaphragm

    @classmethod
    def assign_material_prop(cls,concreteprop=None,steelprop = None):
        if concreteprop is None:
            cls.steelprop = steelprop
        else:
            cls.concreteprop = concreteprop

    # ==================================================================================================

class OpenseesModel(Bridge):
    # set modelbuilder
    @classmethod
    def create_Opensees_model(cls):
        cls.generatemodel(3,6)  # run model generation
        cls.createnodes()       # create nodes of model
        cls.boundaryconditions() # assign boundary conditions to nodes at ends of model (i.e. x = 0 and x = Beam_length)
        cls.materialprop()      # material properties (default concrete and steel)
        #        trans: (1)long beam, (2) transverse  [ x y z]
        cls.ele_transform([0, 0, 1], [1, 0, 0]) # NEED ABSTRACTION
        # - default values are [0,0,1] for longitudinal, [-1,0,0] for transverse
        cls.element_assemble()

    @classmethod
    def generatemodel(cls,ndm,ndf):
        cls.ndm = ndm
        cls.ndf = ndf
        model('basic', '-ndm', cls.ndm, '-ndf', cls.ndf)
        #print("Model DIM = {}, Model DOF = {}".format(ndm, ndf))
    # create nodes
    @classmethod
    def createnodes(cls):
        cls.edgesupport = [] # list containing tags of nodes of support (used in boundaryconditions())
        cls.nodetag = np.zeros((len(cls.ele_z),len(cls.ele_x)))  # array containing the tag relative mesh position
        for y in range(len(cls.ele_z)):  # loop for y grid (ele Z beam positions)
            cls.edgesupport.append(int(y*len(cls.ele_x) + 1))   # first point of support
            for x in range(len(cls.ele_x)):         # loop in eleX (longtidunal)
                y0 = 0  # [current version z axis is vertical axis, future versions allow varying z values]
                #       tag                     x           y=0       z
                node(int(y*len(cls.ele_x) + x + 1), cls.xv[y,x], y0, cls.zv[y,x])  # tag starts from 1
                cls.nodetag[y,x] = int(y*len(cls.ele_x) + x + 1)        # node tag (mesh position)
                # Uncomment line 89 to print node tags and report values
                # print("node tag {tag} has coor x z {x} {z} ".format(tag = int(y*len(cls.ele_x) + x + 1) , x = cls.xv[y,x],z=cls.zv[y,x]))
                #
            cls.edgesupport.append(int(y*len(cls.ele_x) + x + 1))   # second point of support
        #print("Model's nodes layout is:\n",cls.nodetag)
        cls.totalnodes = y*len(cls.ele_x) + x + 1
        #print('Nodes generated =', cls.totalnodes)  # number here is the total nodes
        #print('Support nodes are =',  cls.edgesupport)

    # ==================================================================================================
    # set boundary condition
    @classmethod
    def boundaryconditions(cls):
        countdof = 0 # initialize counter of assigned boundary conditions

        for supp in range(len(cls.edgesupport)): # loop for each cls.edgesupport items
            fixvalpin = [1, 1, 1, 0, 0, 0] # pinned
            fixvalroller = [0,1,1,0,0,0] #roller
            if supp % 2 == 0: #
                fix(cls.edgesupport[supp], *fixvalpin) # x y z, mx my mz
            else:
                fix(cls.edgesupport[supp], *fixvalroller) # x y z, mx my mz
            countdof +=1 # counter
        #print('DOF constrained = ', countdof)

    # ==================================================================================================

    # create material tags
    @classmethod
    def materialprop(cls):
        cls.concrete = 1  # tag for concrete is "1"
        uniaxialMaterial("Concrete01", cls.concrete, *cls.concreteprop)
        cls.concrete = 'Concrete material defined'
    # ==================================================================================================
    @classmethod
    def ele_transform(cls,zaxis,xaxis):
        cls.transfType = 'Linear'  # transformation type
        cls.longitudinalTransf = 1  # tag
        cls.transverseTransf = 2  # tag
        geomTransf(cls.transfType, cls.longitudinalTransf, *zaxis)
        geomTransf(cls.transfType, cls.transverseTransf, *xaxis)
        cls.geo = 'geometrical transform object created'

    # ==================================================================================================
    # define elements
    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @classmethod
    def element_assemble(cls):
        eletypeB = cls.beameletype

    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #  define longitudinal beams
        longbeamtag = cls.nodetag[2:-2,:]  # tags of node correspond to long beam
        cls.numlongbeam = 0  # initialize counter of assigned boundary conditions

        cls.MIDele = np.zeros( (len(longbeamtag) , len(longbeamtag[0]) - 1) )

        for nlongbeam in range(len(longbeamtag)): # loop for each node in Y dir
            for x in range(len(longbeamtag[-1])-1): # minus 1 for number of elem = nnodes - 1
                elenodes = [longbeamtag[nlongbeam,x],longbeamtag[nlongbeam,x+1]]
                #         element  tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
                midtag = (2+nlongbeam)*(len(longbeamtag[-1])-1) + x + 1
                cls.MIDele[nlongbeam][x] = midtag #Element layout for printing
                element(eletypeB, midtag, *elenodes, *cls.longbeam,cls.longitudinalTransf)
                cls.numlongbeam += 1

        #print("Longitudinal beam defined = ", cls.numlongbeam)

        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #   define left and right beams
        leftbeam = cls.nodetag[1,:]  # tags of node correspond to leftbeam
        leftbeameletag = len(longbeamtag[-1])-1
        rightbeam = cls.nodetag[-2,:]  # tags of node correspond to rightbeam
        rightbeameletag = 2* leftbeameletag + cls.numlongbeam
        cls.L_ele = []
        cls.R_ele = []

        for cls.LR in range(len(leftbeam) - 1):  # minus 1 for number of elem = nnodes - 1
            leftelenodes = [leftbeam[cls.LR], leftbeam[cls.LR + 1]]
            rightelenodes = [rightbeam[cls.LR], rightbeam[cls.LR + 1]]
            #         element  tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
            tagL = leftbeameletag + cls.LR + 1
            cls.L_ele.append(tagL) # element tag of left beam for printing
            #         element  tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
            tagR = rightbeameletag + cls.LR + 1
            cls.R_ele.append(tagR) #element tag of right beam for printing

            element(eletypeB, tagL, *leftelenodes, *cls.LRbeam,cls.longitudinalTransf)
            element(eletypeB, tagR, *rightelenodes, *cls.LRbeam,cls.longitudinalTransf)

        #print("Left right beam defined = ", 2*(cls.LR+1))

    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # define edge beam
        leftedge= cls.nodetag[0,:]  # tags of node correspond to leftedge
        rightedge = cls.nodetag[-1,:]  # tags of node correspond to rightedge
        rightedgeeletag = 3 * (len(longbeamtag[-1])-1) + cls.numlongbeam  # countlong is longitudinal beam ele counting
        cls.Ledge_ele = []
        cls.Redge_ele = []
        for cls.edge in range(len(leftedge) - 1):  # minus 1 for number of elem = nnodes - 1
            Ledgenodes = [leftedge[cls.edge], leftedge[cls.edge + 1]]
            Redgenodes = [rightedge[cls.edge], rightedge[cls.edge + 1]]
            cls.Ledge_ele.append(cls.edge + 1)
            cls.Redge_ele.append(rightedgeeletag + cls.edge + 1)
            #         element  tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
            element(eletypeB, cls.edge + 1, *Ledgenodes, *cls.edgebeam, cls.longitudinalTransf)
            element(eletypeB, rightedgeeletag + cls.edge + 1, *Redgenodes, *cls.edgebeam, cls.longitudinalTransf)
        #print("left right edge defined = ", 2*(cls.edge+1))
        #print('Left edge element layout is ',Ledge_ele)
        #print('Right edge element layout is ',Redge_ele)
    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #  define transverse slab
        translab = cls.nodetag[:,1:-1]  # tags of node correspond to slab
        numtranslab = 0  # initialize counter of assigned boundary conditions
        long_ele = 2*(cls.edge+1)+2*(cls.LR+1) + cls.numlongbeam
        cls.Slab_ele = np.zeros((len(translab)-1,len(translab[-1]))) # zero matrix for slab element tag

        for nslab in range(len(translab[-1])): # loop for each node in Y dir (column)
            for y in range(len(translab)-1): # minus 1 for number of elem = nnodes - 1 (row)
                slabnodes = [translab[y,nslab],translab[y+1,nslab]] # the 2 nodes that made up of the slab element
                slabtag = long_ele + (nslab+1)*(len(translab)-1) + y + 1 # corresponding tag for the slab element
                cls.Slab_ele[y][nslab] = slabtag  # add element into the zero matrix correspoding to their location
                #         element  tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
                element(eletypeB,slabtag,*slabnodes, *cls.slab , cls.transverseTransf)
                numtranslab += 1
        #print("Transverse slab defined = ", numtranslab)
        #print('Transver element layout:\n',Slab_ele)
    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #  define diaphragm

        diap_S = cls.nodetag[:,1]
        cls.diapS_ele = np.zeros(((len(diap_S)-1),1)) # Empty matrix for to contain diaphrams' element tag later

        diap_R = cls.nodetag[:, -1]
        cls.diapR_ele = np.zeros(((len(diap_R)-1),1)) # Empty matrix for to contain diaphrams' element tag later

        for diap in range(len(diap_S)-1):
            S_diap = [diap_S[diap], diap_S[diap+1]]
            R_diap = [diap_R[diap], diap_R[diap+1]]
            S_tag = long_ele + diap + 1
            cls.diapS_ele[diap][0] = S_tag
            R_tag = long_ele + numtranslab + (len(diap_S)-1)+ diap + 1
            cls.diapR_ele[diap][0] = R_tag
            element(eletypeB, S_tag,*S_diap, *cls.diaphragm,cls.transverseTransf)
            element(eletypeB, R_tag,*R_diap, *cls.diaphragm,cls.transverseTransf)
        #print("diaphragm elements = ", 2*(diap+1))
        cls.totalele = cls.numlongbeam+2*(cls.LR+1)+2*(cls.edge+1)+numtranslab+2*(diap+1)
        #print("Total Number of elements = ", cls.totalele)
        #print('Left side diagram element layout\n', diapS_ele)
        #print('Right side diagram element layout\n', diapR_ele)



    # ==================================================================================================
    @classmethod
    def time_series(cls):
        timeSeries("Linear", 1)

    # ==================================================================================================

    @classmethod
    def loadpattern(cls):
        pattern("Plain", 1, 1)

    # ==================================================================================================

    @classmethod
    def loadID(cls):
        # This is for Test Load 1000 case for model - 245m example
        load(6, 0, -100000,0,0,0,0) # Fx Fy Fz, M-x , M-y, M-z

        load(17, 0, -100000, 0, 0, 0, 0)  # Fx Fy Fz, M-x , M-y, M-z
        load(28, 0, -100000, 0, 0, 0, 0)  # Fx Fy Fz, M-x , M-y, M-z
        load(39, 0, -100000, 0, 0, 0, 0)  # Fx Fy Fz, M-x , M-y, M-z
        load(50, 0, -100000, 0, 0, 0, 0)  # Fx Fy Fz, M-x , M-y, M-z
        load(61, 0, -100000, 0, 0, 0, 0)  # Fx Fy Fz, M-x , M-y, M-z
        load(72, 0, -100000, 0, 0, 0, 0)  # Fx Fy Fz, M-x , M-y, M-z

    # ==================================================================================================

    # Load methods
    @classmethod
    def loadtype(cls,argument):
        # selected load type is taken as argument str to return method (e.g. point, axle, or UDL)
        method_name = 'load_'+str(argument)
        method = getattr(cls,method_name,lambda:"nothing")
        return method()

    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # find load position given load's (x,0, z) coordinate
    @classmethod
    def load_position(cls,pos,axlwt):
        # method to return 4 beam elements or 1 shell element correspond to the location of which the axle load acts.
        # pos has a X0 and Z0 , need to find the elements and return tag of nodes in the grid where, X0 resides in
        cls.n1 = cls.nodetag[(cls.xv <= pos[0]) * (cls.zv <= pos[1]) ][-1]  # find the reference node
        # from ref node, populate other 3 nodes that forms the grid correspond to location of load
        cls.n2 = cls.n1+1
        cls.n4 = cls.n1+ len(cls.ele_x)  #ordering for clockwise defition
        cls.n3 = cls.n2+len(cls.ele_x)
        #cls.xcor1 = nodeCoord(eleNodes(3)[0])  # coor of first node
        a = (nodeCoord(cls.n2)[0]-nodeCoord(cls.n1)[0])# X dir
        b = (nodeCoord(cls.n4)[2] - nodeCoord(cls.n1)[2])# Z Dir
        cls.zeta = (pos[0]-nodeCoord(cls.n1)[0])/a # X dir
        cls.eta = (pos[1] - nodeCoord(cls.n1)[2]) /b  # Z Dir

        Nzeta = cls.shape_function(cls.zeta,a)
        Neta = cls.shape_function(cls.eta,b)

        Nv = [Nzeta[0]*Neta[0],Nzeta[2]*Neta[0],Nzeta[2]*Neta[2],Nzeta[0]*Neta[2]]
        Nmx = [Nzeta[1]*Neta[0],Nzeta[3]*Neta[0],Nzeta[3]*Neta[2],Nzeta[0]*Neta[3]]
        Nmz = [Nzeta[0]*Neta[1],Nzeta[2]*Neta[1],Nzeta[2]*Neta[3],Nzeta[0]*Neta[3]]
        #   N1 -> cls.n1 , N2 -> cls.n2 . . . . . N4 -> cls.n4
        # assign forces

        for nn in range(0,3):
            load(eval('cls.n%d' % (nn+1)),*np.dot([0,Nmx[nn],Nv[nn],0,0,Nmz[nn]],axlwt))
        #breakpoint()
    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @classmethod
    def shape_function(cls,zeta,a): # using zeta and a as placeholders for normal coor + length of edge element
        # hermite shape functions
        N1 = (1-3*zeta**2+2*zeta**3)
        N2 = (zeta-2*zeta**2+zeta**3)*a
        N3 = (3*zeta**2-2*zeta**3)
        N4 = (-zeta**2+zeta**3)*a
        return [N1, N2, N3, N4]

    #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # single point load
    @classmethod
    def load_singlepoint(cls,nodes, forcevector):
        load(nodes, *forcevector)  # fx fy fz mx my mz

    # multiple point loads of equal force
    @classmethod
    def load_multipoint(cls,NodesVector, forcevector): # force vector input: Fx, Fy, Fz, Mx, My, Mz
        for x in NodesVector:  #for loop lopp thru nodes
          load(x, *forcevector)


    def load_UDLpatch(cls,nodetags,force):
        # takes two nodes , a start and end node, then distribute the load
        for n in range(nodetags[1]-nodetags[0]):
            load(nodetags[0]+n, *force)
        # notes: force must be pre-determined - calculated elsewhere to work out the UDL onto each point


    def load_statictruck(cls,truck_pos,axlwts,axlspc,axlwidth):
        # from truck pos(X>0 and Z>0), determine relative location of front top axle on the bridge model
        # truck_pos is a list [X0, Z0]
        # read direction of axlwts and axlspc is from left (i.e. index = 0).
        # first element of axlspc is a placeholder 0, to account for front of vehicle (i.e. first axle)
        for n in range(len(axlwts)):  # loop do for each axle
            # axl position with respect to bridge
            # X coor, Z coor

            X1 = truck_pos[0] - axlspc[n]  # X coord of axle

            if X1 > 0 :  # check if axle is on bridge
                #   Inputs:                position[x0,z0],        axle weight at position
                cls.load_position([X1, truck_pos[1]], axlwts[n])  # one side axle
                cls.load_position([X1, truck_pos[1]+axlwidth], axlwts[n])  # other side of axle


    # i-node bending moment of element
    def BendingMoment_i(cls,BendingNode):
        cls.BendingNode =  BendingNode
        moment_i = eleForce(cls.BendingNode)[5] # Rotation about Z-axis
        return int(moment_i)

    # j-node bending moment
    def BendingMoment_j(cls,BendingNodeEnd):
        cls.BendingNodeEnd =  BendingNodeEnd
        moment_j = eleForce(cls.BendingNodeEnd)[11] # Rotation about Z-axis
        return int(moment_j)

    # i-node shear force
    def ShearForce_i(cls,ShearNode):
        cls.ShearNode =  ShearNode
        shear_i = eleForce(cls.ShearNode)[1] # Translation about Y-axis
        return int(shear_i)

    # j-node shear force
    def ShearForce_j(cls,ShearNode):
        cls.ShearNode =  ShearNode
        shear_j = eleForce(cls.ShearNode)[7] # Translation about Y-axis
        return int(shear_j)

# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# Model generation objects


