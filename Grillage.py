from xlwt import Workbook
import Bridgemodel
import numpy as np
from Bridge_member import BridgeMember
from openseespy.opensees import *
import matplotlib.pyplot as plt

# Coordinate system: y: vertical axis, x: longitudinal direction, z: transverse direction
# Model inputs
Bridgemodel.wipe()  # clear all previous model

# Material properties
# Concrete
ConcreteProp = [-6.0, -0.004, -6.0, -0.014]
# Steel
SteelProp = []

# BridgeTest Elements Properties
beamelement = "ElasticTimoshenkoBeam"
# input order: Cross-section, Young Modulus, Shear Modulus, Torsion, Second moment area (major y), Second moment area (minor z), element type) N and m
MainBeam = BridgeMember(0.896, 34.6522E9,20E9 ,0.133,0.214,0.259,0.233,0.58,beamelement)
LRBeam = MainBeam
Edgebeam = BridgeMember(0.044625, 34.6522E9, 20E9, 0.26E-3, 0.114E-3, 0.242E-3, 0.0371875, 0.0371875, beamelement)
Slab = BridgeMember(0.4428,34.6522E9 ,20E9 ,2.28E-3 ,0.2233,1.19556E-3,0.369,0.369,beamelement)
Diaphragm = BridgeMember(0.2214,34.6522E9,20E9 ,2.17E-3 ,0.111,0.597E-3,0.1845,0.1845,beamelement)

# Model geomerty
# INPUT ORDER: 1.Transverse length, 2.Beam spacing, 3.Longitudinal(beam) length, 4.Slab spacing, 5.skew, 6.beamtype
BridgeTest = Bridgemodel.OpenseesModel(Transverse_length = 10.35, Beam_spacing=2, Beam_length=20, Slab_spacing=2, skew=0, beamtype=beamelement) #consider number slab spacing of input 4

# Model assign and generation
BridgeTest.assign_beam_member_prop(MainBeam.get_beam_prop, LRBeam.get_beam_prop, Edgebeam.get_beam_prop, Slab.get_beam_prop, Diaphragm.get_beam_prop)
BridgeTest.assign_material_prop(ConcreteProp, SteelProp)
BridgeTest.create_Opensees_model()

#create SOE
Bridgemodel.system("BandSPD")
#create DOF number
Bridgemodel.numberer("RCM")
#create constraint handler
Bridgemodel.constraints("Plain")
#create integrator
Bridgemodel.integrator("LoadControl", 1.0)
#create algorithm
Bridgemodel.algorithm("Linear")
#create analysis object
Bridgemodel.analysis("Static")
#perform the analysis
Bridgemodel.analyze(1)
BridgeTest.time_series()
BridgeTest.loadpattern()

# Print information to avoid repeat when looping model generation
print("X coord =  ", Bridgemodel.OpenseesModel.ele_x)
print("Z coord = ", Bridgemodel.OpenseesModel.ele_z)
print("Model DIM = {}, Model DOF = {}".format(Bridgemodel.OpenseesModel.ndm, Bridgemodel.OpenseesModel.ndf))
print("Model's nodes layout is:\n",Bridgemodel.OpenseesModel.nodetag)
print('Nodes generated =', Bridgemodel.OpenseesModel.totalnodes)  # number here is the total nodes
print('Support nodes are =',  Bridgemodel.OpenseesModel.edgesupport)
print(Bridgemodel.OpenseesModel.concrete)
print(Bridgemodel.OpenseesModel.geo)
print("Total Number of elements = ", Bridgemodel.OpenseesModel.totalele)
print('Longitudinal member element tag:\n',
      Bridgemodel.OpenseesModel.Ledge_ele,'\n',
      Bridgemodel.OpenseesModel.L_ele,'\n',
      Bridgemodel.OpenseesModel.MIDele,'\n',
      Bridgemodel.OpenseesModel.R_ele,'\n',
      Bridgemodel.OpenseesModel.Redge_ele,'\n')
print('The transverse element layout is:')
for i in range(len(Bridgemodel.OpenseesModel.diapS_ele)):
    print(Bridgemodel.OpenseesModel.diapS_ele[i],Bridgemodel.OpenseesModel.Slab_ele[i],Bridgemodel.OpenseesModel.diapR_ele[i],)
print('\n')



# Testing Load -100kN
#TestVector = [0,-100e3, 0, 0, 0, 0] # input = Fx, Fy, Fz, Mx, My, Mz (unit N)
#BridgeTest.load_singlepoint(6, TestVector)
#BridgeTest.load_singlepoint(17, TestVector)
#BridgeTest.load_singlepoint(28, TestVector)
#BridgeTest.load_singlepoint(39, TestVector)
#BridgeTest.load_singlepoint(50, TestVector)
#BridgeTest.load_singlepoint(61, TestVector)
#BridgeTest.load_singlepoint(72, TestVector)

# Multiple point load test
#NodesVector = [28, 39, 17]
#BridgeTest.load_multipoint(NodesVector,TestVector)


# MOVING TRUCK ANALYSIS

#-----------------------------------------------START OF USER INPUT---------------------------------------------------------------------------------------------
# Truck definition USER INPUT HERE
axleswts = [30e3, 50e3, 50e3] # Weight of each axle (N) e.g. 3 axle truck (30kN, 50kN, 50kN)
S_axles = [0, 2, 3]  # Distance between each axle (m) # Future improvement: change the function so that place holder 0  is not required
w = 3  # Width of the truck (m)
initialpos = [0, 5.0875] # Initial position of truck before travel [Xo, Zo] [Longitudianl and transverse direction]
TravelLength = 30 # Intended travel distance (m)
increment = 1 # distance increment (m)
#------------------------------------------------END OF USER INPUT-----------------------------------------------------------------------------------------------


#--------------------------------------------BENDING AND SHEAR ANALYSIS-----------------------------------------------------------------------------------
# NOTE: Black box, change with care

newpos = initialpos
Trucklength = sum(S_axles)
if TravelLength - Trucklength  > Bridgemodel.OpenseesModel.Lx:  # If loop to check whether truck travel out of the bridge
    TravelLength = Bridgemodel.OpenseesModel.Lx + Trucklength

BM = np.zeros( (TravelLength +1 , Bridgemodel.OpenseesModel.totalnodes)) # Array for bending moment
SF = np.zeros( (TravelLength +1 , Bridgemodel.OpenseesModel.totalnodes)) # Array for shear force

wb = Workbook() #Create Excel workbook
sheet1 = wb.add_sheet('Bending') # Create sheet within workbook
sheet2 = wb.add_sheet('Shear')
X_plot = [] # Initialize for plotting

for r in range(TravelLength + 1):
    Bridgemodel.wipe()
    BridgeTest = Bridgemodel.OpenseesModel(Transverse_length = 10.35, Beam_spacing=2, Beam_length=20, Slab_spacing=2, skew=0, beamtype=beamelement)
    BridgeTest.assign_beam_member_prop(MainBeam.get_beam_prop, LRBeam.get_beam_prop, Edgebeam.get_beam_prop, Slab.get_beam_prop, Diaphragm.get_beam_prop)
    BridgeTest.assign_material_prop(ConcreteProp, SteelProp)
    BridgeTest.time_series()
    BridgeTest.loadpattern()
    BridgeTest.create_Opensees_model()
    truckindex = 'Truck position (x,z) = %s' % initialpos
    sheet1.write(r+1,0, truckindex)

    if newpos[0] - Trucklength  > Bridgemodel.OpenseesModel.Lx:
        print('Truck first axle location: ',initialpos)
        print('Truck last axle outside of bridge deck')
        break
    else:
        #print('Truck first axle location: ',initialpos)
        BridgeTest.load_statictruck(initialpos,axleswts,S_axles,w)
        initialpos = newpos
        newpos[0] =  initialpos[0] + increment
        #create SOE
        Bridgemodel.system("BandSPD")
        #create DOF number
        Bridgemodel.numberer("RCM")
        #create constraint handler
        Bridgemodel.constraints("Plain")
        #create integrator
        Bridgemodel.integrator("LoadControl", 1.0)
        #create algorithm
        Bridgemodel.algorithm("Linear")
        #create analysis object
        Bridgemodel.analysis("Static")
        #perform the analysis
        Bridgemodel.analyze(1)

        for c in range(Bridgemodel.OpenseesModel.totalnodes):

            if c + 1 in (Bridgemodel.OpenseesModel.edgesupport[0::2]): # [0::2] give odd element in edgesupport list (left edge support)
                    BM[r][c] = BridgeTest.BendingMoment_i(c+1) # only i-node contribution
                    SF[r][c] = BridgeTest.ShearForce_i(c+1)
                    # Writing in Excel file
                    sheet1.write(r+1,c+1, BM[r][c])
                    sheet2.write(r+1,c+1, SF[r][c])
            elif c + 1 in (Bridgemodel.OpenseesModel.edgesupport[1::2]): # [1::2] give odd element in edgesupport list (right edge support)
                    BM[r][c] = BridgeTest.BendingMoment_j(c+1) # only j-node contribution
                    SF[r][c] = BridgeTest.ShearForce_j(c+1)
                    # Writing in Excel file
                    sheet1.write(r+1,c+1, BM[r][c])
                    sheet2.write(r+1,c+1, SF[r][c])
            else:
                    BM[r][c] = (abs(BridgeTest.BendingMoment_j(c+1)) + abs(BridgeTest.BendingMoment_i(c+1)))/2 # Nodel average of node i of member before and j of member after
                    SF[r][c] = (abs(BridgeTest.ShearForce_j(c+1)) + abs(BridgeTest.ShearForce_i(c+1)))/2
                    # Writing in Excel file
                    sheet1.write(r+1,c+1, BM[r][c])
                    sheet2.write(r+1,c+1, SF[r][c])

        wb.save('Analysis.xls') # save the excel file
        X_plot.append(r)
#--------------------------------------------------- END OF BENDING AND SHEAR ANALYSIS-------------------------------------------------------------------------------

# POST PROCESSING
# Plotting node BM with truck position
    # INPUT nodes interested in interested_node array
interested_node = [39,34]

for nodes in range(len(interested_node)):
    Yplot = []
    for row in BM: #change BM to SF if shear force is aftered
        Yplot.append(row[interested_node[nodes]-1])

    plt.plot(X_plot,Yplot,label = 'Node %s' % interested_node[nodes]) #legend for each line

# Plot editing
plt.legend()
plt.ylabel('Bending Moment (Nm)')
plt.xlabel('Truck location')
plt.grid(True)
plt.show()

#----------------------------------------------------------------------------------------

