
from xlwt import Workbook
import Bridgemodel
import numpy as np
from Bridge_member import BridgeMember
from openseespy.opensees import *

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
BridgeTest = Bridgemodel.OpenseesModel(Transverse_length = 10.35, Beam_spacing=2, Beam_length=38, Slab_spacing=2, skew=0, beamtype=beamelement) #consider number slab spacing of input 4

# Model assign and generation
BridgeTest.assign_beam_member_prop(MainBeam.get_beam_prop, LRBeam.get_beam_prop, Edgebeam.get_beam_prop, Slab.get_beam_prop, Diaphragm.get_beam_prop)
BridgeTest.assign_material_prop(ConcreteProp, SteelProp)
BridgeTest.create_Opensees_model()
BridgeTest.time_series()
BridgeTest.loadpattern()


# Testing Load -100kN
TestVector = [0,-200e3, 0, 0, 0, 0] # input = Fx, Fy, Fz, Mx, My, Mz (unit N)
BridgeTest.load_singlepoint(6, TestVector)
BridgeTest.load_singlepoint(17, TestVector)
BridgeTest.load_singlepoint(28, TestVector)
BridgeTest.load_singlepoint(39, TestVector)
BridgeTest.load_singlepoint(50, TestVector)
BridgeTest.load_singlepoint(61, TestVector)
BridgeTest.load_singlepoint(72, TestVector)

# Ask Justin: "Why do we need this 2 lines??"
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


print(Bridgemodel.eleForce(5))
a = abs(eleForce(5)[11]) + abs(eleForce(6)[5])
b = eleForce(15)[11] - eleForce(16)[5]
c = eleForce(25)[11] - eleForce(26)[5]
d = eleForce(35)[11] - eleForce(36)[5]
e = eleForce(45)[11] - eleForce(46)[5]
f=  eleForce(55)[11] - eleForce(56)[5]
g=  eleForce(65)[11] - eleForce(66)[5]

m = (a + b + c + d + e + f + g)/2
print('m=',a)
print('m=',b)
print('m=',c)
print('m=',d)
print('m=',e)
print('m=',f)
print('m=',g)
handcalcs = (200e3)*7*20/4
print(m)
print(handcalcs)
