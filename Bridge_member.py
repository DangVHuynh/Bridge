# class for bridge member properties
class BridgeMember:
    # input order: Cross-section, Young Modulus, Shear Modulus, Torsion, Second moment area (major y), Second moment area (minor z), element type) unit N and m
    def __init__(self, a, e, g, jx, iy, iz,Avy,Avz,beameletype):
        self.A = a
        self.E = e
        self.G = g
        self.Jx = jx
        self.Iy = iy
        self.Iz = iz
        self.Avy = Avy
        self.Avz = Avz
        self.beameletype = beameletype
    @property
    def get_beam_prop(self):
        if self.beameletype == 'elasticBeamColumn':
        #                           A  E  G  Jx  Iy   Iz                   # N m
            self.formatsummary = [self.A, self.E, self.G, self.Jx, self.Iy, self.Iz]

        else:
            self.formatsummary = [self.E, self.G, self.A, self.Jx, self.Iy, self.Iz, self.Avy, self.Avz]

        return self.formatsummary



# Longbeam = BridgeMember(0.41631,31.113E9 ,1 ,4.23662E-3,0.0380607,0.0484717)
#
# print(Longbeam.E)