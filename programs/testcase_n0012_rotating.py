from LovelacePM.wing import wing_section, wing_quadrant, wing
from LovelacePM.body import body, smooth_angle_defsect_function
from LovelacePM.aircraft import aircraft
from LovelacePM.paneller import Solid
from LovelacePM.aerodynamic_output import plot_Cps, plot_Cds, plot_Cms, plot_Cls    
from LovelacePM.multiprocess_guard import multiprocess_guard
from LovelacePM.xfoil_visc import polar_correction
from LovelacePM.utils import read_airfoil
from math import tan, radians, sin, cos, pi
import numpy as np
import os
import matplotlib.pyplot as plt
import time as tm
import numpy.linalg as lgp

if multiprocess_guard():
    # ordir=os.getcwd()
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #test case to compare with PMARC's test case of a rotating AR 5 NACA 0012 wing

    c=1.0
    AR=5.0
    b=c*AR
    Uinf=0.3048*1.0 #

    a=0.0
    p=3.0*Uinf/(b*2)

    sld=Solid()

    sl=wing_section(afl='n0012', c=c, CA_position=np.array([0.0, -b/2, 0.0]), xdisc=50, closed=True)
    sr=wing_section(afl='n0012', c=c, CA_position=np.array([0.0, b/2, 0.0]), xdisc=50, closed=True)
    wngqd=wing_quadrant(sld, sect1=sl, sect2=sr)
    wng=wing(sld, wingquads=[wngqd])
    acft=aircraft(sld, elems=[wng])

    acft.edit_parameters({'a':a, 'Uinf':Uinf, 'p':p})

    wng.patchcompose(ydisc=30)
    acft.addwake(wakedisc=50, strategy=lambda x: x, offset=b*10)

    acft.eulersolve(wakeiter=1)

    plot_Cps(sld, elems=[wng])

    acft.plotgeometry()

    # os.chdir(ordir)
