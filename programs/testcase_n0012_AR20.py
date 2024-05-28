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
import numpy.linalg as lg

if multiprocess_guard():
    ordir=os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    c=0.5
    AR=20.0
    b=c*AR
    nu=1.72e-5
    rho=1.225
    Uinf=10.0
    Re=Uinf*rho*c

    a=0.0

    n0012_visc=polar_correction('n0012', aseq=[-5.0, 5.0, 0.5])

    sld=Solid()
    sl=wing_section(airfoil='n0012', chord=c, center_position=np.array([0.0, -b/2, 0.0]), n_points_per_side=30, closed=True, correction=n0012_visc, Reynolds=Re)
    sr=wing_section(airfoil='n0012', chord=c, center_position=np.array([0.0, b/2, 0.0]), n_points_per_side=30, closed=True, correction=n0012_visc, Reynolds=Re)
    wngqd=wing_quadrant(sld, sect1=sl, sect2=sr)
    wng=wing(sld, wingquads=[wngqd])
    acft=aircraft(sld, elems=[wng])

    acft.edit_parameters({'a':a, 'Uinf':Uinf})

    wng.patchcompose(ydisc=30)
    acft.addwake()

    acft.eulersolve()
    acft.forces_report()

    plot_Cps(sld, elems=[wng])

    os.chdir(ordir)