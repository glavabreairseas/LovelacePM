from src.LovelacePM.wing import wing_section, wing_quadrant, wing
from src.LovelacePM.body import body, smooth_angle_defsect_function
from src.LovelacePM.aircraft import aircraft
from src.LovelacePM.paneller import Solid
from src.LovelacePM.aerodynamic_output import plot_Cps, plot_Cds, plot_Cms, plot_Cls    
from src.LovelacePM.multiprocess_guard import multiprocess_guard
from src.LovelacePM.xfoil_visc import polar_correction
from src.LovelacePM.utils import read_airfoil
from math import tan, radians
import numpy as np
import os
import matplotlib.pyplot as plt
import time as tm

if multiprocess_guard():
    sld=Solid()

    b=3.0
    c=0.3
    Uinf=20.0

    sect1=wing_section(afl='n4412', c=c, CA_position=np.array([0.0, -b/2, 0.0]), xdisc=30)
    sect2=wing_section(afl='n4412', c=c, CA_position=np.array([0.0, 0.0, 0.0]), xdisc=30)
    sect3=wing_section(afl='n4412', c=c, CA_position=np.array([0.0, b/2, 0.0]), xdisc=30)
    wngqd1=wing_quadrant(sld, sect1=sect1, sect2=sect2, control_names=['aileron', 'tab'], control_axpercs_x=[[0.75, 0.75], [0.9, 0.9]], control_axpercs_thickness=[[0.5, 0.5], [0.5, 0.5]])
    wngqd2=wing_quadrant(sld, sect1=sect2, sect2=sect3)
    wng=wing(sld, wingquads=[wngqd1, wngqd2])
    acft=aircraft(sld, elems=[wng])
    acft.edit_parameters({'Uinf':Uinf, 'aileron':30.0, 'tab':20.0})
    wng.patchcompose(ydisc=70)
    sld.plotgeometry()
    acft.addwake()
    acft.eulersolve()
    print(wngqd1.hinge_moments())
    plot_Cps(sld, elems=[wng])
    plot_Cls(sld, wings=[wng])
    sld.plotgeometry()
    plot_Cps(sld, elems=[wng])
    plot_Cls(sld, wings=[wng])
    plot_Cds(sld, wings=[wng])
    plot_Cms(sld, wings=[wng])