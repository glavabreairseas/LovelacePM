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

    b=1.1963; croot=0.806; taper=0.56; sweep=26.7
    alpha=5.0; Uinf=10.0

    root_sect=wing_section(afl='onerad', c=croot, xdisc=30, sweep=sweep) #CA_position set to origin as default
    left_tip_sect=wing_section(afl='onerad', c=croot*taper, CA_position=np.array([b*tan(radians(sweep)), -b, 0.0]), closed=True, xdisc=30, sweep=sweep)
    right_tip_sect=wing_section(afl='onerad', c=croot*taper, CA_position=np.array([b*tan(radians(sweep)), b, 0.0]), closed=True, xdisc=30, sweep=sweep)

    sld=Solid()

    left_wingquad=wing_quadrant(sld, sect1=left_tip_sect, sect2=root_sect)
    right_wingquad=wing_quadrant(sld, sect1=root_sect, sect2=right_tip_sect)
    wng=wing(sld, wingquads=[left_wingquad, right_wingquad])

    acft=aircraft(sld, elems=[wng])
    acft.edit_parameters({'a':alpha, 'Uinf':Uinf, 'M':0.7})
    acft.plot_input()

    wng.patchcompose(ydisc=50)
    acft.addwake()
    acft.plotgeometry()

    acft.eulersolve()
    acft.forces_report()
    acft.stabreport()
    plot_Cls(sld, wings=[wng])
    plot_Cds(sld, wings=[wng])
    plot_Cms(sld, wings=[wng])
    plot_Cps(sld, elems=[wng])

    results=acft.design_derivatives(wings=[wng])
    print(results)

    os.chdir(ordir)