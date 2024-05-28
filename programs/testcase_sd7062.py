import numpy.random as rnd
from LovelacePM.wing import wing_section, wing_quadrant, wing
from LovelacePM.body import body, smooth_angle_defsect_function
from LovelacePM.aircraft import aircraft
from LovelacePM.paneller import Solid
from LovelacePM.aerodynamic_output import plot_Cps, plot_Cds, plot_Cms, plot_Cls, plot_gammas
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
    a=0.0
    q=0.1
    Uinf=0.05
    b=1.2
    taper=1.0
    croot=0.3

    sld=Solid()

    sect1=wing_section(afl='sd7062', CA_position=np.array([0.0, -b/2, 0.0]), c=croot*taper, xdisc=30, closed=True)#, xstrategy=lambda x: x)
    sect2=wing_section(afl='sd7062', CA_position=np.array([0.0, b/2, 0.0]), c=croot, xdisc=30, closed=True)#, xstrategy=lambda x: x)

    wng1=wing_quadrant(sld, sect1=sect1, sect2=sect2)
    wng=wing(sld, wingquads=[wng1])
    acft=aircraft(sld, elems=[wng], Sref=b*croot*(1+taper)/2)
    wng.patchcompose(ydisc=30)
    acft.edit_parameters({'a':a, 'Uinf':Uinf, 'q':q})
    acft.addwake(wakedisc=30, offset=10.0)#, strategy=lambda x:x)

    acft.eulersolve(wakeiter=1)
    acft.forces_report()
    acft.stabreport()
    plot_Cps(sld, elems=[wng])
    plot_Cls(sld, wings=[wng])
    plot_Cds(sld, wings=[wng])
    plot_Cms(sld, wings=[wng])
    plot_gammas(sld, wings=[wng])
    sld.plotgeometry()

    acft.edit_parameters({'a':a, 'Uinf':Uinf, 'q':q})
    acft.resolve()
    acft.forces_report()
    acft.stabreport()
    plot_Cps(sld, elems=[wng])
    plot_Cls(sld, wings=[wng])
    plot_Cds(sld, wings=[wng])
    plot_Cms(sld, wings=[wng])
    plot_gammas(sld, wings=[wng])
    sld.plotgeometry()

    acft.design_derivatives()