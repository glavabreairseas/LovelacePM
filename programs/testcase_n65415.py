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
    a=-2.0
    Uinf=0.05
    b=5.0
    taper=0.3
    croot=1.0
    quarter_sweep=radians(30)
    twist=-3.0
    incidence=0.0

    sld=Solid()

    sect1=wing_section(afl='n65415', CA_position=np.array([b*tan(quarter_sweep)/2, -b/2, 0.0]), c=croot*taper, xdisc=30, incidence=incidence+twist, closed=True)#, xstrategy=lambda x: x)
    sect2=wing_section(afl='n65415', CA_position=np.array([0.0, 0.0, 0.0]), c=croot, xdisc=30, incidence=incidence)#, xstrategy=lambda x: x)
    sect3=wing_section(afl='n65415', CA_position=np.array([b*tan(quarter_sweep)/2, b/2, 0.0]), c=croot*taper, xdisc=30, incidence=incidence+twist, closed=True)#, xstrategy=lambda x: x)

    wng1=wing_quadrant(sld, sect1=sect1, sect2=sect2)
    wng2=wing_quadrant(sld, sect1=sect2, sect2=sect3)
    wng=wing(sld, wingquads=[wng1, wng2])

    acft=aircraft(sld, elems=[wng], Sref=b*croot*(1+taper)/2)
    acft.edit_parameters({'a':a, 'Uinf':Uinf})

    wng.patchcompose(ydisc=40, ystrategy=lambda x: x)#(np.sin(pi*x-pi/2)+1)/2)

    acft.addwake()

    '''sld.plotnormals(xlim=[-0.6, 0.6], ylim=[-0.6, 0.6], zlim=[-0.6, 0.6], factor=0.1)
    sld.plotnormals(xlim=[-0.2, 0.2], ylim=[-0.8, -0.4], zlim=[-0.2, 0.2], factor=0.1)'''
    acft.eulersolve()
    acft.forces_report()
    acft.stabreport()
    plot_Cps(sld, elems=[wng])
    plot_Cls(sld, wings=[wng])
    plot_Cds(sld, wings=[wng])
    plot_Cms(sld, wings=[wng])
    plot_gammas(sld, wings=[wng])
    sld.plotgeometry()