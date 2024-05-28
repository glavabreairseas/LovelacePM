import numpy as np
import numpy.linalg as lg
import scipy.linalg as slg
import scipy.linalg.lapack as slapack
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as tm

from src.LovelacePM.paneller import *
from src.LovelacePM.utils import *
from src.LovelacePM.wing import *
from src.LovelacePM.body import *
from src.LovelacePM.aircraft import *
from src.LovelacePM.xfoil_visc import *
from src.LovelacePM.aerodynamic_output import *
from src.LovelacePM.multiprocess_guard import *

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