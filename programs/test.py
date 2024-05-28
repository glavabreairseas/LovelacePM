from src.LovelacePM.wing import wing_section, wing_quadrant, wing
from src.LovelacePM.aircraft import aircraft
from src.LovelacePM.paneller import Solid
from src.LovelacePM.aerodynamic_output import plot_Cps, plot_Cds, plot_Cls, plot_Cms
import numpy as np
from math import tan, radians

b=1.1963; croot=0.806; taper=0.56; sweep=26.7

sld=Solid()

root_sect=wing_section(afl='onerad', c=croot, xdisc=30, sweep=sweep) #CA_position set to origin as default
left_tip_sect=wing_section(afl='onerad', c=croot*taper, CA_position=np.array([b*tan(radians(sweep)), -b, 0.0]), closed=True, xdisc=30, sweep=sweep)
right_tip_sect=wing_section(afl='onerad', c=croot*taper, CA_position=np.array([b*tan(radians(sweep)), b, 0.0]), closed=True, xdisc=30, sweep=sweep)

left_wingquad=wing_quadrant(sld, sect1=left_tip_sect, sect2=root_sect)
right_wingquad=wing_quadrant(sld, sect1=root_sect, sect2=right_tip_sect)
wng=wing(sld, wingquads=[left_wingquad, right_wingquad])

acft=aircraft(sld, elems=[wng])
alpha=5.0; Uinf=10.0
acft.edit_parameters({'a':alpha, 'Uinf':Uinf, 'M':0.5})

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