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
    '''
    Script made to test the Euler solution modules with a flat-plate straight wing
    '''

    b=1.0
    c=1.0

    u=0.05

    ys=np.linspace(-b/2, b/2, 50)
    xs=np.linspace(0.0, c, 50)

    wakeon=[]
    coords=[]
    npan=0
    geomlist=[]
    for i in range(len(xs)-1):
        coords=[]
        for j in range(len(ys)-1):
            coords+=[np.array([[xs[i+1], xs[i+1], xs[i], xs[i]], \
                [ys[j], ys[j+1], ys[j+1], ys[j]], [0.0, 0.0, 0.0, 0.0]])]
            if i==len(xs)-2:
                wakeon+=[[npan, -1]]
            npan+=1
        geomlist+=[coords]
    sld=Solid(sldlist=[geomlist])
    sld.genwakepanels(wakecombs=wakeon, a=radians(30.0))
    sld.genvbar(u, a=radians(30.0))
    sld.gennvv()
    t=tm.time()
    sld.genaicm()
    print(tm.time()-t)
    t=tm.time()
    sld.solve(damper=0.0)
    print(tm.time()-t)
    sld.calcpress(Uinf=u)
    sld.plotpress()

    sld.plotgeometry(wake=False)

    sld.calcforces()
    print('CFres: '+str(sld.SCFres))
    print('CMres: '+str(sld.SCMres))

    fig=plt.figure()
    ax=plt.axes(projection='3d')

    xs=[]
    ys=[]
    for i in range(sld.npanels):
        xs+=[sld.panels[i].colpoint[0]]
        ys+=[sld.panels[i].colpoint[1]]
    ax.scatter3D(xs, ys, sld.Cps)
    plt.show()