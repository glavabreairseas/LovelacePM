import numpy as np
import numpy.linalg as lg
import scipy.linalg as slg
import scipy.linalg.lapack as slapack
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as tm

import toolkit

from paneller import *

'''
Script to test the Euler solution modules with a generic sphere geometry
'''

c=0.01
thetas=np.linspace(pi, -pi, 50)
phis=np.linspace(-pi/2, pi/2, 50)
R=0.5
coordlist=[]
conlist=[]
n=0
thlast=0
for i in range(len(phis)-1):
    coordlist+=[[]]
    for j in range(len(thetas)-1):
        coordlist[-1]+=[R*np.array([[cos(phis[i])*cos(thetas[j]), cos(phis[i])*sin(thetas[j]), sin(phis[i])], \
            [cos(phis[i+1])*cos(thetas[j]), cos(phis[i+1])*sin(thetas[j]), sin(phis[i+1])], \
                [cos(phis[i+1])*cos(thetas[j+1]), cos(phis[i+1])*sin(thetas[j+1]), sin(phis[i+1])], \
                    [cos(phis[i])*cos(thetas[j+1]), cos(phis[i])*sin(thetas[j+1]), sin(phis[i])]]).T]
        if j==0:
            thlast=n
        if j==len(thetas)-2:
            conlist+=[[thlast, n]]
        n+=1
t=tm.time()
sld=Solid(coordlist)
#sld.genwakepanels(conlist)
#sld.plotgeometry(wake=True)
sld.genvbar(0.2)
sld.gennvv()
print('Solid generation and pre-processing: '+str(tm.time()-t))
t=tm.time()
sld.genaicm()
print('Generating AICs: '+str(tm.time()-t))
t=tm.time()
sld.solve(damper=c)
print('Solving and post-processing: '+str(tm.time()-t))

'''sld.plotgeometry()
plt.scatter([p.colpoint[0] for p in sld.panels], \
    [lg.norm(sld.delphi[i, :]+sld.vbar[i, :])/lg.norm(sld.vbar[i, :]) \
    for i in range(len(sld.panels))])
plt.show()

for i in range(sld.npanels):
    print(sld.panels[i].nvector@(sld.delphi[i, :]+sld.vbar[i, :]))'''