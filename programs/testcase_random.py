import numpy.random as rnd
from src.LovelacePM.wing import wing_section, wing_quadrant, wing
from src.LovelacePM.body import body, smooth_angle_defsect_function
from src.LovelacePM.aircraft import aircraft
from src.LovelacePM.paneller import Solid
from src.LovelacePM.aerodynamic_output import plot_Cps, plot_Cds, plot_Cms, plot_Cls    
from src.LovelacePM.multiprocess_guard import multiprocess_guard
from src.LovelacePM.xfoil_visc import polar_correction
from src.LovelacePM.utils import read_airfoil
from math import tan, radians, sin, cos, pi
import numpy as np
import os
import matplotlib.pyplot as plt
import time as tm
import numpy.linalg as lg

if multiprocess_guard():
    pan1=rnd.random(12).reshape(3, 4)
    pan2=rnd.random(12).reshape(3, 4)

    '''pan1=np.array([[0.59116381, 0.06286042, 0.71395661, 0.94342679], \
        [0.20698101, 0.25343613, 0.65463535, 0.9292159 ], \
            [0.99920911, 0.43311108, 0.57953426, 0.86468112]])
    pan2=np.array([[0.02830673, 0.45917184, 0.40259338, 0.39839244], \
    [0.67288593, 0.82413215, 0.31487439, 0.73770071], \
    [0.0768672,  0.13285883, 0.13590843, 0.81177605]])'''
    print(pan1)
    print(pan2)

    sld=Solid(sldlist=[pan1, pan2])
    sld.genaicm()
    print(sld.aicm3)