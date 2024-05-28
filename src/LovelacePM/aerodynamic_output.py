import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator as LNDI

from src.LovelacePM.paneller import Solid
from matplotlib import axes


def plot_quarter_chord_cp(
    sld: Solid, half_span: float = 1.0, ax: axes = None, elems: list = []
):
    """Plots the pressure coefficient at quaterchord position
    Works only with flat wings without sweep.

    :param sld: The lovelace Solid object containing the data
    :param half_span: halfspan of the main wing of sld, defaults to 1.0
    :param ax: matplotlib ax to fill with plot, defaults to None
    :param elems: list of elements (lifting surfaces) from which
                    to extract the data, defaults to []

    """
    if len(elems) != 0 and sld.runme:
        # fig = plt.figure()  # noqa F841
        showNow = False
        if ax is None:
            ax = plt.axes()
            showNow = True

        # Get raw data for extrados
        xs = []
        ys = []
        cps = []
        for elem in elems:
            for i in elem.paninds:
                if sld.panels[i].nvector[2] >= 0:
                    xs += [sld.panels[i].colpoint[0]]
                    ys += [sld.panels[i].colpoint[1]]
                    cps += [sld.Cps[i]]
        # Build interpolator
        extra_interp = LNDI(points=np.array([xs, ys]).T, values=cps)

        # Get raw data for intrados
        xs = []
        ys = []
        cps = []
        for elem in elems:
            for i in elem.paninds:
                if sld.panels[i].nvector[2] < 0:
                    xs += [sld.panels[i].colpoint[0]]
                    ys += [sld.panels[i].colpoint[1]]
                    cps += [sld.Cps[i]]
        intra_interp = LNDI(points=np.array([xs, ys]).T, values=cps)

        n_point = 100
        x_plot = np.linspace(-half_span, half_span, n_point)
        plot_points = np.zeros((n_point, 2))
        plot_points[:, 1] = x_plot  # Replace y-coordinate

        ax.plot(
            x_plot / 0.3048, -extra_interp(plot_points), label="extrados"
        )  # abscissa in foot
        ax.plot(x_plot / 0.3048, -intra_interp(plot_points), label="intrados")

        ax.set_xlabel("Spanwise coordinate [ft]")
        ax.set_ylabel("- C_p [-]")
        ax.set_title("Quarterchord pressure distribution")
        plt.legend()
        if showNow:
            plt.show()


def plot_Cps(sld: Solid, ax: axes=None, elems: list=[], xlim=[], ylim=[], zlim=[-2.5, 2.5]):
    if len(elems) != 0 and sld.runme:
        # fig = plt.figure()  # noqa F841
        showNow = False
        if ax is None:
            ax = plt.axes(projection="3d")
            showNow = True
        xs = []
        ys = []
        cps = []
        for elem in elems:
            for i in elem.paninds:
                if sld.panels[i].nvector[2] >= 0:
                    xs += [sld.panels[i].colpoint[0]]
                    ys += [sld.panels[i].colpoint[1]]
                    cps += [sld.Cps[i]]
        ax.scatter3D(xs, ys, cps, "blue")
        xs = []
        ys = []
        cps = []
        for elem in elems:
            for i in elem.paninds:
                if sld.panels[i].nvector[2] < 0:
                    xs += [sld.panels[i].colpoint[0]]
                    ys += [sld.panels[i].colpoint[1]]
                    cps += [sld.Cps[i]]
        ax.scatter3D(xs, ys, cps, "red")
        if len(xlim) != 0:
            ax.set_xlim3d(xlim[0], xlim[1])
        if len(ylim) != 0:
            ax.set_ylim3d(ylim[0], ylim[1])
        if len(zlim) != 0:
            ax.set_zlim3d(zlim[0], zlim[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("C_p")
        if showNow:
            plt.show()


def plot_Cls(sld, ax=None, alpha=0.0, wings=[], axis=1):

    showNow = False
    if ax is None:
        ax = plt.axes()
        showNow = True

    if len(wings) != 0 and sld.runme:
        ypos = [0.0]
        plotcorr = False
        n = 1
        for w in wings:
            if not w.coefavailable:
                w.calc_coefs(axis=axis, alpha=alpha)
            ypos += [np.amax(w.Cls), np.amin(w.Cls)]
            ax.plot(w.ys, w.Cls, label="Inviscid w" + str(n))
            if w.plotcorrections:
                ax.plot(w.ys, w.Cls_corrected, label="Corrected w" + str(n))
                ypos += [np.amax(w.Cls_corrected), np.amin(w.Cls_corrected)]
                plotcorr = True
            n += 1
        ax.set_title("Sectional lift coefficient")
        ax.set_xlabel("y")
        ax.set_ylabel("C_l")
        ax.set_ylim(min(ypos) - 0.1 * abs(min(ypos)), max(ypos) + 0.1 * abs(max(ypos)))
        ax.grid()
        if plotcorr:
            ax.legend()
        if showNow:
            plt.show()


def plot_gammas(sld, alpha=0.0, Uinf=1.0, wings=[], axis=1):
    if len(wings) != 0 and wings[0].sld.runme:
        ypos = [0.0]
        for w in wings:
            if not w.coefavailable:
                w.calc_coefs(axis=axis, alpha=alpha)
            # gammas=w.Cls*w.cs*Uinf/2
            gammas = w.Gammas
            ypos += [np.amax(gammas), np.amin(gammas)]
            plt.plot(w.ys, gammas, "blue", label="Inviscid")
        plt.title("Sectional circulation")
        plt.xlabel("y")
        plt.ylabel("Gamma")
        plt.ylim(min(ypos) - 0.1 * abs(min(ypos)), max(ypos) + 0.1 * abs(max(ypos)))
        plt.grid()
        plt.show()


def plot_Cds(sld, alpha=0.0, wings=[], axis=1):
    if len(wings) != 0 and sld.runme:
        ypos = [0.0]
        plotcorr = False
        n = 1
        for w in wings:
            if not w.coefavailable:
                w.calc_coefs(axis=axis, alpha=alpha)
            ypos += [np.amax(w.Cds), np.amin(w.Cds)]
            plt.plot(w.ys, w.Cds, label="Inviscid w" + str(n))
            if w.plotcorrections:
                plt.plot(w.ys, w.Cds_corrected, label="Corrected w" + str(n))
                ypos += [np.amax(w.Cds_corrected), np.amin(w.Cds_corrected)]
                plotcorr = True
            n += 1
        plt.title("Sectional drag coefficient")
        plt.xlabel("y")
        plt.ylabel("C_d")
        plt.ylim(min(ypos) - 0.1 * abs(min(ypos)), max(ypos) + 0.1 * abs(max(ypos)))
        plt.grid()
        if plotcorr:
            plt.legend()
        plt.show()


def plot_Cms(sld, alpha=0.0, wings=[], axis=1):
    if len(wings) != 0 and sld.runme:
        ypos = [0.0]
        plotcorr = False
        n = 1
        for w in wings:
            if not w.coefavailable:
                w.calc_coefs(axis=axis, alpha=alpha)
            ypos += [np.amax(w.Cms), np.amin(w.Cms)]
            plt.plot(w.ys, w.Cms, label="Inviscid w" + str(n))
            if w.plotcorrections:
                plt.plot(w.ys, w.Cms_corrected, label="Corrected w" + str(n))
                ypos += [np.amax(w.Cms_corrected), np.amin(w.Cms_corrected)]
                plotcorr = True
            n += 1
        plt.title("Sectional quarter-chord torque coefficient")
        plt.xlabel("y")
        plt.ylabel("C_m1/4")
        plt.ylim(min(ypos) - 0.1 * abs(min(ypos)), max(ypos) + 0.1 * abs(max(ypos)))
        plt.grid()
        if plotcorr:
            plt.legend()
        plt.show()
