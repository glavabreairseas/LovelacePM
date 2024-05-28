from pathlib import Path
from typing import Union, Callable
import numpy as np
from scipy.interpolate import CubicSpline
from math import pi, cos, sin
import numpy.linalg as lg
import quaternion  # noqa


def read_airfoil(
    airfoil_database: Union[Path, str],
    airfoil: str,
    n_points_per_airfoil_side: int,
    discretization_strategy: Callable = lambda x: (np.sin(pi * x - pi / 2) + 1) / 2,
    remove_trailing_edge_gap: bool = False,
    extra_intra: bool = False,
    inverted: bool = False,
    closed: bool = False,
    header_lines: int = 1,
) -> np.ndarray:
    """Reads an airfoil definition file.
    The supported format is the standard format for airfoil definition:
    A text file with two columns containing x and y coordinates,
    with x varying between 0 and 1.

    Caveat ! Not all airfoil files are created equal.
    This function supports only files where:
     - The first column (x)
        starts from 1.0 (trailing edge), decreases to 0.0 (leading edge)
        then increases again to 1.0.
     - The second column (y) is first mostly positive (extrados),
        then mostly negative (intrados)

    :param airfoil_database: Path of airfoil database
    :param airfoil: Name of the airfoil
    :param n_points_per_airfoil_side: Number of interpolated points on
                                the extrados and intrados
    :param discretization_strategy: Repartition of the points on each
                                side of the profile
    :param remove_trailing_edge_gap: Close the gap on the trailing edge,
                                defaults to False
    :param extra_intra: Return the extrados and intrados points separately,
                                defaults to False
    :param inverted: flips the airfoil upside down, defaults to False
    :param closed: Flag for tip sections which must be closed, defaults to False
    :param header_lines: Number of header lines in the airfoil file, defaults to 1
    :return: Array of 2D airfoil point coordinates
    """
    with open(Path(airfoil_database, airfoil + ".dat"), "r") as infile:
        aflpts = []
        alltext = infile.read()
        lines = alltext.split("\n")
        for i in range(header_lines, len(lines)):
            linelist = lines[i].split()
            if len(linelist) != 0:
                aflpts += [[float(linelist[0]), float(linelist[1])]]
        aflpts = np.array(aflpts)
        if not inverted:
            # By default, invert y coordinate so that, in fine,
            # the extrados points to negative z
            aflpts[:, 1] = -aflpts[:, 1]

        if n_points_per_airfoil_side != 0:
            xpts = discretization_strategy(
                np.linspace(1.0, 0.0, n_points_per_airfoil_side)
            )
            leading_edge_ind = np.argmin(aflpts[:, 0])
            extra = aflpts[0:leading_edge_ind, :]
            intra = aflpts[leading_edge_ind : np.size(aflpts, 0), :]
            extracs = CubicSpline(np.flip(extra[:, 0]), np.flip(extra[:, 1]))
            extra = np.vstack((xpts, extracs(xpts))).T
            xpts = np.flip(xpts)
            intracs = CubicSpline(intra[:, 0], intra[:, 1])
            intra = np.vstack((xpts, intracs(xpts))).T
            aflpts = np.vstack((extra, intra[1 : np.size(intra, 0), :]))
        aflpts[:, 0] -= 0.25
        intra[:, 0] -= 0.25
        extra[:, 0] -= 0.25

        if remove_trailing_edge_gap:
            midpoint = (aflpts[-1, :] + aflpts[0, :]) / 2
            aflpts[-1, :] = midpoint
            aflpts[0, :] = midpoint

        if extra_intra:
            tipind = np.argmin(aflpts[:, 0])
            extra = aflpts[0:tipind, :]
            intra = aflpts[tipind : np.size(aflpts, 0), :]
            extra = np.flip(extra, axis=0)
            return aflpts, extra, intra

        if closed:
            tipind = np.argmin(aflpts[:, 0])
            extra = aflpts[0 : tipind + 1, :]
            intra = aflpts[tipind : np.size(aflpts, 0), :]
            extra = np.flip(extra, axis=0)
            camberline = (intra + extra) / 2
            aflpts[:, 1] = np.interp(aflpts[:, 0], camberline[:, 0], camberline[:, 1])

    return aflpts


def position_airfoil(
    airfoil_points, position, chord, sweep_deg, twist_deg, dihedral_deg
):
    """3d-izes, Resizes, rotates and translates the airfoil points

    :param airfoil_points: Array of normalized airfoil point positions
    :param position: Target position of the airfoil's origin
    :param chord: Scaled chord of the airfoil
    :param sweep_deg: sweep angle of the airfoil (usually zero in Kide)
    :param twist_deg: twist angle of the airfoil
    :param dihedral_deg: dihedral angle of the airfoil
    :return: Final position of airfoil points
    """
    num_of_points = airfoil_points.shape[0]

    # 3D-izing
    airfoil_points_3d = np.zeros((num_of_points, 3))
    airfoil_points_3d[:, 0] = airfoil_points[:, 0]
    airfoil_points_3d[:, 2] = airfoil_points[:, 1]

    # Resizing
    airfoil_points_3d *= chord

    # Rotation: Sweep, then dihedral, then twist
    q_sweep = np.quaternion(np.cos(sweep_deg / 2), 0, 0, np.sin(sweep_deg / 2))
    q_dihedral = np.quaternion(np.cos(dihedral_deg / 2), np.sin(dihedral_deg / 2), 0, 0)
    q_twist = np.quaternion(np.cos(twist_deg / 2), 0, np.sin(twist_deg / 2), 0)
    q_total = q_sweep * q_dihedral * q_twist

    positioned_points = np.zeros_like(airfoil_points_3d)
    for i in range(num_of_points):
        positioned_points[i, :] = quaternion.rotate_vectors(
            q_total, airfoil_points_3d[i, :]
        )

    # Translation
    positioned_points += position

    return positioned_points


def wing_afl_positprocess(afl, gamma=0.0, c=1.0, ypos=0.0, xpos=0.0, zpos=0.0):
    # position airfoil coordinate in 3D axis system
    R = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos(gamma), -sin(gamma)], [0.0, sin(gamma), cos(gamma)]]
    )
    aflnew = (
        R @ np.vstack((afl[:, 0] * c, np.zeros(np.size(afl, 0)), afl[:, 1] * c))
    ).T
    aflnew[:, 1] += ypos
    aflnew[:, 2] += zpos
    aflnew[:, 0] += xpos
    return aflnew


def trimlist(
    n, l
):  # trim list to defined length based on first element. For input handling
    if (len(l) < n) and len(l) != 0:
        l += [l[0]] * (n - len(l))
    return l


def trim_polars(th):
    if th > pi:
        return th - 2 * pi
    elif th < -pi:
        return th + 2 * pi


def trim_polars_array(thspacing):  # trim polars to eliminate congruous equivalences
    validpos = thspacing >= pi
    thspacing[validpos] -= 2 * pi
    validpos = thspacing <= -pi
    thspacing[validpos] += 2 * pi
    return thspacing


def gen_circdefsect_coords(disc):  # present input coordinates for circular defsect
    thetas = np.linspace(-pi, pi, disc)
    return np.vstack((np.sin(thetas), np.cos(thetas))).T


def linear_pts(p1, p2, n, endpoint=False):
    # function to provide linearly interpolated segments in 2D space,
    # serves as tool for other functions in folder
    eta = np.linspace(0.0, 1.0, n, endpoint=endpoint)
    coords = np.zeros((n, 2))
    for i in range(n):
        coords[i, :] = (1.0 - eta[i]) * p1 + eta[i] * p2
    return coords


def elliptic_pts(p1, p2, center, r_x, r_y, th1, th2, n, endpoint=False):
    # function to provide elliptically interpolated segments in 2D space,
    # serves as tool for other functions in folder
    thspacing = np.linspace(th1, th2, n, endpoint=endpoint)
    coords = np.zeros((n, 2))
    coords[:, 0] = np.sin(thspacing) * r_x + center[0]
    coords[:, 1] = np.cos(thspacing) * r_y + center[1]
    return coords


def gen_squaredefsect_coords(disc):  # present input coordinates for square defsect
    nside = int(disc / 8)
    pts = np.zeros((nside * 8, 2))
    # side 1
    pts[0:nside, 0] = np.linspace(0.0, -1.0, nside, endpoint=False)
    pts[0:nside, 1] = -1.0
    # side 2
    pts[nside : 3 * nside, 1] = np.linspace(-1.0, 1.0, 2 * nside, endpoint=False)
    pts[nside : 3 * nside, 0] = -1.0
    # side 3
    pts[3 * nside : 5 * nside, 0] = np.linspace(-1.0, 1.0, 2 * nside, endpoint=False)
    pts[3 * nside : 5 * nside, 1] = 1.0
    # side 4
    pts[5 * nside : 7 * nside, 1] = np.linspace(1.0, -1.0, 2 * nside, endpoint=False)
    pts[5 * nside : 7 * nside, 0] = 1.0
    # side 5
    pts[7 * nside : 8 * nside, 0] = np.linspace(1.0, 0.0, nside, endpoint=True)
    pts[7 * nside : 8 * nside, 1] = -1.0

    return pts


def smooth_angle_defsect_coords(r_1x, r_2x, r_1y, r_2y, ldisc=30, thdisc=20):
    # coordinates for body.py's smooth coordinate defsect
    n_low = ldisc
    n_sides = ldisc
    n_up = ldisc
    coords = linear_pts(np.array([0.0, -1.0]), np.array([r_1x - 1.0, -1.0]), n_low)
    coords = np.vstack(
        (
            coords,
            elliptic_pts(
                np.array([r_1x - 1.0, -1.0]),
                np.array([-1.0, r_1y - 1.0]),
                np.array([r_1x - 1.0, r_1y - 1.0]),
                r_1x,
                r_1y,
                -pi,
                -pi / 2,
                thdisc,
            ),
        )
    )
    coords = np.vstack(
        (
            coords,
            linear_pts(
                np.array([-1.0, r_1y - 1.0]), np.array([-1.0, 1.0 - r_2y]), n_sides
            ),
        )
    )
    coords = np.vstack(
        (
            coords,
            elliptic_pts(
                np.array([-1.0, 1.0 - r_2y]),
                np.array([r_2x - 1.0, 1.0]),
                np.array([r_1x - 1.0, 1.0 - r_2y]),
                r_2x,
                r_2y,
                -pi / 2,
                0.0,
                thdisc,
            ),
        )
    )
    coords = np.vstack(
        (
            coords,
            linear_pts(np.array([r_2x - 1.0, 1.0]), np.array([1.0 - r_2x, 1.0]), n_up),
        )
    )
    coords = np.vstack(
        (
            coords,
            elliptic_pts(
                np.array([1.0 - r_2x, 1.0]),
                np.array([1.0, 1.0 - r_2y]),
                np.array([1.0 - r_2x, 1.0 - r_2y]),
                r_2x,
                r_2y,
                0.0,
                pi / 2,
                thdisc,
            ),
        )
    )
    coords = np.vstack(
        (
            coords,
            linear_pts(
                np.array([1.0, 1.0 - r_2y]), np.array([1.0, r_1y - 1.0]), n_sides
            ),
        )
    )
    coords = np.vstack(
        (
            coords,
            elliptic_pts(
                np.array([1.0, r_1y - 1.0]),
                np.array([1.0 - r_1x, -1.0]),
                np.array([1.0 - r_1x, r_1y - 1.0]),
                r_1x,
                r_1y,
                pi / 2,
                pi,
                thdisc,
            ),
        )
    )
    coords = np.vstack(
        (
            coords,
            linear_pts(
                np.array([1.0 - r_1x, -1.0]),
                np.array([0.0, -1.0]),
                n_low,
                endpoint=True,
            ),
        )
    )
    return coords


def Mtostream(
    a, b
):  # define coordinate transformation matrix to convert to streamwise coordinate system
    Mtost = np.zeros((3, 3))
    Mtost[0, :] = np.array([cos(a) * cos(b), -cos(a) * sin(b), sin(a)], dtype="double")
    Mtost[1, :] = np.cross(np.array([0.0, 0.0, 1.0]), Mtost[0, :])
    Mtost[1, :] /= lg.norm(Mtost[1, :])
    Mtost[2, :] = np.cross(Mtost[0, :], Mtost[1, :])
    return Mtost


def Mstreamtouni(a, b):  # same but inverted
    return Mtostream(a, b).T


def PG_xmult(
    beta, a, b
):  # matrix to which multiply a point array to apply Prandtl-Glauert's correction
    return Mstreamtouni(a, b) @ np.diag(np.array([1.0, beta, beta])) @ Mtostream(a, b)


def PG_inv_xmult(beta, a, b):  # inverse of the matrix before
    return lg.inv(PG_xmult(beta, a, b))


def PG_vtouni(
    beta, a, b
):  # returns matrix converting incompressible PG calculated velocities
    # to compressible case
    return Mstreamtouni(a, b) @ np.diag(np.array([1.0, beta, beta])) @ Mtostream(a, b)


def PG_unitov(beta, a, b):  # inverse of function in prior entry
    return lg.inv(PG_vtouni(beta, a, b))
