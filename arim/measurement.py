# -*- coding: utf-8 -*-
"""
Tools and methods based on ultrasonic data measurements
"""
from collections import namedtuple
import logging

import numpy as np

from . import geometry as g

_IsometryOxy = namedtuple("_IsometryOxy", "z_o theta phi")

logger = logging.getLogger(__name__)


def find_probe_loc_from_frontwall(frame, couplant, tmin=None, tmax=None):
    """
    Registration process by detection of the frontwall, whose equation is
    assumed to be ``z = 0``. 

    This function:
    0. reset the position of the probe,
    1. detects the frontwall by looking for the extrama in the pulse-echo
    timetraces,
    2. infers the probe angle and standoff from these values
    (use a linear fit),
    3. move the probe.


    ..warning::

        This function changes the input frame.

    Notes: the user is responsible for checking the quality of the linear fit
    by having a look at the output array ``distance_to_surface``.

    The frontwall is assumed to be the line ``z=0.``
        2) move accordingly the probe.

    Parameters
    ----------
    frame : Frame
    couplant : Material
    tmin, tmax : float or None
        Inferior and superior time limits for finding the frontwall.


    Returns
    -------
    probe_standoff : float
        Distance (m) between the reference point of the probe (origin of its PCS) to the surface.
    probe_orientation : float
        Angle (rad) of the probe to the surface.
    time_to_surface : array
        Time between each element and the detected frontwall.

    See Also
    --------
    - :func:`detect_surface_from_extrema`
    - :func:`move_probe_over_flat_surface`
    """
    frame.probe.reset_position()

    # Detect frontwall:
    time_to_surface = detect_surface_from_extrema(frame, tmin, tmax)

    # Move probe:
    distance_to_surface = time_to_surface * couplant.longitudinal_vel / 2
    frame, iso = move_probe_over_flat_surface(
        frame, distance_to_surface, full_output=True
    )

    probe_standoff = iso.z_o
    probe_angle = iso.theta
    logger.info("Probe orientation: {:.2f}°".format(np.rad2deg(iso.theta)))
    logger.info("Probe standoff: {:.2f} mm".format(-1e3 * iso.z_o))

    return probe_standoff, probe_angle, time_to_surface


def move_probe_over_flat_surface(frame, distance_to_surface, full_output=False):
    """
    Translate and rotate the probe such as it is above the plane Oxy (plane z=0) at a given distance.


    The distances passed as arguments must corresponds to a flat surface.
    Perform a linear regression for robustness.

    Use only the distances corresponding to a pulse-echo timetraces. Other distances are discarded.

    **Warning:** this function modifies the points1 (you might want to make a copy before the call).

    Parameters
    ----------
    frame : Frame
    distance_to_surface : ndarray
        Distance between elements and the plane. One per timetrace. Only pulse echo data are used.
    full_output : boolean, optional
        If True, returns also ``(z_op, theta, phi)``. Default: False.

    Returns
    -------
    frame : Frame
        Returns the modified frame.
    isometry : _IsometryOxy
        Tuple ``(z_op, theta, phi)`` returned if ``full_output`` is True.

    Notes
    -----
    For *linear points1*: the points1 must be in the PCS, with all elements on the axis Ox.
    The array will be transformed via a rotation of the angle ``theta`` around the axis Oy
    and a translation, such as the point O(0,0,0) is
    transformed to ``O'(0,0,z_op)``. Remark: if for example you want the first element of the points1 on
    ``O'(0, 0, z_op)``, just apply beforehand a translation on the elements such as the first element is in O.

    *2D array* are not implemented.

    Implemented only: linear points1.

    The points1 must be in the PCS.

    Constraints:
    - The points1 will be in the half-space z < 0.
    - Linear points1: the points1 will be in the space y = 0.

    """
    # probe_type = frame.probe.metadata.get('probe_type', None)
    # if probe_type not in ('linear'):
    #    raise NotImplementedError("Only linear points1 are supported yet (given: ('{}').".format(probe_type))

    O = np.array([0.0, 0.0, 0.0])

    if not frame.probe.pcs.isclose(g.GCS):
        raise ValueError("This function requires that PCS and the GCS are the same.")

    # I/ keep only pulse echo timetraces
    # ---------------------------------

    # Consider only pulse-echo data:
    pulse_echo = frame.tx == frame.rx
    if sum(pulse_echo) < 2:
        raise ValueError("The frame must have at least 2 pulse echo timetraces.")

    distance_to_surface = distance_to_surface[pulse_echo]
    all_locations = frame.probe.locations_pcs
    numelements = frame.probe.numelements

    if np.any(distance_to_surface < 0):
        raise ValueError("Negative distance.")

    # II/ Check the element are all on Ox
    # -----------------------------------------

    on_Ox = np.isclose(np.abs(all_locations.x), all_locations.norm2())
    if not np.all(on_Ox):
        raise NotImplementedError(
            "This function works only with linear points1. The following elements are not on axis Ox: {}".format(
                np.arange(numelements)[on_Ox == False]
            )
        )

    # Let us call A the first element, and B the last element.
    locations_x = all_locations.x[frame.tx[pulse_echo]]
    xA = np.min(locations_x)
    xB = np.max(locations_x)
    assert not np.isclose(xA, xB)

    # III/ Linear regression: distance = p[1] * x + p[0]
    # -----------------------------------------

    # Linear regression to be more robust to small changes in distance_to_surface
    # Distance_to_surface = p[1] * x + p[0]
    p = tuple(reversed(np.polyfit(locations_x, distance_to_surface, 1)))

    # IV/ Get theta and z_0:
    # -----------------------------------------

    # Physically the distance to surface is the distance between each element and its orthogonal projection
    # on the plane Oxy. In other words, the displaced elements have for z coordinates plus-or-minus the distance to
    # surface. We used conventionally minus.

    # Cf 2016-03-10 Find flat surface.pdf:

    z_o = -p[0]
    with np.errstate(all="raise"):
        try:
            theta = np.arcsin(p[1])
        except Exception as e:
            raise RuntimeError(
                "There is no solution: it is likely one circle is in another."
            ) from e

    # V/ Move the points1:
    # -----------------------------------------

    rot = g.rotation_matrix_y(theta)

    frame.probe = frame.probe.rotate(rot)
    frame.probe = frame.probe.translate(np.array((0.0, 0.0, z_o)))

    if full_output:
        isometry = _IsometryOxy(z_o, theta, phi=np.nan)
        return frame, isometry
    else:
        return frame


def detect_surface_from_extrema(frame, tmin=None, tmax=None):
    """

    Parameters
    ----------
    frame
    tmin : float, optional
        Default: -inf
    tmax : float, optional
        Default: +inf

    Returns
    -------
    times_to_surface : ndarray of float
        For each timetrace, time at which occurs the maximum of the absolute signal in [tmin, tmax].

    """
    valid_times_ind = frame.time.window(tmin, tmax)
    valid_times = frame.time.samples[valid_times_ind]

    # Get the timetraces within the window [tmin, tmax].
    data = np.abs(frame.timetraces[:, valid_times_ind])

    # Find the times of the maximum amplitudes (assumed to be the surface):
    times_to_surface = valid_times[np.argmax(data, axis=1)]

    return times_to_surface
