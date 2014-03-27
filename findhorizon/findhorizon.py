"""
Find Black Hole apparent horizons in an axisymmetric spacetime.
===============================================================

Black holes are usually described by their *horizon* enclosing the singularity.
Locating any horizon in a general (typically numerically generated) spacetime
can be very hard - see Thornburg's review [1]_ for details. Here we restrict to
the simpler problem of a specific type of axisymmetric spacetime, where the
equation to solve reduces to a boundary value problem.

Strictly this module constructs *trapped surfaces*, which are surfaces where
null geodesics (light rays) are ingoing. The apparent horizon is the
outermost trapped surface.

Notes
-----

The technical restrictions on the spacetime are

1. axisymmetric, so the singularities are on the z axis;
2. singularities have Brill-Lindquist type;
3. spacetime is conformally flat;
4. coordinates are chosen to obey maximal slicing with no shift;
5. data is time symmetric.

References
----------

.. [1] J. Thornburg, "Event and Apparent Horizon Finders for 3+1 Numerical
    Relativity", Living Reviews in Relativity 10 (3) 2007.
    http://dx.doi.org/10.12942/lrr-2007-3.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.optimize import brentq, root, newton


class Spacetime:

    """
    Define an axisymmetric spacetime.

    For an axisymmetric, vacuum spacetime with Brill-Lindquist singularities
    the only parameters that matter is the locations of the singularities
    (i.e. their z-location) and their bare masses.

    Parameters
    ----------

    z_positions : list of float
        The location of the singularities on the z-axis.
    masses : list of float
        The bare masses of the singularities.
    reflection_symmetry : bool, optional
        Is the spacetime symmetric across the x-axis.

    See also
    --------

    TrappedSurface : class defining the trapped surfaces on a spacetime.

    Examples
    --------

    >>> schwarzschild = Spacetime([0.0], [1.0], True)

    This defines standard Schwarzschild spacetime with unit mass.

    >>> binary = Spacetime([-0.75, 0.75], [1.0, 1.1])

    This defines two black holes, with the locations mirrored but different
    masses.
    """

    def __init__(self, z_positions, masses, reflection_symmetric=False):
        """
        Initialize the spacetime given the location and masses of the
        singularities.
        """

        self.reflection_symmetric = reflection_symmetric

        if reflection_symmetric:
            # Enforce reflection symmetry, in case only positive terms
            # passed in.
            z_plus, z_index = np.unique(np.abs(z_positions),
                                        unique_index=True)
            if (z_plus[0] < np.spacing(1)):  # One singularity at origin
                z_symm = np.zeros((2*len(z_plus)-1, 1))
                masses_symm = np.zeros_like(z_symm)
                z_symm[0] = 0.0
                z_symm[1:len(z_plus)] = z_plus[1:]
                z_symm[len(z_plus):] = -z_plus[1:]
                masses_symm[0] = masses[z_index[0]]
                masses_symm[1:len(z_plus)] = masses[z_index[1:]]
                masses_symm[len(z_plus):] = masses[z_index[1:]]
            else:  # No singularities at origin
                z_symm = np.zeros((2*len(z_plus), 1))
                masses_symm = np.zeros_like(z_symm)
                z_symm[len(z_plus)] = z_plus
                z_symm[len(z_plus):] = -z_plus
                masses_symm[1:len(z_plus)] = masses[z_index]
                masses_symm[len(z_plus):] = masses[z_index]
            z_positions = z_symm
            masses = masses_symm

        self.z_positions = np.array(z_positions)
        self.masses = np.array(masses)
        self.N = len(z_positions)


class TrappedSurface:

    r"""
    Store any trapped surface, centred on a particular point.

    The trapped surface is defined in polar coordinates centred on a point
    on the z-axis; the z-axis is :math:`\theta` = 0 or :math:`\theta` =
    :math:`\pi`.

    Parameters
    ----------

    spacetime : Spacetime
        The spacetime on which the trapped surface lives.
    z_centre : float
        The z-coordinate about which the polar coordinate system describing
        the trapped surface is defined.

    See also
    --------

    Spacetime : class defining the spacetime.

    Notes
    -----

    With the restricted spacetime considered here, a trapped surface
    :math:`h(\theta)` satisfies a boundary value problem with the
    boundary conditions :math:`h'(\theta = 0) = 0 = h'(\theta = \pi)`.
    If the spacetime is reflection symmetric about the x-axis then the
    boundary condition :math:`h'(\theta = \pi / 2) = 0` can be used
    and the domain restricted to :math:`0 \le \theta \le \pi / 2`.

    The shooting method is used here. In the reflection symmetric case
    the algorithm needs a guess for the initial horizon radius,
    :math:`h(\theta = 0)`, and a single condition is enforced at
    :math:`\pi / 2` to match to the boundary condition there.

    In the general case we guess the horizon radius at two points,
    :math:`h(\theta = 0)` and :math:`h(\theta = \pi)` and continuity
    of both :math:`h` *and* :math:`h'` are enforced at the matching point
    :math:`\pi / 2`. The reason for this is a weak coordinate singularity
    on the axis at :math:`\theta = 0, \pi` which makes it difficult to
    integrate *to* these points, but possible to integrate *away* from them.

    Examples
    --------

    >>> schwarzschild = Spacetime([0.0], [1.0], True)
    >>> ts1 = TrappedSurface(schwarzschild)
    >>> ts1.find_r0([0.49, 0.51])
    >>> ts1.solve_given_r0()
    >>> print(round(ts1.r0[0], 9))
    0.5

    This example first constructs the Schwarzschild spacetime which, in this
    coordinate system, has the horizon with radius 0.5. The trapped surface
    is set up, the location of the trapped surface at :math:`\theta = 0` is
    found, which is (to the solver accuracy) at 0.5.
    """

    def __init__(self, spacetime, z_centre=0.0):
        """
        Initialize a horizon centred on a particular point.
        """

        self.z_centre = z_centre
        self.spacetime = spacetime

    def expansion(self, theta, H):
        """
        Compute the expansion for the given spacetime at a fixed point.

        This function gives the differential equation defining the
        boundary value problem.

        Parameters
        ----------

        theta : float
            The angular location at this point.
        H : list of float
            A vector of :math:`(h, h')`.
        """

        h = H[0]
        dhdtheta = H[1]

        z_i = self.spacetime.z_positions
        m_i = self.spacetime.masses

        distance_i = np.zeros_like(z_i)
        z0_minus_zi = np.zeros_like(z_i)
        for i in range(len(z_i)):
            z0_minus_zi[i] = self.z_centre - z_i[i]
            distance_i[i] = np.sqrt(h ** 2 +
                                    2.0 * z0_minus_zi[i] * h * np.cos(theta)
                                    + z0_minus_zi[i] ** 2)

        C2 = 1.0 / (1.0 + (dhdtheta / h) ** 2)
        if (abs(theta) < 1e-16) or (abs(theta - np.pi) < 1e-16):
            cot_theta_dhdtheta_C2 = 0.0
        else:
            cot_theta_dhdtheta_C2 = dhdtheta / (np.tan(theta) * C2)

        psi = 1.0
        dpsi_dr = 0.0
        dpsi_dtheta = 0.0
        for i in range(len(m_i)):
            psi += 0.5 * m_i[i] / distance_i[i]
            dpsi_dr -= 0.5 * m_i[i] * (h + z0_minus_zi[i] * np.cos(theta)) /\
                distance_i[i] ** 3
            dpsi_dtheta += 0.5 * m_i[i] * h * z0_minus_zi[i] * np.sin(theta) /\
                distance_i[i] ** 3

        dHdtheta = np.zeros_like(H)
        dHdtheta[0] = dhdtheta
        dHdtheta[1] = 2.0 * h - cot_theta_dhdtheta_C2 + \
            4.0 * h ** 2 / (psi * C2) * \
            (dpsi_dr - dpsi_dtheta * dhdtheta / h ** 2) + \
            3.0 * dhdtheta ** 2 / h

        return dHdtheta

    # Define the shooting function if using matching (0 <= theta <= pi)
    def shooting_function_full(self, r0):
        r"""
        The function used in the shooting algorithm.

        This is the full algorithm from integrating over
        :math:`0 \le \theta \le \pi`. The difference between the
        solution and its derivative at the matching point is the
        error to be minimized.

        Parameters
        ----------

        r0 : list of float
            Initial guess for the horizon radius, as outlined above.

        Returns
        -------

        list of float
            The error at the matching point.
        """
        # First half of the horizon
        H0 = np.array([r0[0], 0.0])
        solver1 = ode(self.expansion)
        solver1.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
        solver1.set_initial_value(H0, 0.0)
        solver1.integrate(np.pi / 2.0)
        # Second half of the horizon
        H0 = np.array([r0[1], 0.0])
        solver2 = ode(self.expansion)
        solver2.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
        solver2.set_initial_value(H0, np.pi)
        solver2.integrate(np.pi / 2.0)

        return solver1.y - solver2.y

    # Define the shooting function if symmetric (0 <= theta <= pi/2)
    def shooting_function(self, r0):
        r"""
        The function used in the shooting algorithm.

        This is the symmetric algorithm from integrating over
        :math:`0 \le \theta \le \pi / 2`. The difference between the
        derivative at the end point and the boundary condition is the
        error to be minimized.

        Parameters
        ----------

        r0 : float
            Initial guess for the horizon radius, as outlined above.

        Returns
        -------

        float
            The error at the end point.
        """

        H0 = np.array([r0, 0.0])
        solver1 = ode(self.expansion)
        solver1.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
        solver1.set_initial_value(H0, 0.0)
        solver1.integrate(np.pi / 2.0)

        return solver1.y[1]

    def find_r0(self, input_guess, full_horizon=False):
        r"""
        Given some initial guess, find the correct starting location
        for the trapped surface using shooting.

        This finds the horizon radius at :math:`\theta = 0` which,
        together with the differential equation, specifies the trapped
        surface location.

        Parameters
        ----------

        input_guess : list of float
            Two positive reals defining the guess for the initial radius.

            Note that the meaning is different depending on whether this
            is a "full" horizon or not. For a full horizon the numbers
            correspond to the guesses at :math:`\theta = 0, \pi`
            respectively. In the symmetric case where only one guess is
            needed the vector defines the interval within which a *unique*
            root must lie.

        full_horizon : bool, optional
            If the general algorithm is needed (ie, the domain should be
            :math:`0 \le \theta \le \pi` instead of
            :math:`0 \le \theta \le \pi / 2`).

            This parameter is independent of the symmetry of the spacetime.
            If the spacetime is not symmetric this parameter will be
            ignored and the general algorithm always used. If the spacetime
            is symmetric it may still be necessary to use the general
            algorithm: for example, for two singularities it is possible to
            find a trapped surface surrounding just one singularity.
        """

        # Now find the horizon given the input guess
        self.r0 = []
        if (full_horizon or
                not self.spacetime.reflection_symmetric or
                abs(self.z_centre) > 1.e-15):
            sol = root(self.shooting_function_full, input_guess, tol=1.e-12)
            self.r0 = sol.x
        else:
#            sol = brentq(self.shooting_function, input_guess[0],
#                         input_guess[1])
            sol = newton(self.shooting_function, input_guess[1])
            self.r0 = [sol]

    def solve_given_r0(self, full_horizon=False):
        r"""
        Given the correct value for the initial radius, find the horizon.

        This function does not find the correct radius for the trapped
        surface, but solves (in polar coordinates) for the complete
        surface location given the correct initial guess.

        Parameters
        ----------

        full_horizon : bool, optional
            If the general algorithm is needed (ie, the domain should be
            :math:`0 \le \theta \le \pi` instead of
            :math:`0 \le \theta \le \pi / 2`).

            This parameter is independent of the symmetry of the spacetime.
            If the spacetime is not symmetric this parameter will be
            ignored and the general algorithm always used. If the spacetime
            is symmetric it may still be necessary to use the general
            algorithm: for example, for two singularities it is possible to
            find a trapped surface surrounding just one singularity.

        See also
        --------

        find_r0 : finds the correct initial radius.
        """

        dtheta = np.pi / 100.0

        if (full_horizon or not self.spacetime.reflection_symmetric):
            # The solution needs computing for 0 <= theta <= pi
            # First half of the horizon
            theta1 = []
            H1 = []
            H0 = np.array([self.r0[0], 0.0])
            solver1 = ode(self.expansion)
            solver1.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
            solver1.set_initial_value(H0, 0.0)
            theta1.append(0.0)
            H1.append(H0)
            while solver1.successful() and solver1.t < np.pi / 2.0:
                solver1.integrate(solver1.t + dtheta)
                H1.append(solver1.y)
                theta1.append(solver1.t)
            # Second half of the horizon
            theta2 = []
            H2 = []
            H0 = np.array([self.r0[1], 0.0])
            solver2 = ode(self.expansion)
            solver2.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
            solver2.set_initial_value(H0, np.pi)
            theta2.append(np.pi)
            H2.append(H0)
            while solver2.successful() and solver2.t >= np.pi / 2.0 + 1e-12:
                solver2.integrate(solver2.t - dtheta)
                H2.append(solver2.y)
                theta2.append(solver2.t)

            H = np.vstack((np.array(H1), np.flipud(np.array(H2))))
            theta = np.hstack((np.array(theta1),
                               np.flipud(np.array(theta2))))

        else:  # The solution needs computing for 0 <= theta <= pi / 2
            theta1 = []
            H1 = []
            H0 = np.array([self.r0[0], 0.0])
            solver1 = ode(self.expansion)
            solver1.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
            solver1.set_initial_value(H0, 0.0)
            theta1.append(0.0)
            H1.append(H0)
            while solver1.successful() and solver1.t < np.pi / 2.0:
                solver1.integrate(solver1.t + dtheta)
                H1.append(solver1.y)
                theta1.append(solver1.t)

            H = np.vstack((np.array(H1), np.flipud(H1)))
            theta = np.hstack((theta1,
                               np.flipud(np.pi - np.array(theta1))))

        # We now have the solution for 0 <= theta <= pi;
        # fill the remaining angles
        self.H = np.vstack((H, np.flipud(H)))
        self.theta = np.hstack((theta, theta + np.pi))

        return None

    def convert_to_cartesian(self):
        """
        When the solution is known in r, theta coordinates, compute
        the locations in cartesian coordinates (2 and 3d).

        This function assumes that the trapped surface has been located and
        solved for.

        See also
        --------

        solve_given_r0 : find the trapped surface location in polar
                         coordinates.
        """

        self.x = self.H[:, 0] * np.sin(self.theta)
        self.z = self.z_centre + self.H[:, 0] * np.cos(self.theta)

        phi = np.linspace(0.0, 2.0 * np.pi, 20)
        self.X = np.zeros((len(self.theta), len(phi)))
        self.Y = np.zeros_like(self.X)
        self.Z = np.zeros_like(self.X)
        for t in range(len(self.theta)):
            for p in range(len(phi)):
                self.X[t, p] = self.H[t, 0] * np.sin(self.theta[t]) * \
                    np.cos(phi[p])
                self.Y[t, p] = self.H[t, 0] * np.sin(self.theta[t]) * \
                    np.sin(phi[p])
                self.Z[t, p] = self.z_centre + \
                    self.H[t, 0] * np.cos(self.theta[t])
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2 + self.Z ** 2)

        return None

    def plot_2d(self, ax):
        """
        Given a matplotlib axis, plot the trapped surface.

        Plots the surface in the x-z plane, together with the location of
        the singularities: marker style is used to indicate the mass of
        the singularity (will fail badly for masses significantly larger
        than 1).

        Parameters
        ----------

        ax : axis object
            Matplotlib axis on which to do the plot.
        """

        ax.plot(self.x, self.z, 'b-')
        for z, m in zip(self.spacetime.z_positions, self.spacetime.masses):
            ax.plot(0.0, z,
                    'kx', markersize=12, markeredgewidth=1 + int(round(m)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
        ax.axis('equal')


def find_horizon_binary_symmetric(z=0.5, mass=1.0):
    r"""
    Utility function to find horizons for reflection symmetric case.

    This returns the horizon for a spacetime with precisely two singularities
    of identical mass located at :math:`\pm z`.

    Notes
    -----

    The initial guess for the horizon location is based on fitting a cubic
    to the results constructed for :math:`0 \le z \le 0.75` for the unit
    mass case. The radius should scale with the mass. For larger separations
    we should not expect a common horizon.

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass : float, optional
        The mass of the singularities.

    Returns
    -------

    ts : TrappedSurface
        Only returns the single surface found, expected to be the common
        horizon.
    """

    st = Spacetime([-z, z], [mass, mass], True)
    ts = TrappedSurface(st, 0.0)
    # An empirical formula for the required initial guess
    # (ie the value of r0, or h, at theta = 0)
    r0_empirical = mass * (1.0 - 0.0383 * z + 0.945 * z ** 2 - 0.522 * z ** 3)
    # This empirical formula works for the inner horizon if
    # 0.65 < z < 0.72 or so. There is an inner horizon findable
    # down to about 0.47, but the initial guess is very sensitive
    # r0_empirical = mass * (0.204 - 1.6422*z - 0.771*z**2 + 0.5*z**3)
    initial_guess = [0.99 * r0_empirical, 1.01 * r0_empirical]
    try:
        ts.find_r0(initial_guess)
    except ValueError:
        r0 = np.linspace(0.95 * r0_empirical, 1.05 * r0_empirical)
        phi = np.zeros_like(r0)
        for i in range(len(r0)):
            phi[i] = ts.shooting_function(r0[i])
        initial_guess = [r0[np.argmin(phi)], r0[-1]]
        ts.find_r0(initial_guess)
    ts.solve_given_r0()
    ts.convert_to_cartesian()
    return ts


def find_inner_outer_horizon_binary_symmetric(z=0.5, mass=1.0):
    r"""
    Utility function to find horizons for reflection symmetric case.

    This returns two trapped surface for a spacetime with precisely
    two singularities of identical mass located at :math:`\pm z`. The outer
    surface is the apparent horizon; the inner surface is just a trapped
    surface.

    Notes
    -----

    The initial guess for the horizon location is based on fitting a cubic
    to the results constructed for :math:`0 \le z \le 0.75` for the unit
    mass case. The radius should scale with the mass. For larger separations
    we should not expect a common horizon. The inner horizon is based on
    a similar fit but in the narrower range :math:`0.6 \le z \le 0.7` and
    so it is very likely that this function will fail for :math:`z < 0.42`.

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass : float, optional
        The mass of the singularities.

    Returns
    -------

    ts1, ts2 : TrappedSurface
        Returns the trapped surfaces found.
    """

    st = Spacetime([-z, z], [mass, mass], True)
    ts1 = TrappedSurface(st, 0.0)
    ts2 = TrappedSurface(st, 0.0)
    # An empirical formula for the required initial guess
    # (ie the value of r0, or h, at theta = 0)
    r0_empirical = mass * (1.0 - 0.0383 * z + 0.945 * z ** 2 - 0.522 * z ** 3)
    initial_guess = [0.99 * r0_empirical, 1.01 * r0_empirical]
    try:
        ts1.find_r0(initial_guess)
    except ValueError:
        r0 = np.linspace(0.95 * r0_empirical, 1.05 * r0_empirical)
        phi = np.zeros_like(r0)
        for i in range(len(r0)):
            phi[i] = ts1.shooting_function(r0[i])
        initial_guess = [r0[np.argmin(phi)], r0[-1]]
        ts1.find_r0(initial_guess)
    ts1.solve_given_r0()
    ts1.convert_to_cartesian()
    # This empirical formula works for the inner horizon if
    # 0.42 < z < 0.765 or so. It looks likely that the inner horizon
    # persists below 0.42, but eventually it will fail.
    # r0_empirical = mass * (-0.357+4.39*z-5.263*z**2+2.953*z**3)
    r0_empirical = mass * \
        (1.866 - 10.213 * z + 30.744 * z **
         2 - 36.513 * z ** 3 + 16.21 * z ** 4)
    initial_guess = [0.99 * r0_empirical, 1.01 * r0_empirical]
    try:
        ts2.find_r0(initial_guess)
    except ValueError:
        r0 = np.linspace(0.95 * r0_empirical, 1.05 * r0_empirical)
        phi = np.zeros_like(r0)
        for i in range(len(r0)):
            phi[i] = ts2.shooting_function(r0[i])
        initial_guess = [r0[np.argmin(phi)], r0[-1]]
        ts2.find_r0(initial_guess)
    ts2.solve_given_r0()
    ts2.convert_to_cartesian()
    return ts1, ts2


def find_individual_horizon_binary_symmetric(z=0.5, mass=1.0):
    r"""
    Utility function to find horizons for reflection symmetric case.

    This returns two trapped surface for a spacetime with precisely
    two singularities of identical mass located at :math:`\pm z`. These
    should be trapped surfaces about only one singularity.

    Notes
    -----

    The initial guess for the horizon location is based on fitting a cubic
    to the results constructed for :math:`0.45 \le z \le 0.75` for the unit
    mass case. The radius should scale with the mass. For smaller separations
    we should not expect individual horizons.

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass : float, optional
        The mass of the singularities.

    Returns
    -------

    ts1, ts2 : TrappedSurface
        Returns the trapped surfaces found.
    """

    st = Spacetime([-z, z], [mass, mass], True)
    ts1 = TrappedSurface(st, -z)
    ts2 = TrappedSurface(st,  z)
    # An empirical formula for the required initial guess
    # (ie the value of r0, or h, at theta = 0)
    r0_close = mass * \
        (0.002 + 1.027 * z - 1.235 * z ** 2 + 0.816 * z ** 3 - 0.228 * z ** 4)
    r0_far = mass * \
        (0.215 + 0.557 * z - 0.727 * z ** 2 + 0.531 * z ** 3 - 0.163 * z ** 4)
    initial_guess = [r0_close, r0_far]
    ts1.find_r0(initial_guess, True)
    ts1.solve_given_r0(True)
    ts1.convert_to_cartesian()
    initial_guess = [r0_far, r0_close]
    ts2.find_r0(initial_guess, True)
    ts2.solve_given_r0(True)
    ts2.convert_to_cartesian()
    return ts1, ts2


def find_horizon_binary(z=0.5, mass1=1.0, mass2=1.0):
    r"""
    Utility function to find horizons for the general case.

    This returns the horizon for a spacetime with precisely two singularities
    of mass [mass1, mass2] located at :math:`\pm z`. That is, we work in the
    frame where the location of the horizons is symmetric.

    Notes
    -----

    The initial guess for the horizon location is based on fitting a cubic
    to the results constructed for :math:`0 \le z \le 0.75` for the unit
    mass case. The radius should scale with the mass. For larger separations
    we should not expect a common horizon.

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass : float, optional
        The mass of the singularities.

    Returns
    -------

    ts : TrappedSurface
        Only returns the single surface found, expected to be the common
        horizon.
    """

    st = Spacetime([-z, z], [mass1, mass2])
    ts = TrappedSurface(st, 0.0)
    # An empirical formula for the required initial guess
    # (ie the value of r0, or h, at theta = 0)
    # This really is just a guess based on the symmetric case.
    zom = 2.0 * z / (mass1 + mass2)
    r0_empirical = (1.0 - 0.0383 * zom + 0.945 * zom ** 2 -
                    0.522 * zom ** 3) * \
        (mass1 + mass2) / 2.0
    r0_empirical = max(r0_empirical, z + 0.5 * max(mass1, mass2))
    initial_guess = [r0_empirical, r0_empirical]
    ts.find_r0(initial_guess, True)
    ts.solve_given_r0()
    ts.convert_to_cartesian()
    return ts


def plot_horizon_3d(tss):
    """
    Plot a list of horizons.

    Parameters
    ----------

    tss : list of TrappedSurface
        All the trapped surfaces to visualize.
    """
    from mayavi import mlab
    cmaps = ['bone', 'jet', 'hot', 'cool', 'spring', 'summer', 'winter']
    assert len(cmaps) > len(tss)
    extents = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for ts, cm in zip(tss, cmaps):
        mlab.mesh(ts.X, ts.Y, ts.Z, colormap=cm, opacity=0.4)
        extents[0] = min(extents[0], np.min(ts.X))
        extents[1] = max(extents[1], np.max(ts.X))
        extents[2] = min(extents[2], np.min(ts.Y))
        extents[3] = max(extents[3], np.max(ts.Y))
        extents[4] = min(extents[4], np.min(ts.Z))
        extents[5] = max(extents[5], np.max(ts.Z))
    mlab.axes(extent=extents)
    mlab.outline(extent=extents)
    mlab.show()


def solve_plot_symmetric(z=0.5, mass=1.0):
    r"""
    Utility function to find horizons for reflection symmetric case.

    This returns the horizon for a spacetime with precisely two singularities
    of identical mass located at :math:`\pm z`.

    Notes
    -----

    The initial guess for the horizon location is based on fitting a cubic
    to the results constructed for :math:`0 \le z \le 0.75` for the unit
    mass case. The radius should scale with the mass. For larger separations
    we should not expect a common horizon.

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass : float, optional
        The mass of the singularities.
    """

    ts = find_horizon_binary_symmetric(z, mass)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ts.plot_2d(ax)
    plt.show()

    return fig


def solve_plot_symmetric_3d(z=0.5, mass=1.0):
    r"""
    Utility function to plot horizon in 3d for reflection symmetric case.

    This returns the horizon for a spacetime with precisely two singularities
    of identical mass located at :math:`\pm z`.

    Notes
    -----

    The initial guess for the horizon location is based on fitting a cubic
    to the results constructed for :math:`0 \le z \le 0.75` for the unit
    mass case. The radius should scale with the mass. For larger separations
    we should not expect a common horizon.

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass : float, optional
        The mass of the singularities.
    """

    ts = find_horizon_binary_symmetric(z, mass)
    plot_horizon_3d([ts])


def solve_plot_binary(z=0.5, mass1=1.0, mass2=1.0):
    r"""
    Utility function to find horizons for general case.

    This returns the horizon for a spacetime with precisely two singularities
    of different mass located at :math:`\pm z`.

    Notes
    -----

    The initial guess is not easily determined, so performance is poor and
    the algorithm not robust

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass1, mass2 : float, optional
        The mass of the singularities.
    """

    ts = find_horizon_binary(z, mass1, mass2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ts.plot_2d(ax)
    plt.show()

    return fig


def solve_plot_binary_3d(z=0.5, mass1=1.0, mass2=1.0):
    r"""
    Utility function to plot horizons in 3d for general case.

    This returns the horizon for a spacetime with precisely two singularities
    of different mass located at :math:`\pm z`.

    Notes
    -----

    The initial guess is not easily determined, so performance is poor and
    the algorithm not robust

    Parameters
    ----------

    z : float, optional
        The distance from the origin of the singularities (ie the two
        singularities are located at [-z, +z]).
    mass1, mass2 : float, optional
        The mass of the singularities.
    """

    ts = find_horizon_binary(z, mass1, mass2)
    plot_horizon_3d([ts])


if __name__ == "__main__":
#    st = Spacetime([-0.5, 0.5], [1.0, 1.0])
#    ts = TrappedSurface(st)
#    ts.find_r0([1.0, 1.0])
#    ts.solve_given_r0()
    import doctest
    doctest.testmod()
