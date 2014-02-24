import numpy as np
from scipy.integrate import ode
from scipy.optimize import brentq, minimize, root


def expansion(theta, H, params):
    """
    Compute the expansion at a single point.
    """

    h = H[0]
    dhdtheta = H[1]

    singularities = params[0]

    m_i = singularities['masses']
    z_i = singularities['z_positions']

    y = np.array([h * np.sin(theta), h * np.cos(theta)])
    distance_i = np.zeros_like(z_i)
    for i in range(len(z_i)):
        distance_i[i] = np.linalg.norm(y - np.array([0, z_i[i]]), 2)

    C = 1.0 / np.sqrt(1.0 + (dhdtheta / h) ** 2)
    if (abs(theta) < 1e-16) or (abs(theta - np.pi) < 1e-16):
        cot_theta_dhdtheta_C2 = 0.0
    else:
        cot_theta_dhdtheta_C2 = dhdtheta / (np.tan(theta) * C ** 2)

    psi = 1.0
    dpsi_dr = 0.0
    dpsi_dtheta = 0.0
    for i in range(len(m_i)):
        psi += 0.5 * m_i[i] / distance_i[i]
        dpsi_dr += 0.5 * m_i[i] * (z_i[i] * np.cos(theta) - h) / \
            distance_i[i] ** 3
        dpsi_dtheta += 0.5 * m_i[i] * h * (-z_i[i] * np.sin(theta)) / \
            distance_i[i] ** 3

    dHdtheta = np.zeros_like(H)
    dHdtheta[0] = dhdtheta
    dHdtheta[1] = 2.0 * h - cot_theta_dhdtheta_C2 + \
        4.0 * h ** 2 / (psi * C ** 2) * \
        (dpsi_dr - dpsi_dtheta * dhdtheta / h ** 2) + \
        3.0 * dhdtheta ** 2 / h

    return dHdtheta


def shooting_function(h0, singularities, theta0=0.0, theta_end=np.pi / 2.0):
    """
    Find the horizon using shooting. Returns the error at the endpoint.
    """

    theta_out = []
    H = []
    H_0 = np.array([h0, 0.0])
    solver = ode(expansion)
    solver.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
    solver.set_initial_value(H_0, 0.0)
    solver.set_f_params(singularities)
    theta_out.append(theta0)
    H.append(H_0)

    dtheta = (theta_end - theta0) / 10.0

    while solver.successful() and solver.t < theta_end:
        solver.integrate(solver.t + dtheta, step=1)

        H.append(solver.y)
        theta_out.append(solver.t)

    return solver.y[1]


def shooting_function_full(h0, singularities,
                           theta0=0.0, theta_end=np.pi,
                           theta_match=np.pi / 2.0):
    """
    Find the horizon using shooting. Returns the matching error.
    """

    H_0 = np.array([h0[0], 0.0])
    solver1 = ode(expansion)
    solver1.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
    solver1.set_initial_value(H_0, 0.0)
    solver1.set_f_params(singularities)

    dtheta = (theta_match - theta0) / 10.0

    while solver1.successful() and solver1.t < theta_match:
        solver1.integrate(solver1.t + dtheta, step=1)

    H_0 = np.array([h0[1], 0.0])
    solver2 = ode(expansion)
    solver2.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
    solver2.set_initial_value(H_0, theta_end)
    solver2.set_f_params(singularities)

    dtheta = (theta_match - theta_end) / 10.0

    while solver2.successful() and solver2.t > theta_match:
        solver2.integrate(solver2.t + dtheta, step=1)

    return solver1.y - solver2.y


def FindHorizonFull(input_guess, singularities, options=None):
    """
    Find horizons given N singularities (positions and locations),
    assuming 
    1) conformal flatness
    2) axisymmetry
    The input_guess is the radius on the axis.
    """

    # Find the horizon

    sol = root(shooting_function_full, input_guess, args=([singularities],))
    h0 = sol.x

    theta1 = []
    H1 = []
    H_0 = np.array([h0[0], 0.0])
    solver1 = ode(expansion)
    solver1.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
    solver1.set_initial_value(H_0, 0.0)
    solver1.set_f_params([singularities])
    theta1.append(0.0)
    H1.append(H_0)

    dtheta = np.pi / 100.0

    while solver1.successful() and solver1.t < np.pi / 2.0:
        solver1.integrate(solver1.t + dtheta, step=1)

        H1.append(solver1.y)
        theta1.append(solver1.t)

    theta2 = []
    H2 = []
    H_0 = np.array([h0[1], 0.0])
    solver2 = ode(expansion)
    solver2.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
    solver2.set_initial_value(H_0, np.pi)
    solver2.set_f_params([singularities])
    theta2.append(np.pi)
    H2.append(H_0)

    dtheta = -np.pi / 100.0

    while solver2.successful() and solver2.t > np.pi / 2.0:
        solver2.integrate(solver2.t + dtheta, step=1)

        H2.append(solver2.y)
        theta2.append(solver2.t)

    H = np.vstack((np.array(H1), np.flipud(np.array(H2))))
    theta_out = np.hstack((np.array(theta1), np.flipud(np.array(theta2))))

    return theta_out, H


def FindHorizonSymmetric(input_guess, singularities, options=None):
    """
    Find horizons given N singularities (positions and locations),
    assuming 
    1) conformal flatness
    2) axisymmetry
    The input_guess is the radius on the axis at both ends.
    """

    # Find the horizon

    h0 = brentq(shooting_function, input_guess[
                0], input_guess[1], args=([singularities],))

    theta_out = []
    H = []
    H_0 = np.array([h0, 0.0])
    solver = ode(expansion)
    solver.set_integrator("dopri5", atol=1.e-8, rtol=1.e-6)
    solver.set_initial_value(H_0, 0.0)
    solver.set_f_params([singularities])
    theta_out.append(0.0)
    H.append(H_0)

    dtheta = np.pi / 100.0

    while solver.successful() and solver.t < np.pi / 2.0:
        solver.integrate(solver.t + dtheta, step=1)

        H.append(solver.y)
        theta_out.append(solver.t)

    H = np.array(H)
    theta_out = np.array(theta_out)

    return theta_out, H


def FindHorizonBinarySymmetric(z=0.5, mass=1.0):
    """
    Find the common horizon of the axisymmetric system with reflection symmetry
    with singularities at \pm z with equal bare mass.
    """

    singularities = {"masses": np.array([mass, mass]),
                     "z_positions": np.array([-z, z])}
    # An empirical formula for the required initial guess
    # (ie the value of r0, or h, at theta = 0)
    r0_empirical = mass * (0.976 + 0.2 * z + 0.27 * z ** 2)
    initial_guess = [0.95 * r0_empirical, 1.05 * r0_empirical]
    try:
        theta, H = FindHorizonSymmetric(initial_guess, singularities)
    except ValueError:
        r0 = np.linspace(initial_guess[0], initial_guess[1])
        phi = np.zeros_like(r0)
        for i in range(len(r0)):
            phi[i] = shooting_function(r0[i], [singularities])
        initial_guess = [r0[np.argmin(phi)], r0[-1]]
        theta, H = FindHorizonSymmetric(initial_guess, singularities)
    return theta, H


def PlotHorizon2d(ax, theta, H, z_positions, masses):
    """
    Given all theta and H values, plot the x-z plane cut.
    """
    ax.plot(H[:, 0] * np.sin(theta), H[:, 0] * np.cos(theta), 'b-')
    for z, m in zip(z_positions, masses):
        ax.plot(0.0, z,
                'kx', markersize=12, markeredgewidth=1 + int(round(m)))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.axis('equal')


def PlotHorizon3d(ax, theta, H):
    """
    Given all theta and H values, plot the full 3d picture.
    """
    from matplotlib import cm
    phi = np.linspace(0.0, 2.0 * np.pi, 20)
    X = np.zeros((len(theta), len(phi)))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for t in range(len(theta)):
        for p in range(len(phi)):
            X[t, p] = H[t, 0] * np.sin(theta[t]) * np.cos(phi[p])
            Y[t, p] = H[t, 0] * np.sin(theta[t]) * np.sin(phi[p])
            Z[t, p] = H[t, 0] * np.cos(theta[t])
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1, linewidth=0,
                           facecolors=cm.jet(R), antialiased=False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

def PlotHorizonInteractive3d(ax, theta, H):
    """
    Given all theta and H values, plot the full 3d picture.
    """
    from mayavi import mlab
    phi = np.linspace(0.0, 2.0 * np.pi, 20)
    X = np.zeros((len(theta), len(phi)))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for t in range(len(theta)):
        for p in range(len(phi)):
            X[t, p] = H[t, 0] * np.sin(theta[t]) * np.cos(phi[p])
            Y[t, p] = H[t, 0] * np.sin(theta[t]) * np.sin(phi[p])
            Z[t, p] = H[t, 0] * np.cos(theta[t])
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    s = mlab.mesh(X, Y, Z,
                  opacity = 0.4)
    mlab.axes()
    mlab.outline()
    mlab.show()

def PlotHorizonSymmetric(theta, H, z=0.5, mass=1.0, elev=None, azim=None):
    """
    Take the output for a symmetric horizon 
    (ie, theta \in [0, pi/2]) and plot it using 2 and 3d figures.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121)
    half_theta = np.hstack((theta, np.flipud(np.pi - theta)))
    half_H = np.vstack((H, np.flipud(H)))
    all_theta = np.hstack((half_theta, half_theta + np.pi))
    all_H = np.vstack((half_H, half_H))
    PlotHorizon2d(
        ax1, all_theta, all_H, np.array([-z, z]), np.array([mass, mass]))
    ax1.set_title("Singularities at $z = \pm {}$".format(z))
    ax2 = fig.add_subplot(122, projection='3d')
    PlotHorizon3d(ax2, all_theta, all_H)
    if elev is not None:
        ax2.elev = elev
    if azim is not None:
        ax2.azim = azim
    ax2.set_title("Singularities at $z = \pm {}$".format(z))
    plt.show()


def SolvePlotSymmetric(z=0.5, mass=1.0, elev=None, azim=None):

    theta, H = FindHorizonBinarySymmetric(z, mass)
    PlotHorizonSymmetric(theta, H, z, mass, elev, azim)

def SolvePlotASymmetric(z = [-0.5, 0.5], mass = [1.0, 1.0], 
                        elev = None, azim = None):
    """
    Calculate and plot the results even in the full case.
    """
    
    singularities = {"masses": np.array(mass),
                     "z_positions": np.array(z)}
    r0_empirical = np.sum(mass) * (0.97 + 0.2 * np.max(np.abs(z)) + 
                                   0.27 * np.max(np.abs(z))**2)
    input_guess = [r0_empirical, r0_empirical]
    theta, H = FindHorizonFull(input_guess, singularities)
    all_theta = np.hstack((theta, theta + np.pi))
    all_H = np.vstack((H, np.flipud(H)))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121)
    PlotHorizon2d(
        ax1, all_theta, all_H, np.array(z), np.array(mass))
    ax1.set_title("Singularities at $z = \pm {}$".format(z))
    ax2 = fig.add_subplot(122, projection='3d')
    PlotHorizon3d(ax2, all_theta, all_H)
    if elev is not None:
        ax2.elev = elev
    if azim is not None:
        ax2.azim = azim
    ax2.set_title("Singularities at $z = \pm {}$".format(z))
    plt.show()


if __name__ == "__main__":
    singularities = {"masses": np.array([1.0, 1.0]),
                     "z_positions": np.array([-0.75, 0.75])}
    theta, H = FindHorizonSymmetric([1.25, 1.3], singularities)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(H[:, 0] * np.sin(theta),  H[:, 0] * np.cos(theta), 'b-')
    ax1.plot(H[:, 0] * np.sin(theta), -H[:, 0] * np.cos(theta), 'b-')
    ax1.plot(-H[:, 0] * np.sin(theta),  H[:, 0] * np.cos(theta), 'b-')
    ax1.plot(-H[:, 0] * np.sin(theta), -H[:, 0] * np.cos(theta), 'b-')
    ax1.plot(np.zeros(singularities['z_positions'].shape),
             singularities['z_positions'],
             'kx', markersize=12, markeredgewidth=2)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$z$")
    ax1.axis('equal')
    ax2 = fig.add_subplot(122, projection='3d')
    phi = np.linspace(0.0, 2.0 * np.pi, 20)
    all_theta = np.hstack((theta, np.flipud(np.pi - theta)))
    all_H = np.vstack((H, np.flipud(H)))
    X = np.zeros((len(all_theta), len(phi)))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for t in range(len(all_theta)):
        for p in range(len(phi)):
            X[t, p] = all_H[t, 0] * np.sin(all_theta[t]) * np.cos(phi[p])
            Y[t, p] = all_H[t, 0] * np.sin(all_theta[t]) * np.sin(phi[p])
            Z[t, p] = all_H[t, 0] * np.cos(all_theta[t])
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    surf = ax2.plot_surface(X, Y, Z,
                            rstride=1, cstride=1, linewidth=0,
                            facecolors=cm.jet(R), antialiased=False)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_zlabel("$z$")
    plt.show()
