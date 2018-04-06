"""Example implementation of the Lorenz model.

References
----------
- Lorenz, E. (1995). Predictability: a problem partly solved.
  In Proceedings of the Seminar on Predictability, 4-8 September 1995 ,
  volume 1, pages 1–18, Shinfield Park, Reading. European Center on Medium
  Range Weather Forecasting.

"""
from functools import partial

import numpy as np

import elfi


def truth_step(X, Y, h, F, b, c):
    """
    Calculate the time increment in the X and Y variables
    for the Lorenz initial model.

    Parameters
    ----------
    X: 1D ndarray
        Values of X variables at the current time step
    Y: 1D ndarray
        Values of Y variables at the current time step
    h: float
        Coupling constant
    F: float
        Forcing term
    b: float
        Spatial scale ratio
    c: float
        Time scale ratio

    Returns
    -------
    dX_dt: 1D ndarray
        Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dX_dt = np.zeros(X.shape)
    dY_dt = np.zeros(Y.shape)
    for k in range(K):
        dX_dt[k] = (-X[k - 1] * (X[k - 2] - X[(k + 1) % K]) -
                    X[k] + F - ((h * c) / b) * np.sum(Y[k * J: (k + 1) * J]))
    for j in range(J * K):
        dY_dt[j] = (-c * b * Y[(j + 1) % (J * K)] *
                    (Y[(j + 2) % (J * K)] - Y[j - 1]) -
                    c * Y[j] + ((h * c) / b) * X[int(j / J)])

    return dX_dt, dY_dt


def lorenz96(x_initial, y_initial, h, f, b, c,
             time_step=0.0001, num_steps=50000, burn_in=0, skip=0):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Parameters
    ----------
    x_initial: 1D ndarray
        Initial X values.
    y_initial: 1D ndarray
        Initial Y values.
    h: float
        Coupling constant.
    f: float
        Forcing term.
    b: float
        Spatial scale ratio
    c: float
        Time scale ratio
    time_step: float
        Size of the integration time step in MTU
    num_steps: int
        Number of time steps integrated forward.
    burn_in: int
        Number of time steps not saved at the beginning
    skip: int
        Number of time steps skipped between archival

    Returns
    -------
    X_out: [number of timesteps, X size]
        X values at each time step,
    Y_out: [number of timesteps, Y size]
        Y values at each time step
    """

    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    k1_dx_dt = np.zeros(x.shape)
    k2_dx_dt = np.zeros(x.shape)
    k3_dx_dt = np.zeros(x.shape)
    k4_dx_dt = np.zeros(x.shape)
    k1_dy_dt = np.zeros(y.shape)
    k2_dy_dt = np.zeros(y.shape)
    k3_dy_dt = np.zeros(y.shape)
    k4_dy_dt = np.zeros(y.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        k1_dx_dt[:], k1_dy_dt[:] = lorenz96(x, y, h, f, b, c)
        k2_dx_dt[:], k2_dy_dt[:] = lorenz96(x + k1_dx_dt * time_step / 2,
                                            y + k1_dy_dt * time_step / 2,
                                            h, f, b, c)
        k3_dx_dt[:], k3_dy_dt[:] = lorenz96(x + k2_dx_dt * time_step / 2,
                                            y + k2_dy_dt * time_step / 2,
                                            h, f, b, c)
        k4_dx_dt[:], k4_dy_dt[:] = lorenz96(x + k3_dx_dt * time_step,
                                            y + k3_dy_dt * time_step,
                                            h, f, b, c)
        x += ((k1_dx_dt + 2 * k2_dx_dt + 2 * k3_dx_dt + k4_dx_dt)
              / 6 * time_step)
        y += ((k1_dy_dt + 2 * k2_dy_dt + 2 * k3_dy_dt + k4_dy_dt)
              / 6 * time_step)
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = y
            i += 1
    return x_out, y_out, times, steps


def get_model(n_obs=50000, true_params=None, seed_obs=None, stochastic=True):
    """Return a complete Lorenz model in inference task.

    # TODO: Implement stochastic Lorenz model..

    Parameters
    ----------

    n_obs : int, optional
        Number of observations.
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.
    stochastic : bool, optional
        Whether to use the stochastic or deterministic Ricker model.

    Returns
    -------
    m : elfi.ElfiModel

    """

    if stochastic:
        raise Exception("Currently stochastic "
                        "version of Lorenz model doen't supported")
        # TODO: what is the default true parameters?
        # Should we have default parameters?
    else:
        simulator = partial(lorenz96, true_params)
        # TODO: what is the default true parameters?
        # Should we have default parameters?

    m = elfi.ElfiModel(name="lorenz")

    return m
