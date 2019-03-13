import numpy as np
from FOLLOW.arm_2link_todorov_gravity import Arm


def simulate(control, q0=None, dq0=None, print_val=None, time_span=None, dt=None):

    arm = Arm()

    if q0 is None:
        q0 = np.zeros(arm.dim_config_space)
    if dq0 is None:
        dq0 = np.zeros(arm.dim_config_space)

    if time_span is None:
        time_span = 0.001
    elif isinstance(time_span, np.ndarray):
        time_span = float(time_span[0])
    if dt is None:
        dt = 0.001
    elif isinstance(dt, np.ndarray):
        dt = float(dt[0])

    if control.shape[0] is 2:
        control = np.reshape(control, newshape=(1, 2))

    n_steps = int(time_span / dt)

    q = np.zeros((n_steps + 1, 2))
    dq = np.zeros((n_steps + 1, 2))

    q[0] = q0
    dq[0] = dq0

    p_xy = np.zeros((n_steps, 4))

    for i in range(1, int(time_span / dt) + 1):
        res = arm.evolveFns(q[i - 1], dq[i - 1], control[i - 1], dt=dt)
        q[i] = res[0] * dt + q[i - 1]
        dq[i] = res[1] * dt + dq[i - 1]

        if control.shape[0] is not 1:
            x0 = arm.l1 * np.cos(q[i][0])
            x1 = x0 + arm.l2 * np.cos(q[i][0] + q[i][1])
            y0 = arm.l1 * np.sin(q[i][0])
            y1 = y0 + arm.l2 * np.sin(q[i][0] + q[i][1])
            p_xy[i - 1] = np.array([x0, y0, x1, y1])

    return q[1:].astype(np.float32), dq[1:].astype(np.float32), p_xy.astype(np.float32)


def _check_simulate():
    import george
    arm = Arm()
    q0, dq0 = (np.zeros(arm.dim_config_space) for _ in range(2))
    simulation_time_span = 10.
    dt = .001
    t = np.arange(0, simulation_time_span, dt)
    kernel = george.kernels.ExpSquaredKernel(.5)
    np.random.seed(3000)
    gp = george.GP(kernel)
    u = gp.sample(t[::10], arm.dim_config_space) * 4.
    u = np.array([np.interp(t, t[::10], u[i]) for i in range(u.shape[0])])
    # time major
    u = u.T

    import matplotlib.pyplot as plt
    n_arms = 7
    lengths = np.linspace(.3, 1.6, n_arms)
    fig, axes = plt.subplots(n_arms + 1, figsize=(16, 10), sharex=True)
    axes[0].plot(u)
    axes[0].set_ylabel('control torques')
    for i in range(n_arms):
        arm.l1 = lengths[i]
        rq, drq = simulate(arm, simulation_time_span, dt, u, q0, dq0)
        axes[i + 1].plot(rq)
        axes[i + 1].set_ylabel('arm configuration $l_1 = {}$'.format(arm.l1))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    _check_simulate()
