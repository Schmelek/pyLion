import numpy as np
from scipy.optimize import fsolve
from numpy import complex128, sin as sin
from numpy import cos as cos
import matplotlib.pyplot as plt

f = 1


def C(radial_freqs, LD_parameter, offset, tau, ion_number):
    N = ion_number - f
    P = 2*N+1
    C = np.zeros((ion_number, P), dtype=complex128)
    for i in range(N):
        for j in range(P):
            t1 = j * tau / P
            t2 = (j + 1) * tau / P
            w = radial_freqs[i]
            C[i, j] = (
                -offset * cos(offset * t1) * cos(t1 * w) /
                (-(offset ** 2) + w ** 2)
                + offset * cos(offset * t2) * cos(t2 * w) /
                (-(offset ** 2) + w ** 2)
                - w * sin(offset * t1) * sin(t1 * w) /
                (-(offset ** 2) + w ** 2)
                + w * sin(offset * t2) * sin(t2 * w) /
                (-(offset ** 2) + w ** 2)
                + 1j
                * (
                    -offset * sin(t1 * w) * cos(offset * t1) /
                    (-(offset ** 2) + w ** 2)
                    + offset * sin(t2 * w) * cos(offset * t2) /
                    (-(offset ** 2) + w ** 2)
                    + w * sin(offset * t1) * cos(t1 * w) /
                    (-(offset ** 2) + w ** 2)
                    - w * sin(offset * t2) * cos(t2 * w) /
                    (-(offset ** 2) + w ** 2)
                )
            ) * LD_parameter[i, 0]
    excluded_mode = np.argsort(np.abs(radial_freqs-offset))[-(ion_number-N):]
    considered_modes = np.delete(np.arange(ion_number), excluded_mode)
    return C


def D(radial_freqs, LD_parameter, offset, tau, ion_number):
    N = ion_number - f
    P = 2*N+1
    D = np.zeros((P, P))
    if f != 0:
        excluded_mode = np.argsort(
            np.abs(radial_freqs-offset))[-(ion_number-N):]
        considered_modes = np.delete(np.arange(ion_number), excluded_mode)
    else:
        considered_modes = np.arange(N)
    for n in range(P):
        t1p = n * tau / P
        t2p = (n + 1) * tau / P
        t1 = n * tau / P

        def tmp(w): return (
            -(offset ** 2)
            * sin(t1 * w)
            * cos(offset * t1)
            * cos(offset * t1p)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t1 * w)
            * cos(offset * t1)
            * cos(offset * t2p)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t1p * w)
            * cos(offset * t1)
            * cos(offset * t1p)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset ** 2
            * sin(t2p * w)
            * cos(offset * t1)
            * cos(offset * t2p)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1)
            * sin(t1 * w)
            * sin(t1p * w)
            * cos(offset * t1p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1)
            * sin(t1 * w)
            * sin(t2p * w)
            * cos(offset * t2p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1)
            * cos(offset * t1p)
            * cos(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1)
            * cos(offset * t2p)
            * cos(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1p)
            * sin(t1 * w)
            * sin(t1p * w)
            * cos(offset * t1)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1p)
            * cos(offset * t1)
            * cos(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2p)
            * sin(t1 * w)
            * sin(t2p * w)
            * cos(offset * t1)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2p)
            * cos(offset * t1)
            * cos(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1)
            * sin(offset * t1p)
            * sin(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1)
            * sin(offset * t1p)
            * sin(t1p * w)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1)
            * sin(offset * t2p)
            * sin(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1)
            * sin(offset * t2p)
            * sin(t2p * w)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        )
        D[n, n] = sum(
            LD_parameter[a, 0] * LD_parameter[a, 1] * tmp(radial_freqs[a])
            for a in considered_modes)
    for n in range(P):
        for m in range(n):
            t1 = n * tau / P
            t2 = (n + 1) * tau / P
            t1p = m * tau / P
            t2p = (m + 1) * tau / P

            def tmp(w): return (
                -(offset ** 2)
                * sin(t1 * w)
                * cos(offset * t1)
                * cos(offset * t1p)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t1 * w)
                * cos(offset * t1)
                * cos(offset * t2p)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t1p * w)
                * cos(offset * t1)
                * cos(offset * t1p)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset ** 2
                * sin(t1p * w)
                * cos(offset * t1p)
                * cos(offset * t2)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t2 * w)
                * cos(offset * t1p)
                * cos(offset * t2)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset ** 2
                * sin(t2 * w)
                * cos(offset * t2)
                * cos(offset * t2p)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset ** 2
                * sin(t2p * w)
                * cos(offset * t1)
                * cos(offset * t2p)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t2p * w)
                * cos(offset * t2)
                * cos(offset * t2p)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1)
                * sin(t1 * w)
                * sin(t1p * w)
                * cos(offset * t1p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1)
                * sin(t1 * w)
                * sin(t2p * w)
                * cos(offset * t2p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1)
                * cos(offset * t1p)
                * cos(t1 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1)
                * cos(offset * t2p)
                * cos(t1 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1p)
                * sin(t1 * w)
                * sin(t1p * w)
                * cos(offset * t1)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1p)
                * sin(t1p * w)
                * sin(t2 * w)
                * cos(offset * t2)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1p)
                * cos(offset * t1)
                * cos(t1 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1p)
                * cos(offset * t2)
                * cos(t1p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2)
                * sin(t1p * w)
                * sin(t2 * w)
                * cos(offset * t1p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2)
                * sin(t2 * w)
                * sin(t2p * w)
                * cos(offset * t2p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2)
                * cos(offset * t1p)
                * cos(t1p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2)
                * cos(offset * t2p)
                * cos(t2 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2p)
                * sin(t1 * w)
                * sin(t2p * w)
                * cos(offset * t1)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2p)
                * sin(t2 * w)
                * sin(t2p * w)
                * cos(offset * t2)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2p)
                * cos(offset * t1)
                * cos(t1 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2p)
                * cos(offset * t2)
                * cos(t2 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t1)
                * sin(offset * t1p)
                * sin(t1 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t1)
                * sin(offset * t1p)
                * sin(t1p * w)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t1)
                * sin(offset * t2p)
                * sin(t1 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t1)
                * sin(offset * t2p)
                * sin(t2p * w)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t1p)
                * sin(offset * t2)
                * sin(t1p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t1p)
                * sin(offset * t2)
                * sin(t2 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t2)
                * sin(offset * t2p)
                * sin(t2 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t2)
                * sin(offset * t2p)
                * sin(t2p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            )
            D[n, m] = sum(
                tmp(radial_freqs[a]) * LD_parameter[a, 0] * LD_parameter[a, 1]
                for a in considered_modes
            )
            D[m, n] = D[n, m]
    return D


def omega(radial_freqs, LD_parameter, offset, tau, ion_number):
    N = ion_number - f
    P = 2*N+1
    if f != 0:
        excluded_mode = np.argsort(
            np.abs(radial_freqs-offset))[-(ion_number-N):]
        considered_modes = np.delete(np.arange(ion_number), excluded_mode)
    else:
        considered_modes = np.arange(N)
    Cmatr = C(radial_freqs, LD_parameter, offset,
              tau, ion_number)[considered_modes, :]
    Cmatr = np.vstack([Cmatr.real, Cmatr.imag])

    def equations(omega):
        omega = np.array(omega)
        out = np.concatenate(
            (np.dot(Cmatr, omega), np.dot(omega, omega) - 1), axis=None,
        )
        return out.T
    omega = fsolve(equations, np.arange(P))
    D1 = D(radial_freqs, LD_parameter, offset, tau, ion_number)
    if np.dot(omega, (np.dot(D1, omega))) < 0:
        omega = omega * \
            (-np.pi * 7 / (4 * np.dot(omega, (np.dot(D1, omega))))) ** 0.5
    else:
        omega = omega * \
            (np.pi / (4 * np.dot(omega, (np.dot(D1, omega))))) ** 0.5

    return omega


def Fidelity(radial_freqs, LD_parameter, offset, tau, ion_number):
    N = ion_number - f
    P = 2*N+1
    if f != 0:
        excluded_mode = np.argsort(
            np.abs(radial_freqs-offset))[-(ion_number-N):]
        considered_modes = np.delete(np.arange(ion_number), excluded_mode)
    else:
        considered_modes = np.arange(N)
    om = omega(radial_freqs, LD_parameter, offset, tau, ion_number)
    C1 = C(radial_freqs, LD_parameter, offset,
           tau, ion_number)
    D1 = D(radial_freqs, LD_parameter, offset, tau, ion_number)

    def omega_function(t): return np.piecewise(
        t, [((i * tau / P <= t) & (t < (i + 1) * tau / P))
            for i in range(P)], om)

    def Alpha_re(t, mode):
        w = radial_freqs[mode]
        for i in range(P):
            if i * tau / P <= t:
                if t < (i + 1) * tau / P:
                    t1 = i * tau / P
                    return sum(
                        C1.real[mode, j] * om[j] for j in range(i)
                    ) + omega_function(t) * LD_parameter[mode, 0] * (
                        offset * cos(offset * t) * cos(t * w) /
                        (-(offset ** 2) + w ** 2)
                        - offset
                        * cos(offset * t1)
                        * cos(t1 * w)
                        / (-(offset ** 2) + w ** 2)
                        + w * sin(offset * t) * sin(t * w) /
                        (-(offset ** 2) + w ** 2)
                        - w * sin(offset * t1) * sin(t1 * w) /
                        (-(offset ** 2) + w ** 2)
                    )

    def Alpha_im(t, mode):
        w = radial_freqs[mode]
        for i in range(P):
            if t >= i * tau / P:
                if t < (i + 1) * tau / P:
                    t1 = i * tau / P
                    return sum(
                        C1.imag[mode, j] * om[j] for j in range(i)
                    ) + omega_function(t) * LD_parameter[mode, 0] * (
                        offset * sin(t * w) * cos(offset * t) /
                        (-(offset ** 2) + w ** 2)
                        - offset
                        * sin(t1 * w)
                        * cos(offset * t1)
                        / (-(offset ** 2) + w ** 2)
                        - w * sin(offset * t) * cos(t * w) /
                        (-(offset ** 2) + w ** 2)
                        + w * sin(offset * t1) * cos(t1 * w) /
                        (-(offset ** 2) + w ** 2)
                    )
    phonon_number = 5

    betta = 1/(np.tanh(1/5*np.log(1+1/phonon_number)))
    G_1 = np.exp(-sum(Alpha_im(0.9999999*tau, a)**2 +
                      Alpha_re(tau*0.9999999, a)**2 for a in range(ion_number))*betta/2)
    G_2 = np.exp(-sum((Alpha_im(0.9999999*tau, a)**2 +
                       Alpha_re(tau*0.9999999, a)**2)*(LD_parameter[a, 1]/LD_parameter[a, 0])**2 for a in range(ion_number))*betta/2)
    G_plus = np.exp(-sum((Alpha_im(0.9999999*tau, a)**2+Alpha_re(tau*0.9999999, a)
                    ** 2)*(1+LD_parameter[a, 1]/LD_parameter[a, 0])**2 for a in range(ion_number))*betta/2)
    G_minus = np.exp(-sum((Alpha_im(0.9999999*tau, a)**2+Alpha_re(tau*0.9999999, a)
                           ** 2)*(1-LD_parameter[a, 1]/LD_parameter[a, 0])**2 for a in range(ion_number))*betta/2)

    D_ful = np.zeros((P, P))
    for n in range(P):
        t1p = n * tau / P
        t2p = (n + 1) * tau / P
        t1 = n * tau / P

        def tmp(w): return (
            -(offset ** 2)
            * sin(t1 * w)
            * cos(offset * t1)
            * cos(offset * t1p)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t1 * w)
            * cos(offset * t1)
            * cos(offset * t2p)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t1p * w)
            * cos(offset * t1)
            * cos(offset * t1p)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset ** 2
            * sin(t2p * w)
            * cos(offset * t1)
            * cos(offset * t2p)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1)
            * sin(t1 * w)
            * sin(t1p * w)
            * cos(offset * t1p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1)
            * sin(t1 * w)
            * sin(t2p * w)
            * cos(offset * t2p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1)
            * cos(offset * t1p)
            * cos(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1)
            * cos(offset * t2p)
            * cos(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1p)
            * sin(t1 * w)
            * sin(t1p * w)
            * cos(offset * t1)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1p)
            * cos(offset * t1)
            * cos(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2p)
            * sin(t1 * w)
            * sin(t2p * w)
            * cos(offset * t1)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2p)
            * cos(offset * t1)
            * cos(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1)
            * sin(offset * t1p)
            * sin(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1)
            * sin(offset * t1p)
            * sin(t1p * w)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1)
            * sin(offset * t2p)
            * sin(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1)
            * sin(offset * t2p)
            * sin(t2p * w)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        )
        D_ful[n, n] = sum(
            LD_parameter[a, 0] * LD_parameter[a, 1] * tmp(radial_freqs[a])
            for a in range(ion_number))
    for n in range(P):
        for m in range(n):
            t1 = n * tau / P
            t2 = (n + 1) * tau / P
            t1p = m * tau / P
            t2p = (m + 1) * tau / P

            def tmp(w): return (
                -(offset ** 2)
                * sin(t1 * w)
                * cos(offset * t1)
                * cos(offset * t1p)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t1 * w)
                * cos(offset * t1)
                * cos(offset * t2p)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t1p * w)
                * cos(offset * t1)
                * cos(offset * t1p)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset ** 2
                * sin(t1p * w)
                * cos(offset * t1p)
                * cos(offset * t2)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t2 * w)
                * cos(offset * t1p)
                * cos(offset * t2)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset ** 2
                * sin(t2 * w)
                * cos(offset * t2)
                * cos(offset * t2p)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset ** 2
                * sin(t2p * w)
                * cos(offset * t1)
                * cos(offset * t2p)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset ** 2
                * sin(t2p * w)
                * cos(offset * t2)
                * cos(offset * t2p)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1)
                * sin(t1 * w)
                * sin(t1p * w)
                * cos(offset * t1p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1)
                * sin(t1 * w)
                * sin(t2p * w)
                * cos(offset * t2p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1)
                * cos(offset * t1p)
                * cos(t1 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1)
                * cos(offset * t2p)
                * cos(t1 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1p)
                * sin(t1 * w)
                * sin(t1p * w)
                * cos(offset * t1)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1p)
                * sin(t1p * w)
                * sin(t2 * w)
                * cos(offset * t2)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t1p)
                * cos(offset * t1)
                * cos(t1 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t1p)
                * cos(offset * t2)
                * cos(t1p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2)
                * sin(t1p * w)
                * sin(t2 * w)
                * cos(offset * t1p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2)
                * sin(t2 * w)
                * sin(t2p * w)
                * cos(offset * t2p)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2)
                * cos(offset * t1p)
                * cos(t1p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2)
                * cos(offset * t2p)
                * cos(t2 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2p)
                * sin(t1 * w)
                * sin(t2p * w)
                * cos(offset * t1)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2p)
                * sin(t2 * w)
                * sin(t2p * w)
                * cos(offset * t2)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + offset
                * w
                * sin(offset * t2p)
                * cos(offset * t1)
                * cos(t1 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - offset
                * w
                * sin(offset * t2p)
                * cos(offset * t2)
                * cos(t2 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t1)
                * sin(offset * t1p)
                * sin(t1 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t1)
                * sin(offset * t1p)
                * sin(t1p * w)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t1)
                * sin(offset * t2p)
                * sin(t1 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t1)
                * sin(offset * t2p)
                * sin(t2p * w)
                * cos(t1 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t1p)
                * sin(offset * t2)
                * sin(t1p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t1p)
                * sin(offset * t2)
                * sin(t2 * w)
                * cos(t1p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                - w ** 2
                * sin(offset * t2)
                * sin(offset * t2p)
                * sin(t2 * w)
                * cos(t2p * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
                + w ** 2
                * sin(offset * t2)
                * sin(offset * t2p)
                * sin(t2p * w)
                * cos(t2 * w)
                / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            )
            D_ful[n, m] = sum(
                tmp(radial_freqs[a]) * LD_parameter[a, 0] * LD_parameter[a, 1]
                for a in range(ion_number)
            )
            D_ful[m, n] = D_ful[n, m]
    xi = np.dot(om, np.dot((D_ful), om))
    # fig = plt.figure()
    # time = np.arange(0, tau, tau / 10000)

    # for i in np.arange(1, ion_number+1, 1):
    #     Alpha_array_imag = 0
    #     Alpha_array_real = 0
    #     for t in time:
    #         Alpha_array_real = np.append(
    #             Alpha_array_real, Alpha_re(t, i - 1))
    #         Alpha_array_imag = np.append(
    #             Alpha_array_imag, Alpha_im(t, i - 1))
    #     axs = fig.add_subplot(2, 3, i)
    #     axs.plot(Alpha_array_real, Alpha_array_imag)
    #     # axs.set_xlabel("Alpha real")
    #     # axs.set_ylabel("Alpha imaginary")
    #     axs.grid(True)

    # plt.show()
    # plt.bar(np.arange(P), om, color="r")
    # plt.grid()
    # plt.show()

    fidel = 1/8*(2+2*np.sin(xi)*(G_1+G_2)+G_plus+G_minus)
    return fidel
