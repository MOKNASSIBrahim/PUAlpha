import numpy as np
from numba import njit


@njit
def update_props(p, phi, p_ref,
                 rho_w, rho_s,
                 chi_w, chi_s,
                 mu_w, mu_s,
                 rho, chi, mu):
    for i in range(p.shape[0]):
        a = phi[i]
        rho[i] = (1.0 - a) * rho_w + a * rho_s
        chi[i] = (1.0 - a) * chi_w + a * chi_s
        mu[i] = mu_s if a > 0.5 else mu_w
    return rho, chi, mu


@njit
def sound_speed(rho, chi):
    eps = 1e-14
    rr = rho if rho > eps else eps
    cc = chi if chi > eps else eps
    return 1.0 / np.sqrt(rr * cc)


@njit
def eigs(u, rho, chi):
    eps = 1e-14
    rr = rho if rho > eps else eps
    cc = chi if chi > eps else eps
    x = u * u + 4.0 / (rr * cc)
    s = np.sqrt(x) if x > 0.0 else 0.0
    return 0.5 * (u - s), 0.5 * (u + s), u


@njit
def wave_amplitude(u, rho, chi):
    lm, lp, la = eigs(u, rho, chi)
    return max(np.abs(lm), np.abs(lp), np.abs(la))


@njit
def interp_cells(q, Nz, dz, zc):
    xi = zc / dz - 0.5
    i = int(np.floor(xi))
    if i < 0:
        i = 0
    if i > Nz - 2:
        i = Nz - 2
    s = xi - i
    return (1.0 - s) * q[i] + s * q[i + 1], i


@njit
def departure_point(u, Nz, dz, z_face, aV, dt):
    zc = z_face
    for _ in range(5000):
        zc_old = zc
        u_zc, _i = interp_cells(u, Nz, dz, zc_old)
        zc = z_face - aV * dt * u_zc
        if np.abs(zc - zc_old) <= 1e-8:
            return zc
    raise RuntimeError("Characteristic foot failed to converge within 5000 iterations")


@njit
def superbee(r):
    if r <= 0.0:
        return 0.0
    return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))


@njit
def phi_faces_plain(phi):
    Nz = phi.shape[0]
    phiL = np.empty(Nz + 1)
    phiR = np.empty(Nz + 1)
    phiL[0] = phi[0]
    phiR[0] = phi[0]
    phiL[Nz] = phi[Nz - 1]
    phiR[Nz] = phi[Nz - 1]
    for j in range(1, Nz):
        phiL[j] = phi[j - 1]
        phiR[j] = phi[j]
    return phiL, phiR


@njit
def muscl_superbee(phi):
    Nz = phi.shape[0]
    eps = 1e-12
    phiL, phiR = phi_faces_plain(phi)
    if Nz < 3:
        return phiL, phiR

    sigma = np.zeros(Nz)
    for i in range(1, Nz - 1):
        dl = phi[i] - phi[i - 1]
        dr = phi[i + 1] - phi[i]
        r = 0.0 if np.abs(dr) < eps else dl / dr
        sigma[i] = superbee(r) * dr

    for j in range(1, Nz):
        phiL[j] = phi[j - 1] + 0.5 * sigma[j - 1]
        phiR[j] = phi[j] - 0.5 * sigma[j]
    return phiL, phiR


@njit
def compute_alpha_faces(p, u, phi, rho, chi, dz, dt):
    Nz = p.shape[0]
    eps = 1e-12
    alpha_face = np.full(Nz + 1, 0.5)

    Q = np.zeros(Nz)
    for i in range(Nz):
        Q[i] = wave_amplitude(u[i], rho[i], chi[i])

    for j in range(1, Nz):
        S = max(max(Q[j - 1], Q[j]), eps)
        alpha_tilde = 0.5 * dz / (S * dt)

        qim1 = Q[j - 1]
        qi = Q[j]
        qip1 = Q[j + 1] if (j + 1) < Nz else Q[Nz - 1]

        r = (qi - qim1) / (qip1 - qi + eps)
        psi = 0.0 if r <= 0.0 else min(1.0, r)
        a = alpha_tilde + (0.5 - alpha_tilde) * psi
        alpha_face[j] = min(max(a, 0.1), 1.0)
    return alpha_face


@njit
def dt_vfc(p, u, phi, rho, chi, dz, CFL, t, Tf, alpha_eff):
    eps = 1e-12
    fac = np.sqrt(2.0 * alpha_eff)
    denom = 0.0
    for i in range(p.shape[0]):
        val = fac * wave_amplitude(u[i], rho[i], chi[i])
        if val > denom:
            denom = val
    dt = 1.0 if denom < eps else CFL * dz / denom
    if t + dt > Tf:
        dt = Tf - t
    return dt


@njit
def inlet_bc(p_in, rho_in, chi_in, p0, u0, rho0, chi0):
    c0 = sound_speed(rho0, chi0)
    c_in = sound_speed(rho_in, chi_in)
    uB = (u0 - p0 / (rho0 * c0)) + p_in / (rho_in * c_in)
    return p_in, uB


@njit
def outlet_bc_nonreflecting(pN, uN, rhoN, chiN):
    kappa = 1
    cN = sound_speed(rhoN, chiN)
    Rp_int = uN + pN / (rhoN * cN)
    Rm_int = uN - pN / (rhoN * cN)
    Rm_ext = (1.0 - kappa) * Rm_int
    uB = 0.5 * (Rp_int + Rm_ext)
    pB = 0.5 * (rhoN * cN) * (Rp_int - Rm_ext)
    return pB, uB


@njit
def step_vfc(p, u, phi,
             dz, dt,
             rho, chi, tau_w,
             p_in, rho_in, chi_in, phi_in,
             theta, R_hat, g,
             use_superbee_phi, u_data,
             bc_mode, p_out):
    Nz = p.shape[0]
    eps = 1e-12
    inv_dz = 1.0 / dz

    alpha_face = compute_alpha_faces(p, u, phi, rho, chi, dz, dt)

    pF = np.zeros(Nz + 1)
    uF = np.zeros(Nz + 1)
    phiF = np.zeros(Nz + 1)

    if bc_mode == 1:
        pF[0] = p_in
        uF[0] = u[0]
    else:
        pF[0], uF[0] = inlet_bc(p_in, rho_in, chi_in, p[0], u[0], rho[0], chi[0])
    phiF[0] = phi_in

    if use_superbee_phi == 1:
        phiL_face, phiR_face = muscl_superbee(phi)
    else:
        phiL_face, phiR_face = phi_faces_plain(phi)

    for j in range(1, Nz):
        aV = alpha_face[j]
        zc = departure_point(u, Nz, dz, j * dz, aV, dt)

        p_hat, iL = interp_cells(p, Nz, dz, zc)
        u_hat, _i = interp_cells(u, Nz, dz, zc)
        iR = iL + 1

        rho_f = max(0.5 * (rho[iL] + rho[iR]), eps)
        chi_f = max(0.5 * (chi[iL] + chi[iR]), 1e-14)
        RF = max(0.5 * (R_hat[iL] + R_hat[iR]), eps)
        thF = 0.5 * (theta[iL] + theta[iR])
        tauF = 0.5 * (tau_w[iL] + tau_w[iR])

        du_dz = (u[iR] - u[iL]) * inv_dz
        dp_dz = (p[iR] - p[iL]) * inv_dz
        Su_face = 2.0 * tauF / (rho_f * RF) + g * np.sin(thF)

        pF[j] = p_hat + aV * dt * (-(1.0 / chi_f) * du_dz)
        uF[j] = u_hat + aV * dt * (u_hat * du_dz - (1.0 / rho_f) * dp_dz + Su_face)

        phi_up = phiL_face[j] if uF[j] >= 0.0 else phiR_face[j]
        dphi_dz = (phiR_face[j] - phiL_face[j]) * inv_dz
        phiF[j] = phi_up + aV * dt * (-uF[j] * dphi_dz)

    if bc_mode == 1:
        pF[Nz] = p_out
        uF[Nz] = u[Nz - 1]
    else:
        pF[Nz], uF[Nz] = outlet_bc_nonreflecting(p[Nz - 1], u[Nz - 1],
                                                 rho[Nz - 1], chi[Nz - 1])
    phiF[Nz] = phi[Nz - 1]

    p_new = p.copy()
    u_new = u.copy()
    phi_new = phi.copy()

    for i in range(Nz):
        dp_face = (pF[i + 1] - pF[i]) * inv_dz
        du_face = (uF[i + 1] - uF[i]) * inv_dz

        rr = max(rho[i], eps)
        cc = max(chi[i], 1e-14)
        Su_cell = 2.0 * tau_w[i] / (rr * max(R_hat[i], eps)) + g * np.sin(theta[i])

        p_new[i] = p[i] - dt * (u[i] * dp_face + (1.0 / cc) * du_face)
        u_new[i] = u[i] - dt * ((1.0 / rr) * dp_face) + dt * Su_cell
        phi_new[i] = (phi[i] - (dt * inv_dz) * (uF[i + 1] * phiF[i + 1] - uF[i] * phiF[i])
                      + dt * phi[i] * du_face)

    p_new[0] = p_in
    phi_new[0] = phi_in
    u_new[0] = u_new[1]

    if bc_mode == 1:
        p_new[-1] = p_out
        u_new[-1] = u_new[-2]
    else:
        p_new[-1], u_new[-1] = outlet_bc_nonreflecting(p_new[-1], u_new[-1],
                                                       rho[-1], chi[-1])
    phi_new[-1] = phi_new[-2]

    return p_new, u_new, phi_new, alpha_face
