import numpy as np
from numba import njit, prange, get_thread_id


@njit
def _mix_len(r, R, kappa, lam):
    r_sw = R - (lam * R) / (kappa + 1e-12)
    return kappa * (R - r) if r >= r_sw else lam * R


@njit
def _mu_app_wall(tau_w_mag, tau_y, K, n):
    eps = 1e-14
    if tau_w_mag <= tau_y + eps:
        return K
    g = ((tau_w_mag - tau_y) / (K + eps)) ** (1.0 / (n + eps))
    return tau_w_mag / (g + eps)


@njit
def _hb_lam(T, tau_y, K, n):
    eps = 1e-14
    if T <= tau_y + eps:
        return 0.0
    x = (T - tau_y) / (K + eps)
    return 0.0 if x <= 0.0 else x ** (1.0 / (n + eps))


@njit
def shear_rate_from_tau(T, rho, R, r, tau_y, K, n, kappa, lam, turb_on, r_crit):
    eps = 1e-14
    if T <= tau_y + eps:
        return 0.0

    if abs(n - 1.0) < 1e-12:
        if turb_on == 0 or r >= r_crit:
            return (T - tau_y) / (K + eps)
        lm   = _mix_len(r, R, kappa, lam)
        a    = rho * lm * lm
        if a < 1e-20:
            return (T - tau_y) / (K + eps)
        disc = K * K + 4.0 * a * (T - tau_y)
        return (-K + max(disc, 0.0) ** 0.5) / (2.0 * a + eps)

    a = 0.0
    if turb_on == 1 and r < r_crit:
        lm = _mix_len(r, R, kappa, lam)
        a  = rho * lm * lm

    if a < 1e-20:
        return _hb_lam(T, tau_y, K, n)

    g_lo, g_hi = 0.0, 1.0
    for _ in range(28):
        if tau_y + K * (g_hi ** n) + a * g_hi * g_hi >= T:
            break
        g_hi *= 2.0

    for _ in range(28):
        g_mid = 0.5 * (g_lo + g_hi)
        if tau_y + K * (g_mid ** n) + a * g_mid * g_mid >= T:
            g_hi = g_mid
        else:
            g_lo = g_mid
        if (g_hi - g_lo) / (g_hi + 1e-30) < 1e-8:
            break

    return 0.5 * (g_lo + g_hi)


@njit
def ubar_from_tau(tau_w_mag, rho, R, tau_y, K, n, kappa, lam,
                  r, dr, rmid, xi_mid, g_lam, w_lam, g_turb, w_turb):
    Zc  = 808.0
    Cm  = 11.63
    eps = 1e-14
    Nr  = r.shape[0]
    tw  = tau_w_mag

    for j in range(Nr - 1):
        g_lam[j] = shear_rate_from_tau(
            xi_mid[j] * tw, rho, R, rmid[j], tau_y, K, n, kappa, lam, 0, 0.0)

    w_lam[Nr - 1] = 0.0
    for j in range(Nr - 2, -1, -1):
        w_lam[j] = w_lam[j + 1] + g_lam[j] * dr[j]

    Zmax = 0.0
    for j in range(Nr - 1):
        wmid = 0.5 * (w_lam[j] + w_lam[j + 1])
        Z    = abs((R * wmid * rho / (tw + eps)) * g_lam[j])
        if Z > Zmax:
            Zmax = Z

    if Zmax <= Zc:
        I = 0.0
        for j in range(Nr - 1):
            I += 0.5 * (r[j] * w_lam[j] + r[j + 1] * w_lam[j + 1]) * dr[j]
        return (2.0 / (R * R)) * I

    muw     = _mu_app_wall(tw, tau_y, K, n)
    delta_c = min(max(Cm * muw / ((rho + eps) * (tw + eps)) ** 0.5, 0.0), R)
    r_crit  = R - delta_c

    for j in range(Nr - 1):
        g_turb[j] = shear_rate_from_tau(
            xi_mid[j] * tw, rho, R, rmid[j], tau_y, K, n, kappa, lam, 1, r_crit)

    w_turb[Nr - 1] = 0.0
    for j in range(Nr - 2, -1, -1):
        w_turb[j] = w_turb[j + 1] + g_turb[j] * dr[j]

    I = 0.0
    for j in range(Nr - 1):
        I += 0.5 * (r[j] * w_turb[j] + r[j + 1] * w_turb[j + 1]) * dr[j]
    return (2.0 / (R * R)) * I


@njit
def _solve_tau_w(ubar, rho, R, tau_y, K, n, kappa, lam,
                 r, dr, rmid, xi_mid, tau_prev_mag,
                 g_lam, w_lam, g_turb, w_turb):
    eps_u = 1e-12
    if abs(ubar) < eps_u:
        return 0.0

    u_target = abs(ubar)
    eps      = 1e-30
    s        = 1.3
    tw0      = tau_prev_mag if tau_prev_mag >= 1e-6 else max(tau_y * 1.0001, 1e-6)
    tw_lo    = max(tw0 / s, max(tau_y * 1.0001, 1e-6))
    tw_hi    = tw0 * s

    w_hi = ubar_from_tau(tw_hi, rho, R, tau_y, K, n, kappa, lam,
                         r, dr, rmid, xi_mid, g_lam, w_lam, g_turb, w_turb)
    for _ in range(60):
        if w_hi >= u_target:
            break
        tw_hi *= 2.0
        w_hi   = ubar_from_tau(tw_hi, rho, R, tau_y, K, n, kappa, lam,
                                r, dr, rmid, xi_mid, g_lam, w_lam, g_turb, w_turb)

    if w_hi < u_target:
        sgn = 1.0 if ubar >= 0.0 else -1.0
        return -sgn * tw_hi

    w_lo = ubar_from_tau(tw_lo, rho, R, tau_y, K, n, kappa, lam,
                         r, dr, rmid, xi_mid, g_lam, w_lam, g_turb, w_turb)
    if w_lo > u_target:
        tw_lo = max(tau_y * 1.0001, 1e-6)
        w_lo  = ubar_from_tau(tw_lo, rho, R, tau_y, K, n, kappa, lam,
                               r, dr, rmid, xi_mid, g_lam, w_lam, g_turb, w_turb)

    f_lo, f_hi, side = w_lo - u_target, w_hi - u_target, 0

    for _ in range(60):
        tw_c = tw_hi - f_hi * (tw_hi - tw_lo) / (f_hi - f_lo + eps)
        tw_c = min(max(tw_c, tw_lo), tw_hi)
        w_c  = ubar_from_tau(tw_c, rho, R, tau_y, K, n, kappa, lam,
                              r, dr, rmid, xi_mid, g_lam, w_lam, g_turb, w_turb)
        f_c  = w_c - u_target
        if abs(f_c) / (u_target + eps) < 1e-3:
            tw_lo = tw_hi = tw_c
            break
        if (tw_hi - tw_lo) / (tw_hi + eps) < 1e-8:
            break
        if f_c <= 0.0:
            if side == 1:
                f_hi *= 0.5
            tw_lo, f_lo, side = tw_c, f_c, 1
        else:
            if side == -1:
                f_lo *= 0.5
            tw_hi, f_hi, side = tw_c, f_c, -1

    sgn = 1.0 if ubar >= 0.0 else -1.0
    return -sgn * 0.5 * (tw_lo + tw_hi)


@njit(parallel=True)
def compute_tau_w(u, phi, rho, R, muN, tauy1, muS, n1, kappa, lam,
                  r, dr, rmid, xi_mid, tau_w_out,
                  g_lam_all, w_lam_all, g_turb_all, w_turb_all):
    Nz = u.shape[0]
    for i in prange(Nz):
        tid   = get_thread_id()
        a     = phi[i]
        tau_y = a * tauy1
        K     = (1.0 - a) * muN  + a * muS
        n     = (1.0 - a) * 1.0  + a * n1
        tau_w_out[i] = _solve_tau_w(
            u[i], rho[i], R, tau_y, K, n, kappa, lam,
            r, dr, rmid, xi_mid, abs(tau_w_out[i]),
            g_lam_all[tid], w_lam_all[tid],
            g_turb_all[tid], w_turb_all[tid])
    return tau_w_out


@njit
def reconstruct_cell(tau_w, rho, alpha, R, muN, tauy1, muS, n1, kappa, lam,
                     r, dr, rmid, xi_mid, w_out, tau_rz_out,
                     g_lam, w_lam, g_turb, w_turb):
    Zc     = 808.0
    Cm     = 11.63
    eps    = 1e-14
    Nr     = r.shape[0]
    tau_y  = alpha * tauy1
    K      = (1.0 - alpha) * muN + alpha * muS
    n      = (1.0 - alpha) * 1.0 + alpha * n1
    tw_mag = abs(tau_w)

    for j in range(Nr - 1):
        g_lam[j] = shear_rate_from_tau(
            xi_mid[j] * tw_mag, rho, R, rmid[j], tau_y, K, n, kappa, lam, 0, 0.0)

    w_lam[Nr - 1] = 0.0
    for j in range(Nr - 2, -1, -1):
        w_lam[j] = w_lam[j + 1] + g_lam[j] * dr[j]

    Zmax = 0.0
    for j in range(Nr - 1):
        wmid = 0.5 * (w_lam[j] + w_lam[j + 1])
        Z    = abs((R * wmid * rho / (tw_mag + eps)) * g_lam[j])
        if Z > Zmax:
            Zmax = Z

    turb_on = 0;  r_crit = 0.0
    if Zmax > Zc:
        turb_on = 1
        muw     = _mu_app_wall(tw_mag, tau_y, K, n)
        delta_c = min(max(Cm * muw / ((rho + eps) * (tw_mag + eps)) ** 0.5, 0.0), R)
        r_crit  = R - delta_c

    for j in range(Nr - 1):
        g_turb[j] = shear_rate_from_tau(
            xi_mid[j] * tw_mag, rho, R, rmid[j], tau_y, K, n, kappa, lam, turb_on, r_crit)

    w_out[Nr - 1] = 0.0
    for j in range(Nr - 2, -1, -1):
        w_out[j] = w_out[j + 1] + g_turb[j] * dr[j]

    for j in range(Nr):
        tau_rz_out[j] = (r[j] / R) * tau_w

    if tau_w > 0.0:
        for j in range(Nr):
            w_out[j] = -w_out[j]


@njit(parallel=True)
def reconstruct_all(tau_w, rho, phi, R, muN, tauy1, muS, n1, kappa, lam,
                    r, dr, rmid, xi_mid, W_half,
                    g_lam_all, w_lam_all, g_turb_all, w_turb_all):
    Nz           = tau_w.shape[0]
    Nr           = r.shape[0]
    tau_rz_dummy = np.empty(Nr, dtype=np.float64)
    for i in prange(Nz):
        tid = get_thread_id()
        reconstruct_cell(
            tau_w[i], rho[i], phi[i], R, muN, tauy1, muS, n1, kappa, lam,
            r, dr, rmid, xi_mid,
            W_half[:, i], tau_rz_dummy,
            g_lam_all[tid], w_lam_all[tid],
            g_turb_all[tid], w_turb_all[tid])
            
            
