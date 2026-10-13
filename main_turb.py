
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from numba import get_num_threads

from fvc   import update_props, dt_vfc, step_vfc
from tau_w import compute_tau_w, ubar_from_tau
from data_and_postprocessing import radial_metrics

# ── Physical parameters (Table 3) ─────────────────────────────────────────────
R      = 1.0
L      = 1000.0
rho_s  = 1600.0
rho_w  = 1000.0
K      = 0.03
N_HB   = 1.0
TAU_Y  = 0.0
MU_N   = 1e-3
chi    = 8e-10
g_grav = 9.81
D      = 2.0 * R
kappa  = 0.4
lam    = 0.07
c0     = 1.0 / np.sqrt(rho_s * chi)

# ── Numerical parameters ──────────────────────────────────────────────────────
Nr      = 100
zeta    = 0.95
Nz      = 500
CFL     = 0.9
ALPHA   = 0.9
T_FINAL = 10.0
REC     = 20

TW_VALS = [0.06, 0.15, 0.25, 0.50, 0.70, 2.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_turbulence")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.linewidth": 0.8,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "legend.frameon": False,
    "legend.fontsize": 10, "xtick.direction": "in", "ytick.direction": "in",
})
COL_SIM     = "#E87722"
COL_EX      = "black"
LINE_COLORS = ["#4878CF", "#3aaa6e", "#d62728", "#9467bd", "#8c564b", "#17becf"]


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True)


def _scratch():
    nt = get_num_threads()
    return (np.empty((nt, Nr - 1), dtype=np.float64),
            np.empty((nt, Nr),     dtype=np.float64),
            np.empty((nt, Nr - 1), dtype=np.float64),
            np.empty((nt, Nr),     dtype=np.float64))


# ── Reference: Dodge-Metzner friction factor ──────────────────────────────────
def _dodge_metzner(Re, n, tol=1e-12, maxiter=2000):
    A = 4.0 / n**0.75
    C = -0.4 / n**1.2
    f = 0.316 * Re**(-0.25) / 4.0
    for _ in range(maxiter):
        arg = Re * f**((2.0 - n) / 2.0)
        if arg <= 0.0:
            return None
        rhs = A * np.log10(arg) + C
        if rhs <= 0.0:
            return None
        fn = 1.0 / rhs**2
        if abs(fn - f) / (f + 1e-15) < tol:
            return fn
        f = 0.6 * f + 0.4 * fn
    return f


def _Re_MR(V, rho, D, K, n):
    return (rho * V**(2.0 - n) * D**n
            / (8.0**(n - 1) * K * ((3.0 * n + 1.0) / (4.0 * n))**n))


# ── FVC transient run ─────────────────────────────────────────────────────────
def _run_fvc(tw_tgt, r, dr, rmid, xi_mid):
    
    gml, wl, gm, ww = _scratch()

    
    U_expected = ubar_from_tau(tw_tgt, rho_s, R, TAU_Y, K, N_HB, kappa, lam,
                                r, dr, rmid, xi_mid,
                                gml[0], wl[0], gm[0], ww[0])

    p_in = rho_s * c0 * U_expected  +  2.0 * tw_tgt * L / R

    dz    = L / Nz
    p     = np.zeros(Nz)    
    u     = np.zeros(Nz)     
    phi   = np.ones(Nz)      # pure slurry
    tw    = np.zeros(Nz)     

    ra = np.empty(Nz);  ca = np.empty(Nz);  ma = np.empty(Nz)
    update_props(p, phi, 0.0, rho_w, rho_s, chi, chi, MU_N, K, ra, ca, ma)

    R_hat = R * np.ones(Nz)
    theta = np.zeros(Nz)

    sl    = slice(Nz // 4, 3 * Nz // 4)
    t     = 0.0
    step  = 0
    T_arr = [0.0]
    U_arr = [0.0]

    while t < T_FINAL:
        tw = compute_tau_w(u, phi, ra, R, MU_N, TAU_Y, K, N_HB, kappa, lam,
                           r, dr, rmid, xi_mid, tw, gml, wl, gm, ww)
        dt = dt_vfc(p, u, phi, ra, ca, dz, CFL, t, T_FINAL, ALPHA)
        p, u, phi, _ = step_vfc(p, u, phi, dz, dt, ra, ca, tw,
                                  p_in, ra[0], ca[0], 1.0, theta, R_hat, g_grav, 0, 0.0,
                                  0, 0.0)
        update_props(p, phi, 0.0, rho_w, rho_s, chi, chi, MU_N, K, ra, ca, ma)
        t    += dt
        step += 1
        if step % REC == 0:
            T_arr.append(t)
            U_arr.append(float(np.mean(u[sl])))

    T_arr.append(t)
    U_arr.append(float(np.mean(u[sl])))

    
    U_ss  = float(np.mean(u[sl]))
    tw_ss = float(-np.mean(tw[sl]))
    f_c   = tw_ss / (0.5 * rho_s * U_ss**2)
    Re    = _Re_MR(U_ss, rho_s, D, K, N_HB)
    f_dm  = _dodge_metzner(Re, N_HB)

    return dict(
        T=np.array(T_arr), U=np.array(U_arr),
        U_ss=U_ss, tw_ss=tw_ss,
        f_c=f_c, Re=Re, f_dm=f_dm,
    )


# ── Figures ───────────────────────────────────────────────────────────────────
def _fig_u_transient(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.12)

    for i, (tw, res) in enumerate(results):
        col  = LINE_COLORS[i % len(LINE_COLORS)]
        U_ss = res["U_ss"]
        ax.plot(res["T"], res["U"], color=col, lw=2.0,
                label=rf"$\tau_w={tw}$ Pa  ($\bar{{u}}_\infty={U_ss:.4f}$ m/s)")
        ax.axhline(U_ss, color=col, lw=1.3, ls="--", alpha=0.70)

    ax.set_xlabel("$t$  (s)")
    ax.set_ylabel(r"$\bar{u}$  (m/s)")
    ax.legend(loc="lower right", fontsize=9)
    _clean(ax)

    path = os.path.join(OUT_DIR, "fig15_u_transient.png")
    fig.savefig(path, dpi=200);  plt.close(fig)
    print(f"  -> {path}")


def _fig_f_vs_Re(results):
    Re_pts = [res["Re"]  for _, res in results]
    f_pts  = [res["f_c"] for _, res in results]

    Re_line = np.logspace(np.log10(min(Re_pts) * 0.6),
                          np.log10(max(Re_pts) * 1.6), 400)
    f_line, Re_valid = [], []
    for Re in Re_line:
        v = _dodge_metzner(Re, N_HB)
        if v is not None:
            f_line.append(v);  Re_valid.append(Re)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.12)
    ax.loglog(np.array(Re_valid), np.array(f_line),
              color=COL_EX, lw=2.0, ls="--",
              label="Dodge-Metzner (eq. 3.73)", zorder=5)
    ax.loglog(Re_pts, f_pts,
              color=COL_SIM, lw=1.8, marker="o", ms=7,
              label="Simulated (FVC)", zorder=4)
    ax.set_xlabel(r"$Re_\mathrm{MR}$")
    ax.set_ylabel("Fanning friction factor  $f$")
    ax.legend()
    ax.xaxis.set_minor_locator(LogLocator(subs="auto"))
    ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.2, alpha=0.2)

    path = os.path.join(OUT_DIR, "fig16_f_vs_Re.png")
    fig.savefig(path, dpi=200);  plt.close(fig)
    print(f"  -> {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("  BENCHMARK C — Turbulent FVC vs Dodge-Metzner")
    print(f"  R={R} m  L={L} m  Nz={Nz}  Nr={Nr}  CFL={CFL}")
    print(f"  K={K} Pa*s^n  n={N_HB}  kappa={kappa}  lam={lam}  chi={chi}")
    print(f"  Init: p=u=tw=0.  p_in = rho*c0*U_ss + 2*tw*L/R.")
    print(f"  U_ss is read from the solver, not imposed.")
    print("=" * 64)

    r, dr, rmid, xi_mid = radial_metrics(R, Nr, zeta)

    print(f"\n  {'tw (Pa)':>8}  {'U_ss':>10}  {'tw_ss':>8}  "
          f"{'Re_MR':>8}  {'f_FVC':>9}  {'f_DM':>9}  {'f/f_DM':>7}")
    print("  " + "-" * 70)

    t0      = time.perf_counter()
    results = []

    for tw in TW_VALS:
        print(f"    tw = {tw:.3f} Pa ...", end=" ", flush=True)
        res = _run_fvc(tw, r, dr, rmid, xi_mid)
        results.append((tw, res))

        f_dm_s = f"{res['f_dm']:.5f}" if res["f_dm"] else "   N/A"
        ratio  = res["f_c"] / res["f_dm"] if res["f_dm"] else float("nan")
        print(f"done")
        print(f"  {tw:8.3f}  {res['U_ss']:10.5f}  {res['tw_ss']:8.5f}  "
              f"{res['Re']:8.0f}  {res['f_c']:9.5f}  {f_dm_s:>9}  {ratio:7.3f}")

    print(f"\n  Total CPU time: {time.perf_counter() - t0:.1f} s")

    valid    = [(tw, res) for tw, res in results if res["f_dm"] is not None]
    f_c_arr  = np.array([res["f_c"]  for _, res in valid])
    f_dm_arr = np.array([res["f_dm"] for _, res in valid])
    diff     = f_c_arr - f_dm_arr
    l1  = np.mean(np.abs(diff))     / (np.mean(np.abs(f_dm_arr)) + 1e-30)
    l2  = np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(f_dm_arr**2)) + 1e-30)
    li  = np.max(np.abs(diff))      / (np.max(np.abs(f_dm_arr)) + 1e-30)
    print(f"\n  Friction factor errors vs Dodge-Metzner:")
    print(f"    L1={l1*100:.3f}%   L2={l2*100:.3f}%   Linf={li*100:.3f}%")

    print("\n  Generating figures ...")
    _fig_u_transient(results)
    _fig_f_vs_Re(results)
    print("\n  BENCHMARK C COMPLETE")
