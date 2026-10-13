import argparse
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numba import get_num_threads

from fvc   import update_props, dt_vfc, step_vfc
from tau_w import compute_tau_w, reconstruct_cell, reconstruct_all
from data_and_postprocessing import radial_metrics

# ── Physical parameters ───────────────────────────────────────────────────────
R     = 1.0
L     = 1000.0
TAU_W = 5.0
rho_s = 1600.0
rho_w = 1000.0
chi   = 8e-10
g     = 9.81
P_IN  = 10e3 
P_OUT = 0.0

# ── Numerical parameters ──────────────────────────────────────────────────────
Nz    = 500
Nr    = 100
zeta  = 0.95
CFL   = 0.9
ALPHA = 0.9
REC   = 20

OUT_ROOT = "benchmarks_laminar"
os.makedirs(OUT_ROOT, exist_ok=True)

c0    = 1.0 / np.sqrt(rho_s * chi)
dz    = L / Nz
z     = np.linspace(0.5 * dz, L - 0.5 * dz, Nz)
R_hat = R * np.ones(Nz)
theta = np.zeros(Nz)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "legend.frameon": False,
    "legend.fontsize": 10, "xtick.direction": "in", "ytick.direction": "in",
})
COL_SIM = "#E87722"
COL_EX  = "black"


def _clean(ax):
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
    ax.grid(True)

def _inset(fig, rect):
    ax = fig.add_axes(rect)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8);  ax.grid(True, linewidth=0.3)
    return ax

def _zoom(ax, rect, xlim, ylim):
    axins = ax.inset_axes(rect)
    axins.tick_params(labelsize=7);  axins.grid(True, linewidth=0.3)
    axins.set_xlim(*xlim);  axins.set_ylim(*ylim)
    ax.indicate_inset_zoom(axins, edgecolor="0.45", linewidth=0.6, alpha=0.8)
    return axins

def _window(x, y, lo, hi, pad=0.15):
    m = (x >= lo) & (x <= hi)
    a, b = np.min(y[m]), np.max(y[m])
    d = pad * (b - a) if b > a else 1e-9
    return a - d, b + d

def relative_errors(arr, ref):
    d = arr - ref
    return (np.mean(np.abs(d))     / (np.mean(np.abs(ref))     + 1e-30),
            np.sqrt(np.mean(d**2)) / (np.sqrt(np.mean(ref**2)) + 1e-30),
            np.max(np.abs(d))      / (np.max(np.abs(ref))       + 1e-30))

def _scratch():
    nt = get_num_threads()
    return (np.empty((nt, Nr - 1)), np.empty((nt, Nr)),
            np.empty((nt, Nr - 1)), np.empty((nt, Nr)))


# ── FVC transient solver ──────────────────────────────────────────────────────
def _run_fvc(mu_N, tau_y, mu_s, n_hb, U_ss, r, dr, rmid, xi_mid,
             gml, wl, gm, ww, Tf):
    
    p_ref = P_OUT
    p_in  = P_IN

    p    = P_IN - (2.0 * TAU_W / R) * z
    p[0] = P_IN;  p[-1] = P_OUT
    u    = np.zeros(Nz)
    phi  = np.ones(Nz)
    ra   = np.empty(Nz);  ca = np.empty(Nz);  ma = np.empty(Nz)
    update_props(p, phi, p_ref, rho_w, rho_s, chi, chi, mu_N, mu_s, ra, ca, ma)
    tw = np.zeros(Nz)

    sl     = slice(Nz // 4, 3 * Nz // 4)
    t      = 0.0;  step = 0
    T_hist = [0.0];  U_hist = [0.0]
    t0     = time.perf_counter()

    while t < Tf:
        tw = compute_tau_w(u, phi, ra, R, mu_N, tau_y, mu_s, n_hb,
                           0.0, 0.0, r, dr, rmid, xi_mid, tw, gml, wl, gm, ww)
        dt = dt_vfc(p, u, phi, ra, ca, dz, CFL, t, Tf, ALPHA)
        p, u, phi, _ = step_vfc(p, u, phi, dz, dt, ra, ca, tw,
                                  p_in, ra[0], ca[0], 1.0, theta, R_hat, g, 0, 0.0,
                                  1, P_OUT)
        update_props(p, phi, p_ref, rho_w, rho_s, chi, chi, mu_N, mu_s, ra, ca, ma)
        t += dt;  step += 1
        if step % REC == 0:
            T_hist.append(t);  U_hist.append(float(np.mean(u[sl])))

    return dict(p=p, u=u, tw=tw, phi=phi, ra=ra,
                T=np.array(T_hist), U=np.array(U_hist),
                u_ss=float(np.mean(u[sl])), tw_ss=float(-np.mean(tw[sl])),
                cpu=time.perf_counter() - t0, steps=step, t_end=t)


def _reconstruct_midpipe(res, r, dr, rmid, xi_mid, mu_N, tau_y, mu_s, n_hb):
    tw  = float(np.mean(res["tw"][Nz // 4 : 3 * Nz // 4]))
    g1  = np.empty(Nr - 1);  w1 = np.empty(Nr)
    g2  = np.empty(Nr - 1);  w2 = np.empty(Nr)
    w_r = np.empty(Nr);      trz = np.empty(Nr)
    reconstruct_cell(tw, rho_s, 1.0, R, mu_N, tau_y, mu_s, n_hb,
                     0.0, 0.0, r, dr, rmid, xi_mid, w_r, trz, g1, w1, g2, w2)
    return w_r


def _reconstruct_2d(res, r, dr, rmid, xi_mid, mu_N, tau_y, mu_s, n_hb,
                    gml, wl, gm, ww, N_Z=80):
    idx = np.linspace(Nz // 10, 9 * Nz // 10, N_Z, dtype=int)
    W   = np.asfortranarray(np.zeros((r.shape[0], N_Z)))
    reconstruct_all(res["tw"][idx], res["ra"][idx], res["phi"][idx],
                    R, mu_N, tau_y, mu_s, n_hb, 0.0, 0.0,
                    r, dr, rmid, xi_mid, W, gml, wl, gm, ww)
    return W, z[idx]


# ── Shared figures ────────────────────────────────────────────────────────────
def _fig_axial_velocity(res, U_EX, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.13)
    ax.plot(z / 1000, res["u"], color=COL_SIM, lw=1.6, label="Simulated", zorder=3)
    ax.axhline(U_EX, color=COL_EX, lw=1.6, ls="--", label="Exact", zorder=4)
    ax.set_xlabel("z  (km)");  ax.set_ylabel("u  (m/s)")
    ax.set_ylim(U_EX * 0.992, U_EX * 1.016);  ax.legend(loc="upper right");  ax.grid(True)
    lo, hi = 0.075, 0.090
    m = (z / 1000 >= lo) & (z / 1000 <= hi)
    ua, ub = min(res["u"][m].min(), U_EX), max(res["u"][m].max(), U_EX)
    yl = (ua - 0.1 * (ub - ua), ub + 0.1 * (ub - ua))
    axins = _zoom(ax, [0.07, 0.50, 0.36, 0.38], (lo, hi), yl)
    axins.plot(z[m] / 1000, res["u"][m], color=COL_SIM, lw=1.2, marker="o", ms=2.5, zorder=3)
    axins.axhline(U_EX, color=COL_EX, lw=1.4, ls="--", zorder=4)
    fig.savefig(out_path, dpi=200);  plt.close(fig)


def _fig_pressure(res, P_EX, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.13)
    ax.plot(z / 1000, res["p"] / 1e3, color=COL_SIM, lw=1.6, label="Simulated", zorder=3)
    ax.plot(z / 1000, P_EX / 1e3, color=COL_EX, lw=1.6, ls="--", label="Exact", zorder=4)
    ax.set_xlabel("z  (km)");  ax.set_ylabel("p  (kPa)")
    ax.legend(loc="upper right");  ax.grid(True)
    lo, hi = 0.100, 0.105
    m = (z / 1000 >= lo) & (z / 1000 <= hi)
    yl = _window(z / 1000, np.minimum(res["p"], P_EX) / 1e3, lo, hi)
    yh = _window(z / 1000, np.maximum(res["p"], P_EX) / 1e3, lo, hi)
    axins = _zoom(ax, [0.04, 0.08, 0.38, 0.40], (lo, hi), (yl[0], yh[1]))
    axins.plot(z[m] / 1000, res["p"][m] / 1e3, color=COL_SIM, lw=1.2, marker="o", ms=2.5, zorder=3)
    axins.plot(z[m] / 1000, P_EX[m] / 1e3, color=COL_EX, lw=1.4, ls="--", zorder=4)
    fig.savefig(out_path, dpi=200);  plt.close(fig)


def _fig_error_2d(err, r, z_sel, plug_r_ratio, out_path):
    r_sym   = np.concatenate([-r[::-1], r])
    err_sym = np.concatenate([err[::-1, :], err], axis=0)
    vmin    = max(err[err > 0].min(), 1e-10) if (err > 0).any() else 1e-10
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.14)
    pcm = ax.pcolormesh(z_sel / 1000, r_sym / R, err_sym,
                        cmap="viridis", shading="auto",
                        norm=LogNorm(vmin=vmin, vmax=err.max()))
    fig.colorbar(pcm, ax=ax, pad=0.02).set_label("|w_FVC - w_exact|  (m/s)")
    if plug_r_ratio is not None:
        ax.axhline( plug_r_ratio, color="white", lw=1.2, ls="--",
                    label=f"Plug  Rp/R = {plug_r_ratio:.2f}")
        ax.axhline(-plug_r_ratio, color="white", lw=1.2, ls="--")
        ax.legend(loc="lower right", fontsize=9)
    ax.axhline(0, color="white", lw=0.4, ls=":")
    ax.set_xlabel("z  (km)");  ax.set_ylabel("r / R");  _clean(ax)
    fig.savefig(out_path, dpi=200);  plt.close(fig)


def _fig_error_metrics(p_err, u_err, w_err, out_path):
    x, wb = np.arange(3), 0.25
    labels = ["L1", "L2", r"L$\infty$"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.10)
    for bars, vals, lbl, col in [
            (None, p_err, "Pressure p",     "#4393c3"),
            (None, u_err, "Axial vel. u",   "#92c5de"),
            (None, w_err, "Radial vel. w(r)", COL_SIM)]:
        pass
    b1 = ax.bar(x - wb, np.array(p_err)*100, wb, label="Pressure p",
                color="#4393c3", edgecolor="k", linewidth=0.5)
    b2 = ax.bar(x,      np.array(u_err)*100, wb, label="Axial vel. u",
                color="#92c5de", edgecolor="k", linewidth=0.5)
    b3 = ax.bar(x + wb, np.array(w_err)*100, wb, label="Radial vel. w(r)",
                color=COL_SIM,   edgecolor="k", linewidth=0.5)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x);  ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Relative error  (%)");  ax.legend();  _clean(ax)
    fig.savefig(out_path, dpi=200);  plt.close(fig)


def _print_summary(label, name, U_EX, res, metrics):
    sep = "=" * 64
    print(sep);  print(f"  Benchmark {label} - {name}")
    print(f"  R={R} m  L={L} m  Nz={Nz}  Nr={Nr}  CFL={CFL}  chi={chi:.2e}")
    print(sep)
    print(f"  u_ss  = {res['u_ss']:.6f} m/s   (exact {U_EX:.6f})")
    print(f"  tw_ss = {res['tw_ss']:.5f} Pa    (target {TAU_W:.5f})")
    print(f"  t_end = {res['t_end']:.1f} s    steps={res['steps']}  CPU={res['cpu']:.1f} s")
    print(f"\n  {'Quantity':<16} {'L1 (%)':>9} {'L2 (%)':>9} {'Linf (%)':>9}")
    print("  " + "-" * 46)
    for key, lbl in [("pressure","Pressure p"),("axial_vel","Axial vel. u"),
                      ("radial_vel","Radial vel. w")]:
        e = metrics[key]
        print(f"  {lbl:<16} {e[0]*100:>9.4f} {e[1]*100:>9.4f} {e[2]*100:>9.4f}")


# ── Benchmark A ───────────────────────────────────────────────────────────────
class BenchmarkA:
    label = "A";  name = "Newtonian laminar (Hagen-Poiseuille)"
    out_dir = os.path.join(OUT_ROOT, "benchmark_A")
    MU = 10.0;  TAU_Y = 0.0;  N_HB = 1.0;  Tf = 125.5

    def __init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.r, self.dr, self.rmid, self.xi_mid = radial_metrics(R, Nr, zeta)
        self.gml, self.wl, self.gm, self.ww     = _scratch()
        self.U_EX = TAU_W * R / (4.0 * self.MU)
        self.P_EX = P_IN - (2.0 * TAU_W / R) * z

    def w_exact(self, r_arr):
        return (TAU_W * R / (2.0 * self.MU)) * (1.0 - r_arr**2 / R**2)

    def _plot_radial_velocity(self, w_rec):
        r_sym  = np.concatenate([-self.r[::-1], self.r])
        w_sym  = np.concatenate([w_rec[::-1],   w_rec])
        w_ex_s = np.concatenate([self.w_exact(self.r[::-1]), self.w_exact(self.r)])
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.12)
        ax.plot(r_sym/R, w_sym,  color=COL_SIM, lw=1.8, label="Simulated", zorder=3)
        ax.plot(r_sym/R, w_ex_s, color=COL_EX, lw=1.8, ls="--", label="Exact", zorder=4)
        ax.set_xlabel("r / R");  ax.set_ylabel("w  (m/s)")
        ax.legend(loc="upper right");  ax.grid(True)
        m = (r_sym/R >= 0.0) & (r_sym/R <= 0.1)
        yl = _window(r_sym/R, np.minimum(w_sym, w_ex_s), 0.0, 0.1)
        yh = _window(r_sym/R, np.maximum(w_sym, w_ex_s), 0.0, 0.1)
        axins = _zoom(ax, [0.30, 0.12, 0.32, 0.30], (0.0, 0.1), (yl[0], yh[1]))
        axins.plot(r_sym[m]/R, w_sym[m],  color=COL_SIM, lw=1.4, marker="o", ms=2.5, zorder=3)
        axins.plot(r_sym[m]/R, w_ex_s[m], color=COL_EX, lw=1.4, ls="--", zorder=4)
        path = os.path.join(self.out_dir, "fig5_radial_velocity.png")
        fig.savefig(path, dpi=200);  plt.close(fig);  return path

    def execute(self, replot=False):
        pkl = os.path.join(self.out_dir, "results.pkl")
        if replot and os.path.isfile(pkl):
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            res, w_rec = d["res"], d["w_rec"]
        else:
            res   = _run_fvc(self.MU, self.TAU_Y, self.MU, self.N_HB, self.U_EX,
                             self.r, self.dr, self.rmid, self.xi_mid,
                             self.gml, self.wl, self.gm, self.ww, self.Tf)
            w_rec = _reconstruct_midpipe(res, self.r, self.dr, self.rmid, self.xi_mid,
                                         self.MU, self.TAU_Y, self.MU, self.N_HB)
            with open(pkl, "wb") as f:
                pickle.dump({"res": res, "w_rec": w_rec}, f)

        W_2d, z_sel = _reconstruct_2d(res, self.r, self.dr, self.rmid, self.xi_mid,
                                       self.MU, self.TAU_Y, self.MU, self.N_HB,
                                       self.gml, self.wl, self.gm, self.ww)
        err2d = np.abs(W_2d - self.w_exact(self.r)[:, None])
        metrics = dict(pressure   = relative_errors(res["p"], self.P_EX),
                       axial_vel  = relative_errors(res["u"], np.full(Nz, self.U_EX)),
                       radial_vel = relative_errors(w_rec,    self.w_exact(self.r)))

        paths = [self._plot_radial_velocity(w_rec)]
        for fn, key in [("fig6_axial_velocity.png", "u"), ("fig7_pressure.png", "p")]:
            fp = os.path.join(self.out_dir, fn)
            if key == "u":
                _fig_axial_velocity(res, self.U_EX, fp)
            else:
                _fig_pressure(res, self.P_EX, fp)
            paths.append(fp)
        fp = os.path.join(self.out_dir, "fig8_velocity_error_2d.png")
        _fig_error_2d(err2d, self.r, z_sel, None, fp);  paths.append(fp)
        fp = os.path.join(self.out_dir, "fig9_error_metrics.png")
        _fig_error_metrics(metrics["pressure"], metrics["axial_vel"],
                           metrics["radial_vel"], fp);  paths.append(fp)
        return res, w_rec, metrics, paths


# ── Benchmark B ───────────────────────────────────────────────────────────────
class BenchmarkB:
    label = "B";  name = "Non-Newtonian laminar (Bingham / Buckingham-Reiner)"
    out_dir = os.path.join(OUT_ROOT, "benchmark_B")
    MU_B = 10.0;  MU_N = 10.0;  TAU_Y = 1.2;  N_HB = 1.0
    PHI  = TAU_Y / TAU_W;  R_P = PHI * R;  Tf = 112.5

    def __init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.r, self.dr, self.rmid, self.xi_mid = radial_metrics(R, Nr, zeta)
        self.gml, self.wl, self.gm, self.ww     = _scratch()
        self.U_EX = (TAU_W * R / (4.0 * self.MU_B)) * (
            1.0 - 4.0 / 3.0 * self.PHI + self.PHI**4 / 3.0)
        self.P_EX = P_IN - (2.0 * TAU_W / R) * z

    def w_exact(self, r_arr):
        w_p = (TAU_W * R / (2.0 * self.MU_B)) * (1.0 - self.PHI)**2
        w_s = (TAU_W * R / (2.0 * self.MU_B)) * (
            (1.0 - r_arr**2 / R**2) - 2.0 * self.PHI * (1.0 - r_arr / R))
        return np.where(r_arr <= self.R_P, w_p, w_s)

    def _plot_radial_velocity(self, w_rec):
        r_sym  = np.concatenate([-self.r[::-1], self.r])
        w_sym  = np.concatenate([w_rec[::-1],   w_rec])
        w_ex_s = np.concatenate([self.w_exact(self.r[::-1]), self.w_exact(self.r)])
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.12)
        ax.axvspan(-self.R_P/R, self.R_P/R, alpha=0.10, color="#aaaaaa",
                   label=f"Plug  |r/R| <= {self.R_P/R:.2f}")
        ax.plot(r_sym/R, w_sym,  color=COL_SIM, lw=1.8, label="Simulated", zorder=3)
        ax.plot(r_sym/R, w_ex_s, color=COL_EX, lw=1.8, ls="--", label="Exact", zorder=4)
        ax.set_xlabel("r / R");  ax.set_ylabel("w  (m/s)")
        ax.legend(loc="upper right");  ax.grid(True)
        x1 = self.R_P/R - 0.06;  x2 = self.R_P/R + 0.06
        m = (r_sym/R >= x1) & (r_sym/R <= x2)
        yl = _window(r_sym/R, np.minimum(w_sym, w_ex_s), x1, x2)
        yh = _window(r_sym/R, np.maximum(w_sym, w_ex_s), x1, x2)
        axins = _zoom(ax, [0.30, 0.12, 0.32, 0.30], (x1, x2), (yl[0], yh[1]))
        axins.plot(r_sym[m]/R, w_sym[m],  color=COL_SIM, lw=1.4, marker="o", ms=2.5, zorder=3)
        axins.plot(r_sym[m]/R, w_ex_s[m], color=COL_EX, lw=1.4, ls="--", zorder=4)
        path = os.path.join(self.out_dir, "fig10_radial_velocity.png")
        fig.savefig(path, dpi=200);  plt.close(fig);  return path

    def execute(self, replot=False):
        pkl = os.path.join(self.out_dir, "results.pkl")
        if replot and os.path.isfile(pkl):
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            res, w_rec = d["res"], d["w_rec"]
        else:
            res   = _run_fvc(self.MU_N, self.TAU_Y, self.MU_B, self.N_HB, self.U_EX,
                             self.r, self.dr, self.rmid, self.xi_mid,
                             self.gml, self.wl, self.gm, self.ww, self.Tf)
            w_rec = _reconstruct_midpipe(res, self.r, self.dr, self.rmid, self.xi_mid,
                                         self.MU_N, self.TAU_Y, self.MU_B, self.N_HB)
            with open(pkl, "wb") as f:
                pickle.dump({"res": res, "w_rec": w_rec}, f)

        W_2d, z_sel = _reconstruct_2d(res, self.r, self.dr, self.rmid, self.xi_mid,
                                       self.MU_N, self.TAU_Y, self.MU_B, self.N_HB,
                                       self.gml, self.wl, self.gm, self.ww)
        err2d = np.abs(W_2d - self.w_exact(self.r)[:, None])
        metrics = dict(pressure   = relative_errors(res["p"], self.P_EX),
                       axial_vel  = relative_errors(res["u"], np.full(Nz, self.U_EX)),
                       radial_vel = relative_errors(w_rec,    self.w_exact(self.r)))

        paths = [self._plot_radial_velocity(w_rec)]
        for fn, key in [("fig11_axial_velocity.png","u"),("fig12_pressure.png","p")]:
            fp = os.path.join(self.out_dir, fn)
            if key == "u":
                _fig_axial_velocity(res, self.U_EX, fp)
            else:
                _fig_pressure(res, self.P_EX, fp)
            paths.append(fp)
        fp = os.path.join(self.out_dir, "fig13_velocity_error_2d.png")
        _fig_error_2d(err2d, self.r, z_sel, self.R_P/R, fp);  paths.append(fp)
        fp = os.path.join(self.out_dir, "fig14_error_metrics.png")
        _fig_error_metrics(metrics["pressure"], metrics["axial_vel"],
                           metrics["radial_vel"], fp);  paths.append(fp)
        return res, w_rec, metrics, paths


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--bench", choices=["A", "B"], default=None)
    args = parser.parse_args()

    classes = ({"A": [BenchmarkA], "B": [BenchmarkB]}
               .get(args.bench, [BenchmarkA, BenchmarkB]))

    for cls in classes:
        bench = cls()
        res, w_rec, metrics, paths = bench.execute(replot=args.replot)
        _print_summary(bench.label, bench.name, bench.U_EX, res, metrics)
        print("\n  Figures written:")
        for p in paths:
            print(f"    {p}")
        print()
