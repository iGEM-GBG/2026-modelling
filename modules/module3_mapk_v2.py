"""
Module 3 v2: MAPK Cascade (Ste20 -> Ste11 -> Ste7 -> Fus3)
==============================================================
SATURATING (Michaelis-Menten / Goldbeter-Koshland-style) kinetics.
Same topology, same state vector, same rate constants and total pools
as v1 -- the ONLY thing that changed is the functional FORM of each
activation/deactivation term (v1 was pure mass action). This isolates
the kinetics-form change as the only variable between v1 and v2.

    G_betagamma + Ste20   --k_activate_Ste20-->  Ste20*
    Ste20*      + Ste11   --k_activate_Ste11-->  Ste11*
    Ste11*      + Ste7    --k_activate_Ste7-->   Ste7*
    Ste7*       + Fus3    --k_activate_Fus3-->   Fus3*

    Ste20* --k_deactivate_Ste20--> Ste20   (etc. for each tier)

Conservation laws (one per tier, unchanged from v1):
    Ste20_total = Ste20 + Ste20*
    Ste11_total = Ste11 + Ste11*
    Ste7_total  = Ste7  + Ste7*
    Fus3_total  = Fus3  + Fus3*

State vector: y = [Ste20_active, Ste11_active, Ste7_active, Fus3_active]

Output to Module 4: [Fus3*](t)  -->  drives Ste12 activation

Chain: Module 2 [Gbetagamma](t) --> text file --> interp1d --> input here

WHAT CHANGED FROM v1 -- THE KINETICS FORM
-------------------------------------------
v1 activation term:  k_act * driver * (pool_total - pool_active)
v2 activation term:  k_act * driver * (pool_total - pool_active)
                                       ----------------------------
                                       Km_act + (pool_total - pool_active)

Km wraps the SUBSTRATE (the free/inactive pool being converted), NOT
the upstream driver/kinase -- the driver stays a linear multiplier
outside the fraction (standard Michaelis-Menten theory: Km reflects
substrate saturation of the enzyme's capacity, not the enzyme's own
concentration). Km_activate_X = Km_deactivate_X = 0.1 x X_total for
every tier, chosen SMALL relative to the pool deliberately -- this is
what produces zero-order ultrasensitivity (Goldbeter & Koshland 1981).
Km >> pool would reduce algebraically back to something close to v1's
mass action, defeating the point of this upgrade.

Unit-correct rescaling: v1's k_activate has units nM^-1 s^-1 (it
multiplies two concentrations). In v2, the saturating fraction is
dimensionless, so the activation constant needs units s^-1 instead.
Symmetrically, v1's k_deactivate (s^-1) becomes a true Vmax with units
nM/s. The fix is to multiply each v1 constant by that tier's total
pool -- this keeps v2's maximum flux equal to v1's maximum flux,
isolating the kinetics-form change as the only variable.

MODELING STATUS (unchanged from v1)
-------------------------------------
* Ste5 scaffold is NOT modeled explicitly (folded into rate constants).
* Msg5/Ptp2/Ptp3 phosphatase feedback on Fus3 is NOT modeled.

Authors : iGEM team Gothenburg
Date    : 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ──────────────────────────────────────────────────────────────
# INTERMEDIATE I/O
# ──────────────────────────────────────────────────────────────
# v2 reads the SAME Module 2 output as v1. Writes to its OWN output
# files (module3_v2_output_*) so v1 and v2 results can coexist.
INTERMEDIATE_DIR = "../intermediate"
FIGURE_DIR = "../figures"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_module2_output(L_nM):
    """Load [Gbg](t) written by module2_gprotein.py for a given [L]."""
    path = os.path.join(INTERMEDIATE_DIR, f"module2_output_L{L_nM:.1f}nM.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run module1_binding.py then "
            f"module2_gprotein.py first to generate this module's "
            f"input for [L] = {L_nM} nM."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 3]   # t, Gbg (column 3 -- see module2 header)

# ──────────────────────────────────────────────────────────────
# PARAMETERS -- totals and rate constants, see module3_mapk.py (v1)
# for full sourcing (Kofahl & Klipp 2004, Table 2 model values)
# ──────────────────────────────────────────────────────────────
Ste20_total = 153.0   # nM
Ste11_total = 60.6    # nM
Ste7_total  = 58.0    # nM
Fus3_total  = 189.8   # nM

k_activate_Ste20   = 0.083   # nM^-1 s^-1
k_deactivate_Ste20 = 0.017   # s^-1
k_activate_Ste11   = 0.167   # nM^-1 s^-1
k_deactivate_Ste11 = 0.083   # s^-1
k_activate_Ste7    = 0.783   # nM^-1 s^-1
k_deactivate_Ste7  = 0.083   # s^-1
k_activate_Fus3    = 5.75    # nM^-1 s^-1
k_deactivate_Fus3  = 0.833   # s^-1

# Unit-correct rescaling for the Michaelis-Menten form (see docstring).
k_activate_Ste20_v2   = k_activate_Ste20   * Ste20_total   # s^-1
k_deactivate_Ste20_v2 = k_deactivate_Ste20 * Ste20_total   # nM/s
k_activate_Ste11_v2   = k_activate_Ste11   * Ste11_total   # s^-1
k_deactivate_Ste11_v2 = k_deactivate_Ste11 * Ste11_total   # nM/s
k_activate_Ste7_v2    = k_activate_Ste7    * Ste7_total    # s^-1
k_deactivate_Ste7_v2  = k_deactivate_Ste7  * Ste7_total    # nM/s
k_activate_Fus3_v2    = k_activate_Fus3    * Fus3_total    # s^-1
k_deactivate_Fus3_v2  = k_deactivate_Fus3  * Fus3_total    # nM/s

# Km values: designed to guarantee ultrasensitivity (0.1x pool), not a
# measured value -- see docstring.
Km_activate_Ste20   = 0.1 * Ste20_total
Km_deactivate_Ste20 = 0.1 * Ste20_total
Km_activate_Ste11   = 0.1 * Ste11_total
Km_deactivate_Ste11 = 0.1 * Ste11_total
Km_activate_Ste7    = 0.1 * Ste7_total
Km_deactivate_Ste7  = 0.1 * Ste7_total
Km_activate_Fus3    = 0.1 * Fus3_total
Km_deactivate_Fus3  = 0.1 * Fus3_total

# ──────────────────────────────────────────────────────────────
# SATURATING RATE-LAW HELPERS
# ──────────────────────────────────────────────────────────────
def mm_activate(rate_const, driver, free_substrate, Km):
    """Saturating activation: driver (linear) acting on the free/inactive
    substrate pool (wrapped in the Km fraction). Clamped >= 0 to guard
    against solve_ivp overshooting past the tier's physical bounds."""
    free_substrate = max(free_substrate, 0.0)
    return rate_const * driver * free_substrate / (Km + free_substrate)

def mm_deactivate(rate_const, active_substrate, Km):
    """Saturating deactivation: phosphatase/turnover acting on the
    active substrate pool. Same clamp, same reasoning as mm_activate."""
    active_substrate = max(active_substrate, 0.0)
    return rate_const * active_substrate / (Km + active_substrate)

# ──────────────────────────────────────────────────────────────
# ODE DEFINITION
# ──────────────────────────────────────────────────────────────
def mapk_ode_v2(t, y, k_act20, k_deact20, k_act11, k_deact11,
                 k_act7, k_deact7, k_actF, k_deactF,
                 Km_act20, Km_deact20, Km_act11, Km_deact11,
                 Km_act7, Km_deact7, Km_actF, Km_deactF,
                 Ste20_tot, Ste11_tot, Ste7_tot, Fus3_tot, Gbg_interp):
    """
    Saturating (Michaelis-Menten) MAPK cascade ODEs -- v2.

    State vector:
        y[0] = Ste20_active
        y[1] = Ste11_active
        y[2] = Ste7_active
        y[3] = Fus3_active
    """
    Ste20a, Ste11a, Ste7a, Fus3a = y

    Gbg = max(float(Gbg_interp(t)), 0.0)   # numerical safety clamp

    dSte20a_dt = (mm_activate(k_act20, Gbg,    Ste20_tot - Ste20a, Km_act20)
                  - mm_deactivate(k_deact20, Ste20a, Km_deact20))
    dSte11a_dt = (mm_activate(k_act11, Ste20a, Ste11_tot - Ste11a, Km_act11)
                  - mm_deactivate(k_deact11, Ste11a, Km_deact11))
    dSte7a_dt  = (mm_activate(k_act7,  Ste11a, Ste7_tot  - Ste7a,  Km_act7)
                  - mm_deactivate(k_deact7,  Ste7a,  Km_deact7))
    dFus3a_dt  = (mm_activate(k_actF,  Ste7a,  Fus3_tot  - Fus3a,  Km_actF)
                  - mm_deactivate(k_deactF,  Fus3a,  Km_deactF))

    return [dSte20a_dt, dSte11a_dt, dSte7a_dt, dFus3a_dt]

def dense_early_t_eval(t_start, t_end, dense_end=300, n_dense=3000, n_sparse=500):
    """Time grid concentrated in [t_start, dense_end] (where the real
    kinetics happen -- see design discussion: Module 2's own G-protein
    activation is the rate-limiting step, settling over ~1-2 min, not
    the much faster downstream MAPK relay). A uniform grid spread over
    the full multi-hour window is too coarse near t=0 to resolve this;
    this concatenates a dense early segment with a sparse late segment
    that just confirms the long-term plateau.
    """
    t_dense = np.linspace(t_start, dense_end, n_dense)
    t_sparse = np.linspace(dense_end, t_end, n_sparse)[1:]
    return np.concatenate([t_dense, t_sparse])

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
t_start  = 0
t_end    = 10800        # 1 hour, matching v1

L_values_nM = [1e5]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

y0 = [0.0, 0.0, 0.0, 0.0]   # zero basal activity (unchanged from v1;
                            # still valid: Gbg=0 at t=0 collapses the
                            # whole cascade to 0 regardless of kinetics
                            # form -- see design discussion)

results_m3 = {}

for L in L_values_nM:
    t_m2, Gbg_m2 = load_module2_output(L)
    Gbg_interp = interp1d(t_m2, Gbg_m2, kind='cubic', fill_value='extrapolate')

    t_eval = dense_early_t_eval(t_start, t_end)
    sol = solve_ivp(
        fun    = mapk_ode_v2,
        t_span = (t_start, t_end),
        y0     = y0,
        t_eval = t_eval,
        args   = (k_activate_Ste20_v2, k_deactivate_Ste20_v2,
                  k_activate_Ste11_v2, k_deactivate_Ste11_v2,
                  k_activate_Ste7_v2,  k_deactivate_Ste7_v2,
                  k_activate_Fus3_v2,  k_deactivate_Fus3_v2,
                  Km_activate_Ste20, Km_deactivate_Ste20,
                  Km_activate_Ste11, Km_deactivate_Ste11,
                  Km_activate_Ste7,  Km_deactivate_Ste7,
                  Km_activate_Fus3,  Km_deactivate_Fus3,
                  Ste20_total, Ste11_total, Ste7_total, Fus3_total,
                  Gbg_interp),
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    results_m3[L] = sol

    out_path = os.path.join(INTERMEDIATE_DIR, f"module3_v2_output_L{L:.1f}nM.txt")
    header = (
        "Module 3 v2 output -- MAPK cascade (saturating kinetics)\n"
        f"L = {L} nM ; Ste20_total = {Ste20_total} nM ; "
        f"Ste11_total = {Ste11_total} nM ; Ste7_total = {Ste7_total} nM ; "
        f"Fus3_total = {Fus3_total} nM ; Km = 0.1 x total per tier\n"
        "columns: time_s   Ste20_active_nM   Ste11_active_nM   "
        "Ste7_active_nM   Fus3_active_nM"
    )
    np.savetxt(out_path,
               np.column_stack([sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]]),
               header=header, fmt="%.6e")

L_demo = L_values_nM[0]

# ──────────────────────────────────────────────────────────────
# PLOTTING -- each panel saved as its own PDF
# ──────────────────────────────────────────────────────────────

# ── A: Fus3*(t), v2 ───────────────────────────────────────────
figA, axA = plt.subplots(figsize=(6.5, 5))
for (L, sol), c in zip(results_m3.items(), colors):
    axA.plot(sol.t / 60, sol.y[3], color=c, lw=2, label=f"[L] = {L:.0f} nM")
axA.set_xlabel("Time (min)")
axA.set_ylabel("[Fus3*] (nM)")
axA.set_title("Active Fus3 over time (v2, saturating)\n(signal input to Module 4)")
axA.legend(fontsize=9, loc='upper left')
axA.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axA.set_ylim(bottom=0)
figA.tight_layout()
figA.savefig(os.path.join(FIGURE_DIR, "module3v2_A_Fus3_timecourse.pdf"))
plt.close(figA)

# ── B: All four tiers at the simulated dose ───────────────────
figB, axB = plt.subplots(figsize=(6.5, 5))
sol_demo = results_m3[L_demo]
t_min = sol_demo.t / 60
axB.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='Ste20*')
axB.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='Ste11*')
axB.plot(t_min, sol_demo.y[2], lw=2, color='forestgreen', label='Ste7*')
axB.plot(t_min, sol_demo.y[3], lw=2, color='crimson',     label='Fus3*')
axB.set_xlabel("Time (min)")
axB.set_ylabel("Active concentration (nM)")
axB.set_title(f"All four tiers (v2)\n([L] = {L_demo:.0f} nM, wet-lab dose)")
axB.legend(fontsize=9)
axB.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axB.set_ylim(bottom=0)
figB.tight_layout()
figB.savefig(os.path.join(FIGURE_DIR, "module3v2_B_all_tiers.pdf"))
plt.close(figB)

# ── C: Km/pool ratio sensitivity ──────────────────────────────
figC, axC = plt.subplots(figsize=(6.5, 5))
t_m2, Gbg_m2 = load_module2_output(L_demo)
Gbg_interp_fixed = interp1d(t_m2, Gbg_m2, kind='cubic', fill_value='extrapolate')
t_eval = dense_early_t_eval(t_start, t_end)

Km_variants = {
    '0.01x total (very sharp)':   0.01,
    '0.1x total (this script)':   0.1,
    '1x total (mild saturation)': 1.0,
    '10x total (~v1 mass action)': 10.0,
}
variant_colors = ['#d62728', '#2ca02c', '#7f7f7f', '#1f77b4']

for (label, km_frac), c in zip(Km_variants.items(), variant_colors):
    sol_v = solve_ivp(
        mapk_ode_v2, (t_start, t_end), y0, t_eval=t_eval,
        args=(k_activate_Ste20_v2, k_deactivate_Ste20_v2,
              k_activate_Ste11_v2, k_deactivate_Ste11_v2,
              k_activate_Ste7_v2,  k_deactivate_Ste7_v2,
              k_activate_Fus3_v2,  k_deactivate_Fus3_v2,
              km_frac * Ste20_total, km_frac * Ste20_total,
              km_frac * Ste11_total, km_frac * Ste11_total,
              km_frac * Ste7_total,  km_frac * Ste7_total,
              km_frac * Fus3_total,  km_frac * Fus3_total,
              Ste20_total, Ste11_total, Ste7_total, Fus3_total,
              Gbg_interp_fixed),
        method='LSODA', rtol=1e-8, atol=1e-10
    )
    axC.plot(sol_v.t / 60, sol_v.y[3], lw=2, color=c, label=label)

axC.set_xlabel("Time (min)")
axC.set_ylabel("[Fus3*] (nM)")
axC.set_title(f"Km/pool ratio sensitivity\n[L] = {L_demo:.0f} nM")
axC.legend(fontsize=9)
axC.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axC.set_ylim(bottom=0)
figC.tight_layout()
figC.savefig(os.path.join(FIGURE_DIR, "module3v2_C_Km_sensitivity.pdf"))
plt.close(figC)

print("Figures saved: module3v2_{A_Fus3_timecourse,B_all_tiers,"
      "C_Km_sensitivity}.pdf")

# ──────────────────────────────────────────────────────────────
# STEADY-STATE SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n-- Steady-state summary (v2, saturating kinetics) -----------------")
print(f"{'[L] (nM)':>10} {'[Ste20*]_ss':>12} {'[Ste11*]_ss':>12} "
      f"{'[Ste7*]_ss':>12} {'[Fus3*]_ss':>12} {'% of Fus3_tot':>14}")
print("-" * 78)
for L, sol in results_m3.items():
    s20 = sol.y[0][-1]
    s11 = sol.y[1][-1]
    s7  = sol.y[2][-1]
    f3  = sol.y[3][-1]
    pct = 100 * f3 / Fus3_total
    print(f"{L:>10.1f} {s20:>12.3f} {s11:>12.3f} {s7:>12.3f} {f3:>12.3f} {pct:>13.1f}%")
