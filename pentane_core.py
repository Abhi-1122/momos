"""
pentane_core.py
All constants, geometry, energy, MC, MD, umbrella sampling, PMF, and entropy
for the n-pentane conformational sampling project (TraPPE-UA).
"""
import numpy as np

# ─────────────────────────── CELL 1 — Constants ───────────────────────────
kB = 1.380649e-23  # J/K

# TraPPE-UA LJ (SI)
eps_CH3 = 98.0 * kB;  sig_CH3 = 3.75e-10
eps_CH2 = 46.0 * kB;  sig_CH2 = 3.95e-10
eps_mix = np.sqrt(eps_CH3 * eps_CH2)
sig_mix = (sig_CH3 + sig_CH2) / 2.0

# Intramolecular
l_bond  = 1.54e-10
k_theta = 62500.0 * kB
theta_0 = np.radians(114.0)

# Bond-stretch (TraPPE-UA, Martin & Siepmann 1998)
l0_bond  = 1.54e-10          # m  (equilibrium C-C length)
k_bond   = 452900.0 * kB * 1e20   # J/m²  (force constant)

# Torsion coefficients (J)
c1 =  355.03 * kB
c2 =  -68.19 * kB
c3 =  791.32 * kB

# Simulation settings
N_STEPS   = 200_000
N_BINS    = 36
BIN_EDGES   = np.linspace(-np.pi, np.pi, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH   = BIN_EDGES[1] - BIN_EDGES[0]

# Masses
amu   = 1.6605e-27
M_CH3 = 15.035 * amu
M_CH2 = 14.027 * amu
MASSES = np.array([M_CH3, M_CH2, M_CH2, M_CH2, M_CH3])

J_TO_KJMOL = 6.02214076e23 / 1000.0

# Umbrella sampling
N_WINDOWS = 36
PHI_CENTERS = BIN_CENTERS.copy()
K_US = 2000.0 * kB
STEPS_PER_WINDOW = N_STEPS // N_WINDOWS

# ─────────────────────────── CELL 2 — Geometry ───────────────────────────
def dihedral_angle(p0, p1, p2, p3):
    b1 = p1 - p0;  b2 = p2 - p1;  b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))

def bond_angle(p0, p1, p2):
    v1 = p0 - p1;  v2 = p2 - p1
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_a, -1.0, 1.0))

def get_backbone_dihedral(coords):
    return dihedral_angle(coords[0], coords[1], coords[2], coords[3])

# ─────────────────────────── CELL 3 — Energy ─────────────────────────────
def U_LJ(r, eps, sig):
    sr6 = (sig / r)**6
    return 4.0 * eps * (sr6**2 - sr6)

def U_nonbonded(coords):
    pairs = [(0,3,eps_mix,sig_mix),(0,4,eps_CH3,sig_CH3),(1,4,eps_mix,sig_mix)]
    E = 0.0
    for i, j, ep, si in pairs:
        r = np.linalg.norm(coords[i] - coords[j])
        E += U_LJ(r, ep, si)
    return E

def U_angle_energy(coords):
    triples = [(0,1,2),(1,2,3),(2,3,4)]
    E = 0.0
    for i, j, k in triples:
        theta = bond_angle(coords[i], coords[j], coords[k])
        E += 0.5 * k_theta * (theta - theta_0)**2
    return E

def U_bond(coords):
    """
    Harmonic bond-stretch energy over all 4 C-C bonds.
    coords : (5,3) array in metres
    Returns energy in Joules.
    """
    E = 0.0
    for i in range(4):
        r = np.linalg.norm(coords[i+1] - coords[i])
        E += 0.5 * k_bond * (r - l0_bond)**2
    return E

def U_torsion(coords):
    quads = [(0,1,2,3),(1,2,3,4)]
    E = 0.0
    for i, j, k, l in quads:
        phi = dihedral_angle(coords[i], coords[j], coords[k], coords[l])
        E += c1*(1+np.cos(phi)) + c2*(1-np.cos(2*phi)) + c3*(1+np.cos(3*phi))
    return E

def U_total(coords):
    return U_nonbonded(coords) + U_bond(coords) + U_angle_energy(coords) + U_torsion(coords)

def torsion_single(phi):
    """Single-dihedral torsion energy (for plotting the profile)."""
    return c1*(1+np.cos(phi)) + c2*(1-np.cos(2*phi)) + c3*(1+np.cos(3*phi))

# ─────────────────────────── CELL 4 — Build geometry ─────────────────────
def build_pentane_trans():
    """
    All-trans pentane: C0, C2, C4 on x-axis; C1, C3 offset in +y.
    Bond length = 1.54 Å, bond angle = 114°, dihedral = 180° (trans).
    
    For bond angle θ at atom 1: vectors (atom0-atom1) and (atom2-atom1)
    must subtend θ. With even atoms on y=0 and odd atoms at y=h:
      h² = l² (1 + cos θ) / 2
      a² = l² (1 - cos θ) / 2     (a = half x-spacing of even atoms)
    """
    coords = np.zeros((5, 3))
    ct = np.cos(theta_0)
    h = l_bond * np.sqrt((1.0 + ct) / 2.0)   # y-offset of odd atoms
    a = l_bond * np.sqrt((1.0 - ct) / 2.0)    # half x-spacing
    
    # Even atoms (0, 2, 4) on x-axis at y=0
    # Odd atoms  (1, 3)    at y=h
    coords[0] = [0.0,   0.0, 0.0]
    coords[1] = [a,     h,   0.0]
    coords[2] = [2*a,   0.0, 0.0]
    coords[3] = [3*a,   h,   0.0]
    coords[4] = [4*a,   0.0, 0.0]
    return coords

def _rotation_matrix(axis, angle):
    """Rodrigues rotation matrix for rotation around `axis` by `angle` radians."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)

def build_coords_at_phi(phi_target):
    """
    Build pentane with backbone dihedral 0-1-2-3 near phi_target.
    Start from all-trans, rotate atom 0 around the 1->2 bond axis.
    """
    coords = build_pentane_trans()
    phi_current = get_backbone_dihedral(coords)
    delta = phi_target - phi_current
    # Rotate atom 0 around bond axis 1->2 by delta
    axis = coords[2] - coords[1]
    R = _rotation_matrix(axis, delta)
    coords[0] = coords[1] + R @ (coords[0] - coords[1])
    return coords

# ─────────────────────────── CELL 5 — MC Engine ──────────────────────────
def run_MC(coords_init, T, n_steps=N_STEPS, move_size_ang=0.10):
    move_size = move_size_ang * 1e-10
    beta = 1.0 / (kB * T)
    coords = coords_init.copy()
    E_curr = U_total(coords)
    dihedrals = np.zeros(n_steps)
    energies  = np.zeros(n_steps)
    bond_E_traj = []
    n_accept  = 0
    for step in range(n_steps):
        i = np.random.randint(5)
        delta = (np.random.rand(3) - 0.5) * 2.0 * move_size
        coords_new = coords.copy()
        coords_new[i] += delta
        E_new = U_total(coords_new)
        dE = E_new - E_curr
        if dE < 0.0 or np.random.rand() < np.exp(-beta * dE):
            coords = coords_new; E_curr = E_new; n_accept += 1
        dihedrals[step] = get_backbone_dihedral(coords)
        energies[step]  = E_curr
        bond_E_traj.append(U_bond(coords))
    return dihedrals, energies, n_accept / n_steps, bond_E_traj

# ─────────────────────────── CELL 6 — MD Engine ──────────────────────────
def compute_forces(coords):
    forces = np.zeros((5, 3))
    dx = 1e-13
    for i in range(5):
        for j in range(3):
            cp = coords.copy(); cp[i,j] += dx
            cm = coords.copy(); cm[i,j] -= dx
            forces[i,j] = -(U_total(cp) - U_total(cm)) / (2.0 * dx)
    return forces

def run_MD(coords_init, T, n_steps=N_STEPS, dt=2e-15, rescale_every=50):
    coords = coords_init.copy()
    n_atoms = 5
    ndof = 3*n_atoms - 3  # 12
    # Maxwell-Boltzmann velocities
    vel = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        vel[i] = np.random.normal(0, np.sqrt(kB*T/MASSES[i]), 3)
    # Remove COM drift
    total_mass = MASSES.sum()
    v_com = np.sum(MASSES[:,None]*vel, axis=0) / total_mass
    vel -= v_com[None,:]

    dihedrals = np.zeros(n_steps)
    energies  = np.zeros(n_steps)
    T_traj    = np.zeros(n_steps)
    lj_E_traj = []
    bond_E_traj = []
    angle_E_traj = []
    tors_E_traj = []
    forces    = compute_forces(coords)

    for step in range(n_steps):
        # Velocity-Verlet
        acc = forces / MASSES[:,None]
        coords = coords + vel*dt + 0.5*acc*dt**2
        forces_new = compute_forces(coords)
        acc_new = forces_new / MASSES[:,None]
        vel = vel + 0.5*(acc + acc_new)*dt
        forces = forces_new
        # Thermostat
        if (step+1) % rescale_every == 0:
            KE = 0.5 * np.sum(MASSES[:,None] * vel**2)
            T_inst = 2.0*KE / (ndof * kB)
            if T_inst > 0:
                vel *= np.sqrt(T / T_inst)
        KE = 0.5 * np.sum(MASSES[:,None] * vel**2)
        T_traj[step] = 2.0*KE / (ndof * kB)
        dihedrals[step] = get_backbone_dihedral(coords)
        energies[step]  = U_total(coords)
        lj_E_traj.append(U_nonbonded(coords))
        bond_E_traj.append(U_bond(coords))
        angle_E_traj.append(U_angle_energy(coords))
        tors_E_traj.append(U_torsion(coords))

    return dihedrals, energies, T_traj, lj_E_traj, bond_E_traj, angle_E_traj, tors_E_traj

# ─────────────────────── CELL 8/9 — Umbrella Sampling ────────────────────
def U_biased(coords, phi_0, k_us):
    phi = get_backbone_dihedral(coords)
    d_phi = phi - phi_0
    d_phi = (d_phi + np.pi) % (2.0*np.pi) - np.pi
    return U_total(coords) + 0.5*k_us*d_phi**2

def run_US_window(coords_init, T, phi_0, k_us,
                  n_steps=STEPS_PER_WINDOW, move_size_ang=0.10):
    move_size = move_size_ang * 1e-10
    beta = 1.0 / (kB * T)
    coords = coords_init.copy()
    E_curr = U_biased(coords, phi_0, k_us)
    dihedrals = np.zeros(n_steps)
    n_accept  = 0
    for step in range(n_steps):
        i = np.random.randint(5)
        delta = (np.random.rand(3) - 0.5) * 2.0 * move_size
        coords_new = coords.copy()
        coords_new[i] += delta
        E_new = U_biased(coords_new, phi_0, k_us)
        dE = E_new - E_curr
        if dE < 0.0 or np.random.rand() < np.exp(-beta * dE):
            coords = coords_new; E_curr = E_new; n_accept += 1
        dihedrals[step] = get_backbone_dihedral(coords)
    return dihedrals, n_accept / n_steps

# ─────────────────────── CELL 10 — Combine histograms ────────────────────
def make_histogram(dihedrals, n_bins=N_BINS):
    counts, _ = np.histogram(dihedrals, bins=BIN_EDGES)
    total = counts.sum()
    P = counts / total if total > 0 else counts.astype(float)
    return counts, P

def combine_US_histograms(all_window_dihedrals):
    combined = np.concatenate(all_window_dihedrals)
    counts, _ = np.histogram(combined, bins=BIN_EDGES)
    P = counts / counts.sum()
    return counts, P, combined

def wham(window_hists, window_centers, K_US_val, T,
         bin_centers, n_iter=50000, tol=1e-8):
    """
    Self-consistent WHAM to recover unbiased P(phi) from umbrella windows.

    Parameters
    ----------
    window_hists   : list of 1-D arrays, shape (N_BINS,), raw counts per window
    window_centers : 1-D array, shape (N_windows,), phi_0 values in radians
    K_US_val       : float, spring constant in J/rad^2
    T              : float, temperature in K
    bin_centers    : 1-D array, shape (N_BINS,), phi bin centres in radians
    n_iter         : int, max iterations
    tol            : float, convergence threshold on max |delta f|

    Returns
    -------
    P_u  : 1-D array, shape (N_BINS,), normalised unbiased probability
    F_kJ : 1-D array, shape (N_BINS,), PMF in kJ/mol (shifted so min = 0)
    """
    beta   = 1.0 / (kB * T)
    N_w    = len(window_hists)
    N_i    = np.array([h.sum() for h in window_hists], dtype=float)
    H_tot  = np.sum(window_hists, axis=0).astype(float)   # shape (N_BINS,)

    # Bias energies: shape (N_w, N_BINS)
    dphi = (bin_centers[None, :] - window_centers[:, None] + np.pi) \
           % (2 * np.pi) - np.pi
    U_b  = 0.5 * K_US_val * dphi**2                           # J/rad^2 * rad^2 = J

    # Initialise free energy offsets
    f = np.zeros(N_w)

    for iteration in range(n_iter):
        # WHAM denominator: shape (N_BINS,)
        denom = np.sum(N_i[:, None] * np.exp(f[:, None] - beta * U_b), axis=0)
        # Avoid division by zero
        denom = np.where(denom > 0, denom, np.inf)

        P_u = H_tot / denom
        norm = P_u.sum()
        if norm > 0:
            P_u /= norm

        # Update f_i
        f_new = -np.log(
            np.maximum(
                np.sum(P_u[None, :] * np.exp(-beta * U_b), axis=1),
                1e-300
            )
        )
        # Remove gauge freedom: fix f[0] = 0
        f_new -= f_new[0]

        if np.max(np.abs(f_new - f)) < tol:
            break
        f = f_new

    # Shift P_u so it is properly normalised
    P_u = np.where(P_u > 0, P_u, 1e-300)
    F_J  = -kB * T * np.log(P_u)
    F_J  -= F_J.min()
    F_kJ = F_J * J_TO_KJMOL      # convert to kJ/mol

    return P_u, F_kJ

# ─────────────────────── CELL 12 — PMF ───────────────────────────────────
def compute_PMF(P, T):
    with np.errstate(divide='ignore', invalid='ignore'):
        F = -kB * T * np.log(np.where(P > 0, P, np.nan))
    F -= np.nanmin(F)
    return F, F * J_TO_KJMOL

# ─────────────────────── CELL 13 — Entropy ───────────────────────────────
def exploration_entropy(P):
    P_nz = P[P > 0]
    return -np.sum(P_nz * np.log(P_nz))

def running_entropy(dihedrals):
    n = len(dihedrals)
    counts = np.zeros(N_BINS, dtype=float)
    S_traj = np.zeros(n)
    for t in range(n):
        b = np.searchsorted(BIN_EDGES, dihedrals[t], side='right') - 1
        b = np.clip(b, 0, N_BINS - 1)
        counts[b] += 1.0
        total = counts.sum()
        if total > 0:
            P = counts / total
            P_nz = P[P > 0]
            S_traj[t] = -np.sum(P_nz * np.log(P_nz))
    return S_traj, np.mean(S_traj)

def count_barrier_crossings(dihedrals, threshold_deg=60.0):
    threshold = np.radians(threshold_deg)
    dphi = np.diff(dihedrals)
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    return int(np.sum(np.abs(dphi) > threshold))
