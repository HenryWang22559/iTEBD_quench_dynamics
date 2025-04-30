# Full NumPy iTEBD Simulation: Ground State + Real Time Quench
import numpy as np
from scipy.linalg import expm, svd
import matplotlib.pyplot as plt
import time

# --- Parameters ---
chi_max = 30
J = 1.0              # Antiferromagnetic
# Initial state parameters (Ground state)
Hx_i = 5.0
Hz_i = 0.0
# Final state parameters (Quenched)
Hx_f = 2.0
Hz_f = 0.0
# Time steps
dtau = 0.01             # Imaginary time step
dt = 0.01           # Real time step
# Evolution steps
NstepsI = 200         # Imaginary time steps
NstepsR = 100         # Real time steps (Total time T = NstepsR * dt)
# Convergence criterion for ground state energy
CvgCrit = 1.0e-10
# Truncation tolerance for SVD in real time
trunc_err_tol = 1e-8

# --- Physical dimension ---
d = 2

# --- spin-1/2 Operators (NumPy) ---
sx = np.array([[0., 1.], [1., 0.]], dtype=complex)
sy = np.array([[0., -1j], [1j, 0.]], dtype=complex)
sz = np.array([[1., 0.], [0., -1.]], dtype=complex)
si = np.array([[1., 0.], [0., 1.]], dtype=complex)

# --- Hamiltonian Terms (NumPy) ---
H_int = J * np.kron(sz, sz) # Exchange interaction

def H_trans(Hx):
    return -Hx * 0.5 * (np.kron(sx, si) + np.kron(si, sx))

def H_long(Hz):
    return -Hz * 0.5 * (np.kron(sz, si) + np.kron(si, sz))

# --- Initial Hamiltonian and Imaginary Time Evolution Operator ---
H_bond_i = H_int + H_trans(Hx_i) + H_long(Hz_i)
H_bond_i_op = H_bond_i.reshape(d, d, d, d)
U_bond_dtau = expm(-dtau * H_bond_i).reshape(d, d, d, d)

print("Initial H and U_dtau created.")

# --- Final (Quenched) Hamiltonian and Real Time Evolution Operator ---
H_bond_f = H_int + H_trans(Hx_f) + H_long(Hz_f)
H_bond_f_op = H_bond_f.reshape(d, d, d, d)
U_bond_dt = expm(-1j * dt * H_bond_f).reshape(d, d, d, d)

print("Final H and U_dt created.")

def svd_truncate(theta, chi_max, tol=1e-12):
    """Performs SVD and truncates."""
    chi_l, d1, d2, chi_r = theta.shape
    theta_matrix = theta.transpose(0, 1, 2, 3).reshape(chi_l * d1, d2 * chi_r)
    try:
        U, S, Vh = svd(theta_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        print("SVD Warning: Did not converge. Adding noise.")
        noise = np.random.rand(*theta_matrix.shape).astype(complex) * 1e-10
        U, S, Vh = svd(theta_matrix + noise, full_matrices=False)

    threshold = tol * (S[0] if len(S)>0 and S[0] > 1e-16 else tol)
    indices_above_tol = np.where(S > threshold)[0]
    chi_new = 1 if len(indices_above_tol) == 0 else min(indices_above_tol[-1] + 1, chi_max)

    norm_S_sq = np.sum(S**2)
    trunc_err = np.sum(S[chi_new:]**2) / norm_S_sq if norm_S_sq > 1e-16 else 0.0

    S_trunc = S[:chi_new]
    U_trunc = U[:, :chi_new]
    Vh_trunc = Vh[:chi_new, :]

    norm = np.linalg.norm(S_trunc) if len(S_trunc) > 0 else 0.0
    lambda_new = S_trunc / norm if norm > 1e-16 else np.zeros_like(S_trunc)

    A_prime = U_trunc.reshape(chi_l, d1, chi_new)
    B_prime = (Vh_trunc / norm).reshape(chi_new, d2, chi_r) if norm > 1e-16 else Vh_trunc.reshape(chi_new, d2, chi_r)

    return A_prime, lambda_new, B_prime, trunc_err

def itebd_step(A, B, la, lb, U_bond, chi_max, tol=1e-12):
    """Performs one step of the iTEBD update (A-B link)."""
    la = la + 1e-16
    lb = lb + 1e-16

    # 1. Construct theta = lb @ A @ la @ B @ lb
    theta = np.tensordot(A, np.diag(la), axes=(-1, 0))
    theta = np.tensordot(theta, B, axes=(-1, 0))
    theta = np.tensordot(np.diag(lb), theta, axes=(1, 0))
    theta = np.tensordot(theta, np.diag(lb), axes=(-1, 0))

    # 2. Apply gate U_bond
    theta_evolved = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))
    theta_evolved = theta_evolved.transpose(2, 0, 1, 3)

    A_prime, la_new, B_prime, trunc_err = svd_truncate(theta_evolved, chi_max, tol)

    lb_inv = 1.0 / lb
    A_new = np.tensordot(np.diag(lb_inv), A_prime, axes=(-1, 0))
    B_new = np.tensordot(B_prime, np.diag(lb_inv), axes=(-1, 0))

    return A_new, B_new, la_new, trunc_err

def calculate_energy(A, B, la, lb, H_bond_op):
    """Calculates the energy expectation value <H> for a two-site unit cell."""
    la = la + 1e-16
    lb = lb + 1e-16
    theta = np.tensordot(A, np.diag(la), axes=(-1, 0))
    theta = np.tensordot(theta, B, axes=(-1, 0))

    H_theta = np.tensordot(H_bond_op, theta, axes=([2, 3], [1, 2]))
    H_theta = H_theta.transpose(2, 0, 1, 3)

    theta_conj = np.conj(theta)
    energy_numerator = np.tensordot(theta_conj, H_theta, axes=([0,1,2,3],[0,1,2,3]))
    norm_sq = np.tensordot(theta_conj, theta, axes=([0,1,2,3],[0,1,2,3]))

    energy = (energy_numerator / norm_sq).real if abs(norm_sq) > 1e-15 else np.inf
    return energy

def calculate_observables(A, B, la, lb, op):
    """Calculates the expectation value <O> for a single-site operator 'op'."""
    la_safe = la + 1e-16
    lb_safe = lb + 1e-16
    gamma_A = np.tensordot(np.diag(lb_safe), A, axes=(-1, 0))
    gamma_A = np.tensordot(gamma_A, np.diag(la_safe), axes=(-1, 0))

    op_gamma_A = np.tensordot(gamma_A, op, axes=(1, 1))
    op_gamma_A = op_gamma_A.transpose(0, 2, 1)

    gamma_A_conj = np.conj(gamma_A)
    exp_val = np.tensordot(gamma_A_conj, op_gamma_A, axes=([0,1,2],[0,1,2]))
    return exp_val.real

def calculate_entropy(la):
    """Calculates the Von Neumann entanglement entropy from singular values."""
    la_sq = np.abs(la)**2
    la_sq_safe = la_sq[la_sq > 1e-18]
    if len(la_sq_safe) == 0: return 0.0
    entropy = -np.sum(la_sq_safe * np.log(la_sq_safe))
    return entropy

# --- Initialize MPS (Random State) ---
chi_init = 1
A = np.random.rand(chi_init, d, chi_init).astype(complex)
B = np.random.rand(chi_init, d, chi_init).astype(complex)
A /= np.linalg.norm(A); B /= np.linalg.norm(B)
la = np.ones(chi_init) / np.sqrt(chi_init)
lb = np.ones(chi_init) / np.sqrt(chi_init)
print(f"Initial MPS shapes: A={A.shape}, B={B.shape}, la={la.shape}, lb={lb.shape}")

# --- Imaginary Time Evolution Loop ---
print("\n--- Starting Imaginary Time Evolution (Ground State Search) ---")
E_last = np.inf
trunc_errors_A_im = []
trunc_errors_B_im = []
bond_dims_la_im = []
bond_dims_lb_im = []
energies = []
t_start_imag = time.time()
for step in range(NstepsI):
    A, B, la, trunc_err_A = itebd_step(A, B, la, lb, U_bond_dtau, chi_max, tol=CvgCrit/10)
    trunc_errors_A_im.append(trunc_err_A)
    bond_dims_la_im.append(len(la))

    B, A, lb, trunc_err_B = itebd_step(B, A, lb, la, U_bond_dtau, chi_max, tol=CvgCrit/10)
    trunc_errors_B_im.append(trunc_err_B)
    bond_dims_lb_im.append(len(lb))

    current_energy = calculate_energy(A, B, la, lb, H_bond_i_op)
    energies.append(current_energy)
    energy_diff = abs(current_energy - E_last)

    if (step + 1) % 50 == 0: # Print less frequently
        print(f"Im Step {step+1}/{NstepsI}, E={current_energy:.10f}, dE={energy_diff:.2e}, "
              f"TrA={trunc_err_A:.2e}, TrB={trunc_err_B:.2e}, Chi=({len(la)},{len(lb)})")

    if energy_diff < CvgCrit:
        print(f"\n[Converged!] Ground state found after {step+1} steps.")
        print(f"Final Energy = {current_energy:.12f}")
        break
    E_last = current_energy

t_end_imag = time.time()
print(f"Imaginary time evolution took {t_end_imag - t_start_imag:.2f} seconds.")

A_gs = A.copy(); B_gs = B.copy(); la_gs = la.copy(); lb_gs = lb.copy()
print(f"Ground state energy: {energies[-1]:.12f}")

# --- Plotting Imaginary Time Convergence --- (Optional display)
# fig_im, axs_im = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
# axs_im[0].plot(range(1, len(energies) + 1), energies, 'o-', markersize=3)
# axs_im[0].set_ylabel("Energy <H>"); axs_im[0].set_title(f"iTEBD GS Search"); axs_im[0].grid(True)
# axs_im[1].plot(range(1, len(trunc_errors_A_im) + 1), trunc_errors_A_im, 'r.-', label='TrErr A')
# axs_im[1].plot(range(1, len(trunc_errors_B_im) + 1), trunc_errors_B_im, 'b.-', label='TrErr B')
# axs_im[1].set_ylabel("Trunc Err"); axs_im[1].set_yscale('log'); axs_im[1].legend(); axs_im[1].grid(True)
# axs_im[2].plot(range(1, len(bond_dims_la_im) + 1), bond_dims_la_im, 'r.-', label='Chi la')
# axs_im[2].plot(range(1, len(bond_dims_lb_im) + 1), bond_dims_lb_im, 'b.-', label='Chi lb')
# axs_im[2].set_ylabel("Bond Dim"); axs_im[2].set_xlabel("Im Step"); axs_im[2].axhline(chi_max, color='k', ls='--')
# axs_im[2].legend(); axs_im[2].grid(True)
# plt.tight_layout(); # plt.show()

# --- Real Time Evolution Loop ---
print("\n--- Starting Real Time Evolution ---")
A = A_gs.copy(); B = B_gs.copy(); la = la_gs.copy(); lb = lb_gs.copy()

times = np.arange(NstepsR + 1) * dt
Mx_t = np.zeros(NstepsR + 1)
Mz_t = np.zeros(NstepsR + 1)
Entropy_t = np.zeros(NstepsR + 1)
trunc_errors_A_rt = np.zeros(NstepsR)
trunc_errors_B_rt = np.zeros(NstepsR)
bond_dims_la_rt = np.zeros(NstepsR)
bond_dims_lb_rt = np.zeros(NstepsR)

Mx_t[0] = calculate_observables(A, B, la, lb, sx)
Mz_t[0] = calculate_observables(A, B, la, lb, sz)
Entropy_t[0] = calculate_entropy(la)

t_start_real = time.time()
for step in range(NstepsR):
    A, B, la, trunc_err_A = itebd_step(A, B, la, lb, U_bond_dt, chi_max, tol=trunc_err_tol)
    trunc_errors_A_rt[step] = trunc_err_A
    bond_dims_la_rt[step] = len(la)

    B, A, lb, trunc_err_B = itebd_step(B, A, lb, la, U_bond_dt, chi_max, tol=trunc_err_tol)
    trunc_errors_B_rt[step] = trunc_err_B
    bond_dims_lb_rt[step] = len(lb)

    Mx_t[step + 1] = calculate_observables(A, B, la, lb, sx)
    Mz_t[step + 1] = calculate_observables(A, B, la, lb, sz)
    Entropy_t[step + 1] = calculate_entropy(la)

    if (step + 1) % 20 == 0: # Print less frequently
        print(f"RT Step {step+1}/{NstepsR}, t={times[step+1]:.2f}, " \
              f"Mx={Mx_t[step+1]:.4f}, Mz={Mz_t[step+1]:.4f}, S={Entropy_t[step+1]:.4f}, " \
              f"TrA={trunc_err_A:.1e}, TrB={trunc_err_B:.1e}, Chi=({len(la)},{len(lb)})")

t_end_real = time.time()
print(f"Real time evolution took {t_end_real - t_start_real:.2f} seconds.")

# --- Plotting Real Time Results ---
print("\nPlotting real-time evolution results...")
fig_rt, axs_rt = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

axs_rt[0].plot(times, Mx_t, 'o-', markersize=3, label=r'$\langle S_x \rangle$')
axs_rt[0].plot(times, Mz_t, 's-', markersize=3, label=r'$\langle S_z \rangle$')
axs_rt[0].set_ylabel("Magnetization")
axs_rt[0].set_title(f"Real Time Evolution (Hx:{Hx_i}->{Hx_f}, Hz:{Hz_i}->{Hz_f}, J={J}, $\chi$={chi_max})")
axs_rt[0].legend(); axs_rt[0].grid(True)

axs_rt[1].plot(times, Entropy_t, 'o-', markersize=3, color='g')
axs_rt[1].set_ylabel("Entanglement Entropy S")
axs_rt[1].grid(True)

axs_rt[2].plot(times[1:], trunc_errors_A_rt, 'r.-', label='Trunc Err (A-B)')
axs_rt[2].plot(times[1:], trunc_errors_B_rt, 'b.-', label='Trunc Err (B-A)')
axs_rt[2].set_ylabel("Truncation Error"); axs_rt[2].set_yscale('log'); axs_rt[2].legend(); axs_rt[2].grid(True)

axs_rt[3].plot(times[1:], bond_dims_la_rt, 'r.-', label='Bond Dim (la)')
axs_rt[3].plot(times[1:], bond_dims_lb_rt, 'b.-', label='Bond Dim (lb)')
axs_rt[3].set_ylabel("Bond Dimension"); axs_rt[3].set_xlabel("Time (t)")
axs_rt[3].axhline(chi_max, color='k', linestyle='--', label=f'$\chi_{{max}}$={chi_max}')
axs_rt[3].legend(); axs_rt[3].grid(True)

plt.tight_layout()
plt.show()

print("\nFull simulation complete.")
