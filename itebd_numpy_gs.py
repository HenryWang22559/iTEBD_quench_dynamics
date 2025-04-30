# Combined NumPy iTEBD Ground State Search Code
import numpy as np
from scipy.linalg import expm, svd
import matplotlib.pyplot as plt
import time

# --- Parameters ---
chi_max = 30
J = 1.0
Hx_i = 1.0
Hz_i = 0.0
Hx_f = 0.0
Hz_f = 0.5
dtau = 0.01
dt = 0.01 # Not used in this script yet
NstepsI = 200
NstepsR = 100 # Not used in this script yet
CvgCrit = 1.0e-10
trunc_err_tol = 1e-8 # SVD uses tol

# --- Physical dimension ---
d = 2

# --- Operators (NumPy) ---
sx = np.array([[0., 1.], [1., 0.]], dtype=complex)
sy = np.array([[0., -1j], [1j, 0.]], dtype=complex)
sz = np.array([[1., 0.], [0., -1.]], dtype=complex)
si = np.array([[1., 0.], [0., 1.]], dtype=complex)

# --- Hamiltonian Terms (NumPy) ---
H_int = J * np.kron(sz, sz)

def H_trans(Hx):
    return -Hx * 0.5 * (np.kron(sx, si) + np.kron(si, sx))

def H_long(Hz):
    return -Hz * 0.5 * (np.kron(sz, si) + np.kron(si, sz))

# --- Initial Hamiltonian and Imaginary Time Evolution Operator ---
H_bond_i = H_int + H_trans(Hx_i) + H_long(Hz_i)
H_bond_i_op = H_bond_i.reshape(d, d, d, d) # [out_l, out_r, in_l, in_r]
U_bond_dtau = expm(-dtau * H_bond_i).reshape(d, d, d, d)

print("NumPy operators and initial evolution operator created.")
print(f"H_bond_i (shape {H_bond_i_op.shape}), U_bond_dtau (shape {U_bond_dtau.shape})")

# --- Final (Quenched) Hamiltonian and Real Time Evolution Operator ---
H_bond_f = H_int + H_trans(Hx_f) + H_long(Hz_f)
H_bond_f_op = H_bond_f.reshape(d, d, d, d)
U_bond_dt = expm(-1j * dt * H_bond_f).reshape(d, d, d, d)

print("Final evolution operator created (for later use).")
print(f"H_bond_f (shape {H_bond_f_op.shape}), U_bond_dt (shape {U_bond_dt.shape})")


def svd_truncate(theta, chi_max, tol=1e-12):
    """Performs SVD and truncates."""
    chi_l, d1, d2, chi_r = theta.shape
    theta_matrix = theta.transpose(0, 1, 2, 3).reshape(chi_l * d1, d2 * chi_r)

    try:
        U, S, Vh = svd(theta_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        print("SVD did not converge. Adding small noise.")
        noise = np.random.rand(*theta_matrix.shape) * 1e-10
        U, S, Vh = svd(theta_matrix + noise, full_matrices=False)

    threshold = tol * (S[0] if len(S)>0 and S[0] > 0 else tol)
    indices_above_tol = np.where(S > threshold)[0]
    chi_new = 1 if len(indices_above_tol) == 0 else min(indices_above_tol[-1] + 1, chi_max)

    norm_S_sq = np.sum(S**2)
    trunc_err = np.sum(S[chi_new:]**2) / norm_S_sq if norm_S_sq > 0 else 0.0

    S_trunc = S[:chi_new]
    U_trunc = U[:, :chi_new]
    Vh_trunc = Vh[:chi_new, :]

    norm = np.linalg.norm(S_trunc) if len(S_trunc) > 0 else 0.0
    lambda_new = S_trunc / norm if norm > 0 else S_trunc

    A_prime = U_trunc.reshape(chi_l, d1, chi_new)
    B_prime = (Vh_trunc / norm).reshape(chi_new, d2, chi_r) if norm > 0 else Vh_trunc.reshape(chi_new, d2, chi_r)

    return A_prime, lambda_new, B_prime, trunc_err


def itebd_step(A, B, la, lb, U_bond, chi_max, tol=1e-12):
    """Performs one step of the iTEBD update (A-B link)."""
    la = la + 1e-16
    lb = lb + 1e-16

    # 1. Construct theta = lb @ A @ la @ B @ lb
    theta = np.tensordot(A, np.diag(la), axes=(-1, 0)) # (chi_lb, d, chi_la)
    theta = np.tensordot(theta, B, axes=(-1, 0)) # (chi_lb, d, d, chi_lb)
    theta = np.tensordot(np.diag(lb), theta, axes=(1, 0))
    theta = np.tensordot(theta, np.diag(lb), axes=(-1, 0))

    # 2. Apply gate U_bond
    theta_evolved = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2])) # (d, d, chi_lb, chi_lb)
    theta_evolved = theta_evolved.transpose(2, 0, 1, 3) # (chi_lb, d, d, chi_lb)

    # 3. SVD and truncate
    A_prime, la_new, B_prime, trunc_err = svd_truncate(theta_evolved, chi_max, tol)
    # Shapes: A'(chi_lb, d, chi_new), la_new(chi_new), B'(chi_new, d, chi_lb)

    # 4. Absorb inverse singular values (lb_inv)
    lb_inv = 1.0 / lb
    A_new = np.tensordot(np.diag(lb_inv), A_prime, axes=(-1, 0)) # (chi_lb, d, chi_new)
    B_new = np.tensordot(B_prime, np.diag(lb_inv), axes=(-1, 0)) # (chi_new, d, chi_lb)

    return A_new, B_new, la_new, trunc_err


def calculate_energy(A, B, la, lb, H_bond_op):
    """Calculates the energy expectation value <H> for a two-site unit cell."""
    la = la + 1e-16

    # Construct theta = A @ la @ B
    theta = np.tensordot(A, np.diag(la), axes=(-1, 0)) # (chi_lb, d, chi_la)
    theta = np.tensordot(theta, B, axes=(-1, 0)) # (chi_lb, d, d, chi_lb)

    # Apply H_bond_op
    H_theta = np.tensordot(H_bond_op, theta, axes=([2, 3], [1, 2])) # (d, d, chi_lb, chi_lb)
    H_theta = H_theta.transpose(2, 0, 1, 3) # (chi_lb, d, d, chi_lb)

    theta_conj = np.conj(theta)
    energy_numerator = np.tensordot(theta_conj, H_theta, axes=([0,1,2,3],[0,1,2,3]))
    norm_sq = np.tensordot(theta_conj, theta, axes=([0,1,2,3],[0,1,2,3]))

    energy = (energy_numerator / norm_sq).real if abs(norm_sq) > 1e-15 else np.inf
    return energy

# --- Initialize MPS (Random State) ---
chi_init = 1
A = np.random.rand(chi_init, d, chi_init).astype(complex)
B = np.random.rand(chi_init, d, chi_init).astype(complex)
A /= np.linalg.norm(A)
B /= np.linalg.norm(B)
la = np.ones(chi_init) / np.sqrt(chi_init)
lb = np.ones(chi_init) / np.sqrt(chi_init)

print(f"Initial MPS shapes: A={A.shape}, B={B.shape}, la={la.shape}, lb={lb.shape}")

# --- Imaginary Time Evolution Loop ---
print("\n--- Starting Imaginary Time Evolution (Ground State Search) ---")
E_last = np.inf
trunc_errors_A = []
trunc_errors_B = []
bond_dims_la = []
bond_dims_lb = []
energies = []

t_start_imag = time.time()
for step in range(NstepsI):
    # 1. Update A-B bond
    A, B, la, trunc_err_A = itebd_step(A, B, la, lb, U_bond_dtau, chi_max, tol=CvgCrit/10)
    trunc_errors_A.append(trunc_err_A)
    bond_dims_la.append(len(la))

    # 2. Update B-A bond
    B, A, lb, trunc_err_B = itebd_step(B, A, lb, la, U_bond_dtau, chi_max, tol=CvgCrit/10)
    trunc_errors_B.append(trunc_err_B)
    bond_dims_lb.append(len(lb))

    current_energy = calculate_energy(A, B, la, lb, H_bond_i_op)
    energies.append(current_energy)
    energy_diff = abs(current_energy - E_last)

    if (step + 1) % 10 == 0:
        print(f"Step {step+1}/{NstepsI}, E={current_energy:.12f}, dE={energy_diff:.2e}, "
              f"TrA={trunc_err_A:.2e}, TrB={trunc_err_B:.2e}, Chi=({len(la)},{len(lb)})")

    if energy_diff < CvgCrit:
        print(f"\n[Converged!] Ground state found after {step+1} steps.")
        print(f"Final Energy = {current_energy:.12f}")
        break

    E_last = current_energy

    # Optional: Check for bond dimension saturation
    if len(la) >= chi_max and len(lb) >= chi_max and step > 50:
        if np.mean(trunc_errors_A[-5:]) < 1e-10 and np.mean(trunc_errors_B[-5:]) < 1e-10:
            print(f"\n[Converged!] Bond dim saturated at {chi_max} and truncation error low.")
            print(f"Final Energy = {current_energy:.12f}")
            break

t_end_imag = time.time()
print(f"Imaginary time evolution took {t_end_imag - t_start_imag:.2f} seconds.")

A_gs = A.copy()
B_gs = B.copy()
la_gs = la.copy()
lb_gs = lb.copy()

# --- Plotting Convergence ---
print("\nPlotting convergence diagnostics...")
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
axs[0].plot(range(1, len(energies) + 1), energies, 'o-', markersize=3)
axs[0].set_ylabel("Energy <H>")
axs[0].set_title(f"iTEBD Ground State Search (Hx={Hx_i}, Hz={Hz_i}, J={J})")
axs[0].grid(True)
axs[1].plot(range(1, len(trunc_errors_A) + 1), trunc_errors_A, 'r.-', label='Trunc Err (A-B)')
axs[1].plot(range(1, len(trunc_errors_B) + 1), trunc_errors_B, 'b.-', label='Trunc Err (B-A)')
axs[1].set_ylabel("Truncation Error"); axs[1].set_yscale('log'); axs[1].legend(); axs[1].grid(True)
axs[2].plot(range(1, len(bond_dims_la) + 1), bond_dims_la, 'r.-', label='Bond Dim (la)')
axs[2].plot(range(1, len(bond_dims_lb) + 1), bond_dims_lb, 'b.-', label='Bond Dim (lb)')
axs[2].set_ylabel("Bond Dimension"); axs[2].set_xlabel("Imaginary Time Step")
axs[2].axhline(chi_max, color='k', linestyle='--', label=f'chi_max={chi_max}')
axs[2].legend(); axs[2].grid(True)
plt.tight_layout(); plt.show()

print("\nGround state calculation complete. Tensors A_gs, B_gs, la_gs, lb_gs contain the result.")
