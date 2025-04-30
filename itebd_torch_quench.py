# Full PyTorch iTEBD Simulation (GPU accelerated): Ground State + Real Time Quench
import torch
import matplotlib.pyplot as plt
import time

# --- Device Selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Parameters ---
chi_max = 50
J = 1.0              # Antiferromagnetic
# Initial state parameters (Ground state for these)
Hx_i = 10.0
Hz_i = 0.0
# Final state parameters (Quench to these)
Hx_f = 2.0
Hz_f = 0.0
# Time steps
dtau = 0.01             # Imaginary time step (for GS)
dt = 0.01           # Real time step (for quench) 
# Evolution steps
NstepsI = 200         # Imaginary time steps
NstepsR = 1000         # Real time steps (Total time T = NstepsR * dt)
# Convergence criterion for ground state energy
CvgCrit = 1.0e-10
# Truncation tolerance for SVD in real time
trunc_err_tol = 1e-8
# Data type
dtype = torch.complex128 # Use complex128 for higher precision

# --- Physical dimension ---
d = 2

# --- Operators (PyTorch) ---
sx = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype, device=device)
sy = torch.tensor([[0., -1j], [1j, 0.]], dtype=dtype, device=device)
sz = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype, device=device)
si = torch.tensor([[1., 0.], [0., 1.]], dtype=dtype, device=device)

# --- Hamiltonian Terms (PyTorch) ---
# NOTE: User modified H_int in gs.py to -J * kron(sz, sz), keeping it here for consistency
H_int = J * torch.kron(sz, sz) # Matching user's change in gs.py

def H_trans(Hx):
    return -Hx * 0.5 * (torch.kron(sx, si) + torch.kron(si, sx))

def H_long(Hz):
    return -Hz * 0.5 * (torch.kron(sz, si) + torch.kron(si, sz))

# --- Initial Hamiltonian and Imaginary Time Evolution Operator ---
H_bond_i = H_int + H_trans(Hx_i) + H_long(Hz_i)
# torch.linalg.matrix_exp needs a 2D matrix
H_bond_i_mat = H_bond_i.reshape(d*d, d*d)
U_bond_dtau_mat = torch.linalg.matrix_exp(-dtau * H_bond_i_mat)
U_bond_dtau = U_bond_dtau_mat.reshape(d, d, d, d)
H_bond_i_op = H_bond_i.reshape(d, d, d, d) # Keep 4D op for energy calculation
print("Initial H and U_dtau created.")

# --- Final (Quenched) Hamiltonian and Real Time Evolution Operator ---
H_bond_f = H_int + H_trans(Hx_f) + H_long(Hz_f)
# torch.linalg.matrix_exp needs a 2D matrix
H_bond_f_mat = H_bond_f.reshape(d*d, d*d)
U_bond_dt_mat = torch.linalg.matrix_exp(-1j * dt * H_bond_f_mat)
U_bond_dt = U_bond_dt_mat.reshape(d, d, d, d)
H_bond_f_op = H_bond_f.reshape(d, d, d, d) # Keep 4D op for energy calculation
print("Final H and U_dt created.")

def svd_truncate(theta, chi_max, tol=1e-12):
    """Performs SVD and truncates using PyTorch.
       Handles potential non-convergence by adding small noise.
    """
    chi_l, d1, d2, chi_r = theta.shape
    theta_matrix = theta.permute(0, 1, 2, 3).reshape(chi_l * d1, d2 * chi_r)

    try:
        U, S, Vh = torch.linalg.svd(theta_matrix, full_matrices=False)
    except torch._C._LinAlgError: # Catch PyTorch's linalg error
        print("SVD Warning: Did not converge. Adding noise.")
        noise = torch.rand(*theta_matrix.shape, dtype=dtype, device=device) * 1e-10
        U, S, Vh = torch.linalg.svd(theta_matrix + noise, full_matrices=False)

    # S is real for complex svd in PyTorch, thresholding logic is simpler
    threshold = tol * (S[0] if len(S) > 0 and S[0] > 1e-16 else tol)
    indices_above_tol = torch.where(S > threshold)[0]
    chi_new = 1 if len(indices_above_tol) == 0 else min(indices_above_tol[-1].item() + 1, chi_max)

    norm_S_sq = torch.sum(S**2)
    trunc_err = torch.sum(S[chi_new:]**2) / norm_S_sq if norm_S_sq > 1e-16 else 0.0

    S_trunc = S[:chi_new]
    U_trunc = U[:, :chi_new]
    Vh_trunc = Vh[:chi_new, :]

    norm = torch.linalg.norm(S_trunc) if len(S_trunc) > 0 else 0.0
    # Ensure lambda_new is complex if dtype is complex
    lambda_new = (S_trunc / norm).to(dtype) if norm > 1e-16 else torch.zeros_like(S_trunc, dtype=dtype, device=device)

    A_prime = U_trunc.reshape(chi_l, d1, chi_new)
    # Ensure B_prime is complex if dtype is complex
    B_prime_scaled = (Vh_trunc / norm).to(dtype) if norm > 1e-16 else Vh_trunc.to(dtype)
    B_prime = B_prime_scaled.reshape(chi_new, d2, chi_r)

    return A_prime, lambda_new, B_prime, trunc_err.item() # Return error as float

def itebd_step(A, B, la, lb, U_bond, chi_max, tol=1e-12):
    """Performs one step of the iTEBD update (A-B link) using PyTorch.
       Based on user's modified version.
    """
    # Add small value for numerical stability before inverse
    # Do NOT modify la, lb in-place if they are needed later unchanged
    la_stable = la + 1e-16
    lb_stable = lb + 1e-16

    # 1. Construct theta = lb @ A @ la @ B @ lb (User's version)
    theta = torch.tensordot(A, torch.diag(la_stable), dims=([-1], [0]))
    theta = torch.tensordot(theta, B, dims=([-1], [0]))
    theta = torch.tensordot(torch.diag(lb_stable), theta, dims=([1], [0])) # lb acts on left index of A
    theta = torch.tensordot(theta, torch.diag(lb_stable), dims=([-1], [0])) # lb acts on right index of B

    # 2. Apply gate U_bond
    # Indices: U(phys_l', phys_r', phys_l, phys_r), theta(lb_l, phys_l, phys_r, lb_r)
    theta_evolved = torch.tensordot(U_bond, theta, dims=([2, 3], [1, 2])) # Result(phys_l', phys_r', lb_l, lb_r)
    theta_evolved = theta_evolved.permute(2, 0, 1, 3) # Result(lb_l, phys_l', phys_r', lb_r) -> Correct for svd_truncate

    # 3. SVD Truncate
    A_prime, la_new, B_prime, trunc_err = svd_truncate(theta_evolved, chi_max, tol)
    # Shapes: A'(chi_lb, d, chi_new), la_new(chi_new), B'(chi_new, d, chi_lb)

    # 4. Absorb inverse singular values (lb_inv) - User's version
    lb_inv = 1.0 / lb_stable # Use the same lb from the outer bonds
    A_new = torch.tensordot(torch.diag(lb_inv), A_prime, dims=([-1], [0])) # (chi_lb, d, chi_new)
    B_new = torch.tensordot(B_prime, torch.diag(lb_inv), dims=([-1], [0])) # (chi_new, d, chi_lb)

    return A_new, B_new, la_new, trunc_err

def calculate_energy(A, B, la, lb, H_bond_op):
    """Calculates the energy expectation value <H> for a two-site unit cell using PyTorch."""
    # Use user's version from quench script (modified slightly)
    la_stable = la + 1e-16
    lb_stable = lb + 1e-16 # Not used in this calculation as per user code

    theta = torch.tensordot(A, torch.diag(la_stable), dims=([-1], [0]))
    theta = torch.tensordot(theta, B, dims=([-1], [0])) # (chi_lb, d, d, chi_lb)
    theta = torch.tensordot(torch.diag(lb_stable), theta, dims=([-1], [0])) # (chi_lb, d, d, chi_lb)
    theta = torch.tensordot(theta, torch.diag(lb_stable), dims=([-1], [0])) # (chi_lb, d, d, chi_lb)

    H_theta = torch.tensordot(H_bond_op, theta, dims=([2, 3], [1, 2])) # (d,d,chi_lb,chi_lb)
    H_theta = H_theta.permute(2, 0, 1, 3) # (chi_lb, d, d, chi_lb)

    theta_conj = torch.conj(theta)
    # Using einsum for complex inner product
    energy_numerator = torch.einsum('abcd,abcd->', theta_conj, H_theta)
    norm_sq = torch.einsum('abcd,abcd->', theta_conj, theta)

    energy = (energy_numerator / norm_sq).real if abs(norm_sq) > 1e-15 else float('inf')
    return energy.item() # Return as float

def calculate_observables(A, B, la, lb, op):
    la_stable = la + 1e-16
    lb_stable = lb + 1e-16

    # 1. Construct theta = A * la
    # Indices: (chi_lb_left, phys_A, chi_la_right)
    theta = torch.tensordot(A, torch.diag(la_stable), dims=([-1], [0])) # (chi_lb, d, chi_la)
    theta = torch.tensordot(theta, torch.diag(lb_stable), dims=([0], [-1])) # (chi_lb, d, chi_lb)
    theta = theta.permute(2, 0, 1)
    theta_conj = torch.conj(theta)
    norm_sq = torch.einsum('abc,abc->', theta_conj, theta)

    # Avoid division by zero if norm is too small
    if abs(norm_sq) < 1e-15:
        print("Warning: Norm squared is close to zero in calculate_observables.")
        return 0.0

    # 2. Calculate observable
    # Construct theta_opA = op * theta
    theta_opA = torch.tensordot(op, theta, dims=([1],[1]))
    theta_opA = theta_opA.permute(1, 0, 2)
    # Numerator_A = <theta | theta_opA>
    num_A = torch.einsum('abc,abc->', theta_conj, theta_opA) #
    obs_A = (num_A / norm_sq).real
    return obs_A

def calculate_entropy(la):
    """Calculates the Von Neumann entanglement entropy from singular values."""
    # Ensure singular values are positive and filter out zeros/small values
    la_sq = la.real**2 # Schmidt values from svd should be real
    la_sq_clean = la_sq[la_sq > 1e-16]
    if len(la_sq_clean) == 0:
        return 0.0
    entropy = -torch.sum(la_sq_clean * torch.log(la_sq_clean))
    return entropy.item() # Return as float

# --- Initialize MPS (Random State) ---
chi_init = 1
A = torch.rand(chi_init, d, chi_init, dtype=dtype, device=device)
B = torch.rand(chi_init, d, chi_init, dtype=dtype, device=device)
A /= torch.linalg.norm(A); B /= torch.linalg.norm(B)

# Initial Schmidt values (uniform)
la = torch.ones(chi_init, dtype=dtype, device=device) / torch.sqrt(torch.tensor(chi_init, dtype=torch.float64, device=device))
lb = torch.ones(chi_init, dtype=dtype, device=device) / torch.sqrt(torch.tensor(chi_init, dtype=torch.float64, device=device))
la = la.to(dtype)
lb = lb.to(dtype)

# --- Imaginary Time Evolution Loop (Ground State Search) ---
print("\n--- Starting Imaginary Time Evolution (Ground State Search) ---")
E_last = float('inf')
trunc_errors_A_im = []
trunc_errors_B_im = []
bond_dims_la_im = []
bond_dims_lb_im = []
energies = []
t_start_imag = time.time()

for step in range(NstepsI):
    # A-B update
    A, B, la, trunc_err_A = itebd_step(A, B, la, lb, U_bond_dtau, chi_max, tol=CvgCrit/10)
    trunc_errors_A_im.append(trunc_err_A)
    bond_dims_la_im.append(len(la))

    # B-A update
    B, A, lb, trunc_err_B = itebd_step(B, A, lb, la, U_bond_dtau, chi_max, tol=CvgCrit/10)
    trunc_errors_B_im.append(trunc_err_B)
    bond_dims_lb_im.append(len(lb))

    if (step + 1) % 10 == 0:
        # Calculate energy on both bonds for average
        E_bond_AB = calculate_energy(A, B, la, lb, H_bond_i_op)
        E_bond_BA = calculate_energy(B, A, lb, la, H_bond_i_op)
        E_curr = (E_bond_AB + E_bond_BA) / 2.0
        energies.append(E_curr)
        delta_E = abs(E_curr - E_last)
        print(f"Im Step {(step+1):<4}/{NstepsI}, E={E_curr:.10f}, dE={delta_E:.2e}, "
              f"TrA={trunc_err_A:.2e}, TrB={trunc_err_B:.2e}, Chi=({len(la)},{len(lb)})")
        if delta_E < CvgCrit:
            print(f"Converged at step {step + 1}")
            break
        E_last = E_curr

t_end_imag = time.time()
print(f"Imaginary time evolution took {t_end_imag - t_start_imag:.2f} seconds.")

A_gs = A.clone(); B_gs = B.clone(); la_gs = la.clone(); lb_gs = lb.clone()
if energies:
    print(f"Ground state energy: {energies[-1]:.12f}")
else:
     print("Ground state energy not calculated (0 steps or no convergence check).")

# --- Real Time Evolution Loop ---
print("\n--- Starting Real Time Evolution ---")
A = A_gs.clone(); B = B_gs.clone(); la = la_gs.clone(); lb = lb_gs.clone()

# Use numpy for plotting storage - transfer tensors to CPU first
times_np = torch.arange(NstepsR + 1, device='cpu').numpy() * dt
Mx_t_np = torch.zeros(NstepsR + 1, device='cpu').numpy()
Mz_t_np = torch.zeros(NstepsR + 1, device='cpu').numpy()
Entropy_t_np = torch.zeros(NstepsR + 1, device='cpu').numpy()
trunc_errors_A_rt_np = torch.zeros(NstepsR, device='cpu').numpy()
trunc_errors_B_rt_np = torch.zeros(NstepsR, device='cpu').numpy()
bond_dims_la_rt_np = torch.zeros(NstepsR, dtype=torch.int, device='cpu').numpy()
bond_dims_lb_rt_np = torch.zeros(NstepsR, dtype=torch.int, device='cpu').numpy()

# Calculate initial observables
Mx_t_np[0] = calculate_observables(A, B, la, lb, sx)
Mz_t_np[0] = calculate_observables(A, B, la, lb, sz)
Entropy_t_np[0] = calculate_entropy(la) # Entropy of the A-B bond

t_start_real = time.time()
for step in range(NstepsR):
    # A-B update
    A, B, la, trunc_err_A = itebd_step(A, B, la, lb, U_bond_dt, chi_max, tol=trunc_err_tol)
    trunc_errors_A_rt_np[step] = trunc_err_A
    bond_dims_la_rt_np[step] = len(la)

    # B-A update
    B, A, lb, trunc_err_B = itebd_step(B, A, lb, la, U_bond_dt, chi_max, tol=trunc_err_tol)
    trunc_errors_B_rt_np[step] = trunc_err_B
    bond_dims_lb_rt_np[step] = len(lb)

    # Calculate observables after full A-B, B-A cycle
    Mx_t_np[step + 1] = calculate_observables(A, B, la, lb, sx)
    Mz_t_np[step + 1] = calculate_observables(A, B, la, lb, sz)
    Entropy_t_np[step + 1] = calculate_entropy(la) # Use entropy of A-B bond

    if (step + 1) % 20 == 0: # Print less frequently
        print(f"RT Step {(step+1):<4}/{NstepsR}, t={times_np[step+1]:.2f}, "
              f"Mx={Mx_t_np[step+1]:.4f}, Mz={Mz_t_np[step+1]:.4f}, S={Entropy_t_np[step+1]:.4f}, "
              f"TrA={trunc_err_A:.1e}, TrB={trunc_err_B:.1e}, Chi=({len(la)},{len(lb)})")

t_end_real = time.time()
print(f"Real time evolution took {t_end_real - t_start_real:.2f} seconds.")

# --- Plotting Real Time Results --- (Using stored numpy arrays)
print("\nPlotting real-time evolution results...")
fig_rt, axs_rt = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

axs_rt[0].plot(times_np, Mx_t_np, 'o-', markersize=3, label=r'$\langle S_x \rangle$')
axs_rt[0].plot(times_np, Mz_t_np, 's-', markersize=3, label=r'$\langle S_z \rangle$')
axs_rt[0].set_ylabel("Magnetization")
axs_rt[0].set_title(f"Real Time Evolution (Torch/{device}) (Hx:{Hx_i}->{Hx_f}, Hz:{Hz_i}->{Hz_f}, J={J}, $\chi$={chi_max})")
axs_rt[0].legend(); axs_rt[0].grid(True)

axs_rt[1].plot(times_np, Entropy_t_np, 'o-', markersize=3, color='g')
axs_rt[1].set_ylabel("Entanglement Entropy S (bond A-B)")
axs_rt[1].grid(True)

axs_rt[2].plot(times_np[1:], trunc_errors_A_rt_np, 'r.-', label='Trunc Err (A-B)')
axs_rt[2].plot(times_np[1:], trunc_errors_B_rt_np, 'b.-', label='Trunc Err (B-A)')
axs_rt[2].set_ylabel("Truncation Error"); axs_rt[2].set_yscale('log'); axs_rt[2].legend(); axs_rt[2].grid(True)

axs_rt[3].plot(times_np[1:], bond_dims_la_rt_np, 'r.-', label='Bond Dim (la)')
axs_rt[3].plot(times_np[1:], bond_dims_lb_rt_np, 'b.-', label='Bond Dim (lb)')
axs_rt[3].set_ylabel("Bond Dimension"); axs_rt[3].set_xlabel("Time (t)")
axs_rt[3].axhline(chi_max, color='k', linestyle='--', label=f'$\chi_{{max}}$={chi_max}')
axs_rt[3].legend(); axs_rt[3].grid(True)

plt.tight_layout()
plt.show()

print("\nFull simulation complete.")
