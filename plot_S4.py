from calculate_Berry_curvature import *
from calculate_w_coeff import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Parameterize the reciprocal unit cell in u and v:
# We choose u and v in [-2,2). (The physical k is then: k = -offset + u*K1 + v*K2
# where the offset can be chosen so that the cell is centered; here we simply use u,v in [-2,2).)
a1, a2, K1, K2 = compute_lattice_vectors()

Nu, Nv = 150, 150
# Nu, Nv = 50, 50
u_vals = np.linspace(-2, 2, Nu, endpoint=False)
v_vals = np.linspace(-2, 2, Nv, endpoint=False)
U, V = np.meshgrid(u_vals, v_vals)
kx_grid = K1[0] * U + K2[0] * V
ky_grid = K1[1] * U + K2[1] * V

# Compute the effective area element in parameter space.
# The transformation determinant (Jacobian) is |det(K1, K2)|.
jacobian = np.abs(np.linalg.det(np.column_stack((K1, K2))))
print(jacobian)
du_area = u_vals[1] - u_vals[0]
dv_area = v_vals[1] - v_vals[0]
area_element = jacobian * du_area * dv_area

# Define the function to check if a point is in the first BZ.
in_BZ_mask = np.zeros((Nu, Nv), dtype=bool)
for i in range(Nu):
	for j in range(Nv):
		k_pt = np.array([kx_grid[i, j], ky_grid[i, j]])
		in_BZ_mask[i, j] = in_first_BZ(k_pt, K1, K2)


# Cut off wavevector for M function
FourierMax = 5

# Compute Nk^{-2} on the parameter grid.
Nk_inv2_grid = np.zeros((Nu, Nv), dtype=float)

# Define weights w_b.
# calculate S4 for various B1
B1_list = np.linspace(0.0, 5.0, 41)
Err_list = []
s4_list = []
for B1 in B1_list:
	M = lambda x, y: M_func(x, y, B1)
	w_dict, M_approx, err, X, Y, f, f_approx = fourier_truncation_approx(M, a1, a2, Mmax=FourierMax, Nmax=FourierMax)
	Err_list.append(err)
	# For each (u,v) point, get the physical k and then evaluate N_k^{-2}.
	for i, u in enumerate(u_vals):
		for j, v in enumerate(v_vals):
			k_pt = physical_k(u, v, K1, K2)
			Nk_inv2_grid[i, j] = compute_Nk_inv2(k_pt, K1, K2, w_dict, n_max=1)
			kx_grid[i, j] = k_pt[0]
			ky_grid[i, j] = k_pt[1]

	# Compute the Berry curvature on the (u,v) grid.
	Omega_grid = compute_berry_curvature_general(Nk_inv2_grid, u_vals, v_vals, K1, K2)

	# Compute Chern number (integrate Omega over the unit cell and divide by 2π).
	# We integrate only over those grid points for which in_first_BZ(k) is True.
	# Integrate Berry curvature in the first BZ to obtain the Chern number.
	chern_number = np.sum(Omega_grid[in_BZ_mask]) * area_element / (2*np.pi)

	s4 = 12/(2 * 4) * np.sum(2*(Omega_grid[in_BZ_mask])**2) * area_element/(2 * np.pi)
	s4_list.append(s4)

	print("B1:", B1)
	print("Truncation error:", err)
	print("Omega_grid shape:", Omega_grid.shape)
	print("Center point Omega ~", Omega_grid[Nu//2, Nv//2])
	print("Chern number:", chern_number)
	print("S4:", s4)

plt.figure(figsize=(6, 2))
plt.plot(B1_list, np.array(s4_list), lw=2.5)
plt.axhline(3, color='red', lw=1.5, ls='--')
plt.xlabel('$B_1/B_0$')
plt.ylabel('$S_4$')
plt.xlim(0, 5)
plt.ylim(0, 10)
plt.tight_layout()
# Save the figure
filename = __file__
filename = filename.split('.')[0]
plt.savefig(f'{filename}.svg')
plt.savefig(f'{filename}.png')