from calculate_Berry_curvature import *
from calculate_w_coeff import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

a1, a2, K1, K2 = compute_lattice_vectors()

# Parameterize the k-space unit cell in u and v:
# We choose u and v in [0,1). (The physical k is then: k = -offset + u*K1 + v*K2
# where the offset can be chosen so that the cell is centered; here we simply use u,v in [0,1).)
Nu, Nv = 150, 150
FourierMax = 5
u_vals = np.linspace(-2, 2, Nu, endpoint=False)
v_vals = np.linspace(-2, 2, Nv, endpoint=False)

# Compute the effective area element in parameter space.
# The transformation determinant (Jacobian) is |det(K1, K2)|.
jacobian = np.abs(np.linalg.det(np.column_stack((K1, K2))))
du_area = u_vals[1] - u_vals[0]
dv_area = v_vals[1] - v_vals[0]
area_element = jacobian * du_area * dv_area


# For each (u,v) point, get the physical k and then evaluate N_k^{-2}.
kx_grid = np.zeros((Nu, Nv))
ky_grid = np.zeros((Nu, Nv))
in_BZ_mask = np.zeros((Nu, Nv), dtype=bool)
# Compute Nk^{-2} on the parameter grid.
for i, u in enumerate(u_vals):
	for j, v in enumerate(v_vals):
		k_pt = physical_k(u, v, K1, K2)
		kx_grid[i, j] = k_pt[0]
		ky_grid[i, j] = k_pt[1]
		in_BZ_mask[i, j] = in_first_BZ(k_pt, K1, K2)

B1 = 3.0
M = lambda x, y: M_func(x, y, B1)
w_dict, M_approx, err, X, Y, f, f_approx = fourier_truncation_approx(M, a1, a2, Mmax=FourierMax, Nmax=FourierMax)

# For each (u,v) point, get the physical k and then evaluate N_k^{-2}.
Nk_inv2_grid = np.zeros((Nu, Nv), dtype=float)
for i, u in enumerate(u_vals):
	for j, v in enumerate(v_vals):
		k_pt = physical_k(u, v, K1, K2)
		Nk_inv2_grid[i, j] = compute_Nk_inv2(k_pt, K1, K2, w_dict, n_max=1)

# Compute the Berry curvature on the (u,v) grid.
Omega_grid = compute_berry_curvature_general(Nk_inv2_grid, u_vals, v_vals, K1, K2)

# Integrate Berry curvature in the first BZ to obtain the Chern number.
chern_number = np.sum(Omega_grid[in_BZ_mask]) * area_element / (2*np.pi)

s4 = 12/(2 * 4) * np.sum(2*(Omega_grid[in_BZ_mask])**2) * area_element/(2 * np.pi)


print("Omega_grid shape:", Omega_grid.shape)
print("Center point Omega ~", Omega_grid[Nu//2, Nv//2])
print("Chern number:", chern_number)
print("S4:", s4)

# Plot the Berry curvature as a function of physical (kx, ky).
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
# Use pcolormesh for a 2D colormap. Since the grid is a parallelogram, pcolormesh handles it.
plt.pcolormesh(kx_grid, ky_grid, Omega_grid, shading='auto', cmap='plasma_r', vmin=-10, vmax=0)
plt.colorbar(label='$\Omega(k)/l^2$')
plt.xlabel('$k_x l$')
plt.ylabel('$k_y l$')
plt.xlim(-K1[0]*1.2, K1[0]*1.2)
plt.ylim(-K1[0]*1.2, K1[0]*1.2)
plt.title('$B_1/B_0=$'+str(B1))
# aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
# Save the figure
filename = __file__
filename = filename.split('.')[0]
plt.savefig(f'{filename}.svg')
plt.savefig(f'{filename}.png')
# plt.savefig('berry_curvature.png', dpi=300)	
