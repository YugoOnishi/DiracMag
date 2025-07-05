import numpy as np

def is_half_b_still_reciprocal(n1, n2):
    """
    Returns True if (b/2) is a reciprocal-lattice vector.
    For a 2D lattice with fundamental vectors G1, G2,
    b = (n1 * G1 + n2 * G2). Then b/2 is also reciprocal
    if and only if n1, n2 are both even. 
    """
    return (n1 % 2 == 0) and (n2 % 2 == 0)

def eta_b(n1, n2):
    """
    +1 if b/2 is a reciprocal-lattice vector, -1 otherwise.
    """
    return +1 if is_half_b_still_reciprocal(n1, n2) else -1

def k_cross_b(k, b):
    """
    In 2D, the 'cross product' k x b is just the scalar
        k_x * b_y - k_y * b_x
    """
    return k[0]*b[1] - k[1]*b[0]

def compute_Nk_inv2(k, G1, G2, w_dict, n_max=5):
    """
    Numerically compute:
        N_k^{-2} = sum_{b} [ eta_b * w_b * exp(i k x b) * exp(-|b|^2 / 4 ) ]
    
    Here:
      - b = n1*G1 + n2*G2,  with (n1, n2) in [-n_max, n_max]
      - eta_b = +1 if b/2 is reciprocal, else -1
      - w_b    = w_dict[(n1, n2)]  (must be provided)
      - k x b  = 2D cross product => scalar
    """
    val = 0.0 + 0.0j  # accumulate complex sum
    
    # Loop over a finite range of n1, n2
    for n1 in range(-n_max, n_max+1):
        for n2 in range(-n_max, n_max+1):
            # Construct the reciprocal lattice vector b
            b = n1*G1 + n2*G2
            
            # Get eta_b
            eta_val = eta_b(n1, n2)
            
            # Fetch w_b from the dictionary (or 0 if not present)
            #   Depending on how w_b is defined, you might do w_dict.get((n1, n2), 0.0)
            #   or directly assume it is provided for all n1, n2 in range.
            w_val = w_dict.get((n1, n2), 0.0)
            
            # Phase factor: exp(i k x b)
            phase = np.exp(1j * k_cross_b(k, b))
            
            # Gaussian factor: exp(-|b|^2 / 4)
            b_mag_sq = b[0]**2 + b[1]**2
            gauss = np.exp(-0.25 * b_mag_sq)
            
            # Accumulate
            val += eta_val * w_val * phase * gauss
    
    return val

# ---------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Define fundamental reciprocal lattice vectors (2D example).
    # For instance, suppose G1 = (2π/a, 0), G2 = (π/a, √3π/a)
    # You can replace these with whatever your system uses.
    G1 = np.array([1.0, 0.0])   # Example
    G2 = np.array([0.5, np.sqrt(3)/2])  # Example
    
    # Wavevector k in 2D
    k = np.array([0.2, 0.3])
    
    # Suppose we have w_b precomputed or given.
    # For demonstration, we'll just define w_{n1,n2} = exp(-alpha * |b|^2)
    # or store them in a dictionary.
    alpha = 0.2
    n_max_user = 5
    w_dict = {}
    for n1 in range(-n_max_user, n_max_user+1):
        for n2 in range(-n_max_user, n_max_user+1):
            b_vec = n1*G1 + n2*G2
            b_sq  = b_vec[0]**2 + b_vec[1]**2
            w_dict[(n1, n2)] = np.exp(-alpha * b_sq)
    
    # Now compute N_k^{-2} using the code above
    Nk_inv2_val = compute_Nk_inv2(k, G1, G2, w_dict, n_max=n_max_user)
    
    print("N_k^{-2} =", Nk_inv2_val)


def compute_berry_curvature_general(Nk_inv2_grid, u_vals, v_vals, K1, K2):
    """
    Compute the Berry curvature Omega(k) = -1 + Δ_k[ log(N_k) ]
    on a k–grid that is defined by parameters (u, v) such that
       k(u,v) = K1 * u + K2 * v.
    Det(K1, K2) = 2pi is necessary. The area of real space unit cell is 2pi. (The magnetic length is set to be 1)
    
    Here Nk_inv2_grid is the 2D array (shape: (Nu, Nv))
    of N_k^{-2} evaluated on the (u,v)–grid.
    
    The Laplacian in physical k–space is obtained via:
    
      Δ_k f = g^{11} f_{uu} + 2 g^{12} f_{uv} + g^{22} f_{vv},
    
    where the metric components are computed from K1 and K2:
    
      g_{11} = K1·K1, g_{12} = K1·K2, g_{22} = K2·K2,
    and g^{ij} are the elements of the inverse metric.
    
    Finite differences are computed in the (u, v) parameter space,
    where u and v are assumed uniformly spaced (but the lattice can be non–orthogonal).
    
    Parameters
    ----------
    Nk_inv2_grid : 2D ndarray, shape (Nu, Nv)
         Array of N_k^{-2} evaluated on the parameter–grid.
    u_vals : 1D array, length Nu
         Parameter coordinate in the K1 direction.
    v_vals : 1D array, length Nv
         Parameter coordinate in the K2 direction.
    K1, K2 : array-like, shape (2,)
         The two reciprocal lattice basis vectors (in physical k–space).
    
    Returns
    -------
    Omega : 2D ndarray, shape (Nu, Nv)
         Berry curvature at each grid point.
    """
    # Compute N_k and then f = log(N_k)
    Nk_grid = 1.0 / np.sqrt(Nk_inv2_grid)
    f = np.log(Nk_grid)
    
    # Spacings in the parameter space
    du = u_vals[1] - u_vals[0]
    dv = v_vals[1] - v_vals[0]
    
    # Compute second derivatives in parameter space using np.gradient.
    # First derivatives:
    f_u = np.gradient(f, du, axis=0)
    f_v = np.gradient(f, dv, axis=1)
    # Second derivatives:
    f_uu = np.gradient(f_u, du, axis=0)
    f_vv = np.gradient(f_v, dv, axis=1)
    # Mixed derivative: average the two orders.
    f_uv_a = np.gradient(f_u, dv, axis=1)
    f_uv_b = np.gradient(f_v, du, axis=0)
    f_uv = 0.5 * (f_uv_a + f_uv_b)
    
    # Compute the metric from K1, K2.
    g11 = np.dot(K1, K1)
    g12 = np.dot(K1, K2)
    g22 = np.dot(K2, K2)
    det_g = g11 * g22 - g12**2
    # Inverse metric components:
    g11_inv = g22 / det_g
    g12_inv = -g12 / det_g
    g22_inv = g11 / det_g
    
    # Now the Laplacian in physical k-space is:
    lap_f = g11_inv * f_uu + 2.0 * g12_inv * f_uv + g22_inv * f_vv
    
    # Finally, Berry curvature:
    Omega = -1.0 + lap_f
    return Omega

def in_first_BZ(k, K1, K2, tol=1e-10):
    """
    Determine whether a physical k–point lies in the first Brillouin zone (BZ).
    
    The first BZ is defined as the set of points that are as close or closer to
    the origin than to any other nonzero reciprocal-lattice vector.
    
    For a given k, we check against a few nearest reciprocal-lattice vectors.
    
    Parameters:
      k : array-like, shape (2,)
          The physical k–point.
      K1, K2 : array-like, shape (2,)
          The reciprocal lattice basis vectors.
      tol : float, tolerance for equality.
    
    Returns:
      is_inside : bool, True if k is in the first BZ.
    """
    # Compute distance of k from the origin
    dist0 = np.linalg.norm(k)
    
    # Check against several nonzero b = n1*K1 + n2*K2 for n1,n2 in {-1,0,1} (excluding 0,0)
    for n1 in [-2, -1, 0, 1, 2]:
        for n2 in [-2, -1, 0, 1, 2]:
            if n1 == 0 and n2 == 0:
                continue
            b = n1 * K1 + n2 * K2
            # k is in the first BZ if for all nonzero b, ||k|| <= ||k - b||
            if np.linalg.norm(k - b) < dist0 - tol:
                return False
    return True

def physical_k(u, v, K1, K2):
    """
    Map the (u,v) parameters to a physical k–point via a linear transformation:
      k = u*K1 + v*K2.
    
    Here u,v can extend beyond the first BZ (which, for a primitive cell,
    is typically represented by u,v in [-0.5,0.5]). With u,v ∈ [–1,1] the resulting k–grid
    covers an extended region.
    
    Returns:
      k : 1D ndarray (length 2) representing [kₓ, k_y].
    """
    return u * K1 + v * K2

# ----------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Suppose your reciprocal unit cell is defined by the (possibly non–orthogonal)
    # reciprocal lattice vectors K1 and K2. (For example, for a triangular lattice
    # you might choose:
    #   K1 = (K, 0)
    #   K2 = (K/2, sqrt(3)*K/2)
    # with appropriate normalization.)
    # Det(K1, K2) = 2pi is necessary. The area of real space unit cell is 2pi. (the magnetic length is set to be 1)
    # Here we give an example with a triangular unit cell.

    K1 = np.array([1, 0.0]) * np.sqrt(4*np.pi/np.sqrt(3))
    K2 = np.array([-0.5, np.sqrt(3)/2]) * np.sqrt(4*np.pi/np.sqrt(3))
    print("Det(K1, K2)/2pi = ", np.linalg.det(np.column_stack((K1, K2)))/(2*np.pi))

    # Parameterize the unit cell in u and v:
    # We choose u and v in [0,1). (The physical k is then: k = -offset + u*K1 + v*K2
    # where the offset can be chosen so that the cell is centered; here we simply use u,v in [0,1).)
    Nu, Nv = 150, 150
    u_vals = np.linspace(-2, 2, Nu, endpoint=False)
    v_vals = np.linspace(-2, 2, Nv, endpoint=False)


    # Compute Nk^{-2} on the parameter grid.
    Nk_inv2_grid = np.zeros((Nu, Nv), dtype=float)
    kx_grid = np.zeros((Nu, Nv))
    ky_grid = np.zeros((Nu, Nv))
    # Define weights w_b as before.
    w_dict = {
        (0, 0): 1.0,
        (1, 0): 0.1,
        (-1, 0): 0.1,
        (0, 1): 0.1,
        (0, -1): 0.1,
        (1, 1): 0.1,
        (-1, -1): 0.1,
    }
    # For each (u,v) point, get the physical k and then evaluate N_k^{-2}.
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            k_pt = physical_k(u, v, K1, K2)
            Nk_inv2_grid[i, j] = compute_Nk_inv2(k_pt, K1, K2, w_dict, n_max=1)
            kx_grid[i, j] = k_pt[0]
            ky_grid[i, j] = k_pt[1]

    # Compute the Berry curvature on the (u,v) grid.
    Omega_grid = compute_berry_curvature_general(Nk_inv2_grid, u_vals, v_vals, K1, K2)

    # Compute the effective area element in parameter space.
    # The transformation determinant (Jacobian) is |det(K1, K2)|.
    jacobian = np.abs(np.linalg.det(np.column_stack((K1, K2))))
    du_area = u_vals[1] - u_vals[0]
    dv_area = v_vals[1] - v_vals[0]
    area_element = jacobian * du_area * dv_area

    # Compute Chern number (integrate Omega over the unit cell and divide by 2π).
    # Now, extract the Berry curvature within the first Brillouin zone.
    # We integrate only over those grid points for which in_first_BZ(k) is True.
    in_BZ_mask = np.zeros((Nu, Nv), dtype=bool)
    for i in range(Nu):
        for j in range(Nv):
            k_pt = np.array([kx_grid[i, j], ky_grid[i, j]])
            in_BZ_mask[i, j] = in_first_BZ(k_pt, K1, K2)

    # Integrate Berry curvature in the first BZ to obtain the Chern number.
    chern_number = np.sum(Omega_grid[in_BZ_mask]) * area_element / (2*np.pi)
    # chern_number = np.sum(Omega_grid) * area_element / (2*np.pi) / 4


    print("Omega_grid shape:", Omega_grid.shape)
    print("Center point Omega ~", Omega_grid[Nu//2, Nv//2])
    print("Chern number:", chern_number)

    # Now, map the (u,v) grid to physical k–space coordinates.
    # Create arrays for kx and ky of shape (Nu, Nv)
    kx_grid = np.zeros((Nu, Nv))
    ky_grid = np.zeros((Nu, Nv))
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            k_pt = physical_k(u, v, K1, K2)
            kx_grid[i, j] = k_pt[0]
            ky_grid[i, j] = k_pt[1]

    # Plot the Berry curvature as a function of physical (kx, ky).
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    # Use pcolormesh for a 2D colormap. Since the grid is a parallelogram, pcolormesh handles it.
    plt.pcolormesh(kx_grid, ky_grid, Omega_grid, shading='auto', cmap='viridis')
    plt.colorbar(label='Berry Curvature Ω(k)')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.xlim(-K1[0], K1[0])
    plt.ylim(-K1[0], K1[0])
    plt.title('Berry Curvature in Physical k–Space')
    # aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def calculate_varying_B(r, w_dict, G1, G2):
    """
        Given a dictionary of weights w_b, compute the varying part of the magnetic field through the following equation:
        M(r) = e^{-2\tilde{\phi}(r)} = \sum_b w_b \exp(i b r)
        B(r) = \nabla^2 \tilde{phi}(r) = - (M\nabla^2 M - (\nabla M)^2)/(2M^2)= -(1/(2M^2)) \sum_b (b1-b2) b2 w_b1 w_b2 e^{i (b1+b2)r}
    
    
    Parameters:
     r : array-like, shape (2,)
          The position vector in real space where the magnetic field is evaluated.
     
      w_dict : dict
          Dictionary containing weights w_b for each reciprocal lattice vector b.

      G1, G2 : array-like, shape (2,)
          The two reciprocal lattice basis vectors.
    Returns:
      varying_part : float
          The computed varying part of the magnetic field.
    """
    # Placeholder implementation
    varying_part = 0.0
    M = 0.0
    M_numerator = 0.0
    for key, wb in w_dict.items():
        b = key[0] * G1 + key[1] * G2
        M += wb * np.exp(1j * np.dot(b, r))  # Assuming r is defined in the context
        for key2, wb2 in w_dict.items():
            b2 = key2[0] * G1 + key2[1] * G2
            M_numerator += np.dot((b - b2), b2) * wb * wb2 * np.exp(1j * np.dot(b + b2, r))
    
    varying_part = -M_numerator / (2 * M**2)
    return varying_part

# Example usage of the calculate_varying_B function
# --------------------------------------------------
if __name__=="__main__":
    K1 = np.array([1, 0.0]) * np.sqrt(4*np.pi/np.sqrt(3))
    K2 = np.array([-0.5, np.sqrt(3)/2]) * np.sqrt(4*np.pi/np.sqrt(3))
    print("Det(K1, K2)/2pi = ", np.linalg.det(np.column_stack((K1, K2)))/(2*np.pi))

    # real space lattice vectors
    # [a1, a2] is the inverse of [K1, K2]
    a1 = np.array([1.0, 1.0/np.sqrt(3)]) * 2*np.pi / np.sqrt(4*np.pi/np.sqrt(3))
    a2 = np.array([0.0, 2/np.sqrt(3)]) * 2*np.pi / np.sqrt(4*np.pi/np.sqrt(3))
    assert np.allclose(np.dot(a1, K1), 2*np.pi)
    assert np.allclose(np.dot(a2, K2), 2*np.pi)
    assert np.allclose(np.dot(a1, K2), 0)
    assert np.allclose(np.dot(a2, K1), 0)

    # Define weights w_b as before.
    w_dict = {
        (0, 0): 1.0,
        (1, 0): 0.1,
        (-1, 0): 0.1,
        (0, 1): 0.1,
        (0, -1): 0.1,
        (1, 1): 0.1,
        (-1, -1): 0.1,
    }
    # Define weights w_b as before.
    # w_dict = {
    #     (0, 0): 1.0,
    #     (1, 0): 0.,
    #     (-1, 0): 0.,
    #     (0, 1): 0.,
    #     (0, -1): 0.,
    #     (1, 1): 0.,
    #     (-1, -1): 0.,
    # }
    # Create a grid of points in real space
    Nx = 100
    Ny = 100
    # Create a grid of points in real space
    n1_vals  = np.linspace(-2, 2, Nx)
    n2_vals  = np.linspace(-2, 2, Ny)
    N1_vals, N2_vals = np.meshgrid(n1_vals, n2_vals)
    X, Y = a1[0]*N1_vals + a2[0]*N2_vals, a1[1]*N1_vals + a2[1]*N2_vals

    B_varying = np.zeros_like(X, dtype=complex)

    # Compute the varying part of the magnetic field at each point in real space
    for i in range(Nx):
        for j in range(Ny):
            r = np.array([X[i, j], Y[i, j]])
            B_varying[i, j] = calculate_varying_B(r, w_dict, K1, K2)

    print("average of B:", np.mean(B_varying))

    # Plot the varying part of the magnetic field as a color plot
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, np.real(B_varying), shading='auto', cmap='viridis')
    plt.colorbar(label='Varying Magnetic Field $B(r)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Varying Part of Magnetic Field in Real Space')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()



