import numpy as np
import matplotlib.pyplot as plt

def compute_lattice_vectors():
    """
    Returns real-space lattice vectors a1, a2 and reciprocal vectors G1, G2. 
    The lattice is the triangular lattice, and its unit cell area in real space is 2pi. Correspondingly, its area of 1st BZ is also 2pi. 

    Note that the unit of length is the magnetic length l_B = 1. 
    """
    # Suppose your reciprocal unit cell is defined by the (possibly non–orthogonal)
    K1 = np.array([1, 0.0]) * np.sqrt(4*np.pi/np.sqrt(3))
    K2 = np.array([-0.5, np.sqrt(3)/2]) * np.sqrt(4*np.pi/np.sqrt(3))
    # print("Det(K1, K2)/2pi = ", np.linalg.det(np.column_stack((K1, K2)))/(2*np.pi))
    K3 = -K1 - K2
    # real space lattice vectors
    # [a1, a2] is the inverse of [K1, K2]
    a1 = np.array([1.0, 1.0/np.sqrt(3)]) * 2*np.pi / np.sqrt(4*np.pi/np.sqrt(3))
    a2 = np.array([0.0, 2/np.sqrt(3)]) * 2*np.pi / np.sqrt(4*np.pi/np.sqrt(3))

    assert np.allclose(np.dot(a1, K1), 2*np.pi)
    assert np.allclose(np.dot(a2, K2), 2*np.pi)
    assert np.allclose(np.dot(a1, K2), 0)
    assert np.allclose(np.dot(a2, K1), 0)
    assert np.allclose(np.linalg.det(np.column_stack((a1, a2))), 2*np.pi)
    assert np.dot(K1, K1) == np.dot(K2, K2)
    assert np.dot(K2, K2) == np.dot(K3, K3)
    G1 = K1
    G2 = K2
    return a1, a2, G1, G2

# Define the magnetic field
def B_func(x, y, B1):
    a1, a2, G1, G2 = compute_lattice_vectors()
    K1 = G1
    K2 = G2
    K3 = -K1 - K2

    B = B1*(np.cos(K1[0]*x + K1[1]*y) + np.cos(K2[0]*x + K2[1]*y) + np.cos(K3[0]*x + K3[1]*y))
    return B

def M_func(x, y, B1):
    """
    M(r) = e^{-2\phi(r)} is a periodic function defined on the unit cell with peridocity given by a1 and a2.
    \phi(r) is related to the non-uniform magnetic field as B(r)=\nabla^2 \phi(r). 
    """
    a1, a2, G1, G2 = compute_lattice_vectors()
    K1 = G1
    K2 = G2
    K3 = -K1 - K2
    B = B_func(x, y, B1)
    phi = -B/(np.dot(K1, K1))
    return np.exp(-2*phi) 

def fourier_truncation_approx(M_func, a1, a2, Mmax, Nmax, Nu=11, Nv=11, plot=False):
    """
    Compute Fourier coefficients of M_func on the lattice spanned by a1, a2,
    truncate to |m|≤Mmax, |n|≤Nmax, reconstruct the approximation, and optionally plot.

    Parameters
    ----------
    M_func : callable
        Function M(x, y) defined on the unit cell.
    a1, a2 : array_like, shape (2,)
        Real‐space lattice vectors.
    Mmax, Nmax : int
        Truncation limits for Fourier modes in the two lattice directions.
    Nu, Nv : int, optional
        Number of sample points along each lattice coordinate (default 11).
    plot : bool, optional
        If True, display side‐by‐side colorplots of original and truncated functions.

    Returns
    -------
    w_dict : dict
        Dictionary of Fourier coefficients {(n, m): c_{m,n}}.
    M_approx : callable
        Reconstructed approximation function M_approx(x, y).
    err : float
        Maximum absolute error between original and approximation on the sample grid.
    X, Y, f, f_approx : ndarray
        Sample grid coordinates, original sampled M, and its truncated approximation.
    """
    # Reciprocal lattice
    A = np.column_stack((a1, a2))
    Gmat = 2 * np.pi * np.linalg.inv(A)
    G1, G2 = Gmat[0], Gmat[1]

    # Sample grid in lattice coords u, v ∈ [0,1)
    u = np.arange(Nu) / Nu
    v = np.arange(Nv) / Nv
    U, V = np.meshgrid(u, v, indexing='ij')
    X = a1[0]*U + a2[0]*V
    Y = a1[1]*U + a2[1]*V

    # Original sampled values
    f = M_func(X, Y)

    # FFT and extract low‐frequency modes
    F = np.fft.fft2(f) / (Nu * Nv)
    w_dict = {}
    for m in range(-Mmax, Mmax+1):
        for n in range(-Nmax, Nmax+1):
            idx_m = m % Nu
            idx_n = n % Nv
            w_dict[(n, m)] = F[idx_m, idx_n]

    # Reconstruction function
    def M_approx(x, y):
        val = 0+0j
        for (n, m), coeff in w_dict.items():
            kx, ky = m*G1 + n*G2
            val += coeff * np.exp(1j * (kx*x + ky*y))
        return val.real

    # Compute truncated approximation on the grid
    f_approx = np.vectorize(M_approx)(X, Y)

    # Maximum error
    err = np.max(np.abs(f - f_approx))

    # # Plot if requested
    # if plot:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #     pcm1 = ax1.pcolormesh(X, Y, f, shading='auto')
    #     ax1.set_title('Original M(x,y)')
    #     ax1.set_xlabel('x'); ax1.set_ylabel('y')
    #     fig.colorbar(pcm1, ax=ax1)
    #     pcm2 = ax2.pcolormesh(X, Y, f_approx, shading='auto')
    #     ax2.set_title(f'Truncated Fourier (|m|,|n|≤{Mmax})')
    #     ax2.set_xlabel('x'); ax2.set_ylabel('y')
    #     fig.colorbar(pcm2, ax=ax2)
    #     plt.tight_layout()
    #     plt.show()
    #     print("Max error:", err)

    return w_dict, M_approx, err, X, Y, f, f_approx

if __name__ == "__main__":
    # Example usage:
    # Define lattice vectors K1,K2 then compute a1,a2 as in your setup, and M_func(x,y).
    # w_dict, M_approx, err, X, Y, f, f_approx = fourier_truncation_approx(M, a1, a2, Mmax=3, Nmax=3)

    # K1 = np.array([1, 0.0]) * np.sqrt(4*np.pi/np.sqrt(3))
    # K2 = np.array([-0.5, np.sqrt(3)/2]) * np.sqrt(4*np.pi/np.sqrt(3))
    # print("Det(K1, K2)/2pi = ", np.linalg.det(np.column_stack((K1, K2)))/(2*np.pi))
    # K3 = -K1 - K2
    # assert np.dot(K1, K1) == np.dot(K2, K2)
    # assert np.dot(K2, K2) == np.dot(K3, K3)
    # G1 = K1
    # G2 = K2

    # # real space lattice vectors
    # # [a1, a2] is the inverse of [K1, K2]
    # a1 = np.array([1.0, 1.0/np.sqrt(3)]) * 2*np.pi / np.sqrt(4*np.pi/np.sqrt(3))
    # a2 = np.array([0.0, 2/np.sqrt(3)]) * 2*np.pi / np.sqrt(4*np.pi/np.sqrt(3))

    # assert np.allclose(np.dot(a1, K1), 2*np.pi)
    # assert np.allclose(np.dot(a2, K2), 2*np.pi)
    # assert np.allclose(np.dot(a1, K2), 0)
    # assert np.allclose(np.dot(a2, K1), 0)
    a1, a2, G1, G2 = compute_lattice_vectors()
    K1 = G1
    K2 = G2
    K3 = -K1 - K2

    B1 = 1.0
    # 2) Define your function M(x,y)
    def M(x, y):
        """
        M(r) = e^{-2\phi(r)} is a periodic function defined on the unit cell with peridocity given by a1 and a2.
        \phi(r) is related to the non-uniform magnetic field as B(r)=\nabla^2 \phi(r). 
        """
        B = B1*(np.cos(K1[0]*x + K1[1]*y) + np.cos(K2[0]*x + K2[1]*y) + np.cos(K3[0]*x + K3[1]*y))
        phi = -B/(np.dot(K1, K1))
        return np.exp(-2*phi) 

    w_dict, M_approx, err, X, Y, f, f_approx = fourier_truncation_approx(M, a1, a2, Mmax=3, Nmax=3)
    print("Max error:", err)