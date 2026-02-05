import numpy as np

# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------
def polar_Ybus(Ybus):
    """Convierte Ybus cartesiana en magnitud y ángulo (radianes)."""
    Ymag = np.abs(Ybus)
    Yang = np.angle(Ybus)
    return Ymag, Yang


def calc_power_polar(V, theta, Ymag, Yang):
    """Calcula P_i y Q_i usando Ybus en forma polar."""
    n = len(V)
    P = np.zeros(n)
    Q = np.zeros(n)
    for i in range(n):
        for j in range(n):
            P[i] += V[i]*V[j]*Ymag[i,j]*np.cos(theta[i] - theta[j] - Yang[i,j])
            Q[i] += V[i]*V[j]*Ymag[i,j]*np.sin(theta[i] - theta[j] - Yang[i,j])
    return P, Q


def build_jacobian_polar(V, theta, Ymag, Yang, pq, pv):
    """Jacobiano del método Newton-Raphson en forma polar."""
    n = len(V)
    npq = len(pq)

    J11 = np.zeros((n-1, n-1))  # dP/dθ
    J12 = np.zeros((n-1, npq))  # dP/dV
    J21 = np.zeros((npq, n-1))  # dQ/dθ
    J22 = np.zeros((npq, npq))  # dQ/dV

    # --- dP/dθ ---
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                for k in range(n):
                    if k != i:
                        J11[i-1, j-1] -= V[i]*V[k]*Ymag[i,k]*np.sin(theta[i]-theta[k]-Yang[i,k])
            else:
                J11[i-1, j-1] = V[i]*V[j]*Ymag[i,j]*np.sin(theta[i]-theta[j]-Yang[i,j])

    # --- dP/dV ---
    for i in range(1, n):
        for j_idx, j in enumerate(pq):
            if i == j:
                J12[i-1, j_idx] = 2*V[i]*Ymag[i,i]*np.cos(-Yang[i,i])
                for k in range(n):
                    if k != i:
                        J12[i-1, j_idx] += V[k]*Ymag[i,k]*np.cos(theta[i]-theta[k]-Yang[i,k])
            else:
                J12[i-1, j_idx] = V[i]*Ymag[i,j]*np.cos(theta[i]-theta[j]-Yang[i,j])

    # --- dQ/dθ ---
    for i_idx, i in enumerate(pq):
        for j in range(1, n):
            if i == j:
                for k in range(n):
                    if k != i:
                        J21[i_idx, j-1] += V[i]*V[k]*Ymag[i,k]*np.cos(theta[i]-theta[k]-Yang[i,k])
            else:
                J21[i_idx, j-1] = -V[i]*V[j]*Ymag[i,j]*np.cos(theta[i]-theta[j]-Yang[i,j])

    # --- dQ/dV ---
    for i_idx, i in enumerate(pq):
        for j_idx, j in enumerate(pq):
            if i == j:
                J22[i_idx, j_idx] = 2*V[i]*Ymag[i,i]*np.sin(-Yang[i,i])
                for k in range(n):
                    if k != i:
                        J22[i_idx, j_idx] += V[k]*Ymag[i,k]*np.sin(theta[i]-theta[k]-Yang[i,k])
            else:
                J22[i_idx, j_idx] = V[i]*Ymag[i,j]*np.sin(theta[i]-theta[j]-Yang[i,j])

    # Ensamblar Jacobiano
    J = np.vstack([
        np.hstack((J11, J12)),
        np.hstack((J21, J22))
    ])
    return J


# ------------------------------------------------------------
# UNA SOLA ITERACIÓN CON JACOBIANO
# ------------------------------------------------------------
def newton_raphson_one_iter(Ybus, slack, pv, pq, Psp, Qsp, V0, theta0):
    Ymag, Yang = polar_Ybus(Ybus)
    V = V0.copy()
    theta = theta0.copy()
    n = len(V)

    # Calcular potencias
    Pcalc, Qcalc = calc_power_polar(V, theta, Ymag, Yang)

    # Desbalances
    dP = Psp[1:] - Pcalc[1:]
    dQ = Qsp[pq] - Qcalc[pq]
    mismatch = np.concatenate((dP, dQ))

    # Mostrar resultados de potencias
    print("Potencias calculadas (P y Q):")
    for i in range(n):
        print(f"Bus {i+1}:  P = {Pcalc[i]:.6f}  |  Q = {Qcalc[i]:.6f}")
    print("\nDesbalances (ΔP y ΔQ):")
    print(mismatch, "\n")

    # Construir Jacobiano
    J = build_jacobian_polar(V, theta, Ymag, Yang, pq, pv)

    # Resolver el sistema lineal J * dx = mismatch
    dx = np.linalg.solve(J, mismatch)

    # Separar Δθ y ΔV
    dTheta = dx[:n-1]
    dV = dx[n-1:]

    # Actualizar variables
    theta_new = theta.copy()
    V_new = V.copy()
    theta_new[1:] += dTheta
    for i, bus in enumerate(pq):
        V_new[bus] += dV[i]

    # Mostrar resultados
    print("Matriz Jacobiana (forma polar):\n")
    np.set_printoptions(precision=5, suppress=True)
    print(J)
    print("\nΔx (Δθ, ΔV):")
    print(dx)
    print("\nΔθ (en grados):", np.degrees(dTheta))
    print("ΔV:", dV)
    print("\nValores actualizados:")
    for i in range(n):
        print(f"Bus {i+1}:  V = {V_new[i]:.6f}  |  θ = {np.degrees(theta_new[i]):.6f}°")

    return J, mismatch, dx, V_new, theta_new



# ------------------------------------------------------------
# EJEMPLO DE USO
# ------------------------------------------------------------
Ybus = np.array([
    [complex(20,-50), complex(-10,20), complex(-10,30)],
    [complex(-10,20), complex(26,-52), complex(-16,32)],
    [complex(-10,30), complex(-16,32), complex(26,-62)]
], dtype=complex)

slack = 0
pv = [2]     # bus 3 = PV
pq = [1]     # bus 2 = PQ

Psp = np.array([0, -4, 2])
Qsp = np.array([0, -2.5, 0])
V0 = np.array([1.05, 1.00, 1.04])
theta0 = np.radians([0, 0, 0])

J, mismatch = newton_raphson_one_iter(Ybus, slack, pv, pq, Psp, Qsp, V0, theta0)
