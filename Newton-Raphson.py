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
# FLUJO DE POTENCIA NEWTON-RAPHSON (polar)
# ------------------------------------------------------------
def newton_raphson_polar(Ybus, slack, pv, pq, Psp, Qsp, V0, theta0, tol=1e-6, max_iter=20):
    n = len(V0)
    V = V0.copy()
    theta = theta0.copy()
    Ymag, Yang = polar_Ybus(Ybus)

    print("\nIter | Error máx      | Voltajes (pu)               | Ángulos (°)")
    print("-----|----------------|-----------------------------|-----------------------------")

    for it in range(max_iter):
        Pcalc, Qcalc = calc_power_polar(V, theta, Ymag, Yang)

        dP = Psp[1:] - Pcalc[1:]
        dQ = Qsp[pq] - Qcalc[pq]
        mismatch = np.concatenate((dP, dQ))

        error = np.max(np.abs(mismatch))
        print(f"{it:4d} | {error:1.3e} | {np.array2string(V, precision=4)} | {np.degrees(theta)}")

        if error < tol:
            print(f"\n✅ Convergió en {it} iteraciones.\n")
            return V, theta

        J = build_jacobian_polar(V, theta, Ymag, Yang, pq, pv)
        dx = np.linalg.solve(J, mismatch)

        ntheta = n-1
        dTheta = dx[:ntheta]
        dV = dx[ntheta:]

        for i in range(1,n):
            theta[i] += dTheta[i-1]
        for idx, bus in enumerate(pq):
            V[bus] += dV[idx]

    print("\n⚠️ No convergió tras el máximo de iteraciones.\n")
    return V, theta


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

Psp = np.array([0, -4, 2])    # P (pu)
Qsp = np.array([0, -2.5, 0])  # Q (pu)
V0 = np.array([1.05, 1.00, 1.04])
theta0 = np.radians([0, 0, 0])

Vf, thetaf = newton_raphson_polar(Ybus, slack, pv, pq, Psp, Qsp, V0, theta0)

print("Resultados finales:")
for i in range(len(Vf)):
    print(f"Bus {i+1}: |V| = {Vf[i]:.4f} pu, θ = {np.degrees(thetaf[i]):.4f}°")
