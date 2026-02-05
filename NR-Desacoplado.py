import numpy as np

# ============================================================
# MÉTODO NEWTON–RAPHSON DESACOPLADO (FAST DECOUPLED LOAD FLOW)
# ============================================================

def newton_raphson_desacoplado(Ybus, slack, pv, pq, Psp, Qsp, V0, theta0, tol=1e-6, max_iter=50):
    """
    Método Newton–Raphson desacoplado (Fast Decoupled Load Flow).
    """
    n = len(Ybus)
    G = Ybus.real
    B = Ybus.imag

    # Índices útiles
    ang_idx = [i for i in range(n) if i != slack]
    pq_idx = pq.copy()

    # Variables
    V = V0.copy()
    theta = theta0.copy()

    # Submatrices desacopladas
    Bp = -B[np.ix_(ang_idx, ang_idx)]   # para Δθ
    Bpp = -B[np.ix_(pq_idx, pq_idx)]    # para ΔV

    print("-------------------------------------------------------------")
    print("Iter |  Error máx   |  Magnitudes de Voltaje (p.u.)  |  Ángulos (°)")
    print("-------------------------------------------------------------")

    for it in range(max_iter):
        # Cálculo de potencias P y Q
        Pcalc = np.zeros(n)
        Qcalc = np.zeros(n)
        for i in range(n):
            for j in range(n):
                Pcalc[i] += V[i]*V[j]*(G[i,j]*np.cos(theta[i]-theta[j]) + B[i,j]*np.sin(theta[i]-theta[j]))
                Qcalc[i] += V[i]*V[j]*(G[i,j]*np.sin(theta[i]-theta[j]) - B[i,j]*np.cos(theta[i]-theta[j]))

        # Desajustes
        dP = Psp[ang_idx] - Pcalc[ang_idx]
        dQ = Qsp[pq_idx] - Qcalc[pq_idx]

        mism = max(np.max(np.abs(dP)) if len(dP)>0 else 0,
                   np.max(np.abs(dQ)) if len(dQ)>0 else 0)

        # Mostrar resultados intermedios
        print(f"{it+1:3d}  |  {mism:10.3e}  |  " +
              "  ".join([f"{v:.4f}" for v in V]) +
              "  |  " +
              "  ".join([f"{np.degrees(t):.2f}" for t in theta]))

        # Criterio de convergencia
        if mism < tol:
            print("-------------------------------------------------------------")
            print(f"✅ Convergió en {it+1} iteraciones (error {mism:.3e})")
            break

        # 1️⃣ Corrección de ángulos
        rhsP = dP / V[ang_idx]
        dtheta = np.linalg.solve(Bp, rhsP)
        for k, i in enumerate(ang_idx):
            theta[i] += dtheta[k]

        # 2️⃣ Corrección de magnitudes (solo PQ)
        rhsQ = dQ / V[pq_idx]
        dV = np.linalg.solve(Bpp, rhsQ)
        for k, i in enumerate(pq_idx):
            V[i] += dV[k]

        # Mantener voltaje especificado en buses PV
        for i in pv:
            pass

    return V, theta


# ============================================================
# EJEMPLO DE USO
# ============================================================
Ybus = np.array([
    [complex(20,-50), complex(-10,20), complex(-10,30)],
    [complex(-10,20), complex(26,-52), complex(-16,32)],
    [complex(-10,30), complex(-16,32), complex(26,-62)]
], dtype=complex)

slack = 0
pv = [2]      # Bus 3 = PV
pq = [1]      # Bus 2 = PQ

Psp = np.array([0.0, -4.0, 2.0])    # P (p.u.)
Qsp = np.array([0.0, -2.5, 0.0])    # Q (p.u.)
V0 = np.array([1.05, 1.00, 1.04])   # Tensiones iniciales
theta0 = np.radians([0, 0, 0])      # Ángulos iniciales

# ============================================================
# EJECUCIÓN DEL MÉTODO
# ============================================================
Vf, thetaf = newton_raphson_desacoplado(Ybus, slack, pv, pq, Psp, Qsp, V0, theta0)

# ============================================================
# RESULTADOS FINALES
# ============================================================
print("\nResultados finales:")
for i in range(len(Vf)):
    print(f"Bus {i+1}: |V| = {Vf[i]:.4f} p.u., θ = {np.degrees(thetaf[i]):.4f}°")
