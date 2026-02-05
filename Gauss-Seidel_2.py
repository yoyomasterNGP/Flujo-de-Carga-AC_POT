import numpy as np
import pandas as pd

def gauss_seidel_power_flow(Ybus, bus_type, P, Q, V_spec, theta_spec,
                            V_init, Qmin, Qmax, tol=1e-6, max_iter=100, w=1.0):
    nbus = len(bus_type)
    V = V_init.copy()
    convergido = False
    history = []

    # Guardamos los tipos originales para no modificarlos permanentemente
    bus_type_original = bus_type.copy()

    for it in range(max_iter):
        V_prev = V.copy()
        # Reiniciamos tipos al comienzo de cada iteración
        bus_type_iter = bus_type_original.copy()
        Q_iter = np.array(Q, dtype=float)  # copia de Qi (para la tabla)

        # Restaurar magnitud de PV originales (si se habían modificado en iter anterior)
        for i in range(nbus):
            if bus_type_iter[i] == 2:  # si es PV en el arreglo original
                if abs(V[i]) == 0:
                    V[i] = V_spec[i]  # defensa por si V estaba en 0
                else:
                    V[i] = V_spec[i] * V[i] / abs(V[i])

        buses_saturados = []  # indices de PV que se saturan y se tratan como PQ en esta iter

        for i in range(nbus):
            if bus_type_iter[i] == 1:  # Slack
                Q_iter[i] = 0.0
                continue

            # Paso 1: Si es PV (según tipo original en bus_type_iter) -> calcular Q primero
            if bus_type_iter[i] == 2:
                I_i = np.dot(Ybus[i, :], V)
                S_i_calc = V[i] * np.conj(I_i)
                Qcalc = S_i_calc.imag
                Q_iter[i] = Qcalc

                # Si sale de límites, lo tratamos temporalmente como PQ (solo en bus_type_iter)
                if Qcalc < Qmin[i]:
                    Q[i] = Qmin[i]
                    bus_type_iter[i] = 3
                    buses_saturados.append(i)
                elif Qcalc > Qmax[i]:
                    Q[i] = Qmax[i]
                    bus_type_iter[i] = 3
                    buses_saturados.append(i)
                else:
                    Q[i] = Qcalc

            # Paso 2: actualizar V con el Q más reciente (sea PV con Q calculado o PQ con Q dado)
            Si = complex(P[i], Q[i])
            sumYV = np.dot(Ybus[i, :], V) - Ybus[i, i] * V[i]
            V_new = (1 / Ybus[i, i]) * ((Si.conjugate() / V[i].conjugate()) - sumYV)
            V[i] = (1 - w) * V[i] + w * V_new

            # Paso 3: si en esta iteración sigue siendo PV (no temporalmente convertido), normalizar magnitud
            if bus_type_iter[i] == 2:
                # evitar división por cero
                if abs(V[i]) == 0:
                    V[i] = V_spec[i]
                else:
                    V[i] = V_spec[i] * V[i] / abs(V[i])

        # Paso 4: para PQ verdaderos (y para PV que fueron tratados como PQ temporalmente ya pusimos Q=limit)
        # No recalculamos Q de los PQ originales (si quieres mantener Q fijo de PQ), 
        # pero para los PV que se convirtieron temporalmente en PQ queremos guardar su Q resultante:
        for i in buses_saturados:
            # calcular Qi resultante de ese bus tratado ahora como PQ
            I_i = np.dot(Ybus[i, :], V)
            S_i_calc = V[i] * np.conj(I_i)
            Q_iter[i] = S_i_calc.imag
            # Nota: Q[i] ya fue fijado a Qmin o Qmax para la actualización de V en esta iteración.

        # Cálculo de errores (valores absolutos; más seguro que dividir por ángulo/magnitud)
        Vmag_prev = np.abs(V_prev)
        Vang_prev = np.angle(V_prev, deg=True)
        Vmag = np.abs(V)
        Vang = np.angle(V, deg=True)

        error_V_bus = np.abs(Vmag - Vmag_prev)
        error_ang_bus = np.abs(Vang - Vang_prev)

        error_global = max(np.nanmax(error_V_bus), np.nanmax(error_ang_bus))

        # Guardar historial (Iter, |V|..., θ..., errores magnitud..., errores angulo..., error_global)
        history.append([it+1, *Vmag, *Vang, *error_V_bus, *error_ang_bus, error_global])

        if error_global < tol:
            convergido = True
            break

    # Crear DataFrame con las columnas adecuadas
    cols_mag = [f"|V{i+1}|" for i in range(nbus)]
    cols_ang = [f"θ{i+1}(°)" for i in range(nbus)]
    cols_errV = [f"Err|V{i+1}|" for i in range(nbus)]
    cols_errAng = [f"Errθ{i+1}" for i in range(nbus)]

    df = pd.DataFrame(history, columns=["Iter"] + cols_mag + cols_ang + cols_errV + cols_errAng + ["Error_global"])

    return V, it+1, convergido, df


# Ejemplo rápido (copia tu Ybus y datos reales)
if __name__ == "__main__":
    Ybus = np.array([
        [3-9j, -2+6j, -1+3j, 0],
        [-2+6j, 11/3-11j, -2/3+2j, -1+3j],
        [-1+3j, -2/3+2j, 11/3-11j, -2+6j],
        [0, -1+3j, -2+6j, 3-9j]
    ], dtype=complex)

    bus_type = [1, 2, 3, 3]    # 1 Slack, 2 PV, 3 PQ
    P = np.array([0.0, 0.5, -1, 0.3])
    Q = np.array([0.0, 0.0, 0.5, -0.1])
    V_spec = np.array([1.04, 1.04, 0.0, 0.0])
    theta_spec = np.array([0.0, 0.0, 0.0, 0.0])
    V_init = np.array([1.04+0j, 1.04+0j, 1.0+0j, 1.0+0j], dtype=complex)
    Qmin = np.array([0.0, 0.25, -999, -999])
    Qmax = np.array([0.0, 1.0, 999, 999])

    V_final, n_iter, conv, tabla = gauss_seidel_power_flow(
        Ybus, bus_type, P, Q, V_spec, theta_spec, V_init, Qmin, Qmax,
        tol=1e-6, max_iter=100, w=1
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', None)

    print("\n--- Tabla de resultados por iteración ---")
    print(tabla.round(6))

    print("\n--- Tensiones finales ---")
    for i, v in enumerate(V_final):
        print(f"Bus {i+1}: |V|={abs(v):.4f} ∠ {np.angle(v, deg=True):.2f}°")

    print(f"\nIteraciones: {n_iter}, Convergió: {conv}")
    print("SIUUUUUUUU")
