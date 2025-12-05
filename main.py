from typing import List, Tuple


def formato_vector(valores, decimales: int = 4) -> str:
    """Devuelve una lista formateada con n decimales."""
    piezas = []
    for valor in valores:
        if isinstance(valor, float):
            piezas.append(f"{valor:.{decimales}f}")
        else:
            piezas.append(str(valor))
    return "[" + ", ".join(piezas) + "]"


def imprimir_titulo(titulo: str, decorador: str = "=") -> None:
    linea = decorador * len(titulo)
    print(f"\n{linea}\n{titulo}\n{linea}")


def imprimir_subtitulo(texto: str) -> None:
    linea = "-" * len(texto)
    print(f"\n{texto}\n{linea}")


def normalizar_min_max(dataset: List[List[float]]):
    """
    Normaliza un dataset usando la normalizacion Min-Max.
    Cada columna se transforma al rango [0, 1].
    """
    columnas = list(zip(*dataset))
    min_por_columna = [min(col) for col in columnas]
    max_por_columna = [max(col) for col in columnas]

    dataset_normalizado = []
    for fila in dataset:
        fila_normalizada = []
        for indice_columna, valor in enumerate(fila):
            minimo = min_por_columna[indice_columna]
            maximo = max_por_columna[indice_columna]
            rango = maximo - minimo

            if rango == 0:
                valor_normalizado = 0.0
            else:
                valor_normalizado = (valor - minimo) / rango

            fila_normalizada.append(valor_normalizado)
        dataset_normalizado.append(fila_normalizada)

    return dataset_normalizado


def calcular_salida(entradas: List[float], pesos: List[float], sesgo: float) -> float:
    """Calcula la salida del perceptron antes de la funcion de activacion."""
    if len(entradas) != len(pesos):
        raise ValueError("El numero de entradas no coincide con el numero de pesos.")

    salida_lineal = 0.0
    for entrada, peso in zip(entradas, pesos):
        salida_lineal += entrada * peso

    return salida_lineal + sesgo


def actualizar_pesos(pesos: List[float], entradas: List[float], error: float, tasa_aprendizaje: float) -> None:
    """Actualiza los pesos segun la regla clasica del perceptron."""
    for i in range(len(pesos)):
        pesos[i] += tasa_aprendizaje * error * entradas[i]


def actualizar_sesgo(sesgo: float, error: float, tasa_aprendizaje: float) -> float:
    """Actualiza el sesgo del perceptron."""
    return sesgo + tasa_aprendizaje * error


def funcion_activacion(salida_lineal: float) -> int:
    """Funcion de activacion umbral."""
    return 1 if salida_lineal >= 0 else 0


def entrenar_perceptron(
    conjunto_entrenamiento: List[List[float]],
    salidas_esperadas: List[int],
    pesos: List[float],
    sesgo: float,
    tasa_aprendizaje: float,
    epocas: int,
) -> Tuple[List[float], float, list]:
    """Entrena un perceptron simple usando el algoritmo clasico."""
    registro_pesos = []

    imprimir_titulo("Entrenamiento del perceptron")

    for numero_epoca in range(epocas):
        imprimir_subtitulo(f"Epoca {numero_epoca + 1} / {epocas}")

        for indice_muestra, entradas in enumerate(conjunto_entrenamiento):
            salida_lineal = calcular_salida(entradas, pesos, sesgo)
            prediccion = funcion_activacion(salida_lineal)
            error = salidas_esperadas[indice_muestra] - prediccion

            print(
                f"  Muestra {indice_muestra + 1:02d} | entradas={formato_vector(entradas, decimales=3)} | "
                f"salida={salida_lineal:.3f} | prediccion={prediccion} | "
                f"esperado={salidas_esperadas[indice_muestra]} | error={error:+d}"
            )

            if error != 0:
                actualizar_pesos(pesos, entradas, error, tasa_aprendizaje)
                sesgo = actualizar_sesgo(sesgo, error, tasa_aprendizaje)

        registro_pesos.append((pesos.copy(), sesgo))
        print(f"  -> Pesos al final de la epoca: {formato_vector(pesos)} | Sesgo: {sesgo:.4f}")

    return pesos, sesgo, registro_pesos


def predecir(entradas: List[List[float]], pesos: List[float], sesgo: float):
    """Realiza una prediccion usando el perceptron entrenado."""
    predicciones = []
    for muestra in entradas:
        salida_lineal = calcular_salida(muestra, pesos, sesgo)
        prediccion = funcion_activacion(salida_lineal)
        predicciones.append(prediccion)
    return predicciones


def mostrar_predicciones_detalladas(
    nuevas_entradas,
    nuevas_entradas_normalizadas,
    predicciones,
    nuevas_salidas_esperadas,
    pesos_finales,
    sesgo_final,
):
    """Muestra los resultados de las predicciones con formato legible."""
    imprimir_titulo("Predicciones detalladas")

    for i, (entrada, entrada_norm, prediccion, esperado) in enumerate(
        zip(nuevas_entradas, nuevas_entradas_normalizadas, predicciones, nuevas_salidas_esperadas), 1
    ):
        salida_lineal = calcular_salida(entrada_norm, pesos_finales, sesgo_final)

        imprimir_subtitulo(f"Muestra {i}")
        print(f"  Entradas originales  : {formato_vector(entrada, decimales=0)}")
        print(f"  Entradas normalizadas: {formato_vector(entrada_norm)}")
        print(f"  Pesos finales        : {formato_vector(pesos_finales)}")
        print(f"  Sesgo final          : {sesgo_final:.6f}")
        print(f"  Salida lineal        : {salida_lineal:.6f}")
        print(f"  Esperado             : {esperado}")
        print(f"  Prediccion           : {prediccion}")

        clase_pred = "Clase 1" if prediccion == 1 else "Clase 0"
        clase_esp = "Clase 1" if esperado == 1 else "Clase 0"
        print(f"  Clasificacion        : Esperado={clase_esp} | Prediccion={clase_pred}")

        print(f"\n  Calculo detallado:")
        print(f"    salida = sum(entrada_i * peso_i) + sesgo")
        calculo = " + ".join([f"({x:.4f} * {w:.4f})" for x, w in zip(entrada_norm, pesos_finales)])
        print(f"    {calculo} + {sesgo_final:.4f} = {salida_lineal:.4f}")

        print(f"\n  Regla de decision:")
        print(f"    Si salida_lineal >= 0 -> 1")
        print(f"    Si salida_lineal < 0  -> 0")
        print(
            f"    {salida_lineal:.4f} es "
            f"{'mayor o igual' if salida_lineal >= 0 else 'menor'} que 0 -> {prediccion}"
        )


if __name__ == "__main__":
    imprimir_titulo("Configuracion de entrenamiento")

    conjunto_entrenamiento = [
        [85, 2, 80],
        [40, 8, 55],
        [65, 4, 65],
        [90, 1, 85],
    ]
    print(f"Conjunto de entrenamiento: {conjunto_entrenamiento}")

    conjunto_entrenamiento_normalizado = normalizar_min_max(conjunto_entrenamiento)
    salidas_esperadas = [1, 0, 0, 1]

    pesos_iniciales = [0.3, 0.1, 0.2]
    sesgo_inicial = 0.1
    tasa_aprendizaje = 0.1
    epocas = 10

    print(f"Pesos iniciales: {formato_vector(pesos_iniciales)}")
    print(f"Sesgo inicial  : {sesgo_inicial:.4f}")
    print(f"Tasa aprendizaje: {tasa_aprendizaje}")
    print(f"Epocas          : {epocas}")

    pesos_finales, sesgo_final, registro_pesos = entrenar_perceptron(
        conjunto_entrenamiento_normalizado,
        salidas_esperadas,
        pesos_iniciales,
        sesgo_inicial,
        tasa_aprendizaje,
        epocas,
    )

    imprimir_titulo("Resumen de entrenamiento")
    print(f"Pesos finales: {formato_vector(pesos_finales)}")
    print(f"Sesgo final  : {sesgo_final:.4f}")
    print("\nEvolucion por epoca:")
    for epoca, (pesos, sesgo) in enumerate(registro_pesos, start=1):
        print(f"  Epoca {epoca:02d}: Pesos={formato_vector(pesos)} | Sesgo={sesgo:.4f}")

    nuevas_entradas = [
        [75, 3, 70],
        [50, 6, 60],
        [30, 9, 40],
        [95, 0, 90],
    ]
    salidas_esperadas = [1, 0, 0, 1]

    nuevas_entradas_normalizadas = normalizar_min_max(nuevas_entradas)
    predicciones = predecir(nuevas_entradas_normalizadas, pesos_finales, sesgo_final)

    nuevas_salidas_esperadas = salidas_esperadas

    mostrar_predicciones_detalladas(
        nuevas_entradas,
        nuevas_entradas_normalizadas,
        predicciones,
        nuevas_salidas_esperadas,
        pesos_finales,
        sesgo_final,
    )
