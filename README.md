# TFG_PINNs

Este repositorio contiene el código desarrollado para el Trabajo Fin de Grado sobre Physics-Informed Neural Networks (PINNs) aplicado a la resolución de ecuaciones diferenciales en física.

Incluye implementaciones de PINNs para los siguientes sistemas:
- Ecuación diferencial ordinaria simple (EDO)
- Ecuación de Schrödinger independiente del tiempo (TISE)
- Ecuación de Schrödinger dependiente del tiempo (TDSE)
- Átomo de hidrógeno en 1D

Para la EDO simple, cada archivo incluye la construcción de la clase PINN, su arquitectura, el proceso de entrenamiento y la evaluación.

En los casos de TISE, TDSE e hidrógeno 1D, la estructura es la siguiente:
- `pinn.py`: definición de la clase y arquitectura de la PINN
- `adap.py`: entrenamiento con malla adaptativa
- `fija.py`: entrenamiento con malla fija
- `eval.py`: evaluación de resultados

**Nota importante:**  
En el caso de la TISE, los estados deben resolverse en orden creciente (primero n = 0, luego n = 1, etc.), ya que para imponer la ortogonalidad es necesario disponer de los estados anteriores.
