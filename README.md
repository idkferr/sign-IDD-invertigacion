# Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production 

Este repositorio es una versión optimizada y corregida del proyecto oficial [Sign-IDD](https://github.com/NaVi-start/Sign-IDD). Se ha realizado una reestructuración del código para permitir el entrenamiento en hardware **AMD/Intel** mediante **DirectML**, solucionando incompatibilidades críticas de la versión original basada exclusivamente en NVIDIA/CUDA.

## Notas sobre la Instalación y Dependencias

Durante el proceso de configuración inicial, se detectaron y corrigieron problemas críticos en el archivo `requirements.txt` original:

* **Incompatibilidad de Paquetes:** Se presentaron múltiples errores relacionados con la configuración de dependencias que impedían la instalación mediante `pip`.
* **Depuración de Requisitos:** Se eliminaron paquetes de sistema (como `bzip2` y `ca-certificates`) que no deben gestionarse a través de Python, permitiendo una instalación limpia en entornos virtuales (`venv`).
* **Actualización de Seguridad:** Se actualizó el paquete `certifi` para resolver conflictos de dependencias y asegurar la comunicación correcta con los repositorios de modelos.



---

##  Resumen de Impacto
Gracias a estas modificaciones, el proyecto ahora es **completamente reproducible** en máquinas con Windows 10/11 y hardware diverso (AMD/Intel/NVIDIA), superando las barreras de compatibilidad del repositorio oficial.

##  Archivos Modificados y Justificación Técnica

Para lograr la estabilidad del sistema en entornos Windows con GPUs no-NVIDIA, se intervinieron los siguientes módulos centrales:

| Archivo | Función | Modificación Realizada |
| :--- | :--- | :--- |
| **`__main__.py`** | Punto de entrada | Limpieza de advertencias (warnings) y validación de rutas en Windows. |
| **`ACD.py`** | Núcleo de Difusión | Abstracción de `device` (reemplazo de `cuda` por `self.device`) y soporte DirectML. |
| **`batch.py`** | Procesamiento de Datos | Implementación de `_make_device` para carga dinámica en memoria de GPU AMD. |
| **`builders.py`** | Constructores | Ajuste de optimizadores y schedulers para una ejecución más limpia. |
| **`helpers.py`** | Utilidades | Modificación de `load_checkpoint` para carga dinámica de pesos (CPU/GPU). |
| **`loss.py`** | Función de Pérdida | Implementación de épsilon fijo ($10^{-8}$) para evitar colapsos numéricos en DirectML. |
| **`training.py`** | Gestor de Entrenamiento | Mejora en la persistencia de datos y guardado seguro de esqueletos (.skels). |
| **`vocabulary.py`** | Vocabulario | Forzado de `encoding="utf-8"` para soporte universal de caracteres especiales. |



---

##  Configuración del Entorno

### Requisitos Técnicos
* **Python:** 3.11 (Requerido para compatibilidad con `torch-directml`).
* **Hardware:** GPU compatible con DirectX 12 (AMD, Intel, NVIDIA) o CPU.

### Instalación
1. **Crear Entorno Virtual:**
   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate

### Instalar Dependencias
Se ha depurado el archivo requirements.txt eliminando dependencias de sistema (bzip2, ca-certificates) e integrando el soporte para DirectML.
```bash
pip install -r requirements.txt
```

### Preparación de Datos
El proyecto utiliza la estructura de datos del modelo [Progressive Transformer](https://github.com/BenSaunders27/ProgressiveTransformersSLP.git)

```bash
Extraer los datos en la ruta: ./Data/P2014T_Ben/
```
Asegurar que los archivos de vocabulario y metadatos (.csv) estén en: ./Configs/

### Training
Para iniciar el proceso de entrenamiento con detección automática de hardware:

```bash

python __main__.py train ./Configs/Sign-IDD.yaml
```

### Inference
```bash
python __main__.py test ./Configs/Sign-IDD.yaml
```

### Problemas Solucionados
Incompatibilidad de GPU: El código original fallaba en ausencia de drivers CUDA. Se implementó una lógica de selección jerárquica: DirectML > CUDA > CPU.

Errores de Precisión: Las GPUs AMD presentaban errores de "NaN" en el cálculo de pérdida ósea. Se solucionó mediante el uso de épsilon fijo y casting explícito a float32.

Errores de Codificación: Se eliminaron los fallos de lectura de glosas con caracteres especiales en Windows mediante el forzado de UTF-8.

