# Proyecto MLOps

## 1. Setup del proyecto

### 1.1 Pasos Previos

#### _WINDOWS_

- Instalar Python desde la página oficial [Python Downloads](https://www.python.org/downloads/) la versión 3.10, asegurándose de marcar la opción de agregar `python.exe` al `PATH` durante la instalación.

#### _MAC_

- No se requiere un paso adicional aquí, simplemente asegúrate de tener Python 3 instalado. Puedes verificar la instalación ejecutando `python3 --version` en la terminal.

- Opcionalmente, se puede utilizar pyenv para instalar la versión específica requerida de python: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation. Tras esto, simplemente hay que correr `pyenv install 3.10` para obtener la versión especificada.

### 1.2 Crear el entorno virtual (virtual environment)

Es importante crear un entorno virtual para aislar las dependencias del proyecto:

```
python -m venv .venv
```

### 1.3 Activar el entorno virtual

Una vez creado, debes activar el entorno virtual para que puedas instalar y ejecutar paquetes dentro de él:

- **Windows**:

  ```
  .venv\Scripts\activate
  ```

- **Mac/Linux**:
  ```
  source .venv/bin/activate
  ```

### 1.4 Instalar el kernel de IPython

Esto te permitirá usar Jupyter Notebooks en el entorno virtual:

```
pip install ipykernel ipython
```

### 1.5 Registrar el kernel dentro del entorno virtual

Este paso añade el kernel del entorno virtual a Jupyter Notebook para que puedas utilizarlo dentro del entorno:

```
ipython kernel install --user --name=venv
```

### 1.6 Instalar los paquetes de pip

Asegúrate de que todas las dependencias del proyecto estén instaladas correctamente utilizando el archivo `requirements.txt`:

```
pip install -r requirements.txt
```

---

## 2. Pruebas Unitarias

### 2.1 Cómo ejecutar pruebas

Para ejecutar las pruebas unitarias definidas en el proyecto, puedes utilizar `pytest`. Esto te permitirá verificar que el código funcione como se espera:

```
pytest -v .\tests\test_modelclassifier.py
```

---

## 3. Data Versioning Control (DVC)

### 3.1 ¿Qué es DVC?

[DVC (Data Version Control)](https://dvc.org) es una herramienta diseñada para versionar los datos de tu proyecto, similar a cómo Git versiona el código. DVC te permite realizar un seguimiento de los cambios en los datos, asegurarte de que todos los miembros del equipo estén utilizando los mismos datos, y gestionar experimentos reproducibles con pipelines automatizados.

### 3.2 ¿Cómo actualizar la base de datos?

Si la base de datos está almacenada en un remoto (por ejemplo, un almacenamiento en la nube o un servidor compartido), puedes actualizar los datos locales utilizando `dvc pull`. Este comando descarga los archivos de datos necesarios para ejecutar el pipeline desde el almacenamiento remoto configurado.

```
dvc pull
```

### 3.3 Generar el grafo del pipeline

DVC puede representar visualmente el flujo de tu pipeline de datos utilizando grafos acíclicos dirigidos (DAGs). Esto te da una vista general de las dependencias entre las diferentes etapas de tu pipeline de procesamiento de datos:

```
dvc dag
```

### 3.4 ¿Cómo ejecutar los pipelines?

Una vez que tienes un pipeline configurado, puedes ejecutar todas las etapas con el siguiente comando. `dvc repro` automáticamente ejecuta las etapas que no estén actualizadas o que dependan de cambios recientes en los datos o código:

```
dvc repro
```

### 3.5 Documentación adicional:

Para más información sobre cómo crear y gestionar pipelines en DVC, consulta la [documentación oficial](https://dvc.org/doc/start/data-pipelines/data-pipelines).

---

## 4. Estándares de Código

Para garantizar la calidad del código, se ha definido el uso de las siguientes herramientas:

- [black](https://black.readthedocs.io/en/stable/). Para dar formato al código de manera automática.
- [isort](https://pycqa.github.io/isort/). Para ordenar las librerías que se importan en cada archivo de manera automática.
- [flake8](https://flake8.pycqa.org/en/latest/index.html#). Para revisar que el código cumple las guías de estilo definidas en [pep8](https://peps.python.org/pep-0008/).

Para facilitar el manejo de estas herramientas, su configuración se ha hecho utilizando [pre-commit](https://pre-commit.com/). Es necesario correr el siguiente comando para instalar los hooks definidos para las herramientas anteriormente mencionadas:

```
pre-commit install
```

Tras esto, pre-commit correrá de manera automática al realizar cualquier `git commit`, asegurando que todos los archivos que se añadan al repositorio cumplan con los estándares de código definidos.

También se pueden correr manualmente los hooks en todo el repositorio, con el siguiente comando:

```
pre-commit run --all-files
```
## 5. Model Governance
### Documentación y Reproducibilidad

- **MLflow para Gestión de Experimentos**: Cada experimento fue registrado en MLflow, incluyendo sus hiperparámetros y métricas de clasificación.
- **DAG en DVC**: Se creó un DAG en DVC para definir y ejecutar las etapas del proyecto de manera ordenada, asegurando la reproducibilidad del flujo de trabajo.

### Control de Versiones

- **Versionamiento de Datos con DVC y Google Drive**: Los datos fueron versionados con DVC y almacenados en Google Drive para garantizar seguridad y accesibilidad.
- **Registro y Versionado de Modelos en MLflow**: La característica de Model Registry de MLflow permite registrar y versionar modelos, facilitando la restauración de versiones anteriores para revisiones o auditorías.
- **Control de Versiones de Código en GitHub**: El uso de GitHub para el control de versiones del código refuerza la trazabilidad de los cambios en el desarrollo.
### Validación del Modelo

- **Evaluación de Modelos**: Cada modelo fue entrenado y evaluado con métricas clave para el problema de clasificación de riesgos de embarazo.
- **Métricas en MLflow**: Las métricas registradas en MLflow incluyen accuracy, precision, recall y f1 score, lo que permite validar y comparar los modelos.

### Seguridad y Control de Acceso

- **Permisos en DVC y Google Drive**: Al integrar DVC con Google Drive, es posible configurar permisos de edición y visualización para controlar el acceso a los datos.
