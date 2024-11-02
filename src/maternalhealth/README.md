# Proyecto MLOps

## 1. Setup del proyecto
### 1.1 Pasos Previos 
#### *WINDOWS*
- Instalar Python desde la página oficial [Python Downloads](https://www.python.org/downloads/), asegurándose de marcar la opción de agregar `python.exe` al `PATH` durante la instalación.

#### *MAC* 
- No se requiere un paso adicional aquí, simplemente asegúrate de tener Python 3 instalado. Puedes verificar la instalación ejecutando `python3 --version` en la terminal.

### 1.2 Crear el entorno virtual (virtual environment)
Es importante crear un entorno virtual para aislar las dependencias del proyecto:

```
python3 -m venv .venv
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
