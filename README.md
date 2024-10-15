# Proyecto MLOps

## Setup del proyecto *WINDOWS*
Instalar python como lo recomendaron y agregar el python.exe al PATH

### Correr el virtual environment

```
.venv\Scripts\activate
```

### Instalar el kernel dentro del venv
Source: [Using Jupyter Notebook in Virtual Environment](https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/)
```
ipython kernel install --user --name=venv
```

### Instalar paquetes de pip
```
pip install -r requirements.txt
```

## Como ejecutar pruebas?
Dentro de la raiz del proyecto puedes ejecutar el siguiente comando:
```
pytest -v .\tests\test_modelclassifier.py
```
##Change to notify 


# DVC
Para ejecutar los pipelines:
```
dvc repro
```

## Documentacion:
https://dvc.org/doc/start/data-pipelines/data-pipelines
