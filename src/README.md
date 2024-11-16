# Proyecto MLOps

## 1. Correr el proyecto

### 1.1 Pasos Previos

- Seguir las instrucciones para setup del proyecto en [maternalhealth/README.md](./maternalhealth/README.md)
- Haber instalado [docker](https://www.docker.com/).

### 1.2 Recrear los modelos localmente

- Desde `/maternalhealth`, correr el comando `mlflow ui`.
- Abrir otra ventana en la misma ruta, y correr `dvc pull` y `dvc repro` para recrear los modelos localmente.

### 1.3 Desplegar el modelo

Tras analizar las métricas pertinentes, volver al directorio `src` del proyecto, y desplegar el modelo elegido con los siguientes comandos:

```
docker build -t maternal-health-api --build-arg MODEL_NAME={MODEL_NAME} .
```

Nota: Reemplazar `{MODEL_NAME}` con el nombre real del modelo que se quiere desplegar, a elegir entre:

- `decision_tree`
- `knn`
- `logistic`
- `random_forest`
- `svm`
- `xgboost`

```
docker run -p 8000:8000 maternal-health-api:latest
```

### 1.4 Probar la API

Endpoint raíz:

```
curl -X GET http://localhost:8000/ -H 'Content-Type: application/json'
```

Endpoint de predicciones:

```
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"features": [0.3717036160545446, 0.6572080658786049, 0.6096765440652615, 0.34469536238192333, 0.0, 0.9333333333333331]}'
```

Nota: Enviar los features al modelo como una lista ordenada con los parámetros ya preprocesados, en el orden `Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate`.
