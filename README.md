# Análisis de Retrasos en Vuelos Comerciales (EE.UU. 2022)

**Proyecto Final — Introducción a la Ciencia de Datos**  
Especialización en Ciencia de Datos e Inteligencia Artificial  
Universidad de Medellín — 2026-I

**Autores:** Leydi Tatiana Arboleda Vélez — CC 1128272760
             Cindy Velasquez Castaño - CC 1020452905 
             Jhonatan Restrepo Jaramillo - CC 1020454173
             Alejandro Ramirez - CC  
---

## Descripción del Proyecto

Este proyecto aplica técnicas de ciencia de datos para analizar y predecir retrasos en vuelos comerciales de EE.UU. utilizando datos oficiales del gobierno estadounidense. Se abordan dos enfoques complementarios: análisis exploratorio visual, clasificación binaria con modelos de Machine Learning .

**Pregunta central:** ¿Es posible predecir si un vuelo llegará con más de 15 minutos de retraso, y cómo evolucionan los retrasos a lo largo del tiempo?

---

##  Estructura del Proyecto 

```
proyecto_vuelos/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Pipeline de limpieza y transformación
│   ├── model.py               # Entrenamiento y evaluación de modelos
│   └── utils.py               # Funciones auxiliares
│
├── data/
│   ├── raw/
│   │   └── datos.csv          # Datos originales descargados de BTS
│   └── processed/
│       └── datos_limpios.csv  # Datos preprocesados listos para modelar
│
├── models/
│   └── modelo_entrenado.pkl   # Modelo serializado (Random Forest / Logistic)
│
├── tests/
│   ├── test_model.py          # Pruebas unitarias del modelo
│   └── test_api.py            # Pruebas de la API/app de Streamlit
│
├── app.py                     # Aplicación Streamlit (interfaz interactiva)
├── requirements.txt           # Dependencias del proyecto
├── .env                       # Variables de entorno (¡NO subir a Git!)
├── .env.example               # Plantilla de variables de entorno
├── README.md                  # Este archivo
└── .gitignore                 # Archivos excluidos del repositorio
```

---

##  Dataset

| Atributo | Detalle |
|---|---|
| **Fuente** | [Bureau of Transportation Statistics (BTS)](https://transtats.bts.gov) — Gobierno de EE.UU. |
| **Período** | Junio – Diciembre 2022 |
| **Registros** | 3,995,336 vuelos |
| **Variables originales** | 110 columnas |
| **Variables seleccionadas** | 19 columnas relevantes |
| **Variable objetivo** | `Delayed` — 1 si `ArrDelay` > 15 min (estándar FAA) |

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `FlightDate` | Fecha | Fecha del vuelo |
| `Month` / `DayOfWeek` | Ordinal | Mes y día de la semana |
| `Reporting_Airline` | Nominal | Código IATA de la aerolínea |
| `Origin` / `Dest` | Nominal | Aeropuerto de origen y destino |
| `DepDelay` / `ArrDelay` | Continua | Minutos de retraso en salida/llegada |
| `Distance` / `AirTime` | Continua | Distancia (millas) y tiempo en aire (min) |
| `CarrierDelay` / `WeatherDelay` / `NASDelay` | Continua | Minutos de retraso por causa |
| `Cancelled` / `Diverted` | Binaria | Cancelación o desvío del vuelo |
| `Delayed` | Binaria | **Variable objetivo** |

---

## Análisis Exploratorio de Datos (EDA)

El EDA se realizó con **Plotly** y **Seaborn**, respondiendo cuatro preguntas clave:

1. ¿Qué aerolíneas tienen mayor tasa de retrasos?
2. ¿En qué meses y días de la semana se concentran más retrasos?
3. ¿Cuál es la causa principal de los retrasos?
4. ¿Existe relación entre la distancia del vuelo y el retraso?

### Principales hallazgos

- **Distribución de la variable objetivo:** aproximadamente 21% de los vuelos presentaron retraso significativo (>15 min).
- **Estacionalidad semanal:** los ACF/PACF confirmaron picos significativos en lags 7, 14 y 21, evidenciando un ciclo semanal en el número de vuelos retrasados.
- **Causa dominante:** `CarrierDelay` (retrasos atribuidos a la aerolínea) y `DepDelay` (retraso en salida) son los predictores más correlacionados con la variable objetivo.
- **Variación por aerolínea:** existe dispersión notable en la mediana de `ArrDelay` entre aerolíneas; algunas superan consistentemente el umbral de 15 min.
- **Distancia vs. retraso:** no se observa una correlación lineal fuerte entre distancia y retraso de llegada, lo que indica que los retrasos operativos dominan sobre la distancia.

---

## Preprocesamiento

El preprocesamiento utiliza **Pipelines** y **ColumnTransformer** de scikit-learn para garantizar reproducibilidad:

- **Codificación:** `LabelEncoder` para variables ordinales (`DayOfWeek`, `Month`) y `OneHotEncoding` para variables nominales (`Reporting_Airline`, `OriginStateName`).
- **Escalado:** `StandardScaler` sobre predictores continuos.
- **Reducción de dimensionalidad:** selección de las 7 variables con mayor poder predictivo tras análisis de correlación con `Delayed`.
- **Dataset final para modelado:** 2,719,802 registros (después de eliminar vuelos cancelados y con nulos en `ArrDelay`).

---

## Modelos de Machine Learning

### Modelo 1 — Regresión Logística

**Justificación:** variable objetivo binaria (`Delayed`), interpretabilidad de coeficientes.

**Predictores:** `DepDelay`, `Distance`, `AirTime`, `Month`, `DayOfWeek`, `CarrierDelay`, `WeatherDelay`

| Métrica | Valor |
|---|---|
| Pseudo R² (McFadden) | 0.709 |
| Accuracy | 96% |
| AUC-ROC | **0.9597** |
| Precision (retrasado) | 96% |
| Recall (retrasado) | 82% |
| F1-score (retrasado) | 88% |

> El modelo logístico muestra excelente discriminación (AUC ≈ 0.96), impulsado principalmente por `DepDelay` y `CarrierDelay`.

---

### Modelo 2 — Random Forest (Clasificador)

**Justificación:** captura relaciones no lineales entre variables, robustez ante outliers, importancia de variables interpretable.

**Configuración sugerida:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),          # ColumnTransformer definido en preprocessing.py
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',             # Manejo del desbalance de clases
        random_state=42,
        n_jobs=-1
    ))
])
```

**Variables de mayor importancia esperada:** `DepDelay`, `CarrierDelay`, `WeatherDelay`, `DayOfWeek`, `Month`.


---
## Dependencias principales

```
pandas
numpy
scikit-learn
plotly
seaborn
matplotlib
statsmodels
pmdarima
streamlit
joblib
scipy
```

Ver `requirements.txt` para versiones exactas.

---

## GitHub

Este proyecto utiliza GitHub para control de versiones. El archivo `.gitignore` excluye:
- `.env` (credenciales y variables de entorno)
- Archivos de datos crudos pesados (`data/raw/`)
- Entornos virtuales (`venv/`, `__pycache__/`)

---

## Conclusiones

1. **`DepDelay` es el predictor más poderoso:** un retraso en la salida casi siempre se traduce en retraso en la llegada, con una correlación casi perfecta con `Delayed`.
2. **Los retrasos tienen marcada estacionalidad semanal:** los viernes y domingos concentran mayor número de retrasos, mientras que los martes y miércoles son los días con mejor desempeño.
3. **Junio y diciembre son los meses críticos:** junio por alta demanda de verano; diciembre por condiciones climáticas y mayor volumen navideño.
4. **La causa operativa de la aerolínea (`CarrierDelay`) supera al clima:** la mayoría de los retrasos son atribuibles a la operación interna de las aerolíneas, no a factores externos.
5. **La distancia no determina el retraso:** vuelos cortos y largos presentan distribuciones similares de retraso, lo que confirma que el origen del problema es operativo.
6. **Los retrasos de llegada siguen una distribución asimétrica con cola pesada a la derecha:** la mayoría de vuelos retrasados acumula entre 15 y 60 minutos, pero existen casos extremos de varios cientos de minutos.
7. **El modelo SARIMA confirma la previsibilidad de los patrones temporales:** la estacionalidad semanal es estadísticamente significativa (lags 7, 14, 21 en ACF), lo que permite proyectar el volumen de retrasos con antelación.


---

## Licencia

Uso académico — Universidad de Medellín 2026-I.  
Datos: dominio público — [Bureau of Transportation Statistics](https://transtats.bts.gov).
