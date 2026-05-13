# ✈️ Predicción de Retrasos en Vuelos Comerciales

## Descripción del Proyecto

Este proyecto forma parte del programa de **Especialización en Ciencia de Datos e Inteligencia Artificial** y tiene como objetivo analizar y predecir si un vuelo comercial experimentará un retraso significativo en su llegada, definido como un retraso superior a **15 minutos**, umbral estándar utilizado por la Administración Federal de Aviación de los Estados Unidos (FAA).

A través de técnicas de análisis exploratorio de datos (EDA) y modelos de clasificación supervisada, se busca identificar los factores más determinantes en la ocurrencia de retrasos y construir un modelo predictivo confiable que apoye la toma de decisiones en la industria aeronáutica.

**Autores:**
- Cindy Vanessa Velásquez Castaño
- Leidy Tatiana Arboleda Vélez
- Alejandro Ramírez Marín
- Jonathan Restrepo Jaramillo

---

## 🚨 Problema de Negocio

Los retrasos en vuelos comerciales generan pérdidas económicas de miles de millones de dólares anuales para aerolíneas, aeropuertos y pasajeros. Identificar de manera anticipada los vuelos con alta probabilidad de retraso permite a las aerolíneas y operadores aeroportuarios tomar decisiones preventivas: redistribuir tripulaciones, ajustar conexiones, comunicar alertas tempranas a pasajeros y optimizar recursos en tierra.

La incapacidad de predecir estos eventos genera ineficiencias operativas, insatisfacción del cliente y pérdidas económicas evitables.

---

## 🎯 Objetivos del Proyecto

- Realizar un análisis exploratorio exhaustivo de los datos de vuelos del segundo semestre de 2022.
- Identificar los patrones temporales, operativos y geográficos asociados a los retrasos.
- Construir y comparar dos modelos de clasificación binaria para predecir si un vuelo se retrasará más de 15 minutos.
- Seleccionar el modelo con mejor capacidad de detección de retrasos (recall), alineado con el objetivo de negocio.

---

## 📊 Dataset

**Fuente:** Bureau of Transportation Statistics (BTS) — agencia oficial del Departamento de Transporte de los Estados Unidos (DOT).

**Período:** Junio a Diciembre de 2022.

**Acceso:** Público y gratuito en [transtats.bts.gov](https://transtats.bts.gov).

El dataset contiene registros de rendimiento en tiempo real de aerolíneas certificadas que operan en territorio estadounidense, incluyendo información sobre vuelos, retrasos, cancelaciones y sus causas.

### Variables utilizadas

| Variable | Tipo | Descripción |
|---|---|---|
| `FlightDate` | Fecha | Fecha del vuelo (YYYY-MM-DD) |
| `Month` | Ordinal | Mes (6 = Junio … 12 = Diciembre) |
| `DayOfWeek` | Ordinal | Día de la semana (1 = Lunes … 7 = Domingo) |
| `Reporting_Airline` | Nominal | Código IATA de la aerolínea |
| `Origin` / `Dest` | Nominal | Código del aeropuerto de origen/destino |
| `OriginStateName` / `DestStateName` | Nominal | Estado de origen/destino |
| `DepDelay` | Continua | Minutos de retraso en la salida |
| `ArrDelay` | Continua | Minutos de retraso en la llegada |
| `Distance` | Continua | Distancia del vuelo en millas |
| `AirTime` | Continua | Tiempo en el aire en minutos |
| `CarrierDelay` | Continua | Retraso atribuido a la aerolínea (minutos) |
| `WeatherDelay` | Continua | Retraso por condiciones climáticas (minutos) |
| `NASDelay` | Continua | Retraso por el Sistema Aéreo Nacional (minutos) |
| `Flight_Number_Reporting_Airline` | Discreta | Número de vuelo |
| `Cancelled` | Binaria | 1 = Cancelado, 0 = No cancelado |
| `Diverted` | Binaria | 1 = Desviado, 0 = No desviado |
| `Delayed` | Binaria | **Variable objetivo**: 1 si ArrDelay > 15 min |

---

## 🔍 Análisis Exploratorio de Datos (EDA)

El EDA se estructuró en torno a cuatro preguntas de negocio clave:

**Distribución de la variable objetivo:** Aproximadamente el 21% de los vuelos presentaron retrasos superiores a 15 minutos, lo que genera un dataset moderadamente desbalanceado. La distribución de `ArrDelay` fue confirmada como no normal mediante la prueba estadística Kolmogórov-Smirnov.

**Patrones temporales:** El análisis por mes y día de la semana reveló que diciembre concentra la mayor tasa de retrasos del período estudiado, probablemente asociado al incremento de la demanda en temporada navideña. A nivel semanal, los viernes son el día con mayor incidencia de retrasos, mientras que los sábados presentan el mejor desempeño.

**Causas de retraso:** Entre los vuelos clasificados como retrasados, los factores operativos de la aerolínea (`CarrierDelay`, promedio de 22.73 minutos) superan a los retrasos por clima (`WeatherDelay`) y por el Sistema Aéreo Nacional (`NASDelay`), lo que indica que la gestión interna de las aerolíneas es el factor más crítico.

**Relaciones entre variables:** La matriz de correlación evidenció que el retraso en la salida (`DepDelay`) es el predictor más correlacionado con el retraso en la llegada. La distancia del vuelo no mostró correlación significativa con el retraso, lo que sugiere que los retrasos operativos dominan sobre las características geográficas del vuelo. Los boxplots comparativos confirmaron que `DepDelay` diferencia claramente los grupos de vuelos retrasados y no retrasados.

---

## 🤖 Modelos Utilizados

### Regresión Logística

Se entrenó con las variables predictoras numéricas: `DepDelay`, `Distance`, `AirTime`, `Month`, `DayOfWeek`, `CarrierDelay` y `WeatherDelay`. Los datos fueron estandarizados con `StandardScaler` y el modelo fue ajustado con `statsmodels` para acceder a estadísticos completos (coeficientes, p-valores, intervalos de confianza). Se utilizó una división 70/30 para entrenamiento y prueba, con estratificación sobre la variable objetivo.

### Random Forest

Se entrenó con todas las variables disponibles, incluyendo variables categóricas como aerolínea, aeropuerto de origen y destino. Las variables categóricas fueron codificadas mediante `OneHotEncoder`. El modelo fue construido con 100 árboles y profundidad máxima de 10 niveles, usando una división 80/20 para entrenamiento y prueba. Se empleó un `Pipeline` de scikit-learn para encapsular el preprocesamiento y el clasificador.

---

## 📈 Evaluación de Modelos

| Métrica | Regresión Logística | Random Forest |
|---|---|---|
| Accuracy | 0.96 | 0.85 |
| Precision | 0.96 | 0.97 |
| Recall | **0.82** | 0.26 |
| F1-Score | 0.88 | 0.42 |
| AUC-ROC | **0.96** | 0.94 |

**Interpretación:**

El objetivo central de este proyecto es detectar la mayor cantidad posible de vuelos retrasados, lo cual implica maximizar el **recall** (sensibilidad). Bajo esta premisa, los resultados deben leerse en función de cuántos vuelos retrasados logra identificar cada modelo.

La **Regresión Logística** alcanzó un recall de 0.82, lo que significa que identifica correctamente el 82% de los vuelos con retraso real. Su AUC de 0.96 confirma una excelente capacidad discriminativa entre clases. Desde la matriz de confusión, el modelo clasificó correctamente 919,092 vuelos no retrasados e identificó 194,919 casos de retraso, con 42,750 falsos negativos.

El **Random Forest**, aunque ligeramente superior en precisión (0.97 vs 0.96), presenta un recall de solo 0.26, dejando sin detectar el 74% de los vuelos que efectivamente se retrasaron. Este comportamiento lo hace inadecuado para el objetivo planteado, ya que en la práctica implicaría ignorar la mayoría de los retrasos reales.

El análisis de Odds Ratios en la regresión logística confirmó que `CarrierDelay` es el predictor más influyente (OR ≈ 1,640), seguido por `AirTime` y `DepDelay`, mientras que la distancia actúa como factor protector (OR < 1), lo que sugiere que los vuelos largos tienen más capacidad para recuperar retrasos de salida durante el trayecto.

---

## ✅ Conclusión Final

La **Regresión Logística** es el modelo más adecuado para este problema. Su superioridad no radica únicamente en las métricas globales, sino en su alineación con el objetivo de negocio: detectar vuelos retrasados con la mayor precisión posible.

Con un recall de 0.82 frente al 0.26 del Random Forest, y un F1-Score de 0.88 frente a 0.42, la regresión logística demuestra un equilibrio superior entre sensibilidad y precisión. Adicionalmente, su AUC de 0.96 garantiza una excelente capacidad de discriminación en todos los umbrales de clasificación.

Otro valor diferencial de la Regresión Logística es su interpretabilidad: los coeficientes permiten cuantificar el impacto de cada variable en la probabilidad de retraso, lo cual es fundamental para comunicar resultados a equipos operativos no especializados en machine learning. El Random Forest, aunque más preciso en la clase mayoritaria, sacrifica la detección de retrasos reales de manera inaceptable para el contexto del problema.

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- `pandas` — manipulación y análisis de datos
- `numpy` — operaciones numéricas
- `scikit-learn` — modelos de ML, preprocesamiento y métricas
- `statsmodels` — regresión logística con estadísticos detallados
- `matplotlib` y `seaborn` — visualizaciones estáticas
- `plotly` — visualizaciones interactivas
- `scipy` — pruebas estadísticas
- `requests` y `zipfile` — descarga y descompresión de datos desde la BTS

---

## 📁 Estructura del Proyecto

```
proyecto-retrasos-vuelos/
│
├── Proyecto_Final.ipynb        # Notebook principal con EDA y modelos
├── README.md                   # Documentación del proyecto
└── data/                       # (Opcional) Carpeta para datos locales
    └── vuelos_2022_H2.csv      # Dataset descargado de la BTS
```

> **Nota:** Los datos se descargan directamente desde la fuente oficial de la BTS al ejecutar el notebook, por lo que no es necesario disponer de los archivos de forma local.

---

## ▶️ Cómo Ejecutar el Proyecto

**Requisitos previos:**
- Python 3.8 o superior
- Jupyter Notebook o Google Colab
- Conexión a internet (para la descarga automática del dataset)

**Pasos:**

1. Clonar o descargar el repositorio en tu entorno local o abrirlo directamente en Google Colab.

2. Instalar las dependencias necesarias ejecutando en una celda:
   ```bash
   pip install pandas numpy scikit-learn statsmodels matplotlib seaborn plotly scipy requests
   ```

3. Abrir el archivo `Proyecto_Final.ipynb` en Jupyter Notebook o Google Colab.

4. Ejecutar las celdas en orden secuencial. La primera sección descargará automáticamente los datos mensuales de la BTS (junio a diciembre de 2022) y los consolidará en un único DataFrame.

5. Continuar con las secciones de limpieza de datos, EDA, modelado y evaluación.

> **Tiempo estimado de ejecución:** La descarga y procesamiento del dataset puede tomar varios minutos dependiendo de la velocidad de la conexión y los recursos del entorno.

---

## 👤 Autor

Proyecto desarrollado como entrega académica para la **Especialización en Ciencia de Datos e Inteligencia Artificial**.

| Integrante | 
|---|
| Cindy Vanessa Velásquez Castaño |
| Leidy Tatiana Arboleda Vélez |
| Alejandro Ramírez Marín |
| Jonathan Restrepo Jaramillo |

*Fecha de entrega: 12 de mayo de 2026*
