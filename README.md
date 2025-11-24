# data_analyst_bot
Este proyecto implementa un Asistente de Análisis de Datos Inteligente (Data Analyst Bot) en Python. Su objetivo es automatizar la exploración de datos (EDA) permitiendo al usuario hacer preguntas en lenguaje natural sobre un DataFrame.

# Asistente de Análisis de Datos Inteligente (Data Analyst Bot)

Un motor de análisis exploratorio de datos (EDA) automatizado en Python. Este proyecto permite a los usuarios interactuar con un `pandas.DataFrame` utilizando lenguaje natural para generar visualizaciones dinámicas y modelos de Machine Learning explicativos al instante.

## Características Principales

1.  **Búsqueda Semántica (NLP):** Utiliza **TF-IDF** para interpretar la intención del usuario y encontrar las columnas más relevantes, incluso si los nombres no coinciden exactamente.
2.  **Inferencia de Tipos Inteligente:** Detecta tipos de datos reales (Fechas, IDs, Categorías, Texto, Numéricos) analizando la topología de los datos, superando el `dtype` nativo de Pandas.
3.  **AutoML & Feature Importance:** Si la pregunta implica causalidad ("por qué", "impacto"), entrena automáticamente modelos (XGBoost/Regresión) para identificar los drivers principales de una variable.
4.  **Motor de Visualización Dinámico:** Un sistema experto decide qué gráfico es el matemáticamente correcto (Scatter, Series de Tiempo, Violin, Heatmap) basándose en los tipos de datos cruzados.
5.  **Visualización Interactiva:** Genera gráficos ricos e interactivos utilizando **Plotly**.

---

## Instalación y Requisitos

El sistema se adapta al entorno (Jupyter Notebook o Google Colab).

**Dependencias Principales:**
```bash
pip install pandas numpy scikit-learn plotly seaborn
```

Dependencias Opcionales (Recomendadas para AutoML):
El sistema detecta automáticamente si estas librerías están instaladas para mejorar la predicción.
```bash
import pandas as pd
from data_analyst import DataAnalystBot  # Asumiendo que el script se llama data_analyst.py

# 1. Cargar datos
df = pd.read_csv("ventas_empresa.csv")

# 2. Inicializar el Bot
bot = DataAnalystBot(df)

# 3. Análisis Exploratorio (Visualización automática)
# El bot detectará 'fecha' y 'ingresos' y graficará una línea de tiempo.
bot.analizar("evolución de los ingresos en el último año")

# 4. Análisis Causal (AutoML)
# El bot detectará que preguntas 'por qué' y buscará correlaciones fuertes usando ML.
bot.analizar("¿Por qué está variando el margen de ganancia?")
```

## Documentación Técnica
1. Módulo de Preprocesamiento (Funciones Auxiliares)
Conjunto de herramientas para la limpieza e inferencia de metadatos antes del análisis.
clasificar_columna_inteligente(s: pd.Series):
Algoritmo heurístico que determina el uso real de una columna.
Si parece numérico pero tiene baja cardinalidad (<5%) numerico_ordinal.
Si es string pero cumple patrones Regex de fecha = fecha.
Si la cardinalidad es casi total (>95%) = id.
Longitud promedio de texto alta = texto.
perfil_columna(df, col): Genera un "pasaporte" de metadatos para cada variable (estadísticas, nulos, top valores).

2. Clase AutoPredictor (Motor ML)
Encargada de responder preguntas de tipo "¿Qué influye en X?".
- Selección de Modelo:
Detecta si el problema es Regresión (target continuo) o Clasificación (target categórico).
Prioridad de librerías: XGBoost, LightGBM, CatBoost, Sklearn (Linear/Logistic).

Feature Engineering Express:

Imputación de nulos (mediana).
One-Hot Encoding para categorías de baja cardinalidad.
Salida: Retorna el score del modelo y las 10 variables más importantes (Feature Importance).

3. Clase VisualizerEngine (Motor Gráfico)
Cerebro de decisión que asigna el gráfico óptimo según la combinación de variables seleccionadas.
Variable A	Variable B	Gráfico Resultante (Plotly)	Contexto
Numérico	Numérico	Scatter Plot (con línea de tendencia)	Correlación
Fecha	Numérico	Line Plot	Serie de Tiempo
Categoría	Numérico	Violin Plot	Distribución y Outliers
Categoría	Categoría	Heatmap (Crosstab)	Densidad de cruces
Numérico	None	Histogram + Box	Distribución Univariada
Categoría	None	Bar Chart	Frecuencia / Conteo
3+ Numéricas	N/A	Correlation Matrix	Mapa de calor de correlación


4. Clase DataAnalystBot (Orquestador)
Controlador principal que une NLP, ML y Visualización.
Inicialización (__init__):
Indexa todas las columnas creando documentos de texto que incluyen: Nombre de columna + Tipo de dato + Valores frecuentes.
Crea una matriz TF-IDF para búsquedas rápidas.
Búsqueda (_seleccionar_columnas_relevantes):
Calcula la Similitud del Coseno entre la query del usuario y la matriz TF-IDF de las columnas.
Flujo analizar(peticion):
Paso 1: Identifica columnas clave.
Paso 2: Busca palabras clave de causalidad (e.g., "impacta", "causa").
Si existen: Ejecuta AutoPredictor.
Paso 3: Ejecuta VisualizerEngine con las columnas identificadas.
Paso 4: Renderiza los resultados en el output del notebook.


## Notas de Implementación
Muestreo (sample_df): Para garantizar la fluidez de los gráficos interactivos, si el DataFrame excede las 5000 filas, el motor de visualización trabaja sobre una muestra aleatoria (aunque el ML entrena con más datos).
Validación: El motor rechaza automáticamente columnas vacías o con varianza cero (valores constantes) para evitar gráficos rotos.
## Licencia
Este proyecto es de código abierto y está disponible para uso educativo y profesional.
