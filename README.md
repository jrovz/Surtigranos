# Dashboard de Ventas y Análisis RFM - Surtigranos Rovira

Este repositorio contiene el código y los dashboards generados para el análisis de ventas y segmentación de clientes de **Surtigranos Rovira**. Los datos han sido extraídos de **Siigo Nube** y procesados para obtener insights sobre el comportamiento de las ventas y la clasificación de clientes utilizando el análisis **RFM (Recencia, Frecuencia y Valor Monetario)**.

---

## 📌 Contenido del Proyecto

### 1️⃣ Dashboards Generados

#### **📊 Dashboard de Ventas**
- **💰 Total Ventas:** Muestra el total acumulado de ventas.
- **📦 Productos Vendidos:** Cantidad total de productos vendidos.
- **👥 Clientes Activos:** Número de clientes que han realizado compras.
- **🏆 Top 10 Productos por Cantidad Vendida:** Lista de los productos más vendidos.
- **📈 Distribución de Ventas por Producto:** Representación gráfica de la participación de cada producto en las ventas.
- **📉 Relación Precio-Cantidad por Producto:** Comparación entre el precio unitario y la cantidad vendida.
- **💳 Top 10 Clientes por Valor Total de Compras:** Ranking de los clientes con mayor valor de compra.
- **🔄 Frecuencia de Compra por Cliente:** Número de transacciones por cliente.

#### **📊 Análisis RFM de Clientes**
- **🕒 Recencia vs Frecuencia:** Relación entre los días desde la última compra y la cantidad de compras realizadas.
- **🔍 Perfiles RFM de Top 10 Clientes:** Comparación de los clientes más valiosos según Recencia, Frecuencia y Valor Monetario.
- **📊 Correlación entre Métricas RFM:** Matriz de correlación entre las variables RFM.
- **📉 Distribución de Clientes por Valor Monetario:** Representación de la cantidad de clientes según su nivel de gasto.

📂 **Archivos de los dashboards:**
- `surtigranos_dashboard_principal_mejorado.png`
- `surtigranos_dashboard_segmentacion_mejorado.png`

---

## 🏗️ 2️⃣ Estructura del Código

El archivo principal del código es:
- `surtiBI_1.py`: Contiene el procesamiento de datos y la generación de gráficos.

📌 **Librerías utilizadas:**
- `pandas` (Manejo de datos)
- `matplotlib` y `seaborn` (Visualización de datos)
- `numpy` (Operaciones numéricas)

---

## 🚀 3️⃣ Cómo Ejecutar el Proyecto

### 🔧 **Requisitos Previos**

Antes de ejecutar el código, asegúrate de tener instaladas las dependencias. Puedes instalarlas con el siguiente comando:
```bash
pip install pandas matplotlib seaborn numpy
```

### ▶️ **Ejecución**
Ejecuta el script de Python con el siguiente comando:
```bash
python surtiBI_1.py
```
Esto generará los dashboards en formato de imagen y los guardará en el directorio del proyecto.

---

## 🤝 4️⃣ Contacto y Contribución
Si deseas contribuir a este proyecto o tienes dudas, puedes contactarme a través de **GitHub** o mis redes sociales.

---

📅 **Generado el 18/03/2025 - Datos extraídos de Siigo Nube**
