import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import os
from datetime import datetime

def cargar_datos(file_path):
    """
    Carga y limpia los datos de ventas desde un archivo Excel.
    
    Args:
        file_path: Ruta al archivo Excel
    
    Returns:
        DataFrame con los datos de ventas limpios
    """
    try:
        df_ventas = pd.read_excel(file_path, skiprows=7)
        df_ventas = df_ventas.dropna(axis=1, how='all')
        
        # Validación de datos
        columnas_requeridas = ['Nombre cliente', 'Nombre producto', 'Cantidad vendida', 'Total']
        for col in columnas_requeridas:
            if col not in df_ventas.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en el archivo")
        
        # Limpieza adicional
        df_ventas['Total'] = pd.to_numeric(df_ventas['Total'], errors='coerce')
        df_ventas['Cantidad vendida'] = pd.to_numeric(df_ventas['Cantidad vendida'], errors='coerce')
        df_ventas = df_ventas.dropna(subset=['Total', 'Cantidad vendida'])
        
        return df_ventas
    except Exception as e:
        raise Exception(f"Error al cargar datos: {str(e)}")

def analizar_datos(df):
    """
    Realiza análisis estadístico básico de los datos
    
    Args:
        df: DataFrame con datos de ventas
    
    Returns:
        Dict con métricas clave
    """
    df_filtrado = df[df['Nombre cliente'] != 'Ventas Mostrador']
    
    # Cálculo de métricas
    metricas = {
        'total_ventas': df_filtrado['Total'].sum(),
        'promedio_venta': df_filtrado['Total'].mean(),
        'total_productos_vendidos': df_filtrado['Cantidad vendida'].sum(),
        'num_clientes': df_filtrado['Nombre cliente'].nunique(),
        'num_productos': df_filtrado['Nombre producto'].nunique(),
        'producto_mas_vendido': df_filtrado.groupby('Nombre producto')['Cantidad vendida'].sum().idxmax(),
        'cliente_principal': df_filtrado.groupby('Nombre cliente')['Total'].sum().idxmax()
    }
    
    return metricas

def crear_dashboard_mejorado(df, output_dir='output'):
    """
    Crea un dashboard completo con visualizaciones mejoradas
    
    Args:
        df: DataFrame con datos de ventas
        output_dir: Directorio para guardar las imágenes
    
    Returns:
        fig: Figura de matplotlib con el dashboard
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    df_filtrado = df[df['Nombre cliente'] != 'Ventas Mostrador']
    
    # Configuración de estilo
    plt.style.use('ggplot')
    colores_productos = sns.color_palette("viridis", 10)  # Paleta más profesional
    colores_clientes = sns.color_palette("muted", 10)  # Paleta más profesional
    
    # Crear figura con 2x3 subplots para más análisis
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    fig.suptitle('Dashboard de Ventas - SURTIGRANOS-ROVIRA (Clientes Registrados)', 
                 fontsize=28, y=0.98, fontweight='bold')
    
    # Añadir timestamp
    fig.text(0.01, 0.01, f'Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
             fontsize=8, color='gray')
    
    # 1. Top 10 Productos por Cantidad
    top_cantidad = df_filtrado.groupby('Nombre producto')['Cantidad vendida'].sum().sort_values(ascending=False).head(10)
    ax1 = axes[0, 0]
    bars = ax1.barh(top_cantidad.index, top_cantidad.values, color=colores_productos)
    ax1.set_title('Top 10 Productos por Cantidad Vendida', fontsize=18, pad=20, fontweight='bold')
    ax1.set_xlabel('Cantidad vendida', fontsize=14)
    ax1.set_xlim(0, top_cantidad.values.max() * 1.1)  # Mejor escalado
    for i, v in enumerate(top_cantidad.values):
        ax1.text(v + (top_cantidad.values.max() * 0.01), i, f"{v:,.0f}", 
                 va='center', fontsize=10, color='darkslategray')
    
    # 2. Top 10 Productos por Valor
    top_ventas = df_filtrado.groupby('Nombre producto')['Total'].sum().sort_values(ascending=False).head(10)
    ax2 = axes[0, 1]
    wedges, _, _ = ax2.pie(top_ventas.values, labels=None, autopct='%1.1f%%', 
                           explode=[0.05] * len(top_ventas), colors=colores_productos, 
                           shadow=False, startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Distribución de Ventas por Producto (Top 10)', fontsize=18, pad=20, fontweight='bold')
    ax2.legend(wedges, top_ventas.index, title="Productos", 
               loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    # 3. Análisis de relación precio-cantidad
    ax3 = axes[0, 2]
    # Calculamos el precio unitario promedio para cada producto
    df_filtrado['Precio_unitario'] = df_filtrado['Total'] / df_filtrado['Cantidad vendida']
    top_productos = df_filtrado.groupby('Nombre producto')['Cantidad vendida'].sum().sort_values(ascending=False).head(10).index
    df_top = df_filtrado[df_filtrado['Nombre producto'].isin(top_productos)]
    
    sns.scatterplot(data=df_top, x='Precio_unitario', y='Cantidad vendida', 
                    hue='Nombre producto', palette=colores_productos, s=100, alpha=0.7, ax=ax3)
    ax3.set_title('Relación Precio-Cantidad por Producto', fontsize=18, pad=20, fontweight='bold')
    ax3.set_xlabel('Precio Unitario ($)', fontsize=14)
    ax3.set_ylabel('Cantidad Vendida', fontsize=14)
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax3.legend(title="Productos", loc="upper right", fontsize=8)
    
    # 4. Top 10 Clientes por Valor
    top_clientes_valor = df_filtrado.groupby('Nombre cliente')['Total'].sum().sort_values(ascending=False).head(10)
    clientes_primernombre = [nombre.split()[0] for nombre in top_clientes_valor.index]
    
    ax4 = axes[1, 0]
    bars = ax4.bar(clientes_primernombre, top_clientes_valor.values, color=colores_clientes)
    ax4.set_title('Top 10 Clientes por Valor Total de Compras', fontsize=18, pad=20, fontweight='bold')
    ax4.set_ylabel('Total Compras ($)', fontsize=14)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax4.set_xticklabels(clientes_primernombre, rotation=30, ha="right", fontsize=12)
    
    # Añadir etiquetas de valores
    for i, v in enumerate(top_clientes_valor.values):
        ax4.text(i, v + (top_clientes_valor.values.max() * 0.01), f"${v:,.0f}", 
                 ha='center', fontsize=10, color='darkslategray')
    
    # 5. Distribución de ventas por cliente
    ax5 = axes[1, 1]
    ventas_cliente = df_filtrado.groupby('Nombre cliente')['Total'].sum().sort_values(ascending=False).head(10)
    wedges, _, _ = ax5.pie(ventas_cliente.values, labels=None, autopct='%1.1f%%', 
                           explode=[0.05] * len(ventas_cliente), colors=colores_clientes, 
                           shadow=False, startangle=90, textprops={'fontsize': 10})
    ax5.set_title('Distribución de Ventas por Cliente (Top 10)', fontsize=18, pad=20, fontweight='bold')
    ax5.legend(wedges, ventas_cliente.index, title="Clientes", 
               loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    # 6. Frecuencia de compra por cliente
    ax6 = axes[1, 2]
    frecuencia_compra = df_filtrado.groupby('Nombre cliente').size().sort_values(ascending=False).head(10)
    clientes_primernombre_freq = [nombre.split()[0] for nombre in frecuencia_compra.index]
    
    bars = ax6.barh(clientes_primernombre_freq, frecuencia_compra.values, color=colores_clientes)
    ax6.set_title('Frecuencia de Compra por Cliente (Top 10)', fontsize=18, pad=20, fontweight='bold')
    ax6.set_xlabel('Número de Transacciones', fontsize=14)
    
    # Añadir etiquetas de valores
    for i, v in enumerate(frecuencia_compra.values):
        ax6.text(v + (frecuencia_compra.values.max() * 0.01), i, f"{v}", 
                 va='center', fontsize=10, color='darkslategray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3, w_pad=3)
    
    # Guardar métricas en el gráfico
    metricas = analizar_datos(df)
    fig.text(0.01, 0.95, f"Total Ventas: ${metricas['total_ventas']:,.0f}", fontsize=12, color='darkslategray')
    fig.text(0.01, 0.93, f"Productos Vendidos: {metricas['total_productos_vendidos']:,.0f}", fontsize=12, color='darkslategray')
    fig.text(0.01, 0.91, f"Clientes Activos: {metricas['num_clientes']}", fontsize=12, color='darkslategray')
    
    return fig

def generar_dashboard_segmentado(df, output_dir='output'):
    """
    Genera un dashboard adicional con análisis de segmentación RFM
    
    Args:
        df: DataFrame con datos de ventas
        output_dir: Directorio para guardar las imágenes
    
    Returns:
        fig: Figura de matplotlib con el dashboard
    """
    # Implementar Análisis RFM (Recency, Frequency, Monetary)
    df_filtrado = df[df['Nombre cliente'] != 'Ventas Mostrador']
    
    # Simulamos la recencia (en un caso real, necesitaríamos fechas)
    np.random.seed(42)  # Para reproducibilidad
    df_filtrado['Recency'] = np.random.randint(1, 100, size=len(df_filtrado))
    
    # Calculamos Frequency y Monetary
    rfm = df_filtrado.groupby('Nombre cliente').agg({
        'Recency': 'mean',  # Esto es una simulación
        'Nombre producto': 'count',  # Frecuencia
        'Total': 'sum'  # Valor monetario
    }).rename(columns={'Nombre producto': 'Frequency', 'Total': 'Monetary'})
    
    # Normalización de valores para visualización
    rfm_normalized = rfm.copy()
    for col in rfm.columns:
        rfm_normalized[col] = (rfm[col] - rfm[col].min()) / (rfm[col].max() - rfm[col].min())
    
    # Creación del dashboard RFM
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Análisis RFM de Clientes - SURTIGRANOS-ROVIRA', 
                 fontsize=28, y=0.98, fontweight='bold')
    
    # 1. Gráfico de dispersión RFM
    ax1 = axes[0, 0]
    scatter = ax1.scatter(rfm['Recency'], rfm['Frequency'], 
                          s=rfm['Monetary']/rfm['Monetary'].max()*500, 
                          c=rfm['Monetary'], cmap='viridis', alpha=0.6)
    ax1.set_title('Análisis RFM: Recencia vs Frecuencia', fontsize=18, pad=20, fontweight='bold')
    ax1.set_xlabel('Recencia (días desde última compra)', fontsize=14)
    ax1.set_ylabel('Frecuencia (número de compras)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, rfm['Recency'].max() * 1.1)
    ax1.set_ylim(0, rfm['Frequency'].max() * 1.1)
    
    # Añadir colorbar para el valor monetario
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Valor Monetario ($)', fontsize=12)
    
    # 2. Top 10 clientes por valor con desglose RFM
    ax2 = axes[0, 1]
    top_rfm = rfm.sort_values('Monetary', ascending=False).head(10)
    
    # Crear un gráfico de barras apiladas para los valores RFM normalizados
    rfm_top_norm = rfm_normalized.loc[top_rfm.index]
    
    bar_width = 0.8
    indices = np.arange(len(rfm_top_norm))
    
    ax2.bar(indices, rfm_top_norm['Monetary'], bar_width, label='Valor Monetario', color='#f4a261')
    ax2.bar(indices, rfm_top_norm['Frequency'], bar_width, bottom=rfm_top_norm['Monetary'], 
            label='Frecuencia', color='#2a9d8f')
    ax2.bar(indices, rfm_top_norm['Recency'], bar_width, 
            bottom=rfm_top_norm['Monetary'] + rfm_top_norm['Frequency'],
            label='Recencia', color='#e9c46a')
    
    ax2.set_title('Perfiles RFM de Top 10 Clientes', fontsize=18, pad=20, fontweight='bold')
    ax2.set_ylabel('Valor Normalizado', fontsize=14)
    ax2.set_xticks(indices)
    ax2.set_xticklabels([nombre.split()[0] for nombre in rfm_top_norm.index], rotation=45, ha='right', fontsize=12)
    ax2.legend()
    
    # 3. Mapa de calor de correlación entre valores RFM
    ax3 = axes[1, 0]
    corr_matrix = rfm.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax3, vmin=-1, vmax=1, 
                annot_kws={"size": 14}, fmt=".2f", linewidths=0.5)
    ax3.set_title('Correlación entre Métricas RFM', fontsize=18, pad=20, fontweight='bold')
    
    # 4. Distribución de clientes por valor monetario
    ax4 = axes[1, 1]
    sns.histplot(rfm['Monetary'], kde=True, ax=ax4, color='#1f77b4', bins=20)
    ax4.set_title('Distribución de Clientes por Valor Monetario', fontsize=18, pad=20, fontweight='bold')
    ax4.set_xlabel('Valor Monetario Total ($)', fontsize=14)
    ax4.set_ylabel('Número de Clientes', fontsize=14)
    ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3, w_pad=3)
    fig.text(0.01, 0.01, f'Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
             fontsize=8, color='gray')
    
    return fig

def main():
    file_path = "Ventas_por_cliente_por_producto.xlsx"
    output_dir = "output"
    
    try:
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar datos
        print("Cargando datos...")
        df_ventas = cargar_datos(file_path)
        
        # Generar dashboards
        print("Generando dashboard principal...")
        dashboard = crear_dashboard_mejorado(df_ventas, output_dir)
        dashboard.savefig(f'{output_dir}/surtigranos_dashboard_principal.png', dpi=300, bbox_inches='tight')
        
        print("Generando dashboard de segmentación...")
        dashboard_rfm = generar_dashboard_segmentado(df_ventas, output_dir)
        dashboard_rfm.savefig(f'{output_dir}/surtigranos_dashboard_segmentacion.png', dpi=300, bbox_inches='tight')
        
        # Guardar datos procesados
        print("Guardando datos procesados...")
        metricas = analizar_datos(df_ventas)
        with open(f'{output_dir}/metricas_ventas.txt', 'w') as f:
            for k, v in metricas.items():
                f.write(f"{k}: {v}\n")
        
        print("Proceso completo. Los dashboards y datos se encuentran en el directorio 'output'.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()