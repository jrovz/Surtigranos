import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Función para cargar y limpiar los datos
def cargar_datos(file_path):
    # Cargar los datos limpios
    df_ventas = pd.read_excel(file_path, skiprows=7)
    
    # Eliminar columnas vacías
    df_ventas = df_ventas.dropna(axis=1, how='all')
    
    return df_ventas

# Función para crear el dashboard
def crear_dashboard(df):
    # Configuración visual
    plt.style.use('seaborn-v0_8')
    sns.set_palette("viridis")
    
    # Crear una figura principal para organizar los subplots
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('Dashboard de Ventas - SURTIGRANOS-ROVIRA', fontsize=24, y=0.98)
    
    # 1. Top 10 productos más vendidos por cantidad
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    top_cantidad = df.groupby('Nombre producto')['Cantidad vendida'].sum().sort_values(ascending=False).head(10)
    top_cantidad.plot(kind='barh', ax=ax1)
    ax1.set_title('Top 10 Productos por Cantidad Vendida', fontsize=14)
    ax1.set_xlabel('Cantidad vendida')
    ax1.set_ylabel('Producto')
    
    # 2. Top 10 productos por ventas totales
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
    top_ventas = df.groupby('Nombre producto')['Total'].sum().sort_values(ascending=False).head(10)
    top_ventas.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_title('Distribución de Ventas por Producto (Top 10)', fontsize=14)
    ax2.set_ylabel('')
    
    # 3. Margen bruto por producto (Top 10 por margen)
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    # Verificar si la columna 'Margen' ya existe
    if 'Margen' not in df.columns:
        df['Margen'] = df['Subtotal'] - df['Valor bruto'] + df['Descuento']
    top_margen = df.groupby('Nombre producto')['Margen'].sum().sort_values(ascending=False).head(10)
    top_margen.plot(kind='bar', ax=ax3, color='orange')
    ax3.set_title('Top 10 Productos por Margen Bruto', fontsize=14)
    ax3.set_xlabel('Producto')
    ax3.set_ylabel('Margen Bruto')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Distribución de ventas por cliente
    ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=1)
    ventas_cliente = df.groupby('Nombre cliente')['Total'].sum().sort_values(ascending=False).head(10)
    ventas_cliente.plot(kind='pie', autopct='%1.1f%%', ax=ax4)
    ax4.set_title('Distribución de Ventas por Cliente (Top 10)', fontsize=14)
    ax4.set_ylabel('')
    
    # 5. Análisis de productos: Cantidad vs Valor Total
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    producto_analisis = df.groupby('Nombre producto').agg({
        'Cantidad vendida': 'sum',
        'Total': 'sum'
    }).sort_values('Total', ascending=False).head(15)
    
    sns.scatterplot(data=producto_analisis.reset_index(), 
                   x='Cantidad vendida', 
                   y='Total', 
                   size='Total', 
                   hue='Nombre producto',
                   ax=ax5)
    ax5.set_title('Análisis de Productos: Cantidad vs Valor Total de Ventas', fontsize=14)
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar el layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

# Crear una función para generar un reporte de productos clave
def generar_reporte_productos(df):
    # Calcular métricas importantes
    productos_resumen = df.groupby('Nombre producto').agg({
        'Cantidad vendida': 'sum',
        'Valor bruto': 'sum',
        'Total': 'sum',
        'Impuesto cargo': 'sum'
    }).reset_index()
    
    # Calcular margen y rentabilidad
    productos_resumen['Margen'] = productos_resumen['Total'] - productos_resumen['Valor bruto']
    productos_resumen['Rentabilidad_pct'] = (productos_resumen['Margen'] / productos_resumen['Valor bruto'] * 100).round(2)
    
    # Ordenar por Total (ventas) descendente
    productos_resumen = productos_resumen.sort_values('Total', ascending=False)
    
    # Identificar productos más rentables
    productos_mas_rentables = productos_resumen.sort_values('Rentabilidad_pct', ascending=False).head(10)
    
    return productos_resumen, productos_mas_rentables

# Crear una función para analizar clientes
def analizar_clientes(df):
    # Análisis de clientes
    clientes_resumen = df.groupby('Nombre cliente').agg({
        'Código producto': 'nunique',  # Número de productos distintos
        'Cantidad vendida': 'sum',
        'Total': 'sum'
    }).reset_index()
    
    clientes_resumen = clientes_resumen.rename(columns={'Código producto': 'Variedad_productos'})
    clientes_resumen = clientes_resumen.sort_values('Total', ascending=False)
    
    return clientes_resumen

# Función principal para ejecutar el análisis
def main():
    # Definir ruta del archivo - ajusta esta ruta según donde tengas el archivo
    file_path = "Ventas_por_cliente_por_producto.xlsx"
    
    try:
        # Cargar los datos
        df_ventas = cargar_datos(file_path)
        
        print(f"Datos cargados exitosamente. {len(df_ventas)} registros encontrados.")
        
        # Crear dashboard
        dashboard = crear_dashboard(df_ventas)
        dashboard.savefig('surtigranos_dashboard.png', dpi=300, bbox_inches='tight')
        print("Dashboard generado y guardado como 'surtigranos_dashboard.png'")
        
        # Generar reportes
        productos_resumen, productos_rentables = generar_reporte_productos(df_ventas)
        clientes_resumen = analizar_clientes(df_ventas)
        
        # Imprimir o guardar los resultados
        productos_resumen.to_csv('productos_resumen.csv', index=False)
        productos_rentables.to_csv('productos_mas_rentables.csv', index=False)
        clientes_resumen.to_csv('clientes_resumen.csv', index=False)
        
        print("Análisis completado. Se han generado los siguientes archivos:")
        print("- productos_resumen.csv")
        print("- productos_mas_rentables.csv")
        print("- clientes_resumen.csv")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'")
        print("Verifica que el archivo esté en la misma carpeta que este script o proporciona la ruta completa.")
    except Exception as e:
        print(f"Error al procesar los datos: {str(e)}")

# Ejecutar el análisis si se ejecuta este script directamente
if __name__ == "__main__":
    main()