import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

def cargar_datos(file_path):
    df_ventas = pd.read_excel(file_path, skiprows=7)
    df_ventas = df_ventas.dropna(axis=1, how='all')
    return df_ventas

def crear_dashboard_mejorado(df):
    df_filtrado = df[df['Nombre cliente'] != 'Ventas Mostrador']
    plt.style.use('ggplot')
    colores_productos = sns.color_palette("tab10", 10)  # Paleta para productos
    colores_clientes = sns.color_palette("hls", 10)  # Paleta para clientes
   
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Dashboard de Ventas - SURTIGRANOS-ROVIRA (Clientes Registrados)', fontsize=28, y=0.98, fontweight='bold')
    
    top_cantidad = df_filtrado.groupby('Nombre producto')['Cantidad vendida'].sum().sort_values(ascending=False).head(10)
    ax1 = axes[0, 0]
    bars = ax1.barh(top_cantidad.index, top_cantidad.values, color=colores_productos)
    ax1.set_title('Top 10 Productos por Cantidad Vendida', fontsize=18, pad=20, fontweight='bold')
    ax1.set_xlabel('Cantidad vendida', fontsize=14)
    ax1.legend(bars, top_cantidad.index, title="Productos", loc="upper right", fontsize=10)
    
    top_ventas = df_filtrado.groupby('Nombre producto')['Total'].sum().sort_values(ascending=False).head(10)
    ax2 = axes[0, 1]
    wedges, _, _ = ax2.pie(top_ventas.values, labels=None, autopct='%1.1f%%', explode=[0.05] * len(top_ventas), colors=colores_productos, shadow=False, startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Distribución de Ventas por Producto (Top 10)', fontsize=18, pad=20, fontweight='bold')
    ax2.legend(wedges, top_ventas.index, title="Productos", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    top_clientes_valor = df_filtrado.groupby('Nombre cliente')['Total'].sum().sort_values(ascending=False).head(10)
    clientes_primernombre = [nombre.split()[0] for nombre in top_clientes_valor.index]

    
    ax3 = axes[1, 0]
    bars = ax3.bar(clientes_primernombre, top_clientes_valor.values, color=colores_clientes)
    ax3.set_title('Top 10 Clientes por Valor Total de Compras', fontsize=18, pad=20, fontweight='bold')
    ax3.set_ylabel('Total Compras ($)', fontsize=14)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax3.set_xticklabels(clientes_primernombre, rotation=30, ha="right", fontsize=12)

    ventas_cliente = df_filtrado.groupby('Nombre cliente')['Total'].sum().sort_values(ascending=False).head(10)
    ax4 = axes[1, 1]
    wedges, _, _ = ax4.pie(ventas_cliente.values, labels=None, autopct='%1.1f%%', explode=[0.05] * len(ventas_cliente), colors=colores_clientes, shadow=False, startangle=90, textprops={'fontsize': 10})
    ax4.set_title('Distribución de Ventas por Cliente (Top 10)', fontsize=18, pad=20, fontweight='bold')
    ax4.legend(wedges, ventas_cliente.index, title="Clientes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3, w_pad=3)
    return fig

def main():
    file_path = "Ventas_por_cliente_por_producto.xlsx"
    try:
        df_ventas = cargar_datos(file_path)
        dashboard = crear_dashboard_mejorado(df_ventas)
        dashboard.savefig('surtigranos_dashboard_1.png', dpi=300, bbox_inches='tight')
        print("Dashboard mejorado generado correctamente.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()