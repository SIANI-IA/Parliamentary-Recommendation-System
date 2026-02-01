import os
import pickle
import pandas as pd
import numpy as np

def collect_metrics_and_filter(root_folder, output_csv="benchmark_results.csv"):
    """
    Recorre carpetas, extrae métricas y filtra para dejar solo las deseadas para el paper.
    """
    extracted_data = []
    
    print(f"Explorando carpeta: {root_folder} ...")

    # 1. Iterar y extraer
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if not isinstance(data, dict) or "metrics" not in data:
                        continue
                        
                    # Extraer nombre y métricas
                    model_name = data.get("model_name", file.replace(".pkl", ""))
                    metrics = data["metrics"]
                    
                    row = {"Model": model_name.replace("_", "-")}
                    row.update(metrics)
                    extracted_data.append(row)
                    
                except Exception as e:
                    print(f"   [!] Error en {file}: {e}")

    if not extracted_data:
        print("No se encontraron datos.")
        return None

    # 2. Crear DataFrame Completo
    df_full = pd.DataFrame(extracted_data)

    # ---------------------------------------------------------
    # 3. FILTRADO Y ORDENAMIENTO (La parte importante)
    # ---------------------------------------------------------
    
    # Lista de columnas que SI queremos mantener
    # A. Identificadores y Ranking (Prioridad)
    cols_to_keep = ["Model", "R-Precision", "MAP", "nDCG"]
    
    # B. Añadir dinámicamente todos los Recall@... (Ranking)
    #    (Cuidado de no incluir 'Recall (Micro)' o 'Recall (Macro)')
    cols_to_keep += [c for c in df_full.columns if "Recall@" in c]
    
    # C. Métricas de Clasificación Específicas (Solo F1)
    #    Ajusta estos strings si tus keys son ligeramente distintas (ej: "F1-Micro" vs "F1-Score (Micro)")
    target_classif = ["F1-Score (Micro)", "F1-Score (Macro)", "F1-Micro", "F1-Macro"]
    cols_to_keep += [c for c in target_classif if c in df_full.columns]

    # Filtrar: Solo nos quedamos con las columnas que existen en el DF y están en nuestra lista
    final_cols = [c for c in cols_to_keep if c in df_full.columns]
    
    # Crear el DF filtrado
    df_filtered = df_full[final_cols].copy()
    
    # Guardar CSV
    df_filtered.to_csv(output_csv, index=False)
    print(f"\nCSV (Filtrado) guardado en: {output_csv}")
    
    return df_filtered

def generate_latex(df, output_latex="benchmark_table.tex", dataset_name=""):
    if df is None: return

    # Trabajamos sobre una copia para no romper el original con strings de latex
    df_latex = df.copy()
    
    # 1. LOGICA DE RESALTADO (Negrita al máximo por COLUMNA)
    numeric_cols = df_latex.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        # Encontramos el máximo numérico real de la columna
        max_val = df_latex[col].max()
        
        def format_cell(val):
            if pd.isna(val): return "-"
            # Formato base: 3 decimales
            str_val = f"{val:.3f}"
            
            # Si es el máximo (con tolerancia pequeña), poner negrita
            if abs(val - max_val) < 1e-9:
                return f"\\textbf{{{str_val}}}"
            return str_val
        
        # Aplicamos el formato
        df_latex[col] = df_latex[col].apply(format_cell)

    # 2. Limpiar nombres de modelos para LaTeX (escapar guiones bajos)
    df_latex["Model"] = df_latex["Model"].apply(lambda x: x.replace("_", "\\_"))

    # 3. Generar el cuerpo de la tabla
    latex_body = df_latex.to_latex(
        index=False,
        escape=False, # Importante para que funcione el \textbf
        column_format="l" + "c" * (len(df_latex.columns) - 1) # Izq el modelo, Centro los números
    )

    # 4. Envolver en estructura de tabla ajustada al ancho (resizebox)
    # Esto hace que la tabla se encoja si es más ancha que la página vertical
    caption = f"Benchmark Results for {dataset_name} (Best scores in bold)"
    final_latex = """
\\begin{table}[h]
    \\centering
    \\caption{""" + caption + """}
    \\label{tab:results}
    \\resizebox{\\textwidth}{!}{%
""" + latex_body + """
    }
\\end{table}
"""

    print("\nvvvvv CÓDIGO LATEX LISTO (COPIA ESTO) vvvvv\n")
    print(final_latex)
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    
    # Guardar fichero
    with open(output_latex, "w") as f:
        f.write(final_latex)
    print("Nota: Asegúrate de incluir \\usepackage{graphicx} en tu main.tex para que funcione el resizebox.")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    folder_path = "./results/parcanDeb-rec-split"
    dataset_name = os.path.basename(folder_path.rstrip('/'))
    
    if os.path.exists(folder_path):
        output_csv = os.path.join(folder_path, "benchmark_results_filtered.csv")
        df = collect_metrics_and_filter(folder_path, output_csv=output_csv)
        output_latex = os.path.join(folder_path, "benchmark_table.tex")
        generate_latex(df, output_latex=output_latex, dataset_name=dataset_name)
    else:
        print("Carpeta no encontrada.")