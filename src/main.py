import yaml
import sys
import os
from src.trainer_model import train_and_save_model  # O model_trainer, según lo hayas dejado
from src.data_loader import load_and_preprocess_data


def main():
    # 1. Obtener la ruta absoluta del archivo main.py actual
    ruta_script = os.path.abspath(__file__)

    # 2. Obtener el directorio donde está el script (carpeta 'src')
    directorio_src = os.path.dirname(ruta_script)

    # 3. Subir un nivel para llegar a la raíz del proyecto ('project-root-main')
    raiz_proyecto = os.path.dirname(directorio_src)

    # 4. Construir la ruta final hacia el archivo yaml
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')

    try:
        # Usar la nueva ruta dinámica en lugar del string estático
        with open(ruta_config, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta_config}")
        sys.exit(1)

    print("Iniciando pipeline de Machine Learning...")

    # -> Cargando y preprocesando datos...
    print("-> Cargando y preprocesando datos...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print(f"   Datos procesados: {X_train.shape[0]} muestras de entrenamiento.")

    # -> Entrenando y guardando el modelo...
    print(f"-> Entrenando modelo seleccionado: {config['model']['name']}...")
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    # -> Mostrar resultados
    print("\n¡Entrenamiento completado exitosamente! Métricas de validación:")
    for metric_name, value in metrics.items():
        print(f" - {metric_name.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()