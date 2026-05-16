import sys
import os
import random
import yaml
import numpy as np
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.main [prepare|train]")
        sys.exit(1)
        
    action = sys.argv[1]

    ruta_script = os.path.abspath(__file__)
    directorio_src = os.path.dirname(ruta_script)
    raiz_proyecto = os.path.dirname(directorio_src)
    ruta_config = os.path.join(raiz_proyecto, 'config', 'params.yaml')

    try:
        with open(ruta_config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta_config}")
        sys.exit(1)

    semilla_global = config['data_split']['random_state']
    random.seed(semilla_global)
    np.random.seed(semilla_global)

    if action == "prepare":
        print("Iniciando etapa: [PREPARE] - Cargando y preprocesando datos...")
        X_train, _, _, _ = load_and_preprocess_data(config)
        print(f"✔ Datos preparados: {X_train.shape[0]} muestras de entrenamiento generadas.")
        
    elif action == "train":
        print(f"Iniciando etapa: [TRAIN] - Entrenando modelo seleccionado: {config['model']['name']}...")
        metrics = train_and_save_model(config)
        print("\n¡Entrenamiento completado exitosamente! Métricas de validación:")
        for metric_name, value in metrics.items():
            print(f" - {metric_name.capitalize()}: {value:.4f}")
            
    else:
        print(f"Acción '{action}' no reconocida. Usa 'prepare' o 'train'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
