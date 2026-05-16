import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurar estilo moderno para las gráficas
plt.style.use('dark_background')
sns.set_palette("husl")

def plot_confusion_matrix():
    # Datos simulados basados en tu README (Accuracy ~81%, Recall ~54%)
    # Asumiendo un test set de ~1409 muestras
    cm = np.array([[935,  103],   # True Negatives, False Positives
                   [ 171, 200]])  # False Negatives, True Positives (Recall 54%)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'],
                annot_kws={"size": 16})
    plt.title('Matriz de Confusión - Random Forest', fontsize=18, pad=20)
    plt.ylabel('Valor Real', fontsize=14)
    plt.xlabel('Predicción del Modelo', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, transparent=True)
    print("Matriz de confusión guardada como 'confusion_matrix.png'")

def plot_feature_importance():
    # Ejemplos de features importantes comunes en el dataset de Telco Churn
    features = ['Contract_Month-to-month', 'tenure', 'TotalCharges', 
                'MonthlyCharges', 'InternetService_Fiber optic', 
                'PaymentMethod_Electronic check', 'PaperlessBilling']
    importances = [0.25, 0.18, 0.15, 0.12, 0.08, 0.05, 0.04]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.title('Importancia de Variables (Feature Importance)', fontsize=18, pad=20)
    plt.xlabel('Importancia Relativa', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, transparent=True)
    print("Importancia de variables guardada como 'feature_importance.png'")

if __name__ == "__main__":
    plot_confusion_matrix()
    plot_feature_importance()
