# 📡 Bienvenido al Wiki de MLOps Churn Prediction

¡Bienvenido a la documentación oficial del proyecto **MLOps Churn Prediction**! 

Este proyecto simula un entorno laboral real para resolver un problema de clasificación clave en la industria de telecomunicaciones: **Predecir si un cliente cancelará su servicio (Churn)**. A través de este Wiki, documentamos a detalle cada componente de la solución y las prácticas de MLOps aplicadas.

<p align="center">
  <img src="https://raw.githubusercontent.com/RobbeAlex/churn-mlops-project/main/docs/images/banner.png" alt="MLOps Churn Prediction Banner" width="100%">
</p>

## 🎯 Objetivo del Proyecto

Desarrollar un sistema de Machine Learning end-to-end automatizado, reproducible y escalable. Esto incluye desde la ingesta de datos y el entrenamiento de modelos, hasta el despliegue de una API para la inferencia, asegurando prácticas de integración continua (CI) y metodologías rigurosas de desarrollo de software para IA.

## 📚 Índice de Contenidos

Navega por las secciones del Wiki para entender cada aspecto del pipeline:

1. 🏗️ **[Arquitectura y Tecnologías](./Arquitectura-y-Tecnologias)**: Diseño del pipeline MLOps, stack tecnológico usado y flujo de datos.
2. 🚀 **[Guía de Instalación y Uso](./Guia-de-Instalacion-y-Uso)**: Cómo configurar el entorno (Docker/Virtualenv) y ejecutar la API o re-entrenar el modelo.
3. 📊 **[Datos y Preprocesamiento](./Datos-y-Preprocesamiento)**: Todo sobre el dataset *Telco Customer Churn*, su origen, limpieza y transformaciones.
4. 🧠 **[Modelado y Experimentación](./Modelado-y-Experimentacion)**: Detalles del Random Forest, métricas (Accuracy, Recall) y seguimiento con MLflow y DVC.
5. 🌐 **[Despliegue de la API](./Despliegue-de-la-API)**: Uso de la API REST construida con FastAPI, endpoints y ejemplos de *request/response*.
6. ✅ **[Calidad de Código y Pruebas](./Calidad-de-Codigo-y-Pruebas)**: Flujos de CI, `pytest`, cobertura de código y uso de `pre-commit`.
7. ⚖️ **[Consideraciones Éticas](./Consideraciones-Eticas)**: Análisis crítico de los datos, sesgos potenciales e impacto social del modelo.

---

> **Tip para Colaboradores**: Si deseas contribuir, asegúrate de leer primero la sección de **Calidad de Código y Pruebas** para entender nuestros estándares de código antes de enviar un Pull Request.
