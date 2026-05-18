# ⚖️ Consideraciones Éticas

Como parte de un proyecto de Machine Learning, es fundamental abordar las implicaciones éticas y el posible impacto social derivado del uso del modelo en producción. El documento original de análisis ético se encuentra en `docs/ETHICS.md`.

## Propósito del Sistema
El propósito principal de este sistema es **optimizar las estrategias de retención de clientes** para la compañía de telecomunicaciones. Predecir a tiempo el riesgo de abandono permite activar promociones o descuentos para clientes descontentos.

## 1. Impacto y Posibles Consecuencias No Deseadas

- **Pérdida de ingresos por Falsos Positivos:** Identificar erróneamente a clientes leales como "en riesgo de Churn". Si se les ofrecen descuentos masivos y automáticos, la empresa podría incurrir en pérdidas financieras significativas.
- **Intervenciones agresivas:** Si el modelo gatilla demasiadas llamadas automatizadas o promociones repetitivas, puede molestar y terminar provocando el abandono de clientes que inicialmente estaban satisfechos.
- **Falta de retención por Falsos Negativos:** El modelo actualmente tiene un "Recall" bajo (alrededor de 0.54). Esto significa que **falla en identificar a casi la mitad de los clientes que realmente se irán**. Se debe ser transparente con los stakeholders sobre esta limitante para no generar expectativas erróneas sobre el impacto financiero positivo del sistema.

## 2. Sesgos y Justicia (Fairness)

Analizamos el dataset y encontramos potenciales problemas de sesgo demográfico que podrían afectar la equidad del modelo:

- **Género (`gender`):** Aunque en una compañía telefónica el sexo no debería dictar la calidad del servicio, si existiese algún sesgo en los datos históricos de retención (ej. ofreciendo mejores ofertas a hombres que a mujeres o viceversa), el modelo podría aprender a discriminar basándose en esta característica protegida.
- **Tercera Edad (`SeniorCitizen`):** Si esta población presenta una tasa de abandono mayor, el modelo podría penalizarlos asumiendo que "siempre se irán". Esto podría derivar en estrategias de venta agresivas enfocadas injustamente en este grupo demográfico.
- **Capacidad Económica:** Factores como `MonthlyCharges` (cargos mensuales) y `TotalCharges` pueden correlacionarse fuertemente con la situación socioeconómica del cliente. Estrategias basadas ciegamente en este modelo podrían resultar en "redlining digital", donde solo se intenta retener a aquellos con alto poder adquisitivo, ignorando o penalizando a sectores de bajos recursos.

## 3. Privacidad de los Datos (Data Privacy)

- **Pseudonimización:** En la fase de limpieza de datos de este proyecto, la variable `customerID` es eliminada inmediatamente (`src/data_loader.py`). Esto ayuda a mitigar la reidentificación de los clientes, asegurando que las decisiones del algoritmo sean sobre perfiles genéricos y no sobre individuos específicamente identificables.
- Sin embargo, para cumplir con normativas como GDPR, en un entorno de producción real, sería imperativo asegurar mecanismos de encriptación y el derecho al olvido para la información de facturación asociada (`TotalCharges`).

---
> **Conclusión Ética:** El despliegue de este modelo debe usarse de manera **asistencial** (Human-in-the-Loop) en lugar de completamente autónoma, permitiendo que agentes humanos de retención revisen las recomendaciones del algoritmo antes de aplicar promociones indiscriminadas.
