# **Documentación del Dataset: Telco Customer Churn**

## **1\. Descripción del dataset**

Este conjunto de datos contiene información detallada sobre los clientes de una operadora de telecomunicaciones, diseñada para el análisis de la tasa de cancelación (churn).

* **Contenido:** Información demográfica, servicios contratados (teléfono, internet, seguridad, etc.) y detalles financieros de la cuenta.  
* **Dimensiones:** 7,043 registros (filas) y 21 variables (columnas).  
* **Tipos de datos:**  
  * **Numéricos:** tenure (meses), MonthlyCharges (flotante), TotalCharges (flotante).  
  * **Categóricos:** 18 variables que incluyen gender, Contract, InternetService y la etiqueta objetivo Churn.

## **2\. Problema que resuelve**

Se trata fundamentalmente de un problema de **Clasificación Binaria**. El objetivo es predecir si un cliente abandonará la compañía (Yes) o permanecerá en ella (No) durante el próximo ciclo.

## **3\. Aplicaciones prácticas**

| Aplicación | Descripción |
| :---- | :---- |
| Programas de Lealtad | Identificar clientes en riesgo para ofrecerles planes de fidelización. |
| Optimización de Precios | Ajustar tarifas según el impacto observado en la retención por tipo de contrato. |
| Venta Cruzada (Cross-selling) | Identificar qué servicios adicionales (seguridad, backup) ayudan más a retener al cliente. |

## **4\. Implicaciones éticas y sesgos**

* **Desbalance de datos:** La mayoría de los clientes no cancelan, lo que puede causar que los modelos fallen al detectar a los que sí lo hacen.  
* **Sesgo algorítmico:** El uso de datos como la edad (SeniorCitizen) o el género podría generar ofertas desiguales basadas en perfiles protegidos.  
* **Generalización:** Los patrones de este dataset de IBM pueden no reflejar la realidad de mercados fuera de Norteamérica o en años recientes.