# **Análisis Ético, Sesgos y Limitaciones**

Este documento detalla las implicaciones éticas, las limitaciones inherentes y los posibles sesgos identificados en el dataset **Telco Customer Churn** y en los modelos predictivos derivados de él, estructurado para un análisis profundo de la equidad algorítmica.

## **1\. Grupos Subrepresentados y Representatividad**

* **Rangos Etarios:** El dataset carece de la edad exacta del cliente, sustituyéndola únicamente por una variable binaria (SeniorCitizen). Los adultos mayores representan aproximadamente solo el 16% de los datos. Esta falta de granularidad estadística oculta los patrones de comportamiento de grupos intermedios (como adultos jóvenes o de mediana edad) y subrepresenta significativamente a la población de la tercera edad, impidiendo un entrenamiento equitativo para este sector.  
* **Identidad de Género:** La variable gender está restringida estrictamente a un esquema binario (Hombre / Mujer). Aunque estadísticamente estas dos categorías están balanceadas (cerca del 50/50) en el dataset, el modelo excluye y no contabiliza a las personas no binarias o con otras identidades de género, invisibilizando a estas minorías.  
* **Minorías, Raza y Geografía:** El conjunto de datos omite deliberadamente la recopilación de información sobre raza, origen étnico y ubicación geográfica específica (como código postal, ciudad o nivel de ingresos del barrio). Si bien esto añade una capa de anonimato y privacidad, hace algorítmicamente imposible auditar si el servicio de la empresa o el modelo predictivo tiene un rendimiento deficiente o perjudicial hacia comunidades minoritarias específicas.

## **2\. Prejuicios Históricos en las Etiquetas**

La variable objetivo a predecir, Churn (cancelación del servicio), es fundamentalmente una métrica transaccional y de negocio. A diferencia de otros entornos de la inteligencia artificial (como la predicción de reincidencia criminal o la calificación crediticia), esta etiqueta **no suele codificar prejuicios históricos de manera directa**. Sin embargo, existen consideraciones socioeconómicas de alto impacto:

* **Contexto Socioeconómico y Penalización:** Un cliente suele cancelar un servicio por incapacidad de pago temporal o permanente. El dataset muestra una fuerte correlación estadística del "Churn" con los contratos mes a mes (Month-to-month) y pagos mediante cheques electrónicos (Electronic check). El modelo de IA podría aprender a perfilar a los clientes de menores ingresos como un segmento de "alto riesgo de abandono". Esto podría desencadenar decisiones automatizadas donde la empresa les niegue promociones proactivas, limitando los mejores incentivos solo a clientes con alto poder adquisitivo (contratos a 2 años), reproduciendo de forma silenciosa la desigualdad económica.

## **3\. Variables Sensibles Involucradas**

El dataset contiene múltiples atributos personales protegidos que, de ser introducidos a un modelo en producción, requieren auditorías de equidad (Fairness) para evitar la discriminación comercial.

| Variable Sensible | Descripción y Riesgo Ético Algorítmico |
| :---- | :---- |
| **gender** (Género) | Desplegar estrategias de retención de clientes, ajustar precios de ofertas o realizar campañas de marketing diferenciadas basándose únicamente en el género del usuario infringe los principios de equidad de machine learning y podría violar legislaciones de protección al consumidor. |
| **SeniorCitizen** (Edad) | Ignorar sistemáticamente a los adultos mayores en las campañas de retención o asignarles un peso predictivo negativo genera un sesgo algorítmico por edad (edadismo). El modelo podría decidir no ofrecerles soporte técnico gratuito, asumiendo erróneamente un alto riesgo de fuga inevitable. |
| **Partner & Dependents** (Estado Familiar) | Estas columnas revelan explícitamente la estructura familiar íntima del individuo (estado civil y presencia de personas a cargo o hijos). Entrenar modelos que exploten la vulnerabilidad o la necesidad de conectividad de ciertas dinámicas familiares (por ejemplo, familias monoparentales) para maximizar los ingresos es una práctica altamente intrusiva y éticamente cuestionable. |

## **4\. Limitaciones Técnicas del Modelo y Dataset**

* **Desbalance del Objetivo (Class Imbalance):** Dado que la vasta mayoría de los clientes permanece en la empresa y solo una minoría (aprox. 26%) presenta Churn positivo, cualquier modelo estándar tenderá a sesgarse hacia predecir "No Churn" de forma conservadora. Esto limita su sensibilidad real para detectar a los usuarios que más ayuda o intervención necesitan.  
* **Correlación no implica Causalidad:** El algoritmo detectará, por ejemplo, que los usuarios de Fibra Óptica tienen una mayor tasa de cancelación. El modelo asocia la variable, pero ignora el contexto (¿Es la fibra óptica de mala calidad? ¿Es demasiado cara en comparación con la competencia?). Tomar decisiones automatizadas sin investigar la causa raíz llevará a estrategias de negocio contraproducentes.