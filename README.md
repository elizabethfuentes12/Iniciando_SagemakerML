# Iniciando Sagemaker ML

En este tutorial te enseñare un poco de Sagemaker usando un ejemplo sencillo de Machine Learning. 

## ¿Que es Sagemaker?
Aprendizaje automático al alcance de cualquier científico de datos y desarrollador. 

Amazon [SageMaker](https://aws.amazon.com/es/sagemaker/) ayuda a los científicos de datos y a los desarrolladores a preparar, crear, entrenar e implementar con rapidez modelos de aprendizaje automático de alta calidad al poner a disposición un amplio conjunto de capacidades especialmente creadas para el aprendizaje automático.

[Video: Introduction to Amazon SageMaker](https://www.youtube.com/watch?v=Qv_Tr_BCFCQ)

!["sagemaker"](imagenes/sagemaker.png)

## Casos de Uso
- [Mantenimiento predictivo](https://pages.awscloud.com/Implementing-Machine-Learning-Solutions-with-Amazon-SageMaker_2019_0722-MCL_OD.html?&trk=sl_card&trkCampaign=NA-FY19-AWS-DIGMKT-WEBINAR-SERIES-July_2019_0722-MCL&sc_channel=el&sc_campaign=pac_2018-2019_exlinks_ondemand_OTT_evergreen&sc_outcome=Product_Adoption_Campaigns&sc_geo=NAMER&sc_country=mult&trkcampaign=wbnrondemand)
- [Visión artificial](https://aws.amazon.com/es/blogs/iot/sagemaker-object-detection-greengrass-part-1-of-3/)
- [Conducción autónoma](https://aws.amazon.com/es/blogs/machine-learning/labeling-data-for-3d-object-tracking-and-sensor-fusion-in-amazon-sagemaker-ground-truth/)
- [Detección de fraudes](https://www.youtube.com/watch?v=elRQPCHDBPE&t=4s)
- [Predicción de riesgos crediticios](https://www.youtube.com/watch?v=Nlwz4cU68T8)
- [Extracción y análisis de datos a partir de documentos](https://aws.amazon.com/es/blogs/machine-learning/maximizing-nlp-model-performance-with-automatic-model-tuning-in-amazon-sagemaker/)
- [Predicción de pérdida de clientes](https://aws.amazon.com/es/blogs/machine-learning/making-machine-learning-predictions-in-amazon-quicksight-and-amazon-sagemaker/)
- [Previsión de demanda](https://www.youtube.com/watch?v=A04TT68Bd8A)
- [Recomendaciones personalizadas](https://aws.amazon.com/es/blogs/startups/how-dely-uses-amazon-sagemaker-to-deliver-personalized-recipes/)

Fuente y github con modelos: [Casos de uso](https://aws.amazon.com/es/sagemaker/getting-started/)

## Algoritmos integrados de AWS Sagemaker
Fuente: https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/algos.html

AWS Sagemaker proporciona una serie de algoritmos integrados que ayuda a mejorar la preforma del aprendizaje automático. 

| Modelo | Tipos de problemas | Algoritmos |
|:---:|:---:|:---:|
|[Aprendizaje supervisado](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/algos.html#algorithms-built-in-supervised-learning)| Clasificación binaria/de varias clases - Regresión - Previsión de series temporales |Linear Learner Algorithm - Factorization Machines Algorithm - XGBoost Algorithm - K-Nearest Neighbors (k-NN) Algorithm |
|[Aprendizaje no supervisado](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/algos.html#algorithms-built-in-unsupervised-learning)|Ingeniería de características: reducción de la dimensionalidad - Detección de anomalías - Integraciones: convierten objetos de grandes dimensiones en espacio de baja dimensionalidad - Agrupación o agrupación en clústeres - Modelado de temas |  |
|[Análisis de texto](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/algos.html#algorithms-built-in-text-analysis) |Clasificación de textos - Traducción automática de Algoritmo - Resumir texto - Texto a voz |
|[Gema Image Processing](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/algos.html#algorithms-built-in-image-processing) |Clasificación de imágenes y etiquetas múltiple - Detección y clasificación de objetos - Visión artificial |


## Tutorial Crear, entrenar e implementar un modelo de Machine Learning con AWS Sagemaker

Fuente: https://aws.amazon.com/es/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/

En este tutorial, aprenderá a utilizar Amazon SageMaker para crear, entrenar e implementar un modelo de aprendizaje automático (ML). Para este ejercicio, utilizaremos el conocido algoritmo de aprendizaje automático [XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) el cual esta integrado en Sagemaker. 

En este tutorial, asumirá el rol de un desarrollador de aprendizaje automático que trabaja en un banco. Se le solicita desarrollar un modelo de aprendizaje automático para predecir si los clientes se inscribirán para un certificado de depósito. El modelo se entrenará con el conjunto de datos de marketing que contiene la información demográfica de los clientes, sus respuestas a los eventos de marketing y los factores externos.

Los datos se etiquetaron para su conveniencia. Una columna en el conjunto de datos identifica si el cliente está inscrito para algún producto que ofrece el banco. Una versión de este conjunto de datos está disponible para el público en el repositorio de aprendizaje automático a cargo de la Universidad de California, Irvine. Este tutorial implementa un modelo de aprendizaje automático supervisado debido a que los datos están etiquetados. (El aprendizaje no supervisado tiene lugar cuando los conjuntos de datos no están etiquetados).

En este tutorial, hará lo siguiente:

1. Creará una instancia de bloc de notas
2. Preparará los datos
3. Entrenará el modelo para aprender de los datos
4. Implementará el modelo
5. Evaluará el rendimiento de su modelo de aprendizaje automático
 
Los recursos creados y utilizados en este tutorial se pueden utilizar en la capa gratuita de AWS. Recuerde completar el Paso 7 y terminar sus recursos. Si su cuenta ha estado activa con estos recursos por más de dos meses, se cobrará menos de 0,50 USD por ella.

## Paso 1: Abra la consola de Amazon SageMaker

Diríjase a la consola de Amazon SageMaker.

!["sagemaker"](imagenes/sage.png)

Abra SageMaker

## Paso 2: Cree una instancia de bloc de notas de Amazon SageMaker

En este paso, creará una instancia de bloc de notas de Amazon SageMaker. 
 
### 2.a 

Abra notebook instances

!["sagemaker"](imagenes/dos.png)

y seleccione Create notebook Instance en la parte superior derecha

!["sagemaker"](imagenes/dosa.png)

### 2b. 
En la página Crear instancia de bloc de notas, escriba un nombre en el campo Nombre de la instancia de bloc de notas. Este tutorial utiliza MySageMakerInstance como nombre de la instancia, pero puede elegir un nombre diferente si lo desea.

Para este tutorial, puede mantener el Tipo de instancia de bloc de notas predeterminado ml.t2.medium.

Para permitir que la instancia de bloc de notas acceda a Amazon S3 y pueda cargar datos de manera segura en este servicio, se debe especificar un rol de IAM. En el campo Rol de IAM, elija Crear un nuevo rol para que Amazon SageMaker cree un rol con los permisos necesarios y lo asigne a su instancia. De forma alternativa, puede elegir un rol de IAM existente en su cuenta para este fin.

## Ejemplos AWS Sagemaker

https://github.com/aws/amazon-sagemaker-examples