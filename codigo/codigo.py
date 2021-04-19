
#%%
## Paso 3: Prepare los datos
# 3c. 

'''Para preparar los datos, entrenar el modelo de aprendizaje automático e implementarlo, 
deberá importar algunas bibliotecas y definir algunas variables del entorno en su entorno de bloc de notas de Jupyter. 
Ejecute el siguiente código: '''

# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                 
from sagemaker.predictor import csv_serializer   

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")

#%%
### 3d. 
'''En este paso, creará un bucket de S3 que almacenará sus datos para este tutorial.
Copie el siguiente código en la próxima celda de código de su bloc de notas y 
cambie el nombre del bucket de S3 para que sea único. Los nombres de los buckets 
de S3 deben ser únicos a nivel mundial y, además, deben contar con algunas 
restricciones y limitaciones.
'''
bucket_name = 'your-s3-bucket-name' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)

#%%
### 3e. 

'''
A continuación, debe descargar los datos en su instancia de Amazon SageMaker 
y cargarlos en un marco de datos. Copie y Ejecute el siguiente código:
'''

try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  print('Success: downloaded bank_clean.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./bank_clean.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)

#%%

### 3f. 
'''
Ahora, mezclaremos los datos y los dividiremos en datos de entrenamiento y de prueba.

Los datos de entrenamiento (el 70 % de los clientes) se utilizarán durante el ciclo 
de entrenamiento del modelo. Utilizaremos la optimización basada en gradientes para 
refinar de forma iterativa los parámetros del modelo. La optimización basada en gradientes 
es una forma de encontrar valores de parámetros del modelo que minimicen sus errores, 
mediante el uso de gradientes de la función de pérdida del modelo.

Los datos de prueba (el 30 % restante de los clientes) se utilizarán para evaluar el 
rendimiento del modelo y para medir el nivel de generalización de los datos nuevos del
 modelo entrenado.

Copie el siguiente código en una nueva celda de código y seleccione Ejecutar para mezclar
 y dividir los datos:
'''

train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)

#%%
## Paso 4: Entrene el modelo con los datos

'''En este paso, entrenará su modelo de aprendizaje automático con el conjunto de datos de 
entrenamiento. 
'''
### 4a. 
'''Para utilizar un modelo XGBoost prediseñado de Amazon SageMaker, deberá cambiar el formato
 del encabezado y la primera columna de los datos de entrenamiento y cargar los datos desde
  el bucket de S3.

Copie el siguiente código en una nueva celda de código y seleccione Ejecutar para cambiar el 
formato y cargar los datos:
'''

pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

#%%
### 4b. 

'''A continuación, deberá configurar la sesión de Amazon SageMaker, crear una instancia del 
modelo XGBoost (un estimador) y definir los hiperparámetros del modelo. Copie el siguiente 
código en una nueva celda de código y seleccione Ejecutar:
'''

sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(containers[my_region],role, train_instance_count=1, train_instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
#%%
### 4c. 
'''
Con los datos cargados y el estimador XGBoost configurado, entrene el modelo a través de 
la optimización basada en gradientes en una instancia ml.m4.xlarge; copie el siguiente 
código en la próxima celda de código y seleccione Ejecutar.

Luego de algunos minutos, debería comenzar a ver los registros de entrenamiento que se 
generen.
'''

xgb.fit({'train': s3_input_train})
#%%
## Paso 5: Implemente el modelo
'''
En este paso, implementará el modelo entrenado en un punto de enlace, cambiará el formato 
y cargará los datos CSV. Luego, ejecutará el modelo para crear predicciones.
'''
### 5a. 
'''
Para implementar el modelo en un servidor y crear un punto de enlace al que pueda acceder, 
copie el siguiente código en la próxima celda de código y seleccione Ejecutar:
'''

xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')

#%%

### 5b. 
'''
Para predecir si los clientes de los datos de prueba se inscribieron o no en el producto 
del banco, copie el siguiente código en la próxima celda de código y seleccione Ejecutar:
'''
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)

#%%
## Paso 6. Evalúe el rendimiento del modelo
'''
En este paso, evaluará el rendimiento y la precisión del modelo de aprendizaje automático.
'''
### 6a. 

'''Copie y pegue el siguiente código y seleccione Ejecutar para comparar los valores reales con 
los valores predichos en una tabla denominada matriz de confusión.

En función de las predicciones, podemos concluir que usted predijo que un cliente se 
inscribiría para un certificado de depósito exactamente para el 90 % de los clientes en los 
datos de prueba, con una precisión del 65 % (278/429) para los inscritos y del 90 % (10 785/11 928) 
para los no inscritos.
'''

cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*
#%%

## Paso 7: Termine los recursos
'''En este paso, terminará los recursos relacionados con Amazon SageMaker.

Importante: Terminar los recursos que no se utilizan de forma activa reduce los costos, 
y es una práctica recomendada. No terminar sus recursos generará cargos.
'''
### 7a.
''' 
Para eliminar el punto de enlace de Amazon SageMaker y los objetos de su bucket de S3, 
copie, pegue y Ejecute el siguiente código:  
'''
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()