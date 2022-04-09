# FUSA: Datasets y rutinas para entrenar redes neuronales

El archivo `FUSA_taxonomy.json` define las equivalencies entre las clases propias de los datasets y la taxonomía FUSA

Los experimentos esperan que esté instalado el paquete [`fusanet_utils`](https://github.com/fusa-project/fusa-net-utils)

## Instalación de librerías

Crear un entorno virtual con Python 3.8 y activarlo:
```
virtualenv fusa-training --python=python3.8
source fusa-training/bin/activate
```

Luego, instalar las librerías requeridas:
```
pip install -r requirements.txt
```

## Descarga de datasets

### Instalación de DVC

https://dvc.org/doc/install/linux

### Configuración de almacenamiento remoto dvc (patagon)

Primero debemos configurar el usuario del almacenamiento remoto, en este caso se creará un entrada global y por defecto 
```
dvc remote add --default patagon ssh://patagon.uach.cl/home/shared/FUSA/remote-storage
dvc remote modify patagon port 2237
dvc remote modify patagon user SSH_USER
dvc remote modify patagon keyfile SSH_KEY
```
donde `SSH_USER` es el usuario con el cual ingresamos al servidor Patagon y `SSH_KEY` es la ruta donde esta alojada la llave privada.

### Descargar dataset

Con el almacenamiento remoto configurado, sólo debemos traer el dataset:
```
dvc pull
```

## Entrenamientos

Cree una subcarpeta dentro de `experiments`, por ejemplo `experiments/FUSA-tag-vitglobal`

Agregue a esta carpeta
- un archivo de configuración de experimento: `dvc.yaml`
- un archivo de configuración con los parámetros del entrenamiento: `params.yaml`

Puede copiarlos desde `experiments/template` y modificarlos

Para entrenar sin dvc (debug) puede utilizar (desde la subcarpeta)

    python ../run_experiment.py --verbose --train --cuda --root_path ../../ --model_path model.pt

Para entrenar con dvc puede utilizar (desde la subcarpeta)
 
    dvc repro train
    
Para entrenar con dvc en patagon (desde la subcarpeta)

    sbatch slurm_patagon.sh train
    
Esto último asume que existe un contenedor llamado `fusa-torch`

El entrenamiento producirá en la subcarpeta

- Un archivo con el modelo: `model.pt`
- Un archivo con el modelo en formato torch-script: `traced_model.pt`
- Un archivo json que adecua las etiquetas numéricas a los nombres de las clases: `index_to_name.json`
- Si se utilizó dvc: 
  - Un archivo `dvc.lock` (no modifique manualmente este archivo)
  - Una carpeta con las curvas de pérdida: `training_metrics`
  
## Evaluación

Para obtener un reporte de clasificación sobre un dataset con un modelo ya entrenado vaya a la subcarpeta correspondiente, indique el dataset que desea evaluar en `params.yaml` y ejecute

    python ../run_experiment.py --verbose --evaluate  --root_path ../../ --model_path model.pt
    
Para evaluar con dvc puede utilizar (desde la subcarpeta)
 
    dvc repro evaluate
    
Para evaluar con dvc en patagon (desde la subcarpeta)

    sbatch slurm_patagon.sh evaluate
    
Se generará en la carpeta

- Archivo con etiqueta predicha y real: `classification_table.csv`
- Archivo con métricas por clase: `classification_report.csv`

