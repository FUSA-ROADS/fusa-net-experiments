# FUSA: Datasets y rutinas para entrenar redes neuronales

El archivo `FUSA_taxonomy.json` define las equivalencies entre las clases propias de los datasets y la taxonomía FUSA

Los experimentos esperan que esté instalado el paquete [`fusanet_utils`](https://github.com/fusa-project/fusa-net-utils)

## Instalación librerías

Crear un entorno virtual con Python 3.8 y activarlo:
```
virtualenv fusa-training --python=python3.8
source fusa-training/bin/activate
```

Luego, instalar las librerías requeridas:
```
pip install -r requirements.txt
```

## DVC

### Instalación

https://dvc.org/doc/install/linux

### Configuración almacenamiento remoto dvc

Primero debemos configurar el usuario del almacenamiento remoto, en este caso se creará un entrada global y por defecto 
```
dvc remote add --default --global patagon ssh://patagon.uach.cl
dvc remote modify patagon --global port 2237
dvc remote modify patagon --global user SSH_USER
dvc remote modify patagon --global keyfile SSH_KEY
```
donde `SSH_USER` es el usuario con el cual ingresamos al servidor Patagon
y `SSH_KEY` es la ruta donde esta alojada la llave privada.

### Descargar dataset alojado en dvc

Con el almacenamiento remoto configurado, sólo debemos traer el dataset:
```
dvc pull
```
