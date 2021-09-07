# FUSA: Datasets y rutinas para entrenar redes neuronales

El archivo `FUSA_taxonomy.json` define las equivalencies entre las clases propias de los datasets y la taxonomía FUSA

Los experimentos esperan que esté instalado el paquete [`fusanet_utils`](https://github.com/fusa-project/fusa-net-utils)


## DVC

### Instalación

https://dvc.org/doc/install/linux

### Configuración almacenamiento remoto dvc

Primero debemos configurar el usuario del almacenamiento remoto:
```
dvc remote modify patagon user SSH_USER --local
```
donde `SSH_USER` es el usuario con el cual ingresamos al servidor Patagon.

### Descargar dataset alojado en dvc

Con el almacenamiento remoto configurado, sólo debemos traer el dataset:
```
dvc pull
```