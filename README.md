# Adaptadores para datasets de entrenamiento

- El archivo `FUSA_taxonomy.json` define las equivalencies entre las clases propias de los datasets y la taxonomía FUSA
- El archivo `datasets.py` tiene clases `torch.data.utils.Dataset` para los datasets de entrenamiento. También tiene una clase especial `FUSAv1` que es una concatenación de datasets

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