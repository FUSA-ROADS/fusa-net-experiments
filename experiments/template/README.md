# Instrucciones para utilizar este template

Crea una carpeta dentro de `experiments`

Haz un link simbolico a dvc.yaml con

    cd nuevo_experimento
    ln -s ../template/dvc.yaml .

Luego copia `params.yaml` a `nuevo_experimento` y modifícalo a gusto. 

Nota: Si copias toda esta carpeta no olvides borrar el archivo `.dvcignore`
