# Article-Data
Data usada para el articulo publicado en el Número especial en IA e Industria 4.0 en IJCOPI
# Requisitos
- Python 3.0 +
- numpy
- pandas
- matplotlib
- plotly
- mlxtend
> Las librerias se instalan con pip usando el comando *pip install -r REQUIREMENTS.txt*
# Ejecución / Uso
La ejecución del prrograma es simplemente correr el fichero de Python ya que cada archivo tiene seleccionado el conjunto de datos que va a utilizar.

Durante la ejecución del programa tendrá varios *prints* en la consola, los cuales son:
- Un listado con los atributos del archivo (los cuales son declarados adentro del fichero) con la cantidad de datos faltantes.
- Un resumen del atributo actual y la media para las personas diabeticas (1) y las no-diabeticas (0). ***Esto se va a repetir por la cantidad de atributos del conjunto de datos***
- Un listado con los atributos del archivo (los cuales son declarados adentro del fichero) con la cantidad de datos faltantes.
- Una representación gráfica de la correlación de los atributos pertenecientes al conjunto de datos. ***Este queda en formato html llamado ```temp-plot.html```*** 
- Finalmente un Enter final para escribir un archivo ```out.csv``` que crea el fichero con los datos pre-procesados.
> para continuar simplemente hay que dar enter.
