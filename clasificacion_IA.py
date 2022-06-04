import tensorflow as tf
import tensorflow_datasets as tfds

#obtiene los ejemplos para entrenamiento
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
#separamos los datos que servirán para entrenamiento y los datos que servirán para pruebas
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

nombres_clases = metadatos.features['label'].names

#normalizar los datos (pasar de 0-255 a 0-1)
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 #aquí lo pasa de 0-255 a 0-1
    return imagenes, etiquetas

#normalizar los datos de entrenamiento y pruebas con la función construida anteriormente
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)
#agregar los datos a cache (usar memoria en lugar de disco, agilizar el entrenamiento)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()


#crear Modelo
modelo = tf.keras.Sequential([
    tf.keras.Flatten(input_shape=(28,28,1)) #1 -blanco y negro
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #para redes de clasificación
])

#compilar el Modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']

)

num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

TAMANO_LOTE = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)
