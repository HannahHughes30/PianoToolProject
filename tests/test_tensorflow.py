import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("Metal plugin loaded:", tf.test.is_built_with_cuda())

# Simple test
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)
print("Matrix multiplication result:\n", c.numpy())
