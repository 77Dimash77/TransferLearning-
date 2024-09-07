
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))


input_shape = (512, 512, 3)


def load_and_process_image(image_path):
    img = load_img(image_path, target_size=(512, 512))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img


def deprocess_image(img):
    img = img.reshape((512, 512, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Загрузка изображений контента и стиля
content_image_path = 'path_to_content_image.jpg'
style_image_path = 'path_to_style_image.jpg'
content_image = load_and_process_image(content_image_path)
style_image = load_and_process_image(style_image_path)


content_weight = 1e-4
style_weight = 1e-2


def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

num_style_layers = len(style_layers)


def get_model():
    vgg = VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_output = vgg.get_layer(content_layer).output
    style_outputs = [vgg.get_layer(layer).output for layer in style_layers]
    model_outputs = [content_output] + style_outputs
    return Model(vgg.input, model_outputs)


def get_content_and_style_outputs(model, content_path, style_path):
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    return content_outputs, style_outputs

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    content_weight, style_weight = loss_weights

    model_outputs = model(init_image)

    content_output = model_outputs[0]
    style_output = model_outputs[1:]

    content_loss_val = content_loss(content_output, content_features)

    style_loss_val = tf.add_n([style_loss(comb_style, target_gram)
                               for comb_style, target_gram in zip(style_output, gram_style_features)])
    style_loss_val *= style_weight / num_style_layers

    total_loss = content_loss_val * content_weight + style_loss_val
    return total_loss


init_image = np.copy(content_image)


init_image = tf.Variable(init_image, dtype=tf.float32)


opt = tf.optimizers.Adam(learning_rate=5.0)


model = get_model()
content_outputs, style_outputs = get_content_and_style_outputs(model, content_image_path, style_image_path)


style_features = style_outputs[1:]
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
content_features = content_outputs[0]


epochs = 1000
for i in range(epochs):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, (content_weight, style_weight), init_image, gram_style_features, content_features)
    grads = tape.gradient(loss, init_image)
    opt.apply_gradients([(grads, init_image)])
    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss.numpy()}")
        img = deprocess_image(init_image.numpy())
        plt.imshow(img)
        plt.show()


final_image = deprocess_image(init_image.numpy())
plt.imshow(final_image)
plt.show()
