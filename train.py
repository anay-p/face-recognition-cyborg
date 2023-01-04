import tensorflow as tf
# from keras_vggface import utils
from keras_vggface.vggface import VGGFace

# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train_generator = train_datagen.flow_from_directory(
#     './images',
#     target_size=(224, 224),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True
# )

train_generator = tf.keras.utils.image_dataset_from_directory(
    "./images",
    image_size=(224, 224),
    label_mode="categorical"
)

name_list = train_generator.class_names
no_of_classes = len(name_list)
print(name_list)
with open("list.txt", "w") as f:
    for i in range(no_of_classes-1):
        f.write(name_list[i] + ",")
    f.write(name_list[i+1])

base_model = VGGFace(include_top=False, model="senet50", input_shape=(224, 224, 3))

last_layer = base_model.get_layer("avg_pool").output
x = tf.keras.layers.Flatten(name="flatten")(last_layer)
out = tf.keras.layers.Dense(no_of_classes, activation="softmax", name="classifier")(x)
model = tf.keras.Model(base_model.input, out)

for layer in model.layers[:-2]:
    layer.trainable = False
for layer in model.layers[-2:]:
    layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, batch_size=1, verbose=0, epochs=20)
model.save("model.h5")
