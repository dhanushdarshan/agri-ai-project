import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 📂 Dataset path (already confirmed from your screenshot)
train_dir = "/kaggle/input/datasets/emmarex/plantdisease/plantvillage/PlantVillage"

img_size = 224
batch_size = 32

# 📊 Load data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation'
)

print("✅ Data Loaded")

# 🤖 Model (NO internet needed)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights=None   # 🔥 FIXED (no download error)
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model Built")

# 🔥 Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    steps_per_epoch=200,
    validation_steps=50
)

# 💾 Save model
model.save("leaf_model.h5")

print("🎉 MODEL TRAINED & SAVED SUCCESSFULLY")
