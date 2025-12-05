import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Paths

data_dir = "dataset" 
aug_dir = "data_augmented"
os.makedirs(aug_dir, exist_ok=True)

# Augment healthy images

from torchvision import transforms

healthy_input = os.path.join(data_dir, "healthy")
healthy_output = os.path.join(aug_dir, "healthy")
os.makedirs(healthy_output, exist_ok=True)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0))
])

from PIL import Image
for img_name in os.listdir(healthy_input):
    img_path = os.path.join(healthy_input, img_name)
    img = Image.open(img_path).convert('RGB')
    base_name = os.path.splitext(img_name)[0]
    for i in range(10):  # 10x augmentation
        aug_img = transform(img)
        aug_img.save(os.path.join(healthy_output, f"{base_name}_aug{i}.jpg"))
    img.save(os.path.join(healthy_output, f"{base_name}_orig.jpg"))


# Copy mastitis images

mastitis_input = os.path.join(data_dir, "mastitis")
mastitis_output = os.path.join(aug_dir, "mastitis")
os.makedirs(mastitis_output, exist_ok=True)

for img_name in os.listdir(mastitis_input):
    shutil.copy(os.path.join(mastitis_input, img_name),
                os.path.join(mastitis_output, img_name))

# Image Generators

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    aug_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    aug_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Build Model

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# Save model
model.save("mastitis_model.keras")

# Plot metrics

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Acc')
plt.plot(epochs_range, val_acc, label='Validation Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()
