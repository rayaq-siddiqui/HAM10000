from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from .ClassWeightMult import ClassWeightMult


def vgg16(class_weight, freeze_layers=10):
    model = VGG16(weights='imagenet',include_top=False)

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7,activation='softmax')(x)
    out = ClassWeightMult(class_weight)(x)

    model_final = Model(model.input, out)
    for layer in model.layers[0:freeze_layers]:
        layer.trainable = False

    print(model_final.summary())
    return model_final