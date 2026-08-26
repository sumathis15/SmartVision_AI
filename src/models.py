"""Keras transfer-learning builders matching the brief (VGG16, ResNet50, MobileNetV2, EfficientNetB0)."""

from __future__ import annotations

from src.config import IMAGE_SIZE, NUM_CLASSES


def _input_shape():
    return (IMAGE_SIZE, IMAGE_SIZE, 3)


def build_vgg16(learning_rate: float = 1e-4):
    """Freeze convolutional base, custom dense head with dropout."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, optimizers

    base = applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=_input_shape(),
    )
    base.trainable = False

    inputs = layers.Input(shape=_input_shape())
    x = applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="VGG16")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    return model


def build_resnet50(learning_rate: float = 1e-4, unfreeze_last: int = 20):
    """Fine-tune by unfreezing the last `unfreeze_last` layers; GAP + custom head."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, optimizers

    base = applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=_input_shape(),
    )
    for layer in base.layers:
        layer.trainable = False
    if unfreeze_last > 0:
        for layer in base.layers[-unfreeze_last:]:
            layer.trainable = True

    inputs = layers.Input(shape=_input_shape())
    x = applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="ResNet50")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    return model


def build_mobilenetv2(learning_rate: float = 1e-3):
    """Frozen base, compact head, optimized for inference speed."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, optimizers

    base = applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=_input_shape(),
    )
    base.trainable = False

    inputs = layers.Input(shape=_input_shape())
    x = applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="MobileNetV2")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    return model


def build_efficientnetb0(learning_rate: float = 1e-4):
    """Classification head with batch norm; caller enables mixed precision."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, optimizers

    base = applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=_input_shape(),
    )
    base.trainable = False

    inputs = layers.Input(shape=_input_shape())
    x = applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="EfficientNetB0")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    return model


def unfreeze_efficientnet(model, n_blocks: int = 20, learning_rate: float = 1e-5):
    """Stage-2 fine-tune: unfreeze the last n layers of the EfficientNet backbone."""
    import tensorflow as tf
    from tensorflow.keras import optimizers

    base = None
    for layer in model.layers:
        if layer.name.startswith("efficientnet"):
            base = layer
            break
    if base is None:
        return model
    base.trainable = True
    for layer in base.layers[:-n_blocks]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    return model


BUILDERS = {
    "VGG16": build_vgg16,
    "ResNet50": build_resnet50,
    "MobileNetV2": build_mobilenetv2,
    "EfficientNetB0": build_efficientnetb0,
}
