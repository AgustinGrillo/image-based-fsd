from keras import Input, layers, models
import tensorflow as tf


def build_model_v3(post_pro_screen_size):
    inputs = Input(shape=post_pro_screen_size)
    x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    # Model compile
    model.compile(
        loss='mse',
        optimizer='adam',  # 'adam' or 'SGD'
        metrics=['mse']
    )
    return model


def build_model_v5(post_pro_screen_size):
    inputs = Input(shape=post_pro_screen_size)
    # x = layers.Lambda(lambda x: (x / 255.0))(inputs)  DOESNT WORK !
    x = layers.Conv2D(filters=4, kernel_size=(5, 5), activation="relu", strides=(2, 2))(inputs)
    x = layers.Conv2D(filters=8, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=10, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    # x = layers.SpatialDropout2D(rate=0.5)(x)
    x = layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    # Model compile
    model.compile(
        loss='mse',
        optimizer='adam'  # 'adam' or 'SGD'
    )
    return model


################################## MODELS WITH PREPROCCESING INSIDE ##################################

def build_model_v7(screen_size):
    inputs = Input(shape=screen_size)
    x = layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=screen_size)(
        inputs)
    # x = layers.Lambda(lambda m: m[:, :, :, 2:3])(x)  # BLUE: m[:, :, :, 0:1]  GREEN: m[:, :, :, 1:2] RED: m[:, :, :, 2:3]
    # x = layers.Lambda(lambda m: (m / 255.0))(x)
    x = layers.Conv2D(filters=4, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=8, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=10, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    # x = layers.SpatialDropout2D(rate=0.5)(x)
    x = layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    return model


def build_model_v9(screen_size):
    inputs = Input(shape=screen_size)
    x = layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=screen_size)(
        inputs)
    # x = layers.Lambda(lambda m: m[:, :, :, 2:3])(x)  # BLUE: m[:, :, :, 0:1]  GREEN: m[:, :, :, 1:2] RED: m[:, :, :, 2:3]
    x = layers.Lambda(lambda m: (m / 255.0))(x)
    x = layers.Conv2D(filters=24, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=36, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=48, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    # x = layers.SpatialDropout2D(rate=0.5)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    return model


def build_model_v11(screen_size):
    inputs = Input(shape=screen_size)
    x = layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=screen_size)(
        inputs)
    # x = layers.Lambda(lambda m: m[:, :, :, 2:3])(x)  # BLUE: m[:, :, :, 0:1]  GREEN: m[:, :, :, 1:2] RED: m[:, :, :, 2:3]
    x = layers.Lambda(lambda m: (m / 255.0))(x)  # DOESNT WORK !
    x = layers.Conv2D(filters=24, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=36, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=48, kernel_size=(5, 5), activation="relu", strides=(2, 2))(x)
    # x = layers.SpatialDropout2D(rate=0.5)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(10, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    return model


def build_model_v13(screen_size):
    inputs = Input(shape=screen_size)
    x = layers.Cropping2D(cropping=((185, 125), (0, 0)), input_shape=screen_size)(
        inputs)
    # x = layers.Lambda(lambda m: m[:, :, :, 2:3])(x)  # BLUE: m[:, :, :, 0:1]  GREEN: m[:, :, :, 1:2] RED: m[:, :, :, 2:3]
    x = layers.Lambda(lambda m: (m / 255.0))(x)  # DOESNT WORK !
    x = layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=18, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=24, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)

    # x = layers.SpatialDropout2D(rate=0.5)(x)
    x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(10, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    return model


def build_model_v15(screen_size):
    inputs = Input(shape=screen_size)
    x = layers.Cropping2D(cropping=((182, 107), (0, 0)), input_shape=screen_size)(
        inputs)
    # x = layers.Lambda(lambda m: m[:, :, :, 2:3])(x)  # BLUE: m[:, :, :, 0:1]  GREEN: m[:, :, :, 1:2] RED: m[:, :, :, 2:3]
    x = layers.Lambda(lambda m: (m / 255.0))(x)  # DOESNT WORK !
    x = layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=18, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=24, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.Conv2D(filters=42, kernel_size=(3, 3), activation="relu", strides=(2, 2))(x)

    # x = layers.SpatialDropout2D(rate=0.5)(x)
    x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)
    x = layers.Conv2D(filters=48, kernel_size=(3, 3), activation="relu", strides=(1, 1))(x)

    # Flatten layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(10, activation="relu")(x)
    # x = layers.Dropout(rate=0.5)(x)

    # Output layer
    outputs = layers.Dense(2, activation='linear')(x)
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="driverless_model")

    return model
