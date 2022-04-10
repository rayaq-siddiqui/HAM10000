import argparse
from tabnanny import verbose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# user defined imports 
from load_data import load_data
from networks import (
    seq_model,
    simple_seq_model,
    inception_v3,
    resnet50,
    vgg16
)

# print validation statement
print("all resources loaded")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the inputs')
    parser.add_argument(
        '--model', 
        type=str, 
        help='which model would you like to run',
        default='inception_v3'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='how many epochs',
        default=3
    )
    parser.add_argument(
        '--verbose', 
        type=int, 
        help='0,1,2',
        default=1
    )

    args = parser.parse_args()
    model_ = args.model
    epochs_ = args.epochs
    verbose_ = args.verbose

    train_df, val_df, class_weight = load_data()

    if model_ == 'seq_model':
        model = seq_model.seq_model(class_weight)
    elif model_ == 'simple_seq_model':
        model = simple_seq_model.simple_seq_model(class_weight)
    elif model_ == 'inception_v3':
        model = inception_v3.inception_v3(class_weight)
    elif model_ == 'resnet50':
        model = resnet50.resnet50(class_weight)
    elif model_ == 'vgg16':
        model = vgg16.vgg16(class_weight)

    # compiling the model
    optimizer = Adam(learning_rate=0.001)
    loss = SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    # callbacks
    lr_reduction = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=4, 
        verbose=verbose_,
        mode='auto',
        min_delta=0.0001, 
        cooldown=0, 
        min_lr=0.0000001
    )
    checkpoint = ModelCheckpoint(
        "model/ham10000_main_model.h5", 
        save_best_only=True,
        verbose=verbose_
    )

    callbacks = [
        lr_reduction,
        checkpoint
    ]

    # fitting the model
    model.fit(
        train_df,
        validation_data=val_df,
        epochs=epochs_,
        verbose=verbose_,
        callbacks=callbacks
    )
