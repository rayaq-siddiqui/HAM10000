# python3 main.py --model inception_v3 --epochs 3 --verbose 1
# python3 main.py --model efficient_net_v2s --epochs 3 --verbose 1
# python3 main.py --model efficient_net_b7 --epochs 3 --verbose 1


import argparse
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
    vgg16,
    efficient_net_v2s,
    efficient_net_b7
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

    if model_ == 'efficient_net_v2s' or model_ == 'efficient_net_b7':
        im_size = (32,32)
    elif model_ == 'inception_v3':
        im_size = (75,75)
    else:
        im_size = (256,256)

    train_df, val_df, org_df, class_weight = load_data(tar_size=im_size)

    if model_ == 'seq_model':
        model = seq_model.seq_model(class_weight)
    elif model_ == 'simple_seq_model':
        model = simple_seq_model.simple_seq_model(class_weight)
    elif model_ == 'inception_v3':
        model = inception_v3.inception_v3(class_weight, input_shape=(im_size+(3,)))
    elif model_ == 'resnet50':
        model = resnet50.resnet50(class_weight)
    elif model_ == 'vgg16':
        model = vgg16.vgg16(class_weight)
    elif model_ == 'efficient_net_v2s':
        model = efficient_net_v2s.efficient_net_v2s(class_weight, input_shape=(im_size+(3,)))
    elif model_ == 'efficient_net_b7':
        model = efficient_net_b7.efficient_net_b7(class_weight, input_shape=(im_size+(3,)))

    print('layers', len(model.layers))

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
