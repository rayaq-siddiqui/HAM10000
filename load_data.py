import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data():
    def numerizing(x):
        dict = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df': 6
        }
        return dict[x]


    df = pd.read_csv('data/HAM10000_metadata.csv')
    df['image_id'] = df['image_id'].apply(lambda x: '{}.jpg'.format(x))
    df['image_id'] = df['image_id'].apply(lambda x: 'data/HAM10000_images/{}'.format(x))
    df['dx_num'] = df['dx'].apply(lambda x: numerizing(x))

    class_labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    n_classes = 7

    w = {}
    vc = df['dx'].value_counts().items()
    w_sum = 0
    for lab, val in vc:
        w[str(lab)] = val
        w_sum += val
    class_weight = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    ind = 0
    for i in w:
        class_weight[ind] = w_sum / (n_classes * w[i])
        ind += 1

    cw = [val for lab, val in class_weight.items()]
    df_shuffled = shuffle(df)

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 20,
        validation_split=0.2
    )

    train_df = datagen.flow_from_dataframe(
        df_shuffled,
        directory = '',
        x_col = 'image_id',
        y_col = 'dx_num',
        subset='training',
        class_mode='raw',
        target_size=(256, 256),
    )
    val_df = datagen.flow_from_dataframe(
        df_shuffled,
        directory = '',
        x_col = 'image_id',
        y_col = 'dx_num',
        subset='validation',
        class_mode='raw',
        target_size=(256, 256),
    )

    return train_df, val_df, cw
    
