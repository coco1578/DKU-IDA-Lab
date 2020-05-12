import numpy as np

from metrics import *
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD
from scipy.sparse import coo_matrix
from neural_network_architecture import *
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


def preprocess_data(dataset, resize_shape=None):

    X, y = load_svmlight_file(dataset)
    X = np.array(coo_matrix(X, dtype=np.float).todense())

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    # smote = SMOTE()
    # X, y = smote.fit_resample(X, y)

    if resize_shape is not None:
        if len(resize_shape) == 2:
            X = X.reshape(X.shape[0], resize_shape[0], resize_shape[1])
        else:
            X = X.reshape(X.shape[0], resize_shape[0], resize_shape[1], resize_shape[2])

    le = LabelEncoder()
    le.fit(y)
    le_y = le.transform(y)
    number_of_classes = len(set(y))


    return X, y, number_of_classes


if __name__ == '__main__':

    #dataset_path = '../dataset/HIDDEN_10_ShuaiLi_IDA_new_2.spa'
    dataset_path = '../dataset/TBB_10_ShuaiLi_IDA_new_2.spa'
    CV = 5
    EPOCH = 200
    BATCH_SIZE = 32
    CHANNEL_ORDER = 'channels_first'

    result_accuracy_train = list()
    result_accuracy_test = list()
    result_f1_train = list()
    result_f1_test = list()
    result_precision_train = list()
    result_precision_test = list()
    result_recall_train = list()
    result_recall_test = list()
    # result_balanced_error_train = list()
    # result_balanced_error_test = list()


    X, y, number_of_classes = preprocess_data(dataset_path, resize_shape=(1, 3184))

    kf = StratifiedKFold(n_splits=CV, shuffle=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    mcp = ModelCheckpoint('/Users/coco/Desktop/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

    for i, (idx_train, idx_test) in enumerate(kf.split(X, y)):

        X_train = X[idx_train]
        X_test= X[idx_test]
        y_train = y[idx_train]
        y_test= y[idx_test]
        y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

        model = OneJHModel.get_model(input_shape=(1, 3184), number_of_classes=number_of_classes, channel_order=CHANNEL_ORDER)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2)

        scores = model.evaluate(X_train, y_train, verbose=2)
        result_accuracy_train.append(scores[1])
        result_f1_train.append(scores[2])
        result_precision_train.append(scores[3])
        result_recall_train.append(scores[4])

        scores = model.evaluate(X_test, y_test, verbose=2)
        result_accuracy_test.append(scores[1])
        result_f1_test.append(scores[2])
        result_precision_test.append(scores[3])
        result_recall_test.append(scores[4])

    print('------------------------   INFORMATION   ------------------------')
    print('Train accuracy:              {0}'.format(result_accuracy_train))
    print('Test accuracy:               {0}'.format(result_accuracy_test))
    print('Mean train accuracy:         {0}'.format(np.mean(result_accuracy_train)))
    print('Mean test accuracy:          {0}'.format(np.mean(result_accuracy_test)))
    print('SD train accuracy:           {0}'.format(np.std(result_accuracy_train)))
    print('SD test accuracy:            {0}'.format(np.std(result_accuracy_test)))
    print('Train f1 score:              {0}'.format(result_f1_train))
    print('Test f1 score:               {0}'.format(result_f1_test))
    print('Mean train f1 score:         {0}'.format(np.mean(result_f1_train)))
    print('Mean test f1 score:          {0}'.format(np.mean(result_f1_test)))
    print('Train precision:             {0}'.format(result_precision_train))
    print('Test precision:              {0}'.format(result_precision_test))
    print('Mean train precision:        {0}'.format(np.mean(result_precision_train)))
    print('Mean test precision:         {0}'.format(np.mean(result_precision_test)))
    print('Train recall:                {0}'.format(result_recall_train))
    print('Test recall:                 {0}'.format(result_recall_test))
    print('Mean train recall:           {0}'.format(np.mean(result_recall_train)))
    print('Mean test recall:            {0}'.format(np.mean(result_recall_test)))
    # print('Train balanced error:        {0}'.format(result_balanced_error_train))
    # print('Test balanced error:         {0}'.format(result_balanced_error_test))
    # print('Mean train balanced error:   {0}'.format(np.mean(result_balanced_error_train)))
    # print('Mean test balanced error:    {0}'.format(np.mean(result_balanced_error_test)))
    print('-----------------------------------------------------------------')