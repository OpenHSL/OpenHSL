import torch
import numpy as np

from sklearn.model_selection import train_test_split

from openhsl.data.utils import pad_with_zeros


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.name = 'data_name'
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.ignored_labels = [0]
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def create_image_cubes(X,
                       y,
                       windowSize=5,
                       removeZeroLabels=True):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = pad_with_zeros(X, margin=margin)

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def split_train_test_set(X,
                         y,
                         testRatio,
                         randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def get_all_data_loader(X_pca,
                        y,
                        batch_size=16):
    patch_size = 13
    pca_components = X_pca.shape[-1]

    X_all, y_all = create_image_cubes(X_pca, y, windowSize=patch_size, removeZeroLabels=False)
    X_all = X_all.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_all = X_all.transpose(0, 4, 3, 1, 2)
    X_all = TestDS(X_all, y_all)
    all_data_loader = torch.utils.data.DataLoader(dataset=X_all,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)
    return all_data_loader


def create_data_loader(X_pca,
                       y,
                       train_ratio,
                       batch_size=16):
    test_ratio = 1 - train_ratio
    patch_size = 13
    pca_components = X_pca.shape[-1]

    print('\n... ... create data cubes ... ...')
    X, y_all = create_image_cubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = split_train_test_set(X, y_all, test_ratio, randomState=131)
    Xtrain, Xval, ytrain, yval = split_train_test_set(Xtrain, ytrain, 0.1, randomState=131)

    print('Xtrain shape: ', Xtrain.shape)
    print('Xval shape: ', Xval.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求

    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xval = Xval.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xval shape: ', Xval.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xval = Xval.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xval shape: ', Xval.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader

    trainset = TrainDS(Xtrain, ytrain)
    valset = TestDS(Xval, yval)
    testset = TestDS(Xtest, ytest)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=valset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader

