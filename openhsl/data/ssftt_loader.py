import torch
import numpy as np


from openhsl.data.utils import pad_with_zeros, split_train_test_set, create_patches


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


def get_all_data_loader(X_pca,
                        y,
                        batch_size=16):
    patch_size = 13
    pca_components = X_pca.shape[-1]

    X_all, y_all = create_patches(X_pca, y, patch_size=patch_size, remove_zero_labels=False)
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
                       train_sample_percentage,
                       patch_size=13,
                       batch_size=16):

    pca_components = X_pca.shape[-1]

    X, y_all = create_patches(X_pca, y, patch_size=patch_size)

    test_ratio = 1.0 - train_sample_percentage

    Xtrain, Xtest, ytrain, ytest = split_train_test_set(X, y_all, test_ratio)
    Xtrain, Xval, ytrain, yval = split_train_test_set(Xtrain, ytrain, 0.1)

    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xval = Xval.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)

    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xval = Xval.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)

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

