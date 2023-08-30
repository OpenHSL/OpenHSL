import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data as Data

import numpy as np
from einops import rearrange, repeat
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm

from openhsl.data.utils import sample_gt
from openhsl.models.model import Model
from openhsl.data.dataset import get_dataset
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask


def choose_train_and_test_point(train_data: np.ndarray,
                                test_data: np.ndarray,
                                true_data: np.ndarray,
                                num_classes: int):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------

    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
# -------------------------------------------------------------------------------


def mirror_hsi(height,
               width,
               band,
               input_normalize,
               patch=5):
    """
    Mirror padding for HSI

        Parameters
        ----------
        height
        width
        band
        input_normalize
        patch

        Returns
        -------

    """
    padding = patch // 2
    mirrored_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)

    mirrored_hsi[padding: (padding + height), padding: (padding + width), :] = input_normalize

    for i in range(padding):
        mirrored_hsi[padding: (height + padding), i, :] = input_normalize[:, padding - i - 1, :]

    for i in range(padding):
        mirrored_hsi[padding: (height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]

    for i in range(padding):
        mirrored_hsi[i, :, :] = mirrored_hsi[padding * 2 - i - 1, :, :]

    for i in range(padding):
        mirrored_hsi[height + padding + i, :, :] = mirrored_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirrored_hsi.shape[0],
                                                      mirrored_hsi.shape[1],
                                                      mirrored_hsi.shape[2]))
    print("**************************************************")

    return mirrored_hsi
# ----------------------------------------------------------------------------------------------------------------------


# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x: (x + patch), y: (y + patch), :]
    return temp_image
# ----------------------------------------------------------------------------------------------------------------------


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    # 中心区域
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch: (i + 1) * patch * patch, : i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch: (i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i: (i + 1), :(nn - i)] = x_train_reshape[:, 0: 1, (band - nn + i):]
            x_train_band[:, i: (i + 1), (nn - i):] = x_train_reshape[:, 0: 1,:(band - nn + i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch: (nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch: (nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, : i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band
# ----------------------------------------------------------------------------------------------------------------------


# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape, x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
# ----------------------------------------------------------------------------------------------------------------------


# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes +1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
# -------------------------------------------------------------------------------


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_head,
                 dropout,
                 num_channel,
                 mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, (1, 2), 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x


class ViT(nn.Module):
    def __init__(self,
                 image_size,
                 near_band,
                 num_patches,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=1,
                 dim_head=16,
                 dropout=0.,
                 emb_dropout=0.,
                 mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        return self.mlp_head(x)


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
# -------------------------------------------------------------------------------


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res, target, pred.squeeze()


def train_epoch(model, train_loader, criterion, optimizer):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()

            optimizer.zero_grad()
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            loss.backward()
            optimizer.step()

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())
        return top1.avg, objs.avg, tar, pre

# -------------------------------------------------------------------------------


def valid_epoch(model, valid_loader, criterion, optimizer):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()

            batch_pred = model(batch_data)

            loss = criterion(batch_pred, batch_target)

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

        return tar, pre


def test_epoch(model, test_loader):
        pre = np.array([])
        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(test_loader)):
            batch_data = batch_data.cuda()

            batch_pred = model(batch_data)

            _, pred = batch_pred.topk(1, 1, True, True)
            pp = pred.squeeze()
            pre = np.append(pre, pp.data.cpu().numpy())
        return pre


class SpectralFormer(Model):

    def __init__(self, n_classes, n_bands, path_to_weights=None, **kwargs):
        self.model = ViT(image_size=kwargs["patches"],
                         near_band=kwargs["band_patches"],
                         num_patches=n_bands,
                         num_classes=n_classes,
                         dim=27,
                         depth=5,
                         heads=4,
                         mlp_dim=8,
                         dropout=0.1,
                         emb_dropout=0.1,
                         mode=kwargs["mode"])

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

        self.model = self.model.cuda()

    def fit(self,
            X: HSImage,
            y: HSMask,
            fit_params):

        X, y = get_dataset(X, y)

        fit_params.setdefault('optimizer_params', {'learning_rate': 0.01, 'weight_decay': 0})
        fit_params.setdefault('optimizer',
                              torch.optim.Adam(self.model.parameters(),
                                               lr=fit_params['optimizer_params']["learning_rate"],
                                               weight_decay=fit_params['optimizer_params']['weight_decay']))
        fit_params.setdefault('loss', nn.CrossEntropyLoss())

        data = X
        # train, test and concat data of labels
        label = y
        TR, TE = sample_gt(label, 0.4, mode='fixed')

        patches = 1
        band_patches = 1
        batch_size = 256

        num_classes = len(np.unique(label))
        height, width, band = data.shape
        input_normalize = np.zeros(data.shape)

        #for i in range(data.shape[2]):
        #    input_max = np.max(data[:, :, i])
        #    input_min = np.min(data[:, :, i])
        #    input_normalize[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)

        # obtain train and test data
        total_pos_train, total_pos_test, total_pos_true, \
        number_train, number_test, number_true = choose_train_and_test_point(TR,
                                                                             TE,
                                                                             label,
                                                                             num_classes)

        # add padding and mirroring bounds of HSI in these pads
        mirror_image = mirror_hsi(height,
                                  width,
                                  band,
                                  input_normalize,
                                  patch=patches)

        x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image,
                                                                     band,
                                                                     total_pos_train,
                                                                     total_pos_test,
                                                                     total_pos_true,
                                                                     patch=patches,
                                                                     band_patch=band_patches)

        y_train, y_test, y_true = train_and_test_label(number_train,
                                                       number_test,
                                                       number_true,
                                                       num_classes)
        # -------------------------------------------------------------------------------
        # load data
        x_train = torch.from_numpy(x_train_band.transpose((0, 2, 1))).type(torch.FloatTensor)  # [695, 200, 7, 7]
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
        Label_train = Data.TensorDataset(x_train, y_train)
        x_test = torch.from_numpy(x_test_band.transpose((0, 2, 1))).type(torch.FloatTensor)  # [9671, 200, 7, 7]
        y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
        Label_test = Data.TensorDataset(x_test, y_test)
        x_true = torch.from_numpy(x_true_band.transpose((0, 2, 1))).type(torch.FloatTensor)
        y_true = torch.from_numpy(y_true).type(torch.LongTensor)
        Label_true = Data.TensorDataset(x_true, y_true)

        label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)
        label_test_loader = Data.DataLoader(Label_test, batch_size=batch_size, shuffle=True)
        label_true_loader = Data.DataLoader(Label_true, batch_size=200, shuffle=False)

        for epoch in trange(fit_params['epochs']):

            self.model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model=self.model,
                                                             train_loader=label_train_loader,
                                                             criterion=fit_params['loss'],
                                                             optimizer=fit_params['optimizer'])
            print(f"Epoch: {epoch + 1} train_loss: {train_obj} train_acc: {train_acc}")

            self.model.eval()
            tar_v, pre_v = valid_epoch(model=self.model,
                                       valid_loader=label_test_loader,
                                       criterion=fit_params['loss'],
                                       optimizer=fit_params['optimizer'])
        torch.save(self.model.state_dict(), 'spectral_former' + ".pth")

    def predict(self,
                X: HSImage,
                y: HSMask = None):

        data = X.data
        # train, test and concat data of labels
        label = y.data
        TR, TE = sample_gt(label, 0.4, mode='fixed')

        patches = 1
        band_patches = 1
        batch_size = 256

        num_classes = len(np.unique(label))
        height, width, band = data.shape
        input_normalize = np.zeros(data.shape)

        height, width, band  = X.data.shape

        total_pos_train, total_pos_test, total_pos_true, \
        number_train, number_test, number_true = choose_train_and_test_point(TR,
                                                                             TE,
                                                                             label,
                                                                             num_classes)

        # add padding and mirroring bounds of HSI in these pads
        mirror_image = mirror_hsi(height,
                                  width,
                                  band,
                                  input_normalize,
                                  patch=patches)

        x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image,
                                                                     band,
                                                                     total_pos_train,
                                                                     total_pos_test,
                                                                     total_pos_true,
                                                                     patch=patches,
                                                                     band_patch=band_patches)

        y_train, y_test, y_true = train_and_test_label(number_train,
                                                       number_test,
                                                       number_true,
                                                       num_classes)

        x_true = torch.from_numpy(x_true_band.transpose((0, 2, 1))).type(torch.FloatTensor)
        y_true = torch.from_numpy(y_true).type(torch.LongTensor)
        Label_true = Data.TensorDataset(x_true, y_true)
        label_true_loader = Data.DataLoader(Label_true, batch_size=300, shuffle=False)

        self.model.eval()
        # output classification maps
        pre_u = test_epoch(self.model, label_true_loader)

        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

        return prediction_matrix


