import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
from scipy import ndimage


__all__ = [
    "gamma变换",
    "非局部去噪",
    "超分辨",
    "SRCNN",
    'RGB转YCbCr'
]

def gamma变换(image, gamma=1.5):
    '''
    可以增强图像的对比度
    :param image:
    :param gamma:
    :return:
    '''
    # 将图像归一化到 [0, 1] 范围内
    normalized_image = image / 255.0
    # 进行 Gamma 变换
    corrected_image = np.power(normalized_image, gamma)
    # 缩放回 [0, 255] 范围
    corrected_image = (corrected_image * 255).astype(np.uint8)
    return corrected_image

def 非局部去噪(image):
    ### 感觉没啥用
    # 估计图像的噪声水平
    sigma_est = estimate_sigma(image, average_sigmas=True)

    # 非局部均值去噪
    denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)

    return denoised_image




class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

def RGB转YCbCr(img):
    '''
    将RGB编码变为YCbCr编码
    :param img:
    :return:
    '''
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('未知格式：', type(img))

def YCbCr转RGB(img):
    '''
    将YCbCr编码还原为RGB编码
    :param img:
    :return:
    '''
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('未知格式：', type(img))

def 使用SRCNN(image, 模型路径):
    ### 导入模型
    cudnn.benchmark = True  # 加速计算
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(模型路径), strict=True)
    model.eval()

    ### 处理图片
    image = pil_image.fromarray(image)
    image = np.array(image).astype(np.float32)
    ycbcr = RGB转YCbCr(image)
    ## 仅选取其中亮度维，导入网络
    y = ycbcr[:, :, 0]
    y /= 255.
    y = y[None, None, :, :]
    y = torch.from_numpy(y).to(device)
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)  # 将结果进行限制到[0,1]
    ## 还原图片
    preds = preds.mul(255.0).cpu().numpy()[0, 0, :, :]
    output = np.array([preds, ycbcr[:, :, 1], ycbcr[:, :, 2]]).transpose([1, 2, 0])  # 将亮度维叠加回去
    output = np.clip(YCbCr转RGB(output), 0.0, 255.0).astype(np.uint8)
    return output

def 超分辨(image, n, 模型路径):
    '''
    主要算法
    :param image: 图片，三维数组，最后一维为RGB
    :param n: 图片放大倍数
    :param 模型路径:
    :return:
    '''
    image = ndimage.zoom(image, (n, n, 1))  # 将图片的长和宽同时放大n倍
    image = 使用SRCNN(image, 模型路径)
    return image