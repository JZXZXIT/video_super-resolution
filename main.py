import cv2
from MyClasses import *
import os
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cpuinfo
import numpy as np
import PIL.Image as pil_image

def 视频超分辨(放大倍数, 模型路径, 视频路径, 保存路径):
    '''
    处理视频的主函数，可以直接调用使用
    :param 放大倍数: 长和宽均会被放大n倍
    :param 模型路径: 路径+名称，模型只需要保存参数
    :param 视频路径: 视频路径+视频名称
    :param 保存路径: 保存路径+名称
    :return:
    '''

    ### 导入视频
    cap = cv2.VideoCapture(视频路径)
    ## 检查视频文件是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()

    ### 获取基本信息
    ## 获取原始视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### 创建处理后的视频
    output = cv2.VideoWriter(保存路径, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 放大倍数, height * 放大倍数))  # 创建 VideoWriter 对象，用于写入新的视频文件

    ### 循环读取视频帧
    i = 1
    while True:
        ## 输出进度
        print('\r', end="")
        方块数 = int(i * 50 / total_frames)
        方块 = "■" * 方块数
        空格 = " " * (50 - 方块数)
        print(f"正在处理 [{方块}{空格}] {i}/{total_frames}帧", end="")
        ## 读取一帧视频数据
        ret, frame = cap.read()
        ## 如果视频读取完毕，退出循环
        if not ret:
            break
        ## 处理该帧图片
        frame = frame[..., ::-1]  # 将 BGR 转换为 RGB
        frame = 超分辨(frame, 放大倍数, 模型路径)
        frame = gamma变换(frame)
        frame = 非局部去噪(frame)
        frame = frame[..., ::-1]  # 将 RGB 转换回 BGR
        ## 将当前帧写入 VideoWriter 对象中
        output.write(frame)

        i += 1

def 将图片变为网络需要的数据(图片路径:str=None, 图片数组:np.ndarray=None):
    '''
    调用这个函数制作数据集
    保证路径和数组仅有一个位非None
    :param 图片路径:
    :param 图片数组: 保证格式为RGB！！千万不要是RGBN或者BGR等其他格式
    :return:
    '''
    ### 导入图片
    if 图片路径 is not None:
        image = pil_image.open(图片路径).convert('RGB')
    elif 图片数组 is not None:
        pil_image.fromarray(图片数组)
    image = np.array(image).astype(np.float32)

    ### 将RGB转化为YCbCr
    ycbcr = RGB转YCbCr(image)

    ### 仅选取其中亮度维
    y = ycbcr[:, :, 0]
    y /= 255.
    return y

def 训练SRCNN(data_in, data_out, learning_rate, num_epochs, batch_size, str_device, 保存位置, 保存轮次=None):
    '''
    里边集成了一些臃肿的功能，看不懂的话直接调用即可（若正确率部分报错很正常，删除即可，重点关注损失值，正确率不是很准）
    :param data_in: 模型输入，即超分辨前，类型为张量
    :param data_out: 模型输出，即超分辨后，类型为张量
    :param learning_rate: 初始学习率
    :param num_epochs: 训练次数
    :param batch_size: 批大小
    :param str_device: 训练位置，输出‘CPU’ or ‘GPU’
    :param 保存位置: 可以是绝对位置或相对位置
    :param 保存轮次: 支持每n轮保存一次，若不需要，则不要传递参数
    :return:
    '''

    def _保存模型(当前时间, 模型参数, 保存位置, input_size, num_epochs, model, learning_rate, 训练信息):
        ### 创建保存路径
        if not os.path.exists(f"./{保存位置}"):
            os.makedirs(f"./{保存位置}")

        ### 保存模型
        torch.save(model.state_dict(), f'./{保存位置}/{当前时间.strftime("%m_%d_%H_%M_%S")}_model.pth')

        ### 保存介绍文件
        with open(f"./{保存位置}/{当前时间.strftime('%m_%d_%H_%M_%S')}_Introduce.txt", "w") as 介绍文件:
            介绍文件.write(f"{模型参数}\n\n"
                       f"文件名称：{当前时间.strftime('%m_%d_%H_%M_%S')}_model.pth\n文件保存时间：{当前时间}\n\n"
                       f"训练轮次：{num_epochs}\n初始学习率：{learning_rate}\ndata shape：{(-1,) + input_size}\n\n"
                       f"训练结果：\n")
            for i in range(len(训练信息)):
                介绍文件.write(f"{训练信息[i]}\n")

    def MYsummary(model, input_size, batch_size=-1, device="cuda"):
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        from collections import OrderedDict
        import numpy as np

        output = ""

        def register_hook(module):
            def hook(module, input, output):
                def 获取输出shape(输入, 批大小=batch_size):
                    if isinstance(输入, (list, tuple)):
                        # 如果是列表或元组，则遍历每个输出张量
                        # 并将每个张量的形状转换为列表形式，添加到 output_shape 列表中
                        # 初始化 output_shape 列表
                        输出 = []
                        # 遍历每个输出张量 o
                        for io in 输入:
                            # 将修改后的形状列表添加到 output_shape 列表中
                            输出.append(获取输出shape(io, -1))
                    else:
                        # 如果输出不是列表或元组，则将输出张量的形状转换为列表形式
                        输出 = list(输入.size())
                        # 修改 output_shape 列表中的第一个元素为 batch_size
                        输出[0] = 批大小
                    return 输出

                # 将 module 的类名转换成字符串，并提取最后一个部分作为类名
                class_name = str(module.__class__).split(".")[-1].split("'")[0]

                # 获取当前 module 在 summary 中的索引位置，用于构建唯一的键值
                module_idx = len(summary)

                # 生成唯一的键值
                m_key = "%s-%i" % (class_name, module_idx + 1)

                # 创建一个 空的有序字典 对象来保存 module 的信息
                summary[m_key] = OrderedDict()

                # 获取输入张量的形状，并将其转换为列表形式
                summary[m_key]["input_shape"] = list(input[0].size())

                # 修改 input_shape 列表中的第一个元素为 batch_size
                summary[m_key]["input_shape"][0] = batch_size

                # 检查输出是否为列表或元组类型
                summary[m_key]["output_shape"] = 获取输出shape(output)

                # 初始化参数计数变量 params
                params = 0

                # 检查 module 是否具有 weight 属性，并且 weight 对象具有 size 方法
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    # 计算 weight 张量的元素数量，并添加到 params 中
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    # 将 trainable 属性设置为 module.weight 是否需要梯度更新
                    summary[m_key]["trainable"] = module.weight.requires_grad

                # 检查 module 是否具有 bias 属性，并且 bias 对象具有 size 方法
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    # 计算 bias 张量的元素数量，并添加到 params 中
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))

                # 将参数计数值保存到 nb_params 中
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        def 计算total_output(输入):
            输出 = 0
            if isinstance(输入[0], list):
                for part in 输入:
                    输出 += abs(计算total_output(part))
            else:
                输出 += abs(np.prod(输入))
            return abs(输出)

        device = device.lower()
        if device == 'gpu':
            device = 'cuda'
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        ### 创建一个input_size的张量
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        # x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        x = []
        for in_size in input_size:
            size = (2, *in_size)
            data = torch.rand(size).type(dtype)
            x.append(data)
        # print(type(x[0]))

        ###
        # 创建属性
        summary = OrderedDict()  # 创建了一个空的有序字典，会保持元素顺序
        hooks = []

        # 寄存器挂钩
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        分割线1 = "-------------------------------------------------------------------------"
        分割线2 = "========================================================================="
        print(分割线1)
        output += 分割线1 + '\n'
        line_new = "{:>18}  {:>27} {:>15}".format("层类型", "输出形状", "参数数量")
        print(line_new)
        output += line_new + "\n"
        print(分割线2)
        output += 分割线2 + '\n'
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:  # 遍历其中每一个键
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>30} {:>18}".format(
                layer,  # 键名称
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            print(line_new)
            output += line_new + '\n'
            total_params += summary[layer]["nb_params"]
            total_output += 计算total_output(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print(分割线2)
        output += 分割线2 + "\n"
        print("总参数数量: {0:,}".format(total_params))
        output += "总参数数量: {0:,}".format(total_params) + '\n'
        print("可训练参数数量: {0:,}".format(trainable_params))
        output += "可训练参数数量: {0:,}".format(trainable_params) + '\n'
        print("不可训练的参数数量: {0:,}".format(total_params - trainable_params))
        output += "不可训练的参数数量: {0:,}".format(total_params - trainable_params) + '\n'
        print(分割线1)
        output += 分割线1 + '\n'
        print("输入数据大小 (MB): %0.2f" % total_input_size)
        output += "输入数据大小 (MB): %0.2f" % total_input_size + '\n'
        print("前向/反向传播所需内存大小 (MB): %0.2f" % total_output_size)
        output += "前向/反向传播所需内存大小 (MB): %0.2f" % total_output_size + '\n'
        print("参数大小 (MB): %0.2f" % total_params_size)
        output += "参数大小 (MB): %0.2f" % total_params_size + '\n'
        print("模型总大小 (MB): %0.2f" % total_size)
        output += "模型总大小 (MB): %0.2f" % total_size + '\n'
        print(分割线1)
        output += 分割线1 + '\n'

        return output

    ### 获取当前时间
    当前时间 = datetime.now()

    ### 获取训练位置
    if str_device.lower() == "cpu":
        device = torch.device('cpu')
        print(f"程序运行位置：{cpuinfo.get_cpu_info()['brand_raw']}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"程序运行位置：{torch.cuda.get_device_name(device)}")

    ### 处理输入数据
    input_size = tuple(data_in[0].shape)
    data_in, data_out = data_in.to(device), data_out.to(device)

    ### 创建模型
    model = SRCNN()
    model = model.to(device)
    model.train()
    模型参数 = MYsummary(model=model, input_size=input_size, batch_size=batch_size, device=str_device)

    ### 创建数据集和数据加载器
    train_dataset = TensorDataset(data_in, data_out)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ### 创建损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
    ], lr=learning_rate)

    ### 训练模型
    训练信息 = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        for inputs_batch, targets_batch in train_dataloader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            # 前向传播与计算损失
            outputs = model(inputs_batch).clamp(0.0, 1.0)
            loss = criterion(outputs, targets_batch)

            # 反向传播与优化
            optimizer.zero_grad()  # 梯度消零
            loss.backward()
            optimizer.step()  # 更新模型参数

            # 计算损失与正确率
            total_loss += loss.item()
            正确个数 = (outputs == targets_batch).sum().item()
            total_correct += 正确个数 / (np.prod(input_size) * batch_size)

        # 输出准确率和平均损失
        average_loss = total_loss / len(train_dataloader)
        accuracy = total_correct / len(train_dataloader)
        输出 = f'第 [{epoch + 1}/{num_epochs}] 轮, 平均损失: {average_loss}, 准确率: {accuracy * 100:.5f}%'
        print(输出)
        训练信息.append(输出)

        if 保存轮次 is not None:
            if (epoch+1) % 保存轮次 == 0:
                _保存模型(当前时间=当前时间, 模型参数=模型参数, 保存位置=保存位置, input_size=input_size, num_epochs=num_epochs, model=model,
                      learning_rate=learning_rate, 训练信息=训练信息)

    _保存模型(当前时间=当前时间, 模型参数=模型参数, 保存位置=保存位置, input_size=input_size, num_epochs=num_epochs, model=model,
          learning_rate=learning_rate, 训练信息=训练信息)

    print(f'保存成功：  {保存位置}/{当前时间.strftime("%m_%d_%H_%M_%S")}_model.pth')


if __name__ == "__main__":
    pass