import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """返回学习率计划程序

    Parameters:
        optimizer          -- 网络的优化器
        opt (option class) -- 存储所有实验标志；需要是BaseOptions的子类。　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    对于“线性”，我们在第一个<opt.niter>时期保持相同的学习率
    在下一个<opt.niter\u decay>时期内，将速率线性衰减为零。
    对于其他调度器（step、plateau和cosine），我们使用默认的PyTorch调度器。
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class BaseModel(ABC):
    """此类是模型的抽象基类（ABC）.
    要创建子类，需要实现以下五个函数:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.visual_features = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.istest = True if opt.phase == 'test' else False  # 如果是测试，该模式下，没有标注样本；

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的特定于模型的选项，并重写现有选项的默认值.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        从数据加载器解压缩输入数据，并执行必要的预处理步骤。参数：输入（dict）：包括数据本身及其元数据信息。

        Parameters:
            input (dict): includes the data itself and its metadata information.
            从数据加载器解压缩输入数据，并执行必要的预处理步骤。参数：输入（dict）：包括数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def forward(self):
        """向前跑传球；由函数<optimize\u parameters>和<test>调用。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损耗，梯度并更新网络权重；在每次训练迭代中都被调用"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers
            加载和打印网络；创建调度程序
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            存储所有实验标记；需要是BaseOptions的子类
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """在测试期间将模型设为评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()


    def train(self):
        """Make models train mode during train time
        在训练期间使模型成为训练模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """用于测试时间的正向函数。

        这个函数将<forward>函数包装在no\u grad（）中，所以我们不保存backprop的中间步骤
        它还调用<compute\u visuals>来生成额外的可视化结果
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """为visdom和HTML可视化计算额外的输出图像"""
        pass

    def get_image_paths(self):
        """ 返回用于加载当前数据的图像路径"""
        return self.image_paths

    def update_learning_rate(self):
        """更新所有网络的学习率；在每个时代结束时调用"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """返回可视化图像。 train.py将显示具有视觉效果的这些图像，并将图像保存到HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """退货交易损失错误。 train.py将在控制台上打印出这些错误，并将其保存到文件中Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """将所有网络保存到磁盘.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def get_visual(self, name):
        visual_ret = {}
        visual_ret[name] = getattr(self, name)
        return visual_ret

    def pred_large(self, A, B, input_size=256, stride=0):
        """
        输入前后时相的大图，获得预测结果
        假定预测结果中心部分为准确，边缘padding = (input_size-stride)/2
        :param A: tensor, N*C*H*W
        :param B: tensor, N*C*H*W
        :param input_size: int, 输入网络的图像size
        :param stride: int, 预测时的跨步
        :return: pred, tensor, N*1*H*W
        """
        import math
        import numpy as np
        n, c, h, w = A.shape
        assert A.shape == B.shape
        # 分块数量
        n_h = math.ceil((h - input_size) / stride) + 1
        n_w = math.ceil((w - input_size) / stride) + 1
        # 重新计算长宽
        new_h = (n_h - 1) * stride + input_size
        new_w = (n_w - 1) * stride + input_size
        print("new_h: ", new_h)
        print("new_w: ", new_w)
        print("n_h: ", n_h)
        print("n_w: ", n_w)
        new_A = torch.zeros([n, c, new_h, new_w], dtype=torch.float32)
        new_B = torch.zeros([n, c, new_h, new_w], dtype=torch.float32)
        new_A[:, :, :h, :w] = A
        new_B[:, :, :h, :w] = B
        new_pred = torch.zeros([n, 1, new_h, new_w], dtype=torch.uint8)
        del A
        del B
        #
        for i in range(0, new_h - input_size + 1, stride):
            for j in range(0, new_w - input_size + 1, stride):
                left = j
                right = input_size + j
                top = i
                bottom = input_size + i
                patch_A = new_A[:, :, top:bottom, left:right]
                patch_B = new_B[:, :, top:bottom, left:right]
                # print(left,' ',right,' ', top,' ', bottom)
                self.A = patch_A.to(self.device)
                self.B = patch_B.to(self.device)
                with torch.no_grad():
                    patch_pred = self.forward()
                    new_pred[:, :, top:bottom, left:right] = patch_pred.detach().cpu()
        pred = new_pred[:, :, :h, :w]
        return pred

    def load_networks(self, epoch):
        """从磁盘加载所有网络。

        Parameters:
            历元（int）——当前历元；在文件名“%s\u net\uu%s.pth”中使用（epoch，name）
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                # if isinstance(net, torch.nn.DataParallel):
                # net = net.module
                # net = net.module  # 适配保存的module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    # print(key)
                net.load_state_dict(state_dict,strict=False)

    def print_networks(self, verbose):
        """打印网络和（如果详细）网络体系结构中的参数总数

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """为所有网络设置requires\u grad=Fasle以避免不必要的计算
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad




if __name__ == '__main__':

    A = torch.rand([1,3,512,512],dtype=torch.float32)
    B = torch.rand([1,3,512,512],dtype=torch.float32)
