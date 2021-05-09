import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
from util import html
from util.visualizer import save_images
from util.metrics import AverageMeter
import copy
import numpy as np
import torch
import random

# 2021年5月10日02:32:44
def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# set seeds
# seed_torch(2019)
ifSaveImage = False

def make_val_opt(opt):

    val_opt = copy.deepcopy(opt)
    val_opt.preprocess = ''  #
    # 硬编码一些参数进行测试
    val_opt.num_threads = 0   # 测试代码仅支持num_threads = 1
    val_opt.batch_size = 4    # 测试代码仅支持batch_size = 1
    val_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opt.no_flip = True    # 没有翻转如果需要翻转图像的结果，请在此行中注释。
    val_opt.angle = 0
    val_opt.display_id = -1   # 没有视觉显示；测试代码会将结果保存到HTML文件中。
    val_opt.phase = 'val'
    val_opt.split = opt.val_split  # jsonDataset和ListDataset中的函数
    val_opt.isTrain = False
    val_opt.aspect_ratio = 1
    val_opt.results_dir = './results/'
    val_opt.dataroot = opt.val_dataroot
    val_opt.dataset_mode = opt.val_dataset_mode
    val_opt.dataset_type = opt.val_dataset_type
    val_opt.json_name = opt.val_json_name
    val_opt.eval = True

    val_opt.num_test = 2000
    return val_opt


def print_current_acc(log_name, epoch, score):
    """在控制台上打印当前帐户；还可以节省磁盘损失
    Parameters:在控制台上打印当前帐户；还可以将损失保存到磁盘上参数
    """
    message = '(epoch: %d) ' % epoch
    for k, v in score.items():
        message += '%s: %.3f ' % (k, v)
    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def val(opt, model):
    opt = make_val_opt(opt)
    dataset = create_dataset(opt)  # 给定opt.dataset_mode和其他选项来创建数据集
    # model = create_model(opt)      # 给定opt.model和其他选项来创建模型
    # model.setup(opt)               # 常规设置：加载和打印网络；创建调度程序

    web_dir = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # 定义文件目录
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    model.eval()          # 在测试期间将模型设为评估模式
    # 创建一个日志文件来存储训练损失
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ val acc (%s) ================\n' % now)

    running_metrics = AverageMeter()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 仅将我们的模型应用于opt.num_test图片。
            break
        model.set_input(data)  # 从数据加载器解压缩数据
        score = model.test(val=True)           # 运行推断
        running_metrics.update(score)
        visuals = model.get_current_visuals()  # 获取图像结果  可视化
        img_path = model.get_image_paths()     # 获取图像路径
        #if i % 5 == 0:  # save images to an HTML file
            #print('processing (%04d)-th image... %s' % (i, img_path))
        if ifSaveImage:
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    score = running_metrics.get_scores()
    print_current_acc(log_name, epoch, score)
    if opt.display_id > 0:
        visualizer.plot_current_acc(epoch, float(epoch_iter) / dataset_size, score)
    webpage.save()  # save the HTML

    return score[metric_name]

metric_name = 'F1_1'


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options 解析我们的选项，创建检查点目录后缀，并设置GPU设备
    dataset = create_dataset(opt)  # 给定opt.dataset_mode和其他选项来创建数据集create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # 获取数据集中的图像数量。get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # 给定opt.model和其他选项来创建模型create a model given opt.model and other options
    model.setup(opt)               # 常规设置：加载和打印网络；创建调度程序regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # 创建一个可视化工具来显示保存的图像和曲线图create a visualizer that display/save images and plots
    total_iters = 0                # 训练迭代总数the total number of training iterations
    miou_best = 0
    n_epoch_bad = 0
    epoch_best = 0
    time_metric = AverageMeter()   # 计算并存储平均值和当前值
    time_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'time_log.txt') # 路径拼接
    with open(time_log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ training time (%s) ================\n' % now)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # 不同时代的外循环;我们通过<epoch_count>，<epoch_count> + <save_latest_freq>保存模型outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # 整个时期的计时器
        iter_data_time = time.time()    # 每次迭代加载数据的计时器
        epoch_iter = 0                  # 当前时期的训练迭代次数，每个时期重置为0
        model.train()
       # miou_current = val(opt, model)
        for i, data in enumerate(dataset):  # 一个时期内的内循环
            iter_start_time = time.time()  # 每次迭代的计算计时器
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset() # 重置该对象的状态
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            n_epoch = opt.niter + opt.niter_decay

            model.set_input(data)         # 从数据集中解压缩数据并应用预处理
            model.optimize_parameters()   # 计算损失函数，获取梯度，更新网络权重2
            if ifSaveImage:
                if total_iters % opt.display_freq == 0:   # 在visdom上显示图像并将图像保存到HTML文件
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            if total_iters % opt.print_freq == 0:   # 打印培训损失并将日志记录信息保存到磁盘
                losses = model.get_current_losses()   # 得到当前损失
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)  # 可视化
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # 每次<save_latest_freq>迭代都缓存我们的最新模型    保存模型
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)  # 保存模型到磁盘

            iter_data_time = time.time()

        t_epoch = time.time()-epoch_start_time
        time_metric.update(t_epoch)
        print_current_acc(time_log_name, epoch,{"current_t_epoch": t_epoch})  # 打印当前epoch花费时间


        if epoch % opt.save_epoch_freq == 0:              # 在每个<save_epoch_freq>个时期内缓存我们的模型
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            miou_current = val(opt, model)

            if miou_current > miou_best:
                miou_best = miou_current
                epoch_best = epoch
                model.save_networks(str(epoch_best)+"_"+metric_name+'_'+'%0.5f'% miou_best)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # 在每个时期结束时更新学习率。  更新所有网络的学习率；在每个时代结束时调用

    time_ave = time_metric.average()
    print_current_acc(time_log_name, epoch, {"ave_t_epoch": time_ave})
