"""该软件包包含与目标功能，优化和网络架构有关的模块.

要添加一个名为“ dummy”的自定义模型类，您需要添加一个名为“ dummy_model.py”的文件并定义一个继承自BaseModel的子类DummyModel。.
您需要实现以下五个功能:
    -- <__init__>:                      初始化类；首先调用BaseModel .__ init __（self，opt）.
    -- <set_input>:                     从数据集中解压缩数据并应用预处理.
    -- <forward>:                       产生中间结果.
    -- <optimize_parameters>:           计算损耗，梯度并更新网络权重.
    -- <modify_commandline_options>:    （可选）添加特定于模型的选项并设置默认选项.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          指定要绘制和保存的训练损失.
    -- self.model_names (str list):         定义我们的培训中使用的网络.
    -- self.visual_names (str list):        指定要显示和保存的图像.
    -- self.optimizers (optimizer list):    定义和初始化优化器。您可以为每个网络定义一个优化器。如果同时更新两个网络，则可以使用itertools.chain对其进行分组。有关用法，请参见cycle_gan_model.py.

现在，您可以通过指定标志“ --model dummy”来使用模型类.
有关更多详细信息，请参见我们的模板模型类'template_model.py'.
"""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """导入模块“ models [model_name] _model.py”.

    在文件中，名为DatasetNameModel（）的类将
    被实例化。它必须是BaseModel的子类,
    而且不区分大小写.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """返回模型类的静态方法<modify_commandline_options>."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.
    给定选项创建模型。此函数扭曲类CustomDatasetDataLoader。这是该程序包和“ train.py” test.py“之间的主要接口
    此函数扭曲类CustomDatasetDataLoader。
   这是该程序包和“ train.py” test.py“之间的主要接口

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
