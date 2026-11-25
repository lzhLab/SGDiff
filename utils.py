import importlib


def get_obj_from_str(string, reload=False):
    # module是py文件的路径，cls是该py文件当中的模型module类
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 使用importlib.import_module函数导入指定的模块
    # 使用getattr函数获取导入的模块中的指定类对象
    # 返回获取到的类对象
    return getattr(importlib.import_module(module, package=None), cls)


# 利用当前配置现的target和params属性来创建相应的对象
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    # 初始化模型并传入config中的参数
    return get_obj_from_str(config["target"])(**config.get("params", dict()))