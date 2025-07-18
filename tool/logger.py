import sys
import os
import pandas as pd
from tensorboardX import SummaryWriter
from os.path import join
import time
from collections import OrderedDict
from tool.helper import dict_round

#在训练阶段用tensorboard和csv文件记录数据
class Logger:
    def __init__(self,save_name):
        """
        self.log：保存一个 pandas DataFrame，用来累积所有 epoch 的记录
        self.summary：SummaryWriter对象，延迟初始化（第一次写 TensorBoard 时再创建）
        self.name：用于指定保存目录或文件名前缀
        self.time_now：生成当前本地时间的字符串，用作文件名后缀（形如 _2025-06-29-07-12），避免每次运行覆盖旧文件
        """
        self.csv_path = join(save_name,'metrics')
        self.tensorboard_path=join(save_name,'tensorboard')
        os.makedirs(self.csv_path, exist_ok=True)
        os.makedirs(self.tensorboard_path, exist_ok=True)
        #csv文件保存到./save_name/metrics中
        self.csv_file = os.path.join(
            self.csv_path,
            f"log_{time.strftime('%m%d-%H:%M', time.localtime())}.csv"
        )
        self.summary = None
        # Track whether CSV header has been written
        self._csv_header_written = False

    def update(self,epoch,train_log,val_log):
        """
        写入日志，在每次训练或验证结束后调用
        epoch：当前训练轮数
        train_log：一个 dict，包含训练阶段要记录的各项指标（如loss、accuracy等
        val_log：一个 dict，包含验证阶段要记录的指标。
        """
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)

        item = dict_round(item,6)#格式化，保留小数点后6位有效数字
        print(item)
        #保存到CSV文件和TensorBoard
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        """
        将单次记录 item 转为单行 DataFrame tmp
        如果已有 self.log，就 append；否则初始化 self.log = tmp
        每次都把整个累积的 self.log 覆盖写入 CSV，路径为 <save_name>/log_<时间戳>.csv
        """
        df = pd.DataFrame([item])
        # 追加模式：只在第一次写入时输出表头，避免每次重写整个文件。
        df.to_csv(
            self.csv_file,
            mode='a',
            header=not self._csv_header_written,
            index=False
        )
        self._csv_header_written = True

    def update_tensorboard(self,item):
        """
        延迟初始化 SummaryWriter，输出目录为 <save_name>/。
        遍历 item 中除了 "epoch" 外的每个键值对，调用 add_scalar(name, value, global_step) 写入 TensorBoard。
        """
        if self.summary is None:
            self.summary = SummaryWriter(self.tensorboard_path)
        epoch = item['epoch']
        for key, value in item.items():
            if key != 'epoch':
                self.summary.add_scalar(key, value, epoch)

    def save_graph(self,model,input):
        """
        将模型的网络结构可视化并写入 TensorBoard，input：一个用于走一次前向的示例输入张量
        """
        if self.summary is None:
            self.summary = SummaryWriter(self.tensorboard_path)
        # Add model graph
        self.summary.add_graph(model, (input,))
        print("Architecture of model saved to TensorBoard.")
    def close(self):
        """
        关闭底层资源——TensorBoard编写器
        """
        if self.summary:
            self.summary.close()

# 记录打印到终端的信息
class Print_Logger(object):
    """
    重定向 sys.stdout，使得所有 print()的输出同时：
    打印到控制台（self.terminal.write），并追加写入到指定的日志文件（self.log.write）
    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout                          #控制台
        self.log = open(filename, "a", buffering=1)         #如果文件已存在，写入的内容会追加到文件末尾，不会覆盖。buffering=1 表示行缓冲，数据会先缓存在内存中，当遇到换行符 \n 或缓冲区满时，才会将数据刷新到磁盘

    def write(self, message):
        self.terminal.write(message)        #显示信息
        self.log.write(message)             #记录信息

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

    def close(self):
        """
        关闭日志文件
        """
        try:
            self.log.close()
        except Exception:
            pass
# call by
# sys.stdout = Logger(os.path.join(save_path,'test_log.txt'))