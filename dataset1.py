import argparse
import os
import random
import shutil


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)     #获取路径下的所有文件和文件夹的名称
    data_list = []
    GT_list = []

    for filename in filenames:
        """
        规范化文件名
        os.path.splitext(filename)将filename 拆分成一个元组
        假如filename为‘ISIC_0000079.jpg’，那么os.path.splitext(filename)将返回一个元组，['ISIC_0000079','.jpg']
        filename.split('_')根据_划分字符串，然后放到列表中；[:-len('.jpg')]相当于[:-4]，表示从字符串的开头到倒数第 4 个字符（不包括
        """
        ext = os.path.splitext(filename)[-1]            #获取文件后缀名，只处理jpg文件；索引[-1]是取元组中的最后一个元素
        if ext =='.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]           #去掉文件前缀和后缀名;，然后取出最后一个，再去除掉.jpg后缀名
            data_list.append('ISIC_'+filename+'.jpg')
            GT_list.append('ISIC_'+filename+'_segmentation.png')

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arrange = list(range(num_total))
    random.shuffle(Arrange)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='../ISIC/dataset/ISIC2018_Task1-2_Training_Input')
    parser.add_argument('--origin_GT_path', type=str, default='../ISIC/dataset/ISIC2018_Task1_Training_GroundTruth')

    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/valid_GT/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_GT_path', type=str, default='./dataset/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)