#!usr/bin/python3
# -*- coding: UTF-8 -*-


from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
import sys
import tarfile
import random
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'


def download_progress_hook(count, blockSize, totalSize):
    '''下载预处理'''
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write('%s%%' % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, experted_bytes, force=False):
    '''下载数据文件。因为404的原因，无法通过此函数直接下载，故而在浏览器上手动下载了数据，用于后续操作'''
    dest_filename = os.path.join(data_root, filename)

    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ =urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete')

    statinfo = os.stat(dest_filename)
    if statinfo.st_size == experted_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception('Failed to verify' + dest_filename + '. Can you get to it with a browser')

    return dest_filename


train_filename = 'notMNIST_large.tar.gz'
test_filename = 'notMNIST_small.tar.gz'

num_class = 10
# np.random.seed( ) 用于指定随机数生成时所用算法开始的整数值。
# 1.如果使用相同的seed( )值，则每次生成的随即数都相同；
# 2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
# 3.设置的seed()值仅一次有效
np.random.seed(133)


def maybe_extract(filename, force=False):
    '''解压下载的文件，并返回解压文件夹里的文件名列表'''
    root = os.path.splitext(os.path.splitext(filename)[0])[0]

    if os.path.exists(root) and not force:
        print('%s already present - Skipping extraction of %s' % (root, filename))
    else:
        print('Extracting data for %s. This maybe take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_class:
        raise Exception('Expected %d folders, one per class. Found %d instead' % (num_class, len(data_folders)))
    print(data_folders)

    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


def problem_1():
    '''在pycharm随机展示notMNIST_large中的5张图片'''
    root = os.path.splitext(os.path.splitext(train_filename)[0])[0]
    for _ in range(5):
        label_idx = random.randint(0, len(os.listdir(root)) - 1)
        label_root = train_folders[label_idx]
        pngnames = os.listdir(label_root)
        png_idx = random.randint(0, len(pngnames) - 1)
        png = img.imread(os.path.join(label_root, pngnames[png_idx]))  # 读取图片信息，并用ndarray数组形式表示
        plt.imshow(png)
        plt.show()


# _ = problem_1()


image_size = 28
pixel_depth = 255.0


def load_letter(folder, min_num_images):
    '''加载每个字母标签的数据'''
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0: num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation', np.std(dataset))

    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    '''把数据保存在二进制文件中'''
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


def problem_2():
    '''随机展示pickle文件的5个数据图像，以验证归一化后的图像仍然可用'''
    for _ in range(5):
        pickle_idx = np.random.randint(len(train_datasets))
        pickle_file = train_datasets[pickle_idx]
        with open(pickle_file, 'rb') as f:
            letter_set = pickle.load(f)
            sample_idx = np.random.randint(len(letter_set))
            sample_image = letter_set[sample_idx, :, :]
            plt.figure()
            plt.imshow(sample_image)
            plt.show()


# _ = problem_2()


def problem_3():
    '''输出每个pickle中包含的样本数量，以验证每个字母具有均衡的样本数'''
    for pickle_file in train_datasets + test_datasets:
        with open(pickle_file, 'rb') as f:
            letter_set = pickle.load(f)
            print('%s has %d examples' % (pickle_file, letter_set.shape[0]))


# _ = problem_3()


def make_arrays(nb_rows, img_size):
    '''创建待用数组'''
    if nb_rows:
        dataset = np.ndarray(shape=(nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(shape=nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, validation_size=0):
    '''合并数据'''
    num_classes = len(pickle_files)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    validation_dataset, validation_labels = make_arrays(validation_size, image_size)
    vsize_per_class = validation_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if validation_dataset is not None:
                    validation_letter = letter_set[:vsize_per_class, :, :]
                    validation_dataset[start_v: end_v, :, :] = validation_letter
                    validation_labels[start_v: end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class: end_l, :, :]
                train_dataset[start_t: end_t, :, :] = train_letter
                train_labels[start_t: end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return validation_dataset, validation_labels, train_dataset, train_labels


train_size = 200000
validationg_size = 10000
test_size = 10000

validation_dataset, validation_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, validationg_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', validation_dataset.shape, validation_labels.shape)
print('Test:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
    '''进一步打乱数据'''
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]

    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
validation_dataset, validation_labels = randomize(validation_dataset, validation_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)


def problem_4():
    '''通过生成前10个数据的图像，观察是否已是打乱的字母标签，依次来确定随机处理的有效性'''
    for i in range(10):
        sample = train_dataset[i, :, :]
        label = train_labels[i]
        print('label:', label)
        plt.figure()
        plt.imshow(sample)
        plt.show()


# _ = prbolem_4()


pickle_file = os.path.join(data_root, 'notMNIST.picle')

try:
    with open(pickle_file, 'wb') as f:
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'validation_dataset': validation_dataset,
            'validation_labels': validation_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels}
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


def problem_5():
    '''找出训练集和测试集中重复的样本，并计算个数'''
    overlap = {}
    for i, img_1 in enumerate(test_dataset):
        for j,img_2 in enumerate(train_dataset):
            if np.array_equal(img_1, img_2):
                if not i in overlap.keys():
                    overlap[i] = []
                overlap[i].append(j)

    return overlap

# p = problem_5()


def problem_6():
    '''训练一个逻辑回归模型'''
    logical_regressor = LogisticRegression()
    sample_size = 5000
    X_train = train_dataset[: sample_size].reshape(sample_size, 28 * 28)
    y_train = train_labels[: sample_size]
    logical_regressor.fit(X_train, y_train)

    X_validation = validation_dataset[: sample_size].reshape(sample_size, 784)
    y_validation = validation_labels[: sample_size]

    X_test = test_dataset.reshape(test_dataset.shape[0], 784)
    y_test = test_labels

    print('Accuracy on validation data:', logical_regressor.score(X_validation, y_validation))
    print('Accuracy on test data:', logical_regressor.score(X_test, y_test))


_ = problem_6()
