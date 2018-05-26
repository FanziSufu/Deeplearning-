#!usr/bin/python3
# -*- coding: UTF-8 -*-

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE

# 读取文本数据
with open('text8', 'r') as f:
    text = f.read()


# 预处理数据
def preprocess(text, freq=5):
    text = text.lower()
    words = text.split()
    word_counts = Counter(words)

    return [word for word in words if word_counts[word] > freq]


words = preprocess(text)

# 构建映射表
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}

# 对原文本进行vocab到int的转换
int_words = [vocab_to_int[word] for word in words]

# 抽样，去除停用词（高频词）
t = 1e-5
threshold = 0.8
int_word_counts = Counter(int_words)
total_count = len(int_words)
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
train_words = [word for word in int_words if prob_drop[word] < threshold]


# 构造batch
def get_targets(words, idx, window_size=5):
    '''获取input word的上下文单词'''
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    '''构造一个获取batch的生成器'''
    n_batches = len(words) // batch_size
    words = words[: batch_size * n_batches]

    for i in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[i: i + batch_size]
        for idx in range(len(batch)):
            batch_x = batch[idx]
            batch_y = get_targets(batch, idx, window_size)
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


# 配置各种参数
vocab_size = len(vocab)
embedding_size = 128
epochs = 1
batch_size = 1000
window_size = 5
num_sampled = 64
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))


# 构建网络图
graph = tf.Graph()
with graph.as_default():
    train_dataset = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None, None])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_biases = tf.Variable(tf.zeros([vocab_size]))

    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocab_size))

    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


# 运行图
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print('Initialized')

    steps = 1
    losses = 0
    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)

        for x, y in batches:
            feed_dict = {train_dataset: x, train_labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            losses += train_loss

            if steps % 100 == 0:
                print('Epoch: {}/{}'.format(e, epochs),
                      'Steps: {}'.format(steps),
                      'Avg. Training loss: {:.4f}'.format(losses / 100))
                losses = 0

            if steps % 1000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s, %s' % (log, close_word)
                    print(log)

            steps += 1

    final_embeddings = normalized_embeddings.eval()

# 使用TSNE可视化词向量
viz_words = 400
tsne = TSNE()
embed_tsne = tsne.fit_transform(final_embeddings[: viz_words, :])

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(embed_tsne[idx, 0], embed_tsne[idx, 1], color='blue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]))
plt.show()
