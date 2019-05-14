import tensorflow as tf
from imdb_model import Model
from utils import dataset, batch_iter, AverageMeter
import math
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
tf.reset_default_graph()

epochs = 10
output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
output_path = output_dir + "model.ckpt"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def train(model_type='cnn'):
    if model_type=='rnn':
        padding=False
    else:
        padding=True

    model = Model(hidden_size=64)
    logits = model.forward(cell_type='rnn', model_type=model_type)
    loss = model.backword(logits)
    y_pred = tf.argmax(logits, axis=1, name='y_pred')
    y_true = tf.get_default_graph().get_tensor_by_name('y_true:0')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32), name='accuracy')


    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss, name='train_step')
    init = tf.global_variables_initializer()
    data = dataset(padding=padding)
    x_train,y_train, x_val, y_val = data.x_train, data.y_train, data.x_val, data.y_val
    train_losses = []
    train_acces = []
    valid_losses = []
    valid_acces = []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epochs+1):
            avg_train_loss, train_acc, valid_loss, valid_acc = train_per_epoch(model, sess, x_train,y_train, x_val, y_val, epoch, loss, model_type=model_type)
            train_losses.append(avg_train_loss)
            train_acces.append(train_acc)
            valid_losses.append(valid_loss)
            valid_acces.append(valid_acc)
    return train_losses, train_acces, valid_losses, valid_acces


def train_per_epoch(model, sess, train_data,train_labels, test_data, test_labels, epoch, loss, batch_size=64, model_type='rnn'):
    loss_meter = AverageMeter()
    n_minibatches = math.ceil(len(train_data) / batch_size)
    print(f'Epoch{epoch}')
    if model_type == 'rnn':
        with tqdm(total=(n_minibatches)) as prog:
            for i, (train_x, train_x_lengths, train_y) in enumerate(batch_iter(train_data, train_labels, batch_size, use_for=model_type)):
                loss_train, train_acc, _ = sess.run([loss, 'accuracy:0', 'train_step'],
                                                    feed_dict={'sent:0': train_x, 'sent_lengths:0': train_x_lengths,
                                                               'y_true:0': train_y})
                prog.update(1)
                loss_meter.update(loss_train.item())
    else:
        with tqdm(total=(n_minibatches)) as prog:
            for i, (train_x, train_y) in enumerate(batch_iter(train_data, train_labels, batch_size, use_for=model_type)):
                loss_train, train_acc, _ = sess.run([loss, 'accuracy:0', 'train_step'],
                                                    feed_dict={'sent:0': train_x, 'y_true:0': train_y})
                prog.update(1)
                loss_meter.update(loss_train.item())


    print("Average Train Loss: {}".format(loss_meter.avg))
    print('- train_accuracy: {:.2f}'.format(train_acc * 100.0))
    print("Evaluating on dev set", )
    if model.train==True:
        model.train=False
    valid_acc, valid_loss = evaluate(sess, test_data, test_labels, loss, model_type=model_type)
    model.train=True
    print("- valid_accuracy: {:.2f}".format(valid_acc * 100.0))
    print("- valid_loss: {:.2f}".format(valid_loss))
    return loss_meter.avg, train_acc, valid_loss, valid_acc

def evaluate(sess, test_data, test_labels, loss, model_type='rnn'):
    y_pred_test = np.array([])
    loss_meter = AverageMeter()
    n_minibatches = math.ceil(len(test_data) / 64)
    if model_type == 'rnn':
        with tqdm(total=(n_minibatches)) as prog:
            for i, (test_x, test_x_lengths, test_y) in enumerate(batch_iter(test_data, test_labels, 64, shuffle=False, use_for=model_type)):
                y_pred_batch, loss_valid = sess.run(['y_pred:0', loss],
                                                    feed_dict={'sent:0': test_x, 'sent_lengths:0':test_x_lengths,
                                                               'y_true:0': test_y})
                y_pred_test = np.append(y_pred_test, y_pred_batch)
                prog.update(1)
                loss_meter.update(loss_valid.item())
    else:
        with tqdm(total=(n_minibatches)) as prog:
            for i, (test_x, test_y) in enumerate(
                    batch_iter(test_data, test_labels, 128, shuffle=False, use_for=model_type)):
                y_pred_batch, loss_valid = sess.run(['y_pred:0', loss],
                                                    feed_dict={'sent:0': test_x, 'y_true:0': test_y})
                y_pred_test = np.append(y_pred_test, y_pred_batch)
                prog.update(1)
                loss_meter.update(loss_valid.item())
    return np.mean(y_pred_test == test_labels), loss_meter.avg


if __name__ == '__main__':
    train_losses, train_acces, valid_losses, valid_acces = train(model_type='rnn')
    print(f'min val loss:{min(valid_losses)}')
    print(f'max acc:{valid_acces[np.argmin(valid_losses)]}')



