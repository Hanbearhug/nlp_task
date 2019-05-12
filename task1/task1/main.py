import tensorflow as tf
from imdb_model import Model
from utils import dataset, batch_iter, AverageMeter
import math
from tqdm import tqdm
import numpy as np
from datetime import datetime
tf.reset_default_graph()

epochs = 40

def train():
    model = Model(hidden_size=256)
    logits = model.forward()
    loss = model.backword(logits)
    y_pred = tf.argmax(logits, axis=1, name='y_pred')
    y_true = tf.get_default_graph().get_tensor_by_name('y_true:0')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32), name='accuracy')


    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss, name='train_step')
    init = tf.global_variables_initializer()
    data = dataset()
    train_data,train_labels, test_data, test_labels = data.train_data, data.test_labels, data.test_data, data.test_labels
    train_losses = []
    train_acces = []
    valid_losses = []
    valid_acces = []
    with tf.Session() as sess:
        sess.run(init)
        epoch = 1
        for epoch in range(1, epochs+1):
            avg_train_loss, train_acc, valid_loss, valid_acc = train_per_epoch(sess, train_data,train_labels, test_data, test_labels, epoch, loss)
            train_losses.append(avg_train_loss)
            train_acces.append(train_acc)
            valid_losses.append(valid_loss)
            valid_acces.append(valid_acc)
    return train_losses, train_acces, valid_losses, valid_acces


def train_per_epoch(sess, train_data,train_labels, test_data, test_labels, epoch, loss, batch_size=512):
    sent = tf.get_default_graph().get_tensor_by_name('sent:0')
    y_true = tf.get_default_graph().get_tensor_by_name('y_true:0')
    train_step = tf.get_default_graph().get_operation_by_name('train_step')
    accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')

    loss_meter = AverageMeter()
    n_minibatches = math.ceil(len(train_data) / batch_size)
    print(f'Epoch{epoch}')
    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(batch_iter(train_data, train_labels, batch_size)):
            #optimizer.zero_grad()   # remove any baggage in the optimizer
            #loss = 0. # store loss for this batch here

            ### YOUR CODE HERE (~5-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            loss_train, _ = sess.run([loss, train_step], feed_dict={sent: train_x, y_true: train_y})

            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss_train.item())

    print("Average Train Loss: {}".format(loss_meter.avg))
    train_acc = sess.run(accuracy, feed_dict={sent: train_x, y_true: train_y})
    print("Evaluating on dev set", )
    valid_acc, valid_loss = evaluate(sess, test_data, test_labels, loss)
    print("- valid_accuracy: {:.2f}".format(valid_acc * 100.0))
    return loss_meter.avg, train_acc, valid_loss, valid_acc

def evaluate(sess, test_data, test_labels, loss):
    sent = tf.get_default_graph().get_tensor_by_name('sent:0')
    y_pred = tf.get_default_graph().get_tensor_by_name('y_pred:0')
    y_true = tf.get_default_graph().get_tensor_by_name('y_true:0')
    y_pred_test = np.array([])
    loss_meter = AverageMeter()
    n_minibatches = math.ceil(len(test_data) / 512)
    with tqdm(total=(n_minibatches)) as prog:
        for i, (test_x, test_y) in enumerate(batch_iter(test_data, test_labels, 512, shuffle=False)):
            # optimizer.zero_grad()   # remove any baggage in the optimizer
            # loss = 0. # store loss for this batch here

            ### YOUR CODE HERE (~5-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            y_pred_batch, loss_valid = sess.run([y_pred, loss], feed_dict={sent: test_x, y_true:test_y})
            y_pred_test = np.append(y_pred_test, y_pred_batch)

            ### END YOUR CODE

            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss_valid.item())
    return np.mean(y_pred_test==test_labels), loss_meter.avg


if __name__ == '__main__':
    train_losses, train_acces, valid_losses, valid_acces = train()



