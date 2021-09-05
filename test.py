from skimage import io,transform
import numpy as np
from glob import glob
import os
import time
import keras
import matplotlib
from matplotlib import rcParams
matplotlib.use("TkAgg")
config = {
    "font.family":'Times New Roman',
    "font.size": 15,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
import cv2
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from keras import backend as K
from scipy.io import savemat
from skimage.io import imsave
from sklearn.metrics import roc_curve, auc
np.set_printoptions(threshold=np.inf)

w = 128
h = 128
c = 3


def read_img(path):
    # os.path.isdir检测路径是否是目录， os.listdir指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs = []
    labels = []
    # enumerate 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
    for idx, folder in enumerate(cate):
        # 获取指定目录下所有jpg文件，返回list类型
        for im in glob(folder+'/*.jpg'):
            # print('reading the images:%s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h, c), mode="constant")
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)    # 转化为数组


# path = r'F:\TCGA-LGG 脑瘤/'
# test_x, test_y = read_img(path)
# print(test_y)

# saver = tf.train.import_meta_graph(r'F:\model\model3/brain_class.ckpt.meta')     # 模型加载地址
# graph = tf.get_default_graph()      # 获取当前图，为了后续训练时恢复变量
# tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]      # 得到当前图中所有变量的名称
#
# x = graph.get_tensor_by_name('x:0')       # 获取输入变量
# y = graph.get_tensor_by_name('y_:0')     # 获取输出变量
#
# pred = graph.get_tensor_by_name('logits_eval:0')    # 获取网络的输出值
# # 定义评价值
# # correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
# print(pred)
# correct = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), test_y)
# print(correct)
# accuracy = tf.cast(correct, tf.int32)
# # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_y, logits=pred))
#
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('F:\model/model3/'))
#     print("finish load model")
#     # test
#     test_num = int(len(test_y))
#     # test_acc = []
#     # test_loss = []
#     test_accuracy = []
#     for i in range(test_num):
#         test_acc = sess.run([accuracy], feed_dict={x: test_x, y: test_y})
#         print(sess.run(correct))
#         test_accuracy += test_acc
#     print(test_accuracy)
#     print('test_Acc:%f' % (np.sum(test_accuracy) / test_num))
# def plot_confusion_matrix(y_true, y_pred, labels):
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix
    # cmap = plt.cm.binary
    # cm = confusion_matrix(y_true, y_pred)
    # tick_marks = np.array(range(len(labels))) + 0.5
    # np.set_printoptions(precision=2)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(10, 8), dpi=120)
    # ind_array = np.arange(len(labels))
    # x, y = np.meshgrid(ind_array, ind_array)
    # intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    # for x_val, y_val in zip(x.flatten(), y.flatten()):
        # #

        # if (intFlag):
            # c = cm[y_val][x_val]
            # plt.text(x_val, y_val, "%d" % (c,), color='red', family='Times New Roman',fontsize=20, va='center', ha='center')

        # else:
            # c = cm_normalized[y_val][x_val]
            # if (c > 0.0001):
                # #这里是绘制数字，可以对数字大小和颜色进行修改
                # plt.text(x_val, y_val, "%0.3f" % (c,), family='Times New Roman',color='red', fontsize=24, va='center', ha='center')
            # else:
                # plt.text(x_val, y_val, "%d" % (0,), family='Times New Roman',color='red', fontsize=24, va='center', ha='center')
    # if(intFlag):
        # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # else:
        # plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    # plt.gca().set_xticks(tick_marks, minor=True)
    # plt.gca().set_yticks(tick_marks, minor=True)
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    # plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)
    # plt.title('')
    # cb=plt.colorbar()
    # for l in cb.ax.yaxis.get_ticklabels():
        # l.set_family('Times New Roman')
		
    # xlocations = np.array(range(len(labels)))
    # #plt.xticks(xlocations, labels, rotation=90,fontproperties='Times New Roman',size=16)
    # plt.xticks(xlocations, labels, fontproperties='Times New Roman',size=16)
    # plt.yticks(xlocations, labels,fontproperties='Times New Roman',size=16)
    # plt.ylabel('Index of True Classes',fontdict={'family':'Times New Roman','size':21})
    # plt.xlabel('Index of Predict Classes',fontdict={'family':'Times New Roman','size':21})
    # plt.savefig('confusion_matrix.jpg', dpi=300)
    # plt.show()

with tf.Session() as sess:
    path ='f:/目标数据集/test/'
    #path = r'F:\test2/'
    test_x, test_y11 = read_img(path)
    test_x1=test_x[0:200,:,:,:]
    test_y1=test_y11[0:200]
    test_x2=test_x[200:,:,:,:]
    test_y2=test_y11[200:]
    print(test_x.shape)
    print(test_x1.shape)
    print(test_x2.shape)
    print(test_y11.shape)
    print(test_y11)
    start=time.clock()
    saver = tf.train.import_meta_graph('C:/Users/hz/msnet/ckpt2/train.ckpt-4.meta')
    saver.restore(sess, tf.train.latest_checkpoint('C:/Users/hz/msnet/ckpt2/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict1 = {x: test_x1}
    feed_dict2 = {x: test_x2}
    logits = graph.get_tensor_by_name("logits_eval:0")
    classfication_result = sess.run(logits, feed_dict1)
    classfication_result1=sess.run(logits,feed_dict2)
    end=time.clock()
    print('time',end-start)
    print(classfication_result)
    class_pred = np.vstack((classfication_result,classfication_result1))
    pred1 = np.argmax(classfication_result,axis=1)
    pred2 = np.argmax(classfication_result1,axis=1)

    pred = np.hstack((pred1,pred2))
    # classification_pred = np.vstack(classfication_result,classfication_result1)
    # pred = 
    # output1 = []
    # output1 = tf.argmax(classfication_result, 1).eval()
    # output2 = []
    # output2 = tf.argmax(classfication_result1, 1).eval()
    # print(output1)
    # print(output2)
    #pred = np.vstack((output1,output2))
    print(pred)
    # label1=np.concatenate((test_y1,test_y2),axis=0)
    # classfication_result2=np.concatenate((classfication_result,classfication_result1),axis=0)
    # s=tf.argmax(classfication_result2,1).eval()
    # label3=['Abnormal','Normal']
    # plot_confusion_matrix(label1,s,label3)
	
	
	
	
    sum=0
    TP=0
    TN=0
    FP=0
    FN=0

    for i in range(400):
        if pred[i] == test_y11[i]:
            sum=sum+1
            if i<200:
                TN=TN+1
            else:
                TP=TP+1
        elif i<200:
            FN=FN+1
        else:
            FP=FP+1
    print(TP,TN,FP,FN)
    acc=sum/(400)
    print(acc)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    f1=2*recall*precision/(recall+precision)
    print(recall)
    print(precision)
    print(f1)
	
    num_classes=2
    y_test = keras.utils.to_categorical(test_y11,num_classes)
	# print(y_test)
	# print(predict)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], class_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
	# fpr[0].shape==tpr[0].shape==(21, ), fpr[1].shape==tpr[1].shape==(35, ), fpr[2].shape==tpr[2].shape==(33, ) 
	# roc_auc {0: 0.9118165784832452, 1: 0.6029629629629629, 2: 0.7859477124183007}

    plt.figure()
    lw = 3
    plt.plot(fpr[1], tpr[1], color='darkorange',lw=lw, label='ROC curve (area = %0.3f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
	#plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('resnet',dpi=600)
    plt.show()
