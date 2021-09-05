#tensorflow 模块
import tensorflow as tf
#skimage模块下的io transform(图像的形变与缩放)模块
from skimage import io,transform
#glob 文件通配符模块 
import glob
#os 处理文件和目录的模块
import os
#多维数据处理模块
import numpy as np
#
import time
 
#本地数据集地址
path1='F:/目标数据集/train/'
#path1="L:/增强眼底图像/train/"
#本地模型保存地址

 
#将所有的图片resize成100*100
w=128
h=128
c=3
 
def read_img(path):
     #os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
     #os.path.isdir(path)判断path是否是目录
     #b = [x+x for x in list1 if x+x<15 ]  列表生成式,循环list1，当if为真时，将x+x加入列表b
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
         #glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表         
        for im in glob.glob(folder+'/*.jpg'):
            #输出读取的图片的名称             
            #print('reading the images:%s'%(im))
            #io.imread(im)读取单张RGB图片 skimage.io.imread(fname,as_grey=True)读取单张灰度图片
            #读取的图片  
            img=io.imread(im)
            #skimage.transform.resize(image, output_shape)改变图片的尺寸
            img=transform.resize(img,(w,h,c))
            #将读取的图片数据加载到imgs[]列表中
            imgs.append(img)
            #将图片的label加载到labels[]中，与上方的imgs索引对应
            labels.append(idx)
    #将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
 
#调用读取图片的函数，得到图片和labels的数据集
data1,label1=read_img(path1)
 
 
#打乱顺序
#读取data矩阵的第一维数（图片的个数）
num_example=data1.shape[0]
#产生一个num_example范围，步长为1的序列
arr=np.arange(num_example)
#调用函数，打乱顺序
np.random.shuffle(arr)
#按照打乱的顺序，重新排序
num_example=data1.shape[0]
#产生一个num_example范围，步长为1的序列
arr=np.arange(num_example)
#调用函数，打乱顺序
np.random.shuffle(arr)
#按照打乱的顺序，重新排序
data1=data1[arr]
label1=label1[arr]
 
x_train=data1
y_train=label1 
#将所有数据分为训练集和验证集


#将所有数据分为训练集和验证集
# ratio=0.8
# s=np.int(num_example*ratio)
# x_train=data1[:s]
# print(x_train.shape)
# y_train=label1[:s]
# x_val=data1[s:]
# y_val=label1[s:]

#本地数据集地址
path2='F:/目标数据集/test/'
#path2="L:/增强眼底图像/test/"
 
 
# #调用读取图片的函数，得到图片和labels的数据集
data2,label2=read_img(path2)
x_val=data2
y_val=label2
#读取图片+数据预处理

 
#函数声明
 
 
#-----------------构建网络----------------------

#占位符，设置输入参数的大小和格式
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# 设置阈值函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 设置卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = "SAME")

# 设置池化层
def pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = "SAME")
	
def SE_block(x,ratio):
    shape = x.get_shape().as_list()
    channel_out = shape[3]
    out_shape=int(channel_out/ratio)
    # print(shape)
    with tf.variable_scope("squeeze_and_excitation"):
        # 第一层，全局平均池化层
        squeeze = tf.nn.avg_pool(x,[1,shape[1],shape[2],1],[1,shape[1],shape[2],1],padding = "SAME")
        # 第二层，全连接层
        w_excitation1 = weight_variable([1,1,channel_out,out_shape])
        b_excitation1 = bias_variable([out_shape])
        excitation1 = conv2d(squeeze,w_excitation1) + b_excitation1
        excitation1_output = tf.nn.relu(excitation1)
        # 第三层，全连接层
        w_excitation2 = weight_variable([1, 1, out_shape, channel_out])
        b_excitation2 = bias_variable([channel_out])
        excitation2 = conv2d(excitation1_output, w_excitation2) + b_excitation2
        excitation2_output = tf.nn.sigmoid(excitation2)
        # 第四层，点乘
        excitation_output = tf.reshape(excitation2_output,[-1,1,1,channel_out])
        h_output = excitation_output * x

    return h_output

def normalization(input,dim1):
    inputt=input
    a=tf.reduce_max(input,axis=1,keepdims=True)
    b=tf.reduce_max(a,axis=2,keepdims=True)
    max1=b
    c=tf.reduce_min(input,axis=1,keepdims=True)
    d=tf.reduce_min(c,axis=2,keepdims=True)
    min1=d
    # con=tf.constant(1e-10,shape=(batch_size,1,1,dim1))
    con=1e-10
    e=tf.divide(tf.subtract(input,min1),tf.add(tf.subtract(max1,min1),con)) #归一化
    return e
	
# def thresholding(input):
    # x=tf.zeros_like(input)
    # y=tf.ones_like(input)
    # out=tf.where(input>0.2*255,y,x)
    # return out
	
def dilation_conv2d(input,kernel_size, num_o, dilation_factor,name): #空洞卷积
    input_channel=input.shape[3].value
    with tf.variable_scope(name):
        conva1_weights=tf.get_variable('weight',[kernel_size,kernel_size,input_channel,num_o],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conva1_biases=tf.get_variable('bias',[num_o],initializer=tf.constant_initializer(0.0))
        conva1=tf.nn.atrous_conv2d(input,conva1_weights,dilation_factor,padding='SAME')
        output=tf.nn.relu(tf.nn.bias_add(conva1,conva1_biases))
        return output
		
def _add(x_l, name):
    return tf.add_n(x_l, name=name)
		
def ASPP(input, num_o, dilations):
    o = []
    for i, d in enumerate(dilations):
        o.append(dilation_conv2d(input, 3, num_o, d, name='aspp/conv%d' % (i+1)))
    return _add(o, name='aspp/add')
	
def conv2d1(input,kernel_size, num_o,name): #卷积
    input_channel=input.shape[3].value
    with tf.variable_scope(name):
        conva1_weights=tf.get_variable('weight',[kernel_size,kernel_size,input_channel,num_o],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conva1_biases=tf.get_variable('bias',[num_o],initializer=tf.constant_initializer(0.0))
        conva1=tf.nn.conv2d(input,conva1_weights,strides=[1,1,1,1],padding='SAME')
        output=tf.nn.relu(tf.nn.bias_add(conva1,conva1_biases))
        return output

def mlevel_block(input):  #提取多级特征
    o11=conv2d1(input,3,32,'11')
    o12=conv2d1(o11,3,32,'12')
    o13=conv2d1(o12,3,32,'13')
    o21=conv2d1(input,5,32,'21')
    o22=conv2d1(o21,5,32,'22')+o11
    o23=conv2d1(o22,5,32,'23')+o12
    o31=conv2d1(input,7,32,'31')
    o32=conv2d1(o31,7,32,'32')+o21+o11
    o33=conv2d1(o32,7,32,'33')+o22+o12
    return o13+o23+o33


        
 
def inference(input_tensor, train, regularizer):
#-----------------------第一层----------------------------
    branch1=input_tensor[:,:,:,0] #T1c序列
    branch1=tf.expand_dims(branch1,axis=-1)
    branch2=input_tensor[:,:,:,1] #T2序列
    branch2=tf.expand_dims(branch2,axis=-1)
    branch3=input_tensor[:,:,:,2] #T1序列
    branch3=tf.expand_dims(branch3,axis=-1)
    attention=tf.image.flip_left_right(branch2)
    attention1=branch2-attention  #获取差分特征图
    attention1=normalization(attention1,1)
    #attention1=thresholding(attention1)

##feature extraction stage 1
    ##branch1
    with tf.variable_scope('layers-branch1-conv1'):   #提取T1C特征
        convb11_weights=tf.get_variable('weight',[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convb11_biases=tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        convb11=tf.nn.conv2d(branch1,convb11_weights,strides=[1,1,1,1],padding='SAME')
        relub11=tf.nn.relu(tf.nn.bias_add(convb11,convb11_biases))
        

    with tf.variable_scope('layers-branch1-conv2'):
        convb12_weights=tf.get_variable('weight',[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convb12_biases=tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        convb12=tf.nn.conv2d(relub11,convb12_weights,strides=[1,1,1,1],padding='SAME')
        relub12=tf.nn.relu(tf.nn.bias_add(convb12,convb12_biases))
        # relub12=mlevel_block(branch1)
        relub12=tf.multiply(relub12,attention1)+relub12

    with tf.name_scope("branch1-pool1"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool11 = tf.nn.max_pool(relub12, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    #branch2
    with tf.variable_scope('layers-branch2-conv1'):   #提取T2特征
        convb21_weights=tf.get_variable('weight',[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convb21_biases=tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        convb21=tf.nn.conv2d(branch2,convb21_weights,strides=[1,1,1,1],padding='SAME')
        relub21=tf.nn.relu(tf.nn.bias_add(convb21,convb21_biases))

    with tf.variable_scope('layers-branch2-conv2'):
        convb22_weights=tf.get_variable('weight',[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convb22_biases=tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        convb22=tf.nn.conv2d(relub21,convb22_weights,strides=[1,1,1,1],padding='SAME')
        relub22=tf.nn.relu(tf.nn.bias_add(convb22,convb22_biases))
        # relub22=mlevel_block(branch2)
        relub22=tf.multiply(relub22,attention1)+relub22

    with tf.name_scope("branch2-pool1"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool21 = tf.nn.max_pool(relub22, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    #branch3
    with tf.variable_scope('layers-branch3-conv1'):   #提取T1特征
        convb31_weights=tf.get_variable('weight',[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convb31_biases=tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        convb31=tf.nn.conv2d(branch3,convb31_weights,strides=[1,1,1,1],padding='SAME')
        relub31=tf.nn.relu(tf.nn.bias_add(convb31,convb31_biases))

    with tf.variable_scope('layers-branch3-conv2'):
        convb32_weights=tf.get_variable('weight',[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convb32_biases=tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        convb32=tf.nn.conv2d(relub31,convb32_weights,strides=[1,1,1,1],padding='SAME')
        relub32=tf.nn.relu(tf.nn.bias_add(convb32,convb32_biases))
        # relub32=mlevel_block(branch3)
        relub32=tf.multiply(relub32,attention1)+relub32
        # relub32=relub32+relub22
    
    with tf.name_scope("branch1-pool1"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool31 = tf.nn.max_pool(relub32, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

##feature merge
    feat=tf.concat([pool11,tf.multiply(pool21,2),pool31],axis=3)
    with tf.name_scope("attention-attention2"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        attention2 = tf.nn.max_pool(attention1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        attention3 = tf.nn.max_pool(attention2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
    feat=tf.multiply(feat,attention2)
    feat1=SE_block(feat,ratio=4)
    with tf.variable_scope('layers-skip1-conv1'):   #提取T1特征
        convs11_weights=tf.get_variable('weight',[1,1,32,96],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convs11_biases=tf.get_variable('bias',[96],initializer=tf.constant_initializer(0.0))
        convs11=tf.nn.conv2d(pool21,convs11_weights,strides=[1,1,1,1],padding='SAME')
        relus11=tf.nn.relu(tf.nn.bias_add(convs11,convs11_biases))
    feat2=feat1+relus11
    # feat2=ASPP(feat2,96,[1,2,4])

##feature extraction2
    with tf.variable_scope('layer1-conv1'):
        #初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
        conv1_weights = tf.get_variable("weight",[3,3,96,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #初始化偏置conv1_biases，数量为32个
        conv1_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        #卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据，conv1_weights为权重，strides=[1, 1, 1, 1]表示左右上下滑动步长为1，padding='SAME'表示输入和输出大小一样，即补0
        conv1 = tf.nn.conv2d(feat2, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        #激励计算，调用tensorflow的relu函数
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        # relu1=relu1+tf.multiply(relu1,attention2)

    with tf.variable_scope('layers-skip2-conv1'):   #提取T1特征
        convs2_weights=tf.get_variable('weight',[1,1,32,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convs2_biases=tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.0))
        convs2=tf.nn.conv2d(pool21,convs2_weights,strides=[1,1,1,1],padding='SAME')
        relus2=tf.nn.relu(tf.nn.bias_add(convs2,convs2_biases))
        relu1=relu1+relus2

    with tf.name_scope("layer1-pool1"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
		
    with tf.variable_scope('layer2-conv2'):
        #初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
        conv2_weights = tf.get_variable("weight",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #初始化偏置conv1_biases，数量为32个
        conv2_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        #卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据，conv1_weights为权重，strides=[1, 1, 1, 1]表示左右上下滑动步长为1，padding='SAME'表示输入和输出大小一样，即补0
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        #激励计算，调用tensorflow的relu函数
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layers-skip3-conv3'):   #提取T1特征
        convs3_weights=tf.get_variable('weight',[1,1,32,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convs3_biases=tf.get_variable('bias',[256],initializer=tf.constant_initializer(0.0))
        convs3=tf.nn.conv2d(pool21,convs3_weights,strides=[1,1,1,1],padding='SAME')
        relus3=tf.nn.relu(tf.nn.bias_add(convs3,convs3_biases))
        pools3=tf.nn.max_pool(relus3, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        relu2=relu2+pools3

    with tf.name_scope("layer2-pool2"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope('layer3-conv3'):
        #初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
        conv3_weights = tf.get_variable("weight",[3,3,256,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #初始化偏置conv1_biases，数量为32个
        conv3_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
        #卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据，conv1_weights为权重，strides=[1, 1, 1, 1]表示左右上下滑动步长为1，padding='SAME'表示输入和输出大小一样，即补0
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        #激励计算，调用tensorflow的relu函数
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	
    with tf.variable_scope('layers-skip4-conv4'):   #提取T1特征
        convs4_weights=tf.get_variable('weight',[1,1,32,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        convs4_biases=tf.get_variable('bias',[512],initializer=tf.constant_initializer(0.0))
        convs4=tf.nn.conv2d(pool21,convs4_weights,strides=[1,1,1,1],padding='SAME')
        relus4=tf.nn.relu(tf.nn.bias_add(convs4,convs4_biases))
        pools4=tf.nn.max_pool(relus4, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        pools4=tf.nn.max_pool(pools4, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        relu3=relu3+pools4

    # with tf.name_scope("layer3-pool3"):
        # #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        # pool3 = tf.nn.max_pool(relu3, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")    
		
    # with tf.variable_scope('layer4-conv4'):
        # #初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
        # conv4_weights = tf.get_variable("weight",[3,3,512,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
        # #初始化偏置conv1_biases，数量为32个
        # conv4_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.0))
        # #卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据，conv1_weights为权重，strides=[1, 1, 1, 1]表示左右上下滑动步长为1，padding='SAME'表示输入和输出大小一样，即补0
        # conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        # #激励计算，调用tensorflow的relu函数
        # relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
		

		
    with tf.name_scope("layer4-pool4"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool4 = tf.nn.max_pool(relu3, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        pool4=tf.keras.layers.GlobalAveragePooling2D()(pool4)
        # nodes = 4*4*512
        # reshaped = tf.reshape(pool4,[-1,nodes])
 
# #-----------------------第五层----------------------------
    # with tf.variable_scope('layer11-fc1'):
        # #初始化全连接层的参数，隐含节点为1024个 
        # fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      # initializer=tf.truncated_normal_initializer(stddev=0.1))
        # if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        # fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        
        # #使用relu函数作为激活函数
        # fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # #采用dropout层，减少过拟合和欠拟合的程度，保存模型最好的预测效率
        # if train: fc1 = tf.nn.dropout(fc1, 0.5)
        
# #-----------------------第六层----------------------------
    # with tf.variable_scope('layer12-fc2'):
        # #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        # fc2_weights = tf.get_variable("weight", [1024, 512],
                                      # initializer=tf.truncated_normal_initializer(stddev=0.1))
        # if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        # fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
 
        # fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        # if train: fc2 = tf.nn.dropout(fc2, 0.5)
        
#-----------------------第七层----------------------------
    with tf.variable_scope('layer13-fc3'):
        #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(pool4, fc3_weights) + fc3_biases    

    return logit
 
#---------------------------网络结束---------------------------
    
#设置正则化参数为0.0001
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
 
#将上述构建网络结构引入
logits = inference(x,False,regularizer)

 
#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval') 
 
#设置损失函数，作为模型训练优化的参考标准，loss越小，模型越优
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
loss1=tf.add_n(tf.get_collection('losses'))
loss2=loss+loss1
#设置整体学习率为α为0.001
# batch_size=16
# global_step=tf.Variable(0,trainable=False)
# decay_ste=label1.shape[0]/batch_size
# learning_rate_base=0.01
# decay_rate=0.99
# learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,decay_ste,decay_rate)
# train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss2)

#设置预测精度
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 
#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
 
 
#训练和测试数据，可将n_epoch设置更大一些
 
#迭代次数
n_epoch=40  
#每次迭代输入的图片数据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
batch_size=16
saver=tf.train.Saver(max_to_keep=1)
sess=tf.Session()  
#初始化全局参数
sess.run(tf.global_variables_initializer())
max_acc=0
#开始迭代训练，调用的都是前面设置好的函数或变量
for epoch in range(n_epoch):
    start_time = time.time()
 
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    i=0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        i=i+1
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
 
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
    if val_acc>=max_acc:
        max_acc=val_acc;
        saver.save(sess,'ckpt2/train.ckpt',global_step=epoch+1)

#保存模型及模型参数

sess.close()