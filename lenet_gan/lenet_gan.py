from load_mnist import extract_images,extract_labels,save_sample
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
from tensorflow.contrib.layers import *

class lenet_gan():
    def __init__(self,FLAGS):
        self.FLAGS=FLAGS
        self.sess=tf.Session()
        self.noise_dim=100

        self.build_model()
    def build_model(self):
        # placeholder
        self.img_bt=tf.placeholder('float32',[self.FLAGS.batch_size,
                self.FLAGS.img_h,self.FLAGS.img_w,self.FLAGS.img_c])
        self.lbl_bt=tf.placeholder('int64',[self.FLAGS.batch_size])
        self.noise_bt=tf.placeholder("float32",[self.noise_dim])
        # output
        if not self.FLAGS.is_train:
            self.preds,self.logits=self.net(self.img_bt)
            # loss
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.lbl_bt,self.FLAGS.classes),logits=self.logits))
            self.loss_sum=tf.summary.scalar("loss",self.loss)
            # optimize
            self.optim_net = tf.train.AdamOptimizer(self.FLAGS.lr).minimize(self.loss)
            # accuracy
            self.acc = tf.reduce_sum(tf.cast(tf.equal(self.preds, self.lbl_bt), "float32")) / \
                       tf.cast(self.FLAGS.batch_size, "float32")
            self.acc_sum = tf.summary.scalar("acc", self.acc)
            self.net_sum = tf.summary.merge([self.loss_sum, self.acc_sum])
        else:
            self.cr_preds,self.cr_logits=self.critic(self.img_bt)
            self.img_g=tf.multiply(self.actor(),255)
            self.sample=tf.multiply(self.actor(is_reuse=True),255)
            self.cf_preds,self.cf_logits=self.critic(self.img_g,is_reuse=True)
            self.img_sum=tf.summary.image("img",self.img_g)

            self.cr_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.cr_logits),logits=self.cr_logits))
            self.cr_loss_sum=tf.summary.scalar("cr_loss",self.cr_loss)

            self.cf_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.cf_logits),logits=self.cf_logits))
            self.cf_loss_sum = tf.summary.scalar("cf_loss", self.cf_loss)

            self.c_loss = self.cr_loss + self.cf_loss
            self.c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

            self.a_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.cf_logits),logits=self.cf_logits))
            self.a_loss_sum = tf.summary.scalar("a_loss", self.a_loss)
            # trainable_variables
            var_list=tf.trainable_variables()
            self.net_vars=[var for var in var_list if "net_" in var.name]
            self.c_vars=[var for var in var_list if "c_" in var.name]
            self.a_vars=[var for var in var_list if "a_" in var.name]

            self.optim_c=tf.train.AdamOptimizer(self.FLAGS.lr).minimize(self.c_loss,var_list=self.c_vars)
            self.optim_a= tf.train.AdamOptimizer(self.FLAGS.lr).minimize(self.a_loss,var_list=self.a_vars)
            # summary
            # self.h1_sum=tf.summary.image("h1",self.h1[:,:,:,0])
            # self.h2_sum=tf.summary.image("h2",self.h2[:,:,:,0])
            self.c_sum=tf.summary.merge([self.c_loss_sum,self.cr_loss_sum,self.cf_loss_sum])
            self.a_sum=tf.summary.merge([self.a_loss_sum,self.img_sum])
        # writer
        self.writer=tf.summary.FileWriter("./logs",self.sess.graph)
        # saver
        self.saver = tf.train.Saver()
    def train(self):
        # checkpoint
        ckpt=tf.train.get_checkpoint_state(self.FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(self.FLAGS.checkpoint_dir,ckpt_name))
            print("Succeed loading checkpoints!")
        else:
            print("Failed to load checkpoints!")
            if not os.path.exists(self.FLAGS.checkpoint_dir):
                os.makedirs(self.FLAGS.checkpoint_dir)
            self.sess.run(tf.global_variables_initializer())
        # train data
        train_images=extract_images("train_images",mnist_path=self.FLAGS.mnist_path).astype("float32")
        train_labels=extract_labels("train_labels",mnist_path=self.FLAGS.mnist_path).astype("float32")
        test_images=extract_images("test_images",mnist_path=self.FLAGS.mnist_path).astype("float32")
        test_labels=extract_labels("test_labels",mnist_path=self.FLAGS.mnist_path).astype("float32")

        for epoch in range(self.FLAGS.epoch):
            batch_ixs=train_images.shape[0]//self.FLAGS.batch_size
            for ix in range(batch_ixs):
                batch_images=train_images[ix*self.FLAGS.batch_size:(ix+1)*self.FLAGS.batch_size]
                batch_labels=train_labels[ix*self.FLAGS.batch_size:(ix+1)*self.FLAGS.batch_size]
                global_step = epoch * batch_ixs + ix + 1
                # sess.run
                if not self.FLAGS.is_train:
                    _,net_sum,loss,acc=self.sess.run([self.optim_net,self.net_sum,self.loss,self.acc],
                                        feed_dict={self.img_bt:batch_images,self.lbl_bt:batch_labels})
                    self.writer.add_summary(net_sum,global_step=global_step)
                    print("epoch: {0},{1}/{2}, loss: {3:4f},acc: {4:4f}".format(epoch,ix,batch_ixs,loss,acc))
                    # each 100 step
                    if global_step%batch_ixs==0:
                        rand=(np.random.random(self.FLAGS.batch_size)*test_images.shape[0]).astype("int32").tolist()
                        batch_images=test_images[rand]
                        batch_labels=test_labels[rand]
                        all_sum, loss, acc = self.sess.run([self.net_sum, self.loss, self.acc],
                                            feed_dict={self.img_bt: batch_images, self.lbl_bt: batch_labels})
                        print("------------ valid loss: {0:4f},acc {1:4f} -------------".format(loss,acc))
                        self.saver.save(self.sess,self.FLAGS.checkpoint_dir+"/lenet_gan.model",global_step=global_step)
                else:
                    batch_noise=np.random.random(self.noise_dim)
                    _,c_sum,c_loss=self.sess.run([self.optim_c,self.c_sum,self.c_loss],
                            feed_dict={self.img_bt:batch_images,self.noise_bt:batch_noise})
                    _,a_sum,a_loss=self.sess.run([self.optim_a,self.a_sum,self.a_loss],feed_dict={self.noise_bt:batch_noise})
                    # _,a_sum,a_loss = self.sess.run([self.optim_a,self.a_sum,self.a_loss], feed_dict={self.noise_bt: batch_noise})
                    self.writer.add_summary(c_sum, global_step=global_step)
                    self.writer.add_summary(a_sum, global_step=global_step)
                    print("epoch: {0},{1}/{2}, c_loss: {3:4f},a_loss: {4:4f}".format(epoch, ix,batch_ixs,c_loss,a_loss))
                    # each 100 step
                    if global_step % batch_ixs == 0:
                        rand = (np.random.random(self.FLAGS.batch_size) * test_images.shape[0]).astype("int32").tolist()
                        batch_images = test_images[rand]
                        batch_labels = test_labels[rand]
                        c_sum, c_loss = self.sess.run([self.c_sum, self.c_loss],
                                                         feed_dict={self.img_bt: batch_images,self.noise_bt: batch_noise})
                        sample,a_sum, a_loss = self.sess.run([self.sample,self.a_sum, self.a_loss],feed_dict={self.noise_bt: batch_noise})
                        # a_sum, a_loss = self.sess.run([self.a_sum, self.a_loss],feed_dict={self.noise_bt: batch_noise})
                        print("[*] epoch: {0},{1}/{2}, c_loss: {3:4f},a_loss: {4:4f}".format(epoch, ix+1,batch_ixs, c_loss,a_loss))
                        self.saver.save(self.sess, self.FLAGS.checkpoint_dir + "/lenet-gan.model",global_step=global_step)
                        if not os.path.exists("samples"):
                            os.makedirs("samples")
                        save_sample(sample,"samples/"+str(epoch)+'_'+str(ix+1)+'_'+str(batch_ixs)+'.png')
    def predict(self):
        pass

    def net(self,inputs,is_reuse=False):
        with tf.variable_scope("net") as scope:
            if is_reuse:
                scope.reuse_variables()
            self.h1=conv2d(inputs,6,kernel_size=5,padding="SAME",scope="net_conv1_1") # 64*28*28*6
            h1=batch_norm(self.h1)
            h1=max_pool2d(h1,kernel_size=2,padding="SAME",scope="net_pool1") # 64*14*14*6
            self.h2=conv2d(h1,16,kernel_size=5,padding="VALID",scope="net_conv2_1") # 64*10*10*16
            h2=batch_norm(self.h2)
            h2=max_pool2d(h2,kernel_size=2,padding="SAME",scope="net_pool2")# 64*5*5*16
            h3=flatten(h2) # 64*400
            h4=fully_connected(h3,120,scope="net_fc1") # 64*120
            self.h5=fully_connected(h4,84,scope="net_fc2") # 64*84
            logits=fully_connected(self.h5,10,activation_fn=None,reuse=False,scope="net_out") # 64*10
            preds=tf.argmax(tf.sigmoid(logits),axis=1)
            return preds,logits

    def critic(self,inputs,is_reuse=False):
        with tf.variable_scope("critic"):
            self.net(inputs, is_reuse=is_reuse)
            logits=fully_connected(self.h5,1,activation_fn=None,reuse=is_reuse,scope="c_out")
            preds=tf.sigmoid(logits)
            return preds,logits

    def actor(self,is_reuse=False):
        with tf.variable_scope("actor") as scope:
            if is_reuse:
                scope.reuse_variables()
            in_shape=self.FLAGS.batch_size*2*2*16
            w=tf.get_variable("w",shape=[self.noise_dim,in_shape],dtype='float32',initializer=xavier_initializer())
            h1=bias_add(tf.matmul(tf.expand_dims(self.noise_bt,axis=0),w),scope="a_linear_1")
            h1=tf.reshape(h1,[self.FLAGS.batch_size,2,2,16])

            h1=conv2d_transpose(h1,8,kernel_size=3,stride=2,activation_fn=None,scope="a_convtp_1")
            h1 = tf.nn.relu(batch_norm(h1, scope="a_bn_1"))
            h2 = conv2d_transpose(h1, 4, kernel_size=3, stride=2,activation_fn=None,scope="a_convtp_2")
            h2 = tf.nn.relu(batch_norm(h2, scope="a_bn_2"))
            h3 = conv2d_transpose(h2, 2, kernel_size=3, stride=2,activation_fn=None,scope="a_convtp_3")
            h3 = tf.nn.relu(batch_norm(h3, scope="a_bn_3"))
            h4 = conv2d_transpose(h3, 1, kernel_size=3,stride=2,activation_fn=tf.nn.sigmoid, scope="a_convtp_4")
            h4 = h4[:,2:30,2:30,:]
            return h4


