# lenet_gan
	This code is a simple example combining LeNet and GAN.I made a test on LeNet with Gan, the rest work is ongoing, but this may help fresher of gan.

## requirement
    \>=tensorflow 1.0
    pillow
    scipy

## how to use
    flags.DEFINE_integer("epoch",1000,"epoch for train.")
    flags.DEFINE_integer("batch_size",64,"batch size.")
    flags.DEFINE_float("lr",0.0001,"learning rate.")
    flags.DEFINE_string("mnist_path","E:\dataset\mnist","path to mnist dataset.")
    flags.DEFINE_string("checkpoint_dir","checkpoint","checkpoint directory.")
    flags.DEFINE_boolean("is_train",True,"imcompleteness of 'False'!")
    flags.DEFINE_integer("classes",10,"classes to classify.")
    flags.DEFINE_integer("img_h",28,"image height.")
    flags.DEFINE_integer("img_w",28,"image width.")
    flags.DEFINE_integer("img_c",1,"image color.")

1.*you may need to download mnist dataset first [link](http://yann.lecun.com/exdb/mnist/),then modify the mnist_path*.

2.*run script: python3 main.py*
