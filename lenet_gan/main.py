import tensorflow as tf
import pprint
from lenet_gan import lenet_gan

flags=tf.app.flags
flags.DEFINE_integer("epoch",100,"epoch for train.")
flags.DEFINE_integer("batch_size",64,"batch size.")
flags.DEFINE_float("lr",0.0001,"learning rate.")
flags.DEFINE_string("mnist_path","E:\dataset\mnist","path to mnist dataset.")
flags.DEFINE_string("checkpoint_dir","checkpoint","checkpoint directory.")
flags.DEFINE_boolean("is_train",True,"imcompleteness of 'False'!")
flags.DEFINE_integer("classes",10,"classes to classify.")
flags.DEFINE_integer("img_h",28,"image height.")
flags.DEFINE_integer("img_w",28,"image width.")
flags.DEFINE_integer("img_c",1,"image color.")
FLAGS=flags.FLAGS

def main(_):
    # print parameters
    pp=pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    # train or test
    LG=lenet_gan(FLAGS)
    if FLAGS.is_train:
        LG.train()
    else:
        LG.test()
if __name__=="__main__":
    tf.app.run()
