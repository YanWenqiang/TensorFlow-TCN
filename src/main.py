import argparse
import os
from model import TCN

parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="train")

parser.add_argument("--nb_filters", default = 100, help="Number of filters", type=int)
parser.add_argument("--kernel_size", default = 2, help="Size of the kernel",type=int)
parser.add_argument("--nb_stacks", default = 5, type=int)
parser.add_argument("--batch_size", default=2, help="Batch size for use in training", type=int)
parser.add_argument("--epochs", default=1, help="Number of epochs for training", type=int)
parser.add_argument("--num_classes",   default=2, help="Number of classes", type=int)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate for optimizer", type=float)
parser.add_argument("--dilations", default="1,2,4,8,16")
parser.add_argument("--activation", default="relu_norm")
parser.add_argument("--use_skip_connections", default=True, type=bool)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--return_sequences", default=False, type=bool)
parser.add_argument("--max_len", default=10, type=int)
parser.add_argument("--vocab_size", default=20000, type=int)
parser.add_argument("--embed_size", default=128, type=int)
parser.add_argument("--save_dir", default="./models")

parser.add_argument("--train_data", default="data/training", help="Directory for training data")
parser.add_argument("--val_data", default="data/validation", help="Directory for validation data")
parser.add_argument("--infer_data", default="data/infer")
parser.add_argument("--output_folder", default="data/output")
parser.add_argument("--checkpoint", default="./models/checkpoint", help="Directory for storing model checkpoints")



def main():
    FLAGS = parser.parse_args()
    tcn = TCN(FLAGS)

    if FLAGS.mode == "train":
        # train_path = FLAGS.train_data
        # valid_path = FLAGS.valid_data
        tx = [[1,2,3], [1,2,3], [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
        ty = [1,1,1,1,1,1,1,1,1]
        vx = [[1,2,3], [1,2,3], [1,2,3],[1,2,3], [1,2,3], [1,2,3],[1,2,3], [1,2,3], [1,2,3]]
        vy = [1,1,1,1,1,1,1,1,1]
        train_data = (tx, ty)
        valid_data = (vx, vy)
        tcn.train(train_data, valid_data)
    elif FLAGS.mode == "infer":
        # infer_data = []
        ix =  [[1,2,3], [1,2,3], [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
        ckpt = os.path.join(FLAGS.checkpoint, "model.ckpt")
        tcn.infer(ix)

if __name__ == "__main__":
    main()