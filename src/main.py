import argparse

from model import TCN

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="infer")
parser.add_argument("--train_data", default="data/training",
                    help="Directory for training images")
parser.add_argument("--val_data", default="data/validation",
                    help="Directory for validation images")
parser.add_argument("--ckpt", default="models/model.ckpt",
                    help="Directory for storing model checkpoints")
parser.add_argument("--layers_per_block", default="2,3,3",
                    help="Number of layers in dense blocks")
parser.add_argument("--batch_size", default=8,
                    help="Batch size for use in training", type=int)
parser.add_argument("--epochs", default=5,
                    help="Number of epochs for training", type=int)
parser.add_argument("--num_threads", default=2,
                    help="Number of threads to use for data input pipeline", type=int)
parser.add_argument("--growth_k", default=16, help="Growth rate for Tiramisu", type=int)
parser.add_argument("--num_classes",   default=2, help="Number of classes", type=int)
parser.add_argument("--learning_rate", default=1e-4,
                    help="Learning rate for optimizer", type=float)
parser.add_argument("--infer_data", default="data/infer")
parser.add_argument("--output_folder", default="data/output")

def main():
    FLAGS = parser.parse_args()
    tcn = TCN(FLAGS)

    if FLAGS.mode == "train":
        tcn.train()
    elif FLAGS.mode == "infer":
        tcn.infer()

if __name__ == "__main__":
    main()