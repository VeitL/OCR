import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data') 
parser.add_argument('--recurrent_conv', type=bool, default=True) 
parser.add_argument('--keep_prob', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--num_devices', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--lr_decay_rate', type=float, default=0.3)
parser.add_argument('--lr_decay_intv', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--log_interval', type=int, default=20) # print every # iterations
parser.add_argument('--checkpoint_interval', type=int, default=5) # save model every # epochs
parser.add_argument('--checkpoint_basedir', type=str, default='./outputs/') # output model folder
parser.add_argument('--checkpoint', type=str, default='./outputs/checkpoint-final.ckpt') # checkpoint name (for testing)

cfgs = parser.parse_args()
cfgs.train_dir = os.path.join(cfgs.data_dir, 'train')
cfgs.val_dir = os.path.join(cfgs.data_dir, 'val')
cfgs.test_dir = os.path.join(cfgs.data_dir, 'test')