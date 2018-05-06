import argparse
import os
import shutil
import subprocess as sp
import sys
import tensorflow as tf
import time

sys.path.insert(0, '.')
from stylenet import net


json_tempalte = """{
  "name": "%s",
  "version": 1,
  "description": "Add styles from famous paintings to any photo in a fraction of a second.",
  "srcPath": "%s",
  "destPath": "%s",
  "interfaces": [
    {
      "name": "Transfer",
      "inputs": [
        {
          "name": "input",
          "internalName": "%s",
          "description": "Raw image."
        }
      ],
      "outputs": [
        {
          "name": "output",
          "internalName": "%s",
          "description": "New image."
        }
      ]
    }
  ]
}
"""


def normalize(model_name):
    return ''.join(model_name.split())


def export(args):
    batch_shape = (args.batch_size, args.height, args.width, 3)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
    preds = net(img_placeholder)
    model_dir = args.checkpoint
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.isdir(args.checkpoint):
            ckpt = tf.train.get_checkpoint_state(args.checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, args.checkpoint)
            model_dir = os.path.dirname(args.checkpoint)

        serving_dir = os.path.abspath(model_dir.rstrip('\\/') + '.serving')
        if os.path.exists(serving_dir):
            shutil.rmtree(serving_dir)
            time.sleep(1)
        os.makedirs(serving_dir)
        saver.save(sess, os.path.join(serving_dir, 'fns.ckpt'))
    
    transfer_name = normalize(args.name)
    json_path = os.path.join(serving_dir, 'export.json')
    export_dir = serving_dir.rstrip('\\/') + '.export'
    if os.path.exists(export_dir): shutil.rmtree(export_dir)
    with open(json_path, 'w') as fout:
        fout.write(json_tempalte % (transfer_name, repr(serving_dir).strip('\'\"'), repr(export_dir).strip('\'\"'), img_placeholder.name, preds.name))
    
    process = sp.Popen(['sonoma_tf_export', json_path], shell=True)
    process.wait()
    if process.returncode != 0:
        raise Exception('Fail to run sonoma_tf_export! Either no Sonoma Python package is installed or %s is of wrong format.' % json_path);


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='dir or .ckpt file to load checkpoint from')
    parser.add_argument('--name', type=str, default='StyleTransferNetwork',
                        help='model name used in Sonoma')
    parser.add_argument('--height', type=int, default=240,
                        help='image height')
    parser.add_argument('--width', type=str, default=320,
                        help='image width')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')
    args, _ = parser.parse_known_args()
    export(args)