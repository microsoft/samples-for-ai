import argparse
import os
import shutil
import sys
import tensorflow as tf

sys.path.insert(0, '.')
from stylenet import net
from utils import info, error, fail


def export(args):
    ckpt_dir = os.path.expanduser(args.ckpt_dir)
    export_dir = os.path.expanduser(args.export_dir)
    if os.path.exists(export_dir):
        info('Deleting the folder containing SavedModel at ' + export_dir)
        shutil.rmtree(export_dir)
    
    # Construct the serving graph
    batch_shape = (args.batch_size, args.height, args.width, 3)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape)
    preds = net(img_placeholder / 255.0)    
    saver = tf.train.Saver()
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    
    with tf.Session() as sess:
        # Restore the checkpoint
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            info('Restoring from ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            fail("No checkpoint found int " + ckpt_dir)
        
        # Write the SavedModel
        info('Exporting SavedModel to ' + export_dir)
        serving_signatures = {
            'Transfer': #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.saved_model.signature_def_utils.predict_signature_def(
                { tf.saved_model.signature_constants.PREDICT_INPUTS: img_placeholder },
                { tf.saved_model.signature_constants.PREDICT_OUTPUTS: preds }
            )
        }
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=serving_signatures,
                                             legacy_init_op=tf.saved_model.main_op.main_op(),
                                             clear_devices=True)
        builder.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='where the checkpoint is stored')
    parser.add_argument('--export_dir', type=str, default='export', help='where to write SavedModel')
    parser.add_argument('--height', type=int, default=240, help='image height')
    parser.add_argument('--width', type=str, default=320, help='image width')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    args, _ = parser.parse_known_args()
    export(args)