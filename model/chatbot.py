from model import gnmt_model
from model import model_helper
from utils import nmt_utils
from utils import misc_utils as utils
import tensorflow as tf

from utils.data_utils import clean_text


class Bot():

    def __init__(self, data_dir, id_speaker, num_translations_per_input=1,
                 scope=None):
        self.data_dir = data_dir
        self.hparams = utils.load_hparams(self.data_dir)
        self.hparams.infer_batch_size = 1
        self.ckpt = tf.train.latest_checkpoint(data_dir)

        self.id = id_speaker
        self.num_translations_per_input = num_translations_per_input
        model_creator = gnmt_model.GNMTModel

        self.infer_model = model_helper.create_infer_model(model_creator,
                                                           self.hparams,
                                                           scope)

    def get_answer(self, question, id_spk):
        infer_data = [clean_text(question)]
        infer_model = self.infer_model
        with tf.Session(
                graph=infer_model.graph,
                config=utils.get_config_proto()) as sess:
            loaded_infer_model = model_helper.load_model(
                    infer_model.model, self.ckpt, sess, "infer")
            sess.run(
                    infer_model.iterator.initializer,
                    feed_dict={
                        infer_model.src_placeholder: infer_data,
                        infer_model.batch_size_placeholder: 1,
                        infer_model.src_speaker_placeholder: id_spk,
                        infer_model.tgt_speaker_placeholder: self.id
                        })

            nmt_outputs, _ = loaded_infer_model.decode(sess)
            translation = []
            for beam_id in range(self.num_translations_per_input):

                # Set set_id to 0 because batch_size of 1
                translation.append(nmt_utils.get_translation(
                        nmt_outputs=nmt_outputs[beam_id],
                        sent_id=0,
                        tgt_eos=self.hparams.eos,
                        subword_option=self.hparams.subword_option))

        return translation
