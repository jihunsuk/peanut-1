import tensorflow as tf
import numpy as np
from . import Dialogue
from . import model
import os
import sys


class chatbot:
    # def kakao_input(self, inputs):
    #     self.inputs = inputs

    def __init__(self):
        self.dialogue = Dialogue.Dialogue()
        self.dialogue.load_vocab(os.path.dirname(os.path.abspath(__file__))+'/data/words.npy')

        self.model = model.Hred(self.dialogue.voc_size, self.dialogue.embedding_matrix, False, 1)
        self.sess = tf.Session()
        #self.inputs = ''
        # 모델 불러오기
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.abspath(__file__))+'/model')
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self, inputs):
        sentences = []

        sentences.append(inputs.strip())
        reply = self.get_replay(sentences)
        return reply

    def _decode(self, enc_inputs, dec_inputs):
        enc_len = []
        dec_len = []

        enc_batch = []
        dec_batch = []

        for i in range(0, len(enc_inputs)):
            enc_input = enc_inputs[i]
            if len(enc_input) > 25:
                enc_input = enc_inputs[i][0:25]
            dec_input = dec_inputs[i]
            if len(dec_input) > 24:
                dec_input = dec_inputs[i][0:24]

            enc, dec, _ = self.dialogue.transform(enc_input, dec_input, 25, 25)
            enc_batch.append(enc)
            dec_batch.append(dec)
            enc_len.append(len(enc_input))
            dec_len.append(len(dec_input)+1)

        context_size = len(enc_inputs)
        b = np.max(dec_len, 0)

        return self.model.predict(self.sess, [enc_batch], [enc_len], [dec_batch], [dec_len], [b], context_size)

    # msg에 대한 응답을 반환
    def get_replay(self, sentences):

        enc_input = [self.dialogue.tokens_to_ids(self.dialogue.tokenizer(sentence)) for sentence in sentences]
        dec_input = enc_input

        outputs = self._decode(enc_input, dec_input)
        reply = self.dialogue.decode([outputs[len(enc_input)-1]], True)
        reply = self.dialogue.cut_eos(reply)

        return reply
#
# vocab_path = './data/words.npy'
#
# def main(_):
#
#     print("깨어나는 중 입니다. 잠시만 기다려주세요...\n")
#
#     Chatbot = chatbot(vocab_path)
#     Chatbot.run()
#
# if __name__ == "__main__":
#     tf.app.run()
