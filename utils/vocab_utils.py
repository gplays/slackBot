# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from . import misc_utils as utils

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
    """Check if vocab_file doesn't exist, create from corpus_file."""
    if tf.gfile.Exists(vocab_file):
        utils.print_out("# Vocab file %s exists" % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            # Verify if the vocab starts with unk, sos, eos
            # If not, prepend those tokens & generate a new vocab file
            if not unk:
                unk = UNK
            if not sos:
                sos = SOS
            if not eos:
                eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                utils.print_out("The first 3 vocab words [%s, %s, %s]"
                                " are not [%s, %s, %s]" %
                                (vocab[0], vocab[1], vocab[2], unk, sos, eos))
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir,
                                              os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                        tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = new_vocab_file
    else:
        print(os.path.join(out_dir, os.path.basename(vocab_file)))
        print(vocab_file)
        raise ValueError("vocab_file '{}' does not exist.".format(vocab_file))

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
    src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
                tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table


def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:

    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding
      dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            if len(tokens) < 2 or (emb_size and emb_size == len(tokens)):
                print("BAD FORMATTED LINE: length = {}".format(len(tokens)))
            else:
                word = tokens[0]
                vec = list(map(float, tokens[1:]))
                emb_dict[word] = vec
                if emb_size:
                    assert emb_size == len(
                            vec), "All embedding size should be the same. " \
                                  "Embedding of size {} read, " \
                                  "expected {}.\n" \
                                  "Full line:\n" \
                                  "{}".format(len(vec), emb_size, line)
                else:
                    emb_size = len(vec)
    return emb_dict, emb_size


def build_vocab_file(questions, answers, vocab_file, out_dir, threshold=10):
    """

    :param questions:
    :type questions:
    :param answers:
    :type answers:
    :param vocab_file:
    :type vocab_file:
    :param out_dir:
    :type out_dir:
    :param threshold:
    :type threshold:
    :return:
    :rtype:
    """
    # Create a dictionary for the frequency of the vocabulary
    vocab = {}
    for question in questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Remove rare words from the vocabulary.
    # We will aim to replace fewer than 5% of words with <UNK>
    filtered_vocab = [word for word, count in vocab.items()
                      if count >= threshold]

    print("Size of total vocab:", len(vocab))
    print("Size of vocab we will use:", len(filtered_vocab))

    # Add the unique tokens to the vocabulary dictionaries.
    codes = [UNK, SOS, EOS]

    final_vocab = codes + [_ for _ in vocab]
    new_vocab_file = os.path.join(out_dir,
                                  os.path.basename(vocab_file))

    with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
        for word in final_vocab:
            f.write("%s\n" % word)

    return new_vocab_file


def build_speaker_file(src_speakers, tgt_speaker, speaker_file, out_dir):
    speakers = {}
    for speaker in src_speakers:
        if speaker[0] not in speakers:
            speakers[speaker[0]] = 1

    for speaker in tgt_speaker:
        if speaker[0] not in speakers:
            speakers[speaker[0]] = 1

    with open(out_dir + "/" + speaker_file, "w") as f:
        f.write("\n".join(speakers.keys()))
