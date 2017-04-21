import numpy as np
import json
import random


def extract_chars_from_end(char_seq, max_seq_len_cutoff):
    if len(char_seq) > max_seq_len_cutoff:
        char_seq = char_seq[-max_seq_len_cutoff:]
    return char_seq

def pad_sentence(char_seq, max_seq_len_cutoff, padding_char=" "):
    num_padding = max_seq_len_cutoff - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq

def string_to_int_conversion(char_seq, alphabet):
    char_list = []
    for char in char_seq:
        char_list.append(alphabet.find(char))

    x = np.asarray(char_list)
    return x


def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index, max_seq_len_cutoff):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch_text = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]

    x_batch = []
    for text in x_batch_text:
        text_end_extracted = extract_chars_from_end(list(text.lower()), max_seq_len_cutoff)
        padded = pad_sentence(text_end_extracted, max_seq_len_cutoff=max_seq_len_cutoff)
        text_to_int = string_to_int_conversion(padded, alphabet)
        x_batch.append(text_to_int)
    x_batch = np.asarray(x_batch, dtype=np.int8)


    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data(dataset_path, n_classes):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

    dataset_file = open(dataset_path, "r", encoding='utf-8')
    dataset_content = dataset_file.readlines()

    x = []
    y = []
    for element in dataset_content:
        element = element.lower()
        element = element.split("\t")
        label = int(element[0])
        text = element[1].strip()
        if (len(text) == 0):
            continue


        x.append(text)
        tmp_lable = np.zeros(n_classes)
        if(n_classes == 2):
            tmp_lable[label] = 1
        else:
            tmp_lable[label - 1] = 1
        y.append(tmp_lable)


    y = np.array(y, dtype=np.int8)
    return [x, y]


def batch_iter(data, batch_size, max_seq_len_cutoff, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        random.shuffle(data)

    x_shuffled, y_shuffled = zip(*data)


    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index, max_seq_len_cutoff=max_seq_len_cutoff)
        batch = list(zip(x_batch, y_batch))
        yield batch
