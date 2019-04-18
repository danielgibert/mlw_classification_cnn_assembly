import numpy as np
import json

def parse_labels(labels_batch, num_classes):
    """
    Convert integers to one-hot vectors

    Parameters
    ----------
    labels_batch: list
        Batch of labels
    num_classes: int
        Number of classes/families

    Return
    ------
    y_batch: list of one-hot vectors
    """
    y_batch = []
    for label in labels_batch:
        y = np.zeros(num_classes)
        y[label] = 1
        y_batch.append(y)
    return y_batch


def parse_inputs(input_x, vocabulary_mapping):
    if type(input_x) is str:
        program = [vocabulary_mapping[word] for word in input_x.split(",")]
        return np.array(program, dtype=np.int32)
    else:
        x_batch = []
        for X in input_x:
            program = [vocabulary_mapping[word] for word in X.split(",")]
            x_batch.append(program)
        return np.array(x_batch, dtype=np.int32)

def load_embeddings(embeddings_filepath, vocabulary_mapping, embedding_size=4):
    embeddings_2darray = np.zeros((len(vocabulary_mapping.keys()), embedding_size))
    with open(embeddings_filepath, "r") as embeddings_file:
        embeddings_data = embeddings_file.readlines()
        for line in embeddings_data:
            line = line.strip()
            line = line.split(" ")
            word = line[0]
            word_embedding = [float(e) for e in line[1:]]
            word_ID = vocabulary_mapping[word]
            embeddings_2darray[word_ID] = word_embedding
    return embeddings_2darray

def load_vocabulary(vocabulary_filepath):
    """
    It reads and stores in a dictionary-like structure the data from the file passed as argument

    Parameters
    ----------
    vocabulary_filepath: str
        JSON-like file

    Return
    ------
    vocabulary_dict: dict
    """
    with open(vocabulary_filepath, "r") as vocab_file:
        vocabulary_dict = json.load(vocab_file)
    return vocabulary_dict

def store_metadata(output_file, inverse_vocabulary_dict):
    """
    Stores the metadata file used for the embeddings' labels

    Parameters
    ----------
    output_file: str
        Filename
    inverse_vocabulary_dict: dict
        Dictionary-like structure containing the mapping between opcodes and their corresponding assignments

    """
    with open(output_file, "w") as metadata_file:
        for key in sorted(inverse_vocabulary_dict):
            metadata_file.write("{}\n".format(inverse_vocabulary_dict[key]))


def get_GPU_index(GPU_indices=[0,1], current_index=None):
    if current_index == None:
        return 0
    else:
        if (current_index+1) == len(GPU_indices):
            return 0
        else:
            current_index += 1
            return current_index