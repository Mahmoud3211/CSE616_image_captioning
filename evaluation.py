import os
import re
from tkinter import image_names
import numpy as np
import matplotlib.pyplot as plt
import pickle
from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization
from config import *
from data import *
from model import *
from tqdm import tqdm

from_disk = pickle.load(open("save_models/tv_layer.pkl", "rb"))
vectorization = TextVectorization.from_config(from_disk['config'])
vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorization.set_weights(from_disk['weights'])
vocab = vectorization.get_vocabulary()

index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
captions_mapping, text_data = load_captions_data("Flickr8k.token.txt")

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)

valid_images = list(valid_data.keys())

cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder,
)

checkpoint_ = tf.train.Checkpoint(model=caption_model)
status = checkpoint_.restore(tf.train.latest_checkpoint(checkpoint_path))

def calculate_single_blue(sample_img, real_captions):
    img = decode_and_resize(sample_img)
    img = img.numpy().clip(0, 255).astype(np.uint8)

    # Pass the image to the CNN
    img = tf.expand_dims(img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    # print("Predicted Caption: ", decoded_caption)
    real = []
    for real_cap in real_captions:
        real_cap = real_cap.replace("<start> ", "").replace(" <end>", "").strip()
        real.append(real_cap.split())
        # print("Real Caption: ", real_cap)

    single_bleu = sentence_bleu(real, decoded_caption.split())
    return single_bleu


def calculate_avrage_bleu(images, test=False):
    # Select a random image from the validation dataset
    total_bleu = []
    for sample_img in tqdm(list(images.keys())):
        # Read the image from the disk
        real_captions = images[sample_img]
        if not test:
            single = calculate_single_blue(sample_img, real_captions)
        else:
            real = []
            for real_cap in real_captions:
                real_cap = real_cap.replace("<start> ", "").replace(" <end>", "").strip()
                real_cap = real_cap.split()
                real.append(real_cap)

            single = sentence_bleu(real, real_cap.split())

        total_bleu.append(single)
    return np.mean(total_bleu)


print("Average Bleu Score for validation data is :", calculate_avrage_bleu(valid_data, False))