from evaluation import *
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

assert calculate_avrage_bleu(train_data, True) == 1.0