# -*- coding: utf-8 -*-
"""Baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Pnpt-LhAAQLv3C0a0jwS5pBQtR5OjtEd

# Base line model - Encoder-Decoder without Attention

Pre-processing steps are based on: https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

Also the model used is based on the seq2seq translation tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

The idea was to try to handle summarization as a translation problem.

Evaluation results and summaries were obtained using 10 observations. 

* BLEU-4 score for training: 0.57
* BLEU-4 score for validation: 0.46
"""

# Importing libraries
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import collections

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings

from bs4 import BeautifulSoup

!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

"""## Data Pre-processing

The data we are working with is amazon food reviews, it is publicly available in [kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews).

In order to have the data ready for the model, we followed the following steps:
* Drop duplicates and NA values
* Expand Contractions (Ex: aren't -> are not)
* Text to lowercase, remove html tags, remove ('s) 
* Remove text inside parenthesis, eliminate punctuations and special characters
* Remove stopwords, and remove short words.
"""

# Import Data
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
data = pd.read_csv("/content/drive/My Drive/reviews.csv", nrows=100000)

# Drop duplicates and NA values
data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)

# Expanded contractions
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

# Text cleaning
# lowercase, remove html tags, contraction mapping, remove ('s), remove text inside parenthesis,
# eliminate punctuations and special characters, remove stopwords, remove short words.
stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    new_string = text.lower()
    new_string = BeautifulSoup(new_string, "lxml").text
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"', '', new_string)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])    
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string) 
    tokens = [w for w in new_string.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t))

# Summary Cleaning
# lowercase, remove html tags, contraction mapping, remove ('s), remove text inside parenthesis,
# eliminate punctuations and special characters, remove stopwords, remove short words.
def summary_cleaner(text):
    new_string = re.sub('"','', text)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])    
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    new_string = new_string.lower()
    tokens=new_string.split()
    new_string=''
    for i in tokens:
        if len(i)>1:                                 
            new_string=new_string + i + ' '  
    return new_string.strip()

cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(summary_cleaner(t))

# Cleaning text and summary columns
data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

# Assigning pandas series to variables
text = data['cleaned_text']
summary = data['cleaned_summary']

"""### Data Splitting

Our data will be splitted using 90% train data and 10% validation data.
"""

## Data Splitting
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(text, summary, test_size=0.1, 
                                                  random_state=0, shuffle=True)

"""We need to create a vocabulary for the text (review) and the summary. 
Here, they are treated as if each one of them were a Language. Both of them will have their own vocabulary 
and word to index dictionary. Also, we include a Start of Sentence (SOS), End of Sentence (EOS), 
and unknown (UNK) tokens in each language. The UNK token is necessary specially for our validation data 
because there are words that are not in our vocabulary."""

UNK_token = 0
SOS_token = 1
EOS_token = 2

# Creates a vocabulary for a language. word to index, index to word,
# and word frequency dictionaries are created.
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<UNK>":0}
        self.word2count = {}
        self.index2word = {0: "<UNK>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3  # Count SOS, EOS, and UNK

    def addSentence(self, sentence):
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Returns an instance of the Lang class for each of the text and summary
# provided. Also, a list of text-summary pairs is returned.
def readLangs(lang1, lang2):
    pairs = list(zip(lang1, lang2))
    input_lang = Lang('text')
    output_lang = Lang('summary')

    return input_lang, output_lang, pairs

# We filter each pair of text-summary to a certain length
text_max_len = 80
summary_max_len = 10

def filterPair(p):
    return len(p[0].split(' ')) < text_max_len and \
        len(p[1].split(' ')) < summary_max_len


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Prepares data by calling previous functions.
# Reads data, filters it, and creates a vocabulary.
def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData(x_train, y_train)
print(random.choice(pairs))

pairs[1]

val_pairs = list(zip(x_val, y_val))
print("Read %s sentence pairs" % len(val_pairs))
val_pairs = filterPairs(val_pairs)
print("Trimmed to %s sentence pairs" % len(val_pairs))

val_pairs[0]

# Vectorizing the data and converting each sentence into a tensor
# To train for each pair, we need an input tensor and a target tensor

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, lang.word2index['<UNK>']) for word in word_tokenize(sentence)]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

"""## Encoder"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""## Decoder"""

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""#### Model training

Training process:
"""

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion, max_length=text_max_len):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach() # detach from history as input
        
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Estimate of time left in the training process. 

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0    # Reset every print_every
    plot_loss_total = 0     # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor[:text_max_len], target_tensor[:summary_max_len], 
                     encoder, decoder, encoder_optimizer, decoder_optimizer, 
                     criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

"""### Evaluation

The evaluation process is similar to the training process
"""

def evaluate(encoder, decoder, sentence, max_length=text_max_len):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(summary_max_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# Generates a summary and compares it with the target summary in the data set
def evaluateRandomly(encoder, decoder, pairs=pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

from nltk.translate.bleu_score import corpus_bleu

# Calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4 score for a number of a dataset 
# observations.
def evaluate_bleu(encoder, decoder, pairs=pairs, n=10):
    """To speed up testing, we only evaluate BLEU score on n test sentences."""
    references = []
    predictions = []
    for pair in pairs[:n]:
        references.append([pair[1].split(' ')])
        output_words = evaluate(encoder, decoder, pair[0][:text_max_len])
        predictions.append(output_words)
    blue_1 = corpus_bleu(references, predictions, weights=(1, 0, 0, 0))
    blue_2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0))
    blue_3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0))
    blue_4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1 score:', blue_1)
    print('BLEU-2 score:', blue_2)
    print('BLEU-3 score:', blue_3)
    print('BLEU-4 score:', blue_4)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

# TO speed up training, you can reduce 75000 to 5000
trainIters(encoder=encoder1, decoder=decoder1, n_iters=75000, print_every=5000)

"""### Training set evaluation"""

evaluateRandomly(encoder1, decoder1) # n=10
evaluate_bleu(encoder1, decoder1) # n=10

evaluate_bleu(encoder1, decoder1, n=len(pairs)) # whole train set

"""### Validation set evaluation"""

evaluateRandomly(encoder1, decoder1, val_pairs) # n=10
evaluate_bleu(encoder1, decoder1, val_pairs) # n=10

evaluate_bleu(encoder1, decoder1, val_pairs, n=len(val_pairs)) # whole val set