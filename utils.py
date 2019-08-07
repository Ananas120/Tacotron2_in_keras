# -*-config:utf-8 -* 

import numpy as np
import matplotlib.pyplot as plt

from threading import Thread

def time_to_string(secondes):
    secondes = int(secondes)
    h = secondes // 3600
    h = "" if h == 0 else "{}h ".format(h)
    m = (secondes % 3600) // 60
    m = "" if m == 0 else "{}min ".format(m)
    s = ((secondes % 300) % 60)
    return "{}{}{}s".format(h, m, s)    
    
def build_heatmap(fourrier, factor=8):
    cote = len(fourrier[0]) // factor
    image = np.ones((cote, cote, 3))
    longueur = cote // len(fourrier)
    for i in range(len(fourrier)):
        for j in range(longueur):
            image[:cote,i*longueur + j, 0] += fourrier[i,:cote]
            image[:cote,i*longueur + j, 1] -= fourrier[i,:cote]
            
    image[np.where(image > 1)] = 1
    image[np.where(image < 0)] = 0
    return image

def string_to_array(string):
    tab = []
    mot = ""
    for c in string:
        if (c >= '0' and c <= '9') or c == '.':
            mot += c
        else:
            if mot != "":
                tab.append(int(mot))
                mot = ""
    return np.array(tab)

def print_dict(dico, max_list_items=10):
    for k, v in dico.items():
        if type(v) is list:
            print("- {} \t\t= {} length = {}".format(k, v[:max_list_items], len(v)))
        else:
            print("- {} \t\t= {}".format(k, v))
        
def arrondir(nombre, nb_decimal=2):
    nombre = int(nombre * 10**nb_decimal)
    return nombre / 10 ** nb_decimal
        
def string_to_unicode(phrase):
    return np.array([ord(c) for c in phrase])

def string_to_code(phrase, vocab):
    code = []
    for c in phrase:
        try:
            code.append(vocab.at[c, 'numero'])
        except:
            pass
    return np.array(code)

def string_to_word_code(phrase, vocab, UKN_WORD='ukn'):
    code = []
    for mot in phrase.split(" "):
        if mot in vocab:
            code.append(vocab.at[mot, 'numero'])
        else:
            code.append(vocab.at[UKN_WORD, 'numero'])
    return np.array(code)

def unicode_to_string(code):
    return "".join([chr(c) for c in code])

def code_to_string(code, vocab, min_code=3):
    string = ""
    max_code = len(vocab) + min_code
    for c in code:
        if c >= min_code and c < max_code:
            string += vocab.at[c-min_code, 'caractere']
    return string

def word_code_to_string(code, vocab, min_code=3):
    string = []
    max_code = len(vocab) + min_code
    for c in code:
        if c >= min_code and c < max_code:
            string.append(vocab.at[c-min_code, 'mot'])
    return " ".join(string)


class Pthread(Thread):
    def __init__(self, fonction, *args, **kwargs):
        Thread.__init__(self)
        self.fonction = fonction
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        self.fonction(*self.args, **self.kwargs)
        
class DataWindow(object):
    def __init__(self, window_length=100, data=[]):
        self.window_length = window_length
        self.buffer = data
        
    def append(self, value):
        self.buffer.append(value)
            
    def mean(self):
        data_len = min(self.window_length, len(self.buffer))
        if data_len == 0: return float("inf")
        return np.mean(np.array(self.buffer[-data_len:]))
    
    def max(self):
        data_len = min(self.window_length, len(self.buffer))
        return np.max(np.array(self.buffer[-data_len:]))

    def min(self):
        data_len = min(self.window_length, len(self.buffer))
        return np.min(np.array(self.buffer[-data_len:]))
    
    def get_data(self):
        return self.buffer
