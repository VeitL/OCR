import numpy as np 

def build_dictionary():
    # chars = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    char2id = dict({})
    id2char = dict({})
    ind = 0 
    for c in chars:
        if c not in char2id:
            char2id[c] = ind 
            id2char[ind] = c 
            ind += 1
    num_chars = len(char2id.keys())
    return char2id, id2char, num_chars

def label_to_token_ids(char2id, label):
    return [char2id[c] for c in label]

def token_ids_to_label(id2char, ids):
    return "".join([id2char[id_] if id_ in id2char else '' for id_ in ids])
