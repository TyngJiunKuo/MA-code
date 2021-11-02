import torch
import torch.utils.data.dataloader as dataloader
import numpy as np
import copy
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
import os
import json

try:
    import captum
except:
    os.system('pip3 install captum --user')
from captum.attr import Saliency, LayerIntegratedGradients, IntegratedGradients, GradientShap, LayerGradientShap, \
        DeepLift, TokenReferenceBase, visualization, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer, ShapleyValueSampling

import annoy
from annoy import AnnoyIndex

import stanza
# from nltk import wordnet

import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

try:
    import lemminflect
except:
    os.system('pip3 install lemminflect --user')
import lemminflect
from lemminflect import getLemma

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from scipy.stats import spearmanr
import random

#stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, ner', tokenize_pretokenized=True) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion, pad=1):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, length = batch.text
        if pad:
            predictions = model(text, length)
        else:
            predictions = model(text)
        loss = criterion(predictions, batch.label)
        acc = multi_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
def bert_train(model, train_data, optimizer, batch_size, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    iterator = iter(dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True))
    for batch in iterator:
        optimizer.zero_grad()
        text, label = batch[0].to(device), batch[1].to(device)
        predictions = model(text)
        #loss = criterion(predictions, label.unsqueeze(1).float())
        loss = criterion(predictions, label)
        acc = multi_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, pad=1):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, length = batch.text
            if pad:
                predictions = model(text, length).squeeze(1)
            else:
                predictions = model(text).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = multi_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
def bert_evaluate(model, eval_data, batch_size, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    iterator = iter(dataloader.DataLoader(eval_data, batch_size=batch_size, shuffle=True, pin_memory=True))
    with torch.no_grad():
        for batch in iterator:
            text, label = batch[0].to(device), batch[1].to(device)
            predictions = model(text)
#             loss = criterion(predictions, label.unsqueeze(1).float())
            loss = criterion(predictions, label)
            acc = multi_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def multi_accuracy(preds, y):
    preds_label = torch.argmax(preds, -1)
    correct = (preds_label == y).float()
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def best_index(valid_attack, which_criter): # find the index of the best perturbation
    # which_criter: 'top_correlation', 'pos_correlation, 'neg_correlation'
    correlation = [v[which_criter] for v in valid_attack]
    return np.argmin(correlation)


def intersect_ratio(orig, new):  # numpy array of index
    top_ratio = len(np.intersect1d(orig, new))/len(orig)  # how many original topk features are still in topk
    return top_ratio


def criteria(saliency1, saliency2):
    normal_s1 = normalize_feature_scores(saliency1)
    normal_s2 = normalize_feature_scores(saliency2)
    pos_1 = normal_s1 > 0
    pos1_index = np.where(pos_1 == True)[0]
    neg1_index = np.where(pos_1 == False)[0]
    #index_1 = np.argsort(normal_s1)

    pos_2 = normal_s2 > 0
    pos2_index = np.where(pos_2 == True)[0]
    neg2_index = np.where(pos_2 == False)[0]
    #index_2 = np.argsort(normal_s2)

    top_1 = np.argsort(np.abs(normal_s1))[-10:]
    top_2 = np.argsort(np.abs(normal_s2))[-10:]
    top_correlation = intersect_ratio(top_1, top_2)
    pos_correlation = intersect_ratio(pos1_index, pos2_index)
    neg_correlation = intersect_ratio(neg1_index, neg2_index)
    rank_correl, _ = spearmanr(normal_s1, normal_s2)     #here have to use distribution
    return rank_correl, top_correlation, pos_correlation, neg_correlation 
    
def generate_embedding(test_sample, NET, model_type):
    model = NET.model
    model.eval()
    # disable cudnn for rnn backpropogation
    torch.backends.cudnn.enabled = False
    if model_type == 'rnn':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_type == 'bert':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'bert.embeddings')
        
    if isinstance(test_sample, str):
        test_tensor = NET.inp_to_tensor(test_sample)
        input_emb = interpretable_emb.indices_to_embeddings(test_tensor)
    else:  # test_sample is an embedding instance.
        input_emb = test_sample
    if input_emb.shape != torch.Size([1, 1, 100]):
        input_emb = input_emb.sum(dim = 1)
    else:
        input_emb = input_emb.reshape(1, -1)
    #print(input_emb.shape)
    remove_interpretable_embedding_layer(model, interpretable_emb)
    return input_emb


def generate_saliency(test_sample, target, NET, model_type):
    model = NET.model
    model.eval()
    # disable cudnn for rnn backpropogation
    torch.backends.cudnn.enabled = False
    if model_type == 'rnn':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_type == 'bert':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'bert.embeddings')
        
    if isinstance(test_sample, str):
        test_tensor = NET.inp_to_tensor(test_sample)
        input_emb = interpretable_emb.indices_to_embeddings(test_tensor)
    else:  # test_sample is an embedding instance.
        input_emb = test_sample
    if NET.pad == 0:
        length_tensor = None
    else:
        length_tensor = torch.LongTensor([input_emb.shape[1]]).to(device)
    sa = Saliency(model)
    if model_type == 'rnn':
        attr_sa = sa.attribute(input_emb, target=target, additional_forward_args=length_tensor, abs=False).detach()  # for binary
    elif model_type == 'bert':
        attr_sa = sa.attribute(input_emb, target=target, abs=False).detach()
    #    remove_interpretable_embedding_layer(model, interpretable_emb)  # remove inerpreteable_embedding layer
    #except:
    remove_interpretable_embedding_layer(model, interpretable_emb)
    return attr_sa, input_emb


def generate_integrated_gradient(test_sample, target, NET, model_type):
    model = NET.model
    model.eval()
    torch.backends.cudnn.enabled = False
    
    if model_type == 'rnn':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_type == 'bert':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'bert.embeddings')
    
    if isinstance(test_sample, str):
        test_tensor = NET.inp_to_tensor(test_sample).to(device)
        input_emb = interpretable_emb.indices_to_embeddings(test_tensor)
    else:
        input_emb = test_sample
    if NET.pad==0:
        length_tensor = None
    else:
        length_tensor = torch.LongTensor([input_emb.shape[1]]).to(device)
    IG = IntegratedGradients(model)
    if model_type == 'rnn':
        attr_ig = IG.attribute(input_emb, target=target, additional_forward_args=length_tensor, return_convergence_delta=False).detach()
    elif model_type == 'bert':
        attr_ig = IG.attribute(input_emb, target=target, return_convergence_delta=False).detach()
    #except:
    remove_interpretable_embedding_layer(model, interpretable_emb)
        #return 0
    return attr_ig, input_emb


def generate_deep_lift(test_sample, target, NET, model_type):
    model = NET.model
    model.eval()
    torch.backends.cudnn.enabled = False
    # model.zero_grad()  # with this line, the model somehow was destroyed.
    #try:
    
    if model_type == 'rnn':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_type == 'bert':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'bert.embeddings')
        
    if isinstance(test_sample, str):
        test_tensor = NET.inp_to_tensor(test_sample)
        input_emb = interpretable_emb.indices_to_embeddings(test_tensor)
    else:
        input_emb = test_sample
    if NET.pad==0:
        length_tensor = None
    else:
        length_tensor = torch.LongTensor([input_emb.shape[1]]).to(device)
    DL = DeepLift(model)
    if model_type == 'rnn':
        attr_ig = DL.attribute(input_emb, target=target, additional_forward_args=length_tensor, return_convergence_delta=False).detach()
    elif model_type == 'bert':
        attr_ig = DL.attribute(input_emb, target=target, return_convergence_delta=False).detach()
    remove_interpretable_embedding_layer(model, interpretable_emb)
    #except:
    #    remove_interpretable_embedding_layer(model, interpretable_emb)
    return attr_ig, input_emb
    
    
def generate_shapley(test_sample, target, NET, model_type):
    model = NET.model
    model.eval()
    torch.backends.cudnn.enabled = False
    # model.zero_grad()  # with this line, the model somehow was destroyed.
    #try:
    
    if model_type == 'rnn':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_type == 'bert':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'bert.embeddings')
        
    if isinstance(test_sample, str):
        test_tensor = NET.inp_to_tensor(test_sample)
        input_emb = interpretable_emb.indices_to_embeddings(test_tensor).to(device)
    else:
        input_emb = test_sample
    if NET.pad==0:
        length_tensor = None
    else:
        length_tensor = torch.LongTensor([input_emb.shape[1]]).to(device)
    Shap = ShapleyValueSampling(model)
    feature_mask = torch.tensor(range(input_emb.shape[1])).repeat_interleave(input_emb.shape[2]).reshape(input_emb.shape[1], -1).to(device)
    
    if model_type == 'rnn':
        attr_shap = Shap.attribute(input_emb, target=target, additional_forward_args=length_tensor, feature_mask=feature_mask).detach()
    elif model_type == 'bert':
        attr_shap = Shap.attribute(input_emb, target=target, feature_mask=feature_mask).detach()
    remove_interpretable_embedding_layer(model, interpretable_emb)
    #except:
    #    remove_interpretable_embedding_layer(model, interpretable_emb)
    return attr_shap, input_emb



def get_saliency(NET, input, target, method):   # need to cover the attributions for perturbed embeddings
    if method == 'simple':
        saliency, inp_emb = generate_saliency(input, target, NET, model_type)
    elif method == 'integrated_gradient':
        saliency, inp_emb = generate_integrated_gradient(input, target, NET, model_type)
    elif method == 'deeplift':
        saliency, inp_emb = generate_deep_lift(input, target, NET, model_type)
    elif method == 'shapley':
        saliency, inp_emb = generate_shapley(input, target, NET, model_type)    
    else:
        raise ValueError('Not implemented other methods yet.')
    return saliency, inp_emb


def check_prediction(gt_label, test_sample, NET):
    score, label = NET.generate(test_sample)
    if label != gt_label:
        print('Model prediction is incorrect.')
        return 0
    else:
        conf = NET.confidence(score, label)
        return conf


def batch_loop(list_obj, bsz=64):
    for i in range(0, len(list_obj), bsz):
        yield np.array(list_obj[i: i+bsz])


def build_index_embedding(embedding, metric, save_dir):
    length, dimension = embedding.shape       # snetence length, embedding dim
    t = AnnoyIndex(dimension, 'euclidean')    # t is a (embedding dim) dimension vector with euclidean matric
    if os.path.isfile(save_dir):
        t.load(save_dir)
    else:
        for i in range(length):
            norm_emb = embedding[i]/np.linalg.norm(embedding[i])  #L2
            t.add_item(i, norm_emb)
        t.build(100)
        t.save(save_dir)
    return t


def neighbors(target_emb, embedding_index, k):   # both embedding and target_emb should be numpy.
    norm_emb = target_emb/np.linalg.norm(target_emb)
    index, distance = embedding_index.get_nns_by_vector(norm_emb, k, include_distances=True)
    return index, [1-d for d in distance]
    '''
    final = []
    for i in range(len(embedding)):
        similarity = torch.cosine_similarity(embedding[i].view(1, -1), target_emb).item()
        final.append(similarity)
    final_np = np.array(final)
    index_order = np.argsort(final_np)
    nearest_neighbor = index_order[-10:]
    return nearest_neighbor, final_np[nearest_neighbor]
    '''

def normalize_feature_scores(attribution, model = 'rnn', tokens = None):
    attribution_norm = attribution.sum(dim=2).squeeze(0)
    attribution_norm = attribution_norm / torch.norm(attribution_norm)
    attribution_norm = attribution_norm.cpu().detach().numpy()
    if model == 'bert':
        attribution_norm = merg_score(tokens, attribution_norm)
        return attribution_norm
    else:
        return attribution_norm


def percentage_feature_scores(attribution, model = None):
    attribution_norm = [float(i) / sum(np.abs(attribution)) for i in np.abs(attribution)]
    #print('attribution after percentage normalization {}'.format(attribution_norm))
    arg_attribution_norm = np.argsort(attribution_norm)[::-1]
    #print('arg sort index: {}'.format(arg_attribution_norm))
    return attribution_norm, arg_attribution_norm
    
    
def merg_subword(tokens):
    subwords_index = []
    subwords = []
    sub = False
    for i in range(len(tokens)):
        if (i < len(tokens) -1) and (tokens[i+1][0] == '#'):
            subwords.append(i)
            sub = True
        elif tokens[i][0] == '#':
            subwords.append(i)
            sub = True
        elif tokens[i][0] != '#':
            sub = False
        if (not sub) and subwords != []:
            subwords_index.append(subwords)
            subwords = []
    return subwords_index


def merg_score(tokens, importance):
    sub_id = 0
    score_id = 0
    score = []
    subwords_index = merg_subword(tokens)
    #print('len subwords_index: {}'.format(len(subwords_index)))
    if len(subwords_index) == 0:
        return importance
    else:
        while score_id < len(importance):
            if score_id in subwords_index[sub_id]:
                sum_score = sum(importance[i] for i in subwords_index[sub_id]) / len(subwords_index[sub_id])
                score_id += len(subwords_index[sub_id])
                if sub_id < len(subwords_index)-1:
                    sub_id += 1
                    #print('sub id: {}'.format(sub_id))
                score.append(sum_score)
            else:
                sum_score = importance[score_id]
                score_id += 1
                score.append(sum_score)
    return score


def all_jaccard_similarity(org, adv):
    sim1 = jaccard_similarity(org[0], adv[0])
    sim2 = jaccard_similarity(org[1], adv[1])
    sim3 = jaccard_similarity(org[2], adv[2])
    sim4 = jaccard_similarity(org[3], adv[3])
    sim5 = jaccard_similarity(org[4], adv[4])
    sim6 = jaccard_similarity(org[5], adv[5])
    sim7 = jaccard_similarity(org[6], adv[6])
    sim8 = jaccard_similarity(org[7], adv[7])
    sim9 = jaccard_similarity(org[8], adv[8])
    return [sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9]

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return round(len(s1.intersection(s2)) / len(s1.union(s2)), 3)


def word4jaccard(attr, attr_sort, token, model):
    if model == 'bert':
        attr_sort = list(attr_sort)
        attr_sort.remove(0)
        attr_sort.remove(max(attr_sort))
    jaccard_words1, _ = importent_words(attr, attr_sort, token, 0.1)
    jaccard_words2, _ = importent_words(attr, attr_sort, token, 0.2) 
    jaccard_words3, _ = importent_words(attr, attr_sort, token, 0.3) 
    jaccard_words4, _ = importent_words(attr, attr_sort, token, 0.4) 
    jaccard_words5, _ = importent_words(attr, attr_sort, token, 0.5) 
    jaccard_words6, _ = importent_words(attr, attr_sort, token, 0.6) 
    jaccard_words7, _ = importent_words(attr, attr_sort, token, 0.7) 
    jaccard_words8, _ = importent_words(attr, attr_sort, token, 0.8) 
    jaccard_words9, word_index = importent_words(attr, attr_sort, token, 0.9)
    return [jaccard_words1, jaccard_words2, jaccard_words3, jaccard_words4, jaccard_words5,\
                        jaccard_words6, jaccard_words7, jaccard_words8, jaccard_words9], word_index

def importent_words(norm, norm_argsort, tokens, p):
    percent = 0
    words = []
    words_index = []

    for index in norm_argsort:
        words_index.append(index)
        percent += norm[index]
        words.append(tokens[index])
        if (percent >= p) or (len(words) == len(norm_argsort)):
            return words, words_index

def check_token(sen, NET, model):
    if model == 'rnn':
        token = NET.TEXT.tokenize(sen)
        sentence = ' '.join(token)
    elif model == 'bert':
        token = NET.model.tokenizer.tokenize(sen)
        sentence = NET.model.tokenizer.convert_tokens_to_string(token)
    return sentence
    

def eraser4all(NET, token, label, score, conf, word_index, model = None):
    erasers = eraser(NET, token, label, score, conf, word_index, model)
    return erasers
'''
def eraser4all(NET, words, token, label, score, conf):
    eraser_01 = eraser(NET, words[0], token, label, score, conf)
    eraser_02 = eraser(NET, words[1], token, label, score, conf)
    eraser_03 = eraser(NET, words[2], token, label, score, conf)
    eraser_04 = eraser(NET, words[3], token, label, score, conf)
    eraser_05 = eraser(NET, words[4], token, label, score, conf)
    eraser_06 = eraser(NET, words[5], token, label, score, conf)
    eraser_07 = eraser(NET, words[6], token, label, score, conf)
    eraser_08 = eraser(NET, words[7], token, label, score, conf)
    eraser_09 = eraser(NET, words[8], token, label, score, conf)
    return [eraser_01, eraser_02, eraser_03, eraser_04, eraser_05, eraser_06, eraser_07, eraser_08, eraser_09]
'''

def eraser(NET, token, label, score, conf, word_index, model):
    SEN_SCORE = []
    SEN_LABEL = []
    RATE_SCORE = []
    #RATE_CONF = []
    
    add_token = []
    if model == 'bert':
        add_token = ['[CLS]'] + ['[PAD]'] * (len(token)-2) + ['[SEP]']
    else:
        add_token += ['<pad>'] * len(token)
    
    #text_org = ' '.join(add_token)
    #score_org, _ = NET.generate(text_org)
    #conf_org = NET.confidence(score_org, label)
    #score_org = score_org.squeeze(0).detach().cpu().numpy()[label]
    score = score.squeeze(0).detach().cpu().numpy()[label]
    SEN_LABEL.append(label)

    for w_index in word_index:
        add_token[w_index] = token[w_index]
        add_text = ' '.join(add_token)
        #print(add_text)
        
        score_add, label_add = NET.generate(add_text)
        #conf_add = NET.confidence(score_add, label)
        #rate_conf = abs(conf_add - conf_org)
        score_add = score_add.squeeze(0).detach().cpu().numpy()[label]
        rate_score = abs(score - score_add)
        
        SEN_LABEL.append(label_add)
        SEN_SCORE.append(score_add)
        RATE_SCORE.append(rate_score)
        #RATE_CONF.append(rate_conf)

        #score_org = score_add
        #conf_org = conf_add
        
    #SEN_CONF.append(conf)
    #score = score.squeeze(0).detach().cpu().numpy()[label]
    SEN_SCORE.append(score)
    return SEN_SCORE, SEN_LABEL, RATE_SCORE#, RATE_CONF
'''
def eraser(NET, word, token, label, score, conf, word_index):
    SEN_SCORE = []
    comprehensive_score = []
    SEN_CONF = []
    comprehensive_conf = []
    REMOVED_SCORE = []
    sufficiency_score = []
    REMOVED_CONF = []
    sufficiency_conf = []
    
    SEN_CONF.append(conf)
    REMOVED_CONF.append(conf)
    score = score.squeeze(0).detach().cpu().numpy()[label]
    SEN_SCORE.append(score)
    REMOVED_SCORE.append(score)

    rmv_token = copy.deepcopy(token)
    for w in word:
        rmv_token.remove(w)
        if rmv_token == []:
            rmv_token += ['<pad>']
        rmv_text = ' '.join(rmv_token)
        
        score_rmv, _ = NET.generate(rmv_text)
        conf_rmv = NET.confidence(score_rmv, label)
        SEN_CONF.append(conf_rmv)
        score_rmv = score_rmv.squeeze(0).detach().cpu().numpy()[label]
        SEN_SCORE.append(score_rmv)
        
        if comprehensive_score == []:
            com_score_rmv = score - score_rmv
        else:
            com_score_rmv = comprehensive_score[-1] - score_rmv
        comprehensive_score.append(com_score_rmv)
        if comprehensive_conf == []:
            com_conf_rmv = conf - conf_rmv
        else:
            com_conf_rmv = comprehensive_conf[-1] - conf_rmv
        comprehensive_conf.append(com_conf_rmv)
  
        score_suff, _ = NET.generate(w)
        conf_suff = NET.confidence(score_suff, label)
        REMOVED_CONF.append(conf_suff)
        score_suff = score_suff.squeeze(0).detach().cpu().numpy()[label]
        REMOVED_SCORE.append(score_suff)
        
        if sufficiency_score == []:
            suff = score - score_suff
        else:
            suff = sufficiency_score[-1] - score_suff
        sufficiency_score.append(suff)
        if sufficiency_conf == []:
            suff_conf = conf - conf_suff 
        else:
            suff_conf = sufficiency_conf[-1] - conf_suff
        sufficiency_conf.append(suff_conf)
        
    return SEN_SCORE, SEN_CONF, comprehensive_score, comprehensive_conf, REMOVED_SCORE, REMOVED_CONF, sufficiency_score, sufficiency_conf
    
    
def augmentation(org, adv):
    id = 0
    changed_id = []
    for o, a in zip(org.split(), adv.split()):
        if  o != a:
            changed_id.append(id)
            id += 1
        else:
            id += 1
    return changed_id
'''

def visualize_features(text, score, pred_label, gt_label,  attribution, delta=0, attribution_label='pos'):
    if not isinstance(text, list):
        raise ValueError('{} has to be tokenized items.'.format(text))
    attribution_norm = normalize_feature_scores(attribution)
    vis_data_record = []
    # score = torch.sigmoid(score).item()
    vis_data_record.append(visualization.VisualizationDataRecord(attribution_norm, score, pred_label, gt_label,
                                          attribution_label, attribution_norm.sum(), text, delta))
    visualization.visualize_text(vis_data_record)


def visualize_embedding(attribution, save_to_dir):
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Generate the values
    x_vals = range(attribution.shape[0])
    y_vals = range(attribution.shape[1])
    z_vals = attribution.reshape(-1).detach().numpy()
    X, Y = np.meshgrid(x_vals, y_vals)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    C = []
    for j in z_vals:
        if j > 0:
            C.append('red')
        elif j == 0:
            C.append('yellow')
        else:
            C.append('blue')
    p_3d = ax.scatter(X, Y, z_vals, c=C)
    # cbar=plt.colorbar(p_3d) 
    ax.set_xlabel('Words')
    ax.set_ylabel('Embedding dimension')
    ax.set_zlabel('Gradient value')

    plt.savefig(save_to_dir)
    plt.show()
    
def load_json(file):
    with open(file, 'r')as f:
        data = json.load(f)
    return data


class WordNet:
    def __init__(self):
        self.artikel = ['a', 'an', 'the']

    def ner(self, token_lists):  # [[word1, word2, word3,...]]
        # nlp = stanza.Pipeline(lang='en', processors='tokenize, ner', tokenize_pretokenized=True)
        tagged = nlp(token_lists)
        NER = tagged.entities
        NER_PERSON = [a.text for a in NER if a.type =='PERSON']
        NER_GPE = [a.text for a in NER if a.type =='GPE']    # city, country, etc.
        return NER_PERSON, NER_GPE

    def get_upos(self, token, tag):
        # nlp = stanza.Pipeline(lang='en', processors='tokenize, ner', tokenize_pretokenized=True)
        tagged = nlp(token)
        pos_word = [w.text for s in tagged.sentences for w in s.words if w.upos == tag]
        pos_id = [w.id-1 for s in tagged.sentences for w in s.words if w.upos == tag]
        return pos_word, pos_id
        
    def get_xpos(self, token, tag):
        # nlp = stanza.Pipeline(lang='en', processors='tokenize, ner', tokenize_pretokenized=True)
        tagged = nlp(token)
        pos_word = [w.text for s in tagged.sentences for w in s.words if w.xpos == tag]
        pos_id = [w.id-1 for s in tagged.sentences for w in s.words if w.xpos == tag]
        return pos_word, pos_id
        
    def verb_upos(self, token):
        # nlp = stanza.Pipeline(lang='en', processors='tokenize, ner', tokenize_pretokenized=True)
        #lemmatizer = WordNetLemmatizer()
        tagged = nlp(token)
        
        present_word = [w.text for s in tagged.sentences for w in s.words if (w.upos == 'VERB') and ((w.xpos == 'VB') or (w.xpos == 'VBG') \
                                                                                                     or (w.xpos == 'VBP') or ((w.xpos == 'VBZ')))]          
                                                              
        present_id = [w.id-1 for s in tagged.sentences for w in s.words if (w.upos == 'VERB') and ((w.xpos == 'VB') or (w.xpos == 'VBG') \
                                                                                                     or (w.xpos == 'VBP') or ((w.xpos == 'VBZ')))] 
                                                                       
        past_word = [w.text for s in tagged.sentences for w in s.words if (w.upos == 'VERB') and ((w.xpos == 'VBD') or (w.xpos == 'VBN'))]

        past_id = [w.id-1 for s in tagged.sentences for w in s.words if (w.upos == 'VERB') and ((w.xpos == 'VBD') or (w.xpos == 'VBN'))]
        return present_word, present_id, past_word, past_id
    
    def adj_upos(self, token):
        tagged = nlp(token)
        #adj_word = [w.text for s in tagged.sentences for w in s.words if w.xpos == 'JJ']
        #adj_id = [w.id-1 for s in tagged.sentences for w in s.words if w.xpos == 'JJ']
        
        #adj_r_word = [lemmatizer.lemmatize(w.text, 'a') for s in tagged.sentences for w in s.words if (w.upos == 'ADJ') and (w.xpos == 'JJR')]
        adj_word = []
        adj_id = []
        compar_word = []
        compar_word_id = []
        compar_more = [] #for adjective with more
        compar_more_id = []
        super_word = []
        super_word_id = []
        super_most = []
        super_most_id = []
        for s in tagged.sentences:
            for i in range(len(s.words)):
                #if i < len(s.words)-1:
                if (s.words[i].xpos == 'JJ') and (s.words[i-1].text != 'more') and (s.words[i-1].text != 'most'):
                    adj_word.append(s.words[i].text)
                    adj_id.append(s.words[i].id-1)
                elif (s.words[i].xpos == 'JJ') and (s.words[i-1].text == 'more'):
                    compar_more.append('most') # change more to most, cause in cpmarative -> simple adjective doesn't need to change words, only delete it
                    compar_more_id.append(s.words[i-1].id-1)
                elif (s.words[i].xpos == 'JJR') and (s.words[i].id not in compar_more_id):
                    compar_word.append(lemmatizer.lemmatize(s.words[i].text, 'a'))
                    compar_word_id.append(s.words[i].id-1)
                elif (s.words[i].xpos == 'JJ') and (s.words[i-1].text == 'most'):
                    super_most.append('more') # change most to more cause superlative will be only changed into comparative
                    super_most_id.append(s.words[i-1].id-1)
                elif (s.words[i].xpos == 'JJS') and (s.words[i].id not in super_most_id):
                    super_word.append(lemmatizer.lemmatize(s.words[i].text, 'a'))
                    super_word_id.append(s.words[i].id-1)
                #else:
                #    if s.words[i].xpos == 'JJ':
                #        compar_word.append(lemmatizer.lemmatize(s.words[i].text, 'a'))
                #        compar_word_id.append(s.words[i].id)
                #    elif s.words[i].xpos == 'JJS':
                #        super_word.append(lemmatizer.lemmatize(s.words[i].text, 'a'))
                #        super_word_id.append(s.words[i].id)
        #adj_r_id = [w.id-1 for s in tagged.sentences for w in s.words if (w.upos == 'ADJ') and (w.xpos == 'JJR')]
        
        #adv_s_word = [lemmatizer.lemmatize(w.text, 'a') for s in tagged.sentences for w in s.words if (w.upos == 'ADJ') and (w.xpos == 'JJS')]
        for i in range(len(super_word)):
            if super_word[i] == 'best':
                super_word[i] = 'good'
            elif super_word[i] == 'most':
                super_word[i] = 'many'
        for i in range(len(compar_word)):
            if compar_word[i] == 'more':
                compar_word[i] = 'many'
        #adj_s_id = [w.id-1 for s in tagged.sentences for w in s.words if (w.upos == 'ADJ') and (w.xpos == 'JJS')]
        return adj_word, adj_id, compar_word, compar_word_id, compar_more, compar_more_id, super_word, super_word_id, super_most, super_most_id

    #def adj(self, token_lists):
    #    nlp = stanza.Pipeline(lang='en', processors='tokenize, ner', tokenize_pretokenized=True)
    #    tagged = nlp(token_lists)
    #    adj = [w.text for s in tagged.sentences for w in s.words if w.upos == 'ADJ']
    #    return adj
    
    def synonym(self, token, pos):
        pos_synonym = []
        pos_syn = []
        pos_word, pos_id = self.get_upos(token, pos)

        if pos == 'ADJ':
            synset = [wordnet.synsets(w, 'a') for w in pos_word]
        elif pos == 'VERB':
            synset = [wordnet.synsets(w, 'v') for w in pos_word]
        elif pos == 'NOUN':
            synset = [wordnet.synsets(w, 'n') for w in pos_word]
        
        for syn, w in zip(synset, pos_word):
            temp = list(set([word for s in syn for word in s.lemma_names() if word != w]))
            pos_syn.append(temp)

        for synonym in pos_syn:
            temp_syn = []
            for syn in synonym:
                syn = syn.split('_')
                if len(syn) > 1:
                    syn = ' '.join(syn)
                    temp_syn.append(syn)
                else:
                    temp_syn.append(syn[0])
            pos_synonym.append(temp_syn)
        
            
        #for syn, w in zip(synset, pos_word):
        #    temp = list(set([syn for s in synset for syn in s.lemma_names() if syn != w]))
        #    
         #   temp = list(set([s.name().split('.')[0] for s in syn if s.name().split('.')[0] != w]))
         #   pos_synonym.append(temp)
        return pos_word, pos_synonym, pos_id
    
    #def synonym(self, token, pos, seed = 20):
       #synonyms = {}
       #for t in token:
       #    synset = wordnet.synsets(token, pos=pos)  # pos constrain
       #    synonym = list(set([syn for s in wn for syn in s.lemma_names() if syn != token]))
       #    synonyms[token] = synonym
       #random.seed(seed)
       #size = len(token)
       #adv_id = random.choice(range(size))
       #synset = wordnet.synsets(token[adv_id], pos=pos)  # pos constrain
       #synonyms = list(set([s.name().split('.')[0] for s in synset  if s.name().split('.')[0] != token[adv_id]]))
       #synonyms = list(set([syn for s in synset for syn in s.lemma_names() if syn != token[adv_id]]))
       #token.remove(token[adv_id])
       #return synonyms, token, adv_id

    #def synonym(self, token, pos):
    #   synset = wordnet.synsets(token, pos=pos)  # pos constrain
    #   synonyms = [s.lemma_names() for s in synset]
    #   return synonyms

    def antonym(self, token, pos, seed = 20):
       #synonyms = {}
       #for t in token:
       #    synset = wordnet.synsets(token, pos=pos)  # pos constrain
       #    synonym = list(set([syn for s in wn for syn in s.lemma_names() if syn != token]))
       #    synonyms[token] = synonym
       random.seed(seed)
       size = len(token)
       adv_id = random.choice(range(size))
       synset = wordnet.synsets(token[adv_id], pos=pos)  # pos constrain
       antonym = list(set([syn.antonyms()[0].name() for s in synset for syn in s.lemmas() if syn.antonyms()]))
       token.remove(token[adv_id])
       return synonyms, token, adv_id

    def synonym_antonym(self, token, pos, type, seed = 20):
        random.seed(seed)
        size = len(token)
        adv_id = random.choice(range(size))
        synset = wordnet.synsets(token[adv_id], pos=pos)  # pos constrain
        if type == 'synonym':
             synonym = list(set([syn for s in synset for syn in s.lemma_names() if syn != token[adv_id]]))
             token.remove(token[adv_id])
             return synonym, token, adv_id
        elif type == 'antonym':
            antonym = list(set([syn.antonyms()[0].name() for s in synset for syn in s.lemmas() if syn.antonyms()]))
            token.remove(token[adv_id])
            return antonym, token, adv_id


def get_upos(tokens):  # get part-of-speech tags
    sent = ' '.join(tokens[0])
    tagged = nlp(tokens)
    pos = [w.upos for s in tagged.sentences for w in s.words]
    entities = tagged.entities
    entities_index = []
    for e in entities:
        if e.type == 'PERSON':
            name = e.text
            s_id = sent[:e.start_char].count(' ')
            e_id = sent[e.start_char:e.end_char].count(' ')
            location = [s_id]
            while e_id:
                s_id += 1
                location += [s_id]
                e_id -= 1
            entities_index.append((name, location))

    return pos, entities_index





