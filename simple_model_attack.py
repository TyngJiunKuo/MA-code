import utils
import torch
import numpy as np
import copy
import random
import os
import json
import pickle
import datasets_helper, models

from utils import best_index, generate_saliency, generate_integrated_gradient, generate_deep_lift,\
    intersect_ratio, visualize_features, normalize_feature_scores, jaccard_similarity, importent_words,\
    generate_shapley, word4jaccard, eraser4all, all_jaccard_similarity, percentage_feature_scores
from tqdm import tqdm
from collections import Counter

from textattack.augmentation import EmbeddingAugmenter, CheckListAugmenter

# noun singular <=> plural
from pattern.text.en import pluralize, singularize
# adj type normal <=> comparative, comparative <=> superlative
from pattern.text.en import comparative, superlative
# verb tense present <=> past, past <=> present
from pattern.text.en import conjugate, INFINITIVE, PRESENT, PAST, FUTURE, SG, PL

import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('opinion_lexicon')
from nltk.corpus import opinion_lexicon
pos = opinion_lexicon.positive()
neg = opinion_lexicon.negative()
opinion_tokens = pos + neg

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ManualAttack:
    def __init__(self, test_sample, gt_label, NET, method = 'intergrated gradient', model_type = 'rnn'):
        self.test_sample = test_sample
        self.gt_label = gt_label        #sentence label
        self.NET = NET       # here NET is only a simple model with sum_embedding plus Linear layer.
        self.feature_method = method
        self.model_type = model_type
        self.init_dense_weights()
        self.saliency1, _ = self.get_saliency(NET, self.test_sample, gt_label, method)

    def init_dense_weights(self):
        #self.dense_params = list(self.NET.model.parameters())[-2]    #linear fc parameter, output_dim = 2
        #self.class_vector = self.dense_params[self.gt_label, :].detach().cpu().numpy() # get parameters for labe 1 or 0
        #self.objective = copy.deepcopy(self.class_vector)
        #self.embedding = self.NET.model.embedding(self.NET.inp_to_tensor(self.test_sample)).detach().cpu().numpy()
        #self.emb_weights = copy.deepcopy(self.NET.model.embedding.weight.detach().cpu().numpy())  #vocab's embedding(25002)
        #self.embedding_index = utils.build_index_embedding(self.emb_weights, 'cosine', '/home/kuo/code/data/simple_embedding_euclidean_index.ann') 
        #self.emb_ids  = self.NET.TEXT.vocab.stoi 
        
        if self.model_type == 'rnn':
            self.tokens = self.NET.TEXT.tokenize
        elif self.model_type == 'bert':
            self.tokens = self.NET.model.tokenizer.tokenize
            #self.length = len(self.tokens(self.test_sample))
        return self.tokens#, self.objective, self.embedding, self.emb_weights, self.emb_ids, 

    def more_objective(self, magnitude):
        # move magnitude step towards the direction of the class vector, resulting to higher confidence score.
        direction = np.sign(self.class_vector)
        self.objective = direction * magnitude   #magnitude??

    def get_hidden(self, NET, text):
        text = self.NET.inp_to_tensor(text)
        size = len(text)
        self.NET.model.set_hook()
        self.NET.model(text, [torch.tensor(size)])
        hidden = self.NET.model.get_text_hidden()
        self.NET.model.remove_hook()
        return hidden

    def traverse_all_vocabs(self, tolerance):
        sum_emb = np.sum(self.embedding, 1).squeeze(0) 
        sum_adv = sum_emb.copy()
        obj = self.objective.copy()
        vocabs = self.emb_weights.copy()
        size = vocabs.shape[0]   #25002
        print('vocabulary size : {}'.format(size))
        words = []
        for i in range(tolerance):
            similarities = []
            for j in range(size):
                embedding = vocabs[j]
                sum_try = np.add(sum_adv, embedding)  
                similarity = np.inner(obj, sum_try) / (
                            np.linalg.norm(obj) * np.linalg.norm(sum_try))
                similarities.append(similarity)

            best_id = np.argmax(similarities)
            score = similarities[best_id]
            sum_adv = np.add(sum_adv, vocabs[best_id])
            print('best score {}'.format(score))
            word = self.NET.TEXT.vocab.itos[best_id]
            words.append(word)
            print('added word: {}'.format(word))
            print('\n\n')
        return words

    def add_words(self, tolerance):
        sum_emb = np.sum(self.embedding, 1).squeeze(0)  # (100, )
        obj = self.objective.copy()
        # tensor_obj = torch.from_numpy(obj).unsqueeze(0)
        words = []
        for i in range(tolerance):
            delta = obj - sum_emb  # without normalization
            # delta = obj - sum_emb/(length+i)
            k_neighbors, scores = utils.neighbors(delta.reshape(-1), self.embedding_index, 10000)
            similarities = []
            for neighbor in k_neighbors:
                neighbor_emb = self.emb_weights[neighbor]
                sum_adv = np.add(sum_emb, neighbor_emb)
                # tensor_adv = torch.from_numpy(sum_adv).unsqueeze(0)
                similarity = np.inner(sum_adv.reshape(-1), obj.reshape(-1))/\
                             (np.linalg.norm(sum_adv.reshape(-1)) * np.linalg.norm(obj.reshape(-1)))
                # similarity = torch.cosine_similarity(tensor_obj, tensor_adv).numpy()
                similarities.append(similarity)

            best_id = np.argmax(similarities)
            word = self.NET.TEXT.vocab.itos[k_neighbors[best_id]]
            # if word not in opinion_tokens:
            words.append(word)
            # else:
                # continue
            sum_emb = np.add(sum_emb, self.emb_weights[k_neighbors[best_id]])
            # print('added word: {}'.format(word))
            # print('cosine similarity between sentences and class {}'.format(similarities[best_id]))
        return [w for w in words if w not in opinion_tokens]

    def replace(self, text, words, seed=20):
        text_nlp = utils.nlp(text)
        # tokens = text.split()
        adv_tokens = self.tokens.copy()
        visited = []
        for word in words:
            word_nlp = utils.nlp([[word]])
            word_pos = word_nlp.sentences[0].words[0].upos
            text_pos = [p.text for s in text_nlp.sentences for p in s.words if p.upos == word_pos]
            text_pos = [p for p in text_pos if p not in visited]
            if text_pos:
                random.seed(seed)
                candi_word = random.choice(text_pos)
                candi_id = self.tokens.index(candi_word)
                visited.append(candi_word)
                adv_tokens[candi_id] = word
            else:  # not enough candidates to perturb
                return 0
        return ' '.join(adv_tokens)

    def feature_diff(self, adv_text): # feature_diff(self, adv_text)
        saliency_adv, _ = self.get_saliency(self.NET, adv_text, self.gt_label, self.feature_method)
        return utils.criteria(self.saliency1, saliency_adv)

    def get_saliency(self, NET, input, target, method):  # need to cover the attributions for perturbed embeddings
        if method == 'simple':
            saliency, inp_emb = generate_saliency(input, target, NET, self.model_type)
        elif method == 'integrated_gradient':
            saliency, inp_emb = generate_integrated_gradient(input, target, NET, self.model_type)
        elif method == 'deeplift':
            saliency, inp_emb = generate_deep_lift(input, target, NET, self.model_type)
        elif method == 'shapley':
            saliency, inp_emb = generate_shapley(input, target, NET, self.model_type)
        else:
            raise ValueError('Not implemented other methods yet.')
        return saliency, inp_emb

    def visualize_perturb(self, adv_input, feature_method, adv_label, conf):
        # score, label = self.NET.generate(adv_input)
        # conf = self.NET.confidence(score, label)
        saliency, _ = self.get_saliency(self.NET, adv_input, self.gt_label, feature_method)
        #visualize_features(self.tokens(adv_input), conf, label, label, saliency)
        return self.tokens(adv_input), conf, adv_label, self.gt_label, saliency.cpu()

    def visualize_orig(self, feature_method, conf):
       #  score, label = self.NET.generate(self.test_sample)
        # conf = self.NET.confidence(score, label)
        saliency, _ = self.get_saliency(self.NET, self.test_sample, self.gt_label, feature_method)
        #visualize_features(self.tokens(self.test_sample), conf, label, self.gt_label, saliency)
        return self.tokens(self.test_sample), conf, self.gt_label, self.gt_label, saliency.cpu()
    


class ArtikelAttack:
    def __init__(self, test_sample, NET, model):
        self.artikel = utils.WordNet().artikel
        self.NET = NET
        self.test_sample = test_sample
        if model == 'rnn':
            self.tokens = self.NET.TEXT.tokenize
        elif model == 'bert':
            self.tokens = self.NET.model.tokenizer.tokenize

    def change_artikel(self, artikel_id, adv_tokens, index, seed = 20):
        random.seed(seed)
        adv_artikel = adv_tokens[artikel_id[index]]
        if (adv_artikel == "a") or (adv_artikel == "an"):
            adv_tokens[artikel_id[index]] = "the"
        else:
            adv_tokens[artikel_id[index]] = random.choice(["a", "an"])
        return adv_tokens

    def get_artikel_id(self, adv_tokens):
         artikel = self.artikel
         artikel_id =  [i for i in range(len(adv_tokens)) for a in artikel if adv_tokens[i] == a]
         return artikel_id

class EntityAttack(ManualAttack):
    def __init__(self, test_sample, gt_label, NET, model_type):
        super(ManualAttack, self).__init__()

        self.test_sample = test_sample
        self.gt_label = gt_label
        self.NET = NET
        self.model_type = model_type

        self.init_variable()

    def init_variable(self):
        #self.objective, self.embedding, self.emb_weights, self.emb_ids, 
        self.tokens = self.init_dense_weights()
    
        
    def text_change(self, n_word, n_id, text):
        #sen = text
        text[n_id] = n_word
        return text

# for adjective        
    def type_exchange(self):
        n2r, n2r_id, r2s, r2s_id, more2most, more2most_id, s2r, s2r_id, most2more, most2more_id = utils.WordNet().adj_upos(self.test_sample)
        r2n = copy.deepcopy(r2s)
        r2n_id = copy.deepcopy(r2s_id)
        more2none_id = copy.deepcopy(more2most_id)
        n2r = [comparative(w) for w in n2r]
        s2r = [comparative(w) for w in s2r]
        r2s = [superlative(w) for w in r2s]
        none_index_n2r = []
        none_index_r2s = []
        for i in range(len(n2r)):
            if n2r[i] == 'possibler':
                n2r[i] = 'more possible'
            if n2r[i] == 'firster':
                none_index_n2r.append(i)
            if n2r[i] == 'oker':
                n2r[i] = 'more ok'
            if n2r[i] == 'watchabler':
                n2r[i] = 'more watchable'
            if n2r[i] == 'entirer':
                n2r[i] = 'more entire'
            if n2r[i] == 'elser':
                none_index_n2r.append(i)
            if n2r[i] == 'enjoyabler':
                n2r[i] = 'more enjoyable'
            if n2r[i] == 'more second':
                none_index_n2r.append(i)
            if n2r[i] == 'more average':
                none_index_n2r.append(i)
            if n2r[i] == 'otherer':
                none_index_n2r.append(i)
            if n2r[i] == 'samer':
                none_index_n2r.append(i)
            if n2r[i] == 'wholer':
                none_index_n2r.append(i)
            if n2r[i] == 'more [CLS]':
                none_index_n2r.append(i)

        for i in range(len(r2s)):
            if r2s[i] == 'possiblest':
                r2s[i] = 'most possible'
            if r2s[i] == 'firstest':
                none_index_r2s.append(i)
            if r2s[i] == 'okest':
                r2s[i] = 'most ok'
            if r2s[i] == 'watchablest':
                r2s[i] = 'most watchable'
            if r2s[i] == 'entirest':
                r2s[i] = 'most entire'
            if r2s[i] == 'elsest':
                none_index_r2s.append(i)
            if r2s[i] == 'enjoyablest':
                r2s[i] = 'most enjoyable'
            if r2s[i] == 'most second':
                none_index_r2s.append(i)                  
            if r2s[i] == 'most average':
                none_index_r2s.append(i)
            if r2s[i] == 'otherest':
                none_index_r2s.append(i)
            if r2s[i] == 'samest':
                none_index_r2s.append(i)
            if r2s[i] == 'wholest':
                none_index_r2s.append(i)

        n2r = [e for i, e in enumerate(n2r) if i not in none_index_n2r]
        n2r_id = [e for i, e in enumerate(n2r_id) if i not in none_index_n2r]
        r2s = [e for i, e in enumerate(r2s) if i not in none_index_r2s]
        r2s_id = [e for i, e in enumerate(r2s_id) if i not in none_index_r2s]
        
        return n2r, n2r_id, r2s, r2s_id, more2most, more2most_id, s2r, s2r_id, most2more, most2more_id, r2n, r2n_id, more2none_id

# for synonym       
    def change_synonym(self, pos):
        pos_w, syn_w, pos_id = utils.WordNet().synonym(self.test_sample, pos)
        synonym = []
        for p, syn in zip(pos_w, syn_w):
            similarity = []
            if syn:
                for s in syn:
                    sim = torch.cosine_similarity(utils.generate_embedding(p, self.NET, self.model_type),\
                                                       utils.generate_embedding(s, self.NET, self.model_type)).item()
                    similarity.append(sim)
                arg_index = np.argsort(similarity)
                index = -1
                sim_word = syn[arg_index[index]]
                while(True):
                    if ('-' in sim_word) and (abs(index) < len(arg_index)):
                        index -= 1
                        sim_word = syn[arg_index[index]]
                    else:
                        break
                #sim_word = syn[arg_index[-1]]
                synonym.append(sim_word)
            else:
                synonym.append([])
        empty_id = [i for i, e in enumerate(synonym) if e == []]
        if empty_id:
            synonym = [e for i, e in enumerate(synonym) if i not in empty_id]
            pos_id = [e for i, e in enumerate(pos_id) if i not in empty_id]
        for i in range(len(synonym)):
            if synonym[i] == 'creepy-crawly':
                synonym[i] = 'weird'
            if synonym[i] == "ship's company":
                synonym[i] = 'ship company'
        '''
            if synonym[i] == 'well-disposed':
                synonym[i] = 'kindly'
            if synonym[i] == 'one-time':
                synonym[i] = 'previous'
            if synonym[i] == 'forward-looking':
                synonym[i] = 'fashion'
        '''
        return synonym, pos_id
        
# for verb        
    def pres_past(self):
        pres_w, pres_id, past_w, past_id = utils.WordNet().verb_upos(self.test_sample)
        pres2past = [conjugate(w, tense = PAST, person = 1, number = SG) for w in pres_w]
        pres_id = []
        for i in range(len(pres2past)):
            if (pres2past[i] == '-lrb-ed') or (pres2past[i] == 'rrb-ed') or (pres2past[i] == '[cls]ed'):
                pres_id.append(i)
                #pres2past.pop(i)
                #pres_id.pop(i)
               # pres2past[i] = '-lrb-'
            #if pres2past[i] == '-rrb-ed':
               # pres2past[i] = '-rrb-'
        pres2past = [e for i, e in enumerate(pres2past) if i not in pres_id]
        pres_id = [e for i, e in enumerate(pres_id) if i not in pres_id]

        past2pres = [conjugate(w, tense = PRESENT, person = 1, number = SG) for w in past_w]
        for i in range(len(past2pres)):
            if past2pres[i] == 're-create':
                past2pres[i] = 'recreate'
        return pres2past, pres_id, past2pres, past_id

# for noun        
    def get_noun(self):
        nn, nn_id = utils.WordNet().get_xpos(self.test_sample, 'NN')
        nns, nns_id = utils.WordNet().get_xpos(self.test_sample, 'NNS')
        return nn, nn_id, nns, nns_id
        
    def pl_sing(self):
        sin2plu_0index = []
        plu2sin_0index = []
        
        sin, sin_id, plu, plu_id = self.get_noun()
        sin2plu = [pluralize(s) for s in sin]
        for i in range(len(sin2plu)):
            if sin2plu[i] == 's-slsrsbs-s':
                sin2plu_0index.append(i)
                #sin2plu.pop(i)
                #sin_id.pop(i)
                #sin2plu[i] = '-lrb-'
            if sin2plu[i] == 's-srsrsbs-s':
                sin2plu_0index.append(i)
                #sin2plu[i] = '-rrb-'
            if sin2plu[i] == "ols'":
                sin2plu[i] = 'ols'
            if sin2plu[i] == "'ems":
                sin2plu[i] = 'ems'

        sin2plu = [e for i, e in enumerate(sin2plu) if i not in sin2plu_0index]
        sin_id = [e for i, e in enumerate(sin_id) if i not in sin2plu_0index]
        plu2sin = [singularize(p) for p in plu]
        return sin2plu, sin_id, plu2sin, plu_id


 

def artikel_attack(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT = []
    VISUAL = []
    LABEL_CHANGED = []
    JACCARD = []
    CONFI = []
    SCORE = []
    ERASER_STATS = []

    for test_sample, gt_label in tqdm(test_samples):
        similarity = []
        preturb_text = []
        visual = []
        label_changed = []
        jaccard = []
        confi =[]
        eraser_stats = []
        
        # to let text will be correctly split in further
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Artikelattack = ArtikelAttack(test_sample, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            adv_tokens = Artikelattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            adv_tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(adv_tokens)
        tokens_size = len(adv_tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            print('org attr: {}'.format(org_attr))
            visual.append([adv_tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            print('percentage attr: {}'.format(org_attr))
            print('word sort: {}'.format(org_attr_sort))
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, adv_tokens, model)
            #erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Manualattack.tokens(test_sample))
            visual.append([adv_tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, adv_tokens, model)
            #erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, org_attr_sort, model)

        adv_id = Artikelattack.get_artikel_id(adv_tokens)
        ids_size = len(adv_id)
        '''
        org_attr = Manualattack.saliency1
        #org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        if model == 'rnn':
            org_attr, org_attr_sort = normalize_feature_scores(org_attr, model)
        elif moedel == 'bert':
            org_attr, org_attr_sort = normalize_feature_scores(org_attr, model, adv_tokens)

        if model == 'rnn':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, adv_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, adv_tokens, label, score, conf)
        elif model == 'bert':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, bert_tokens, label, score, conf)
        
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, adv_tokens)
            erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
 
        for i in range(ids_size):
            adv_tokens = Artikelattack.change_artikel(adv_id, adv_tokens, i)
            adv_text = ' '.join(adv_tokens)
            adv_split = adv_text.split()
            adv_size = len(adv_split)
            '''
            if model == 'bert':
              adv_bert = copy.deepcopy(adv_text)
              adv_text = '[CLS] ' + adv_bert + ' [SEP]'
              adv_tokens = Artikelattack.tokens(adv_text)
            else:
                adv_text = adv_text
            '''    
            counter_adv = Counter(adv_split)
            preturb = list((counter_org & counter_adv).elements())
            percent = (adv_size - len(preturb)) / tokens_size
            
            adv_score, adv_label = RNN_emb.generate(adv_text)
            adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
            adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
            if model == 'rnn':
                adv_attr = normalize_feature_scores(adv_attr)
                adv = [adv_split, conf, adv_label, label, adv_attr]
                adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
            elif model == 'bert':
                adv_attr = normalize_feature_scores(adv_attr, model, Manualattack.tokens(adv_text))
                adv = [adv_split, conf, adv_label, label, adv_attr]  
                adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)

            '''
            adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
            adv = [adv_tokens, adv_conf, adv_label, label, adv_attr.cpu()]
            adv_attr, adv_attr_sort= normalize_feature_scores(adv_attr)
            
            if model == 'rnn':
                adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_tokens)
                erasers = eraser4all(RNN_emb, adv_jaccard, adv_tokens, label, score, conf)
            elif model == 'bert':
                adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
            
            if model == 'rnn': 
                adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_tokens)
                erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, adv_word_index)
            elif model == 'bert':
                adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_tokens)
                erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, adv_word_index)
            '''   
            all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
            
            if percent <= 0.2:
                similarity.append(all_sim)
                preturb_text.append(adv_text)
                visual.append(adv)
                label_changed.append(adv_label)
                jaccard.append(adv_jaccard)
                confi.append(adv_conf)
                eraser_stats.append(erasers)
            else:
              break

        SIMILARITY.append(similarity)
        PRETURB_TEXT.append(preturb_text)
        VISUAL.append(visual)
        LABEL_CHANGED.append(label_changed)
        JACCARD.append(jaccard)
        CONFI.append(confi)
        ERASER_STATS.append(eraser_stats)
    
    '''
    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_artikel_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_ig_artikel_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_ig_artikel_visual.pkl')
        file_label = os.path.join(directory, 'sst_ig_artikel_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_artikel_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_ig_artikel_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_artikel_eraser.pkl')
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_artikel_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_dl_artikel_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_dl_artikel_visual.pkl')
        file_label = os.path.join(directory, 'sst_dl_artikel_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_artikel_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_dl_artikel_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_artikel_eraser.pkl')
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_artikel_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_s_artikel_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_s_artikel_visual.pkl')
        file_label = os.path.join(directory, 'sst_s_artikel_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_artikel_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_s_artikel_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_artikel_eraser.pkl')
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_artikel_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_sh_artikel_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_sh_artikel_visual.pkl')
        file_label = os.path.join(directory, 'sst_sh_artikel_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_artikel_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_sh_artikel_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_artikel_eraser.pkl')  

    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)                      
    with open(file_preturb_text, 'wb') as f:
        pickle.dump(PRETURB_TEXT, f)
    with open(file_visual, 'wb') as f:
        pickle.dump(VISUAL, f)
    with open(file_label, 'wb') as f:
        pickle.dump(LABEL_CHANGED, f) 
    with open(file_confi, 'wb') as f:
        pickle.dump(CONFI, f)
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f)
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f)
    '''

def noun_pl_sin(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT = []
    VISUAL = []
    LABEL_CHANGED = []
    JACCARD = []
    CONFI = []
    ERASER_STATS = []
    
    count = 0
    
    for test_sample, gt_label in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text = []
        visual = []
        label_changed = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            visual.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            visual.append([tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)

        nn, nn_id, nns, nns_id = Entityattack.pl_sing() # nn singular, nns plural
        nn_size = len(nn_id)
        nns_size = len(nns_id)
            
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
        
        if nn:
          for i in range(nn_size):
              tokens = Entityattack.text_change(nn[i], nn_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr] 
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              
              if percent <= 0.2: 
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        if nns: 
          for i in range(nns_size):
              #print('nns: {}'.format(tokens))
              tokens = Entityattack.text_change(nns[i], nns_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)        
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              #print('nns advtext: {}'.format(adv_text))
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  #print('adv arrt len: {}'.format(len(adv_attr)))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
                  
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
              
        SIMILARITY.append(similarity)
        PRETURB_TEXT.append(preturb_text)
        VISUAL.append(visual)
        LABEL_CHANGED.append(label_changed)
        JACCARD.append(jaccard)
        CONFI.append(confi)
        ERASER_STATS.append(eraser_stats)

    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_noun_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_ig_noun_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_ig_noun_visual.pkl')
        file_label = os.path.join(directory, 'sst_ig_noun_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_noun_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_ig_noun_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_noun_eraser.pkl')
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_noun_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_dl_noun_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_dl_noun_visual.pkl')
        file_label = os.path.join(directory, 'sst_dl_noun_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_noun_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_dl_noun_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_noun_eraser.pkl')
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_noun_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_s_noun_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_s_noun_visual.pkl')
        file_label = os.path.join(directory, 'sst_s_noun_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_noun_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_s_noun_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_noun_eraser.pkl')
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_noun_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_sh_noun_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_sh_noun_visual.pkl')
        file_label = os.path.join(directory, 'sst_sh_noun_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_noun_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_sh_noun_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_noun_eraser.pkl')  

    
    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)                      
    with open(file_preturb_text, 'wb') as f:
        pickle.dump(PRETURB_TEXT, f)
    with open(file_visual, 'wb') as f:
        pickle.dump(VISUAL, f)
    with open(file_label, 'wb') as f:
        pickle.dump(LABEL_CHANGED, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f)   
    with open(file_confi, 'wb') as f:
        pickle.dump(CONFI, f)     
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f)                    


def verb_tense(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT = []
    VISUAL = []
    LABEL_CHANGED = []
    JACCARD = []
    CONFI = []
    ERASER_STATS = []
    
    count = 0
    
    for test_sample, gt_label in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text = []
        visual = []
        label_changed = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            visual.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            visual.append([tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)
        
        '''
        bert_sample = copy.deepcopy(test_sample)
        bert_sample = '[CLS] ' + bert_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)

        conf = RNN_emb.confidence(score, label)
        confi.append(conf)

        #adv_text = copy.deepcopy(test_sample)
        tokens = Entityattack.tokens(test_sample)
        bert_tokens = Entityattack.tokens(bert_sample)
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        visual.append([tokens, conf, label, label, Manualattack.saliency1.cpu()])
        

        org_attr = Manualattack.saliency1
        org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        
        if model == 'rnn':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, tokens, label, score, conf)
        elif model == 'bert':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, bert_tokens, label, score, conf)
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''    
        
        pres, pres_id, past, past_id = Entityattack.pres_past()
        pres_size = len(pres_id)
        past_size = len(past_id)
        
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
        
        if pres:
          for i in range(pres_size):
              print(tokens)
              tokens = Entityattack.text_change(pres[i], pres_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              print(adv_text)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
      
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
              
        if past: 
          for i in range(past_size):
              tokens = Entityattack.text_change(past[i], past_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
              
        SIMILARITY.append(similarity)
        PRETURB_TEXT.append(preturb_text)
        VISUAL.append(visual)
        LABEL_CHANGED.append(label_changed)
        JACCARD.append(jaccard)
        CONFI.append(confi)
        ERASER_STATS.append(eraser_stats)
    
    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_verb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_ig_verb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_ig_verb_visual.pkl')
        file_label = os.path.join(directory, 'sst_ig_verb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_verb_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_ig_verb_confidence.pkl') 
        file_eraser = os.path.join(directory, 'sst_ig_verb_eraser.pkl')
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_verb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_dl_verb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_dl_verb_visual.pkl')
        file_label = os.path.join(directory, 'sst_dl_verb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_verb_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_dl_verb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_verb_eraser.pkl')
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_verb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_s_verb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_s_verb_visual.pkl')
        file_label = os.path.join(directory, 'sst_s_verb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_verb_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_s_verb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_verb_eraser.pkl')
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_verb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_sh_verb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_sh_verb_visual.pkl')
        file_label = os.path.join(directory, 'sst_sh_verb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_verb_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_sh_verb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_verb_eraser.pkl')  

    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)                   
    with open(file_preturb_text, 'wb') as f:
        pickle.dump(PRETURB_TEXT, f)
    with open(file_visual, 'wb') as f:
        pickle.dump(VISUAL, f)
    with open(file_label, 'wb') as f:
        pickle.dump(LABEL_CHANGED, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f) 
    with open(file_confi, 'wb') as f:
        pickle.dump(CONFI, f)  
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f)            

def adj_compar_super(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT = []
    VISUAL = []
    LABEL_CHANGED = []
    JACCARD = []
    CONFI = []
    ERASER_STATS = []

    count = 0
    
    for test_sample, gt_label in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text = []
        visual = []
        label_changed = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            visual.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            visual.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)
        
        '''
        bert_sample = copy.deepcopy(test_sample)
        bert_sample = '[CLS] ' + bert_sample + ' [SEP]'

        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)

        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        #adv_text = copy.deepcopy(test_sample)
        tokens = Entityattack.tokens(test_sample)
        bert_tokens = Entityattack.tokens(bert_sample)
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        visual.append([tokens, conf, label, label, Manualattack.saliency1.cpu()])

        org_attr = Manualattack.saliency1
        org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        
        if model == 'rnn':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, tokens, label, score, conf)
        elif model == 'bert':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, bert_tokens, label, score, conf)
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''
        
        n2r, n2r_id, r2s, r2s_id, more2most, more2most_id, s2r,\
                    s2r_id, most2more, most2more_id, r2n, r2n_id, more2none_id = Entityattack.type_exchange()

        n2r_size = len(n2r_id)
        r2s_size = len(r2s_id)
        s2r_size = len(s2r_id)
        r2n_size = len(r2n_id)
        more2most_size = len(more2most_id)
        most2more_size = len(most2more_id)
        more2none_size = len(more2none_id)
        
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)

        if n2r:
          for i in range(n2r_size):
              tokens = Entityattack.text_change(n2r[i], n2r_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
      
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
  
        
        if r2s:
          for i in range(r2s_size):
              tokens = Entityattack.text_change(r2s[i], r2s_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        if more2most:
          for i in range(more2most_size):
              tokens = Entityattack.text_change(more2most[i], more2most_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        if s2r: 
          for i in range(s2r_size):
              tokens = Entityattack.text_change(s2r[i], s2r_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        if r2n: 
          for i in range(r2n_size):
              tokens = Entityattack.text_change(r2n[i], r2n_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
                
        if most2more: 
          for i in range(most2more_size):
              tokens = Entityattack.text_change(most2more[i], most2more_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
                  
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
                
        if more2none_id: 
          for i in range(more2none_size):
              tokens = Entityattack.text_change('', more2none_id[i], tokens)
              none_token = [i for i in tokens if i != '']
              adv_text = ' '.join(none_token)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
                  
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text.append(adv_text)
                  visual.append(adv)
                  label_changed.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break
              
        SIMILARITY.append(similarity)
        PRETURB_TEXT.append(preturb_text)
        VISUAL.append(visual)
        LABEL_CHANGED.append(label_changed)
        JACCARD.append(jaccard)
        CONFI.append(confi)
        ERASER_STATS.append(eraser_stats)

    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_adj_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_ig_adj_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_ig_adj_visual.pkl')
        file_label = os.path.join(directory, 'sst_ig_adj_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_adj_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_ig_adj_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_adj_eraser.pkl')
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_adj_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_dl_adj_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_dl_adj_visual.pkl')
        file_label = os.path.join(directory, 'sst_dl_adj_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_adj_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_dl_adj_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_adj_eraser.pkl')
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_adj_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_s_adj_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_s_adj_visual.pkl')
        file_label = os.path.join(directory, 'sst_s_adj_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_adj_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_s_adj_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_adj_eraser.pkl')
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_adj_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_sh_adj_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_sh_adj_visual.pkl')
        file_label = os.path.join(directory, 'sst_sh_adj_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_adj_jaccard.pkl') 
        file_confi = os.path.join(directory, 'sst_sh_adj_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_adj_eraser.pkl')  

    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)                    
    with open(file_preturb_text, 'wb') as f:
        pickle.dump(PRETURB_TEXT, f)
    with open(file_visual, 'wb') as f:
        pickle.dump(VISUAL, f)
    with open(file_label, 'wb') as f:
        pickle.dump(LABEL_CHANGED, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f) 
    with open(file_confi, 'wb')as f:
        pickle.dump(CONFI, f)  
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f)          

        
def adj_syn(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT_ASYN = []
    VISUAL_ASYN = []
    LABEL_CHANGED_ASYN = []
    JACCARD = []
    CONFI_ASYN = []
    ERASER_STATS = []
    
    count = 0
    
    for test_sample, gt_label in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text_asyn = []
        visual_asyn = []
        label_changed_asyn = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text_asyn.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed_asyn.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            #print('org attr len:{}'.format(len(org_attr)))
            visual_asyn.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            visual_asyn.append([tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)
        '''
        bert_sample = copy.deepcopy(test_sample)
        bert_sample = '[CLS] ' + bert_sample + ' [SEP]'

        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)

        conf = RNN_emb.confidence(score, label)
        confi_asyn.append(conf)

        #adv_text = copy.deepcopy(test_sample)
        tokens = Entityattack.tokens(test_sample)
        bert_tokens = Entityattack.tokens(bert_sample)
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        visual_asyn.append([tokens, conf, label, label, Manualattack.saliency1.cpu()])

    
        org_attr = Manualattack.saliency1
        org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        
        if model == 'rnn':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, tokens, label, score, conf)
        elif model == 'bert':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, bert_tokens, label, score, conf)
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''
        
        syn, syn_id = Entityattack.change_synonym('ADJ')
        syn_size = len(syn_id)
        
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
        
        if syn:
          for i in range(syn_size):
              #print('tokens: {}'.format(tokens))
              tokens = Entityattack.text_change(syn[i], syn_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              #print('adv text: {}'.format(adv_text))
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  #print('adv attr len: {}'.format(len(adv_attr)))
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]  
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text_asyn.append(adv_text)
                  visual_asyn.append(adv)
                  label_changed_asyn.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        SIMILARITY.append(similarity)
        PRETURB_TEXT_ASYN.append(preturb_text_asyn)
        VISUAL_ASYN.append(visual_asyn)
        LABEL_CHANGED_ASYN.append(label_changed_asyn)
        JACCARD.append(jaccard)
        CONFI_ASYN.append(confi)
        ERASER_STATS.append(eraser_stats)

    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_asyn_similarity.pkl')
        file_preturb_text_asyn = os.path.join(directory, 'sst_ig_asyn_preturb_text.pkl')
        file_visual_asyn = os.path.join(directory, 'sst_ig_asyn_visual.pkl')
        file_label_asyn = os.path.join(directory, 'sst_ig_asyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_asyn_jaccard.pkl') 
        file_confi_asyn = os.path.join(directory, 'sst_ig_asyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_asyn_eraser.pkl')
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_asyn_similarity.pkl')
        file_preturb_text_asyn = os.path.join(directory, 'sst_dl_asyn_preturb_text.pkl')
        file_visual_asyn = os.path.join(directory, 'sst_dl_asyn_visual.pkl')
        file_label_asyn = os.path.join(directory, 'sst_dl_asyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_asyn_jaccard.pkl') 
        file_confi_asyn = os.path.join(directory, 'sst_dl_asyn_confidence.pkl')  
        file_eraser = os.path.join(directory, 'sst_dl_asyn_eraser.pkl')  
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_asyn_similarity.pkl')
        file_preturb_text_asyn = os.path.join(directory, 'sst_s_asyn_preturb_text.pkl')
        file_visual_asyn = os.path.join(directory, 'sst_s_asyn_visual.pkl')
        file_label_asyn = os.path.join(directory, 'sst_s_asyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_asyn_jaccard.pkl') 
        file_confi_asyn = os.path.join(directory, 'sst_s_asyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_asyn_eraser.pkl')  
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_asyn_similarity.pkl')
        file_preturb_text_asyn = os.path.join(directory, 'sst_sh_asyn_preturb_text.pkl')
        file_visual_asyn = os.path.join(directory, 'sst_sh_asyn_visual.pkl')
        file_label_asyn = os.path.join(directory, 'sst_sh_asyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_asyn_jaccard.pkl') 
        file_confi_asyn = os.path.join(directory, 'sst_sh_asyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_asyn_eraser.pkl')  
          
    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f) 
    with open(file_preturb_text_asyn, 'wb') as f:
        pickle.dump(PRETURB_TEXT_ASYN, f)
    with open(file_visual_asyn, 'wb') as f:
        pickle.dump(VISUAL_ASYN, f)
    with open(file_label_asyn, 'wb') as f:
        pickle.dump(LABEL_CHANGED_ASYN, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f) 
    with open(file_confi_asyn, 'wb') as f:
        pickle.dump(CONFI_ASYN, f) 
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f) 

def verb_syn(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT_VSYN = []
    VISUAL_VSYN = []
    LABEL_CHANGED_VSYN = []
    JACCARD = []
    CONFI_VSYN = []
    ERASER_STATS = []

    count = 0
    
    for test_sample, gt_label in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text_vsyn = []
        visual_vsyn = []
        label_changed_vsyn = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text_vsyn.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed_vsyn.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            visual_vsyn.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            visual_vsyn.append([tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)
        '''
        bert_sample = copy.deepcopy(test_sample)
        bert_sample = '[CLS] ' + bert_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)

        conf = RNN_emb.confidence(score, label)
        confi_vsyn.append(conf)

        #adv_text = copy.deepcopy(test_sample)
        tokens = Entityattack.tokens(test_sample)
        print('sentence tokens {}'.format(tokens))
        bert_tokens = Entityattack.tokens(bert_sample)
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        visual_vsyn.append([tokens, conf, label, label, Manualattack.saliency1.cpu()])
    
        org_attr = Manualattack.saliency1
        org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''
        
        syn, syn_id = Entityattack.change_synonym('VERB')
        syn_size = len(syn_id)
        
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
        
        if syn:
          for i in range(syn_size):
              tokens = Entityattack.text_change(syn[i], syn_id[i], tokens)
              adv_text = ' '.join(tokens)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]  
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)

              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text_vsyn.append(adv_text)
                  visual_vsyn.append(adv)
                  label_changed_vsyn.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        SIMILARITY.append(similarity)
        PRETURB_TEXT_VSYN.append(preturb_text_vsyn)
        VISUAL_VSYN.append(visual_vsyn)
        LABEL_CHANGED_VSYN.append(label_changed_vsyn)
        JACCARD.append(jaccard)
        CONFI_VSYN.append(confi)
        ERASER_STATS.append(eraser_stats)
    
    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_vsyn_similarity.pkl')
        file_preturb_text_vsyn = os.path.join(directory, 'sst_ig_vsyn_preturb_text.pkl')
        file_visual_vsyn = os.path.join(directory, 'sst_ig_vsyn_visual.pkl')
        file_label_vsyn = os.path.join(directory, 'sst_ig_vsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_vsyn_jaccard.pkl') 
        file_confi_vsyn = os.path.join(directory, 'sst_ig_vsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_vsyn_eraser.pkl') 
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_vsyn_similarity.pkl')
        file_preturb_text_vsyn = os.path.join(directory, 'sst_dl_vsyn_preturb_text.pkl')
        file_visual_vsyn = os.path.join(directory, 'sst_dl_vsyn_visual.pkl')
        file_label_vsyn = os.path.join(directory, 'sst_dl_vsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_vsyn_jaccard.pkl') 
        file_confi_vsyn = os.path.join(directory, 'sst_dl_vsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_vsyn_eraser.pkl') 
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_vsyn_similarity.pkl')
        file_preturb_text_vsyn = os.path.join(directory, 'sst_s_vsyn_preturb_text.pkl')
        file_visual_vsyn = os.path.join(directory, 'sst_s_vsyn_visual.pkl')
        file_label_vsyn = os.path.join(directory, 'sst_s_vsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_vsyn_jaccard.pkl') 
        file_confi_vsyn = os.path.join(directory, 'sst_s_vsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_vsyn_eraser.pkl')  
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_vsyn_similarity.pkl')
        file_preturb_text_vsyn = os.path.join(directory, 'sst_sh_vsyn_preturb_text.pkl')
        file_visual_vsyn = os.path.join(directory, 'sst_sh_vsyn_visual.pkl')
        file_label_vsyn = os.path.join(directory, 'sst_sh_vsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_vsyn_jaccard.pkl') 
        file_confi_vsyn = os.path.join(directory, 'sst_sh_vsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_vsyn_eraser.pkl')  
      
    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)        
    with open(file_preturb_text_vsyn, 'wb') as f:
        pickle.dump(PRETURB_TEXT_VSYN, f)
    with open(file_visual_vsyn, 'wb') as f:
        pickle.dump(VISUAL_VSYN, f)
    with open(file_label_vsyn, 'wb') as f:
        pickle.dump(LABEL_CHANGED_VSYN, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f) 
    with open(file_confi_vsyn, 'wb') as f:
        pickle.dump(CONFI_VSYN, f) 
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f) 
  
        
def noun_syn(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT_NSYN = []
    VISUAL_NSYN = []
    LABEL_CHANGED_NSYN = []
    JACCARD = []
    CONFI_NSYN = []
    ERASER_STATS = []

    count = 0
    
    for test_sample, gt_label in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text_nsyn = []
        visual_nsyn = []
        label_changed_nsyn = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text_nsyn.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed_nsyn.append(label)
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)
        
        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            visual_nsyn.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            visual_nsyn.append([tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)
        '''
        bert_sample = copy.deepcopy(test_sample)
        bert_sample = '[CLS] ' + bert_sample + ' [SEP]'
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)

        conf = RNN_emb.confidence(score, label)
        confi_nsyn.append(conf)

        tokens = Entityattack.tokens(test_sample)
        bert_tokens = Entityattack.tokens(bert_sample)
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        visual_nsyn.append([tokens, conf, label, label, Manualattack.saliency1.cpu()])
    
        org_attr = Manualattack.saliency1
        org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        
        if model == 'rnn':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, tokens, label, score, conf)
        elif model == 'bert':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, bert_tokens, label, score, conf)
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''
        syn, syn_id = Entityattack.change_synonym('NOUN')
        syn_size = len(syn_id)
           
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
        
        if syn:
          for i in range(syn_size):
              #print(tokens)
              tokens = Entityattack.text_change(syn[i], syn_id[i], tokens)
              adv_text = ' '.join(tokens)
              #print(adv_text)
              adv_split = adv_text.split()
              adv_size = len(adv_split)
              '''
              bert_text = copy.deepcopy(adv_text)
              bert_text = '[CLS] ' + bert_text + ' [SEP]'
              adv_bert_tokens = Entityattack.tokens(bert_text)
              '''
              
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
            
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              if model == 'rnn':
                  adv_attr = normalize_feature_scores(adv_attr)
                  adv = [adv_split, conf, adv_label, label, adv_attr]
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
              elif model == 'bert':
                  adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                  adv = [adv_split, conf, adv_label, label, adv_attr]  
                  adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                  adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                  #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
              '''
              counter_adv = Counter(adv_split)
              preturb = list((counter_org & counter_adv).elements())
              percent = (adv_size - len(preturb)) / tokens_size
              
              adv_score, adv_label = RNN_emb.generate(adv_text)
              adv_conf = RNN_emb.confidence(adv_score, adv_label)
                  
              adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
              adv = [adv_split, adv_conf, adv_label, label, adv_attr.cpu()]
              adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            
              if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_split, label, score, conf)
              elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
              if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split)
                  erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_word_index)
              elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
              '''
              all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
             
              if percent <= 0.2:
                  similarity.append(all_sim)
                  preturb_text_nsyn.append(adv_text)
                  visual_nsyn.append(adv)
                  label_changed_nsyn.append(adv_label)
                  jaccard.append(adv_jaccard)
                  confi.append(adv_conf)
                  eraser_stats.append(erasers)
              else:
                break

        SIMILARITY.append(similarity)
        PRETURB_TEXT_NSYN.append(preturb_text_nsyn)
        VISUAL_NSYN.append(visual_nsyn)
        LABEL_CHANGED_NSYN.append(label_changed_nsyn)
        JACCARD.append(jaccard)
        CONFI_NSYN.append(confi)
        ERASER_STATS.append(eraser_stats)

    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_nsyn_similarity.pkl')
        file_preturb_text_nsyn = os.path.join(directory, 'sst_ig_nsyn_preturb_text.pkl')
        file_visual_nsyn = os.path.join(directory, 'sst_ig_nsyn_visual.pkl')
        file_label_nsyn = os.path.join(directory, 'sst_ig_nsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_nsyn_jaccard.pkl') 
        file_confi_nsyn = os.path.join(directory, 'sst_ig_nsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_nsyn_eraser.pkl')  
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_nsyn_similarity.pkl')
        file_preturb_text_nsyn = os.path.join(directory, 'sst_dl_nsyn_preturb_text.pkl')
        file_preturb_text_nsyn = os.path.join(directory, 'sst_dl_nsyn_preturb_text.pkl')
        file_visual_nsyn = os.path.join(directory, 'sst_dl_nsyn_visual.pkl')
        file_label_nsyn = os.path.join(directory, 'sst_dl_nsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_nsyn_jaccard.pkl') 
        file_confi_nsyn = os.path.join(directory, 'sst_dl_nsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_nsyn_eraser.pkl')  
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_nsyn_similarity.pkl')
        file_preturb_text_nsyn = os.path.join(directory, 'sst_s_nsyn_preturb_text.pkl')
        file_visual_nsyn = os.path.join(directory, 'sst_s_nsyn_visual.pkl')
        file_label_nsyn = os.path.join(directory, 'sst_s_nsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_nsyn_jaccard.pkl') 
        file_confi_nsyn = os.path.join(directory, 'sst_s_nsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_nsyn_eraser.pkl')  
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_nsyn_similarity.pkl')
        file_preturb_text_nsyn = os.path.join(directory, 'sst_sh_nsyn_preturb_text.pkl')
        file_visual_nsyn = os.path.join(directory, 'sst_sh_nsyn_visual.pkl')
        file_label_nsyn = os.path.join(directory, 'sst_sh_nsyn_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_nsyn_jaccard.pkl') 
        file_confi_nsyn = os.path.join(directory, 'sst_sh_nsyn_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_nsyn_eraser.pkl')  
            
            
    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)                     
    with open(file_preturb_text_nsyn, 'wb') as f:
        pickle.dump(PRETURB_TEXT_NSYN, f)
    with open(file_visual_nsyn, 'wb') as f:
        pickle.dump(VISUAL_NSYN, f)
    with open(file_label_nsyn, 'wb') as f:
        pickle.dump(LABEL_CHANGED_NSYN, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f) 
    with open(file_confi_nsyn, 'wb') as f:
        pickle.dump(CONFI_NSYN, f) 
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f) 
        
def emb_augmentation(test_samples, RNN_emb, feature_method, directory, model):
    SIMILARITY = []
    PRETURB_TEXT = []
    VISUAL = []
    LABEL_CHANGED = []
    JACCARD = []
    CONFI = []
    ERASER_STATS = []
    
    count = 0
    
    for data in tqdm(test_samples):
        count += 1
        similarity = []
        preturb_text = []
        visual = []
        label_changed = []
        jaccard = []
        confi = []
        eraser_stats = []
        
        test_sample = data['org']
        adv_sample = data['perturb']
        
        test_sample = test_sample.lower()
        test_sample = utils.check_token(test_sample, RNN_emb, model)
        preturb_text.append(test_sample)
        score, label = RNN_emb.generate(test_sample)
        label_changed.append(label)
        
        
        # add special tokens for bert
        if model == 'bert':
            #bert_sample = copy.deepcopy(test_sample)
            test_sample = '[CLS] ' + test_sample + ' [SEP]'
            #print('test sample: {}'.format(test_sample))
        
        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)
        
        conf = RNN_emb.confidence(score, label)
        confi.append(conf)

        if model == 'rnn':
            tokens = Entityattack.tokens(test_sample)
        elif model == 'bert':
            #bert tokenizer will produce subwords, here I don't want any subwords
            tokens = test_sample.split()
            #print('len test sample: {}'.format(len(tokens)))
        
        #here keep special tokens in count, cause it doesn't affect the resule so much
        counter_org = Counter(tokens)
        tokens_size = len(tokens)
        
        org_attr = Manualattack.saliency1
        #print('len org attr: {}'.format(len(org_attr[0])))
        if model == 'rnn':
            org_attr = normalize_feature_scores(org_attr)
            visual.append([tokens, conf, label, label, org_attr])
            org_attr, org_attr_sort = percentage_feature_scores(org_attr)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort)
        elif model == 'bert':
            org_attr = normalize_feature_scores(org_attr, model, Entityattack.tokens(test_sample))
            #print('len normalize org attr: {}'.format(len(org_attr)))
            visual.append([tokens, conf, label, label, org_attr])  
            org_attr, org_attr_sort = percentage_feature_scores(org_attr, model)
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens, model)
            #erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index, model)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, org_attr_sort, model)
        '''
        bert_sample = copy.deepcopy(test_sample)
        bert_sample = '[CLS] ' + bert_sample + ' [SEP]'

        Manualattack = ManualAttack(test_sample, label, RNN_emb, feature_method, model)
        Entityattack = EntityAttack(test_sample, label, RNN_emb, model)

        conf = RNN_emb.confidence(score, label)
        confi.append(conf)

        tokens = Entityattack.tokens(test_sample)
        bert_tokens = Entityattack.tokens(bert_sample)
        
        visual.append([tokens, conf, label, label, Manualattack.saliency1.cpu()])
    
        org_attr = Manualattack.saliency1
        org_attr, org_attr_sort = normalize_feature_scores(org_attr)
        

        if model == 'rnn':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, tokens, label, score, conf)
        elif model == 'bert':
            org_jaccard = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, org_jaccard, bert_tokens, label, score, conf)
  
        if model == 'rnn':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, tokens)
            erasers = eraser4all(RNN_emb, tokens, label, score, conf, word_index)
        elif model == 'bert':
            org_jaccard, word_index = word4jaccard(org_attr, org_attr_sort, bert_tokens)
            erasers = eraser4all(RNN_emb, bert_tokens, label, score, conf, word_index)
        '''     
        
        jaccard.append(org_jaccard)
        eraser_stats.append(erasers)
        
        for sample in adv_sample:
            sample = sample.lower()
            sample = utils.check_token(sample, RNN_emb, model)
            
            if model == 'bert':
                #bert_sample = copy.deepcopy(test_sample)
                adv_text = '[CLS] ' + sample + ' [SEP]'
                adv_split = adv_text.split()
            else:
                adv_text = copy.deepcopy(sample)
                adv_split = Entityattack.tokens(adv_text)
            
            adv_score, adv_label = RNN_emb.generate(sample)
            adv_conf = RNN_emb.confidence(adv_score, adv_label)
            
            adv_attr, _ = Manualattack.get_saliency(RNN_emb, adv_text, adv_label, feature_method)
            if model == 'rnn':
                adv_attr = normalize_feature_scores(adv_attr)
                adv = [adv_split, conf, adv_label, label, adv_attr]
                adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr)
                adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index)
                erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort)
            elif model == 'bert':
                adv_attr = normalize_feature_scores(adv_attr, model, Entityattack.tokens(adv_text))
                adv = [adv_split, conf, adv_label, label, adv_attr]  
                adv_attr, adv_attr_sort = percentage_feature_scores(adv_attr, model)
                adv_jaccard, word_index = word4jaccard(adv_attr, adv_attr_sort, adv_split, model)
                #erasers = eraser4all(RNN_emb, adv_split, label, score, conf, word_index, model)
                erasers = eraser4all(RNN_emb, adv_split, label, score, conf, adv_attr_sort, model)
            '''  
            adv_score, adv_label = RNN_emb.generate(sample)
            adv_conf = RNN_emb.confidence(adv_score, adv_label)

            adv_bert_sample = copy.deepcopy(sample)
            adv_bert_sample = '[CLS] ' + adv_bert_sample + ' [SEP]'
            
            adv_tokens = Entityattack.tokens(sample)
            adv_bert_tokens =Entityattack.tokens(adv_bert_sample)

            adv_attr, _ = Manualattack.get_saliency(RNN_emb, sample, adv_label, feature_method)
            adv = [adv_tokens, adv_conf, adv_label, label, adv_attr.cpu()]
            adv_attr,  adv_attr_sort= normalize_feature_scores(adv_attr)
            if model == 'rnn': 
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_tokens, label, score, conf)
            elif model == 'bert':
                  adv_jaccard = word4jaccard(adv_attr, adv_attr_sort, adv_bert_tokens)
                  erasers = eraser4all(RNN_emb, adv_jaccard, adv_bert_tokens, label, score, conf)
            if model == 'rnn': 
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_tokens)
                  erasers = eraser4all(RNN_emb, adv_tokens, label, score, conf, adv_word_index)
            elif model == 'bert':
                  adv_jaccard, adv_word_index = word4jaccard(adv_attr, adv_attr_sort, adv_tokens)
                  erasers = eraser4all(RNN_emb, adv_bert_tokens, label, score, conf, adv_word_index)
            '''
            all_sim = all_jaccard_similarity(org_jaccard, adv_jaccard)
            
            similarity.append(all_sim)
            preturb_text.append(sample)
            visual.append(adv)
            label_changed.append(adv_label)
            jaccard.append(adv_jaccard)
            confi.append(adv_conf)
            eraser_stats.append(erasers)
                
        SIMILARITY.append(similarity)
        PRETURB_TEXT.append(preturb_text)
        VISUAL.append(visual)
        LABEL_CHANGED.append(label_changed)
        JACCARD.append(jaccard)
        CONFI.append(confi)
        ERASER_STATS.append(eraser_stats)
    
    if feature_method == 'integrated_gradient':
        file_similarity = os.path.join(directory, 'sst_ig_emb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_ig_emb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_ig_emb_visual.pkl')
        file_label = os.path.join(directory, 'sst_ig_emb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_ig_emb_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_ig_emb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_ig_emb_eraser.pkl')  
    elif feature_method == 'deeplift':
        file_similarity = os.path.join(directory, 'sst_dl_emb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_dl_emb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_dl_emb_visual.pkl')
        file_label = os.path.join(directory, 'sst_dl_emb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_dl_emb_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_dl_emb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_dl_emb_eraser.pkl')  
    elif feature_method == 'simple':
        file_similarity = os.path.join(directory, 'sst_s_emb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_s_emb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_s_emb_visual.pkl')
        file_label = os.path.join(directory, 'sst_s_emb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_s_emb_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_s_emb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_s_emb_eraser.pkl')
    elif feature_method == 'shapley':
        file_similarity = os.path.join(directory, 'sst_sh_emb_similarity.pkl')
        file_preturb_text = os.path.join(directory, 'sst_sh_emb_preturb_text.pkl')
        file_visual = os.path.join(directory, 'sst_sh_emb_visual.pkl')
        file_label = os.path.join(directory, 'sst_sh_emb_label.pkl')
        file_jaccard = os.path.join(directory, 'sst_sh_emb_jaccard.pkl')
        file_confi = os.path.join(directory, 'sst_sh_emb_confidence.pkl')
        file_eraser = os.path.join(directory, 'sst_sh_emb_eraser.pkl')
 
    with open(file_similarity, 'wb')as f:
        pickle.dump(SIMILARITY, f)                    
    with open(file_preturb_text, 'wb') as f:
        pickle.dump(PRETURB_TEXT, f)
    with open(file_visual, 'wb') as f:
        pickle.dump(VISUAL, f)
    with open(file_label, 'wb') as f:
        pickle.dump(LABEL_CHANGED, f)   
    with open(file_jaccard, 'wb') as f:
        pickle.dump(JACCARD, f) 
    with open(file_confi, 'wb') as f:
        pickle.dump(CONFI, f) 
    with open(file_eraser, 'wb') as f:
        pickle.dump(ERASER_STATS, f) 

def main():
    #'''
    directory = '/home/kuo/code/sst_results_rnn/'
    RNN_emb = models.Classifier('/home/kuo/code/model/LSTM1.pt',  only_vocab=False, pretrained=1, model_type='RNN')
    model = 'rnn'
    #'''
    '''
    directory = '/home/kuo/code/sst_results_bert/'
    RNN_emb = models.Classifier('/home/kuo/code/model/bert1.pt',  only_vocab=False, pretrained=1, model_type='bert')
    model = 'bert'
    '''
    #aug_directory = '/home/kuo/code/data/'

    #if not os.path.isdir(directory):
    #    os.makedirs(directory)

    #test_samples = datasets_helper.random_sst_test_sample()
    #test_samples = utils.load_json('/home/kuo/code/data/aug_data.json')

    '''
    augmenter = CheckListAugmenter(pct_words_to_swap = 0.2, transformations_per_example=10)
    columns = ['org', 'perturb']
    data = []
    with open('/home/kuo/code/data/checklist_aug_data.json', 'w') as outfile:
        for test_sample, gt_label in test_samples:
            test_text = augmenter.augment(test_sample)
            if len(test_text) > 5:
                test_text = random.sample(test_text, 5)
            values = [test_sample, test_text]
            entry = dict(zip(columns, values))
            data.append(entry)
        json.dump(data, outfile)
    '''

    '''
    correct = 0
    wrong = 0
    for test_sample, gt_label in test_samples:
        score, label = RNN_emb.generate(test_sample)
        if gt_label == 'positive':
            if label == 0:
                wrong += 1
            else:
                correct += 1
        if gt_label == 'negative':
            if label == 0:
                correct += 1
            else:
                wrong += 1
        print('gt label:{}'.format(gt_label))
        print('predicted label: {}'.format(label))
    print('correct:{}'.format(correct))
    print('wrong:{}'.format(wrong))
    '''
    test_samples = [('this movie is great .', 1), ('this movie is awesome .', 1)] 

    feature_method = 'shapley'
    artikel_attack(test_samples, RNN_emb, feature_method, directory, model)
    '''
    artikel_attack(test_samples, RNN_emb, feature_method, directory, model)
    noun_pl_sin(test_samples, RNN_emb, feature_method, directory, model)
    verb_tense(test_samples, RNN_emb, feature_method, directory, model)
    adj_compar_super(test_samples, RNN_emb, feature_method, directory, model)
    adj_syn(test_samples, RNN_emb, feature_method, directory, model)
    verb_syn(test_samples, RNN_emb, feature_method, directory, model)
    noun_syn(test_samples, RNN_emb, feature_method, directory, model)
    '''
    #emb_augmentation(test_samples, RNN_emb, feature_method, directory, model)
    
    #feature_method = 'simple'
    '''
    artikel_attack(test_samples, RNN_emb, feature_method, directory, model)
    noun_pl_sin(test_samples, RNN_emb, feature_method, directory, model)
    verb_tense(test_samples, RNN_emb, feature_method, directory, model)
    adj_compar_super(test_samples, RNN_emb, feature_method, directory, model)
    adj_syn(test_samples, RNN_emb, feature_method, directory, model)
    verb_syn(test_samples, RNN_emb, feature_method, directory, model)
    noun_syn(test_samples, RNN_emb, feature_method, directory, model)
    '''
    #emb_augmentation(test_samples, RNN_emb, feature_method, directory, model)
    
    #feature_method = 'integrated_gradient'
    '''
    adj_compar_super(test_samples, RNN_emb, feature_method, directory, model)
    artikel_attack(test_samples, RNN_emb, feature_method, directory, model)
    noun_pl_sin(test_samples, RNN_emb, feature_method, directory, model)
    verb_tense(test_samples, RNN_emb, feature_method, directory, model)
    adj_syn(test_samples, RNN_emb, feature_method, directory, model)
    verb_syn(test_samples, RNN_emb, feature_method, directory, model)
    noun_syn(test_samples, RNN_emb, feature_method, directory, model)
    '''
    #emb_augmentation(test_samples, RNN_emb, feature_method, directory, model)


    #feature_method = 'shapley'
    '''
    adj_compar_super(test_samples, RNN_emb, feature_method, directory, model)
    artikel_attack(test_samples, RNN_emb, feature_method, directory, model)
    noun_pl_sin(test_samples, RNN_emb, feature_method, directory, model)
    verb_tense(test_samples, RNN_emb, feature_method, directory, model)
    adj_syn(test_samples, RNN_emb, feature_method, directory, model)
    verb_syn(test_samples, RNN_emb, feature_method, directory, model)
    noun_syn(test_samples, RNN_emb, feature_method, directory, model)
    '''
    #emb_augmentation(test_samples, RNN_emb, feature_method, directory, model)
    
if __name__ == '__main__':
    main()
