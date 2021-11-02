# here defines a variaties of classification models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import datasets_helper
from utils import epoch_time, train, evaluate, bert_train, bert_evaluate
from networks import RNN, CNN, SumEmbedding, BertBase
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

embedding_dim = 100
unit = 500
dropout = 0.3
out_dim = 2
min_length = 20
classify = torch.softmax
n_filters = 100
filter_sizes = [3, 4, 5]
num_layers = 1
# pad = 1


class Classifier:
    def __init__(self, model_dir, only_vocab=False, activation='relu', pretrained=1, model_type='RNN', pad=True):  
        self.model_dir = model_dir
        self.activation = activation
        self.model_type = model_type
        self.pad = pad
        if model_type == 'bert':
            self.init_model(pretrained)
        
        #if only_vocab:
        #    self.TEXT, self.LABEL = datasets_helper.load_sst(only_vocab=only_vocab)
        #    #self.TEXT, self.LABEL = datasets_helper.load_imdb(only_vocab=only_vocab)
        if model_type == 'RNN':
            self.TEXT, self.LABEL, self.train_iterator, self.valid_iterator, self.test_iterator = datasets_helper.load_sst(only_vocab=only_vocab)
            #self.TEXT, self.LABEL, self.train_iterator, self.valid_iterator, self.test_iterator = datasets_helper.load_imdb(only_vocab=only_vocab)
        elif model_type == 'bert':
            self.train_iterator, self.valid_iterator, self.test_iterator = datasets_helper.load_data_for_bert('sst', self.model.tokenizer)
        
        if model_type == 'RNN':
            self.input_dim = len(self.TEXT.vocab)
            self.pad_idx = self.TEXT.vocab.stoi[self.TEXT.pad_token]
            self.init_model(pretrained)
            
        criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        
    def init_model(self, pretrained):
        if self.model_type == 'RNN':
            self.init_RNN_params(embedding_dim, unit, out_dim, dropout)
        elif self.model_type == 'CNN':
            self.init_CNN_params(embedding_dim, n_filters, filter_sizes, out_dim, dropout, self.activation)
        elif self.model_type == 'SumEmb':
            self.init_simple_params(embedding_dim, out_dim)
        elif self.model_type  == 'bert':
            self.init_bert_params(out_dim)
        else:
            raise ValueError('only support RNN, CNN, SumEmb for now')

        if pretrained:
            self.load_model(self.model_dir)
        else:
            print('please train first.')

    def init_RNN_params(self, embedding_dim, unit, output_dim, dropout, use_pretrained=1):
        self.model = RNN(self.input_dim, embedding_dim, unit, output_dim, dropout, self.pad_idx).to(device)
        if use_pretrained:
           self.load_embedding()

    def init_CNN_params(self, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, activation,
                        use_pretrainemb=1):
        self.model = CNN(self.input_dim, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, self.pad_idx, activation).to(device)
        self.pad = False
        if use_pretrainemb:
           self.load_embedding()
           
    def init_bert_params(self, output_dim, use_pretrained=0):
        self.model = BertBase(output_dim).to(device)
        if use_pretrained:
           self.load_embedding()

    def init_simple_params(self, EMBEDDING_DIM, OUTPUT_DIM, use_pretrainemb=1):
        self.model = SumEmbedding(self.input_dim, EMBEDDING_DIM, OUTPUT_DIM, self.pad_idx)
        self.pad = False
        if use_pretrainemb:
            self.load_embedding()

    def load_embedding(self):
        pretrained_embeddings = self.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]
        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
        self.model.embedding.weight.data[self.pad_idx] = torch.zeros(embedding_dim)
        self.model.embedding.weight.requires_grad = True  # fix embedding weights

    def load_model(self, model_dir):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_dir))
        else:
            # self.model.load_state_dict(torch.load(model_dir, map_location=device))
            self.model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
        self.model.eval()  # if don't set it to eval() model, the prediction always varies.

    def inp_to_tensor(self, sentence):
        if self.model_type == 'RNN':
            tokens = self.TEXT.tokenize(sentence)
            indexed = [self.TEXT.vocab.stoi[t] for t in tokens]
            input_tensor = torch.tensor(indexed, device=device).unsqueeze(0)
        elif self.model_type == 'bert':
            tokens = self.model.tokenizer.tokenize(sentence)
            indexed = self.model.tokenizer.convert_tokens_to_ids(tokens)
            #indexed = self.model.tokenizer.add_special_tokens_single_sentence(indexed)   # add [CLS] and [SEP].
            input_tensor = torch.tensor(indexed, device=device).unsqueeze(0)
        return input_tensor

    def predict(self, input_tensor):
        if self.model_type == 'RNN':
            if self.pad:
                length_tensor = torch.LongTensor([input_tensor.shape[1]]).to(device)
                score = self.model(input_tensor, length_tensor)
            else:
                score = self.model(input_tensor)
            label = torch.argmax(score, -1).item() 
        elif self.model_type == 'bert':
            score = self.model(input_tensor)
            label = torch.argmax(score, -1).item() 
        return score, label  # need sigmoid_score ?

    def generate(self, text):
        if isinstance(text, str):
            input = self.inp_to_tensor(text)
        else:
            input = text  # if input is embedding.
            # length = torch.LongTensor([text.shape[1]])
        input = input.to(device)
        score, label = self.predict(input)
        return score, label

    def confidence(self, score, label):  # return the confidence score of a class
        conf = classify(score, -1).squeeze(0).detach().cpu().numpy()[label]
        return conf
    
    def train_bert_model(self, n_epochs, train_data, dev_data):
        lrlast = .00002
        lrmain = .00001
        batch_size = 32
        optimizer = optim.Adam([{"params": self.model.bert.parameters(), "lr": lrmain}, {"params": self.model.fc.parameters(), "lr": lrlast}])
        
        #model_dir = '/home/kuo/code/model/bert_sst.pt'
        
        #criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device)
        #criterion = criterion.to(device)
        self.model = self.model.to(device)
        
        #epochs_num = 5  # let's only train for 5 epochs
        best_valid_loss = float('inf')
        
        for epoch in tqdm(range(n_epochs)):
            start_time = time.time()
            train_loss, train_acc = bert_train(self.model, train_data, optimizer, batch_size, self.criterion)
            valid_loss, valid_acc = bert_evaluate(self.model, dev_data, batch_size, self.criterion)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_dir)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    
    def train_model(self, n_epochs, train_iterator, valid_iterator):
        optimizer = optim.Adam(self.model.parameters())
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        criterion = criterion.to(device)

        # N_EPOCHS = 10
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = train(self.model, train_iterator, optimizer, criterion, pad=self.pad)
            valid_loss, valid_acc = evaluate(self.model, valid_iterator, criterion, pad=self.pad)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_dir)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def main():
    n_epochs = 3
    save_model_dir = '/home/kuo/code/model/bert2.pt'
    RNN_classifier = Classifier(save_model_dir, only_vocab=False, pretrained=0, model_type='bert')
    RNN_classifier.train_bert_model(n_epochs, RNN_classifier.train_iterator, RNN_classifier.valid_iterator)
    #RNN_classifier.train_model(n_epochs, RNN_classifier.train_iterator, RNN_classifier.valid_iterator)

if __name__ == '__main__':
    main()

