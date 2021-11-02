# here defines a variaties of classification models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import datasets_helper
from utils import epoch_time, train, evaluate
import time
from pytorch_transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

embedding_dim = 100
unit = 500
dropout = 0.3
out_dim = 2
min_length = 20
classify = torch.softmax
num_layers = 1
pad = 1


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, unit, output_dim, dropout, pad_idx):

        super(RNN, self).__init__()
        self.hidden_size = unit
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, int(unit/2), num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(unit, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_lengths):
        if type(input_lengths) == list:
            input_lengths = input_lengths
        else:
            input_lengths = input_lengths.cpu()
        # text = [batch size, sent len]
        bsz = input.shape[0]
        if len(input.shape) == 2:
            # text index input
            embedded = self.embedding(input)
        elif len(input.shape) == 3:
            # embedding vectors
            embedded = input
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.rnn(packed_embedded)
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_out)
        # hidden is not batch first yet
        hidden = hidden.permute(1, 0, 2)
        # print('hidden shape: {}'.format(hidden.shape))
        out = self.dropout(torch.cat((hidden[:, -2, :], hidden[:, -1, :]), dim=1))  # last layer, bidirection
        # print('out shape: {}'.format(out.shape))
        return self.fc(out)

    def set_hook(self, layer='rnn'):
        hook_handlers = []

        def hook(module, input, output):
            setattr(module, "text_represent", output)
        for n, m in self.named_modules():
            if n == layer:
                h = m.register_forward_hook(hook)
                hook_handlers.append(h)
        self.hook_handlers = hook_handlers

    def remove_hook(self):
        for h in self.hook_handlers:
            h.remove()

    def get_text_hidden(self, layer='rnn'):
        for n, m in self.named_modules():
            if n == layer:
                text_represent = m.text_represent
                hidden = text_represent[0]
                #hidden = text_represent[1][0]  # rnn output: (out, (hidden, cell))
                #hidden = hidden.permute(1, 0, 2)  # batch first
                #hidden = torch.cat((hidden[:, -2, :], hidden[:, -1, :]), dim=1)
                return hidden


class SumEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_dim, pad_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, out_dim)

    def forward(self, input):
        if len(input.shape) == 2:
            # text index input
            embedded = self.embedding(input)
        elif len(input.shape) == 3:
            # embedding vectors
            embedded = input
        # embedded = self.embedding(input)  # [batch, length]
        sum_emb = torch.sum(embedded, dim = 1)  # sum of all word embeddings -> [batch, dim]
        out = self.fc(sum_emb)
        return out


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx, activation='relu'):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = F.relu

        elif activation == 'softplus':
            self.activation = F.softplus

        else:
            print('only support `relu` and `softplus` for now')

    def forward(self, input):
        # text = [batch size, sent len]
        if len(input.shape) == 2:
            # text index input
            embedded = self.embedding(input)
        elif len(input.shape) == 3:
            # embedding vectors
            embedded = input

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [self.activation(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def set_hook(self, layer='convs.'):
        hook_handlers = []

        def hook(module, input, output):
            setattr(module, "text_represent", output)

        for n, m in self.named_modules():
            if n.startswith(layer):
                h = m.register_forward_hook(hook)
                hook_handlers.append(h)
        self.hook_handlers = hook_handlers

    def get_text_hidden(self, layer='convs.'):
        hiddens = []
        for n, m in self.named_modules():
            if n.startswith(layer):
                text_represent = m.text_represent
                hidden = text_represent  # rnn output: (out, (hidden, cell)
                hiddens.append(hidden)
        # get representation
        conved = [torch.relu(h).squeeze(3) for h in hiddens]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        text_represent = torch.cat(pooled, dim=1)
        return text_represent

    def remove_hook(self):
        for h in self.hook_handlers:
            h.remove()
            
            
class BertParams:
    def __init__(self):
        self.model_name = 'BERTModel.pt'
        self.pretrained = 'bert-base-uncased'
        self.max_sent_length = 254
        self.hidden = 768  # to be checked


class BertBase(nn.Module):
    def __init__(self, out_dim):   # output dimension depends on the task.
        super(BertBase, self).__init__()
        self.Params = BertParams()
        self.tokenizer = BertTokenizer.from_pretrained(self.Params.pretrained, cache_dir='/home/kuo/code/model/bert')   # download pretrained bert model to the cache dir.
        self.tokenizer = BertTokenizer.from_pretrained(self.Params.pretrained, cache_dir='/home/kuo/code/model/bert')
        self.bert = BertModel.from_pretrained(self.Params.pretrained, cache_dir='/home/kuo/code/model/bert')
        #self.bert.eval()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.Params.hidden, out_dim)

    def forward(self, input):
        if len(input.shape) < 3 : 
            bert_out = self.bert(input)
        else:   
            # when the input is already embedding, input.shape == (batch, sequence_length, embedding_dimension)
            # We seperate the embedding step only to simplify the gradient based feature methods, where you can directly feed embedding vectors as input.
            bert_out = self.get_output_from_embedding(input)

        encoded = bert_out[1]  # CLS  # use the pooled output from the last layer.
        encoded = self.dropout(encoded)
        return self.fc(encoded)

    def get_output_from_embedding(self, embedding, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(embedding.shape[0], embedding.shape[1]).to(embedding)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.bert.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
                head_mask = head_mask.to(dtype=next(self.bert.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.bert.config.num_hidden_layers
        encoder_outputs = self.bert.encoder(embedding,
                                             extended_attention_mask,
                                             head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


if __name__ == '__main__':
   pass
