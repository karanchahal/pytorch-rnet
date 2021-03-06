import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import coloredlogs, logging
# Create a logger object.
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    ''' Simple GRU Encoder, converts words to embeddings if not using word embeddings
        and then runs them through a 3 layer GRU cell, outputs a probability of the vocabulary,
        adds the conv feature map as the hidden state
    '''
    def __init__(self,vocab_size,embed_size=300,hidden_size=75,num_layers=3,batch_size=1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True,bidirectional=True,dropout=0.2)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)


        self.linear_passage = nn.Linear(hidden_size*2,hidden_size)
        self.linear_question = nn.Linear(hidden_size*2,hidden_size)
        self.linear_hidden = nn.Linear(hidden_size*2,hidden_size)
        self.linear_vt = nn.Linear(hidden_size,hidden_size)
        self.linear_vt_answer_rec = nn.Linear(hidden_size,1)
        self.linear_self_passage_word = nn.Linear(hidden_size*2,hidden_size)
        self.linear_self_passage = nn.Linear(hidden_size*2,hidden_size)

        self.linear_h_p = nn.Linear(hidden_size*2,hidden_size)
        self.linear_h_a = nn.Linear(hidden_size,hidden_size)

        self.gated_attention = nn.Linear(hidden_size*3,hidden_size*3)
        self.gated_attention_self_matching = nn.Linear(hidden_size*3,hidden_size*3)

        self.tanh = nn.Tanh()

        self.gated_attention_rnn = nn.GRU(hidden_size*3, hidden_size, num_layers=1, batch_first=True, bidirectional=True,dropout=0.2) # bidirectional input + c_t of 75 size = 150+75 = 225
        self.gated_attention_self_matching_rnn = nn.GRU(hidden_size*3, hidden_size, num_layers=1, batch_first=True,bidirectional=True,dropout=0.2) # passage word size + c_t = 55 + 75 = 150 
        
        self.answer_recurrent_network = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True) 

        self.init_weights()

    def init_weights(self): 
        '''Initialize the weights'''
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.uniform_(-0.1, 0.1)
                m.bias.data.fill_(0)
            elif isinstance(m,nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)

    def gated_attn(self,passage_word,question):
        

        last_hidden = self.hidden_qa_passage
        last_hidden = torch.cat((last_hidden[0,:,:],last_hidden[1,:,:]),dim=1)

        h = self.linear_hidden(last_hidden)
        p = self.linear_passage(passage_word)
        q = self.linear_question(question)

        a_t = F.softmax(self.linear_vt(self.tanh(q + h + p))) # stj = vt*tanh(question, passage_word, last_hidden_layer) ; a_t = sigmoid(s_t)
        c_t = torch.sum(a_t*q,dim=1) # 1,75
        

        final_inp = torch.cat((passage_word,c_t),dim=1)

        gate = F.sigmoid(self.gated_attention(final_inp))
        gated_input = (final_inp*gate).unsqueeze(1)


        x,self.hidden_qa_passage = self.gated_attention_rnn(gated_input,self.hidden_qa_passage)
        h = self.hidden_qa_passage
        h = torch.cat((h[0,:,:],h[1,:,:]),dim=1)

        return h
    
    def self_matching_attn(self,passage_word,passage):

        last_hidden = self.hidden_self_matching
        last_hidden = torch.cat((last_hidden[0,:,:],last_hidden[1,:,:]),dim=1)

        p_w = self.linear_self_passage_word(passage_word)
        p = self.linear_self_passage(passage)
        inp = p + p_w

        a_t = F.softmax(self.linear_vt(self.tanh(p + p_w))) # stj = vt*tanh(passage,passage_word) ; a_t = sigmoid(s_t)
        c_t = torch.sum(a_t*p,dim=1)
        
        final_inp = torch.cat((passage_word,c_t),dim=1)
        gate = F.sigmoid(self.gated_attention_self_matching(final_inp))
        gated_input = (final_inp*gate).unsqueeze(1)

        x,self.hidden_self_matching = self.gated_attention_self_matching_rnn(gated_input,self.hidden_self_matching)

        h = self.hidden_self_matching
        h = torch.cat((h[0,:,:],h[1,:,:]),dim=1)
        
        return h
    
    def pointer_network(self,h_p):
        
        hidden_layer = self.hidden_ans_recurrent_pointer_network # 1,1,75
        
        h_p = self.linear_h_p(h_p)
        h_a = self.linear_h_a(hidden_layer)
        
        a = self.linear_vt_answer_rec(self.tanh(h_p + h_a))
        
        a_t = a.squeeze(2) #1, x

        pointer = F.log_softmax(a_t)
        c_t = torch.sum(a*h_p,dim=1).unsqueeze(0) # 1,1,75
        x , self.hidden_ans_recurrent_pointer_network = self.answer_recurrent_network(c_t,self.hidden_ans_recurrent_pointer_network)
        
        return pointer



    def build_question_aware_passage(self,passage,question):
        passage_words = passage.size(1)
        v = None
        for i in range(passage_words):
            passage_word = passage[:,i,:]
            x = self.gated_attn(passage_word,question)
            
            if isinstance(v,torch.autograd.Variable):
                v = torch.cat((v,x),dim=0)
            else:
                v = x
        
        v = v.unsqueeze(0)
        return v
    
    def build_self_matching_attention(self,passage_1,passage_2):
        passage_words = passage_1.size(1)
        h = None
        for i in range(passage_words):
            passage_word = passage_1[:,i,:]
            x = self.self_matching_attn(passage_word,passage_2)
            
            if isinstance(h,torch.autograd.Variable):
                h = torch.cat((h,x),dim=0)
            else:
                h = x

        h = h.unsqueeze(0)
        return h
    

    def forward(self,passage,question):
        

        passage = self.embedding(passage) # 1, 149 , 75
        question = self.embedding(question) # 1, 13, 75

       
        # packed = pack_padded_sequence(x,lengths,batch_first=True)
        # x, _ = self.lstm(x)
        u_p,self.hidden_p = self.gru(passage,self.hidden_p) # 1, 149 , 150 , 75*2 as bidirectional
        u_q,self.hidden_q = self.gru(question,self.hidden_q)# 1, 13 , 150 , 75*2 as bidirectional
        
        self.hidden_ans_recurrent_pointer_network = self.initHiddenAnsReccNetwork(u_q) # batch size by hidden state size
        
        v = self.build_question_aware_passage(u_p,u_q)
        h = self.build_self_matching_attention(v,v)

        # print(h.size()) # batch size, num words in passage , hidden vector size (1, 149, 75)
        start_index = self.pointer_network(h)
        end_index = self.pointer_network(h)

        return start_index,end_index
    
    def initHidden(self,question):
        self.hidden_p = self.initHiddenUtil(layers=3)
        self.hidden_q = self.initHiddenUtil(layers=3)
        self.hidden_qa_passage = self.initHiddenUtil(layers=1)
        self.hidden_self_matching = self.initHiddenUtil(layers=1)
        
    
    def initHiddenAnsReccNetwork(self,question):
        

        q = self.linear_question(question)   
        a_t = F.sigmoid(self.linear_vt(self.tanh(q)))
        r_q = torch.sum(a_t*q,dim=1).unsqueeze(0)
        return r_q
        
    def initHiddenUtil(self,layers):
        # multiplied by 2 as all rnns are bi directional presently
        return Variable(torch.zeros(layers*2,self.batch_size,self.hidden_size))
   
