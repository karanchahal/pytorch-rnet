import json
from word_model import Vocab ,WordModel
import pickle
import coloredlogs, logging
from encoder import Encoder
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')
coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(levelname)s %(message)s')

dataset = json.load(open('data/dev-v1.1.json'))

word_model = WordModel()
logger.warning('Loading Vocab ...')
word_model.load_vocab()
vocab_size = word_model.vocab.length()
encoder = Encoder(vocab_size=vocab_size)
optimiser = torch.optim.SGD(encoder.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

def train_model(context,question,answer,target_start,target_end):
    context,question,answer = Variable(context),Variable(question),Variable(answer)
    context = context.unsqueeze(0)
    question = question.unsqueeze(0)
    answer = answer.unsqueeze(0)
    
    optimiser.zero_grad()
    encoder.initHidden(question)
   
    target_start = Variable(torch.LongTensor([target_start]))
    target_end = Variable(torch.LongTensor([target_end]))
    
   
    s,e = encoder(context,question)

    loss1 = criterion(s,target_start)
    loss2  = criterion(e,target_end)

    loss = loss1 + loss2 # ?? what needs to be done to minimise loss 

    print(loss.data[0])
    loss.backward()
    optimiser.step()

for input in dataset['data']:
    for paragraph in input['paragraphs']:
        context = paragraph['context']
        context_ids = word_model.parse(context)

        questions_answers = paragraph['qas']
        for qa in questions_answers:
            print(qa['question'])
            question = qa['question']
            question_ids = word_model.parse(question,qa=True)

            answers = qa['answers']
            while True:
                for ans in answers:
                    ans_start = ans['answer_start']
                    ans_text = ans['text']
                    ans_end = ans_start + len(ans_text)
                    # print(context[ans_start:ans_end])
                
                    ans_ids = word_model.parse(ans_text,qa=True)
                
                    target_start,target_end = word_model.get_target_ids(context,context_ids,ans_start,ans_end)

                    if target_start != -1:
                        train_model(context_ids,question_ids,ans_ids,target_start,target_end)

                #     break
                # break 
        break
    break
