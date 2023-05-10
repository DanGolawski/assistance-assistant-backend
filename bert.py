import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

def get_answer(question, paragraph):
   model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
   tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

               
   encoding = tokenizer.__call__(text=question,text_pair=paragraph)

   inputs = encoding['input_ids']  #Token embeddings
   sentence_embedding = encoding['token_type_ids']  #Segment embeddings
   tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

   start_scores, end_scores = model(return_dict=False, input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))

   start_index = torch.argmax(start_scores)

   end_index = torch.argmax(end_scores)

   answer = ' '.join(tokens[start_index:end_index+1])


   corrected_answer = ''

   for word in answer.split():
      
      #If it's a subword token
      if word[0:2] == '##':
         corrected_answer += word[2:]
      else:
         corrected_answer += ' ' + word

   return corrected_answer