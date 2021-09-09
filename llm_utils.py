# util for getting contextual embeddings from
# large language models like BERT, RoBERTa, &c.
# based on https://huggingface.co/transformers/quickstart.html

import numpy as np
import math
import torch
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel, \
                         BertTokenizer, BertTokenizerFast, BertModel

def get_contextual_embeddings_batched_just_CLS_token(contexts, questions, model_name, use_gpu):

    print('Torch version:', torch.__version__)

    if model_name == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    elif model_name == 'roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('roberta-large')
    elif model_name == 'roberta-large-mnli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-mnli_fine-tuned')
    elif model_name == 'roberta-large-qnli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-qnli_fine-tuned')
    elif model_name == 'roberta-large-snli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-snli_fine-tuned')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()
    if use_gpu:
        model.to('cuda')

    batch_size = 32
    max_length = 80
    num_docs = len(contexts)
    unrolled_num_features = model.config.hidden_size

    docs_by_hidden_features = np.empty((num_docs, unrolled_num_features))

    num_batches = math.ceil(float(num_docs) / float(batch_size))
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_docs)
        elements_in_batch = end_idx - start_idx
        
        c_batch = contexts[start_idx:end_idx]
        q_batch = questions[start_idx:end_idx]
        tokenized_output = tokenizer(c_batch, q_batch, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        #print(tokenized_output)
        tokens_tensor = tokenized_output['input_ids']
        # tell the model to ignore the padding we added to get uniform sentence size in the batch
        attention_tensor = tokenized_output['attention_mask']
        if batch % 10 == 0:
            print('Batch indices: [{}, {}) / {}'.format(start_idx, end_idx, num_docs))

        # get everything on the gpu
        if use_gpu:
            tokens_tensor = tokens_tensor.to('cuda')
            attention_tensor = attention_tensor.to('cuda')
        # BERT needs token_type_ids to differentiate context from question
        if model_name == 'bert':
            segments_tensors = tokenized_output['token_type_ids']
            if use_gpu:
                segments_tensors = segments_tensors.to('cuda')

        # Predict hidden states features for each layer
        with torch.no_grad():
            if model_name == 'bert':
                outputs = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor)
            else:
                outputs = model(tokens_tensor, attention_mask=attention_tensor)
                
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers = outputs[0]
        # bring the hidden layers back over to the cpu
        if use_gpu:
            encoded_layers = encoded_layers.cpu()
        encoded_CLS = encoded_layers[:, 0, :]
        # We have encoded the [CLS] that starts our input sequence in a FloatTensor of shape (batch size, model hidden dimension)
        assert tuple(encoded_CLS.shape) == (elements_in_batch, model.config.hidden_size)
        
        # unroll and store
        docs_by_hidden_features[start_idx:end_idx, :] = encoded_CLS


    print('cleaning up!')
    if use_gpu:
        del tokens_tensor
        del attention_tensor
        if model_name == 'bert':
            del segments_tensors
        del outputs
        del encoded_layers
        torch.cuda.empty_cache() 

    print('done!!')
    return docs_by_hidden_features


def get_contextual_embeddings_batched_mean_hidden_tokens(contexts, questions, model_name, use_gpu):

    print('Torch version:', torch.__version__)

    if model_name == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    elif model_name == 'roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('roberta-large')
    elif model_name == 'roberta-large-mnli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-mnli_fine-tuned')
    elif model_name == 'roberta-large-qnli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-qnli_fine-tuned')
    elif model_name == 'roberta-large-snli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-snli_fine-tuned')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()
    if use_gpu:
        model.to('cuda')

    batch_size = 32
    max_length = 80
    num_docs = len(contexts)
    unrolled_num_features = model.config.hidden_size

    docs_by_hidden_features = np.empty((num_docs, unrolled_num_features))

    num_batches = math.ceil(float(num_docs) / float(batch_size))
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_docs)
        elements_in_batch = end_idx - start_idx
        
        c_batch = contexts[start_idx:end_idx]
        q_batch = questions[start_idx:end_idx]
        tokenized_output = tokenizer(c_batch, q_batch, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        #print(tokenized_output)
        tokens_tensor = tokenized_output['input_ids']
        # tell the model to ignore the padding we added to get uniform sentence size in the batch
        attention_tensor = tokenized_output['attention_mask']
        if batch % 10 == 0:
            print('Batch indices: [{}, {}) / {}'.format(start_idx, end_idx, num_docs))

        # get everything on the gpu
        if use_gpu:
            tokens_tensor = tokens_tensor.to('cuda')
            attention_tensor = attention_tensor.to('cuda')
        # BERT needs token_type_ids to differentiate context from question
        if model_name == 'bert':
            segments_tensors = tokenized_output['token_type_ids']
            if use_gpu:
                segments_tensors = segments_tensors.to('cuda')

        # Predict hidden states features for each layer
        with torch.no_grad():
            if model_name == 'bert':
                outputs = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor)
            else:
                outputs = model(tokens_tensor, attention_mask=attention_tensor)
                
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers = outputs[0]
        # bring the hidden layers back over to the cpu
        if use_gpu:
            encoded_layers = encoded_layers.cpu()
        encoded_mean = torch.mean(encoded_layers, 1)
        # We have encoded the [CLS] that starts our input sequence in a FloatTensor of shape (batch size, model hidden dimension)
        assert tuple(encoded_mean.shape) == (elements_in_batch, model.config.hidden_size)
        
        # unroll and store
        docs_by_hidden_features[start_idx:end_idx, :] = encoded_mean


    print('cleaning up!')
    if use_gpu:
        del tokens_tensor
        del attention_tensor
        if model_name == 'bert':
            del segments_tensors
        del outputs
        del encoded_layers
        torch.cuda.empty_cache() 

    print('done!!')
    return docs_by_hidden_features


def get_contextual_embeddings_batched(contexts, questions, model_name, use_gpu):

    print('Torch version:', torch.__version__)

    if model_name == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    elif model_name == 'roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('roberta-large')
    elif model_name == 'roberta-large-mnli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-mnli_fine-tuned')
    elif model_name == 'roberta-large-qnli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-qnli_fine-tuned')
    elif model_name == 'roberta-large-snli':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('./roberta-large-snli_fine-tuned')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()
    if use_gpu:
        model.to('cuda')

    batch_size = 32
    max_length = 80
    num_docs = len(contexts)
    unrolled_num_features = max_length * model.config.hidden_size

    docs_by_hidden_features = np.empty((num_docs, unrolled_num_features))

    num_batches = math.ceil(float(num_docs) / float(batch_size))
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_docs)
        elements_in_batch = end_idx - start_idx
        
        c_batch = contexts[start_idx:end_idx]
        q_batch = questions[start_idx:end_idx]
        tokenized_output = tokenizer(c_batch, q_batch, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        
        tokens_tensor = tokenized_output['input_ids']
        # tell the model to ignore the padding we added to get uniform sentence size in the batch
        attention_tensor = tokenized_output['attention_mask']
        if batch % 10 == 0:
            print('Batch indices: [{}, {}) / {}'.format(start_idx, end_idx, num_docs))
        

        # get everything on the gpu
        if use_gpu:
            tokens_tensor = tokens_tensor.to('cuda')
            attention_tensor = attention_tensor.to('cuda')
        # BERT needs token_type_ids to differentiate context from question
        if model_name == 'bert':
            segments_tensors = tokenized_output['token_type_ids']
            if use_gpu:
                segments_tensors = segments_tensors.to('cuda')

        # Predict hidden states features for each layer
        with torch.no_grad():
            if model_name == 'bert':
                outputs = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor)
            else:
                outputs = model(tokens_tensor, attention_mask=attention_tensor)
                
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers = outputs[0]
        # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
        assert tuple(encoded_layers.shape) == (elements_in_batch, max_length, model.config.hidden_size)
        # bring the hidden layers back over to the cpu
        if use_gpu:
            encoded_layers = encoded_layers.cpu()
        # unroll and store
        docs_by_hidden_features[start_idx:end_idx, :] = np.reshape(encoded_layers, (elements_in_batch, unrolled_num_features))
        
    print('cleaning up!')
    if use_gpu:
        del tokens_tensor
        del attention_tensor
        if model_name == 'bert':
            del segments_tensors
        del outputs
        del encoded_layers
        torch.cuda.empty_cache() 

    print('done!!')
    return docs_by_hidden_features

