import transformers
import torch
import numpy as np
import adatest

_embedding_cache = {}

def embed(strings, normalize=True):

    # this code block is from the sentence_transformers documentation
    if adatest.embedding_model is None:

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Load model from HuggingFace Hub
        tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/stsb-roberta-base-v2')
        model = transformers.AutoModel.from_pretrained('sentence-transformers/stsb-roberta-base-v2')

        # Tokenize sentences
        def embed_model(sentences):
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, max pooling.
            return mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

        adatest.embedding_model = embed_model

    # find which strings are not in the cache
    new_strings = []
    for s in strings:
        if s not in _embedding_cache:
            new_strings.append(s)
            _embedding_cache[s] = None # so we don't embed the same string twice
    
    # embed the new strings
    if len(new_strings) > 0:
        new_embeds = adatest.embedding_model(new_strings)
        for i,s in enumerate(new_strings):
            if normalize:
                _embedding_cache[s] = new_embeds[i] / np.linalg.norm(new_embeds[i])
            else:
                _embedding_cache[s] = new_embeds[i]
    
    return [_embedding_cache[s] for s in strings]

def cos_sim(a, b):
    """ Cosine distance between two vectors.
    
    Based on the sentence_transformers implementation.
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))