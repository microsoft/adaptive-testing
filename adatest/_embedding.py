import transformers
import torch
import numpy as np
import adatest
import PIL

_embedding_cache = {}

def embed(strings, normalize=True):

    # find which strings are not in the cache
    new_text_strings = []
    new_image_urls = []
    for s in strings:
        if s not in _embedding_cache:
            if s.startswith("__IMAGE="):
                new_image_urls.append(s)
            else:
                new_text_strings.append(s)
            _embedding_cache[s] = None # so we don't embed the same string twice
    
    # embed the new text strings
    if len(new_text_strings) > 0:
        new_embeds = _text_embedding_model()(new_text_strings)
        for i,s in enumerate(new_text_strings):
            if normalize:
                _embedding_cache[s] = new_embeds[i] / np.linalg.norm(new_embeds[i])
            else:
                _embedding_cache[s] = new_embeds[i]

    # embed the new image urls
    if len(new_image_urls) > 0:
        new_embeds = _image_embedding_model()([url[8:] for url in new_image_urls])
        for i,s in enumerate(new_image_urls):
            if normalize:
                _embedding_cache[s] = new_embeds[i] / np.linalg.norm(new_embeds[i])
            else:
                _embedding_cache[s] = new_embeds[i]
    
    return [_embedding_cache[s] for s in strings]

def _text_embedding_model():
    """ Get the text embedding model.
    
    Much of this code block is from the sentence_transformers documentation.
    """
    if adatest.text_embedding_model is None:

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
        
        adatest.text_embedding_model = embed_model
    
    return adatest.text_embedding_model

def _image_embedding_model():
    if adatest.image_embedding_model is None:
        import clip  # pylint: disable=import-outside-toplevel

        model, preprocess = clip.load("ViT-L/14", device="cpu", jit=True)

        def embed_model(urls):
            with torch.no_grad():
                out = []
                for url in urls:
                    image = adatest.utils.get_image(url)
                    image_emb = model.encode_image(preprocess(image).unsqueeze(0).to("cpu"))
                    image_emb /= image_emb.norm(dim=-1, keepdim=True)
                    image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
                    out.append(image_emb)
            return np.vstack(out)
        
        adatest.image_embedding_model = embed_model
    
    return adatest.image_embedding_model

def cos_sim(a, b):
    """ Cosine distance between two vectors.
    
    Based on the sentence_transformers implementation.
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))