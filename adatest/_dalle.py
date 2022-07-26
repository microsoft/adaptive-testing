import io
import os
import json
import requests
import time
import pprint
from PIL import Image
import random
# from IPython import display
from base64 import b64decode
import urllib3
import string
import random
import uuid
import logging

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUBKEY = os.environ.get("SUBKEY")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

auth_headers_openai = {
    "authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

auth_headers_babel = headers = {
    'Ocp-Apim-Subscription-Key': SUBKEY,
}

http = urllib3.PoolManager()

def generate_image(prompt, image_type, batch_size=1, seed=None, retry=True):
    """
    prompt: text
    image_type: illustration, photo or unknown
    """
    if seed is not None:
        payload = {
            "image_type": image_type,
            "batch_size": batch_size,
            "caption": prompt,
            "seed": seed
        }
    else:
        payload = {
            "image_type": image_type,
            "batch_size": batch_size,
            "caption": prompt
        }

    def send_dalle_request():
        log.debug(f"Sending DALLE request: {payload}")
        return http.request('POST',
                "https://babel-dalle2-access.azure-api.net/dalle2/text2im",
                payload, auth_headers_babel)

    response = send_dalle_request()
    log.debug(f"DALLE response status {response.status} for prompt {prompt}")

    if (response.status == 408 or response.status == 429) and retry:
        retry_count = 0
        while retry_count < 3:
            log.debug(f"DALLE response status {response.status}. Retrying in 15 seconds...")
            time.sleep(15)
            response = send_dalle_request()
            if response.status != 408 and response.status != 429:
                break
            retry_count += 1

    if response.status != 200:
        log.debug(f"DALLE response error. Status {response.status}, reason {response.reason}, msg {response.msg}")
        raise Exception(f"Error generating image. Status {response.status}, reason {response.reason}, msg {response.msg}")
    
    result = json.loads(response.data)
    result["caption"] = prompt
    result["image_type"] = image_type
    return result


# def inspect_image(binary_image, size=(200, 200)):
#     buffer = io.BytesIO(binary_image)
#     buffer.seek(0)
#     if size is not None:
#         image = Image.open(buffer).resize(size)
#     else:
#         image = Image.open(buffer)

#     display(image)

def generate_images_from_prompt(prompt, image_type, batch_size=1, seed=None):
    result = generate_image(prompt, image_type, batch_size=batch_size, seed=seed)
    return result


def save_image_to_disk(b64_image, path, filename=None):
    os.makedirs(path, exist_ok=True)
    bytes = b64decode(b64_image)
    if filename is None:
        filename = str(uuid.uuid4()) + ".png"
    filepath = os.path.join(path, filename)
    with open(filepath, "wb") as f:
        f.write(bytes)
    log.debug(f"Saved image to {filepath}")
    return filename


def generate_novel_topical_prompts(example_prompts, headers, topic="", model="text-davinci-001", max_tokens=100):
    examples = list(map(lambda e: f'"{e}"', example_prompts))
    prompt = ""
    for example in examples:
        prompt = prompt + topic + ":\n" + example + "\n\n"
    
    prompt = prompt + topic + ':\n"'

    result = requests.post(
        "https://api.openai.com/v1/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "n": 10,
            "stream": False,
            "logprobs": None,
            "stop": '"'
        }),
        headers=headers)

    result_json = result.json()
    choices = example_prompts + list(map(lambda c: c["text"], result_json["choices"]))

    return choices


def random_perturbation(phrase):
    p = 0.7
    new_phrase = []
    words = phrase.split(' ')
    for word in words:
        outcome = random.random()
        if outcome <= p:
            ix = random.choice(range(len(word)))
            new_word = ''.join([word[w] if w != ix else random.choice(string.ascii_letters) for w in range(len(word))])
            new_phrase.append(new_word)
        else:
            new_phrase.append(word)

    new_phrase = ' '.join([w for w in new_phrase])
    
    return new_phrase
