import random
from nbformat import read
import numpy as np







def get_few_shot_instances():

    few_shot_instances = [
    '''Concept: dog
    -
    Parent: animal
    -
    Sibling: cat
    Sibling: dolphin
    Sibling: elephant
    -
    Children: chihuahua
    Children: labrador
    Children: golden retriever''',
    '''Concept: red
    -
    Sibling: blue
    Sibling: yellow
    Sibling: green
    -
    Parent: color
    -
    Children: crimson
    Children: magenta
    Children: maroon''',
    '''Concept: person
    -
    Parent: named entity
    -
    Children: nickname
    Children: last name
    Children: first name
    -
    Sibling: location
    Sibling: company
    Sibling: date
    '''
    ]
    few_shot_instances = [i.replace("Sibling", "Same level") for i in few_shot_instances]
    few_shot_instances = [i.replace("Parent", "Superset") for i in few_shot_instances]
    few_shot_instances = [i.replace("Children", "Subset") for i in few_shot_instances]

    return few_shot_instances


def make_prompt(concept, 
                few_shot_instances='',
                problem = ''):
    if problem: 
        instruction=' I am analysing ' + problem +'. Given a concept in %s, give topics related to the concept at different levels. ' %problem
    else : 
        instruction =  'Given a concept topic, give topics related to the concept at different levels. '
    prompt = '' 
    if instruction:
        prompt += instruction + '\n-------\n'
    if len(few_shot_instances)>0:
        tmp = [x for x in few_shot_instances]
        random.shuffle(tmp)
        for x in tmp:
            prompt += x + '\n-------\n'
    if concept: 
        prompt += '%s\n' % concept
    
    return prompt


def get_parent(response):


    lines = [choice["text"] for choice in response["choices"]]
    scores = [choice["logprobs"]['token_logprobs'] for choice in response["choices"]]
    suggested = []
    for i, line in enumerate(lines):
        suggested.append(line)

    #log.debug("suggested_tests", suggested_tests)
    a= list(zip(suggested, [np.sum(s) for s in scores]))
    print(a)
    ret = []
    for x in [x[0] for x in sorted(a, key=lambda z: -z[1])]:

        xi = x.split('\n')
        for i in xi: 
            if ('Main topic:' in i) :
                for text in i.split("'"):
                    if ('Main topic:' not in text) and (len(text)>2) and (text not in ret):
                        ret.append(text)
    return ret



def just_parent(response, parent='', children='', sibling='',  nprompts=3, n=5):

    lines = [choice["text"] for choice in response["choices"]]
    scores = [choice["logprobs"]['token_logprobs'] for choice in response["choices"]]
    suggested = []
    for i, line in enumerate(lines):
        suggested.append(line)

    #log.debug("suggested_tests", suggested_tests)
    a= list(zip(suggested, [np.sum(s) for s in scores]))#, [np.mean(s) for s in scores]))
#     prompts = [make_prompt(concept, parent, children) for _ in range(nprompts)]
#     a = complete_prompt(prompts, stop='-')
#     ret = []
#     for x in [x[0] for x in sorted(a, key=lambda z: -z[1])]:
#         if x not in ret:
#             ret.append(x)

    ret = []
    for x in [x[0] for x in sorted(a, key=lambda z: -z[1])]:
        # print(x)
        # xi = x.split('\n-\n')
        # for i in xi: 
        #     if (i not in ret) and ('Superset:' in i):
        #         ret.append(i.split('Superset: ')[1])
        if 'Superset: ' in x:
            z  = x.split('Superset: ')[1].split('\n')[0]
            # print(x.split('Superset:'))
            ret.append(z)
    return set(ret)

def parent_and_siblings(response,parent='', children='', sibling='',nprompts=3, n=5):
    lines = [choice["text"] for choice in response["choices"]]
    scores = [choice["logprobs"]['token_logprobs'] for choice in response["choices"]]
    suggested = []
    for i, line in enumerate(lines):
        suggested.append(line)

    #log.debug("suggested_tests", suggested_tests)
    a= list(zip(suggested, [np.sum(s) for s in scores]))#, [np.mean(s) for s in scores]))
#     print(a)
    ret = []
    for x in [x[0] for x in sorted(a, key=lambda z: -z[1])]:
        if x not in ret:
            ret.append(x)
    return ret

def just_siblings(response,parent='', children='',sibling='', nprompts=3, n=15):
    lines = [choice["text"] for choice in response["choices"]]
    scores = [choice["logprobs"]['token_logprobs'] for choice in response["choices"]]
    suggested = []
    for i, line in enumerate(lines):
        suggested.append(line)

    #log.debug("suggested_tests", suggested_tests)
    a= list(zip(suggested, [np.sum(s) for s in scores]))#, [np.mean(s) for s in scores]))
    ret = []
#     print(a)
    for x in [x[0] for x in sorted(a, key=lambda z: -z[1])]:
        # xi = x.split('\n')
        # for i in xi: 
        #     if ('Same level:' in i) and (i not in ret):
        #       ret.append(i.split('Same level: ')[1])
        if 'Same level: ' in x:
            z  = x.split('Same level: ')[1].split('\n')[0]
            print(x.split('Same level:'))
            ret.append(z)
    return set(ret)

  

def just_children(response,parent='', children='',sibling='', nprompts=3, n=15):
    lines = [choice["text"] for choice in response["choices"]]
    scores = [choice["logprobs"]['token_logprobs'] for choice in response["choices"]]
    suggested = []
    for i, line in enumerate(lines):
        suggested.append(line)

    #log.debug("suggested_tests", suggested_tests)
    a= list(zip(suggested, [np.sum(s) for s in scores]))#, [np.mean(s) for s in scores]))
    ret = []
#     print(a)
    for x in [x[0] for x in sorted(a, key=lambda z: -z[1])]:
        # xi = x.split('\n')
        # for i in xi: 
        #     if ('Same level:' in i) and (i not in ret):
        #       ret.append(i.split('Same level: ')[1])
        if 'Subset: ' in x:
            z  = x.split('Subset: ')[1].split('\n')[0]
            print(x.split('Subset:'))
            ret.append(z)
    return set(ret)

    # def make_prompt(concept, 
#                 parent='',
#                 children= '', 
#                 sibling='',
#                 few_shot_instances='',
#                 instruction='Given a concept, give me the parent concept and three examples of sibling concepts and three examples of children concepts.'):
#     # few_shot_instances: list of (scenario, question, harm_type(optional))
#     prompt = '' 
#     if instruction:
#         prompt += instruction + '\n-------\n'
#     if few_shot_instances:
#         tmp = [x for x in few_shot_instances]
#         random.shuffle(tmp)
#         for x in tmp:
#             prompt += x + '\n-------\n'
#     if concept: 
#         prompt += 'Concept: %s\n-\n' % concept
#     if parent:
#         prompt += 'Superset: %s\n-\n' % parent
#     if len(children) > 0: 
#         for i in children[:-1]: 
#             prompt += 'Subset: %s\n' %i
#         prompt += 'Subset: %s\n-\n' %children[-1]
#     if len(sibling) > 0: 
#         for i in sibling[:-1]: 
#             prompt += 'Same level: %s\n' %i
#         prompt += 'Same level: %s\n-\n' %sibling[-1]
#     return prompt
