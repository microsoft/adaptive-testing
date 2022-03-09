import re

p = re.compile(r"(fuck|shit|bitch|nigger)", re.IGNORECASE)

def clean_string(string):
    return p.sub("***", string)