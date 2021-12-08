import pandas as pd
import os
import string
import re

def preprocessing():
    nl_files = ['nl_option_output.txt', 'nl_policy_output.txt']
    rlang_files= ['rlang_option_output.txt', 'rlang_policy_output.txt']
    type_tag = ['option', 'policy']
    accumulated_data = []
    type_counter = 0
    for nl_file, rlang_file in zip(nl_files,rlang_files):
        print('here')
        nl_file = '../data/nl/' + nl_file
        rlang_file = '../data/' + rlang_file
        with open(nl_file) as nl, open(rlang_file) as rlang: 
            for nl_string, rlang_string in zip(nl, rlang):
                accumulated_data.append([nl_string.strip(), rlang_string.strip(), type_tag[type_counter]])
        type_counter+=1
    
    df = pd.DataFrame(accumulated_data, columns=['natural_language', 'rlang', 'string_type'])

    # convert source and target text to Lowercase 
    df.natural_language=df.natural_language.apply(lambda x: x.lower())
    df.rlang=df.rlang.apply(lambda x: x.lower())

    # Remove quotes from source and target text
    df.natural_language=df.natural_language.apply(lambda x: re.sub("'", '', x))
    df.rlang=df.rlang.apply(lambda x: re.sub("'", '', x))

    # create a set of all special characters
    # special_characters= set(string.punctuation)

    # Remove all the special characters
    # df.natural_language = df.natural_language.apply(lambda x: ''.join(char1 for char1 in x if char1 not in special_characters))
    # df.rlang = df.rlang.apply(lambda x: ''.join(char1 for char1 in x if char1 not in special_characters))

    # Remove extra spaces
    df.natural_language=df.natural_language.apply(lambda x: x.strip())
    df.rlang=df.rlang.apply(lambda x: x.strip())
    df.natural_language=df.natural_language.apply(lambda x: re.sub(" +", " ", x))
    df.rlang=df.rlang.apply(lambda x: re.sub(" +", " ", x))

    # Add start and end tokens to target sequences
    df.rlang = df.rlang.apply(lambda x : '[start] '+ x + ' [end]')
    
    df.to_csv('../data/nl_to_rlang_data.csv')

    return df


    

