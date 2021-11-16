import os

from nltk import Tree
from random import randint, seed

script_dir = os.path.dirname(__file__)

def transform(t):
    label = t.label()
    # switch statement based on t.label
    if label == 'Effect':
        return transform_Effect(t)

def transform_Effect(t):
    out_tree = Tree('Effect',[])
    out_tree.append('Effect E')
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            possible_statements = [
                'entails that',
                'states that',
                'suggests that',
                'means that',
                'makes it so that'
            ]
            random_index = randint(0, len(possible_statements) - 1)
            out_tree.append(possible_statements[random_index])
            out_tree.append(transform_VariableName(elt))

        #ignores functors like :, \n, \t
        if label == 'ConditionalStatement':
            possible_statements = [
                'entails that',
                'states that',
                'suggests that',
                'means that',
                'leads to',
                'makes it so that'
            ]
            random_index = randint(0, len(possible_statements) - 1)
            out_tree.append(possible_statements[random_index])
            out_tree.append(transform_ConditionalStatement(elt))
        
        if label == 'Assignment':
            out_tree.append(transform_Assignment(elt))

    return out_tree


#Assignment / Assignment Options 
def transform_Assignment(t):
    out_tree = Tree('BoolExp', [])
    for elt in t:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'AssignmentOptions':
            out_tree.append(transform_AssignmentOptions(elt))
        if label == 'Value':
            out_tree.append(transform_Value(elt))
    return out_tree


def transform_AssignmentOptions(t):
    out_tree = Tree('Assignment',[])
    r = randint(0, 1)
    elt = t[0]
    if elt == '+=':
        possible_plus_statements = [
            'increases by',
            'gets larger by'
        ]
        out_tree.append(possible_plus_statements[randint(0, len(possible_plus_statements) - 1)])
    if elt == '-=':
        possible_minus_statements = [
            'decreases by',
            'gets smaller by'
        ]
        out_tree.append(possible_minus_statements[randint(0, len(possible_minus_statements) - 1)])
    if elt == '*=':
        possible_mult_statements = [
            'increase by a multiple of',
            'multiplies by'
        ]
        out_tree.append(possible_mult_statements[randint(0, len(possible_mult_statements) - 1)])
    if elt == '/=':
        possible_div_statements = [
            'decreases by a multiple of',
            'divides by'
        ]
        out_tree.append(possible_div_statements[randint(0, len(possible_div_statements) - 1)])
    return out_tree


def transform_ConditionalStatement(t):
    out_tree = Tree('ConditionalStatement', [])
    for elt in t:
        label = elt.label()
        if label == 'If':
            out_tree.append(transform_If(elt))
        if label == 'Else':
            out_tree.append(transform_Else(elt))
    return out_tree

def transform_If(t):
    out_tree = Tree('If', ['if'])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolExp':
            out_tree.append(transform_BoolExp(elt))
            out_tree.append(',')
        if label == 'Assignment':
            out_tree.append(transform_Assignment(elt))
    return out_tree

def transform_Else(t):
    out_tree = Tree('Else', [', or'])
    possible_else_statements = [
        'otherwise',
        'if not',
        'as a last resort',
        'if no other options are possible'
    ]
    random_index = randint(0, len(possible_else_statements) - 1)

    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'Assignment':
            out_tree.append(transform_Assignment(elt))
    
    out_tree.append(possible_else_statements[random_index]) # if not, as a last resort, if no other options are possible
    return out_tree


#BoolExp
def transform_BoolExp(t):
    out_tree = Tree('BoolExp', [])
    for elt in t:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolTest':
            out_tree.append(transform_BoolTest(elt))
        if label == 'Value':
            out_tree.append(transform_Value(elt))
    return out_tree


#BoolTest
def transform_BoolTest(t):
    out_tree = Tree('BoolTest',[])
    r = randint(0, 1)
    elt = t[0]
    if elt == '==':
        possible_equals_statements = [
            'is equal to',
            'is exactly the same as'
        ]
        out_tree.append(possible_equals_statements[randint(0, len(possible_equals_statements) - 1)])
    if elt == '!=':
        possible_not_equals_statements = [
            'is not equal to',
            'is not the same as'
        ]
        out_tree.append(possible_not_equals_statements[randint(0, len(possible_not_equals_statements) - 1)])
    if elt == '>':
        possible_greater_statements = [
            'is greater than',
            'is larger than'
        ]
        out_tree.append(possible_greater_statements[randint(0, len(possible_greater_statements) - 1)])
    if elt == '<':
        possible_less_than_statements = [
            'is less than',
            'is smaller than'
        ]
        out_tree.append(possible_less_than_statements[randint(0, len(possible_less_than_statements) - 1)])
    return out_tree



def transform_Value(t):
    return t[0]

def transform_VariableName(t):
    return t[0]

def transform_Conj(t):
    pass




#main function
def main():
    seed(0)

    output_file = "nl_effect_output.txt"
    with open(os.path.join(script_dir, "../data/tokenized_effect_output.txt"), 'r') as f_input:
        with open(os.path.join(script_dir, '../data/nl/nl_effect_output.txt'), 'w') as f_output:
            lines = f_input.readlines()
            total_lines = len(lines)
            print("Writing transform file for EFFECT to ", output_file)
            print(f'Transforming {total_lines} total statements')
            print('This may take a while...\n...')

            for i in range(total_lines):
                if (i % 1000 == 0):
                    print(f'Finished transforming {i}/{total_lines} statements')
                
                tokenized_rlang = lines[i]
                in_tree = Tree.fromstring(tokenized_rlang)
                out_tree = transform(in_tree)
                
                output = ' '.join(out_tree.leaves())
                f_output.write(output + '\n')
                
if __name__ == '__main__':
    main()