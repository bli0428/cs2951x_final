import os

from nltk import Tree
from random import randint, seed

script_dir = os.path.dirname(__file__)

def transform(t):
    label = t.label()
    # switch statement based on t.label
    if label == 'Option':
        return transform_Option(t)

def transform_Option(t):
    out_tree = Tree('Option',[])
    # out_tree.append('you have the option to')
    for i, elt in enumerate(t[1:]):
        if elt == 'init':
            out_tree.append(transform_Init(elt))
        if type(elt) is not str and elt.label() == 'BoolExp':
            out_tree.append(transform_BoolExp(elt))
        if type(elt) is not str and elt.label() == 'Policy':
            out_tree.append(transform_SubPolicy(elt))
        if elt == 'until':
            out_tree.append(transform_Termination(t[i+2]))
    return out_tree
def transform_Init(t):
    possible_statements = [
                'if',
                'when',
                'once',
                'whenever'
            ]
    random_index = randint(0, len(possible_statements) - 1)
    return possible_statements[random_index]

def transform_SubPolicy(t):
    out_tree = Tree('Policy', [])
    possible_statements = [
        'you should',
        'take the option to',
        'you are able to',
        'you have the choice to'
    ]
    random_index = randint(0, len(possible_statements) - 1)

    out_tree.append(possible_statements[random_index])

    for elt in t[1:]:
        label = elt.label()
        if label == 'ExecuteVariableName':
            out_tree.append(transform_ExecuteVariableName(elt))
    
    return out_tree

def transform_Termination(t):
    possible_statements = [
                'until',
                'until finally',
            ]
    random_index = randint(0, len(possible_statements) - 1)
    return possible_statements[random_index]

def transform_Execute(t):
    print(t)
    if len(t) <= 1:
        return
    out_tree = Tree('Execute', ['do', t[1]])
    return out_tree

def transform_ExecuteVariableName(t):
    return Tree('Execute',[t[0]])

def transform_BoolExp(t):
    out_tree = Tree('BoolExp', [])
    for elt in t:
        label = elt.label()
        if label == 'VariableName' or label == 'VariableName2':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolTest':
            out_tree.append(transform_BoolTest(elt))
        if label == 'Value':
            out_tree.append(transform_Value(elt))
    return out_tree


def transform_BoolTest(t):
    out_tree = Tree('BoolTest',[])
    r = randint(0, 1)
    elt = t[0]
    if elt == '==':
        possible_equals_statements = [
            'is equal to',
            'is'
        ]
        out_tree.append(possible_equals_statements[randint(0, len(possible_equals_statements) - 1)])
    if elt == '!=':
        possible_not_equals_statements = [
            'is not equal to',
            'is not'
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
    if elt == '<=':
        possible_less_than_equal_statements = [
            'is at most',
            'is less than or equal to'
        ]
        out_tree.append(possible_less_than_equal_statements[randint(0, len(possible_less_than_equal_statements) - 1)])
    if elt == '>=':
        possible_greater_than_equal_statements = [
            'is at least',
            'is greater than or equal to'
        ]
        out_tree.append(possible_greater_than_equal_statements[randint(0, len(possible_greater_than_equal_statements) - 1)])
    return out_tree

def transform_Value(t):
    return t[0]

def transform_VariableName(t):
    return t[0]

def main():
    seed(0)

    output_file = "nl_option_output.txt"
    with open(os.path.join(script_dir, "../data/tokenized_option_output.txt"), 'r') as f_input:
        with open(os.path.join(script_dir, f"../data/nl/{output_file}"), 'w') as f_output:
            lines = f_input.readlines()
            total_lines = len(lines)
            print(f"Writing transform file for OPTION to ", output_file)
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