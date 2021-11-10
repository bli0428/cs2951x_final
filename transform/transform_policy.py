import os

from nltk import Tree
from random import randint, seed

script_dir = os.path.dirname(__file__)

def transform(t):
    label = t.label()
    # switch statement based on t.label
    if label == 'Program':
        return transform_Program(t[0])

def transform_Program(t):
    label = t.label()

    if label == 'Statement':
        return transform_Statement(t[0])

def transform_Statement(t):
    label = t.label()
    if label == 'Policy':
        return transform_Policy(t)

def transform_Policy(t):
    out_tree = Tree('Policy',[])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            possible_statements = [
                'entails that you',
                'states that you should',
                'suggests that it would be advantageous to',
                'means that it would be good to',
                'is a strategy where you',
            ]
            random_index = randint(0, len(possible_statements) - 1)

            out_tree.append(transform_VariableName(elt))
            out_tree.append(possible_statements[random_index])

        #ignores functors like :, \n, \t
        if label == 'ConditionalExecute':
            out_tree.append(transform_ConditionalExecute(elt))
    return out_tree

def transform_ConditionalExecute(t):
    out_tree = Tree('ConditionalExecute', [])
    for elt in t:
        label = elt.label()
        if label == 'If':
            out_tree.append(transform_If(elt))
        if label == 'Elif':
            out_tree.append(transform_Elif(elt))
        if label == 'Else':
            out_tree.append(transform_Else(elt))
    return out_tree

#note: current structure of if statement parse clashes with this because 'Execute' is included as string literal.
def transform_If(t):
    out_tree = Tree('If', ['if'])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolExp':
            out_tree.append(transform_BoolExp(elt))
        if label == 'Execute':
            out_tree.insert(0, transform_Execute(elt))
    return out_tree

def transform_Elif(t):
    out_tree = Tree('Elif', ['otherwise, if'])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'BoolExp':
            out_tree.append(transform_BoolExp(elt))
        if label == 'Execute':
            out_tree.append(transform_Execute(elt))
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
        if label == 'Execute':
            out_tree.append(transform_Execute(elt))
    
    out_tree.append(possible_else_statements[random_index]) # if not, as a last resort, if no other options are possible
    return out_tree

def transform_Execute(t):
    if len(t) <= 1:
        return
    out_tree = Tree('Execute', ['do', t[1]])
    return out_tree

#should include conjs too, but recursion errors means I put that off.
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
    pass

def transform_VariableName(t):
    return t[0]

def transform_Conj(t):
    pass

def main():
    seed(0)

    output_file = "nl_policy_output.txt"
    with open(os.path.join(script_dir, "../data/tokenized_policy_output.txt"), 'r') as f_input:
        with open(os.path.join(script_dir, f"../data/nl/{output_file}"), 'w') as f_output:
            lines = f_input.readlines()
            total_lines = len(lines)
            print(f"Writing tokenized file for POLICY to ", output_file)
            print(f'Tokenizing {total_lines} total statements')
            print('This may take a while...\n...')

            for i in range(total_lines):
                if (i % 1000 == 0):
                    print(f'Finished parsing {i}/{total_lines} statements')
                
                tokenized_rlang = lines[i]
                in_tree = Tree.fromstring(tokenized_rlang)
                out_tree = transform(in_tree)
                
                output = ' '.join(out_tree.leaves())
                f_output.write(output + '\n')
                
if __name__ == '__main__':
    main()