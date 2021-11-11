import nltk
from random import randint, seed
import os

script_dir = os.path.dirname(__file__)

version_count = 0
def transform(t):
    label = t.label()
    if label == 'Program':
        return transform_Program(t[0])

def transform_Program(t):
    label = t.label()
    if label == 'Statement':
        return transform_Statement(t[0])

def transform_Statement(t):
    label = t.label()
    if label == 'MarkovFeature':
        return transform_MarkovFeature(t)

def transform_MarkovFeature(t):
    global version_count
    possible_statements = [
                t[1][0] + ' has been changed:',
                'Something has changed:',
                t[1][0] + ' has transformed:',
                t[1][0] + ' has been modified:',
            ]
    out_tree = nltk.Tree('MarkovFeature',[])

    out_tree.append(possible_statements[version_count])
    for elt in t[1:]:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(transform_VariableName(elt))
        if label == 'Operation':
            out_tree.append(transform_Operation(elt))
        if label == 'Assignment':
            out_tree.append(transform_Assignment(elt))
    return out_tree

def transform_VariableName(t):
    return t[0]

def transform_Operation(t):
    if t[0] == '+':
        return "plus"
    if t[0] == '-':
        return "minus"
    if t[0] == '*':
        return "multiplied by"
    if t[0] == '/':
        return "divided by"

def transform_Assignment(t):
    return "is now"


def main():
    seed(0)
    global version_count

    output_file = "nl_markov_feature_output.txt"
    with open(os.path.join(script_dir, "../data/tokenized_markov_feature_output.txt"), 'r') as f_input:
        with open(os.path.join(script_dir, f"../data/nl/{output_file}"), 'w') as f_output:
            lines = f_input.readlines()
            total_lines = len(lines)
            print(f"Writing tokenized file for POLICY to ", output_file)
            print(f'Tokenizing {total_lines} total statements')
            print('This may take a while...\n...')

            for i in range(4): 
                for i in range(total_lines):
                    if (i % 100 == 0):
                        print(f'Finished parsing {i}/{total_lines} statements')
                    
                    tokenized_rlang = lines[i]
                    in_tree = nltk.Tree.fromstring(tokenized_rlang)
                    out_tree = transform(in_tree)
                    output = ' '.join(out_tree.leaves())
                    f_output.write(output + '\n')
                version_count+=1
                
if __name__ == '__main__':
    main()