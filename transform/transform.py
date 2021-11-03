import nltk

# f = open('tokenized_policy_output.txt')
# for i in range(1):
#     l = f.readline()
#     t = nltk.Tree.fromstring(l)



def transform(t):

    # switch statement based on t.label
    pass

def transform_Policy(t):
    pass

def transform_ConditionalExecute(t):
    out_tree = nltk.Tree.fromstring('ConditionalExecute')
    for elt in t:
        label = elt.label()
        if label == 'If':
            cpt = transform_If(elt)
        if label == 'Elif':
            cpt = transform_Elif(elt)
        if label == 'Else':
            cpt = transform_Else(elt)
    pass

def transform_If(t):
    out_tree = nltk.Tree('If', ['if'])
    for elt in t:
        if elt == 'VariableName':
            out_tree.append(elt[0])
        if elt == 'Execute':
            out_tree.insert(0, transform_Execute(elt))

    return out_tree

def transform_Elif(t):
    out_tree = nltk.Tree('Elif', ['otherwise, if'])
    for elt in t:
        if elt == 'VariableName':
            out_tree.append(elt[0])
        if elt == 'Execute':
            out_tree.append(transform_Execute(elt))
    
    return out_tree

def transform_Else(t):
    out_tree = nltk.Tree('Else', [transform_Execute(t[1]), 'otherwise'])

    pass

def transform_Execute(t):
    out_tree = nltk.tree('Execute', ['do', t[[1]]])
    return out_tree
    
def transform_BoolExp(t):
    out_tree = nltk.Tree('BoolExp', [])
    for elt in t:
        label = elt.label()
        if label == 'VariableName':
            out_tree.append(elt[0])
        if label == 'BoolTest':
            out_tree.append(transform_BoolTest(elt))
        if label == 'Value':
            out_tree.append(transform_Value(elt))
    
    return out_tree

def transform_BoolTest(t):
    out_tree = nltk.Tree('BoolTest',[])
    elt = t[0]
    if elt == '==':
        out_tree.append('is equal to')
    if elt == '!=':
        out_tree.append('is not equal to')
    if elt == '>':
        out_tree.append('is greater than')
    if elt == '<':
        out_tree.append('is less than')
    return out_tree

def transform_Value(t):
    pass

def transform_VariableName(t):
    pass

def transform_Conj(t):
    pass

#Newline, Colon, Tab are cut out for the English translation