terminalsFather = ["value", "name", "member", "qualifier"]

def tree_to_variable_index(root):
    if root.name == 'type':
        return []
    if len(root.child) == 0 and root.father.name in terminalsFather:
        if root.father.name == 'member' and root.father.father.name == 'MethodInvocation':
            return []
        return [root.id]
    ans = []
    for x in root.child:
        ans += tree_to_variable_index(x)
    return ans
def dfg_java(root, states, nodes):
    assignment=['Assignment']
    def_statement=['VariableDeclarator']
    increment_statement=['update_expression']
    if_statement=['IfStatement','else_statement']
    for_statement=['ForStatement']
    enhanced_for_statement=['enhanced_for_statement']
    while_statement=['while_statement']
    tenary_statement=['TernaryExpression']
    DFG = []
    states = states.copy()
    if len(root.child) == 0:
        #print(root.name)
        if root.father.name in terminalsFather:
            if root.father.name == 'member' and root.father.father.name == 'MethodInvocation':
                return [], states
            if root.name in states:
                return [(root.id, 'comefrom', states[root.name].copy())], states
            else:
                states[root.name] = [root.id]
                return [], states
        else:
            return [], states
        return [], states
        #identifier
    elif root.name in def_statement:
        vname = root.getchildbyname('name')
        rvalue = root.getchildbyname('initializer')
        if rvalue is None:
            #no initializer
            return [], states
            pass
        else:
            vids = tree_to_variable_index(vname)
            rids = tree_to_variable_index(rvalue)
            temp, states = dfg_java(rvalue, states,nodes)
            DFG += temp
            for vid in vids:
                for rid in rids:
                    DFG.append((vid, 'comefrom', [rid]))
                states[nodes[vid].name] = [vid]
        return DFG, states
    elif root.name in assignment:
        leftnode = root.getchildbyname('expressionl')
        rightnode = root.getchildbyname('value')
        temp, states = dfg_java(rightnode, states,nodes)
        DFG = temp
        vids = tree_to_variable_index(leftnode)
        rids = tree_to_variable_index(rightnode)
        for vid in vids:
            for rid in rids:
                DFG.append((vid, 'computedfrom', [rid]))
            states[nodes[vid].name] = [vid]
        return DFG, states
    elif root.name in increment_statement:
        DFG = []
        vids = tree_to_variable_index(root)
        for vid in vids:
            for vid2 in vids:
                DFG.append((vid, 'computedfrom', [vid2]))
            states[nodes[vid].name] = [vid]
        return DFG, states
    elif root.name in if_statement:
        DFG = []
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else_statement' in root.name:
            tag = True
        for child in root.child:
            if 'else_statement' in child.name:
                tag=True
            if child.name not in if_statement and flag is False:
                temp,current_states=dfg_java(child,current_states,nodes)
                DFG+=temp
            else:
                flag=True
                temp,new_states=dfg_java(child,states,nodes)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        #print('if', others_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root.name in tenary_statement:
        DFG = []
        current_states=states.copy()
        others_states=[]

        for child in root.child:
            if 'if_true' in child.name:
                temp, current_states = dfg_java(child, current_states, nodes)
            if 'if_false' in child.name:
                temp,new_states=dfg_java(child,states,nodes)
                others_states.append(new_states)
        others_states.append(current_states)

        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states

    elif root.name in for_statement:
        DFG = []
        try:
            statements = root.child[1].child[0].child[0].child
        except:
            return [], states
        for child in statements:
            temp,states=dfg_java(child,states,nodes)
            DFG+=temp
        flag = False
        for child in statements:
            if flag:
                temp,states=dfg_java(child,states,nodes)
                DFG+=temp                
            elif child.name=="LocalVariableDeclaration":
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1]) not in dic:
                dic[(x[0],x[1])]=[x[2]]
            else:
                dic[(x[0],x[1])][0]=sorted(list(set(dic[(x[0],x[1])][0]+x[2])))
        DFG=[(x[0],x[1],y[0]) for x,y in sorted(dic.items(),key=lambda t:t[0][0])]
        return DFG, states
    elif root.name in enhanced_for_statement:
        return [], states
    elif root.name in while_statement:
        DFG = []
        try:
            statements = root.child[1].child[0].child[0].child
        except:
            return [], states
        for i in range(2):
            for child in statements:
                temp,states=dfg_java(child,states,nodes)
                DFG+=temp    
        dic={}
        for x in DFG:
            if (x[0],x[1]) not in dic:
                dic[(x[0],x[1])]=[x[2]]
            else:
                dic[(x[0],x[1])][0]=sorted(list(set(dic[(x[0],x[1])][0]+x[2])))
        DFG=[(x[0],x[1],y[0]) for x,y in sorted(dic.items(),key=lambda t:t[0][0])]
        return DFG, statements
                #dic[(x[0],x[1])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
    elif root.name == 'type':
        return [], states
    if True:
        DFG=[]
        for child in root.child:
            if child.name not in []:
                temp,states=dfg_java(child,states,nodes)
                DFG+=temp
            #if root.name == 'MethodDeclaration':
            #    print(states)
        return DFG, states
s = ['MethodDeclaration', 'modifiers', 'public_ter', '^', '^', 'return_type', 'ReferenceType', 'name', 'TimeSeries_ter', '^', '^', '^', '^', 'name', 'meth0', '^', '^', 'parameters', 'FormalParameter', 'type', 'ReferenceType', 'name', 'RegularTimePeriod_ter', '^', '^', '^', '^', 'name', 'par0', '^', '^', '^', 'FormalParameter', 'type', 'ReferenceType', 'name', 'RegularTimePeriod_ter', '^', '^', '^', '^', 'name', 'par1', '^', '^', '^', '^', 'throws', 'CloneNotSupportedException_ter', '^', '^', 'body', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '==_ter', '^', '^', 'operandl', 'MemberReference', 'member', 'par0', '^', '^', '^', '^', 'operandr🚀null', 'Literal🚀null', 'value🚀null', 'null_ter', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'ThrowStatement', 'expression', 'ClassCreator', 'type', 'ReferenceType', 'name', 'IllegalArgumentException_ter', '^', '^', '^', '^', 'arguments', 'Literal🚀string', 'value🚀string', '<string>_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '==_ter', '^', '^', 'operandl', 'MemberReference', 'member', 'par1', '^', '^', '^', '^', 'operandr🚀null', 'Literal🚀null', 'value🚀null', 'null_ter', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'ThrowStatement', 'expression', 'ClassCreator', 'type', 'ReferenceType', 'name', 'IllegalArgumentException_ter', '^', '^', '^', '^', 'arguments', 'Literal🚀string', 'value🚀string', '<string>_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '>_ter', '^', '^', 'operandl', 'MethodInvocation', 'qualifier', 'par0', '^', '^', 'arguments', 'MemberReference', 'member', 'par1', '^', '^', '^', '^', 'member', 'compareTo_ter', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '0_ter', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'ThrowStatement', 'expression', 'ClassCreator', 'type', 'ReferenceType', 'name', 'IllegalArgumentException_ter', '^', '^', '^', '^', 'arguments', 'Literal🚀string', 'value🚀string', '<string>_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'LocalVariableDeclaration', 'type', 'BasicType', 'name', 'boolean_ter', '^', '^', '^', '^', 'declarators', 'VariableDeclarator', 'name', 'loc0', '^', '^', 'initializer', 'Literal🚀bool', 'value🚀bool', 'false_ter', '^', '^', '^', '^', '^', '^', '^', 'LocalVariableDeclaration', 'type', 'BasicType', 'name', 'int_ter', '^', '^', '^', '^', 'declarators', 'VariableDeclarator', 'name', 'loc1', '^', '^', 'initializer', 'MethodInvocation', 'arguments', 'MemberReference', 'member', 'par0', '^', '^', '^', '^', 'member', 'getIndex_ter', '^', '^', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '<_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc1', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '0_ter', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'StatementExpression', 'expression', 'Assignment', 'expressionl', 'MemberReference🚀numeric', 'member🚀numeric', 'loc1', '^', '^', '^', '^', 'value', 'BinaryOperation🚀numeric', 'operator🚀numeric', '+_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc1', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '1_ter', '^', '^', '^', '^', '^', '^', 'type', '=_ter', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '==_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc1', '^', '^', '^', '^', 'operandr', 'This', 'selectors', 'MemberReference', 'member', 'data_ter', '^', '^', '^', 'MethodInvocation', 'member', 'size_ter', '^', '^', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'StatementExpression', 'expression', 'Assignment', 'expressionl', 'MemberReference🚀bool', 'member🚀bool', 'loc0', '^', '^', '^', '^', 'value', 'Literal🚀bool', 'value🚀bool', 'true_ter', '^', '^', '^', '^', 'type', '=_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'LocalVariableDeclaration', 'type', 'BasicType', 'name', 'int_ter', '^', '^', '^', '^', 'declarators', 'VariableDeclarator', 'name', 'loc2', '^', '^', 'initializer', 'MethodInvocation', 'arguments', 'MemberReference', 'member', 'par1', '^', '^', '^', '^', 'member', 'getIndex_ter', '^', '^', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '<_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '0_ter', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'StatementExpression', 'expression', 'Assignment', 'expressionl', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'value', 'BinaryOperation🚀numeric', 'operator🚀numeric', '+_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '1_ter', '^', '^', '^', '^', '^', '^', 'type', '=_ter', '^', '^', '^', '^', '^', 'StatementExpression', 'expression', 'Assignment', 'expressionl', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'value', 'BinaryOperation🚀numeric', 'operator🚀numeric', '-_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '1_ter', '^', '^', '^', '^', '^', '^', 'type', '=_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'BinaryOperation🚀bool', 'operator🚀bool', '<_ter', '^', '^', 'operandl🚀numeric', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'operandr🚀numeric', 'Literal🚀numeric', 'value🚀numeric', '0_ter', '^', '^', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'StatementExpression', 'expression', 'Assignment', 'expressionl', 'MemberReference🚀bool', 'member🚀bool', 'loc0', '^', '^', '^', '^', 'value', 'Literal🚀bool', 'value🚀bool', 'true_ter', '^', '^', '^', '^', 'type', '=_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'IfStatement', 'condition', 'MemberReference🚀bool', 'member🚀bool', 'loc0', '^', '^', '^', '^', 'then_statement', 'BlockStatement', 'statements', 'LocalVariableDeclaration', 'type', 'ReferenceType', 'name', 'TimeSeries_ter', '^', '^', '^', '^', 'declarators', 'VariableDeclarator', 'name', 'loc3', '^', '^', 'initializer', 'Cast', 'type', 'ReferenceType', 'name', 'TimeSeries_ter', '^', '^', '^', '^', 'expression', 'SuperMethodInvocation', 'member', 'clone_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'StatementExpression', 'expression', 'Assignment', 'expressionl', 'MemberReference', 'qualifier', 'loc3', '^', '^', 'member', 'data_ter', '^', '^', '^', '^', 'value', 'ClassCreator', 'type', 'ReferenceType', 'name', 'java_ter', '^', '^', 'sub_type', 'ReferenceType', 'name', 'util_ter', '^', '^', 'sub_type', 'ReferenceType', 'name', 'ArrayList_ter', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', 'type', '=_ter', '^', '^', '^', '^', '^', 'ReturnStatement', 'expression', 'MemberReference', 'member', 'loc3', '^', '^', '^', '^', '^', '^', '^', '^', 'else_statement', 'BlockStatement', 'statements', 'ReturnStatement', 'expression', 'MethodInvocation', 'arguments', 'MemberReference🚀numeric', 'member🚀numeric', 'loc1', '^', '^', '^', 'MemberReference🚀numeric', 'member🚀numeric', 'loc2', '^', '^', '^', '^', 'member', 'meth0', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^', '^']

class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.sibiling = None
    def getchildbyname(self, name):
        for x in self.child:
            if x.name == name:
                return x
        return None

'''node = Node(s[0], 0)
currnode = node
idx = 1
nodes = [node]
for x in s[1:]:
    if x != "^":
        nnode = Node(x, idx)
        idx += 1
        nnode.father = currnode
        currnode.child.append(nnode)
        currnode = nnode
        nodes.append(nnode)
    else:
        currnode = currnode.father'''
#dfg, s = dfg_java(node, {}, nodes)
#print(dfg)