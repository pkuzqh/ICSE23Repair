import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
import h5py
from tqdm import tqdm
import json
from scipy import sparse
from parse_dfg import *
sys.setrecursionlimit(500000000)
class Graph:
    def __init__(self):
        self.row = []
        self.col = []
        self.val = []
        self.edge = {}
        self.rowNum = 0
        self.colNum = 0
    def addEdge(self, r, c, v):
        if (r, c) in self.edge:
            print(r, c)
            assert(0)
        self.edge[(r, c)] = len(self.row)
        self.row.append(r)
        self.col.append(c)
        self.val.append(v)
        '''self.edge[(c, r)] = len(self.row)
        self.row.append(c)
        self.col.append(r)
        self.val.append(v)'''
    def editVal(self, r, c, v):
        self.val[self.edge(r, c)] = v
    def updateval(self, index, v):
        self.val[index] = v
    def normlize(self):
        r = {}
        c = {}
        for i  in range(len(self.row)):
            if self.row[i] not in r:
                r[self.row[i]] = 0
            r[self.row[i]] += 1
            if self.col[i] not in c:
                c[self.col[i]] = 0
            c[self.col[i]] += 1
        for i in range(len(self.row)):
            self.val[i] = 1 / math.sqrt(r[self.row[i]]) * 1 / math.sqrt(c[self.col[i]])
class PreSumDataset(data.Dataset):
    def __init__(self, config, dataName="train"):
        self.train_path = "train_process.txt"
        self.val_path = "dev_process.txt"  # "validD.txt"
        self.test_path = "test_process.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.Var_Len = config.varlen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50
        self.edgedic = pickle.load(open('edge.pkl', 'rb'))
        self.ruledict = pickle.load(open("rule.pkl", "rb"))
        self.rrdict = {}
        for x in self.ruledict:
            self.rrdict[self.ruledict[x]] = x
        if not os.path.exists("nl_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        #print(self.Nl_Voc)
        if dataName == "train":
            if os.path.exists("data_pre.pkl"):
                self.data = pickle.load(open("data_pre.pkl", "rb"))
                return
            data = pickle.load(open('process_datacopy.pkl', 'rb'))
            print(len(data))
            train_size = int(len(data) / 8 * 7)
            self.data = self.preProcessData(data)

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
        self.Nl_Voc["<emptynode>"] = len(self.Nl_Voc)
        self.Code_Voc["<emptynode>"] = len(self.Code_Voc)

    def init_dic(self):
        print("initVoc")
        #f = open(self.train_path, "r", encoding='utf-8')
        #lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        nls = []
        rules = []
        data = pickle.load(open('process_datacopy.pkl', 'rb'))
        for x in data:
            if len(x['rule']) > self.Code_Len:
                continue
            nls.append(x['input'])
        '''for i in tqdm(range(int(len(lines) / 5))):
            data = lines[5 * i].strip().lower().split()
            nls.append(data)
            rulelist = lines[5 * i + 1].strip().split()
            tmp = []
            for x in rulelist:
                if int(x) >= 10000:
                    tmp.append(data[int(x) - 10000])
            rules.append(tmp)
        f.close()
        nl_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=0)
        code_voc = VocabEntry.from_corpus(rules, size=50000, freq_cutoff=10)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id'''
        code_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=10)
        self.Code_Voc = code_voc.word2id
        for x in self.ruledict:
            print(x)
            lst = x.strip().lower().split()
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.Code_Voc:
                    self.Code_Voc[y] = len(self.Code_Voc)
            #rules.append([lst[0]] + lst[2:])
        #print(self.Code_Voc)
        if 'numeric' not in self.Code_Voc:
            self.Code_Voc['numeric'] = len(self.Code_Voc)
        if 'string' not in self.Code_Voc:
            self.Code_Voc['string'] = len(self.Code_Voc)
        if 'bool' not in self.Code_Voc:
            self.Code_Voc['bool'] = len(self.Code_Voc)
        if 'ptype' not in self.Code_Voc:
            self.Code_Voc['ptype'] = len(self.Code_Voc)
        self.Nl_Voc = self.Code_Voc
        #print(self.Code_Voc)
        assert("root" in self.Code_Voc)
        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
        print(maxNlLen, maxCodeLen, maxCharLen)
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def preProcessOne(self, data):
        #print(tree)
        #print(self.nl[0])
        inputNl = []
        inputNlchar = []
        inputPos = []
        inputNlad = []
        Nl = []
        for x in data:
            inputpos = x['prob']
            tree = x['tree']
            inputpos = self.pad_seq(inputpos, self.Nl_Len)
            nl = tree.split()
            Nl.append(nl)
            node = Node('root', 0)
            currnode = node
            idx = 1
            nltmp = ['root']
            nodes = [node]
            for j, x in enumerate(nl[1:]):
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(x.father.id)
                        nladdata.append(1)
                    for s in x.father.child:
                        if x.id < self.Nl_Len and s.id < self.Nl_Len:
                            nladrow.append(x.id)
                            nladcol.append(s.id)
                            nladdata.append(1)
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)
            nl = nltmp
            #tmp = GetFlow()
            #for p in range(len(tmp)):
            #    for l in range(len(tmp[0])):
            #        nladrow.append(p)
            #        nladcol.append(l)
            #        nladdata.append(1)
            '''for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(x.father.id)
                        nladdata.append(1)
                    for s in x.father.child:
                        if x.id < self.Nl_Len and s.id < self.Nl_Len:
                            nladrow.append(x.id)
                            nladcol.append(s.id)
                            nladdata.append(1)
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)'''
            nl = nltmp
            inputnls = self.pad_seq(self.Get_Em(nl, self.Nl_Voc), self.Nl_Len)
            nlad = sparse.coo_matrix((nladdata, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNl.append(inputnls)
            inputNlad.append(nlad)
            inputPos.append(inputpos)
            inputNlchar.append(inputnlchar)
        self.data = [inputNl, inputNlad, inputPos, inputNlchar]
        self.nl = Nl
        return
        #return np.array([inputnls]), np.array([nlad.toarray()]), np.array([inputpos]), np.array([inputnlchar])
    def preProcessData(self, dataFile):
        #lines = dataFile.readlines()
        inputNl = []
        inputNlad = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        inputPos = []
        inputVar = []
        inputType = []
        inputUseable = []
        nls = []
        vardics = []
        maxl = 0
        for i in tqdm(range(len(dataFile))):
            if len(dataFile[i]['varwithtype']) > self.Var_Len:
                continue
            child = {}
            nl = dataFile[i]['input']#lines[5 * i].lower().strip().split()
            node = Node(nl[0], 0)
            currnode = node
            idx = 1
            nltmp = [nl[0]]
            nodes = [node]
            inputres = []
            vars = dataFile[i]['varwithtype']
            vardict = {}
            for k, va in enumerate(vars):
                vardict[va[0]] = k
            for x in nl[1:]:
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            dfg, _ = dfg_java(node, {}, nodes)
            gcode = Graph()
            for edge in dfg:
                if edge[1] not in self.edgedic:
                    self.edgedic[edge[1]] = len(self.edgedic)
                for idx in edge[2]:
                    gcode.addEdge(edge[0], idx, self.edgedic[edge[1]])
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.name in vardict:
                    inputres.append(vardict[x.name] + 1)
                else:
                    inputres.append(0)
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        if 'child->father' not in self.edgedic:
                            self.edgedic['child->father'] = len(self.edgedic) 
                        gcode.addEdge(x.id, x.father.id, self.edgedic['child->father'])
                        #nladrow.append(x.id)
                        #nladcol.append(x.father.id)
                        #nladdata.append(1)
                    idx = x.father.child.index(x)
                    if idx > 0:
                        if 'left->ad' not in self.edgedic:
                            self.edgedic['left->ad'] = len(self.edgedic)
                        gcode.addEdge(x.id, x.father.child[idx - 1].id, self.edgedic['left->ad'])
                    if idx < len(x.father.child) - 1:
                        if 'right->ad' not in self.edgedic:
                            self.edgedic['right->ad'] = len(self.edgedic)
                        gcode.addEdge(x.id, x.father.child[idx + 1].id, self.edgedic['right->ad'])
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        if 'father->child' not in self.edgedic:
                            self.edgedic['father->child'] = len(self.edgedic)
                        gcode.addEdge(x.id, s.id, self.edgedic['father->child'])
                        #nladrow.append(x.id)
                        #nladcol.append(s.id)
                        #nladdata.append(1)
            nl = nltmp
            vars = dataFile[i]['varwithtype']
            tmpvar = []
            tmptype = []
            tmpuseable = []
            for k, x in enumerate(vars):
                tmpvar.append(x[0])
                tmptype.append(x[1])
                tmpuseable.append(x[2] + 5) 
                for p, nodet in enumerate(nl):
                    if nodet == x[0]:
                        if 'comef' not in self.edgedic:
                            self.edgedic['comef'] = len(self.edgedic)
                        if 'usev' not in self.edgedic:
                            self.edgedic['usev'] = len(self.edgedic)
                        gcode.addEdge(p, self.Nl_Len + k, self.edgedic['comef'])
                        gcode.addEdge(self.Nl_Len + k, p, self.edgedic['usev'])
            if 'dtype' not in self.edgedic:
                self.edgedic['dtype'] = len(self.edgedic)
            if 'stype' not in self.edgedic:
                self.edgedic['stype'] = len(self.edgedic)
            for k, v in enumerate(tmpvar):
                for j, v in enumerate(tmpvar):
                    if tmptype[k] == tmptype[j]:
                        gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['stype'])
                    else:
                        gcode.addEdge(self.Nl_Len + k, self.Nl_Len + j, self.edgedic['dtype'])
            inputVar.append(self.pad_seq(self.Get_Em(tmpvar, self.Nl_Voc), self.Var_Len))
            inputType.append(self.pad_seq(self.Get_Em(tmptype, self.Nl_Voc), self.Var_Len))
            inputUseable.append(self.pad_seq(tmpuseable, self.Var_Len))
            maxl = max(len(vars), maxl)
            #continue
            nls.append((dataFile[i]['input'], dataFile[i]['varwithtype']))
            if i == 6:
                print(dataFile[i]['varwithtype'])
            inputpos = dataFile[i]['problist']
            #for j in range(len(inputpos)):
            #    inputpos[j] = inputpos[j]
            inputPos.append(self.pad_seq(inputpos, self.Nl_Len))
            
            inputnls = self.Get_Em(nl, self.Nl_Voc)
            inputNl.append(self.pad_seq(inputnls, self.Nl_Len))
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            
            inputres = self.pad_seq(inputres, self.Nl_Len)

            inputRes.append(inputres)

            nlad = sparse.coo_matrix((gcode.val, (gcode.row, gcode.col)), shape=(self.Nl_Len + self.Var_Len, self.Nl_Len + self.Var_Len))
            inputNlad.append(nlad)
        print(maxl)
        batchs = [inputNl, inputNlad, inputRes, inputPos, inputVar, inputType, inputUseable]
        self.data = batchs
        self.nl = nls
        #self.code = codes
        if self.dataName == "train":
            open("data_pre.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        return batchs

    def __getitem__(self, offset):
        ans = []
        for i in range(len(self.data)):
            d = self.data[i][offset]
            if i == 1:
                tmp = d.toarray().astype(np.int32)
                ans.append(tmp)
            else:
                ans.append(np.array(d))
        return ans
    def __len__(self):
        return len(self.data[0])
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
    
#dset = SumDataset(args)
