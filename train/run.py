import sys
sys.path.append(".")
import torch
from torch import optim
from Dataset import SumDataset, Graph
import os
from tqdm import tqdm
from model.Model import *
import numpy as np
from copy import deepcopy
import pickle
from ScheduledOptim import *
import sys
from Searchnode import Node
import json
import torch.nn.functional as F
import traceback
from blcDP import BalancedDataParallel
from apex import amp
import copy
#from pythonBottom.run import finetune
#from pythonBottom.run import pre
#wandb.init("sql")
from memory_profiler import profile
import random
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':500,
    'CodeLen':60,
    'batch_size':96,
    'tbsize':4,
    'embedding_size':256,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'num_steps':50,
    'rulenum':10,
    'cnum':695,
    'use_apex':False,
    'mask_value':-1e9,
    'gradient_accumulation_steps':1,
    'max_grad_norm':5,
    'seed':19970316,
    'varlen':45,
    'mask_id':-1,
    'lookLoss':False,
    'patience':5,
    'max_num_trials':10
})
os.environ["CUDA_VISIBLE_DEVICES"]="1, 5, 6, 7"
#os.environ['CUDA_LAUNCH_BLOCKING']="4"
def save_model(model, dirs='checkpointSearch/', optimizer=None, amp=None):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    '''if args.use_apex:
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'amp':amp.state_dict()
        }
        torch.save(checkpoint, dirs + 'best_model.ckpt')
    else:'''
    torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt'))
def load_model_apex(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt')['model'])
use_cuda = True#torch.cuda.is_available()
onelist = ['root', 'body', 'statements', 'block', 'arguments', 'initializers', 'parameters', 'case', 'cases', 'selectors']
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans
def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    return ans
def getRulePkl(vds):
    inputruleparent = []
    inputrulechild = []
    for i in range(args.cnum):
        rule = vds.rrdict[i].strip().lower().split()
        inputrulechild.append(vds.pad_seq(vds.Get_Em(rule[2:], vds.Code_Voc), vds.Char_Len))
        inputruleparent.append(vds.Code_Voc[rule[0].lower()])
    return np.array(inputruleparent), np.array(inputrulechild)
def getAstPkl(vds):
    rrdict = {}
    for x in vds.Code_Voc:
        rrdict[vds.Code_Voc[x]] = x
    inputchar = []
    for i in range(len(vds.Code_Voc)):
        rule = rrdict[i].strip().lower()
        inputchar.append(vds.pad_seq(vds.Get_Char_Em([rule])[0], vds.Char_Len))
    return np.array(inputchar)
def evalacc(model, dev_set, test=False):
    antimask = gVar(getAntiMask(args.CodeLen))
    a, b = getRulePkl(dev_set)
    tmpast = getAstPkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    if test:
        devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=len(dev_set),
                                              shuffle=False, drop_last=True, num_workers=1)
        batch_size = len(dev_set)
    else:
        devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
        batch_size = args.batch_size
    model = model.eval()
    accs = []
    tcard = []
    loss = []
    ploss = []
    antimask2 = antimask.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1)
    rulead = gVar(pickle.load(open("train/data/rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.tbsize, 1, 1)
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(args.tbsize, 1).long()
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        with torch.no_grad():
            if test:
                l, pre = model.module(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7], devBatch[8], devBatch[9], devBatch[10], devBatch[11], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, devBatch[5])     
            else:
                l, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7], devBatch[8], devBatch[9], devBatch[10], devBatch[11], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, devBatch[5])
            ploss.append(l)
            if(args.lookLoss or test):
                print(l)
            loss.append(l.mean().item())
            pred = pre.argmax(dim=-1)
            resmask = torch.gt(devBatch[5], 0)
            acc = (torch.eq(pred, devBatch[5]) * resmask).float()#.mean(dim=-1)
            #predres = (1 - acc) * pred.float() * resmask.float()
            accsum = torch.sum(acc, dim=-1)
            resTruelen = torch.sum(resmask, dim=-1).float()
            cnum = torch.eq(accsum, resTruelen).sum().float()
            #print((torch.eq(accsum, resTruelen)))
            acc = acc.sum(dim=-1) / resTruelen
            accs.append(acc.mean().item())
            tcard.append(cnum.item())
                        #print(devBatch[5])
                        #print(predres)
    if(args.lookLoss):
        exit(0)
    tnum = np.sum(tcard)
    acc = np.mean(accs)
    loss1 = np.mean(loss)

    return acc, tnum, loss1
from pretrainDataset import PreSumDataset
def pretrain():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    opt_level = 'O1'
    train_set = PreSumDataset(args, "train")
    print(len(train_set.rrdict))
    print(train_set.ruledict['start -> copyword3'])
    args.mask_id = train_set.Nl_Voc['[mask]']
    rulead = gVar(pickle.load(open("train/data/rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.tbsize, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = getAstPkl(train_set)
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(train_set.Code_Voc))).unsqueeze(0).repeat(args.tbsize, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    #dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    print(len(test_set))
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
    model = Decoder(args)
    #load_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0
    maxL = 1e10
    if use_cuda:
        if args.use_apex:
            model = model.cuda()
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        else:
            model = model.cuda()
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        #model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    antimask = gVar(getAntiMask(args.CodeLen))
    #model.to()
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):

            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss = model(dBatch[0], dBatch[1], dBatch[3], dBatch[4], dBatch[5], dBatch[6], tmpindex2, tmpchar, dBatch[2])
            loss = torch.mean(loss)# + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            #optimizer.zero_grad()
            if j % 100 == 0:
                closs = loss.item()
                
                if closs < maxL:
                    print('find better loss %f'%closs)
                    maxL = closs
                    save_model(model)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
                
            if args.use_apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            j += 1
def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    opt_level = 'O1'
    train_set = SumDataset(args, "train")
    print(len(train_set.rrdict))
    print(train_set.ruledict['start -> copyword3'])

    rulead = gVar(pickle.load(open("train/data/rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.tbsize, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = getAstPkl(train_set)
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).repeat(args.tbsize, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.tbsize, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(train_set.Code_Voc))).unsqueeze(0).repeat(args.tbsize, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    print(len(test_set))
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=1)
    model = Decoder(args)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0
    maxL = 1e10
    maxLoss = 1e10
    num_trial = patience = 0
    isBetter = False
    if args.lookLoss:
        load_model(model, 'train/models/checkModel19-400/')
    if use_cuda:
        if args.use_apex:
            model = model.cuda()
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        else:
            model = model.cuda()
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        #model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    #load_model(model)
    antimask = gVar(getAntiMask(args.CodeLen))
    #model.to()
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):
            #print(j)
            isBetter = False
            if j % 200 == 0:
                evalacc(model, test_set, True)
                acc2, tnum2, l = evalacc(model, dev_set)
                #print("for dev " + str(acc) + " " + str(tnum) + " max is " + str(maxC))
                print("Epch%d for dev "%epoch + str(acc2) + " " + str(tnum2) + " max is " + str(maxC2) + "loss is " + str(l))
                #exit(0)
                if maxL > l:#if maxC2 < tnum2 or maxC2 == tnum2 and maxAcc2 < acc2:
                    maxC2 = tnum2
                    #maxAcc2 = acc2
                    maxL = l
                    print("find better loss " + str(maxL))
                    isBetter = True
                    #save_model(model.module, 'checkModel/')
                if maxAcc2 < acc2:
                    maxAcc2 = acc2
                    print("find better maxloss " + str(maxAcc2))
                    save_model(model.module, 'train/models/checkpointAcc/')
                if isBetter:
                    patience = 0
                    print('save model to [%s]' % 'checkModel/', file=sys.stderr)
                    save_model(model.module, 'train/models/checkModel%d-%d/'%(epoch, j))
                    save_model(model.module, 'train/models/checkModel/')
                    torch.save(optimizer.state_dict(), 'train/models/checkModel/optim.bin')

                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == args.max_num_trials:
                            print('early stop!', file=sys.stderr)
                            exit(0)
                        lr = optimizer.param_groups[0]['lr'] * 0.5
                        model.module.load_state_dict(torch.load('train/models/checkModel/best_model.ckpt'))
                        model = model.cuda()
                        optimizer.load_state_dict(torch.load('train/models/checkModel/optim.bin'))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                        # Load the best save model
                        #load_model(model, 'checkModel/')
                        # Reinitialize the optimizer
                else:
                    patience += 1
            antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[6], dBatch[7], dBatch[8], dBatch[9], dBatch[10], dBatch[11], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, dBatch[5])
            loss = torch.mean(loss)# + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            #optimizer.zero_grad()
            #print(loss)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
                
            if args.use_apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()#_and_update_lr()
                    optimizer.zero_grad()
            j += 1

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    if sys.argv[1] == "train":
        if not os.path.exists('train/data/traindata.pkl'):
            SumDataset(args, 'train')
            os.system('python3 split.py')
        train()
    if sys.argv[1] == 'pretrain':
        pretrain()
    if sys.argv[1] == 'loss':
        args.lookLoss = True
        train()
    else:
        assert(0)
        profile = line_profiler.LineProfiler()
        profile.enable()
        test()
        profile.disable()
        profile.print_stats(sys.stdout)
     #test()




