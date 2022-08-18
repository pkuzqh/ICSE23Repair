import subprocess
from tqdm import tqdm
import time
import os
card = [4, 5, 6, 7]
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-1', 'Closure-10', 'Closure-14', 'Closure-15', 'Closure-18', 'Closure-31', 'Closure-33', 'Closure-38', 'Closure-51', 'Closure-62', 'Closure-63', 'Closure-70', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-107', 'Closure-118', 'Closure-113', 'Closure-124', 'Closure-125', 'Closure-129', 'Lang-6', 'Lang-16', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-38', 'Lang-39', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-5', 'Math-25', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-57', 'Math-58', 'Math-59', 'Math-69', 'Math-70', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Lang-43', 'Math-50', 'Math-98', 'Time-7', 'Mockito-38', 'Mockito-22', 'Mockito-29', 'Mockito-34', 'Closure-104', 'Math-27']
#lst = ['Lang-39', 'Lang-50', 'Lang-60', 'Lang-63', 'Math-88', 'Math-82', 'Math-20', 'Math-28', 'Math-6', 'Math-72', 'Math-79', 'Math-8']#['Closure-38', 'Closure-123', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
prlist = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']
ids = [range(1, 27), list(range(1, 134)), list(range(1, 66)), range(1, 107), range(1, 39), list(range(1, 28)), list(range(1, 25)), list(range(1, 23)), list(range(1, 13)), list(range(1, 15)), list(range(1, 14)), list(range(1, 40)), list(range(1, 6)), list(range(1, 64))]
lst = []
for k, x in enumerate(prlist):
    for y in ids[k]:
        if os.path.exists("patchforcompile/" + x + "-" + str(y) + ".json"):
            continue
        lst.append(x + "-" + str(y))
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-14', 'Closure-15', 'Closure-62', 'Closure-63', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-104', 'Closure-118', 'Closure-124', 'Lang-6', 'Lang-26', 'Lang-33', 'Lang-38', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Math-5', 'Math-27', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-50', 'Math-57', 'Math-59', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7', 'Mockito-29']
#lst = ['Math-50', 'Math-57', 'Math-59', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7', 'Mockito-29']

#lst = ['Chart-1', 'Chart-4', 'Chart-5', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-14', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-14', 'Closure-15', 'Closure-33', 'Closure-38', 'Closure-62', 'Closure-63', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-104', 'Closure-118', 'Closure-124', 'Closure-126', 'Lang-6', 'Lang-10', 'Lang-24', 'Lang-26', 'Lang-33', 'Lang-38', 'Lang-39', 'Lang-45', 'Lang-46', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Math-4', 'Math-5', 'Math-27', 'Math-30', 'Math-32', 'Math-33', 'Math-34', 'Math-41', 'Math-50', 'Math-57', 'Math-58', 'Math-59', 'Math-65', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7', 'Mockito-29', 'Mockito-38']
#recoder = ['Chart-3', 'Closure-2', 'Closure-7', 'Closure-21', 'Closure-46', 'Closure-57', 'Closure-109', 'Math-96']
#lst1 = lst + recoder
#lst = []
cardnum = {}
for c in card:
    cardnum[c] = 0
processcard = {}
jobs = []
for x in lst:
    #if 'Math' not in x:
    #    continue
    while True:
        for process in jobs:
            if process.poll() is not None:
                cardnum[processcard[process.pid]] -= 1
                jobs.remove(process)
        tmpcard = -1
        for y in card:
            if cardnum[y] < 3:
                tmpcard = y
                break
        if tmpcard != -1:
            break
        time.sleep(1)
    cardnum[tmpcard] += 1
    p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(tmpcard) + " python3 testDefect4j1.py " + x, shell=True)
    processcard[p.pid] = tmpcard
    jobs.append(p)
for p in jobs:
    p.wait()
    #subprocess.run(["python3", "run.py"])
