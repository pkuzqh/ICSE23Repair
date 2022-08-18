import subprocess
from tqdm import tqdm
import time
import os
card = [0, 1, 2 , 3, 4]
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-1', 'Closure-10', 'Closure-14', 'Closure-15', 'Closure-18', 'Closure-31', 'Closure-33', 'Closure-38', 'Closure-51', 'Closure-62', 'Closure-63', 'Closure-70', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-107', 'Closure-118', 'Closure-113', 'Closure-124', 'Closure-125', 'Closure-129', 'Lang-6', 'Lang-16', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-38', 'Lang-39', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-5', 'Math-25', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-57', 'Math-58', 'Math-59', 'Math-69', 'Math-70', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Lang-43', 'Math-50', 'Math-98', 'Time-7', 'Mockito-38', 'Mockito-22', 'Mockito-29', 'Mockito-34', 'Closure-104', 'Math-27']
#lst = ['Lang-39', 'Lang-50', 'Lang-60', 'Lang-63', 'Math-88', 'Math-82', 'Math-20', 'Math-28', 'Math-6', 'Math-72', 'Math-79', 'Math-8']#['Closure-38', 'Closure-123', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
lst = []
for x in os.listdir('../bugs-QuixBugs/bugs/'):
    if os.path.exists('patchQuix/%s.json'%x):
        continue
    lst.append(x)
for i in tqdm(range(int(len(lst) / 10) + 1)):
            print(lst[i])
            jobs = []
            for j in range(10):
                if 10 * i + j >= len(lst):
                    continue
                cardn =int(j / 2)
                p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python3 testQuixbug.py " + lst[10 * i + j], shell=True)#subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card) + " python3 testDefect4j.py " + lst[12 * i + j], stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, shell=True)#subprocess.run(["python3", "run.py"])
                #p = subprocess.Popen("python3 run.py", stdout=subprocess.PIPE, shell=True)#subprocess.run(["python3", "run.py"])
                jobs.append(p)
                time.sleep(2)
                #jobs.append(1)
            for p in jobs:
                p.wait()