Correct Patch: 
```
               Cli(5): 5, 18, 25, 28, 32
               Codec(3): 2, 7, 17
               Compress(4): 19, 27, 31, 32
               Csv(5): 4, 5, 9, 11, 15
               JacksonCore(2): 5, 8
               Jsoup(10): 24, 26, 39, 40, 61, 64, 68, 77, 85, 88
               Jxpath(2): 12, 21
               Gson(1): 6
```
Description of Correctness
```
Cli(5): 
5, 18, 25, 28, 32: Identical to groundtruth.
Codec(3):
2, 7, 17: Identical to groundtruth.
Compress(4):
19, 27, 31, 32: Identical to groundtruth.
Csv(5):
4, 5, 9, 11: Identical to groundtruth.
15: The generated patch also removes the corresponding condition branch.
JacksonCore(2): 
5, 8: Identical to groundtruth.
Jsoup(10):
24, 26, 39, 40, 61, 64, 68, 85, 88: Identical to groundtruth.
77: "normalName" is the lowercase of the name. 
Jxpath(2):
12, 21: Identical to groundtruth.
Gson(1):
6: Identical to groundtruth.
```
