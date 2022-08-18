import pickle
#proj = pickle.load(open('process_datacopy.pkl', 'rb'))
p = pickle.load(open('nl_voc.pkl', 'rb'))
print(p['indexOf_ter'])
assert(0)
for i, x in enumerate(proj):
    if 'indexOf_ter' in x['input']:
        print(x)
    if i > 20:
        break
        #assert(0)