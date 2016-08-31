import sys
from collections import Counter
import gzip
import codecs

wc = Counter()
thr = int(sys.argv[1])
l = []

try:
    gzip_file = gzip.open(sys.argv[2], 'rb')
    sys.stdin = codecs.getreader('iso-8859-1')(gzip_file)
except IndexError:
    print('using console input', file=sys.stderr)

try:
    sys.stdout = open(sys.argv[3], 'w')
except IndexError:
    print('using console output', file=sys.stderr)

for i, w in enumerate(sys.stdin):
    # print('D:', i, w)
    tokens = w.split('\t')
    if len(tokens) > 1:
        w = tokens[1]

    if i % 1000000 == 0:
        # if i > 10000000: break
        print(i, len(wc), file=sys.stderr)
        wc.update(l)
        l = []
    l.append(w.strip().lower())
wc.update(l)

for w, c in [(w, c) for w, c in wc.items()]:
    if c < thr or w == '':
        wc.pop(w)

# sorting the counter in descending order of count
for w, c in sorted([(w, c) for w, c in wc.items() if c > thr], key=lambda x: -x[1]):
    print(" ".join([w, str(c)]))
