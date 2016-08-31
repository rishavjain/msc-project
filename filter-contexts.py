import sys
import re
import gzip
import codecs
from collections import Counter
import time

def read_conll(fh):
    root = (0, '*root*', -1, 'rroot')
    tokens = [root]
    for line in fh:
        # if lower:
        line = line.lower()
        tok = line.strip().split('\t')

        if tok == ['']:
            if len(tokens) > 1:
                yield tokens
            tokens = [root]
        else:
            # if len(tok) == 7:
            tokens.append((int(tok[0]), tok[1], int(tok[5]), tok[6]))
    if len(tokens) > 1:
        yield tokens


def read_vocab(fh, THR):
    v = {}
    for i, line in enumerate(fh):
        line = line.lower()
        line = line.strip().split()

        if len(line) != 2:
            # print('read_vocab: invalid format at line {}: {}'.format(i, line), file=sys.stderr)
            continue

        if int(line[1]) >= THR:
            v[line[0]] = int(line[1])
            # else:
            #     print('read_vocab: less than THR, %s %s' % (line[0], line[1]), file=sys.stderr)
    return v


def read_context(fh):
    cvocab = Counter()
    for i, line in enumerate(fh):
        line = line.strip().split()

        if len(line) != 2:
            continue

        c = tuple(line[0].split('::|::'))
        cvocab[c] = int(line[1])
    return cvocab


def filter_cvocab(cvocab, THR):
    iter = 0
    while True:
        print('iter {}:\tlen(cvocab)={}'.format(iter, len(cvocab)), end='', file=sys.stderr)
        sys.stderr.flush()

        ncvocab = Counter()
        t = ()

        flag = 0

        for c, n in cvocab.items():
            if n < THR:
                if len(c) > 1:
                    ncvocab.update({ c[1:]: n })
                flag += 1
            else:
                ncvocab.update({ c: n })

        # if len(ncvocab) > 0:
        #     cvocab.update(ncvocab)
        if len(t) > 0:
            ncvocab.update(t)

        cvocab = ncvocab

        if flag == 0:
            break

        print('\tremoved entries : {}'.format(flag), file=sys.stderr)
        iter += 1

    return cvocab

if __name__ == '__main__':

    vocab_file = open(sys.argv[1], encoding='iso-8859-1')

    context_file = open(sys.argv[2], encoding='iso-8859-1')

    try:
        THR = int(sys.argv[3])
    except IndexError:
        THR = 100

    try:
        WIN_SIZE = int(sys.argv[4])
    except IndexError:
        WIN_SIZE = 1

    try:
        gzip_file = gzip.open(sys.argv[5], 'rb')
        sys.stdin = codecs.getreader('iso-8859-1')(gzip_file)
    except IndexError:
        print('using console input', file=sys.stderr)

    try:
        sys.stdout = open(sys.argv[6], 'w')
    except IndexError:
        print('using console output', file=sys.stderr)

    wvocab = read_vocab(vocab_file, THR)
    cvocab = filter_cvocab(read_context(context_file), THR)

    startt = time.time()
    print(file=sys.stderr)
    for i, conll in enumerate(read_conll(sys.stdin)):
        if i % 50000 == 0:
            endt = time.time()
            print('sentences processed: {}K, speed: {} words/sec'.format(i/1000, (50000/(endt-startt) if (endt-startt)>1 else 0.0)), file=sys.stderr)
            startt = time.time()
            sys.stderr.flush()

        deps = {}

        for tok in conll[1:]:
            par_ind = tok[2]
            par = conll[par_ind]
            m = tok[1]
            if m not in wvocab: continue
            rel = tok[3]

            if rel == 'prep': continue  # this is the prep. we'll get there (or the PP is crappy)
            if rel == 'pobj' and par[0] != 0:

                ppar = conll[par[2]]
                rel = "%s:%s" % (par[3], par[1])
                h = ppar[1]
                ih = ppar[0]
            else:
                h = par[1]
                ih = par[0]
            if h not in wvocab and h != '*root*': continue

            deps[(tok[0], m)] = (ih, h,  rel)


        for m in deps:
            d = []
            curr = m
            for depth in range(WIN_SIZE):
                d += [deps[curr], ]

                if deps[curr][:2] in deps:
                    curr = deps[curr][:2]
                else:
                    break

            d1 = tuple(["I_".join((x[2], x[1])) for x in d])
            while len(d1) > 0:
                if d1 in cvocab:
                    print(m[1], '::|::'.join(d1))
                    break
                d1 = d1[:-1]

            if d[-1][1] != '*root*':
                idx = -1
            elif len(d) > 1:
                idx = -2
                while len(d)+idx >= 0 and d[idx][1] not in wvocab:
                    idx -= 1
            else:
                continue

            h = d[idx][1]
            d2 = [[m + (d[0][2],),],] + [[(d[i][0], d[i][1], d[i+1][2]),] for i in range(0,len(d)+idx)]
            d2 = d2[0]

            d1 = tuple(["_".join((x[2], x[1])) for x in d2])
            while len(d1) > 0:
                if d1 in cvocab:
                    print(h, '::|::'.join(d1))
                    break

                idx -= 1
                if len(d)+idx < 0:
                    break

                h = d[idx][1]
                d2 = [[m + (d[0][2],),],] + [[(d[i][0], d[i][1], d[i+1][2]),] for i in range(0,len(d)+idx)]
                d2 = d2[0]
                d1 = tuple(["_".join((x[2], x[1])) for x in d2])
