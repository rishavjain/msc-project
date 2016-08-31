import sys
import re
import gzip
import codecs
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
        if lower:
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


if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.stderr.write(
            "Usage: parsed-file | %s <vocab-file> [<min-count>] > deps-file \n" % sys.argv[0])
        sys.exit(1)

    vocab_file = sys.argv[1]

    try:
        THR = int(sys.argv[2])
    except IndexError:
        THR = 100

    try:
        WIN_SIZE = int(sys.argv[3])
    except IndexError:
        WIN_SIZE = 1

    lower = True

    print("vocab_file:", vocab_file, file=sys.stderr)
    print("THR:", THR, file=sys.stderr)
    print("WIN_SIZE:", WIN_SIZE, file=sys.stderr)

    try:
        gzip_file = gzip.open(sys.argv[4], 'rb')
        sys.stdin = codecs.getreader('iso-8859-1')(gzip_file)
    except IndexError:
        print('using console input', file=sys.stderr)

    try:
        sys.stdout = open(sys.argv[5], 'w')
    except IndexError:
        print('using console output', file=sys.stderr)

    wvocab = read_vocab(open(vocab_file, encoding='iso-8859-1'), THR)
    print("vocab:", len(wvocab), file=sys.stderr)

    startt = time.time()
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

        # WIN_SIZE = 5
        for m in deps:
            d = []
            curr = m
            for depth in range(WIN_SIZE):
                d += [deps[curr], ]

                if deps[curr][:2] in deps:
                    curr = deps[curr][:2]
                else:
                    break

            print(m[1], '::|::'.join(["I_".join((x[2], x[1])) for x in d]))


            if d[-1][1] != '*root*':
                idx = -1
            elif len(d) > 1:
                idx = -2
                while idx >= 0 and d[idx][1] not in wvocab:
                    idx -= 1
            else:
                continue

            d[idx] = list(d[idx])
            h = d[idx][1]
            d = [[m + (d[0][2],),],] + [[(d[i][0], d[i][1], d[i+1][2]),] for i in range(0,len(d)+idx)]

            print(h, '::|::'.join(["_".join((x[2], x[1])) for x in d[0]]))

            # sys.stdout.flush()