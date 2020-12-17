from indicnlp.tokenize import indic_tokenize

from sys import stdin


for line in stdin:
    print(' '.join(indic_tokenize.trivial_tokenize(line, 'ta')), end='')


