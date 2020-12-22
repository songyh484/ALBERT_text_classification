import pandas as pd


def get_balance_corpus(corpus_size, corpus_pos, corpus_neg):
    sample_size = corpus_size // 2
    pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size), \
                                   corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)])

    print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
    print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
    print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])

    print(pd_corpus_balance)
    return pd_corpus_balance


path = './data/'

pd_all = pd.read_csv(path + 'ChnSentiCorp_htl_all.csv')

'''
print('评论数目（总体）：%d' % pd_all.shape[0])
print('评论数目（正向）：%d' % pd_all[pd_all.label==1].shape[0])
print('评论数目（负向）：%d' % pd_all[pd_all.label==0].shape[0])
'''
pd_positive = pd_all[pd_all.label == 1]
pd_negative = pd_all[pd_all.label == 0]
ChnSentiCorp_htl_ba_6000 = get_balance_corpus(2000, pd_positive, pd_negative)
ChnSentiCorp_htl_ba_6000.to_csv('./data/train_htl.txt', sep=' ', header=None, index=False)

ChnSentiCorp_htl_ba_100 = get_balance_corpus(100, pd_positive, pd_negative)
ChnSentiCorp_htl_ba_100.to_csv('./data/test_htl.txt', sep=' ', header=None, index=False)


ChnSentiCorp_htl_ba_6000.sample(10)

