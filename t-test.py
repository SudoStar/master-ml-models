from scipy import stats

imp_nine_ind = [39.98, 49.97, 50.56, 50.18, 59.62, 58.74, 51.05, 57.58, 47.59]
# imp_nine_ind = [51.7]
imp_nine_hol = 50.98

print(stats.ttest_1samp(a=imp_nine_ind, popmean=imp_nine_hol))
