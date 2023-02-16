# %%
import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# %%
# Variable to store results data
csvs = [pd.DataFrame(dict()), pd.DataFrame(dict()), pd.DataFrame(dict())]
id_numbers = ['607', '8169', '5738']


# %%
csvs_dir = os.path.join('results', 'model_csvs')

for file_name in glob.glob(csvs_dir + '/*.csv'):
    for i in range(len(id_numbers)):
        if id_numbers[i] in file_name:
            csvs[i] = pd.read_csv(file_name, index_col=0)


# %%
for tables_csvs, ids in zip(csvs, id_numbers):
    sns.set(font_scale = 2)
    plt.figure(figsize=(20, 14))

    # hplt = sns.histplot(tables_csvs.melt(),
    #                         x='value',
    #                         hue='variable',
    #                         element='step',
    #                         fill=True,
    #                         bins=50,
    #                         log_scale=False,
    #                         cumulative=True,
    #                         alpha=0.7)
    tables_csvs.rename(columns={"y_hat": "$\hat{y}$"}, inplace=True)

    hplt = sns.ecdfplot(data=tables_csvs.melt(),
                        x='value',
                        hue='variable',
                        log_scale=True,
                        )

    plt.setp(hplt.get_legend().get_texts(), fontsize='22') # for legend text
    plt.setp(hplt.get_legend().get_title(), fontsize='32') # for legend title

    hplt.set_ylabel('Probability', fontsize=30)
    hplt.set_xlabel('CDR (log)', fontsize=30)

    hplt.axes.set_title(f'CDF of tower {ids} value and predictions', fontsize=50)

    #plt.legend(['y', '$\hat{y}$'], title='Variable')

    plt.savefig(os.path.join('results', 'model_csvs', f'{ids}_results_english.png'))
    plt.show()
# %%
