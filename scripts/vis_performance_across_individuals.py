import matplotlib.pyplot as plt
import pandas as pd
import sys,os

from plot_utils import plot_scattermeans

performance_data_dir = '../performance_data'
performance_data_file = 'lmu_nengo_230615-204923.csv'

prdf = pd.read_csv(os.path.join(performance_data_dir,performance_data_file),index_col=0)
print(prdf.head())

fig,ax = plt.subplots(1,1,figsize=(3,3))
plot_scattermeans(prdf,
                features = ['accuracy','specificity','sensitivity'],
                ax = ax,
                colors = ['tab:red','tab:green','tab:orange'],
                xlabels = ['Accuracy','Specificity','Sensitivity'],
                min = 0.0, max = 1.)
fig.tight_layout()
plt.show()
