import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from imblearn.ensemble import BalancedRandomForestClassifier

from src.analysis import classical, classification_report
from src.processing import processData
from src.utils import pkl_save_obj

# Process Data

process_data = processData()

dfww = (process_data.get_pT_sorted_events('data/jjWpmWpm_undecayed_01.csv')
        .append(process_data.get_pT_sorted_events('data/jjWpmWpm_undecayed_02.csv'),
                                                  ignore_index=True)
        .append(process_data.get_pT_sorted_events('data/jjWpmWpm_undecayed_03.csv'),
                                                  ignore_index=True))

X = (dfww
     .drop('n_lon', axis = 1))
y = (dfww['n_lon'] == 2)

skf = StratifiedKFold(n_splits=5,
                      random_state=30)

# Classic Machine Learning
scores = dict()

## $\Delta\phi_{jj}$
X_phijj = (X['delta_phi.jj']
           .values
           .reshape(-1, 1))

phi_jj = LogisticRegression(solver='liblinear')

scores['Delta phi_jj'] = classical(phi_jj, X_phijj, y, skf)

print('\n' + 'Delta phi_jj' + '\n----------')
classification_report(scores['Delta phi_jj'])
print('\n')

## Random Forest
rfc_names = ['Random Forest 1000/5',
             'Random Forest 1000/10',
             'Random Forest 500/None',
             'Random Forest 1000/None',
             'Weighted Random Forest 1000/10, 1:2',
             'Weighted Random Forest 1000/10, 1:3',
             'Weighted Random Forest 1000/None, 1:3',
             'Weighted Random Forest 1000/None, balanced subsample',
             'Balanced Random Forest 2000/10',
             'Balanced Random Forest 1000/None',
             'Balanced Random Forest 2000/None']

rfc_params = [{'n_estimators':1000, 'max_depth':5},
              {'n_estimators':1000, 'max_depth':10},
              {'n_estimators':500},
              {'n_estimators':1000},
              {'n_estimators':1000, 'max_depth':10, 'class_weight':{0:1, 1:2}},
              {'n_estimators':1000, 'max_depth':10, 'class_weight':{0:1, 1:3}},
              {'n_estimators':1000, 'class_weight':{0:1, 1:3}},
              {'n_estimators':1000, 'class_weight':'balanced_subsample'},
              {'n_estimators':2000, 'max_depth':10},
              {'n_estimators':1000},
              {'n_estimators':2000}]


for i in range(len(rfc_names)):
    name = rfc_names[i]
    params = rfc_params[i]
    params['n_jobs'] = -1

    if name[:8] == 'Balanced':
        rfc = BalancedRandomForestClassifier(**params)
    else:
        rfc = RandomForestClassifier(**params)

    scores[name] = classical(rfc, X.values, y, skf)

    print('\n' + name + '\n----------')
    classification_report(scores[name])
    print('\n')

# Save Results
pkl_save_obj(scores, 'classical_scores_gs4')
