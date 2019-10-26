import os
import numpy as np
import pandas as pd


class getRawData():
    def __init__(self):
        self.p_mu = ['px', 'py', 'pz', 'e']
        self.particles = ['j1', 'j2', 'l1', 'v1', 'l2', 'v2']
        self.raw_features = [p_i + '.' + particle for particle in self.particles for p_i in self.p_mu]
        self.raw_targets = ['spin.W1', 'spin.W2']
        self.df_features = pd.DataFrame(columns=self.raw_features)
        self.df_targets = pd.DataFrame(columns=self.raw_targets)

    def load_csv_file(self, file_name, col_names):
        return pd.read_csv(file_name, header=None, names=col_names)

    def get_raw_targets_and_features(self, raw_data_dir):
        all_files = os.listdir(raw_data_dir)
        csv_files = [file_name for file_name in all_files if file_name.endswith('.csv')]
        csv_files.sort()
        for csv_file in csv_files:
            file_name = os.path.join(raw_data_dir, csv_file)
            if csv_file.endswith('decayed_1.csv'):
                df_new = self.load_csv_file(file_name, self.raw_features)
                self.df_features = self.df_features.append(df_new, ignore_index=True)
            else:
                df_new = self.load_csv_file(file_name, self.raw_targets)
                self.df_targets = self.df_targets.append(df_new, ignore_index=True)

        return self.df_targets, self.df_features


class featureEngineering():
    def __init__(self):
        self.l_rename = {'pt.l1':'pt.l2', 'eta.l1':'eta.l2', 'phi.l1':'phi.l2', 'zepp.l1':'zepp.l2', 'pt.l2':'pt.l1', 'eta.l2':'eta.l1', 'phi.l2':'phi.l1', 'zepp.l2':'zepp.l1'}
        self.j_rename = {'pt.j1':'pt.j2', 'eta.j1':'eta.j2', 'phi.j1':'phi.j2', 'pt.j2':'pt.j1', 'eta.j2':'eta.j1', 'phi.j2':'phi.j1'}

    def pt(self, px, py):
        # transverse momentum
        return np.sqrt(px**2 + py**2)

    def eta(self, pz, e):
        # pseudorapidity
        return np.arctanh(pz / e)

    def phi(self, px, py):
        # azimuthal angle
        return np.arctan2(px, py)

    def mm(self, e, px, py, pz):
        # invariant mass
        return np.sqrt(e**2 - (px**2 + py**2 + pz**2))

    def engineer_targets(self, targets):
        # target: 2 longitundally polarized Ws
        has_long_pol = ((targets['spin.W1'] == 0) & (targets['spin.W2'] == 0)).astype('int')
        return pd.DataFrame(has_long_pol, columns=['has_long_pol'])

    def engineer_features(self, df, features):
        # pt
        df['pt.j1'] = self.pt(features['px.j1'], features['py.j1'])
        df['pt.j2'] = self.pt(features['px.j2'], features['py.j2'])
        df['pt.l1'] = self.pt(features['px.l1'], features['py.l1'])
        df['pt.l2'] = self.pt(features['px.l2'], features['py.l2'])

        # eta
        df['eta.j1'] = self.eta(features['pz.j1'], features['e.j1'])
        df['eta.j2'] = self.eta(features['pz.j2'], features['e.j2'])
        df['eta.l1'] = self.eta(features['pz.l1'], features['e.l1'])
        df['eta.l2'] = self.eta(features['pz.l2'], features['e.l2'])

        # phi
        df['phi.j1'] = self.phi(features['px.j1'], features['py.j1'])
        df['phi.j2'] = self.phi(features['px.j2'], features['py.j2'])
        df['phi.l1'] = self.phi(features['px.l1'], features['py.l1'])
        df['phi.l2'] = self.phi(features['px.l2'], features['py.l2'])

        # missing transverse energy
        df['et.miss'] = self.pt(features['px.v1'] + features['px.v2'], features['py.v1'] + features['py.v2'])
        df['phi.miss'] = -1.0 * self.phi(features['px.v1'] + features['px.v2'], features['py.v1'] + features['py.v2'])

        # dijet system
        df['mm.jj'] = self.mm(features['e.j1'] + features['e.j2'], features['px.j1'] + features['px.j2'], features['py.j1'] + features['py.j2'], features['pz.j1'] + features['pz.j2'])
        df['delta_eta.jj'] = np.abs(df['eta.j1'] - df['eta.j2'])
        df['delta_phi.jj'] = np.minimum(np.abs(df['phi.j1'] - df['phi.j2']), 2 * np.pi - np.abs(df['phi.j1'] - df['phi.j2']))

        # Zeppenfeld variables
        df['zepp.l1'] = (df['eta.l1'] - (df['eta.j1'] + df['eta.j2']) / 2) / df['delta_eta.jj']
        df['zepp.l2'] = (df['eta.l2'] - (df['eta.j1'] + df['eta.j2']) / 2) / df['delta_eta.jj']

        # R_{ll,jj}
        df['r.lljj'] = np.sqrt(((df['eta.l1'] + df['eta.l2']) - (df['eta.j1'] + df['eta.j2']))** 2 + ((df['phi.l1'] + df['phi.l2']) - (df['phi.j1'] + df['phi.j2']))**2) / 2

        # cuts on leptons
        cut_1 = (df['pt.l1'] >= 20)
        cut_2 = (df['pt.l1'] >= 20)
        cut_3 = (np.abs(df['eta.l1']) <= 2.4)
        cut_4 = (np.abs(df['eta.l2']) <= 2.4)

        return df[cut_1 & cut_2 & cut_3 & cut_4]

    def pt_sorting(self, df):
        df11 = df[(df['pt.j1'] > df['pt.j2']) & (df['pt.l1'] > df['pt.l2'])]
        df12 = df[(df['pt.j1'] > df['pt.j2']) & (df['pt.l1'] < df['pt.l2'])]
        df21 = df[(df['pt.j1'] < df['pt.j2']) & (df['pt.l1'] > df['pt.l2'])]
        df22 = df[(df['pt.j1'] < df['pt.j2']) & (df['pt.l1'] < df['pt.l2'])]

        df12 = df12.rename(columns=self.l_rename)
        df21 = df21.rename(columns=self.j_rename)
        df22 = df22.rename(columns=self.l_rename).rename(columns=self.j_rename)

        dfs = df11.append(df12, ignore_index=True).append(df21, ignore_index=True).append(df22, ignore_index=True)

        return dfs.sample(frac=1).reset_index(drop=True)


if __name__ == '__main__':
    raw_data_dir = '/Users/christopherwmurphy/Documents/Research/NNtest/code_v2/raw_data'
    raw_targets, raw_features = getRawData().get_raw_targets_and_features(raw_data_dir)

    targets = featureEngineering().engineer_targets(raw_targets)
    features = featureEngineering().engineer_features(targets, raw_features)
    df = featureEngineering().pt_sorting(features)

    engineered_data_dir = '/Users/christopherwmurphy/Documents/Research/NNtest/code_v2/processed_data'
    df.to_csv(os.path.join(engineered_data_dir, 'samples.csv'), index=False)
