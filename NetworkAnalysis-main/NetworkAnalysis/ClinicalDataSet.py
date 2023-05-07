import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


class ClinicalDataSet(pd.DataFrame):
    @classmethod
    def from_file(cls, path, sep=',', header=0, column_index=0, patient_axis='auto'):
        network_df = pd.read_csv(path, sep=sep, header=header, column_index=column_index)
        return cls(network_df, patient_axis)

    def __init__(self, dataframe, patient_axis='auto', event_flag_col=None, event_time_col=None):
        self.event_flag_col = event_flag_col
        self.event_time_col = event_time_col

        if patient_axis == 'auto':
            if dataframe.shape[0] > dataframe.shape[1]:
                self.genes = dataframe.columns.values
                self.observations = dataframe.index
                super().__init__(dataframe)

            else:
                self.samples = dataframe.columns.values
                self.observations = dataframe.index
                super().__init__(dataframe.transpose())

        elif patient_axis == 1:
            self.genes = dataframe.index
            self.observations = dataframe.columns.values
            super().__init__(dataframe.transpose())

        elif patient_axis == 0:
            self.samples = dataframe.columns.values
            self.observations = dataframe.index
            super().__init__(dataframe)

        else:
            print('Please provide a correct axis number (0,1)')

    def getSurvivalMeasures(self, data, measure='HazardRatio'):
        '''
        :param os_group: a column or series object indicating the samples that have a particular condition
        :param measure: specifies the measure that is used to evaluate the survival characteristics of the os_group
        :return: survival_df
        '''
        pass

    def KaplanMeierPlot(self, groups, event_time_col=None, event_observed=None, group_names=None, savepath=None):

        if event_time_col is None:
            if self.event_time_col is None:
                raise Exception('Please specify the column that contains the event times.')
            else:
                event_time_col = self.event_time_col

        if event_observed is None:
            if self.event_flag_col is None:
                raise Exception('Please specify the columns that contains the event flags.')
            else:
                event_observed = self.event_flag_col

        # groups is an array of size nrow(clinical_df) with integers (0,1,2,3...)
        # indicating to which group each sample belongs
        group_labels = np.unique(groups)

        if group_names is None:
            group_names = group_labels

        kmf = KaplanMeierFitter()
        id = 0

        for i in group_labels:
            df = self.loc[groups == i]
            kmf.fit(df[event_time_col], event_observed=df[event_observed], label=group_names[i])

            if id == 0:
                ax = kmf.plot(ci_show=False, linewidth=2)
            else:
                kmf.plot(ax=ax, ci_show=False, linewidth=2)
            id += 1
        plt.xlabel('Survival times (months)', fontsize=14)
        plt.ylabel('Proportion alive', fontsize=14)
        plt.legend(fontsize=14)
        if savepath is not None:
            print('Saving figure to: ' + savepath)
            plt.savefig(savepath, dpi=1000, bbox_inches="tight")
        plt.show()
