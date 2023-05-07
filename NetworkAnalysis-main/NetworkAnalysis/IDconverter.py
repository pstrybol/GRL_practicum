import pandas as pd

class IDconverter(dict):

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    @classmethod
    def from_df(cls, df, colnames=None, **kwargs):
        assert isinstance(df, pd.DataFrame)

        if len(df.columns) == 2:
            df = df
        elif colnames is not None:
            df = df[list(colnames)]
        else:
            try:
                df = df.iloc[:, :2]
                print('No column names provided, taking the first two columns as inputs.')

            except IndexError:
                raise IOError('Make sure that the provided dataframe contains at least two columns.')

        return cls(zip(df.iloc[:, 0], df.iloc[:, 1]), **kwargs)

    @classmethod
    def from_file(cls, path, sep=',', header=0, index_col=None, **kwargs):

        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col, **kwargs)
        IDconverter.from_df(df, **kwargs)

    def getReverted(self):
        return IDconverter({v: k for k, v in self.items()})

    def mapdf(self):
        pass
