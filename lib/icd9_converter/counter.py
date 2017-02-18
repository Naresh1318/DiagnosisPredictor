"""
Functions for counting the occurences or tracking the dates of ICD9
codes within a set of categories.  These can be used for calculating
comorbidity scores and for other reporting functions.
"""

import pandas as pd
import numpy as np


class Counter(object):
    """
   Summaries of ICD9 code category matches to data.

    A 'category' is a set of ICD9 codes, or initial substrings of ICD9
    codes.  A code associated with a particular subject matches a
    category if it matches one of the ICD9 codes defining the
    category.  This class computes the number of matches for each
    subject to codes in each category, and optionally calculates the
    first and last date at which a code category is matched, based on
    provided service dates.

    Parameters
    ----------
    codes_full : list-like or dict-like
        List of ICD9 codes in a category, or a dict mapping from
        category names to lists of ICD9 codes belonging to the
        separate categories.
    codes_initial : list-like or dict-like
        List of ICD9 code initial substrings, or a dict mapping
        category names to lists of ICD9 code initial substrings.  The
        category will match any code whose initial substring matches
        an element of the corresponding list.
    date_var : string
        A column name containing a service date associated with each
        code.

    Attributes
    ----------
    table : Series or DataFrame

        A Series or DataFrame whose rows correspond to subjects.
        Columns containing '[N]' contain the number of matches in a
        service category for each subject, and columns containing
        '[first]' or '[last]' contain the first or last service date
        where a match occured.

    Notes
    -----
    This class does not perform any standardization or validation of
    the ICD9 codes.  The subject-associated codes must match the
    target codes either exactly (for `full_codes`) or must match their
    initial substrings exactly (for `initial_codes`).

    The `update` method allows calculations to be done on large data
    sets that cannot be stored in memory.  Pass through the data set
    in chunks and call `update` on each chunk.

    Examples
    --------
    Calculate the number of codes matching any of the Elixhauser
    'renal' category codes.

    >>> counter = icd9.Counter(codes_full={'Renal': icd9.elixComorbid['Renal']})
    >>> counter.update(chunk1, 'id')

    Calculate the number of codes matching any of the Elixhauser
    'renal' category codes, and the first and last date at which a
    match occured.

    >>> counter = icd9.Counter(codes_full={'Renal': icd9.elixComorbid['Renal']},
                               calculate_dates=True)
    >>> counter.update(chunk1, 'id', 'date')

    Calculate the Elixhauser comorbidity score, defined as the number
    of ICD9 categories in which a subject has at least one reported
    ICD9 code.  `chunk1` and `chunk2` are pandas.Series objects whose
    keys are subject ids, and whose values are ICD9 codes.

    >>> counter = icd9.Counter(codes_full=icd9.elixComorbid)
    >>> counter.update(chunk1, 'id')
    >>> counter.update(chunk2, 'id')
    >>> elix = (counter.table > 0).sum(1)

    Set up a counter that will count the number of codes per subject
    in two categories.  The first category consists of exactly two
    codes: `12345` and `54321`.  The second category consists of any
    code beginning with the string `44`, and any string beginning with
    the string `323`.

    >>> cat1 = ['12345', '54321']
    >>> cat2 = ['44', '323']
    >>> full = {'group1': cat1}
    >>> init = {'group2': cat2}
    >>> counter = icd9.Counter(codes_full=full, codes_initial=init)
    >>> counter.update(chunk)
    """

    def __init__(self, calculate_dates=False, codes_full=None, codes_initial=None):

        self.codes_full = codes_full if codes_full is not None else {}
        self.codes_initial = codes_initial if codes_initial is not None else {}
        self.calculate_dates = calculate_dates

        columns = set(self.codes_full.keys()) | set(self.codes_initial.keys())
        columns = list(columns)
        columns.sort()
        self.table = pd.DataFrame()
        for col in columns:
            self.table[col + " [N]"] = pd.Series([], dtype=np.float64)
            if self.calculate_dates:
                self.table[col + " [first]"] = pd.Series([], dtype='datetime64[ns]')
                self.table[col + " [last]"] = pd.Series([], dtype='datetime64[ns]')

        self._matcher = {}
        for col in self.codes_initial:
            self._matcher[col] = self._create_matcher(col)


    def _create_matcher(self, col):
        return lambda x : any([x.startswith(y) for y in self.codes_initial[col]])


    def _update_sums(self, idval, col):
        if idval.shape[0] == 0:
            return
        val = idval.groupby(idval.values).agg(len)
        vname = col + " [N]"
        self.table.loc[:, vname].fillna(val, inplace=True)
        self.table.loc[:, vname] = self.table.loc[:, vname].add(val, fill_value=0)


    def _update_dates(self, codes, col, id_var, date_var):
        if codes.shape[0] == 0:
            return
        gb = codes.groupby(codes[id_var])
        for typ, func, vname in (("min", np.min, col + " [first]"),
                                 ("max", np.max, col + " [last]")):
            val = gb.agg({date_var: func})
            self.table.loc[:, vname].fillna(val[date_var], inplace=True)
            tdiff = self.table.loc[val.index, vname] - val[date_var]
            if typ == "min":
                ix = tdiff > 0
            else:
                ix = tdiff < 0
            if not ix.any():
                continue
            ix = ix.index[ix]
            self.table.loc[ix, vname] = val.loc[ix, date_var]


    def update(self, codes, id_var, date_var=None):
        """
        Update the table of ICD9 code counts according to a given set
        of values.

        Parameters
        ----------
        codes : pandas.DataFrame object
            A pandas.DataFrame object whose index contains subject
            identifiers.  The DataFrame should have a column
            containing subject identifiers, a column containing code
            values, and optionally a column containing time values.
        id_var : string
            The name of the column of `codes` containing subjects id's.
        date_var : string
            The name of the colum of `codes` containing servie times.
        """

        cols = set(codes.columns) - set([id_var])
        if date_var is not None:
            cols = cols - set([date_var])
        assert(len(cols) == 1)
        code_var = list(cols)[0]

        # Add rows for new index values
        ix = np.setdiff1d(codes[id_var].unique(), self.table.index)
        df = pd.DataFrame(index=ix, dtype=np.float64)
        for col in self.table.columns:
            if "[N]" in col:
                df[col] = 0.
                df[col] = df[col].astype(np.float64)
            else:
                df[col] = pd.NaT
                df[col] = df[col].astype('datetime64[ns]')
        self.table = self.table.append(df)

        for col in self.codes_full.keys():
            ix = codes[code_var].isin(self.codes_full[col])
            self._update_sums(codes.loc[ix, id_var], col)
            if self.calculate_dates:
                self._update_dates(codes.loc[ix, :], col, id_var, date_var)

        for col in self.codes_initial.keys():
            ix = codes[code_var].apply(self._matcher[col])
            self._update_sums(codes.loc[ix, id_var], col)
            if self.calculate_dates:
                self._update_dates(codes.loc[ix, :], col, id_var, date_var)
