import icd9
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal


#
# Test without dates
#

cat1 = ["12345", "54321"]
cat2 = ["44", "323"]
full = {"group1": cat1}
init = {"group2": cat2}
counter = icd9.Counter(codes_full=full, codes_initial=init)

chunk = pd.DataFrame()
chunk['id'] = [1, 2, 3, 4, 5]
chunk['code'] = ["12345", "12345", "32", "441", "54321"]
counter.update(chunk, 'id')

expected = pd.DataFrame([[1, 0], [1, 0], [0, 0], [0, 1], [1, 0]],
                        index=[1, 2, 3, 4, 5], dtype=np.float64,
                        columns=["group1 [N]", "group2 [N]"])

counter.table = counter.table.loc[:, expected.columns]
assert_frame_equal(counter.table, expected)

chunk = pd.DataFrame()
chunk['id'] = [1, 2, 5, 6, 6]
chunk['code'] = ["12345", "440", "32", "441", "54321"]
counter.update(chunk, 'id')

expected = pd.DataFrame([[2, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]],
                        index=[1, 2, 3, 4, 5, 6], dtype=np.float64,
                        columns=["group1 [N]", "group2 [N]"])

# The columns need to be in the same order.
counter.table = counter.table.loc[:, expected.columns]
assert_frame_equal(counter.table, expected)


#
# Test with dates
#

cat1 = ["12345"]
cat2 = ["66"]
cat3 = ["44"]
full = {"group1": cat1}
init = {"group1": cat2, "group2": cat3}
dt = icd9.Counter(codes_full=full, codes_initial=init, calculate_dates=True)

chunk = pd.DataFrame()
chunk['id'] = [1, 1, 2, 3, 4, 5]
chunk["code"] = ["12345", "12345", "4424", "99", "12345", "6600"]
chunk["date"] = ["2014-6-1", "2014-4-1", "2014-5-1", "2014-5-1", "2014-3-1", "2014-5-1"]
chunk["date"] = pd.to_datetime(chunk["date"])
dt.update(chunk, 'id', 'date')

chunk = pd.DataFrame()
chunk["id"] = [1, 1, 2, 2, 5, 5, 5, 5]
chunk["code"] = ["66xx", "12345", "99", "12345", "12345", "4400", "66", "663"]
chunk["date"] = ["2014-2-1", "2014-8-1", "2014-5-1", "2014-4-1", "2014-5-1", "2014-6-1", "2014-7-1", "2014-4-1"]
chunk["date"] = pd.to_datetime(chunk["date"])
dt.update(chunk, 'id', 'date')

df = [['index', 'group1 [N]', 'group1 [first]', 'group1 [last]', 'group2 [N]', 'group2 [first]', 'group2 [last]'],
      [1, 4, '2014-02-01', '2014-08-01', 0, 'NaT', 'NaT'],
      [2, 1, '2014-04-01', '2014-04-01', 1, '2014-05-01', '2014-05-01'],
      [3, 0, 'NaT', 'NaT', 0, 'NaT', 'NaT'],
      [4, 1, '2014-03-01', '2014-03-01', 0, 'NaT', 'NaT'],
      [5, 4, '2014-04-01', '2014-07-01', 1, '2014-06-01', '2014-06-01']]

expected = pd.DataFrame(df[1:], columns=df[0])
expected = expected.set_index('index')
expected['group1 [first]'] = pd.to_datetime(expected['group1 [first]'])
expected['group1 [last]'] = pd.to_datetime(expected['group1 [last]'])
expected['group2 [first]'] = pd.to_datetime(expected['group2 [first]'])
expected['group2 [last]'] = pd.to_datetime(expected['group2 [last]'])
expected['group1 [N]'] = expected['group1 [N]'].astype(np.float64)
expected['group2 [N]'] = expected['group2 [N]'].astype(np.float64)
expected.index.name = None

# The columns need to be in the same order.
assert_frame_equal(dt.table, expected)


