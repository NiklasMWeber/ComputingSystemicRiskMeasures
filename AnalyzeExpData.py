import os
import pandas as pd

path_to_package = os.path.abspath(os.getcwd()).split('GNN')[0] + 'GNN/LearnRiskMeasure/'
dt_string = '2025_04_17_22_37_59_Outer_Bench_AVAR'
out_path = path_to_package + 'Experiments/' + dt_string + '/'
filename = out_path + 'exp_log.txt'

# 2025_04_17_14_08_51_Inner_AVAR
# 2025_04_18_02_02_24_Inner_Bench_AVAR
# 2025_04_17_22_21_36_Outer_AVAR
# 2025_04_17_22_37_59_Outer_Bench_AVAR
def read_file_to_list_of_lists(filename):
    with open(filename, 'r') as file:
        list_of_lists = [line.strip().split('\t') for line in file]
    return list_of_lists

list_of_lists = read_file_to_list_of_lists(filename)
list_of_lists = [line for line in list_of_lists if not len(line) <= 2]
for line in list_of_lists:
    line[1] = int(line[1])

list_of_info = [line for line in list_of_lists if line[2] == 'Info']
relevant_info = ['Datatype', 'Model', 'Seed']  # 'Batch Size', 'Learning Rate',
exp_ids = list(set([int(line[1]) for line in list_of_info]))

id_to_tuple = {}
for id in exp_ids:
    tuple = []
    for k, info in enumerate(relevant_info):
        filtered_list = [line for line in list_of_info if line[1] == id and line[3] == info]
        element = filtered_list[0][4]
        tuple.append(element)
    id_to_tuple[id] = tuple

id_to_tuple_df = pd.DataFrame(id_to_tuple.items(), columns=['Id', 'Tuple'])
id_to_tuple_df[relevant_info] = pd.DataFrame(id_to_tuple_df['Tuple'].tolist(), index=id_to_tuple_df.index)
id_to_tuple_df.drop(columns=['Tuple'], inplace=True)

list_of_results = [line for line in list_of_lists if line[2] in ['Final', 'Best']]
results_df = pd.DataFrame(list_of_results, columns = ['Time', 'Id', 'Final', 'Epoch', 'Train Risk Label', 'Train Risk', 'Val Risk Label', 'Val Risk', 'Test Risk Label', 'Test Risk', 'Capital Label', 'Capital'])
results_df[['Train Risk',  'Val Risk', 'Test Risk',  'Capital']] = results_df[['Train Risk',  'Val Risk', 'Test Risk',  'Capital']].astype('float64')
results_df = results_df.drop(['Time', 'Final', 'Epoch', 'Train Risk Label',  'Val Risk Label',  'Test Risk Label',  'Capital Label'], axis=1)

merged_results_df = pd.merge(results_df, id_to_tuple_df, on='Id')
stats = merged_results_df.groupby(['Datatype', 'Model'])[['Train Risk', 'Val Risk', 'Test Risk', 'Capital']].agg(['mean', 'std']).reset_index()
stats.columns = ['Datatype', 'Model', 'Train Risk Mean', 'Train Risk Std', 'Val Risk Mean', 'Val Risk Std', 'Test Risk Mean', 'Test Risk Std', 'Capital Mean', 'Capital Std']

results = pd.merge(merged_results_df, stats, on=['Datatype', 'Model'])

results_no_dup = results.drop(['Id', 'Seed', 'Train Risk', 'Val Risk', 'Test Risk', 'Capital'], axis=1).drop_duplicates()
results_no_dup.to_csv(out_path + dt_string + '.csv', index=False, sep=';', decimal='.')

print('done')