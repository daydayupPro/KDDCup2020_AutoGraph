# generate features
import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import node_classification
import time
from collections import Counter
from utils import normalize_features

def dayday_feature(data, n_class=2, label_most_common_1=19, flag_unlabel=0):
    t1 = time.time()
    data = data.copy()
    x = data['fea_table'].copy()
    num_nodes = x.shape[0]
    nodes_all = list(x.index)
    df = data['edge_file'].copy()
    max_weight = max(df['edge_weight'])
    df.rename(columns={'edge_weight': 'weight'}, inplace=True)
    
    degree_in_1st = np.zeros(num_nodes)
    degree_out_1st = np.zeros(num_nodes)
    weight_in_1st = np.zeros(num_nodes)
    weight_out_1st = np.zeros(num_nodes)
    for source, target, weight in df.values:
        source = int(source)
        target = int(target)
        degree_in_1st[target] += 1
        degree_out_1st[source] += 1
        weight_in_1st[target] += weight
        weight_out_1st[source] += weight
    
    degree_1st_diff = degree_in_1st - degree_out_1st
    weight_1st_diff = weight_in_1st - weight_out_1st

    features_1 = np.concatenate([        
        degree_in_1st.reshape(-1, 1),
        degree_out_1st.reshape(-1, 1),
        weight_in_1st.reshape(-1, 1),
        weight_out_1st.reshape(-1, 1),
        degree_1st_diff.reshape(-1, 1),
        weight_1st_diff.reshape(-1, 1)
    ], axis=1)
    
    features_in_1st = pd.DataFrame({"node_index": np.arange(num_nodes), "degree_in_1st": degree_in_1st, "weight_in_1st": weight_in_1st})
    df_degree_in_1st = pd.merge(left=df, right=features_in_1st, left_on="src_idx", right_on="node_index", how="left")
    df_degree_in_1st_info = df_degree_in_1st.groupby('dst_idx')['degree_in_1st'].agg({
        'degree_in_1st_sum': np.sum, 'degree_in_1st_mean': np.mean, 'degree_in_1st_min': np.min, 'degree_in_1st_max': np.max, 'degree_in_1st_median': np.median
    })
    df_weight_in_1st_info = df_degree_in_1st.groupby('dst_idx')['weight_in_1st'].agg({
        'weight_in_1st_sum': np.sum, 'weight_in_1st_mean': np.mean, 'weight_in_1st_min': np.min, 'weight_in_1st_max': np.max, 'weight_in_1st_median': np.median
    })

    df_degree_in_2nd = pd.DataFrame({"node_index": df_degree_in_1st_info.index, "degree_in_2nd": df_degree_in_1st_info['degree_in_1st_sum']})
    df_degree_in_2nd = pd.merge(left=df, right=df_degree_in_2nd, how="left", left_on="src_idx", right_on="node_index")
    df_degree_in_2nd_info = df_degree_in_2nd.groupby('dst_idx')['degree_in_2nd'].agg({
        'degree_in_2nd_sum': np.sum, 'degree_in_2nd_mean': np.mean, 'degree_in_2nd_min': np.min, 'degree_in_2nd_max': np.max, 'degree_in_2nd_median': np.median
    })
    features_2_index = df_degree_in_1st_info.index
    features_2_t = np.hstack([df_degree_in_1st_info.values, df_weight_in_1st_info.values, df_degree_in_2nd_info.values])
    features_2 = np.zeros((num_nodes, features_2_t.shape[1]))
    for i, index in enumerate(features_2_index):
        features_2[index] = features_2_t[i]


    train_y = data['train_label'].copy()
    df_info_in = pd.merge(left=df, right=train_y, how='left', left_on='src_idx', right_on='node_index')
    if flag_unlabel == 0:
        df_info_in.dropna(inplace=True)
    else:
        df_info_in.fillna(-1, inplace=True)

    df_labels_in_count = df_info_in.pivot_table(index=["dst_idx"], columns='label', aggfunc='size', fill_value=0)
    df_labels_in_precent = pd.crosstab(index=df_info_in.dst_idx, columns=df_info_in.label, normalize='index')

    df_labels_in_without_most_common = df_info_in.copy()
    df_labels_in_without_most_common = df_labels_in_without_most_common[df_labels_in_without_most_common.label != label_most_common_1]
    df_labels_in_precent_without_most_common = pd.crosstab(index=df_labels_in_without_most_common.dst_idx, columns=df_labels_in_without_most_common.label, normalize='index')

    df_labels_weight_count_in = df_info_in.pivot_table(index=['dst_idx'], columns='label', values='weight', aggfunc='sum', fill_value=0)
    df_labels_weight_percent_in = pd.crosstab(index=df_info_in.dst_idx, columns=df_info_in.label, values=df_info_in.weight, aggfunc='sum', normalize='index')

    df_labels_weight_percent_in_without_most_common = pd.crosstab(
        index=df_labels_in_without_most_common.dst_idx, columns=df_labels_in_without_most_common.label, values=df_labels_in_without_most_common.weight, 
        aggfunc='sum', normalize='index')

    features_3_index = list(df_labels_in_count.index)
    features_3_t = np.hstack((df_labels_in_count.values, df_labels_in_precent.values, df_labels_weight_count_in.values, df_labels_weight_percent_in.values))
    features_3 = np.zeros((num_nodes, features_3_t.shape[1]))
    for i, index in enumerate(features_3_index):
        features_3[index] = features_3_t[i]

    labels_in_temp = features_3[:, :n_class]
    labels_weight_in_temp = features_3[:, 2*n_class:3*n_class]
    features_labels_all_in_2nd = np.zeros((num_nodes, n_class))
    features_labels_weight_all_in_2nd = np.zeros((num_nodes, n_class))
    for source, target, weight in df.values:
        source = int(source)
        target = int(target)
        features_labels_all_in_2nd[source] += labels_in_temp[target]
        features_labels_weight_all_in_2nd[source] += labels_weight_in_temp[target]
    features_labels_all_in_2nd_percent = np.delete(features_labels_all_in_2nd, label_most_common_1, axis=1)
    features_labels_all_in_2nd_percent = normalize_features(features_labels_all_in_2nd_percent)

    features_out_1st = pd.DataFrame({"node_index": np.arange(num_nodes), "degree_out_1st": degree_out_1st, "weight_out_1st": weight_out_1st})
    df_degree_out_1st = pd.merge(left=df, right=features_out_1st, left_on="dst_idx", right_on="node_index", how="left")
    df_degree_out_1st_info = df_degree_out_1st.groupby('src_idx')['degree_out_1st'].agg({
        'degree_out_1st_sum': np.sum, 'degree_out_1st_mean': np.mean, 'degree_out_1st_min': np.min, 'degree_out_1st_max': np.max, 'degree_out_1st_median': np.median
    })
    df_weight_out_1st_info = df_degree_out_1st.groupby('src_idx')['weight_out_1st'].agg({
        'weight_out_1st_sum': np.sum, 'weight_out_1st_mean': np.mean, 'weight_out_1st_min': np.min, 'weight_out_1st_max': np.max, 'weight_out_1st_median': np.median
    })

    df_degree_out_2nd = pd.DataFrame({"node_index": df_degree_out_1st_info.index, "degree_out_2nd": df_degree_out_1st_info['degree_out_1st_sum']})
    df_degree_out_2nd = pd.merge(left=df, right=df_degree_out_2nd, how="left", left_on="dst_idx", right_on="node_index")
    df_degree_out_2nd_info = df_degree_out_2nd.groupby('src_idx')['degree_out_2nd'].agg({
        'degree_out_2nd_sum': np.sum, 'degree_out_2nd_mean': np.mean, 'degree_out_2nd_min': np.min, 'degree_out_2nd_max': np.max, 'degree_out_2nd_median': np.median
    })
    features_4_index = df_degree_out_1st_info.index
    features_4_t = np.hstack([df_degree_out_1st_info.values, df_weight_out_1st_info.values, df_degree_out_2nd_info.values])
    features_4 = np.zeros((num_nodes, features_4_t.shape[1]))
    for i, index in enumerate(features_4_index):
        features_4[index] = features_4_t[i]

    df_info_out = pd.merge(left=df, right=train_y, how='left', left_on='dst_idx', right_on='node_index')
    if flag_unlabel == 0:
        df_info_out.dropna(inplace=True)
    else:
        df_info_out.fillna(-1, inplace=True)

    df_labels_out_count = df_info_out.pivot_table(index=["src_idx"], columns='label', aggfunc='size', fill_value=0)
    df_labels_out_precent = pd.crosstab(index=df_info_out.src_idx, columns=df_info_out.label, normalize='index')

    df_labels_out_without_most_common = df_info_out.copy()
    df_labels_out_without_most_common = df_labels_out_without_most_common[df_labels_out_without_most_common.label != label_most_common_1]
    df_labels_out_precent_without_most_common = pd.crosstab(index=df_labels_out_without_most_common.src_idx, columns=df_labels_out_without_most_common.label, normalize='index')

    df_labels_weight_count_out = df_info_out.pivot_table(index=['src_idx'], columns='label', values='weight', aggfunc='sum', fill_value=0)
    df_labels_weight_percent_out = pd.crosstab(index=df_info_out.src_idx, columns=df_info_out.label, values=df_info_out.weight, aggfunc='sum', normalize='index')
    df_labels_weight_percent_out_without_most_common = pd.crosstab(
        index=df_labels_out_without_most_common.src_idx, columns=df_labels_out_without_most_common.label, values=df_labels_out_without_most_common.weight, 
        aggfunc='sum', normalize='index')

    features_5_index = list(df_labels_out_count.index)
    features_5_t = np.hstack((df_labels_out_count.values, df_labels_out_precent.values, df_labels_weight_count_out.values, df_labels_weight_percent_out.values))
    features_5 = np.zeros((num_nodes, features_5_t.shape[1]))
    for i, index in enumerate(features_5_index):
        features_5[index] = features_5_t[i]

    features_merge = np.concatenate([
        features_1,
        features_2,
        features_3,
        features_4,
        features_5,
        features_labels_all_in_2nd,
        features_labels_all_in_2nd_percent
    ], axis=1)
    features_merge = np.unique(features_merge, axis=1)
    features_merge = np.delete(features_merge, np.argwhere(np.sum(features_merge, axis=0)==0), axis=1)

    return features_merge



def dayday_feature_old(data, flag_unlabel=0):
    t1 = time.time()
    data = data.copy()
    x = data['fea_table'].copy()
    num_nodes = x.shape[0]
    nodes_all = list(x.index)
    df = data['edge_file'].copy()
    max_weight = max(df['edge_weight'])
    df.rename(columns={'edge_weight': 'weight'}, inplace=True)
    
    degree_in_1st = np.zeros(num_nodes)
    degree_out_1st = np.zeros(num_nodes)
    weight_in_1st = np.zeros(num_nodes)
    weight_out_1st = np.zeros(num_nodes)
    for source, target, weight in df.values:
        source = int(source)
        target = int(target)
        degree_in_1st[target] += 1
        degree_out_1st[source] += 1
        weight_in_1st[target] += weight
        weight_out_1st[source] += weight
    
    degree_1st_diff = degree_in_1st - degree_out_1st
    weight_1st_diff = weight_in_1st - weight_out_1st

    features_1 = np.concatenate([        
        degree_in_1st.reshape(-1, 1),
        degree_out_1st.reshape(-1, 1),
        weight_in_1st.reshape(-1, 1),
        weight_out_1st.reshape(-1, 1),
        degree_1st_diff.reshape(-1, 1),
        weight_1st_diff.reshape(-1, 1)
    ], axis=1)
    
    features_in_1st = pd.DataFrame({"node_index": np.arange(num_nodes), "degree_in_1st": degree_in_1st, "weight_in_1st": weight_in_1st})
    df_degree_in_1st = pd.merge(left=df, right=features_in_1st, left_on="src_idx", right_on="node_index", how="left")
    df_degree_in_1st_info = df_degree_in_1st.groupby('dst_idx')['degree_in_1st'].agg({
        'degree_in_1st_sum': np.sum, 'degree_in_1st_mean': np.mean, 'degree_in_1st_min': np.min, 'degree_in_1st_max': np.max, 'degree_in_1st_median': np.median
    })
    df_weight_in_1st_info = df_degree_in_1st.groupby('dst_idx')['weight_in_1st'].agg({
        'weight_in_1st_sum': np.sum, 'weight_in_1st_mean': np.mean, 'weight_in_1st_min': np.min, 'weight_in_1st_max': np.max, 'weight_in_1st_median': np.median
    })

    df_degree_in_2nd = pd.DataFrame({"node_index": df_degree_in_1st_info.index, "degree_in_2nd": df_degree_in_1st_info['degree_in_1st_sum']})
    df_degree_in_2nd = pd.merge(left=df, right=df_degree_in_2nd, how="left", left_on="src_idx", right_on="node_index")
    df_degree_in_2nd_info = df_degree_in_2nd.groupby('dst_idx')['degree_in_2nd'].agg({
        'degree_in_2nd_sum': np.sum, 'degree_in_2nd_mean': np.mean, 'degree_in_2nd_min': np.min, 'degree_in_2nd_max': np.max, 'degree_in_2nd_median': np.median
    })
    features_2_index = df_degree_in_1st_info.index
    features_2_t = np.hstack([df_degree_in_1st_info.values, df_weight_in_1st_info.values, df_degree_in_2nd_info.values])
    features_2 = np.zeros((num_nodes, features_2_t.shape[1]))
    for i, index in enumerate(features_2_index):
        features_2[index] = features_2_t[i]

    train_y = data['train_label'].copy()
    df_info_in = pd.merge(left=df, right=train_y, how='left', left_on='src_idx', right_on='node_index')
    if flag_unlabel == 0:
        df_info_in.dropna(inplace=True)
    else:
        df_info_in.fillna(-1, inplace=True)

    df_labels_in = df_info_in.loc[:, ['dst_idx', 'label']]
    df_labels_in.rename(columns={'dst_idx': 'node_index','label': 'src_label'}, inplace=True)
    df_labels_in_count = df_labels_in.pivot_table(index=["node_index"], columns='src_label', aggfunc='size', fill_value=0)
    df_labels_in_precent = pd.crosstab(index=df_labels_in.node_index, columns=df_labels_in.src_label, normalize='index')

    df_labels_weight_count_in = df_info_in.pivot_table(index=['dst_idx'], columns='label', values='weight', aggfunc='sum', fill_value=0)
    df_labels_weight_percent_in = pd.crosstab(index=df_info_in.dst_idx, columns=df_info_in.label, values=df_info_in.weight, aggfunc='sum', normalize='index')

    features_3_index = list(df_labels_in_count.index)
    features_3_t = np.hstack((df_labels_in_count.values, df_labels_in_precent.values, df_labels_weight_count_in.values, df_labels_weight_percent_in.values))
    features_3 = np.zeros((num_nodes, features_3_t.shape[1]))
    for i, index in enumerate(features_3_index):
        features_3[index] = features_3_t[i]

    features_out_1st = pd.DataFrame({"node_index": np.arange(num_nodes), "degree_out_1st": degree_out_1st, "weight_out_1st": weight_out_1st})
    df_degree_out_1st = pd.merge(left=df, right=features_out_1st, left_on="dst_idx", right_on="node_index", how="left")
    df_degree_out_1st_info = df_degree_out_1st.groupby('src_idx')['degree_out_1st'].agg({
        'degree_out_1st_sum': np.sum, 'degree_out_1st_mean': np.mean, 'degree_out_1st_min': np.min, 'degree_out_1st_max': np.max, 'degree_out_1st_median': np.median
    })
    df_weight_out_1st_info = df_degree_out_1st.groupby('src_idx')['weight_out_1st'].agg({
        'weight_out_1st_sum': np.sum, 'weight_out_1st_mean': np.mean, 'weight_out_1st_min': np.min, 'weight_out_1st_max': np.max, 'weight_out_1st_median': np.median
    })
    df_degree_out_2nd = pd.DataFrame({"node_index": df_degree_out_1st_info.index, "degree_out_2nd": df_degree_out_1st_info['degree_out_1st_sum']})
    df_degree_out_2nd = pd.merge(left=df, right=df_degree_out_2nd, how="left", left_on="dst_idx", right_on="node_index")
    df_degree_out_2nd_info = df_degree_out_2nd.groupby('src_idx')['degree_out_2nd'].agg({
        'degree_out_2nd_sum': np.sum, 'degree_out_2nd_mean': np.mean, 'degree_out_2nd_min': np.min, 'degree_out_2nd_max': np.max, 'degree_out_2nd_median': np.median
    })
    features_4_index = df_degree_out_1st_info.index
    features_4_t = np.hstack([df_degree_out_1st_info.values, df_weight_out_1st_info.values, df_degree_out_2nd_info.values])
    features_4 = np.zeros((num_nodes, features_4_t.shape[1]))
    for i, index in enumerate(features_4_index):
        features_4[index] = features_4_t[i]

    df_info_out = pd.merge(left=df, right=train_y, how='left', left_on='dst_idx', right_on='node_index')
    if flag_unlabel == 0:
        df_info_out.dropna(inplace=True)
    else:
        df_info_out.fillna(-1, inplace=True)

    df_labels_out = df_info_out.loc[:, ['src_idx', 'label']]
    df_labels_out.rename(columns={'src_idx': 'node_index','label': 'dst_label'}, inplace=True)
    df_labels_out_count = df_labels_out.pivot_table(index=["node_index"], columns='dst_label', aggfunc='size', fill_value=0)
    df_labels_out_precent = pd.crosstab(index=df_labels_out.node_index, columns=df_labels_out.dst_label, normalize='index')
    df_labels_weight_count_out = df_info_out.pivot_table(index=['src_idx'], columns='label', values='weight', aggfunc='sum', fill_value=0)
    df_labels_weight_percent_out = pd.crosstab(index=df_info_out.src_idx, columns=df_info_out.label, values=df_info_out.weight, aggfunc='sum', normalize='index')

    features_5_index = list(df_labels_out_count.index)
    features_5_t = np.hstack((df_labels_out_count.values, df_labels_out_precent.values, df_labels_weight_count_out.values, df_labels_weight_percent_out.values))
    features_5 = np.zeros((num_nodes, features_5_t.shape[1]))
    for i, index in enumerate(features_5_index):
        features_5[index] = features_5_t[i]
    features_merge = np.concatenate([
        features_1,
        features_2,
        features_3,
        features_4,
        features_5
    ], axis=1)
    features_merge = np.unique(features_merge, axis=1)
    features_merge = np.delete(features_merge, np.argwhere(np.sum(features_merge, axis=0)==0), axis=1)

    return features_merge