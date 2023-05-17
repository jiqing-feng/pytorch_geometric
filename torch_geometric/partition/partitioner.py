import json
import os
from typing import List, Optional, Union

import torch

from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.local_feature_store import LocalFeatureStore
from torch_geometric.data.local_graph_store import LocalGraphStore
from torch_geometric.loader import ClusterData
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType, as_str


def prepare_dir(root_path: str, child_path: Optional[str] = None):
    dir_path = root_path
    if child_path is not None:
        dir_path = os.path.join(root_path, child_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def record_meta_info(
    output_dir: str,
    num_parts: int,
    is_hetero: bool = False,
    node_types: Optional[List[NodeType]] = None,
    edge_types: Optional[List[EdgeType]] = None,
):
    r""" Save partitioning meta info into the output directory.
    """
    meta = {
        'num_parts': num_parts,
        'hetero_graph': is_hetero,
        'node_types': node_types,
        'edge_types': edge_types
    }
    meta_file = os.path.join(output_dir, 'META.json')
    with open(meta_file, "w") as outfile:
        json.dump(meta, outfile, sort_keys=True, indent=4)


def record_node_mapping(output_dir: str, node_map: torch.Tensor,
                        ntype: Optional[NodeType] = None):
    r""" Save a partition book of graph nodes into the output directory.
    """
    if ntype is not None:
        subdir = prepare_dir(output_dir, 'node_map')
        fpath = os.path.join(subdir, f'{as_str(ntype)}.pt')
    else:
        fpath = os.path.join(output_dir, 'node_map.pt')
    torch.save(node_map, fpath)


def record_edge_mapping(output_dir: str, edge_map: torch.Tensor,
                        etype: Optional[EdgeType] = None):
    r""" Save a partition book of graph edges into the output directory.
    """
    if etype is not None:
        subdir = prepare_dir(output_dir, 'edge_map')
        fpath = os.path.join(subdir, f'{EdgeTypeStr(etype)}.pt')
    else:
        fpath = os.path.join(output_dir, 'edge_map.pt')
    torch.save(edge_map, fpath)


class Partitioner():
    r""" Base class for partitioning graphs and features.
    """
    def __init__(self, output_dir: str, num_parts: int,
                 data: Union[Data, HeteroData],
                 device: torch.device = torch.device('cpu')):

        self.output_dir = prepare_dir(output_dir)

        self.num_parts = num_parts
        assert self.num_parts > 1
        self.data = data
        self.is_hetero = False
        if isinstance(data, HeteroData):
            self.is_hetero = True
            self.node_types = data.node_types
            self.edge_types = data.edge_types
        else:
            self.node_types = None
            self.edge_types = None
        self.device = device

    def generate_partition(self):
        r""" Partition graph and feature data into different parts.
    """
        # save meta info for partition.
        print("save metadata for partition info")
        record_meta_info(self.output_dir, self.num_parts, self.is_hetero,
                         self.node_types, self.edge_types)

        input_data = self.data
        if self.is_hetero:
            input_data = self.data.to_homogeneous()
        cluster_data = ClusterData(input_data, num_parts=self.num_parts,
                                   log=True, keep_inter_cluster_edges=True)
        perm = cluster_data.perm
        partptr = cluster_data.partptr
        node_partition_mapping = torch.arange(input_data.num_nodes,
                                              dtype=torch.int64)
        edge_partition_mapping = torch.arange(input_data.num_edges,
                                              dtype=torch.int64)

        if self.is_hetero:
            edge_type_num = len(input_data._edge_type_names)
            node_type_num = len(input_data._node_type_names)

            for pid in range(self.num_parts):
                # save graph partition
                print(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = partptr[pid]
                end_pos = partptr[pid + 1]
                part_edge_ids = cluster_data[pid].eid
                edge_type = cluster_data[pid].edge_type
                node_type = cluster_data[pid].node_type
                graph_store = LocalGraphStore()
                edge_feature_store = LocalFeatureStore()
                for etype_id in range(edge_type_num):
                    edge_name = input_data._edge_type_names[etype_id]
                    mask = (edge_type == etype_id)
                    local_row_ids = torch.masked_select(edge_index[0], mask)
                    local_col_ids = torch.masked_select(edge_index[1], mask)
                    global_row_ids = perm[local_row_ids + start_pos]
                    global_col_ids = perm[local_col_ids]
                    type_edge_ids = torch.masked_select(part_edge_ids, mask)
                    edge_partition_mapping[type_edge_ids] = pid

                    node_num = global_row_ids.shape[0]
                    graph_store.put_edge_index(
                        edge_index=(global_row_ids, global_col_ids),
                        edge_type=edge_name, layout='coo',
                        size=(node_num, node_num))
                    # save edge feature partition
                    if cluster_data[pid].edge_attr is not None:
                        print(f"save edge feature for edge type: {edge_name}")
                        type_edge_feat = cluster_data[pid].edge_attr[mask, :]
                        edge_feature_store.put_tensor(type_edge_feat,
                                                      group_name=f'part_{pid}',
                                                      attr_name=edge_name,
                                                      index=None)
                        edge_feature_store.set_global_ids(
                            type_edge_ids, group_name=f'part_{pid}',
                            attr_name=edge_name)

                subdir = prepare_dir(self.output_dir, f'part_{pid}')
                torch.save(graph_store, os.path.join(subdir, 'graph.pt'))
                if len(edge_feature_store.get_all_tensor_attrs()) > 0:
                    torch.save(edge_feature_store,
                               os.path.join(subdir, 'edge_feats.pt'))

                # save node feature partition
                print(f"save node feature for part: {pid}")
                node_ids = perm[start_pos:end_pos]
                node_partition_mapping[node_ids] = pid
                if cluster_data[pid].x is not None:
                    offset = 0
                    node_feature_store = LocalFeatureStore()
                    for ntype_id in range(node_type_num):
                        node_name = input_data._node_type_names[ntype_id]
                        mask = (node_type == ntype_id)
                        type_node_id = torch.masked_select(node_ids, mask)
                        type_node_id = type_node_id - offset
                        offset = offset + self.data.num_nodes_dict[node_name]
                        type_node_feat = cluster_data[pid].x[mask, :]
                        node_feature_store.put_tensor(type_node_feat,
                                                      group_name=f'part_{pid}',
                                                      attr_name=node_name)
                        node_feature_store.set_global_ids(
                            type_node_id, group_name=f'part_{pid}',
                            attr_name=node_name, index=None)
                    torch.save(node_feature_store,
                               os.path.join(subdir, 'node_feats.pt'))

            # save node partition book
            print("save node partition book")
            for ntype_id in range(node_type_num):
                node_name = input_data._node_type_names[ntype_id]
                mask = (input_data.node_type == ntype_id)
                type_node_map = torch.masked_select(node_partition_mapping,
                                                    mask)
                record_node_mapping(self.output_dir, type_node_map, node_name)

            # save edge partition book
            print("save edge partition book")
            for etype_id in range(edge_type_num):
                edge_name = input_data._edge_type_names[etype_id]
                mask = (input_data.edge_type == etype_id)
                type_edge_map = torch.masked_select(edge_partition_mapping,
                                                    mask)
                record_edge_mapping(self.output_dir, type_edge_map, edge_name)

        else:  # homo graph
            for pid in range(self.num_parts):
                # save graph partition
                print(f"save graph partition for part: {pid}")
                edge_index = cluster_data[pid].edge_index
                start_pos = partptr[pid]
                end_pos = partptr[pid + 1]
                local_row_ids = edge_index[0]
                local_col_ids = edge_index[1]
                global_row_ids = perm[local_row_ids + start_pos]
                global_col_ids = perm[local_col_ids]
                edge_ids = cluster_data[pid].eid
                graph_store = LocalGraphStore()
                node_num = global_row_ids.shape[0]
                graph_store.put_edge_index(
                    edge_index=(global_row_ids, global_col_ids),
                    edge_type=None, layout='coo', size=(node_num, node_num))
                subdir = prepare_dir(self.output_dir, f'part_{pid}')
                torch.save(graph_store, os.path.join(subdir, 'graph.pt'))

                edge_partition_mapping[edge_ids] = pid
                # save edge feature partition
                if cluster_data[pid].edge_attr is not None:
                    print(f"save edge feature for part: {pid}")
                    edge_feature_store = LocalFeatureStore()
                    edge_feature_store.put_tensor(cluster_data[pid].edge_attr,
                                                  group_name=f'part_{pid}',
                                                  attr_name=None, index=None)
                    edge_feature_store.set_global_ids(edge_ids,
                                                      group_name=f'part_{pid}',
                                                      attr_name=None)
                    torch.save(edge_feature_store,
                               os.path.join(subdir, 'edge_feats.pt'))

                # save node feature partition
                print(f"save node feature for part: {pid}")
                node_ids = perm[start_pos:end_pos]
                if cluster_data[pid].x is not None:
                    node_feature_store = LocalFeatureStore()
                    node_feature_store.put_tensor(cluster_data[pid].x,
                                                  group_name=f'part_{pid}',
                                                  attr_name=None, index=None)
                    node_feature_store.set_global_ids(node_ids,
                                                      group_name=f'part_{pid}',
                                                      attr_name=None)
                    torch.save(node_feature_store,
                               os.path.join(subdir, 'node_feats.pt'))

                node_partition_mapping[node_ids] = pid

            # save node/edge partition mapping info
            print("save partition mapping info for nodes/edges")
            record_node_mapping(self.output_dir, node_partition_mapping,
                                self.node_types)
            record_edge_mapping(self.output_dir, edge_partition_mapping,
                                self.edge_types)
