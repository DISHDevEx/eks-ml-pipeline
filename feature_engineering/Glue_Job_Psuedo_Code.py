##Filter by cluster, node, pod, and container all of the indexes that have anomolous behavior in them. We will later on purge our datasets of the clusters that behave in these manners to train on clean data

filter_Condition_Cluster = np.where( df_cluster['cluster_failed_node_count']>0 ) 



filter_Condition_Node = np.where( 
                                    (df_node['node_memory_failcnt']>0) |
                                    (df_node['node_memory_hierarchical_pgfault']>0) |
                                    (df_node['node_memory_hierarchical_pgmajfault']>0)|
                                    (df_node['node_memory_pgfault']>0)|
                                    (df_node['node_memory_pgmajfault']>0)|
                                    (df_node['node_network_rx_dropped']>0)|
                                    (df_node['node_network_rx_errors']>0)|
                                    (df_node['node_network_tx_dropped']>0)|
                                    (df_node['node_network_tx_errors']>0)
                                   
                                    
                                )









filter_Condition_Pod = np.where( 
                                    (df_pod['pod_memory_failcnt']>0) |
                                    (df_pod['pod_memory_hierarchical_pgfault']>0) |
                                    (df_pod['pod_memory_hierarchical_pgmajfault']>0)|
                                    (df_pod['pod_memory_pgfault']>0)|
                                    (df_pod['pod_memory_pgmajfault']>0)|
                                    (df_pod['pod_network_rx_dropped']>0)|
                                    (df_pod['pod_network_rx_errors']>0)|
                                    (df_pod['pod_network_tx_dropped']>0)|
                                    (df_pod['pod_network_tx_errors']>0) |
                                    (df_pod['pod_status'] == 'Failed')
                                )




filter_Condition_Container = np.where( 
                                    (df_container['container_memory_failcnt']>0) |
                                    (df_container['container_memory_hierarchical_pgfault']>0) |
                                    (df_container['container_memory_hierarchical_pgmajfault']>0)|
                                    (df_container['container_memory_pgmajfault']>0)
                                    
                                )


unhealthy_clusternames_cluster = df_cluster.loc[filter_Condition_Cluster]['ClusterName'].unique().tolist()
unhealthy_clusternames_node = df_node.loc[filter_Condition_Node]['ClusterName'].unique().tolist()
unhealthy_clusternames_pod = df_pod.loc[filter_Condition_Pod]['ClusterName'].unique().tolist()
unhealthy_clusternames_container = df_container.loc[filter_Condition_Container]['ClusterName'].unique().tolist()



### FIND UNHEALTHY
all_unhealthy_cluster_names = list(set(unhealthy_clusternames_cluster+unhealthy_clusternames_node+unhealthy_clusternames_pod+unhealthy_clusternames_container))




# Create non anomalous data
df_node_non_anomalous = df_node[~df_node['ClusterName'].isin(all_unhealthy_cluster_names)]



##Feature set to be extracted: CPU(%), Memory(%), Total Network Bytes
Extract CPU, Memory, Network per node and save

for instance_id in df_node_non_anomalous:
    minmaxscaler(features)
    


