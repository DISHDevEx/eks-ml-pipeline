

##ingest data
(1) Read in data

##init an empty structure to hold instance
(2) instance_dfs = []


##get all usable nodeIDs
(3) iterate through all nodeID Dataframes:
    
    drop nulls['node_cpu_utilization','node_memory_utilization','node_network_total_bytes'] in nodeID Dataframe
    
    
    if nodeID Dataframe has: 
        a) Number of timestamps >= 60
        c) Max time delta <= 75 seconds
        d) Min time delta >= 45 seconds
        
        
        Then: instance_dfs.append(nodeID Dataframe)

            
(4) #Create trianing samples that LSTM can take in with shape: [Numofsamples,Timesteps,NumFeatures]

##global variable
time_steps = 12
batch_size = 6
n_samples = batch_size*1000000000 ##made up number ideally it should be a very large number of samples


features = ['node_cpu_utilization','node_memory_utilization','node_network_total_bytes']

x_train = np.zeros((n_samples,time_steps,len(features)))
for b in range(n_samples):
    
    ##pick random df, and normalize
    df = random.choice(instance_dfs)
    df = df.drop(columns = ['InstanceId'])
    df = df.set_index('Timestamp')
    df = df.sort_index()
    df[features] = scaler.fit_transform(df[features])
    
    
    
    sample = np.zeros((n_samples,len(features)))
    ##make sure length of df is atleast 40
    first_time = random.choice(range(len(df)-time_steps))
    df.head()
    sample = df[features].iloc[first_time:first_time+time_steps]
    x_train[b] = sample


(5) Save x_train in s3   
s3.save(x_train)