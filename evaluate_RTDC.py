import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from io_utils import parse_args
from tqdm import tqdm

# --- estimating few-shot class covariance
def cov_estimate(sigma, in_sigma, Sc, base_cov, lamb = 0.1, gamma = 0.05):
    n_way = len(Sc)                                                       
    n_shot = len(Sc[0])
    if n_shot == 1: 
        if len(sigma):
            return sigma, in_sigma
        else:
            sigma = lamb*base_cov + gamma*np.eye(base_cov.shape[0])
            in_sigma = np.linalg.inv(sigma) 
            return [sigma for i in range(n_way)],[in_sigma for i in range(n_way)]
    else:         
        sigma = []
        in_sigma = []
        for i in range(n_way):
            cov_support = np.cov(Sc[i].T)
            sigma.append((1-lamb)*cov_support + lamb*base_cov + gamma*np.eye(base_cov.shape[0]))
            in_sigma.append(np.linalg.inv(sigma[i]))
        return sigma, in_sigma

if __name__ == '__main__':       
    
    params = parse_args('test')
    n_ways = 5
    n_shot = 1
    n_queries = 15 
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    
    num_sampled = 150
    topk = 15
    lamb = 1.0
    gamma = 0.05
    
	
    # ---- data loading and fsl tasks generating
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries, 'method': params.method}
    FSLTask.loadDataSet(params.dataset, params.method)
    FSLTask.setRandomStates(cfg)       
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)    
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    beta = 0.5
    base_cov = []
    base_features_path = "./checkpoints/%s/%s/last/base_features.plk"%(params.dataset, params.method)
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            feature = np.power(feature[:, ] ,beta)      
            cov = np.cov(feature.T)                    
            base_cov.append(cov)
            
    base_cov = np.array(base_cov)/(len(base_cov))       
    base_cov = np.sum(base_cov,axis = 0)
    
    # ---- classification for each task
    print('Start classification for %d tasks...'%(n_runs))
    
    sigma = []
    in_sigma = []
    acc_list = []
    for it in tqdm(range(n_runs)):
        
        support_data = ndatas[it][:n_lsamples].numpy()   
        support_label = labels[it][:n_lsamples].numpy()  
        query_data = ndatas[it][n_lsamples:].numpy()     
        query_label = labels[it][n_lsamples:].numpy()    
        

        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        
        Sc = [support_data[j::n_ways] for j in range(n_ways)]       
        
        # --- distribution estimation
        prototype = [np.mean(Sc[j], axis=0) for j in range(n_ways)]        
        sigma, in_sigma = cov_estimate(sigma, in_sigma, Sc, base_cov, lamb = lamb, gamma = gamma)
        
        # --- relational matrix
        R = []
        for i in range(n_usamples): 
            dist = []
            for j in range(n_ways): 
                mx = query_data[i]-prototype[j]
                d = np.sqrt(np.dot(np.dot(mx,in_sigma[j]),mx.T)) 
                dist.append(-d)      
            dist = np.exp(dist)                       
            d_sum = np.sum(dist)
            R.append([dist[j]/d_sum for j in range(n_ways)])
        B = np.array(R).T                                    

        # --- topk similar query samples for each class   
        K = np.argpartition(B, -topk)[:,-topk:] 
        topk_query_samples = [[query_data[j] for j in K[i]] for i in range(n_ways)] 
        X = [np.concatenate([Sc[i], topk_query_samples[i]]) for i in range(n_ways)] 
        

        # --- distribution calibration and generating samples      
        sampled_data = []
        sampled_label = []

        for i in range(n_ways):
            mean = np.mean(X[i], axis=0)  
            sampled_data.append(np.random.multivariate_normal(mean=mean, cov=sigma[i], size=num_sampled))  
            sampled_label.extend([support_label[i]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * num_sampled, -1)
            
        # ---- train classifier
        classifier = LogisticRegression(max_iter=2000).fit(X=np.concatenate([support_data, sampled_data]), y=np.concatenate([support_label, sampled_label]))
        
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc*100)
        
    sqr_n = np.sqrt(it+1)
    print(it+1, params.method, '%s %d way %d shot  ACC: %4.2f%% +- %4.2f%%'%(params.dataset, n_ways, n_shot,
                      float(np.mean(acc_list)),1.96*np.std(acc_list)/sqr_n))
