import argparse

import numpy as np

from mkl.acquisition import GreedyNRanking
from mkl.data import DataManager, Hdf5Dataset
from mkl.dense import DenseGaussianProcessregressor
    
# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', type=int, help='num concurrent runs to perform')
    parser.add_argument('-i', action='store', type=int, help='total iterations to run including random sample at start')
    args = parser.parse_args()

    N_CONCURRENT = args.n
    NUM_ITERATIONS = args.i


    # set up core objects
    X_ref = Hdf5Dataset('All_mof_descriptors_pre_transformed.csv')
    manager = DataManager(N_CONCURRENT)
    models = [DenseGaussianProcessregressor(data_set=X_ref) for _ in range(N_CONCURRENT)]
    acquisition = GreedyNRanking()


    #logging.info('taking inital random sample')
    initial_points = list(np.random.choice(len(X_ref), N_CONCURRENT, replace=False))
    results = simulate_mofs(initial_points)


    #logging.info('updating reference values for `y`')
    for k, x, y in results:
        manager.add_entry(k, (x, y))
        
        
    # perform iterations
    for _ in range(NUM_ITERATIONS - 1):  # total num not including random sampled    
        queued = []
        sampled = manager.get_all_sampled()

        for k in range(N_CONCURRENT):
            Xk, yk = manager.get_X_y(k)
            models[k] = models[k].fit(Xk, yk)  # 1. fit models to data collected so far
            posterior = models[k].sample_y()  # 2. use model to make evaluation for ALL data points
            ranked = acquisition(posterior)[::-1]  # want index of best ranking to be at start of list
                    
            # 3. for each model, choose the highest ranked AVAILABLE mof
            for rk in ranked:
                if rk not in sampled and rk not in queued:  # not already sampled or queued to be run
                    queued.append(rk)
                    break  # stop at this point, only want to queue N_CONCURRENT things
                
        # 4. run the simulations concurrently  
        results = simulate_mofs(queued)
        
        # 5. take the collected results and update data manager with them
        for k, x, y in results:
            manager.add_entry(k, (x, y))
            
        # 6. write status of each concurrent screening so far
        for k in range(N_CONCURRENT):
            manager.write_to_file(k)
        
        # 7. repeat the loop
        
        
    # a = perf_counter()
    
    # raspa = RaspaRegistry('cif_list.txt', simulation_dir='raspa_dir')
    
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(raspa.run_simulation, idx) for idx in range(len(raspa))]
        
    # output = [a.result() for a in futures]
        
    # for o in output:
    #     print(o)
    
    # b = perf_counter()
    # print(b-a)
    