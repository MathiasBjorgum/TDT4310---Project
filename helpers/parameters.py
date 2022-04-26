knn_params = [{'n_neighbors': [3, 5, 7],
         'weights': ['uniform', 'distance'],
         "leaf_size": [20,30,40,50]
         }]

svc_params = [{"kernel": ["rbf", "poly", "sigmoid"],
                "C": [1, 1.5, 2]}]

dt_params = [{"criterion": ["gini", "entropy"], 
                "ccp_alpha": [0, 0.1, 0.5]}]

rf_params = [{"n_estimators": [80, 100, 120], 
                "criterion":["gini", "entropy"]}]

nb_params = [{"alpha": [0.5,1,2]}]