# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:29:52 2019

@author: Lambert Rosique
"""
import json
import lightgbm as lgb
import fraud_utils as fu
import requests
import time

''' Modèle LGB '''
def train_model_lgb():
    global X_train, X_test, y_train, y_test
    # Découpage des données en train/test sets
    X_train, X_test, y_train, y_test = fu.split_train_test(fu.data_creditcard.drop('Class', axis=1), fu.data_creditcard['Class'], test_size=0.2)
    
    # Datasets
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    #lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
    
    # Parameters
    parameters = {'num_leaves': 2**8,
                  'learning_rate': 0.1,
                  'is_unbalance': True,
                  'min_split_gain': 0.1,
                  'min_child_weight': 1,
                  'reg_lambda': 1,
                  'subsample': 1,
                  'objective':'binary',
                  #'device': 'gpu', #comment if you're not using GPU
                  'task': 'train'
                  }
    num_rounds = 300
    
    # Training
    clf = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)
    
    # Affichage de quelques métriques pour évaluer notre modèle
    y_prob = clf.predict(X_test)
    y_pred = fu.binarize_prediction(y_prob, threshold=0.5)
    metrics = fu.classification_metrics_binary(y_test, y_pred)
    metrics2 = fu.classification_metrics_binary_prob(y_test, y_prob)
    metrics.update(metrics2)
    cm = metrics['Confusion Matrix']
    metrics.pop('Confusion Matrix', None)
    
    print(json.dumps(metrics, indent=4, sort_keys=True))
    fu.plot_confusion_matrix(cm, ['no fraud (negative class)', 'fraud (positive class)'])
    
    # Sauvegarde du modèle entraîné
    clf.save_model(fu.BASELINE_MODEL)

train_model_lgb()

''' Serveur temps-réel '''
### IMPORTANT ###
# Démarrer le serveur grâce à la commande
# python ./api.py

def add_points_map(nb):
    max_try=120
    cpt = 0
    while cpt < max_try and not fu.test_server_online():
        cpt += 1
        print("Server is not ready ("+str(cpt)+")")
        time.sleep(1)
    if cpt == max_try:
        raise Exception("Max attempts reached")
    print("Server ready for real-time fraud detection !")
    print("You can now go to "+fu.URL_API + '/map')
    for i in range(nb):
        # Récupération d'une fraude
        vals = y_test[y_test == 1].index.values
        X_target = X_test.loc[vals[0]]
        dict_query = X_target.to_dict()
        # Affichage sur la carte via un appel au webservice
        headers = {'Content-type':'application/json'}
        end_point_map = fu.URL_API + '/predict_map'
        requests.post(end_point_map, data=json.dumps(dict_query), headers=headers)
    
        fu.wait_random_time(0.5,5)

add_points_map(25)