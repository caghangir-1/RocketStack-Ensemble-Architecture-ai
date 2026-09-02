''' ==============================================================================
# Copyright (c) 2024-2026. All rights reserved.
#
# Project: RocketStack: Level-aware deep recursive ensemble learning architecture
# Author: Çağatay Demirel

# This source code is licensed under the GNU General Public License v3.0 (GPLv3).
============================================================================== '''

# ========================= Libraries =========================

#======= Sklearn ============
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
                            log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

import lightgbm as lgb
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.exceptions import ConvergenceWarning

#==== Dimension Reduction Functions ========
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
#==== Dimension Reduction Functions ========

#======= Sklearn ============

#====== Other AI ========
from xgboost import XGBClassifier
import catboost as ctb
#====== Other AI ========

#====== Others =======
import numpy as np
import time
import optuna

import math
import warnings
from numpy.random import seed
from numpy.random import randint
#====== Others =======

# =========== Tensorflow ===========
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply
# =========== Tensorflow ===========

# ========================= Libraries =========================

class RocketStack():

    def SFE_fit(self, xtrain, ytrain, kk):
         k = 5
         if len(kk) == 0:
             seed(1)
             kk= randint(1, np.size(xtrain, 1))
        #print(sf)
         sf= []
         pos=[]
         for i in range(0, np.size(xtrain, 1)):
             if(kk[i]==1):
                 sf.append(i) 

        
         pos=np.transpose(sf)

         model  = KNeighborsClassifier(n_neighbors =1,metric='euclidean') 
         X=xtrain[:,pos]
         scores = cross_val_score(model, X, ytrain, cv=5)

         
         cost = sum(scores)/k

         
         return cost*100
     
    #% =============== SFE Run ====================
    def SFE_run(self, Input, Target, UR, UR_Max, UR_Min, Max_FEs, Max_Run, Run, Cost):
        
        np.random.seed(31) #to initialize weights same every time
        
        X_New = None
        while (Run <= Max_Run):
            
            EFs = 1
            
            X = np.random.randint(0, 2, np.size(Input, 1))   # Initialize an Individual X
            Fit_X = self.SFE_fit(Input, Target, X)                    # Calculate the Fitness of X
            Nvar = np.size(Input, 1)                         # Number of Features in Dataset
        
        
            while (EFs <= Max_FEs):
        
                X_New = np.copy(X)
                # Non-selection operation:
        
                U_Index = np.where(X == 1)                      # Find Selected Features in X
                NUSF_X = np.size(U_Index, 1)                    # Number of Selected Features in X
                UN = math.ceil(UR*Nvar)                         # The Number of Features to Unselect: Eq(2)
                # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
                # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
                K1 = np.random.randint(0, NUSF_X, UN)           # Generate UN random number between 1 to the number of slected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = U_Index[0][[res1]]                          # K=index(U)
                X_New[K] = 0                                    # Set X_New (K)=0 
        
        
               # Selection operation:
                if np.sum(X_New) == 0:
                    S_Index = np.where(X == 0)                  # Find non-selected Features in X
                    NSF_X = np.size(S_Index, 1)                 # Number of non-selected Features in X
                    SN = 1                                      # The Number of Features to Select
                    K1 = np.random.randint(0, NSF_X, SN)        # Generate SN random number between 1 to the number of non-selected features in X
                    res = np.array([*set(K1)])
                    res1 = np.array(res)
                    K = S_Index[0][[res1]]
                    X_New = np.copy(X)
                    X_New[K] = 1                                # Set X_New (K)=1
        
        
                Fit_X_New = self.SFE_fit(Input,Target,X_New)             # Calculate the Fitness of X_New
        
                if Fit_X_New > Fit_X:
                    X = np.copy(X_New)
                    Fit_X = Fit_X_New
        
                UR = (UR_Max-UR_Min)*((Max_FEs-EFs)/Max_FEs)+UR_Min  # Eq(3)
                Cost[EFs-1,Run-1]=Fit_X
                # print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, Fit_X, np.sum(X), Run))
                EFs = EFs+1
        
            Run = Run+1
        cost1=Cost[-1,:]
        
        print('\n')
        
        print('*************************SFE Algorithm**********************************')
        
        print('\n Best = {} '.format(max(cost1)))
        print('\n Mean = {} '.format(cost1.mean(0)))
        print('\n Worst = {}  '.format(min(cost1)))
        print('\n std  = {}'.format(np.std(cost1)))
        
        selected_features = np.where(X_New == 1)[0]
        
        return selected_features
    #% =============== SFE Run ==================== 
    
    def reduce_dimensionality_with_autoencoder(self, train_X, test_X, encoding_dim, epochs=50, batch_size=32, validation_split=0.2):
        
        """
        Reduces the dimensionality of the train and test data using an autoencoder.
        
        Parameters:
        train_X (np.ndarray): Training data.
        test_X (np.ndarray): Test data.
        encoding_dim (int): The dimension to reduce the data to.
        epochs (int): Number of training epochs for the autoencoder.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of the training data to use for validation.
        
        Returns:
        np.ndarray: Encoded training features.
        np.ndarray: Encoded test features.
        """
        
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)
        
        input_dim = train_X_scaled.shape[1]
        
        # Define the autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)
        
        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train the autoencoder
        autoencoder.fit(train_X_scaled, train_X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split, verbose=0)
        
        # Encode the train and test data using the encoder part of the autoencoder
        encoded_train_features = encoder.predict(train_X_scaled)
        encoded_test_features = encoder.predict(test_X_scaled)
        
        return encoded_train_features, encoded_test_features
     
    def reduce_dimensionality_with_autoencoder2(self, train_X, test_X, epochs=50, batch_size=32, validation_split=0.2):
  
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)
        
        input_dim = train_X_scaled.shape[1]
    
        # Define the autoencoder model with 2 encoding and 2 decoding layers
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(input_dim // 2, activation='relu')(input_layer)
        encoded = Dense(input_dim // 3, activation='relu')(encoded)
        
        decoded = Dense(input_dim // 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)
    
        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')
    
        # Train the autoencoder
        autoencoder.fit(train_X_scaled, train_X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split, verbose=0)
    
        # Encode the train and test data using the encoder part of the autoencoder
        encoded_train_features = encoder.predict(train_X_scaled)
        encoded_test_features = encoder.predict(test_X_scaled)
        
        return encoded_train_features, encoded_test_features
    
    def attention_feature_selector_mc(self, train_X, train_Y, test_X, percentile=25, epochs=10, batch_size=32, verbose=1):
        
        """
        Attention-based feature selection using a fixed percentile threshold.
    
        Parameters:
            train_X (ndarray): Training feature set of shape (n_samples, n_features).
            train_Y (ndarray): Training labels of shape (n_samples,).
            percentile (int): Percentile of features to retain (e.g., 25 for top 25%).
            epochs (int): Number of training epochs for the attention model.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity level for training (0 = silent, 1 = progress bar).
    
        Returns:
            selected_feature_indices (ndarray): Indices of selected features.
        """
        
        # Get the number of features
        input_dim = train_X.shape[1]
        
        num_classes = len(np.unique(train_Y))
        
        # Build and train the attention model
        attention_model = self.build_attention_model_multiclass(input_dim, num_classes)
        attention_model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=verbose)
     
        # Extract attention weights
        attention_layer_model = Model(inputs=attention_model.input, outputs=attention_model.get_layer("attention_weights").output)
        attention_weights = attention_layer_model.predict(train_X)
     
        # Percentile-Based Feature Selection
        mean_attention_weights = attention_weights.mean(axis=0)
        threshold = np.percentile(mean_attention_weights, 100 - percentile)
        selected_feature_indices = np.where(mean_attention_weights > threshold)[0]
     
        print(f"Initial Feature Count: {input_dim}")
        print(f"Selected Feature Count: {len(selected_feature_indices)}")
        
        train_X_selected = train_X[:, selected_feature_indices]
        test_X_selected = test_X[:, selected_feature_indices]
        
        return train_X_selected, test_X_selected
    
    def build_attention_model_multiclass(self, input_dim, num_classes):
         
        input_layer = Input(shape=(input_dim,))
        
        # Attention mechanism
        attention_weights = Dense(input_dim, activation="softmax", name="attention_weights")(input_layer)
        weighted_features = Multiply(name="weighted_features")([input_layer, attention_weights])
        
        # Multi-class classification output
        output_layer = Dense(num_classes, activation="softmax")(weighted_features)
        
        # Compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        return model

    def AscentTheRocket_binary_massive_exploration(self, train_X, train_Y, hyperparameterOptimization=False, num_of_level=10, iffeatselection=True, n_trials=50,
                                                                       level369=True, indices_to_normalized=None, specific_folds=None, blur_strength='light', left_model_threshold=2):
                
        ''' ============== Blending protocol =============
        1) original: (if no intermeditate feature selection)
        2) cumulative: (if intermediate feature selection)
         ============== Blending protocol ============= '''
         
        ''' Indexing the model order: 
            
        ======= Stand-alone models ======= (standalone_preds: [19])
        1) xgb --> Bayesopt possibility
        2) lgbm --> Bayesopt possibility
        3) randomforest --> Bayesopt possibility
        4) ctb_clf --> Bayesopt possibility
        5) sgd
        6) svc
        7) bagging
        8) adaboost
        9) knn
        10) LDA
        11) gpc
        
        12) extratrees
        13) gradient boosting
        14) calibrated ridge classifier
        15) logistic regression
        16) calibrated passive aggressive
        17) benoulli
        18) gaussian
        20) MLP
        21) histgradient boosting
        ======= Stand-alone models =======
        '''

        # Clean up logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", message="[LightGBM]")
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        
        np.random.seed(31) #to initialize weights same every time
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #this cv will be used eternally in this very function
    
        ''' ================= Metric initialization hub ================ '''
        num_of_folds = 5
        num_of_models = 20
        # num_of_classes = 2
        
        ''' ======== Metric initialization individual ======== '''
        accuracies_ind = np.zeros((num_of_models, num_of_folds))
        f1scores_ind = np.zeros((num_of_models, num_of_folds))
        precisions_ind = np.zeros((num_of_models, num_of_folds))
        recalls_ind = np.zeros((num_of_models, num_of_folds))
        log_lossess_ind = np.zeros((num_of_models, num_of_folds))
        roc_auc_score_ind = np.zeros((num_of_models, num_of_folds))
        elapsedtimes_individual = np.zeros(num_of_models)
        ''' ======== Metric initialization individual ======== '''
        
        ''' ======== Metric initialization blend ensemble (L1 -- L10) ======== '''   
        accuracies_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        f1scores_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        precisions_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        recalls_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        log_lossess_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        roc_auc_score_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))

        elapsedtimes_blendensemble_levels = np.zeros((num_of_folds, num_of_level, num_of_models))
        elapsedtimes_blendensemble_until_meta_learning = np.zeros((num_of_folds, num_of_level))
        elapsedtimes_blendensemble_just_meta_learning = np.zeros((num_of_folds, num_of_level, num_of_models))
        
        number_of_feautures_blendensemble_levels = np.zeros((num_of_folds, num_of_level))
        ''' ======== Metric initialization blend ensemble (L1 -- L10) ======== '''
        
        ''' ======= Metric initialization stack-of-stacking ======== (Features + L1....L10) '''
        accuracies_stackofstack = np.zeros((num_of_folds, num_of_models))
        f1scores_stackofstack = np.zeros((num_of_folds, num_of_models))
        precisions_stackofstack = np.zeros((num_of_folds, num_of_models))
        recalls_stackofstack = np.zeros((num_of_folds, num_of_models))
        log_lossess_stackofstack = np.zeros((num_of_folds, num_of_models))
        roc_auc_score_stackofstack = np.zeros((num_of_folds, num_of_models))
        
        elapsedtimes_stackofstacking = np.zeros((num_of_folds, num_of_models))
        ''' ======= Metric initialization stack-of-stacking ======== (Features + L1....L10) '''
         
        ''' ======= INTERNAL OPTIMIZATION FUNCTION (Nested CV) ======= '''
        def get_optimized_model_for_fold(model_name, X_train, y_train):
            """ Runs Optuna on the fold's training data to find best params. """
            
            def objective(trial):
                # --- Group 1: Heavy Hitters ---
                if model_name == 'xgb':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'n_jobs': -1, 'random_state': 42, 'verbosity': 0
                    }
                    model = XGBClassifier(**params)
                elif model_name == 'lgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'n_jobs': -1, 'random_state': 42, 'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params)
                elif model_name == 'ctbclf':
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 500),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                        'thread_count': -1, 'random_seed': 42, 'verbose': False, 'allow_writing_files': False
                    }
                    model = ctb.CatBoostClassifier(**params)
                elif model_name == 'randforest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'n_jobs': -1, 'random_state': 42
                    }
                    model = RandomForestClassifier(**params)
                # --- Group 2: Linear / SVM ---
                elif model_name == 'svc':
                    params = {
                        'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
                        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                        'gamma': 'scale',  # Explicitly set gamma to scale for stability
                        'probability': True, 
                        'random_state': 42,
                        'max_iter': 2000,   # HARD LIMIT: Stops infinite loops on hard margins
                        'cache_size': 2000  # Give it 2GB RAM to speed up matrix calcs
                    }
                    model = SVC(**params)
                elif model_name == 'sgd':
                    params = {
                        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                        'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
                        'loss': 'log_loss', 'max_iter': 1000, 'n_jobs': -1, 'random_state': 42
                    }
                    model = SGDClassifier(**params)
                # --- Group 3: Others ---
                elif model_name == 'knn':
                    n_neighbors = trial.suggest_int('n_neighbors', 3, 50)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                elif model_name == 'mlp':
                    params = {
                        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                        'max_iter': 500, 'random_state': 42
                    }
                    model = MLPClassifier(**params)
                # --- Fallbacks for simple models (still tuned lightly) ---
                elif model_name == 'bagging':
                    model = BaggingClassifier(n_estimators=trial.suggest_int('n', 10, 50), n_jobs=-1, random_state=42)
                elif model_name == 'adaboost':
                    model = AdaBoostClassifier(n_estimators=trial.suggest_int('n', 50, 150), algorithm='SAMME', random_state=42)
                elif model_name == 'lda':
                    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') # Simple tune
                elif model_name == 'gpc':
                    model = GaussianProcessClassifier(random_state=42, n_jobs=-1, max_iter_predict=20)
                elif model_name == 'extra_trees':
                    model = ExtraTreesClassifier(n_estimators=trial.suggest_int('n', 100, 300), n_jobs=-1, random_state=42)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingClassifier(n_estimators=trial.suggest_int('n', 100, 300), random_state=42)
                elif model_name == 'hist_gradient_boosting':
                    model = HistGradientBoostingClassifier(max_iter=trial.suggest_int('n', 100, 300), random_state=42)
                # --- Wrappers ---
                elif model_name == 'calibrated_ridge':
                    alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
                    model = CalibratedClassifierCV(RidgeClassifier(alpha=alpha), cv=3)
                elif model_name == 'logistic_regression':
                     model = LogisticRegression(C=trial.suggest_float('C', 0.1, 10.0, log=True), max_iter=1000, n_jobs=-1)
                elif model_name == 'calibrated_passive_aggressive':
                    C = trial.suggest_float('C', 0.01, 10.0, log=True)
                    model = CalibratedClassifierCV(PassiveAggressiveClassifier(C=C, max_iter=1000), cv=3)
                elif model_name == 'bernoulli_nb':
                    model = BernoulliNB(alpha=trial.suggest_float('alpha', 0.1, 2.0))
                elif model_name == 'gaussian_nb':
                    model = GaussianNB(var_smoothing=trial.suggest_float('var', 1e-9, 1e-5, log=True))
                else: 
                    return 0
                
                # Fast internal CV (3-fold) for HPO
                try:
                    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=1).mean()
                except:
                    return 0.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials) # 50 trials per fold to keep runtime managed
            best = study.best_params
            
            # Return fresh model with best params
            if model_name == 'xgb': return XGBClassifier(**best, n_jobs=-1, random_state=42, verbosity=0)
            if model_name == 'lgbm': return lgb.LGBMClassifier(**best, n_jobs=-1, verbose=-1, random_state=42)
            if model_name == 'ctbclf': return ctb.CatBoostClassifier(**best, thread_count=-1, verbose=False, allow_writing_files=False, random_seed=42)
            if model_name == 'randforest': return RandomForestClassifier(**best, n_jobs=-1, random_state=42)
            if model_name == 'svc': return SVC(**best, probability=True, random_state=42)
            if model_name == 'sgd': return SGDClassifier(**best, loss='log_loss', max_iter=1000, n_jobs=-1, random_state=42)
            if model_name == 'knn': return KNeighborsClassifier(**best, n_jobs=-1)
            if model_name == 'mlp': return MLPClassifier(**best, max_iter=500, random_state=42)
            if model_name == 'bagging': return BaggingClassifier(n_estimators=best['n'], n_jobs=-1, random_state=42)
            if model_name == 'adaboost': return AdaBoostClassifier(n_estimators=best['n'], algorithm='SAMME', random_state=42)
            if model_name == 'extra_trees': return ExtraTreesClassifier(n_estimators=best['n'], n_jobs=-1, random_state=42)
            if model_name == 'gradient_boosting': return GradientBoostingClassifier(n_estimators=best['n'], random_state=42)
            if model_name == 'hist_gradient_boosting': return HistGradientBoostingClassifier(max_iter=best['n'], random_state=42)
            if model_name == 'calibrated_ridge': return CalibratedClassifierCV(RidgeClassifier(alpha=best['alpha']), cv=5)
            if model_name == 'logistic_regression': return LogisticRegression(C=best['C'], max_iter=1000, n_jobs=-1)
            if model_name == 'calibrated_passive_aggressive': return CalibratedClassifierCV(PassiveAggressiveClassifier(C=best['C'], max_iter=1000), cv=5)
            if model_name == 'bernoulli_nb': return BernoulliNB(alpha=best['alpha'])
            if model_name == 'gaussian_nb': return GaussianNB(var_smoothing=best['var'])
            # Fallbacks
            if model_name == 'lda': return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            if model_name == 'gpc': return GaussianProcessClassifier(random_state=42, n_jobs=-1)
            return None
        ''' ======= INTERNAL OPTIMIZATION FUNCTION END ======= '''

        ''' ========= Model Name Map ========== '''
        model_names_list = ['xgb', 'lgbm', 'randforest', 'ctbclf', 'sgd', 'svc', 'bagging', 'adaboost', 'knn', 
                            'lda', 'gpc', 'extra_trees', 'gradient_boosting', 'calibrated_ridge', 'logistic_regression', 
                            'calibrated_passive_aggressive', 'bernoulli_nb', 'gaussian_nb', 'mlp', 'hist_gradient_boosting']
        
        # Define DEFAULTS (used if HPO=False, or as base for cloning)
        defaults = {
            'xgb': XGBClassifier(n_jobs=-1, random_state=42, verbosity=0),
            'lgbm': lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42),
            'randforest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'ctbclf': ctb.CatBoostClassifier(silent=True, thread_count=-1, random_seed=42),
            'sgd': SGDClassifier(loss='log_loss', n_jobs=-1, random_state=42),
            'svc': SVC(probability=True, random_state=42),
            'bagging': BaggingClassifier(n_jobs=-1, random_state=42),
            'adaboost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
            'knn': KNeighborsClassifier(n_jobs=-1),
            'lda': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            'gpc': GaussianProcessClassifier(n_jobs=-1, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_jobs=-1, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'calibrated_ridge': CalibratedClassifierCV(RidgeClassifier(), cv=5),
            'logistic_regression': LogisticRegression(n_jobs=-1),
            'calibrated_passive_aggressive': CalibratedClassifierCV(PassiveAggressiveClassifier(), cv=5),
            'bernoulli_nb': BernoulliNB(),
            'gaussian_nb': GaussianNB(),
            'mlp': MLPClassifier(random_state=42),
            'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=42)
        }

        ''' ================================================= Recursive blend ensemble hub v1.2 ============================================= '''
        
        ''' ============ Parameter first initialization ============ '''
        num_probabilities = 1 #only 1 prob is enough to represent 2 class probabilities
        average = 'weighted'
        numbers_of_selected_models = np.zeros(num_of_level)
        
        selected_models_in_each_level = dict()
        for n_fold in range(5):
            selected_models_in_each_level[n_fold] = dict()
            for featsel in range(2):
                selected_models_in_each_level[n_fold][featsel] = dict()
                for level in range(num_of_level):
                    selected_models_in_each_level[n_fold][featsel][level] = dict()
        
        model_names = model_names_list 
        ''' ============ Parameter first initialization ============ '''
        
        exit_flag = False #for breaking the nested loops if necessary
        for n_fold, (train_idx, test_idx) in enumerate(cv.split(train_X, train_Y)):
            
            if specific_folds is not None:
                if n_fold not in specific_folds:
                    continue  # Skip this fold and move to the next
            
            print('Entered to blending fold %d' % n_fold)
            
            ''' ========= initialize train / test folds (reset for each feature selection strategy) ==========='''
            temp_train_X, temp_test_X = train_X[train_idx], train_X[test_idx] #initialize expandable train, test feature sets
            y_train, y_test = train_Y[train_idx], train_Y[test_idx]
            
            # =========== Only some numerical values need to be normalized ============
            if(indices_to_normalized is not None):
                scaler = StandardScaler()
                temp_train_X[:, indices_to_normalized] = scaler.fit_transform(temp_train_X[:, indices_to_normalized])
                temp_test_X[:, indices_to_normalized] = scaler.transform(temp_test_X[:, indices_to_normalized])
            # =========== Only some numerical values need to be normalized ============
            
            temp_stackofstack_train_X, temp_stackofstack_test_X = temp_train_X.copy(), temp_test_X.copy()
            ''' ========= initialize train / test folds (reset for each feature selection strategy) ==========='''
            
            ''' ======= CRITICAL: LEVEL-0 HPO (OPTIMIZED) & INDIVIDUAL EVALUATION (Moved Inside Loop) =========== '''
            
            current_fold_stacking_models = []
            
            print(f'Starting Level-0 Optimization & Evaluation for Fold {n_fold}...')
            for idx, name in enumerate(model_names):
                
                # 1. OPTIMIZE (Nested CV) or LOAD DEFAULT
                if hyperparameterOptimization:
                    print(f"  > Tuning {name} on Fold {n_fold} training set...")
                    try:
                        begin = time.time()
                        learner = get_optimized_model_for_fold(name, temp_train_X, y_train)
                        end = time.time()
                        print(end-begin)
                        if learner is None: learner = clone(defaults[name])
                    except:
                        learner = clone(defaults[name])
                else:
                    learner = clone(defaults[name])
                
                # 2. INDIVIDUAL EVALUATION (Train on this fold, Test on this fold)
                # This replaces the global cross_validate to ensure we use the Fold-Specific Optimized Model
                t_start = time.time()
                learner.fit(temp_train_X, y_train)
                t_end = time.time()
                elapsedtimes_individual[idx] += (t_end - t_start) # Accumulate time across folds
                
                # Predictions for Metrics
                preds = learner.predict(temp_test_X)
                probs = learner.predict_proba(temp_test_X)
                
                # Store Metrics (Fold-wise)
                accuracies_ind[idx, n_fold] = accuracy_score(y_test, preds)
                f1scores_ind[idx, n_fold] = f1_score(y_test, preds, average='weighted')
                precisions_ind[idx, n_fold] = precision_score(y_test, preds, average='weighted', zero_division=0)
                recalls_ind[idx, n_fold] = recall_score(y_test, preds, average='weighted')
                log_lossess_ind[idx, n_fold] = log_loss(y_test, probs)
                roc_auc_score_ind[idx, n_fold] = roc_auc_score(y_test, probs[:, 1])
                
                # 3. Add to Stacking List (Name, Fitted_Model_Copy_or_New_Instance)
                # Note: For stacking, we usually re-fit or use cross_val_predict. 
                # We pass the FRESH instance (not fitted) to the stacking loop, but initialized with BEST PARAMS.
                # Clone creates a fresh unfitted estimator with the same params.
                current_fold_stacking_models.append((name, clone(learner)))

            print(f'Level-0 Optimization & Evaluation for Fold {n_fold} Completed.')
            
            # Replace the global list with this fold's optimized list
            temp_stacking_models = current_fold_stacking_models 
            ''' ======= CRITICAL: LEVEL-0 HPO (OPTIMIZED) & INDIVIDUAL EVALUATION (Moved Inside Loop) =========== '''

            selected_modelindices_levelN_truefalse = np.ones(len(temp_stacking_models), dtype=bool)
            
            begin_blending = time.time()
            # Qcustoms = np.zeros(10)
            for level in range(num_of_level): #1 to 10
        
                ''' ========== Parameter intermediate initialization ========== '''
                num_basefeats = temp_train_X.shape[1]
                num_estimators = len(temp_stacking_models)
                
                num_feats = num_estimators * num_probabilities + num_basefeats #blended total amount
                ''' ========== Parameter intermediate initialization ========== '''
                
                ''' =========== Out-of-fold prediction for level-N for all learners ========== '''
                temp_oof_train_X = np.zeros((int(len(temp_train_X)), num_feats))
                temp_oof_test_X = np.zeros((int(len(temp_test_X)), num_feats))
                
                begin_innermostloop = time.time()
                
                temp_roc_auc_modelwise = np.zeros((len(temp_stacking_models)))
                
                for j, (learner_name, learner) in enumerate(temp_stacking_models):
                        
                    # NOTE: learner here is the OPTIMIZED model for this fold (from Level-0 HPO)
                    # cross_val_predict here performs the inner CV for OOF generation
                    n_jobs = -1 
                    temp_oof_predictions = cross_val_predict(learner, temp_train_X, y_train, cv=5, method='predict_proba', n_jobs=n_jobs) #OOF

                    learner.fit(temp_train_X, y_train)
                    
                    print('Entered to OOF prediction of feat selection %d, level %d, and model no %d' % (0, level+1, j))
   
                    ''' ====== Temp level-N train & test sets stacking ======= '''                        
                    if(num_probabilities == 1): #in case if the classification is binary, you need only one probability to represent 2 classess
                        temp_oof_train_X[:, j * num_probabilities  : (j+1) * num_probabilities] = temp_oof_predictions[:,0][..., np.newaxis] #scalar to 2D array conversion
                        temp_oof_test_X[:, j * num_probabilities  : (j+1) * num_probabilities] = learner.predict_proba(temp_test_X)[:,0][..., np.newaxis] #scalar to 2D array conversion
                    else:
                        temp_oof_train_X[:, j * num_probabilities  : (j+1) * num_probabilities] = temp_oof_predictions
                        temp_oof_test_X[:, j * num_probabilities  : (j+1) * num_probabilities] = learner.predict_proba(temp_test_X)
                    ''' ====== Temp level-N train & test sets stacking ======= '''
                    
                    temp_roc_auc_modelwise[j] = roc_auc_score(y_train, temp_oof_predictions[:, 1])
                ''' =========== Out-of-fold prediction for level-N for all learners ========== '''
                
                ''' ======== Blending with input feature set ======== '''
                temp_oof_train_X[:, -1 * num_basefeats:] = temp_train_X
                temp_oof_test_X[:, -1 * num_basefeats:] = temp_test_X
                ''' ======== Blending with input feature set ======== '''
                
                ''' ======= Final feature selection of level-N blended ensemble train & test datasets ======= '''
                initial_feature_size = temp_oof_train_X.shape[1]
                
                if(level369 == True):
                    
                    if((iffeatselection == True) and (level % 3 == 0) and (level > 0)):
                        
                        UR = 0.3
                        UR_Max = 0.3
                        UR_Min = 0.001
                        Max_FEs = 200
                        Max_Run = 5
                        Run = 1
                        Cost = np.zeros([Max_FEs, Max_Run])
                        
                        selected_feaure_indices = self.SFE_run(Input=temp_oof_train_X, Target=y_train, UR=UR, UR_Max=UR_Max, UR_Min=UR_Min, Max_FEs=Max_FEs, Max_Run=Max_Run, Run=Run, Cost=Cost)
                        
                        temp_oof_train_X, temp_oof_test_X = temp_oof_train_X[:, selected_feaure_indices], temp_oof_test_X[:, selected_feaure_indices]
                        
                        print('Boruta feature selection of level %d done successfully' % level)
                        print('Selected number of features are F=%d out of %d' % (len(selected_feaure_indices), initial_feature_size))
                        
                else:
                    
                    if((iffeatselection == True) and (level > 0)):
                        
                        UR = 0.3
                        UR_Max = 0.3
                        UR_Min = 0.001
                        Max_FEs = 200
                        Max_Run = 5
                        Run = 1
                        Cost = np.zeros([Max_FEs, Max_Run])
                        
                        selected_feaure_indices = self.SFE_run(Input=temp_oof_train_X, Target=y_train, UR=UR, UR_Max=UR_Max, UR_Min=UR_Min, Max_FEs=Max_FEs, Max_Run=Max_Run, Run=Run, Cost=Cost)
                        
                        temp_oof_train_X, temp_oof_test_X = temp_oof_train_X[:, selected_feaure_indices], temp_oof_test_X[:, selected_feaure_indices]
                        
                        print('Boruta feature selection of level %d done successfully' % level)
                        print('Selected number of features are F=%d out of %d' % (len(selected_feaure_indices), initial_feature_size))
   
                ''' ======= Final feature selection of level-N blended ensemble train & test datasets ======= '''
                
                until_meta_learning = time.time()
                
                accumulated_just_meta_learnings_for_this_fold = np.sum(elapsedtimes_blendensemble_just_meta_learning[n_fold, 0 : level, :])
                elapsedtimes_blendensemble_until_meta_learning[n_fold, level] = until_meta_learning - (begin_blending + accumulated_just_meta_learnings_for_this_fold)
                        
                ''' ======== Meta-model evaluation of final blended ensemble train set with test set ========= '''
                for ifselected, selected_indice in zip(selected_modelindices_levelN_truefalse, range(20)):
                    
                    if(ifselected == True):
                        
                        final_estimator = clone(current_fold_stacking_models[selected_indice][1]) #initialize for each fold using Optimized Model
                        
                        end_blending = time.time()
                        
                        begin_metalearning = time.time()
                        final_estimator.fit(temp_oof_train_X, y_train)
                        end_metalearning = time.time()
                                                 
                        elapsedtimes_blendensemble_levels[n_fold, level, selected_indice] = (until_meta_learning - begin_blending) + (end_metalearning - begin_metalearning) 
                        elapsedtimes_blendensemble_just_meta_learning[n_fold, level, selected_indice] = end_metalearning - begin_metalearning
                                             
                        final_predictions = final_estimator.predict(temp_oof_test_X)
                        final_prediction_probs = final_estimator.predict_proba(temp_oof_test_X)
                    
                        ''' ========== Temp results logging ========== '''
                        accuracies_blnd[n_fold, level, selected_indice] = accuracy_score(y_test, final_predictions)
                        f1scores_blnd[n_fold, level, selected_indice] = f1_score(y_test, final_predictions, average=average)
                        precisions_blnd[n_fold, level, selected_indice] = precision_score(y_test, final_predictions, average=average)
                        recalls_blnd[n_fold, level, selected_indice] = recall_score(y_test, final_predictions, average=average)
                        log_lossess_blnd[n_fold, level, selected_indice] = log_loss(y_test, final_prediction_probs)
                        roc_auc_score_blnd[n_fold, level, selected_indice] = roc_auc_score(y_test, final_prediction_probs[:,1])
                        ''' ========== Temp results logging ========== '''

                ''' ======== Meta-model evaluation of final blended ensemble train set with test set ========= '''
                
                
                ''' ================================== Ensemble Level-N model elimination randomized OOF-based [heuristical] ========================== '''                
                print('Model elimination process has begun...')

                threshold_percentile_init = 10
                
                # --- Compute range and apply strength factor ---
                range_val = np.max(temp_roc_auc_modelwise) - np.min(temp_roc_auc_modelwise)
                if blur_strength == 'light':
                    noise_scale = 0.05 * range_val
                elif blur_strength == 'moderate':
                    noise_scale = 0.10 * range_val
                elif blur_strength == 'oof-strict':
                    print('no noise induction')
                    noise_scale = 0
                else:
                    raise ValueError("Invalid blur_strength")
                
                print(f'[Blurrization] Level {level+1}: noise_scale = {noise_scale:.6f} (range-based, strength = {blur_strength})')
                
                # --- Inject Gaussian noise to blur OOF-based scores ---
                if(blur_strength != 'oof-strict'):
                    
                    noisy_accuracy_modelwise = temp_roc_auc_modelwise + np.random.normal(
                        loc=0.0,
                        scale=noise_scale,
                        size=temp_roc_auc_modelwise.shape
                    )
                    
                else:
                    
                    noisy_accuracy_modelwise = temp_roc_auc_modelwise
                
                # --- Determine new percentile threshold based on noisy scores ---
                temp_std = np.std(noisy_accuracy_modelwise)
                new_threshold_percentile = threshold_percentile_init / 2 + (temp_std ** 2) * 80
                Qcustom = np.percentile(noisy_accuracy_modelwise, new_threshold_percentile)
                
                # --- Prune based on blurred threshold ---
                selected_modelindices_levelN_previously = np.argwhere(selected_modelindices_levelN_truefalse == True)[:, 0]
                selected_modelindices_levelN = np.argwhere(noisy_accuracy_modelwise >= Qcustom)[:, 0]
                indices_to_be_selected = selected_modelindices_levelN_previously[selected_modelindices_levelN]
                
                selected_modelindices_levelN_truefalse = np.zeros(len(selected_modelindices_levelN_truefalse), dtype=bool)
                selected_modelindices_levelN_truefalse[indices_to_be_selected] = True
                
                # --- Log selected models ---
                numbers_of_selected_models[level] = len(selected_modelindices_levelN)
                print('The number of selected models in level %d is %d' % (level + 1, numbers_of_selected_models[level]))
                
                selected_modelindices_levelN_currently = np.argwhere(selected_modelindices_levelN_truefalse == True)[:, 0]
                model_names = model_names_list # reset names to full list to index correctly? No, need mapping. 
                selected_models_in_each_level[n_fold][level] = [model_names_list[indice] for indice in selected_modelindices_levelN_currently]
                
                # --- Check exit condition or continue stacking ---
                if numbers_of_selected_models[level] < left_model_threshold:
                    exit_flag = True
                    break
                else:
                    temp_stacking_models = [current_fold_stacking_models[indice] for indice in selected_modelindices_levelN_currently]
                ''' ================================== Ensemble Level-N model elimination randomized OOF-based [heuristical] ========================== ''' 
                
                ''' ===== approximating total time required ==== '''
                end_innermostloop = time.time()
                total_time_for_innermostloop = end_innermostloop - begin_innermostloop
                print('Total approximate time required is %d seconds' % (total_time_for_innermostloop * 100))
                ''' ===== approximating total time required ==== '''
                
                ''' ======= Recursively change the initial train / test data blended --> input data ====== '''
                temp_train_X, temp_test_X = temp_oof_train_X.copy(), temp_oof_test_X.copy()
                
                number_of_feautures_blendensemble_levels[n_fold, level] = np.shape(temp_oof_train_X)[1]
                ''' ======= Recursively change the initial train / test data blended --> input data ====== '''
                
                ''' ======= Stack-of-stacking ======== '''
                temp_stackofstack_train_X, temp_stackofstack_test_X = np.column_stack((temp_stackofstack_train_X, temp_train_X)), np.column_stack((temp_stackofstack_test_X, temp_test_X))
                ''' ======= Stack-of-stacking ======== '''
                
            ''' ========= Stack-of-stacking evaluation ========== '''
            for selected_indice in range(20):
                                     
                final_estimator = clone(current_fold_stacking_models[selected_indice][1]) #initialize for each fold using Optimized
                
                end_blending = time.time()
                
                begin_metalearning = time.time()
                final_estimator.fit(temp_stackofstack_train_X, y_train)
                end_metalearning = time.time()
                                     
                elapsedtimes_stackofstacking[n_fold, selected_indice] = (end_blending - begin_blending) + (end_metalearning - begin_metalearning) 
                
                final_predictions = final_estimator.predict(temp_stackofstack_test_X)
                final_prediction_probs = final_estimator.predict_proba(temp_stackofstack_test_X)
                
                ''' ========== Temp results logging ========== '''
                accuracies_stackofstack[n_fold, selected_indice] = accuracy_score(y_test, final_predictions)
                f1scores_stackofstack[n_fold, selected_indice] = f1_score(y_test, final_predictions, average=average)
                precisions_stackofstack[n_fold, selected_indice] = precision_score(y_test, final_predictions, average=average)
                recalls_stackofstack[n_fold, selected_indice] = recall_score(y_test, final_predictions, average=average)
                log_lossess_stackofstack[n_fold, selected_indice] = log_loss(y_test, final_prediction_probs)
                roc_auc_score_stackofstack[n_fold, selected_indice] = roc_auc_score(y_test, final_prediction_probs[:,1])
                ''' ========== Temp results logging ========== '''
                
            ''' ========= Stack-of-stacking evaluation ========== '''
                    
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        
        '''
        What has been happening here:
        1) this is elapsed time handler section of the function
        2) I am trying to restructure the results for more meaningful storing as dictionaries but need re-ordering the mess
        3) I want with and without normalized versions of:
         3.1) fold-wise individual model elapsed times + each blend level elapsed time
         3.2) averaged of those folds as another dictionary
        4) stack-of-stacking results as separate dictionary without normalization
        '''
      
        ''' ==================== Initial summation operation ==================== '''
        elapsedtimes_blendensemble_levels_sum = np.sum(elapsedtimes_blendensemble_levels, axis=(0,2))
        elapsedtimes_blendensemble_levels_sum_perfold = np.sum(elapsedtimes_blendensemble_levels, axis=(2))
      
        elapsedtimes_stackofstacking_sum = np.sum(elapsedtimes_stackofstacking, axis=(0,1))
        ''' ==================== Initial summation operation ==================== '''
      
        ''' ============================================== Without stack-of-stacking ================================================== '''
        ''' ============================================== Without stack-of-stacking ================================================== '''
      
        ''' =========== Elapsed time without normalization and without stack-of-stacking =========== '''
        elapsedtimes_individual_withoutnorm = elapsedtimes_individual
        elapsedtimes_blendensemble_levels_withoutnorm = elapsedtimes_blendensemble_levels_sum
        elapsedtimes_blendensemble_levels_withoutnorm_perfold = elapsedtimes_blendensemble_levels_sum_perfold
        ''' =========== Elapsed time without normalization and without stack-of-stacking =========== '''
        
        ''' =========== Elapsed time normalizer without stack-of-stacking =========== '''
        maximum_elapsed_time = max(np.max(elapsedtimes_individual), np.max(elapsedtimes_blendensemble_levels_sum))
      
        elapsedtimes_individual_norm = elapsedtimes_individual / maximum_elapsed_time
        elapsedtimes_blendensemble_levels_norm = elapsedtimes_blendensemble_levels_sum / maximum_elapsed_time
        
        ''' ======== Special per fold treatment ======== '''
        elapsedtimes_individual_norm_perfold = np.zeros((5,20))
        elapsedtimes_blendensemble_levels_norm_perfold = np.zeros((5,10))
        for fold in range(5):
            maximum_elapsed_time_perfold = max(np.max(elapsedtimes_individual / 5), np.max(elapsedtimes_blendensemble_levels_sum_perfold[fold]))
            elapsedtimes_individual_norm_perfold[fold] = (elapsedtimes_individual / 5) / maximum_elapsed_time_perfold
            elapsedtimes_blendensemble_levels_norm_perfold[fold] = elapsedtimes_blendensemble_levels_sum_perfold[fold] / maximum_elapsed_time_perfold
        ''' ======== Special per fold treatment ======== '''
        
        ''' =========== Elapsed time normalizer without stack-of-stacking =========== '''
        
        ''' =========== Elapsed time without stack-of-stacking: numpy --> dict ============ '''
        elapsedtimes_individual_norm_dict, elapsedtimes_blendensemble_levels_norm_dict = {}, {}
        elapsedtimes_individual_norm_perfold_dict, elapsedtimes_blendensemble_levels_norm_perfold_dict = {}, {}
        elapsedtimes_individual_withoutnorm_dict, elapsedtimes_blendensemble_levels_withoutnorm_dict = {}, {}
      
        model_names = model_names_list 
        for i in range(len(model_names)):
            elapsedtimes_individual_norm_dict[model_names[i]] = elapsedtimes_individual_norm[i]
            elapsedtimes_individual_withoutnorm_dict[model_names[i]] = elapsedtimes_individual_withoutnorm[i]
      
        for i in range(num_of_level):
            elapsedtimes_blendensemble_levels_norm_dict['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_norm[i]
      
            elapsedtimes_blendensemble_levels_withoutnorm_dict['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_withoutnorm[i]
                
        ''' ======== Special per fold treatment from numpy to dict ======== '''
        for fold in range(5):
            temp_elapsedtimes_blendensemble_levels_norm_fold_dict = dict()
            temp_elapsedtimes_individual_norm_fold_dict = dict()
      
            for i in range(num_of_level):
                temp_elapsedtimes_blendensemble_levels_norm_fold_dict['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_norm_perfold[fold,i]
                
                temp_elapsedtimes_individual_norm_fold_dict[model_names[i]] = elapsedtimes_individual_norm_perfold[fold, i]
      
            elapsedtimes_blendensemble_levels_norm_perfold_dict['fold_' + str(fold)] = temp_elapsedtimes_blendensemble_levels_norm_fold_dict
            elapsedtimes_individual_norm_perfold_dict['fold_' + str(fold)] = temp_elapsedtimes_individual_norm_fold_dict
        ''' ======== Special per fold treatment from numpy to dict ======== '''
      
        ''' =========== Elapsed time without stack-of-stacking: numpy --> dict ============ '''
        
        ''' ============================================== Without stack-of-stacking ================================================== '''
        ''' ============================================== Without stack-of-stacking ================================================== '''
      
        ''' ============================================== With stack-of-stacking ================================================== '''
        ''' ============================================== With stack-of-stacking ================================================== '''
      
        ''' =========== Elapsed time normalizer with stack-of-stacking =========== '''
        maximum_elapsed_time2 = max(np.max(elapsedtimes_individual), np.max(elapsedtimes_stackofstacking_sum))
        elapsedtimes_individual_norm2 = elapsedtimes_individual / maximum_elapsed_time2
        elapsedtimes_blendensemble_levels_norm2 = elapsedtimes_blendensemble_levels_sum / maximum_elapsed_time2
        elapsedtimes_stackofstacking_norm = elapsedtimes_stackofstacking_sum / maximum_elapsed_time2
        ''' =========== Elapsed time normalizer with stack-of-stacking =========== '''
        
        ''' =========== Elapsed time with stack-of-stacking: numpy --> dict ============ '''
        elapsedtimes_individual_norm_dict2, elapsedtimes_blendensemble_levels_norm_dict2, elapsedtimes_stackofstacking_norm_dict = {}, {}, {}
              
        for i in range(len(model_names)):
            elapsedtimes_individual_norm_dict2[model_names[i]] = elapsedtimes_individual_norm2[i]
            
        for i in range(num_of_level):
            elapsedtimes_blendensemble_levels_norm_dict2['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_norm2[i]
                
        elapsedtimes_stackofstacking_norm_dict = elapsedtimes_stackofstacking_norm
        ''' =========== Elapsed time with stack-of-stacking: numpy --> dict ============ '''
        
        ''' ============================================== With stack-of-stacking ================================================== '''
        ''' ============================================== With stack-of-stacking ================================================== '''
      
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''

        ''' ===================== Return results ==================== '''
        
        # ============= Averaging & std non-zeros =============
        accuracies_blnd_nan = np.where(accuracies_blnd == 0, np.nan, accuracies_blnd)
        accuracies_blnd_mean = np.nanmean(accuracies_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        accuracies_blnd_std = np.nanstd(accuracies_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values

        f1scores_blnd_nan = np.where(f1scores_blnd == 0, np.nan, f1scores_blnd)
        f1scores_blnd_mean = np.nanmean(f1scores_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        f1scores_blnd_std = np.nanstd(f1scores_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values

        precisions_blnd_nan = np.where(precisions_blnd == 0, np.nan, precisions_blnd)
        precisions_blnd_mean = np.nanmean(precisions_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        precisions_blnd_std = np.nanstd(precisions_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values

        recalls_blnd_nan = np.where(recalls_blnd == 0, np.nan, recalls_blnd)
        recalls_blnd_mean = np.nanmean(recalls_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        recalls_blnd_std = np.nanstd(recalls_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values

        log_lossess_blnd_nan = np.where(log_lossess_blnd == 0, np.nan, log_lossess_blnd)
        log_lossess_blnd_mean = np.nanmean(log_lossess_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        log_lossess_blnd_std = np.nanstd(log_lossess_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values

        roc_auc_score_blnd_nan = np.where(roc_auc_score_blnd == 0, np.nan, roc_auc_score_blnd)
        roc_auc_score_blnd_mean = np.nanmean(roc_auc_score_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        roc_auc_score_blnd_std = np.nanstd(roc_auc_score_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values
        # ============= Averaging & std non-zeros =============
        
        # ============= Averaging & std of stack of stacking =============
        accuracies_stackofstack_mean = np.mean(accuracies_stackofstack, axis=0)
        accuracies_stackofstack_std = np.std(accuracies_stackofstack, axis=0)

        precisions_stackofstack_mean = np.mean(precisions_stackofstack, axis=0)
        precisions_stackofstack_std = np.std(precisions_stackofstack, axis=0)
        
        f1scores_stackofstack_mean = np.mean(f1scores_stackofstack, axis=0)
        f1scores_stackofstack_std = np.std(f1scores_stackofstack, axis=0)
        
        recalls_stackofstack_mean = np.mean(recalls_stackofstack, axis=0)
        recalls_stackofstack_std = np.std(recalls_stackofstack, axis=0)
        
        log_lossess_stackofstack_mean = np.mean(log_lossess_stackofstack, axis=0)
        log_lossess_stackofstack_std = np.std(log_lossess_stackofstack, axis=0)
        
        roc_auc_score_stackofstack_mean = np.mean(roc_auc_score_stackofstack, axis=0)
        roc_auc_score_stackofstack_std = np.std(roc_auc_score_stackofstack, axis=0)
        # ============= Averaging & std of stack of stacking =============
            
        all_results = {'individual_results': {'accuracy_ind': accuracies_ind, 'f1scores_ind': f1scores_ind, 'precisions_ind': precisions_ind,
                                             'recalls_ind': recalls_ind, 'log_lossess_ind': log_lossess_ind, 'auc_rocs_ind': roc_auc_score_ind}, \
                      'individual_avg_results': {'avg_accuracy_ind': np.mean(accuracies_ind, axis=1), 'std_accuracy_ind': np.std(accuracies_ind, axis=1), 
                                                 'avg_f1scores_ind': np.mean(f1scores_ind, axis=1), 'avg_precisions_ind': np.mean(precisions_ind, axis=1),
                                                 'avg_recalls_ind': np.mean(recalls_ind, axis=1) , 'avg_log_lossess_ind': np.mean(log_lossess_ind, axis=1),
                                                 'auc_rocs_ind': np.mean(roc_auc_score_ind, axis=1)}, \
                               
                      'blendensemble_results': {'accuracy_blnd': accuracies_blnd, 'f1scores_blnd': f1scores_blnd,
                                                 'precisions_blnd': precisions_blnd, 'recalls_blnd': recalls_blnd,
                                                 'log_losses_blnd': log_lossess_blnd, 'roc_aucs_blnd': roc_auc_score_blnd}, \
                      'blendensemble_avg_results': {'avg_accuracy_blnd': accuracies_blnd_mean , 'std_accuracy_blnd': accuracies_blnd_std,
                                                 'avg_f1scores_blnd': f1scores_blnd_mean, 'std_f1scores_blnd': f1scores_blnd_std,
                                                 'avg_precisions_blnd': precisions_blnd_mean, 'std_precisions_blnd': precisions_blnd_std,
                                                 'avg_recalls_blnd': recalls_blnd_mean, 'std_recalls_blnd': recalls_blnd_std,
                                                 'avg_log_losses_blnd': log_lossess_blnd_mean, 'std_log_losses_blnd': log_lossess_blnd_std,
                                                 'avg_roc_aucs_blnd': roc_auc_score_blnd_mean, 'std_roc_aucs_blnd': roc_auc_score_blnd_std}, \
                        
                      'stackofstacking_results': {'accuracies_stackofstack': accuracies_stackofstack, 'f1scores_stackofstack': f1scores_stackofstack,
                                                 'precisions_stackofstack': precisions_stackofstack, 'recalls_stackofstack': recalls_stackofstack,
                                                 'log_lossess_stackofstack': log_lossess_stackofstack, 'roc_aucs_stackofstack': roc_auc_score_stackofstack}, \
                      'stackofstacking_avg_results': {'accuracies_stackofstack_mean': accuracies_stackofstack_mean , 'accuracies_stackofstack_std': accuracies_stackofstack_std,
                                                 'f1scores_stackofstack_mean': f1scores_stackofstack_mean, 'f1scores_stackofstack_std': f1scores_stackofstack_std,
                                                 'precisions_stackofstack_mean': precisions_stackofstack_mean, 'precisions_stackofstack_std': precisions_stackofstack_std,
                                                 'recalls_stackofstack_mean': recalls_stackofstack_mean, 'recalls_stackofstack_std': recalls_stackofstack_std,
                                                 'log_lossess_stackofstack_mean': log_lossess_stackofstack_mean, 'log_lossess_stackofstack_std': log_lossess_stackofstack_std,
                                                 'roc_aucs_stackofstack_mean': roc_auc_score_stackofstack_mean, 'roc_aucs_stackofstack_std': roc_auc_score_stackofstack_std}, \
                            
                      'elapsedtimes_withoutstackofstacking_normed': {'individual_models_normed': elapsedtimes_individual_norm_dict, 'blend_ensemble_level1-10_normed': elapsedtimes_blendensemble_levels_norm_dict}, \
                      'elapsedtimes_withoutstackofstacking_normed_perfold': {'individual_models_normed_perfold': elapsedtimes_individual_norm_perfold_dict, 
                                                                             'blend_ensemble_level1-10_normed_perfold': elapsedtimes_blendensemble_levels_norm_perfold_dict}, \
   
                      'elapsedtimes_withoutstackofstacking_withoutnormed': {'individual_models_normed': elapsedtimes_individual_withoutnorm_dict, 
                                                                            'blend_ensemble_level1-10_normed': elapsedtimes_blendensemble_levels_withoutnorm_dict}, \
                      'elapsedtimes_withstackofstacking': {'individual_models2': elapsedtimes_individual_norm_dict2, 'blend_ensemble_level1-10': elapsedtimes_blendensemble_levels_norm_dict2,
                                                           'elapsedtimes_stackofstacking_norm_dict': elapsedtimes_stackofstacking_norm_dict}, \
                      'other information': {'number_of_feautures_blendensemble_levels': number_of_feautures_blendensemble_levels, 
                                            'bare_elapsedtimes_blendensemble_levels': elapsedtimes_blendensemble_levels, 
                                            'elapsedtimes_blendensemble_until_meta_learning': elapsedtimes_blendensemble_until_meta_learning,
                                            'elapsedtimes_blendensemble_just_meta_learning': elapsedtimes_blendensemble_just_meta_learning}
                              }
        
        return all_results
    
    def AscentTheRocket_multiclass_massive_exploration(self, train_X, train_Y, hyperparameterOptimization=False, num_of_level=10, iffeatselection_or_not=True, level369=True, 
                                                                                         feat_selection_type='attentionlayer', stackoverstacking=False, indices_to_normalized=None, left_model_threshold=5, 
                                                                                         specific_folds=None, blur_strength='light'):
        
        ''' ============== Blending protocol =============
        1) original: (if no intermeditate feature selection)
        2) cumulative: (if intermediate feature selection)
         ============== Blending protocol ============= '''
         
        ''' Indexing the model order: 
            
        ======= Stand-alone models ======= (14 models)
        1) xgb 
        2) lgbm 
        3) randomforest 
        4) svc
        5) bagging
        6) adaboost
        7) knn
        8) extra_trees
        9) gradient_boosting
        10) logistic_regression
        11) bernoulli_nb
        12) gaussian_nb
        13) mlp
        14) histgradient boosting
        ======= Stand-alone models =======
        '''
        
        # --- Imports for HPO and Models ---

        # Clean up logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", message="[LightGBM]")
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        
        np.random.seed(31) 
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    
        ''' ================= Metric initialization hub ================ '''
        num_of_folds = 5
        num_of_models = 14 # Adjusted to match your list (SGD removed)
        num_of_classes = len(np.unique(train_Y))
        
        ''' ======== Metric initialization individual ======== '''
        accuracies_ind = np.zeros((num_of_models, num_of_folds))
        f1scores_ind = np.zeros((num_of_models, num_of_folds))
        precisions_ind = np.zeros((num_of_models, num_of_folds))
        recalls_ind = np.zeros((num_of_models, num_of_folds))
        log_lossess_ind = np.zeros((num_of_models, num_of_folds))
        elapsedtimes_individual = np.zeros(num_of_models)
        ''' ======== Metric initialization individual ======== '''
        
        ''' ======== Metric initialization blend ensemble (L1 -- L10) ======== '''   
        accuracies_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        f1scores_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        precisions_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        recalls_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))
        log_lossess_blnd = np.zeros((num_of_folds, num_of_level, num_of_models))

        elapsedtimes_blendensemble_levels = np.zeros((num_of_folds, num_of_level, num_of_models))
        elapsedtimes_blendensemble_until_meta_learning = np.zeros((num_of_folds, num_of_level))
        elapsedtimes_blendensemble_just_meta_learning = np.zeros((num_of_folds, num_of_level, num_of_models))
        
        number_of_feautures_blendensemble_levels = np.zeros((num_of_folds, num_of_level))
        ''' ======== Metric initialization blend ensemble (L1 -- L10) ======== '''
        
        ''' ======= Metric initialization stack-of-stacking ======== (Features + L1....L10) '''
        accuracies_stackofstack = np.zeros((num_of_folds, num_of_models))
        f1scores_stackofstack = np.zeros((num_of_folds, num_of_models))
        precisions_stackofstack = np.zeros((num_of_folds, num_of_models))
        recalls_stackofstack = np.zeros((num_of_folds, num_of_models))
        log_lossess_stackofstack = np.zeros((num_of_folds, num_of_models))
        
        elapsedtimes_stackofstacking = np.zeros((num_of_folds, num_of_models))
        ''' ======= Metric initialization stack-of-stacking ======== (Features + L1....L10) '''
         
        ''' ======= INTERNAL OPTIMIZATION FUNCTION (Nested CV) ======= '''
        def get_optimized_model_for_fold(model_name, X_train, y_train):
            """ Runs Optuna on the fold's training data to find best params (Multi-Class). """
            
            def objective(trial):
                # --- Group 1: Heavy Hitters ---
                if model_name == 'xgb':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'objective': 'multi:softprob', 'num_class': num_of_classes,
                        'n_jobs': -1, 'random_state': 42, 'verbosity': 0
                    }
                    model = XGBClassifier(**params)
                elif model_name == 'lgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                        'objective': 'multiclass', 'num_class': num_of_classes,
                        'n_jobs': -1, 'random_state': 42, 'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params)
                elif model_name == 'randforest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'n_jobs': -1, 'random_state': 42
                    }
                    model = RandomForestClassifier(**params)
                # --- Group 2: Linear / SVM ---
                elif model_name == 'svc':
                    params = {
                        'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
                        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                        'probability': True, 'random_state': 42,
                        'decision_function_shape': 'ovr',
                        'max_iter': 2000, 'cache_size': 2000 # Hard limits for speed
                    }
                    model = SVC(**params)
                # --- Group 3: Others ---
                elif model_name == 'knn':
                    n_neighbors = trial.suggest_int('n_neighbors', 3, 50)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                elif model_name == 'mlp':
                    params = {
                        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                        'max_iter': 500, 'random_state': 42
                    }
                    model = MLPClassifier(**params)
                # --- Fallbacks for simple models ---
                elif model_name == 'bagging':
                    model = BaggingClassifier(n_estimators=trial.suggest_int('n', 10, 50), n_jobs=-1, random_state=42)
                elif model_name == 'adaboost':
                    model = AdaBoostClassifier(n_estimators=trial.suggest_int('n', 50, 150), algorithm='SAMME', random_state=42)
                elif model_name == 'extra_trees':
                    model = ExtraTreesClassifier(n_estimators=trial.suggest_int('n', 100, 300), n_jobs=-1, random_state=42)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingClassifier(n_estimators=trial.suggest_int('n', 100, 300), random_state=42)
                elif model_name == 'hist_gradient_boosting':
                    model = HistGradientBoostingClassifier(max_iter=trial.suggest_int('n', 100, 300), random_state=42)
                elif model_name == 'logistic_regression':
                     model = LogisticRegression(C=trial.suggest_float('C', 0.1, 10.0, log=True), max_iter=1000, n_jobs=-1)
                elif model_name == 'bernoulli_nb':
                    model = BernoulliNB(alpha=trial.suggest_float('alpha', 0.1, 2.0))
                elif model_name == 'gaussian_nb':
                    model = GaussianNB(var_smoothing=trial.suggest_float('var', 1e-9, 1e-5, log=True))
                else: 
                    return 0
                
                # Fast internal CV (3-fold) for HPO - Use Accuracy for simplicity in HPO
                try:
                    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=1).mean()
                except:
                    return 0.0

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30) # 30 trials per fold
            best = study.best_params
            
            # Return fresh model with best params
            if model_name == 'xgb': return XGBClassifier(**best, objective='multi:softprob', num_class=num_of_classes, n_jobs=-1, random_state=42, verbosity=0)
            if model_name == 'lgbm': return lgb.LGBMClassifier(**best, objective='multiclass', num_class=num_of_classes, n_jobs=-1, verbose=-1, random_state=42)
            if model_name == 'randforest': return RandomForestClassifier(**best, n_jobs=-1, random_state=42)
            if model_name == 'svc': return SVC(**best, probability=True, random_state=42, decision_function_shape='ovr', max_iter=2000, cache_size=2000)
            if model_name == 'knn': return KNeighborsClassifier(**best, n_jobs=-1)
            if model_name == 'mlp': return MLPClassifier(**best, max_iter=500, random_state=42)
            if model_name == 'bagging': return BaggingClassifier(n_estimators=best['n'], n_jobs=-1, random_state=42)
            if model_name == 'adaboost': return AdaBoostClassifier(n_estimators=best['n'], algorithm='SAMME', random_state=42)
            if model_name == 'extra_trees': return ExtraTreesClassifier(n_estimators=best['n'], n_jobs=-1, random_state=42)
            if model_name == 'gradient_boosting': return GradientBoostingClassifier(n_estimators=best['n'], random_state=42)
            if model_name == 'hist_gradient_boosting': return HistGradientBoostingClassifier(max_iter=best['n'], random_state=42)
            if model_name == 'logistic_regression': return LogisticRegression(C=best['C'], max_iter=1000, n_jobs=-1)
            if model_name == 'bernoulli_nb': return BernoulliNB(alpha=best['alpha'])
            if model_name == 'gaussian_nb': return GaussianNB(var_smoothing=best['var'])
            return None
        ''' ======= INTERNAL OPTIMIZATION FUNCTION END ======= '''

        ''' ========= Model Name Map (14 Models) ========== '''
        model_names_list = ['xgb', 'lgbm', 'randforest', 'svc', 'bagging', 'adaboost', 'knn',
                            'extra_trees', 'gradient_boosting', 'logistic_regression',
                            'bernoulli_nb', 'gaussian_nb', 'mlp', 'hist_gradient_boosting']
        
        # Define DEFAULTS (used if HPO=False, or as base for cloning)
        defaults = {
            'xgb': XGBClassifier(n_jobs=-1, random_state=42, verbosity=0),
            'lgbm': lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42),
            'randforest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'svc': SVC(probability=True, random_state=42),
            'bagging': BaggingClassifier(n_jobs=-1, random_state=42),
            'adaboost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
            'knn': KNeighborsClassifier(n_jobs=-1),
            'extra_trees': ExtraTreesClassifier(n_jobs=-1, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(n_jobs=-1),
            'bernoulli_nb': BernoulliNB(),
            'gaussian_nb': GaussianNB(),
            'mlp': MLPClassifier(random_state=42),
            'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=42)
        }

        ''' ================================================= Recursive blend ensemble hub v1.2 ============================================= '''
        
        ''' ============ Parameter first initialization ============ '''
        num_probabilities = len(np.unique(train_Y))
        if(num_probabilities == 2):
            num_probabilities = 1 #only 1 prob is enough to represent 2 class probabilities
            average = 'binary'
        else:
            average = 'weighted'
        # num_probabilities logic handled in loop (depends on model)
        # For multi-class, we usually stack all class probabilities
        
        numbers_of_selected_models = np.zeros(num_of_level)
        
        selected_models_in_each_level = dict()
        for n_fold in range(5):
            selected_models_in_each_level[n_fold] = dict()
            for featsel in range(2):
                selected_models_in_each_level[n_fold][featsel] = dict()
                for level in range(num_of_level):
                    selected_models_in_each_level[n_fold][featsel][level] = dict()
        
        model_names = model_names_list 
        ''' ============ Parameter first initialization ============ '''
        
        exit_flag = False 
        for n_fold, (train_idx, test_idx) in enumerate(cv.split(train_X, train_Y)):
            
            if specific_folds is not None:
                if n_fold not in specific_folds:
                    continue 
            
            print('Entered to blending fold %d' % n_fold)
            
            ''' ========= initialize train / test folds ==========='''
            temp_train_X, temp_test_X = train_X[train_idx], train_X[test_idx] 
            y_train, y_test = train_Y[train_idx], train_Y[test_idx]
            
            if(indices_to_normalized is not None):
                scaler = StandardScaler()
                temp_train_X[:, indices_to_normalized] = scaler.fit_transform(temp_train_X[:, indices_to_normalized])
                temp_test_X[:, indices_to_normalized] = scaler.transform(temp_test_X[:, indices_to_normalized])
            
            temp_stackofstack_train_X, temp_stackofstack_test_X = temp_train_X.copy(), temp_test_X.copy()
            ''' ========= initialize train / test folds ==========='''
            
            # =========================================================================
            #  CRITICAL: LEVEL-0 HPO (OPTIMIZED) & INDIVIDUAL EVALUATION (Moved Inside Loop)
            # =========================================================================
            current_fold_stacking_models = []
            
            print(f'Starting Level-0 Optimization & Evaluation for Fold {n_fold}...')
            for idx, name in enumerate(model_names):
                
                # 1. OPTIMIZE (Nested CV) or LOAD DEFAULT
                if hyperparameterOptimization:
                    print(f"  > Tuning {name} on Fold {n_fold} training set...")
                    try:
                        learner = get_optimized_model_for_fold(name, temp_train_X, y_train)
                        if learner is None: learner = clone(defaults[name])
                    except:
                        learner = clone(defaults[name])
                else:
                    learner = clone(defaults[name])
                
                # 2. INDIVIDUAL EVALUATION (Train on this fold, Test on this fold)
                t_start = time.time()
                learner.fit(temp_train_X, y_train)
                t_end = time.time()
                elapsedtimes_individual[idx] += (t_end - t_start) 
                
                preds = learner.predict(temp_test_X)
                probs = learner.predict_proba(temp_test_X)
                
                accuracies_ind[idx, n_fold] = accuracy_score(y_test, preds)
                f1scores_ind[idx, n_fold] = f1_score(y_test, preds, average='weighted')
                precisions_ind[idx, n_fold] = precision_score(y_test, preds, average='weighted', zero_division=0)
                recalls_ind[idx, n_fold] = recall_score(y_test, preds, average='weighted')
                log_lossess_ind[idx, n_fold] = log_loss(y_test, probs)
                
                # 3. Add to Stacking List 
                current_fold_stacking_models.append((name, clone(learner)))

            print(f'Level-0 Optimization & Evaluation for Fold {n_fold} Completed.')
            
            temp_stacking_models = current_fold_stacking_models 
            # =========================================================================
            
            selected_modelindices_levelN_truefalse = np.ones(len(temp_stacking_models), dtype=bool)
            
            begin_blending = time.time()
            for level in range(num_of_level): 
        
                ''' ========== Parameter intermediate initialization ========== '''
                num_basefeats = temp_train_X.shape[1]
                num_estimators = len(temp_stacking_models)
                
                # Multi-class stacking dimension calculation:
                # Each model outputs N_Classes probabilities
                num_feats = num_estimators * num_of_classes + num_basefeats 
                ''' ========== Parameter intermediate initialization ========== '''
                
                ''' =========== Out-of-fold prediction for level-N for all learners ========== '''
                temp_oof_train_X = np.zeros((int(len(temp_train_X)), num_feats))
                temp_oof_test_X = np.zeros((int(len(temp_test_X)), num_feats))
                
                begin_innermostloop = time.time()
                
                temp_accuracy_modelwise = np.zeros((len(temp_stacking_models)))
                
                for j, (learner_name, learner) in enumerate(temp_stacking_models): 
                        
                    n_jobs = -1 
                    temp_oof_predictions = cross_val_predict(learner, temp_train_X, y_train, cv=5, method='predict_proba', n_jobs=n_jobs) #OOF
                    print('If NAN in temp_oof_prediction of model no: %d is %s' % (j, np.isnan(temp_oof_predictions).any()))
    
                    learner.fit(temp_train_X, y_train)
                    
                    print('Entered to OOF prediction, level %d, and model no %d' % (level+1, j))
    
                    ''' ====== Temp level-N train & test sets stacking (Multi-Class) ======= '''                        
                    # Stack ALL class probabilities
                    start_idx = j * num_of_classes
                    end_idx = (j+1) * num_of_classes
                    
                    temp_oof_train_X[:, start_idx : end_idx] = temp_oof_predictions
                    temp_oof_test_X[:, start_idx : end_idx] = learner.predict_proba(temp_test_X)
                    ''' ====== Temp level-N train & test sets stacking (Multi-Class) ======= '''
                    
                    y_pred_labels = np.argmax(temp_oof_predictions, axis=1)
                    temp_accuracy_modelwise[j] = accuracy_score(y_train, y_pred_labels)
                
                ''' =========== Out-of-fold prediction for level-N for all learners ========== '''
                    
                ''' ======== Blending with input feature set ======== '''
                temp_oof_train_X[:, -1 * num_basefeats:] = temp_train_X
                temp_oof_test_X[:, -1 * num_basefeats:] = temp_test_X
                ''' ======== Blending with input feature set ======== '''
                    
                ''' ================================== Final feature selection ================================== '''
                initial_feature_size = temp_oof_train_X.shape[1]
                
                if((level369 == True) and (iffeatselection_or_not == True)): 
                    
                    if((level % 3 == 0) and (level > 0)):
   
                        if(feat_selection_type == 'SFE'):
                            UR = 0.3
                            UR_Max = 0.3
                            UR_Min = 0.001
                            Max_FEs = 200
                            Max_Run = 5
                            Run = 1
                            Cost = np.zeros([Max_FEs, Max_Run])
                            
                            selected_feaure_indices = self.SFE_run(Input=temp_oof_train_X, Target=y_train, UR=UR, UR_Max=UR_Max, UR_Min=UR_Min, Max_FEs=Max_FEs, Max_Run=Max_Run, Run=Run, Cost=Cost)
                            
                            temp_oof_train_X, temp_oof_test_X = temp_oof_train_X[:, selected_feaure_indices], temp_oof_test_X[:, selected_feaure_indices]
                            
                            print('SFE feature selection of level %d done successfully' % level)
                            print('Selected number of features are F=%d out of %d' % (len(selected_feaure_indices), initial_feature_size))
                            
                        elif(feat_selection_type == 'autoencoder_2layers'): 
                            temp_oof_train_X, temp_oof_test_X = self.reduce_dimensionality_with_autoencoder(temp_oof_train_X, temp_oof_test_X, encoding_dim=int(initial_feature_size / 3))
                            
                            print('Autoencoder x2 feature selection of level %d done successfully' % level)
                                                                    
                        elif(feat_selection_type == 'autoencoder_4layers'): 
                            temp_oof_train_X, temp_oof_test_X = self.reduce_dimensionality_with_autoencoder2(temp_oof_train_X, temp_oof_test_X)
                            
                            print('Autoencoder x4 feature selection of level %d done successfully' % level)
                            
                        elif(feat_selection_type == 'attentionlayer'):
                            temp_oof_train_X, temp_oof_test_X = self.attention_feature_selector_mc(temp_oof_train_X, y_train, temp_oof_test_X, percentile=25, epochs=50, batch_size=32)
       
                elif((level369 == False) and (iffeatselection_or_not == True)): 
                    
                    if(level > 0):
                        if(feat_selection_type == 'SFE'):
                            UR = 0.3
                            UR_Max = 0.3
                            UR_Min = 0.001
                            Max_FEs = 200
                            Max_Run = 5
                            Run = 1
                            Cost = np.zeros([Max_FEs, Max_Run])
                            
                            selected_feaure_indices = self.SFE_run(Input=temp_oof_train_X, Target=y_train, UR=UR, UR_Max=UR_Max, UR_Min=UR_Min, Max_FEs=Max_FEs, Max_Run=Max_Run, Run=Run, Cost=Cost)
                            
                            temp_oof_train_X, temp_oof_test_X = temp_oof_train_X[:, selected_feaure_indices], temp_oof_test_X[:, selected_feaure_indices]
                            
                            print('SFE feature selection of level %d done successfully' % level)
                            print('Selected number of features are F=%d out of %d' % (len(selected_feaure_indices), initial_feature_size))
                            
                        elif(feat_selection_type == 'autoencoder_2layers'): 
                            temp_oof_train_X, temp_oof_test_X = self.reduce_dimensionality_with_autoencoder(temp_oof_train_X, temp_oof_test_X, encoding_dim=int(initial_feature_size / 3))
                            print('Autoencoder x2 feature selection of level %d done successfully' % level)
                                                                    
                        elif(feat_selection_type == 'autoencoder_4layers'): 
                            temp_oof_train_X, temp_oof_test_X = self.reduce_dimensionality_with_autoencoder2(temp_oof_train_X, temp_oof_test_X)
                            print('Autoencoder x4 feature selection of level %d done successfully' % level)
                            
                        elif(feat_selection_type == 'attentionlayer'):
                            temp_oof_train_X, temp_oof_test_X = self.attention_feature_selector_mc(temp_oof_train_X, y_train, temp_oof_test_X, percentile=25, epochs=50, batch_size=32)
    
                ''' ================================== Final feature selection ================================== '''
    
                print('If NAN in train data is %s' % np.isnan(temp_oof_train_X).any())
                print('If NAN in test data is %s' % np.isnan(temp_oof_test_X).any())
                
                until_meta_learning = time.time()
                
                accumulated_just_meta_learnings_for_this_fold = np.sum(elapsedtimes_blendensemble_just_meta_learning[n_fold, 0 : level, :])
                elapsedtimes_blendensemble_until_meta_learning[n_fold, level] = until_meta_learning - (begin_blending + accumulated_just_meta_learnings_for_this_fold)
                
                ''' ======== Meta-model evaluation ========= '''
                for ifselected, selected_indice in zip(selected_modelindices_levelN_truefalse, range(num_of_models)):
                    
                    if(ifselected == True):
                        
                        print('meta-model evaluation with selected indice of %d' % selected_indice)
                        
                        final_estimator = clone(current_fold_stacking_models[selected_indice][1]) #initialize for each fold using Optimized
                        
                        begin_metalearning = time.time()
                        final_estimator.fit(temp_oof_train_X, y_train)
                        end_metalearning = time.time()
                                                 
                        elapsedtimes_blendensemble_levels[n_fold, level, selected_indice] = (until_meta_learning - begin_blending) + (end_metalearning - begin_metalearning) 
                        elapsedtimes_blendensemble_just_meta_learning[n_fold, level, selected_indice] = end_metalearning - begin_metalearning
                        
                        final_predictions = final_estimator.predict(temp_oof_test_X)
                        final_prediction_probs = final_estimator.predict_proba(temp_oof_test_X)
                                                 
                        ''' ========== Temp results logging ========== '''     
                        accuracies_blnd[n_fold, level, selected_indice] = accuracy_score(y_test, final_predictions)
                        f1scores_blnd[n_fold, level, selected_indice] = f1_score(y_test, final_predictions, average=average)
                        precisions_blnd[n_fold, level, selected_indice] = precision_score(y_test, final_predictions, average=average)
                        recalls_blnd[n_fold, level, selected_indice] = recall_score(y_test, final_predictions, average=average)
                        log_lossess_blnd[n_fold, level, selected_indice] = log_loss(y_test, final_prediction_probs)
                        ''' ========== Temp results logging ========== '''
    
                ''' ======== Meta-model evaluation ========= '''
                
                ''' ================================== Ensemble Level-N model elimination ========================== '''
                
                print('Model elimination process has begun...')
                
                threshold_percentile_init = 10
                
                # --- Compute range and apply strength factor ---
                range_val = np.max(temp_accuracy_modelwise) - np.min(temp_accuracy_modelwise)
                if blur_strength == 'light':
                    noise_scale = 0.05 * range_val
                elif blur_strength == 'moderate':
                    noise_scale = 0.10 * range_val
                else:
                    raise ValueError("Invalid blur_strength")
                
                print(f'[Blurrization] Level {level+1}: noise_scale = {noise_scale:.6f} (range-based, strength = {blur_strength})')
                
                # --- Inject Gaussian noise to blur OOF-based scores ---
                noisy_accuracy_modelwise = temp_accuracy_modelwise + np.random.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=temp_accuracy_modelwise.shape
                )
                
                # --- Determine new percentile threshold based on noisy scores ---
                temp_std = np.std(noisy_accuracy_modelwise)
                new_threshold_percentile = threshold_percentile_init / 2 + (temp_std ** 2) * 80
                Qcustom = np.percentile(noisy_accuracy_modelwise, new_threshold_percentile)
                
                # --- Prune based on blurred threshold ---
                selected_modelindices_levelN_previously = np.argwhere(selected_modelindices_levelN_truefalse == True)[:, 0]
                selected_modelindices_levelN = np.argwhere(noisy_accuracy_modelwise >= Qcustom)[:, 0]
                indices_to_be_selected = selected_modelindices_levelN_previously[selected_modelindices_levelN]
                
                selected_modelindices_levelN_truefalse = np.zeros(len(selected_modelindices_levelN_truefalse), dtype=bool)
                selected_modelindices_levelN_truefalse[indices_to_be_selected] = True
                
                # --- Log selected models ---
                numbers_of_selected_models[level] = len(selected_modelindices_levelN)
                print('The number of selected models in level %d is %d' % (level + 1, numbers_of_selected_models[level]))
                
                selected_modelindices_levelN_currently = np.argwhere(selected_modelindices_levelN_truefalse == True)[:, 0]
                selected_models_in_each_level[n_fold][level] = [model_names[indice] for indice in selected_modelindices_levelN_currently]
                
                # --- Check exit condition or continue stacking ---
                if numbers_of_selected_models[level] < left_model_threshold:
                    exit_flag = True
                    break
                else:
                    temp_stacking_models = [current_fold_stacking_models[indice] for indice in selected_modelindices_levelN_currently]
                ''' ================================== Ensemble Level-N model elimination ========================== '''
                
                ''' ===== approximating total time required ==== '''
                end_innermostloop = time.time()
                total_time_for_innermostloop = end_innermostloop - begin_innermostloop
                print('Total approximate time required is %d seconds' % (total_time_for_innermostloop * 100))
                ''' ===== approximating total time required ==== '''
                
                ''' ======= Recursively change the initial train / test data blended --> input data ====== '''
                if(stackoverstacking == True):
                    temp_train_X, temp_test_X = np.column_stack((temp_train_X, temp_oof_train_X)).copy(), np.column_stack((temp_test_X, temp_oof_test_X)).copy()
                else:
                    temp_train_X, temp_test_X = temp_oof_train_X.copy(), temp_oof_test_X.copy()
    
                number_of_feautures_blendensemble_levels[n_fold, level] = np.shape(temp_train_X)[1]
                ''' ======= Recursively change the initial train / test data blended --> input data ====== '''
                
                ''' ======= Stack-of-stacking ======== '''
                if(stackoverstacking == False):
                    temp_stackofstack_train_X, temp_stackofstack_test_X = np.column_stack((temp_stackofstack_train_X, temp_train_X)), np.column_stack((temp_stackofstack_test_X, temp_test_X))
                ''' ======= Stack-of-stacking ======== '''
                
            ''' ========= Stack-of-stacking evaluation ========== '''
            if(stackoverstacking == False):
                
                for selected_indice in range(num_of_models):
                                            
                    final_estimator = clone(current_fold_stacking_models[selected_indice][1]) #initialize for each fold using Optimized
                    
                    end_blending = time.time()
                    
                    begin_metalearning = time.time()
                    final_estimator.fit(temp_stackofstack_train_X, y_train)
                    end_metalearning = time.time()
                                         
                    elapsedtimes_stackofstacking[n_fold, selected_indice] = (end_blending - begin_blending) + (end_metalearning - begin_metalearning) 
                    
                    final_predictions = final_estimator.predict(temp_stackofstack_test_X)
                    final_prediction_probs = final_estimator.predict_proba(temp_stackofstack_test_X)
                    
                    ''' ========== Temp results logging ========== '''
                    accuracies_stackofstack[n_fold, selected_indice] = accuracy_score(y_test, final_predictions)
                    f1scores_stackofstack[n_fold, selected_indice] = f1_score(y_test, final_predictions, average=average)
                    precisions_stackofstack[n_fold, selected_indice] = precision_score(y_test, final_predictions, average=average)
                    recalls_stackofstack[n_fold, selected_indice] = recall_score(y_test, final_predictions, average=average)
                    log_lossess_stackofstack[n_fold, selected_indice] = log_loss(y_test, final_prediction_probs)
                    ''' ========== Temp results logging ========== '''
                    
            ''' ========= Stack-of-stacking evaluation ========== '''
        
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        
        '''
        What has been happening here:
        1) this is elapsed time handler section of the function
        2) I am trying to restructure the results for more meaningful storing as dictionaries but need re-ordering the mess
        3) I want with and without normalized versions of:
         3.1) fold-wise individual model elapsed times + each blend level elapsed time
         3.2) averaged of those folds as another dictionary
        4) stack-of-stacking results as separate dictionary without normalization
        '''
        
        # elapsedtimes_blendensemble_levels = np.random.rand(num_of_folds, num_of_level, num_of_models)
      
        ''' ==================== Initial summation operation ==================== '''
        # elapsedtimes_stackofstacking --> num_of_folds, num_of_models
        # elapsedtimes_blendensemble_levels --> num_of_folds, num_of_level, num_of_models
        # elapsedtimes_individual --> num_of_models
        
        elapsedtimes_blendensemble_levels_sum = np.sum(elapsedtimes_blendensemble_levels, axis=(0,2))
        elapsedtimes_blendensemble_levels_sum_perfold = np.sum(elapsedtimes_blendensemble_levels, axis=(2))
      
        elapsedtimes_stackofstacking_sum = np.sum(elapsedtimes_stackofstacking, axis=(0,1))
        ''' ==================== Initial summation operation ==================== '''
      
        ''' ============================================== Without stack-of-stacking ================================================== '''
        ''' ============================================== Without stack-of-stacking ================================================== '''
      
        ''' =========== Elapsed time without normalization and without stack-of-stacking =========== '''
        elapsedtimes_individual_withoutnorm = elapsedtimes_individual
        elapsedtimes_blendensemble_levels_withoutnorm = elapsedtimes_blendensemble_levels_sum
        elapsedtimes_blendensemble_levels_withoutnorm_perfold = elapsedtimes_blendensemble_levels_sum_perfold
        ''' =========== Elapsed time without normalization and without stack-of-stacking =========== '''
        
        ''' =========== Elapsed time normalizer without stack-of-stacking =========== '''
        maximum_elapsed_time = max(np.max(elapsedtimes_individual), np.max(elapsedtimes_blendensemble_levels_sum))
      
        elapsedtimes_individual_norm = elapsedtimes_individual / maximum_elapsed_time
        elapsedtimes_blendensemble_levels_norm = elapsedtimes_blendensemble_levels_sum / maximum_elapsed_time
        
        ''' ======== Special per fold treatment ======== '''
        elapsedtimes_individual_norm_perfold = np.zeros((5,14))
        elapsedtimes_blendensemble_levels_norm_perfold = np.zeros((5,10))
        for fold in range(5):
            maximum_elapsed_time_perfold = max(np.max(elapsedtimes_individual / 5), np.max(elapsedtimes_blendensemble_levels_sum_perfold[fold]))
            elapsedtimes_individual_norm_perfold[fold] = (elapsedtimes_individual / 5) / maximum_elapsed_time_perfold
            elapsedtimes_blendensemble_levels_norm_perfold[fold] = elapsedtimes_blendensemble_levels_sum_perfold[fold] / maximum_elapsed_time_perfold
        ''' ======== Special per fold treatment ======== '''
        
        ''' =========== Elapsed time normalizer without stack-of-stacking =========== '''
        
        ''' =========== Elapsed time without stack-of-stacking: numpy --> dict ============ '''
        elapsedtimes_individual_norm_dict, elapsedtimes_blendensemble_levels_norm_dict = {}, {}
        elapsedtimes_individual_norm_perfold_dict, elapsedtimes_blendensemble_levels_norm_perfold_dict = {}, {}
        elapsedtimes_individual_withoutnorm_dict, elapsedtimes_blendensemble_levels_withoutnorm_dict = {}, {}
      
        model_names = model_names_list 
        for i in range(len(model_names)):
            elapsedtimes_individual_norm_dict[model_names[i]] = elapsedtimes_individual_norm[i]
            elapsedtimes_individual_withoutnorm_dict[model_names[i]] = elapsedtimes_individual_withoutnorm[i]
      
        for i in range(num_of_level):
            elapsedtimes_blendensemble_levels_norm_dict['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_norm[i]
      
            elapsedtimes_blendensemble_levels_withoutnorm_dict['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_withoutnorm[i]
                
        ''' ======== Special per fold treatment from numpy to dict ======== '''
        for fold in range(5):
            temp_elapsedtimes_blendensemble_levels_norm_fold_dict = dict()
            temp_elapsedtimes_individual_norm_fold_dict = dict()
      
            for i in range(num_of_level):
                temp_elapsedtimes_blendensemble_levels_norm_fold_dict['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_norm_perfold[fold,i]
                
                temp_elapsedtimes_individual_norm_fold_dict[model_names[i]] = elapsedtimes_individual_norm_perfold[fold, i]
      
                # temp_elapsedtimes_individual_norm_fold_dict['_level_' + str(i+1)] = elapsedtimes_individual_norm_perfold[fold,i]
      
            elapsedtimes_blendensemble_levels_norm_perfold_dict['fold_' + str(fold)] = temp_elapsedtimes_blendensemble_levels_norm_fold_dict
            elapsedtimes_individual_norm_perfold_dict['fold_' + str(fold)] = temp_elapsedtimes_individual_norm_fold_dict
        ''' ======== Special per fold treatment from numpy to dict ======== '''
      
        ''' =========== Elapsed time without stack-of-stacking: numpy --> dict ============ '''
        
        ''' ============================================== Without stack-of-stacking ================================================== '''
        ''' ============================================== Without stack-of-stacking ================================================== '''
      
        ''' ============================================== With stack-of-stacking ================================================== '''
        ''' ============================================== With stack-of-stacking ================================================== '''
      
        ''' =========== Elapsed time normalizer with stack-of-stacking =========== '''
        maximum_elapsed_time2 = max(np.max(elapsedtimes_individual), np.max(elapsedtimes_stackofstacking_sum))
        elapsedtimes_individual_norm2 = elapsedtimes_individual / maximum_elapsed_time2
        elapsedtimes_blendensemble_levels_norm2 = elapsedtimes_blendensemble_levels_sum / maximum_elapsed_time2
        elapsedtimes_stackofstacking_norm = elapsedtimes_stackofstacking_sum / maximum_elapsed_time2
        ''' =========== Elapsed time normalizer with stack-of-stacking =========== '''
        
        ''' =========== Elapsed time with stack-of-stacking: numpy --> dict ============ '''
        elapsedtimes_individual_norm_dict2, elapsedtimes_blendensemble_levels_norm_dict2, elapsedtimes_stackofstacking_norm_dict = {}, {}, {}
              
        for i in range(len(model_names)):
            elapsedtimes_individual_norm_dict2[model_names[i]] = elapsedtimes_individual_norm2[i]
            
        for i in range(num_of_level):
            elapsedtimes_blendensemble_levels_norm_dict2['_level_' + str(i+1)] = elapsedtimes_blendensemble_levels_norm2[i]
                
        elapsedtimes_stackofstacking_norm_dict = elapsedtimes_stackofstacking_norm
        ''' =========== Elapsed time with stack-of-stacking: numpy --> dict ============ '''
        
        ''' ============================================== With stack-of-stacking ================================================== '''
        ''' ============================================== With stack-of-stacking ================================================== '''
      
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        ''' ================================================================ Elapsed time handler  ================================================================== '''
        
        ''' ===================== Return results ==================== '''
        
        # ============= Averaging & std non-zeros =============
        accuracies_blnd_nan = np.where(accuracies_blnd == 0, np.nan, accuracies_blnd)
        accuracies_blnd_mean = np.nanmean(accuracies_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        accuracies_blnd_std = np.nanstd(accuracies_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values
        
        f1scores_blnd_nan = np.where(f1scores_blnd == 0, np.nan, f1scores_blnd)
        f1scores_blnd_mean = np.nanmean(f1scores_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        f1scores_blnd_std = np.nanstd(f1scores_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values
        
        precisions_blnd_nan = np.where(precisions_blnd == 0, np.nan, precisions_blnd)
        precisions_blnd_mean = np.nanmean(precisions_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        precisions_blnd_std = np.nanstd(precisions_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values
        
        recalls_blnd_nan = np.where(recalls_blnd == 0, np.nan, recalls_blnd)
        recalls_blnd_mean = np.nanmean(recalls_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        recalls_blnd_std = np.nanstd(recalls_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values
        
        log_lossess_blnd_nan = np.where(log_lossess_blnd == 0, np.nan, log_lossess_blnd)
        log_lossess_blnd_mean = np.nanmean(log_lossess_blnd_nan, axis=0) # Compute the mean along the last dimension, ignoring NaN values
        log_lossess_blnd_std = np.nanstd(log_lossess_blnd_nan, axis=0) # Compute the std along the last dimension, ignoring NaN values
        # ============= Averaging & std non-zeros =============
        
        # ============= Averaging & std of stack of stacking =============
        accuracies_stackofstack_mean = np.mean(accuracies_stackofstack, axis=0)
        accuracies_stackofstack_std = np.std(accuracies_stackofstack, axis=0)
        
        precisions_stackofstack_mean = np.mean(precisions_stackofstack, axis=0)
        precisions_stackofstack_std = np.std(precisions_stackofstack, axis=0)
        
        f1scores_stackofstack_mean = np.mean(f1scores_stackofstack, axis=0)
        f1scores_stackofstack_std = np.std(f1scores_stackofstack, axis=0)
        
        recalls_stackofstack_mean = np.mean(recalls_stackofstack, axis=0)
        recalls_stackofstack_std = np.std(recalls_stackofstack, axis=0)
        
        log_lossess_stackofstack_mean = np.mean(log_lossess_stackofstack, axis=0)
        log_lossess_stackofstack_std = np.std(log_lossess_stackofstack, axis=0)
        # ============= Averaging & std of stack of stacking =============
            
        all_results = {'individual_results': {'accuracy_ind': accuracies_ind, 'f1scores_ind': f1scores_ind, 'precisions_ind': precisions_ind,
                                             'recalls_ind': recalls_ind, 'log_lossess_ind': log_lossess_ind}, \
                      'individual_avg_results': {'avg_accuracy_ind': np.mean(accuracies_ind, axis=1), 'std_accuracy_ind': np.std(accuracies_ind, axis=1), 
                                                 'avg_f1scores_ind': np.mean(f1scores_ind, axis=1), 'avg_precisions_ind': np.mean(precisions_ind, axis=1),
                                                 'avg_recalls_ind': np.mean(recalls_ind, axis=1) , 'avg_log_lossess_ind': np.mean(log_lossess_ind, axis=1)}, \
                               
                      'blendensemble_results': {'accuracy_blnd': accuracies_blnd, 'f1scores_blnd': f1scores_blnd,
                                                 'precisions_blnd': precisions_blnd, 'recalls_blnd': recalls_blnd,
                                                 'log_losses_blnd': log_lossess_blnd}, \
                      'blendensemble_avg_results': {'avg_accuracy_blnd': accuracies_blnd_mean , 'std_accuracy_blnd': accuracies_blnd_std,
                                                 'avg_f1scores_blnd': f1scores_blnd_mean, 'std_f1scores_blnd': f1scores_blnd_std,
                                                 'avg_precisions_blnd': precisions_blnd_mean, 'std_precisions_blnd': precisions_blnd_std,
                                                 'avg_recalls_blnd': recalls_blnd_mean, 'std_recalls_blnd': recalls_blnd_std,
                                                 'avg_log_losses_blnd': log_lossess_blnd_mean, 'std_log_losses_blnd': log_lossess_blnd_std}, \
                        
                      'stackofstacking_results': {'accuracies_stackofstack': accuracies_stackofstack, 'f1scores_stackofstack': f1scores_stackofstack,
                                                 'precisions_stackofstack': precisions_stackofstack, 'recalls_stackofstack': recalls_stackofstack,
                                                 'log_lossess_stackofstack': log_lossess_stackofstack}, \
                      'stackofstacking_avg_results': {'accuracies_stackofstack_mean': accuracies_stackofstack_mean , 'accuracies_stackofstack_std': accuracies_stackofstack_std,
                                                 'f1scores_stackofstack_mean': f1scores_stackofstack_mean, 'f1scores_stackofstack_std': f1scores_stackofstack_std,
                                                 'precisions_stackofstack_mean': precisions_stackofstack_mean, 'precisions_stackofstack_std': precisions_stackofstack_std,
                                                 'recalls_stackofstack_mean': recalls_stackofstack_mean, 'recalls_stackofstack_std': recalls_stackofstack_std,
                                                 'log_lossess_stackofstack_mean': log_lossess_stackofstack_mean, 'log_lossess_stackofstack_std': log_lossess_stackofstack_std}, \
                            
                      'elapsedtimes_withoutstackofstacking_normed': {'individual_models_normed': elapsedtimes_individual_norm_dict, 'blend_ensemble_level1-10_normed': elapsedtimes_blendensemble_levels_norm_dict}, \
                      'elapsedtimes_withoutstackofstacking_normed_perfold': {'individual_models_normed_perfold': elapsedtimes_individual_norm_perfold_dict, 
                                                                             'blend_ensemble_level1-10_normed_perfold': elapsedtimes_blendensemble_levels_norm_perfold_dict}, \
   
                      'elapsedtimes_withoutstackofstacking_withoutnormed': {'individual_models_normed': elapsedtimes_individual_withoutnorm_dict, 
                                                                            'blend_ensemble_level1-10_normed': elapsedtimes_blendensemble_levels_withoutnorm_dict}, \
                      'elapsedtimes_withstackofstacking': {'individual_models2': elapsedtimes_individual_norm_dict2, 'blend_ensemble_level1-10': elapsedtimes_blendensemble_levels_norm_dict2,
                                                           'elapsedtimes_stackofstacking_norm_dict': elapsedtimes_stackofstacking_norm_dict}, \
                      'other information': {'number_of_feautures_blendensemble_levels': number_of_feautures_blendensemble_levels, 
                                            'bare_elapsedtimes_blendensemble_levels': elapsedtimes_blendensemble_levels, 
                                            'elapsedtimes_blendensemble_until_meta_learning': elapsedtimes_blendensemble_until_meta_learning,
                                            'elapsedtimes_blendensemble_just_meta_learning': elapsedtimes_blendensemble_just_meta_learning}
                              }
        
        return all_results        
        
    # ==================================================================================================
    # Trainable / sklearn-like RocketStack wrappers
    # ==================================================================================================
    class _ColumnSelector:
        """Small fitted transformer used to preserve SFE/attention-selected columns."""
        def __init__(self, selected_indices):
            self.selected_indices = np.asarray(selected_indices, dtype=int)

        def transform(self, X):
            return np.asarray(X)[:, self.selected_indices]

    class _KerasEncoderTransformer:
        """Fitted autoencoder encoder + scaler, so new data can be transformed after training."""
        def __init__(self, scaler, encoder):
            self.scaler = scaler
            self.encoder = encoder

        def transform(self, X):
            X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
            return self.encoder.predict(X_scaled, verbose=0)

    class _RocketStackTrainableModel:
        """
        Sklearn-like fitted RocketStack object.

        The object stores, for each recursive level:
          1) fitted base learners used to generate level-N probability features,
          2) the fitted feature selector/reducer, if any,
          3) the best OOF-scoring meta-learner fitted on the full level-N training representation.

        Typical use:
            rs = RocketStack()
            model = rs.AscentTheRocket_binary_model(level=5, return_model='best')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        """

        def __init__(self, parent, problem_type='binary', num_of_level=10, return_model='best',
                     iffeatselection=True, feat_selection_type='SFE', feature_selection_levels=(3, 6, 9),
                     stackoverstacking=False, indices_to_normalized=None, blur_strength='light',
                     left_model_threshold=2, meta_scoring=None, cv_splits=5, random_state=42,
                     n_jobs=-1, autoencoder_epochs=50, autoencoder_batch_size=32,
                     attention_epochs=50, attention_batch_size=32, verbose=True,
                     hyperparameterOptimization=False, n_trials=30):
            if return_model not in ('best', 'all'):
                raise ValueError("return_model must be either 'best' or 'all'.")
            if problem_type not in ('binary', 'multiclass'):
                raise ValueError("problem_type must be either 'binary' or 'multiclass'.")
            if hyperparameterOptimization:
                raise NotImplementedError(
                    "The trainable model wrapper currently supports the default RocketStack model pool. "
                    "Hyperparameter optimization can be added later, but is intentionally disabled here "
                    "to keep the stored train/test pipeline stable and reproducible."
                )

            self.parent = parent
            self.problem_type = problem_type
            self.num_of_level = int(num_of_level)
            self.return_model = return_model
            self.iffeatselection = bool(iffeatselection)
            self.feat_selection_type = feat_selection_type
            self.feature_selection_levels = feature_selection_levels
            self.stackoverstacking = bool(stackoverstacking)
            self.indices_to_normalized = indices_to_normalized
            self.blur_strength = blur_strength
            self.left_model_threshold = int(left_model_threshold)
            self.meta_scoring = meta_scoring
            self.cv_splits = int(cv_splits)
            self.random_state = int(random_state)
            self.n_jobs = n_jobs
            self.autoencoder_epochs = int(autoencoder_epochs)
            self.autoencoder_batch_size = int(autoencoder_batch_size)
            self.attention_epochs = int(attention_epochs)
            self.attention_batch_size = int(attention_batch_size)
            self.verbose = bool(verbose)
            self.hyperparameterOptimization = hyperparameterOptimization
            self.n_trials = int(n_trials)

            self.is_fitted_ = False
            self.levels_ = []
            self.model_names_list_ = None
            self.defaults_ = None
            self.classes_ = None
            self.n_classes_ = None
            self.initial_scaler_ = None
            self.best_level_ = None
            self.best_model_name_ = None
            self.best_oof_score_ = None
            self.effective_num_levels_ = 0

        # ------------------------- public sklearn-like API -------------------------
        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            if X.ndim != 2:
                raise ValueError("X must be a 2D array-like object.")
            if len(X) != len(y):
                raise ValueError("X and y must contain the same number of samples.")

            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            if self.problem_type == 'binary' and self.n_classes_ != 2:
                raise ValueError("AscentTheRocket_binary_model requires exactly two classes.")
            if self.problem_type == 'multiclass' and self.n_classes_ < 2:
                raise ValueError("AscentTheRocket_multiclass_model requires at least two classes.")

            self.model_names_list_, self.defaults_ = self._make_model_pool()
            if self.meta_scoring is None:
                self.meta_scoring = 'roc_auc' if self.problem_type == 'binary' else 'accuracy'

            np.random.seed(self.random_state)
            rng = np.random.default_rng(self.random_state)
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

            X_current = self._prepare_initial_X_fit(X)
            X_stack = X_current.copy()
            active_indices = list(range(len(self.model_names_list_)))
            self.levels_ = []

            for level_idx in range(self.num_of_level):
                level_number = level_idx + 1
                if self.verbose:
                    print(f"[RocketStack-{self.problem_type}] Fitting level {level_number}/{self.num_of_level} with {len(active_indices)} active models")

                X_level_raw, fitted_base_models, modelwise_scores = self._make_level_features_fit(
                    X_current=X_current,
                    y=y,
                    active_indices=active_indices,
                    cv=cv
                )

                feature_transformer = self._fit_feature_transformer(
                    X_level_raw,
                    y,
                    level_number=level_number
                )
                X_level = feature_transformer.transform(X_level_raw) if feature_transformer is not None else X_level_raw

                if self.stackoverstacking:
                    X_meta_input = np.column_stack((X_current, X_level))
                else:
                    X_meta_input = X_level

                meta_scores = self._score_meta_candidates(X_meta_input, y, active_indices, cv)
                best_global_index = max(meta_scores, key=meta_scores.get)
                best_model_name = self.model_names_list_[best_global_index]

                meta_estimators = {}
                if self.return_model == 'all':
                    for global_index in active_indices:
                        name = self.model_names_list_[global_index]
                        meta_estimators[name] = clone(self.defaults_[name]).fit(X_meta_input, y)
                    best_meta_estimator = meta_estimators[best_model_name]
                else:
                    best_meta_estimator = clone(self.defaults_[best_model_name]).fit(X_meta_input, y)

                selected_next_indices, noisy_scores, threshold = self._prune_active_models(
                    active_indices=active_indices,
                    modelwise_scores=modelwise_scores,
                    rng=rng
                )

                level_info = {
                    'level': level_number,
                    'input_n_features': X_current.shape[1],
                    'output_n_features': X_meta_input.shape[1],
                    'active_model_indices': active_indices.copy(),
                    'active_model_names': [self.model_names_list_[i] for i in active_indices],
                    'fitted_base_models': fitted_base_models,
                    'feature_transformer': feature_transformer,
                    'stackoverstacking': self.stackoverstacking,
                    'best_meta_model_index': best_global_index,
                    'best_meta_model_name': best_model_name,
                    'best_oof_score': meta_scores[best_global_index],
                    'best_meta_estimator': best_meta_estimator,
                    'meta_oof_scores': {self.model_names_list_[k]: v for k, v in meta_scores.items()},
                    'meta_estimators': meta_estimators if self.return_model == 'all' else None,
                    'modelwise_oof_scores': {self.model_names_list_[active_indices[i]]: float(modelwise_scores[i])
                                             for i in range(len(active_indices))},
                    'modelwise_noisy_scores': {self.model_names_list_[active_indices[i]]: float(noisy_scores[i])
                                               for i in range(len(active_indices))},
                    'pruning_threshold': float(threshold),
                    'selected_next_model_indices': selected_next_indices.copy(),
                    'selected_next_model_names': [self.model_names_list_[i] for i in selected_next_indices]
                }
                self.levels_.append(level_info)

                if self.verbose:
                    print(f"[RocketStack-{self.problem_type}] Level {level_number}: best meta-model = {best_model_name}, OOF score = {meta_scores[best_global_index]:.6f}")
                    print(f"[RocketStack-{self.problem_type}] Level {level_number}: retained {len(selected_next_indices)} models for next level")

                # Stop if pruning leaves too few models for a meaningful next recursive level.
                if len(selected_next_indices) < self.left_model_threshold:
                    if self.verbose:
                        print(f"[RocketStack-{self.problem_type}] Early stop after level {level_number}: fewer than left_model_threshold models remain")
                    break

                if self.stackoverstacking:
                    X_current = np.column_stack((X_current, X_level)).copy()
                else:
                    X_current = X_level.copy()

                if not self.stackoverstacking:
                    X_stack = np.column_stack((X_stack, X_current))

                active_indices = selected_next_indices.copy()

            self.effective_num_levels_ = len(self.levels_)
            self.best_level_ = self.effective_num_levels_
            final_info = self.levels_[self.best_level_ - 1]
            self.best_model_name_ = final_info['best_meta_model_name']
            self.best_oof_score_ = final_info['best_oof_score']
            self.is_fitted_ = True
            return self

        def predict(self, X, level=None, model_name=None):
            estimator, X_level = self._prepare_prediction_input(X, level=level, model_name=model_name)
            return estimator.predict(X_level)

        def predict_proba(self, X, level=None, model_name=None):
            estimator, X_level = self._prepare_prediction_input(X, level=level, model_name=model_name)
            if not hasattr(estimator, 'predict_proba'):
                raise AttributeError(f"Selected estimator {estimator.__class__.__name__} does not provide predict_proba().")
            return estimator.predict_proba(X_level)

        def score(self, X, y, metric='accuracy', level=None, model_name=None):
            y = np.asarray(y)
            if metric == 'accuracy':
                return accuracy_score(y, self.predict(X, level=level, model_name=model_name))
            if metric == 'f1':
                average = 'binary' if self.problem_type == 'binary' else 'weighted'
                return f1_score(y, self.predict(X, level=level, model_name=model_name), average=average)
            if metric == 'precision':
                average = 'binary' if self.problem_type == 'binary' else 'weighted'
                return precision_score(y, self.predict(X, level=level, model_name=model_name), average=average, zero_division=0)
            if metric == 'recall':
                average = 'binary' if self.problem_type == 'binary' else 'weighted'
                return recall_score(y, self.predict(X, level=level, model_name=model_name), average=average)
            if metric == 'log_loss':
                return log_loss(y, self.predict_proba(X, level=level, model_name=model_name))
            if metric in ('roc_auc', 'auc'):
                probs = self.predict_proba(X, level=level, model_name=model_name)
                if self.problem_type == 'binary':
                    return roc_auc_score(y, probs[:, 1])
                return roc_auc_score(y, probs, multi_class='ovr')
            raise ValueError("metric must be one of: accuracy, f1, precision, recall, log_loss, roc_auc")

        def get_level_summary(self):
            if not self.is_fitted_:
                raise RuntimeError("Call fit() before get_level_summary().")
            return [
                {
                    'level': info['level'],
                    'best_meta_model_name': info['best_meta_model_name'],
                    'best_oof_score': info['best_oof_score'],
                    'active_model_names': info['active_model_names'],
                    'selected_next_model_names': info['selected_next_model_names'],
                    'input_n_features': info['input_n_features'],
                    'output_n_features': info['output_n_features']
                }
                for info in self.levels_
            ]

        # ------------------------- internals -------------------------
        def _make_model_pool(self):
            if self.problem_type == 'binary':
                model_names_list = ['xgb', 'lgbm', 'randforest', 'ctbclf', 'sgd', 'svc', 'bagging', 'adaboost', 'knn',
                                    'lda', 'gpc', 'extra_trees', 'gradient_boosting', 'calibrated_ridge',
                                    'logistic_regression', 'calibrated_passive_aggressive', 'bernoulli_nb',
                                    'gaussian_nb', 'mlp', 'hist_gradient_boosting']
                defaults = {
                    'xgb': XGBClassifier(n_jobs=-1, random_state=self.random_state, verbosity=0),
                    'lgbm': lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=self.random_state),
                    'randforest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.random_state),
                    'ctbclf': ctb.CatBoostClassifier(silent=True, thread_count=-1, random_seed=self.random_state),
                    'sgd': SGDClassifier(loss='log_loss', n_jobs=-1, random_state=self.random_state),
                    'svc': SVC(probability=True, random_state=self.random_state),
                    'bagging': BaggingClassifier(n_jobs=-1, random_state=self.random_state),
                    'adaboost': AdaBoostClassifier(algorithm='SAMME', random_state=self.random_state),
                    'knn': KNeighborsClassifier(n_jobs=-1),
                    'lda': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                    'gpc': GaussianProcessClassifier(n_jobs=-1, random_state=self.random_state),
                    'extra_trees': ExtraTreesClassifier(n_jobs=-1, random_state=self.random_state),
                    'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
                    'calibrated_ridge': CalibratedClassifierCV(RidgeClassifier(), cv=5),
                    'logistic_regression': LogisticRegression(n_jobs=-1, max_iter=1000),
                    'calibrated_passive_aggressive': CalibratedClassifierCV(PassiveAggressiveClassifier(), cv=5),
                    'bernoulli_nb': BernoulliNB(),
                    'gaussian_nb': GaussianNB(),
                    'mlp': MLPClassifier(random_state=self.random_state),
                    'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=self.random_state)
                }
            else:
                model_names_list = ['xgb', 'lgbm', 'randforest', 'svc', 'bagging', 'adaboost', 'knn',
                                    'extra_trees', 'gradient_boosting', 'logistic_regression',
                                    'bernoulli_nb', 'gaussian_nb', 'mlp', 'hist_gradient_boosting']
                defaults = {
                    'xgb': XGBClassifier(n_jobs=-1, random_state=self.random_state, verbosity=0),
                    'lgbm': lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=self.random_state),
                    'randforest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.random_state),
                    'svc': SVC(probability=True, random_state=self.random_state, decision_function_shape='ovr'),
                    'bagging': BaggingClassifier(n_jobs=-1, random_state=self.random_state),
                    'adaboost': AdaBoostClassifier(algorithm='SAMME', random_state=self.random_state),
                    'knn': KNeighborsClassifier(n_jobs=-1),
                    'extra_trees': ExtraTreesClassifier(n_jobs=-1, random_state=self.random_state),
                    'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
                    'logistic_regression': LogisticRegression(n_jobs=-1, max_iter=1000),
                    'bernoulli_nb': BernoulliNB(),
                    'gaussian_nb': GaussianNB(),
                    'mlp': MLPClassifier(random_state=self.random_state),
                    'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=self.random_state)
                }
            return model_names_list, defaults

        def _prepare_initial_X_fit(self, X):
            X_prepared = np.asarray(X, dtype=float).copy()
            if self.indices_to_normalized is not None:
                idx = np.asarray(self.indices_to_normalized, dtype=int)
                self.initial_scaler_ = StandardScaler()
                X_prepared[:, idx] = self.initial_scaler_.fit_transform(X_prepared[:, idx])
            return X_prepared

        def _prepare_initial_X_predict(self, X):
            X_prepared = np.asarray(X, dtype=float).copy()
            if self.indices_to_normalized is not None:
                if self.initial_scaler_ is None:
                    raise RuntimeError("Initial scaler is missing. Was the model fitted correctly?")
                idx = np.asarray(self.indices_to_normalized, dtype=int)
                X_prepared[:, idx] = self.initial_scaler_.transform(X_prepared[:, idx])
            return X_prepared

        def _make_level_features_fit(self, X_current, y, active_indices, cv):
            n_samples = X_current.shape[0]
            n_basefeats = X_current.shape[1]
            n_probability_features = 1 if self.problem_type == 'binary' else self.n_classes_
            X_level = np.zeros((n_samples, len(active_indices) * n_probability_features + n_basefeats))
            fitted_base_models = []
            modelwise_scores = np.zeros(len(active_indices), dtype=float)

            for local_idx, global_idx in enumerate(active_indices):
                model_name = self.model_names_list_[global_idx]
                learner_template = clone(self.defaults_[model_name])

                try:
                    oof_probs = cross_val_predict(
                        learner_template,
                        X_current,
                        y,
                        cv=cv,
                        method='predict_proba',
                        n_jobs=self.n_jobs
                    )
                except Exception as exc:
                    raise RuntimeError(f"OOF prediction failed for model '{model_name}'. Original error: {exc}")

                fitted_learner = clone(self.defaults_[model_name]).fit(X_current, y)
                fitted_base_models.append({
                    'name': model_name,
                    'global_index': global_idx,
                    'estimator': fitted_learner
                })

                start = local_idx * n_probability_features
                end = (local_idx + 1) * n_probability_features
                if self.problem_type == 'binary':
                    X_level[:, start:end] = oof_probs[:, 0].reshape(-1, 1)
                    try:
                        modelwise_scores[local_idx] = roc_auc_score(y, oof_probs[:, 1])
                    except Exception:
                        pred_labels = self.classes_[np.argmax(oof_probs, axis=1)]
                        modelwise_scores[local_idx] = accuracy_score(y, pred_labels)
                else:
                    X_level[:, start:end] = oof_probs
                    pred_labels = self.classes_[np.argmax(oof_probs, axis=1)]
                    modelwise_scores[local_idx] = accuracy_score(y, pred_labels)

                if self.verbose:
                    print(f"  OOF done: {model_name}")

            X_level[:, -n_basefeats:] = X_current
            return X_level, fitted_base_models, modelwise_scores

        def _make_level_features_predict(self, X_current, level_info):
            n_samples = X_current.shape[0]
            n_basefeats = X_current.shape[1]
            fitted_base_models = level_info['fitted_base_models']
            n_probability_features = 1 if self.problem_type == 'binary' else self.n_classes_
            X_level = np.zeros((n_samples, len(fitted_base_models) * n_probability_features + n_basefeats))

            for local_idx, base_info in enumerate(fitted_base_models):
                estimator = base_info['estimator']
                probs = estimator.predict_proba(X_current)
                start = local_idx * n_probability_features
                end = (local_idx + 1) * n_probability_features
                if self.problem_type == 'binary':
                    X_level[:, start:end] = probs[:, 0].reshape(-1, 1)
                else:
                    X_level[:, start:end] = probs

            X_level[:, -n_basefeats:] = X_current
            transformer = level_info['feature_transformer']
            X_level = transformer.transform(X_level) if transformer is not None else X_level
            if level_info['stackoverstacking']:
                X_level = np.column_stack((X_current, X_level))
            return X_level

        def _fit_feature_transformer(self, X_level_raw, y, level_number):
            if not self.iffeatselection:
                return None
            if self.feature_selection_levels is not None and level_number not in self.feature_selection_levels:
                return None
            if self.feature_selection_levels is None and level_number <= 1:
                return None

            initial_feature_size = X_level_raw.shape[1]
            if self.verbose:
                print(f"[RocketStack-{self.problem_type}] Feature selection/reduction at level {level_number}: {self.feat_selection_type}")

            if self.feat_selection_type == 'SFE':
                UR = 0.3
                UR_Max = 0.3
                UR_Min = 0.001
                Max_FEs = 200
                Max_Run = 5
                Run = 1
                Cost = np.zeros([Max_FEs, Max_Run])
                selected_indices = self.parent.SFE_run(
                    Input=X_level_raw,
                    Target=y,
                    UR=UR,
                    UR_Max=UR_Max,
                    UR_Min=UR_Min,
                    Max_FEs=Max_FEs,
                    Max_Run=Max_Run,
                    Run=Run,
                    Cost=Cost
                )
                if self.verbose:
                    print(f"  selected {len(selected_indices)} / {initial_feature_size} features")
                return RocketStack._ColumnSelector(selected_indices)

            if self.feat_selection_type == 'autoencoder_2layers':
                encoding_dim = max(1, int(initial_feature_size / 3))
                return self._fit_autoencoder_transformer(X_level_raw, encoding_dim=encoding_dim, mode='2layers')

            if self.feat_selection_type == 'autoencoder_4layers':
                encoding_dim = max(1, int(initial_feature_size / 3))
                return self._fit_autoencoder_transformer(X_level_raw, encoding_dim=encoding_dim, mode='4layers')

            if self.feat_selection_type == 'attentionlayer':
                attention_model = self.parent.build_attention_model_multiclass(initial_feature_size, self.n_classes_)
                class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
                y_attention = np.asarray([class_to_index[label] for label in y], dtype=int)
                attention_model.fit(
                    X_level_raw,
                    y_attention,
                    epochs=self.attention_epochs,
                    batch_size=self.attention_batch_size,
                    validation_split=0.2,
                    verbose=0 if not self.verbose else 1
                )
                attention_layer_model = Model(
                    inputs=attention_model.input,
                    outputs=attention_model.get_layer("attention_weights").output
                )
                attention_weights = attention_layer_model.predict(X_level_raw, verbose=0)
                mean_attention_weights = attention_weights.mean(axis=0)
                threshold = np.percentile(mean_attention_weights, 75)
                selected_indices = np.where(mean_attention_weights > threshold)[0]
                if len(selected_indices) == 0:
                    selected_indices = np.array([int(np.argmax(mean_attention_weights))])
                if self.verbose:
                    print(f"  selected {len(selected_indices)} / {initial_feature_size} attention-weighted features")
                return RocketStack._ColumnSelector(selected_indices)

            raise ValueError("feat_selection_type must be one of: SFE, autoencoder_2layers, autoencoder_4layers, attentionlayer")

        def _fit_autoencoder_transformer(self, X, encoding_dim, mode='2layers'):
            X = np.asarray(X, dtype=float)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            input_dim = X_scaled.shape[1]

            input_layer = Input(shape=(input_dim,))
            if mode == '2layers':
                encoded = Dense(encoding_dim, activation='relu')(input_layer)
                decoded = Dense(input_dim, activation='sigmoid')(encoded)
            else:
                encoded = Dense(max(1, input_dim // 2), activation='relu')(input_layer)
                encoded = Dense(max(1, input_dim // 3), activation='relu')(encoded)
                decoded = Dense(max(1, input_dim // 2), activation='relu')(encoded)
                decoded = Dense(input_dim, activation='sigmoid')(decoded)

            autoencoder = Model(inputs=input_layer, outputs=decoded)
            encoder = Model(inputs=input_layer, outputs=encoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(
                X_scaled,
                X_scaled,
                epochs=self.autoencoder_epochs,
                batch_size=self.autoencoder_batch_size,
                shuffle=True,
                validation_split=0.2,
                verbose=0
            )
            return RocketStack._KerasEncoderTransformer(scaler=scaler, encoder=encoder)

        def _score_meta_candidates(self, X_meta_input, y, active_indices, cv):
            meta_scores = {}
            for global_idx in active_indices:
                model_name = self.model_names_list_[global_idx]
                estimator = clone(self.defaults_[model_name])
                try:
                    scores = cross_val_score(
                        estimator,
                        X_meta_input,
                        y,
                        cv=cv,
                        scoring=self.meta_scoring,
                        n_jobs=self.n_jobs
                    )
                    score = float(np.nanmean(scores))
                except Exception:
                    # Fallback keeps training robust for models/scorers that fail on tiny folds.
                    preds = cross_val_predict(
                        estimator,
                        X_meta_input,
                        y,
                        cv=cv,
                        method='predict',
                        n_jobs=self.n_jobs
                    )
                    score = float(accuracy_score(y, preds))
                meta_scores[global_idx] = score
            return meta_scores

        def _prune_active_models(self, active_indices, modelwise_scores, rng):
            modelwise_scores = np.asarray(modelwise_scores, dtype=float)
            threshold_percentile_init = 10
            range_val = np.max(modelwise_scores) - np.min(modelwise_scores)

            if self.blur_strength == 'light':
                noise_scale = 0.05 * range_val
            elif self.blur_strength == 'moderate':
                noise_scale = 0.10 * range_val
            elif self.blur_strength in ('oof-strict', 'strict', None):
                noise_scale = 0.0
            else:
                raise ValueError("blur_strength must be 'light', 'moderate', or 'oof-strict'.")

            noisy_scores = modelwise_scores + rng.normal(loc=0.0, scale=noise_scale, size=modelwise_scores.shape)
            temp_std = np.std(noisy_scores)
            new_threshold_percentile = threshold_percentile_init / 2 + (temp_std ** 2) * 80
            threshold = np.percentile(noisy_scores, new_threshold_percentile)

            selected_local = np.argwhere(noisy_scores >= threshold)[:, 0]
            selected_global = [active_indices[i] for i in selected_local]
            return selected_global, noisy_scores, threshold

        def _prepare_prediction_input(self, X, level=None, model_name=None):
            if not self.is_fitted_:
                raise RuntimeError("Call fit() before predict(), predict_proba(), or score().")
            if level is None:
                level = self.best_level_
            level = int(level)
            if level < 1 or level > self.effective_num_levels_:
                raise ValueError(f"level must be between 1 and {self.effective_num_levels_}.")

            X_current = self._prepare_initial_X_predict(X)
            X_level = None
            for info in self.levels_[:level]:
                X_level = self._make_level_features_predict(X_current, info)
                X_current = X_level.copy()

            info = self.levels_[level - 1]
            if model_name is None:
                estimator = info['best_meta_estimator']
            else:
                if info['meta_estimators'] is None:
                    raise ValueError("This object was fitted with return_model='best'. Fit with return_model='all' to select a meta-model by name at prediction time.")
                if model_name not in info['meta_estimators']:
                    raise ValueError(f"model_name must be one of {list(info['meta_estimators'].keys())}.")
                estimator = info['meta_estimators'][model_name]
            return estimator, X_level

    def AscentTheRocket_binary_model(self, level=10, return_model='best', hyperparameterOptimization=False,
                                     iffeatselection=True, level369=True, feature_selection_levels=None,
                                     feat_selection_type='SFE', indices_to_normalized=None,
                                     blur_strength='light', left_model_threshold=2, meta_scoring='roc_auc',
                                     cv_splits=5, random_state=42, n_jobs=-1, verbose=True,
                                     autoencoder_epochs=50, autoencoder_batch_size=32,
                                     attention_epochs=50, attention_batch_size=32, n_trials=30, **kwargs):
        """
        Create a trainable sklearn-like RocketStack binary classifier.

        Parameters
        ----------
        level : int
            Number of recursive RocketStack levels to train.
        return_model : {'best', 'all'}
            'best' stores only the best OOF-scoring meta-model per level.
            'all' stores all fitted meta-models per level and still uses the best one by default.
        feature_selection_levels : tuple/list or None
            User-facing recursive levels at which feature selection is applied. If None and level369=True,
            defaults to (3, 6, 9). If None and level369=False, feature selection is applied after every
            level except level 1.
        """
        if 'return' in kwargs:
            return_model = kwargs.pop('return')
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {list(kwargs.keys())}")
        if feature_selection_levels is None:
            feature_selection_levels = tuple(lv for lv in (3, 6, 9) if lv <= level) if level369 else None

        return RocketStack._RocketStackTrainableModel(
            parent=self,
            problem_type='binary',
            num_of_level=level,
            return_model=return_model,
            iffeatselection=iffeatselection,
            feat_selection_type=feat_selection_type,
            feature_selection_levels=feature_selection_levels,
            stackoverstacking=False,
            indices_to_normalized=indices_to_normalized,
            blur_strength=blur_strength,
            left_model_threshold=left_model_threshold,
            meta_scoring=meta_scoring,
            cv_splits=cv_splits,
            random_state=random_state,
            n_jobs=n_jobs,
            autoencoder_epochs=autoencoder_epochs,
            autoencoder_batch_size=autoencoder_batch_size,
            attention_epochs=attention_epochs,
            attention_batch_size=attention_batch_size,
            verbose=verbose,
            hyperparameterOptimization=hyperparameterOptimization,
            n_trials=n_trials
        )

    def AscentTheRocket_multiclass_model(self, level=10, return_model='best', hyperparameterOptimization=False,
                                         iffeatselection_or_not=True, level369=True, feature_selection_levels=None,
                                         feat_selection_type='attentionlayer', stackoverstacking=False,
                                         indices_to_normalized=None, blur_strength='light', left_model_threshold=5,
                                         meta_scoring='accuracy', cv_splits=5, random_state=42, n_jobs=-1,
                                         verbose=True, autoencoder_epochs=50, autoencoder_batch_size=32,
                                         attention_epochs=50, attention_batch_size=32, n_trials=30, **kwargs):
        """
        Create a trainable sklearn-like RocketStack multi-class classifier.

        Use exactly like a sklearn estimator:
            model = RocketStack().AscentTheRocket_multiclass_model(level=5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        """
        if 'return' in kwargs:
            return_model = kwargs.pop('return')
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {list(kwargs.keys())}")
        if feature_selection_levels is None:
            feature_selection_levels = tuple(lv for lv in (3, 6, 9) if lv <= level) if level369 else None

        return RocketStack._RocketStackTrainableModel(
            parent=self,
            problem_type='multiclass',
            num_of_level=level,
            return_model=return_model,
            iffeatselection=iffeatselection_or_not,
            feat_selection_type=feat_selection_type,
            feature_selection_levels=feature_selection_levels,
            stackoverstacking=stackoverstacking,
            indices_to_normalized=indices_to_normalized,
            blur_strength=blur_strength,
            left_model_threshold=left_model_threshold,
            meta_scoring=meta_scoring,
            cv_splits=cv_splits,
            random_state=random_state,
            n_jobs=n_jobs,
            autoencoder_epochs=autoencoder_epochs,
            autoencoder_batch_size=autoencoder_batch_size,
            attention_epochs=attention_epochs,
            attention_batch_size=attention_batch_size,
            verbose=verbose,
            hyperparameterOptimization=hyperparameterOptimization,
            n_trials=n_trials
        )
