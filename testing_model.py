class testing_model:
  def __init__ (self, X_train, y_train, X_test, y_test, scores):
    
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.scores = scores
    
     
  def KNeighbors (self):
    scores_def = scores
    parameters = {'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                              'n_neighbors': [k for k in range (1, 11)],
                              'p': [p for p in range (1, 11, 2)]}
  
    for score in scores_def:
      clf = GridSearchCV(estimator = KNeighborsClassifier(), scoring = score, param_grid = parameters, cv = 5, iid = False)
      clf.fit(X_train, y_train)
      score_test = clf.best_estimator_.score(X_test, y_test) 
      print(score)
      print(clf.best_estimator_)
      print(clf.best_score_)
    print('accuracy_test: ', score_test)

    
  def Bernoulli (self):
    scores_def = scores
    parameters = {'alpha': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 1.], 
                  'binarize': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 1.],
                  'fit_prior': ['True', 'False']}
  
    for score in scores_def:
      clf = GridSearchCV(estimator = BernoulliNB(), scoring = score, param_grid = parameters, cv = 5, iid = False)
      clf.fit(X_train, y_train)
      score_test = clf.best_estimator_.score(X_test, y_test) 
      print(score)
      print(clf.best_estimator_)
      print(clf.best_score_)
    print('accuracy_test: ', score_test)   

    
  def Gaussian (self):
    scores_def = scores
    parameters = {'var_smoothing':  [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 1.]}
  
    for score in scores_def:
      clf = GridSearchCV(estimator = GaussianNB(), scoring = score, param_grid = parameters, cv = 5, iid = False)
      clf.fit(X_train, y_train)
      score_test = clf.best_estimator_.score(X_test, y_test) 
      print(score)
      print(clf.best_estimator_)
      print(clf.best_score_)
    print('accuracy_test: ', score_test) 
    
    
  def Multinomial (self):
    scores_def = scores
    parameters = {'alpha': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 1.], 'fit_prior': ['True', 'False']}
  
    for score in scores_def:
      clf = GridSearchCV(estimator = MultinomialNB(), scoring = score, param_grid = parameters, cv = 5, iid = False)
      clf.fit(X_train, y_train)
      score_test = clf.best_estimator_.score(X_test, y_test) 
      print(score)
      print(clf.best_estimator_)
      print(clf.best_score_)
    print('accuracy_test: ', score_test) 
    
    
  def Complement (self):
    scores_def = scores
    parameters = {'alpha': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 1.], 'fit_prior': ['True', 'False'], 'norm': ['True', 'False']}
  
    for score in scores_def:
      clf = GridSearchCV(estimator = Complement(), scoring = score, param_grid = parameters, cv = 5, iid = False)
      clf.fit(X_train, y_train)
      score_test = clf.best_estimator_.score(X_test, y_test) 
      print(score)
      print(clf.best_estimator_)
      print(clf.best_score_)
    print('accuracy_test: ', score_test)