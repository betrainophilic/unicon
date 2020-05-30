def model_predictions(Pclass,Age,SibSp,Parch,Fare,male,Q,S):
    import pickle
    x=[[Pclass,Age,SibSp,Parch,Fare,male,Q,S]]
    randomforest=pickle.load(open('titanic_model.sav','rb'))
    prediction=randomforest.predict(x)
    if prediction==0:
        prediction='Not survived'
    elif prediction==1:
        prediction='Survived'
    else:
        prediction='Error'
    return prediction
