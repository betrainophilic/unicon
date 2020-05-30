from django.shortcuts import render
from . import ml_model

def home(request):
    return render(request,'index.html')

def result(request):
    Pclass=int(request.GET['pclass'])
    Age=int(request.GET['age'])
    SibSp=int(request.GET['sibsp'])
    Parch=int(request.GET['parch'])
    Fare=int(request.GET['fare'])
    male=int(request.GET['male'])
    Q=int(request.GET['q'])
    S=int(request.GET['s'])
    prediction=ml_model.model_predictions(Pclass,Age,SibSp,Parch,Fare,male,Q,S)
    return render(request,'result.html',{'prediction':prediction})
