import pickle
from flask import Flask,request
app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')  #decorator #rootpage
def welcome():
    return"Welcome to the flask page"


@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return str(prediction)
    

@app.route('/predict_file',methods=["POST"])
def predict_note_authentication1():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return str(prediction)
    
    












if __name__=='__main__':
    app.run()
    
