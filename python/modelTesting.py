import pandas as pd
from python.class_def import TextSelector,NumberSelector,DenseTransformer
import pickle
import joblib

def test(df):
    try:
        #df = df.dropna()

        features = [c for c in df.columns.values if
                    c not in ['Username', 'Userid', 'Rating', 'Helpfulness', 'Comment', 'TimeCreated']]
        #print("List of features : \n", *features, sep="\n")


        with open('models/model_RF.pkl', 'rb') as f:
            model_SVM = pickle.load(f)

        X_test = df[features]

        pred_RF = model_SVM.predict(X_test)

        X_test = X_test.reset_index()


        newdf = pd.DataFrame(pred_RF, columns=['Label'])
        df_merged = pd.concat([X_test, newdf], axis=1)

        return df_merged

    except Exception as e:
            print(e)

def customtest(df,modeltype):
    features = [c for c in df.columns.values if
                c not in ['Helpfulness', 'Comment']]

    if modeltype == "Random_Forest":
        model = "model_RF.pkl"
    elif modeltype == "Support_Vector_Machine":
        model = "model_SVMV2.pkl"
    else:
        model = "model_NB.pkl"

    with open('models/'+model, 'rb') as f:
        model = pickle.load(f)
    #print("List of features : \n", *features, sep="\n")
    X_test = df[features]

    pred_RF = model.predict(X_test)
    return pred_RF