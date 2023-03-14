# save this as app.py
import json
from python.class_def import TextSelector,NumberSelector,DenseTransformer

from flask import Flask, request, render_template, jsonify
import pandas as pd

from urllib.parse import urlparse

from flask_mysqldb import MySQL

from python.preprocessing import checkUrl, obtainData, cleaning, preprocessingMalay, runPreprocessingEnglish
from python.modelTesting import test,customtest

# Model Instantiate
import malaya
preprocessModel = malaya.preprocessing.preprocessing(normalize=['number'], annotate=['elongated', 'repeated'])
normalizer = malaya.normalize.normalizer()
modelMalayEnglishTranslator = malaya.translation.ms_en.transformer(model='noisy-base')

# Threading
from threading import Thread

# Actual App Configurations
app = Flask(__name__)

app.config['MYSQL_HOST'] = #Server used; localhost or 192.....
app.config['MYSQL_USER'] = #Your selecter user;   root,local,....
app.config['MYSQL_PASSWORD'] = #Your defined pass
app.config['MYSQL_DB'] = #Your db name
app.config['MYSQL_DATABASE_CHARSET'] = #charset used in the database
mysql = MySQL(app)


def prepConnection():
    cur = mysql.connection.cursor()
    return cur


def terminateDB(cursor):
    cursor.close()



@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/submitCustom', methods=['POST'])
def submitCustom():

    try:

        if 'text' not in request.form:
            return jsonify({'requirements': 'Please dont change js'})
        elif not request.form['text'].split():
            return jsonify({'requirements': 'Enter your sample review please'})
        elif 'ourmodel'not in request.form:
            return jsonify({'requirements': 'Please pick a model please'})


        text = request.form['text'].strip()
        modeltype = request.form['ourmodel'].strip()

        if modeltype in ('Random_Forest','Naive_Bayes','Support_Vector_Machine'):

            customData = [[text, 0]]
            customDF = pd.DataFrame(customData, columns=['Comment', 'Helpfulness'])

            customDF, TotalOriginalRecords = cleaning(customDF)

            customDF = preprocessingMalay(customDF, preprocessModel, normalizer, modelMalayEnglishTranslator)

            customDF, TotalProperRecords = runPreprocessingEnglish(customDF)

            result = customtest(customDF,modeltype)

            if result[0] == 0:
                return jsonify({'spam': 'The review input is predicted as spam'})
            else:
                return jsonify({'ham': 'Awesome, the model predict your review as genuine!'})
        else:
            return jsonify({'requirements': 'Please pick the right model at least'})



    except Exception as e:
        print(e)



@app.route('/mainExe', methods=['POST'])
def mainExe():

    url = request.form['url']

    topleveldomain = urlparse(url).netloc

    #Step 1 check url is valid or not
    if topleveldomain == "shopee.com.my":

        # checkurl(URL put here) in preprocessing.py function

        shop_id, item_id , validate = checkUrl(url)
        if validate:

            cur = prepConnection()
            cur.callproc('getInsertRecords', (shop_id,item_id))
            row_headers = [x[0] for x in cur.description]
            data = cur.fetchall()

            result = []
            for rv in data:
                result.append(dict(zip(row_headers, rv)))

            terminateDB(cur)
            if len(result) > 0:

                try:
                    if len(result[0]['Username']) > 0:

                        return sendDataToJs(shop_id, item_id,False,result)

                except Exception:
                    #print("Getting Record Problem")
                    pass

                try:
                    if result[0]['AllowInsert'] == 'Allow Insert!':
                        return processing(shop_id,item_id)
                except Exception:
                    #print("Something happen")
                    pass

            else:
                return jsonify({'reportError': 'No Review Has Been Made !'})
        else:
            return jsonify({'reportError': 'URL must come from an existing product!'})
    else:
        return jsonify({'reportError': 'Enter only url from Shopee.com.my'})



@app.route('/preprocessing', methods=['POST'])
def processing(shop_id,item_id):

    try:
        df = obtainData(shop_id,item_id)

        df, TotalOriginalRecords = cleaning(df)

        df = preprocessingMalay(df,preprocessModel,normalizer, modelMalayEnglishTranslator)

        df, TotalProperRecords = runPreprocessingEnglish(df)

        expandedDF = df[['Comment','Rating','Helpfulness','TimeCreated','Userid','Username']].copy()
        expandedDF = expandedDF.reset_index()

        df = test(df)

        df_merged = pd.concat([df, expandedDF], axis=1)

        df_dict_fastBoi = df_merged.to_dict(orient="records")


        #Warning SQL Query Here
        insertIntoDB(df_dict_fastBoi,item_id,TotalOriginalRecords)

        return sendDataToJs(shop_id, item_id, True)

    except Exception as e:
        print(f"Error Located at app.processing \n{e}")



def jsonReviewComment(QueryResult, wantJson):
    try:
        if wantJson:
            return jsonify(QueryResult)
        else:
            return QueryResult
    except Exception as e:
        print(f"Error Located at app.jsonReviewComment \n{e}")


def readFromDBStatOnly(QueryResult, wantJson):

    try:
        FalseLabel = {
            "TotalRecords": QueryResult[0]["TotalRecords"],
            "TotalEmoji": QueryResult[0]["TotalEmoji"],
            "TotalPunc": QueryResult[0]["TotalPunc"],
            "TotalCaps": QueryResult[0]["TotalCaps"],
            "AvgTextReview": QueryResult[0]["AvgTextReview"],
            "AvgEmoji": QueryResult[0]["AvgEmoji"],
            "AvgPunc":QueryResult[0]["AvgPunc"],
            "AvgCaps":QueryResult[0]["AvgCaps"]
        }
    except Exception as e:
        FalseLabel = {
            "TotalRecords": 0,
            "TotalEmoji": 0,
            "TotalPunc": 0,
            "TotalCaps": 0,
            "AvgTextReview": 0,
            "AvgEmoji": 0,
            "AvgPunc": 0,
            "AvgCaps": 0
        }
        pass
    try:
        TrueLabel = {
            "TotalRecords": QueryResult[1]["TotalRecords"],
            "TotalEmoji": QueryResult[1]["TotalEmoji"],
            "TotalPunc": QueryResult[1]["TotalPunc"],
            "TotalCaps": QueryResult[1]["TotalCaps"],
            "AvgTextReview": QueryResult[1]["AvgTextReview"],
            "AvgEmoji": QueryResult[1]["AvgEmoji"],
            "AvgPunc":QueryResult[1]["AvgPunc"],
            "AvgCaps":QueryResult[1]["AvgCaps"]
        }
    except Exception as e:
        TrueLabel = {
            "TotalRecords": 0,
            "TotalEmoji": 0,
            "TotalPunc": 0,
            "TotalCaps": 0,
            "AvgTextReview": 0,
            "AvgEmoji": 0,
            "AvgPunc": 0,
            "AvgCaps": 0
        }
        pass

    return TrueLabel,FalseLabel


def insertIntoDB(df_dict_fastBoi, item_id,TotalOriginalRecords):
    try:
        cur = prepConnection()

        Query = "INSERT INTO Review (TextReview,Rating,Helpfulness,Label,ReviewCreatedAt,TotalLengthManglish,TotalEmoji,TotalPunc,TotalCaps,UserID,ProductID,FrequencyAdverbs,FrequencyVerbs,FrequencyAdjectives,FrequencyNouns) VALUES " \
                "( %(Comment)s,%(Rating)s,%(Helpfulness)s,%(Label)s,%(TimeCreated)s,%(LengthManglish)s,%(TotalEmoji)s,%(TotalPunct)s,%(TotalCaps)s,%(Userid)s,"+item_id+",%(FrequencyAdverbs)s,%(FrequencyVerbs)s,%(FrequencyAdjectives)s,%(FrequencyNouns)s )"



        for records in df_dict_fastBoi:
            cur.callproc('getInsertUsers', (records['Userid'], records['Username']))

        cur.callproc('dropReviewForUpdate', (item_id,TotalOriginalRecords))


        cur.executemany(Query, df_dict_fastBoi)
        mysql.connection.commit()

        terminateDB(cur)

    except Exception as e:
        print(f"Error Located at app.insertIntoDB \n{e}")


def sendDataToJs(shop_id, item_id, skip, result=[]):
    try:
        if skip:

            cur = prepConnection()

            cur.callproc('getInsertRecords', (shop_id, item_id))
            row_headers = [x[0] for x in cur.description]
            data = cur.fetchall()

            terminateDB(cur)

            result = []
            for rv in data:
                result.append(dict(zip(row_headers, rv)))
    except Exception as e:
        print(f"Error Located at sendDataToJs#1 \n{e}")

    jsonResultDB = jsonReviewComment(result, False)

    try:
        cur = prepConnection()

        cur.callproc('getStatisticResults', (item_id,))
        row_headers = [x[0] for x in cur.description]
        data = cur.fetchall()

        terminateDB(cur)

        statResult = []
        for rv in data:
            statResult.append(dict(zip(row_headers, rv)))

        TrueLabel, FalseLabel = readFromDBStatOnly(statResult, False)
    except Exception as e:
        print(f"Error Located at sendDataToJs#2 \n{e}")

    #Get Stat Result for only Original Records ONLY!
    try:

        cur = prepConnection()

        cur.callproc('getCountOriRecords', (item_id,))
        row_headers = [x[0] for x in cur.description]
        result = cur.fetchall()

        terminateDB(cur)

        reviewCount = []
        for rv in result:
            reviewCount.append(dict(zip(row_headers, rv)))


        OriginalRecordsCount = {
            "OriginalRecordsCount": reviewCount[0]['Records'],
        }


        newPackedJSON = {
            "TrueLabel": TrueLabel,
            "FalseLabel": FalseLabel,
            "OriginalRecordsCount": OriginalRecordsCount,
            "Result": jsonResultDB
        }
        return jsonify(newPackedJSON)

    except Exception as e:
        print(f"Error Located at app.sendDataToJs#3 \n{e}")



#Remove when on the go
if __name__ == "__main__":

    app.run(debug=True)
    app.run(host="0.0.0.0")



