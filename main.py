from flask import Response, request, jsonify, g, app, Flask
import flask_mysqldb
import re
import math
import requests
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import datetime
from noun_supersense_api import manager
import nltk
from nltk.corpus import wordnet as wn
import spacy
from random import randint, randrange
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['MYSQL_HOST'] = 'yuriom.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'yuriom'
app.config['MYSQL_PASSWORD'] = 'Piedmont'
app.config['MYSQL_DB'] = 'yuriom$api'
mysql = flask_mysqldb.MySQL(app)

def distance(w0, w1):
    c = mysql.connection.cursor()
    d0 = None
    if (c.execute("SELECT * FROM words WHERE word=%s", (w0,))) > 0:
        d0 = c.fetchone()
    d1 = None
    if (c.execute("SELECT * FROM words WHERE word=%s", (w1,))) > 0:
        d1 = c.fetchone()

    # if d0 == None or d1 == None:
    #     return None
    # a = 0.0
    # for i in range (0, 300):
    #     a = a + (d0[i + 1] * d1[i + 1])
    # b = 0.0
    # for i in range (0, 300):
    #     b = b + (d0[i + 1] * d0[i + 1])
    # c = 0.0
    # for i in range (0, 300):
    #     c = c + (d1[i + 1] * d1[i + 1])
    # return a / (math.sqrt(b) * math.sqrt(c))
    if d0 == None or d1 == None:
        return None
    a = []
    b = []
    for i in range (0, 300):
        a.append(d0[i + 1])
        b.append(d1[i + 1])

    return dot(a, b)/(norm(a)*norm(b))

@app.route('/', methods=['GET'])
def home():
    with open('/home/yuriom/csc/index.html', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return lines

@app.route('/thresholds.html', methods=['GET'])
def threshold():
    with open('/home/yuriom/csc/thresholds.html', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return lines

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    with open('/home/yuriom/csc/favicon.ico', 'rb') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, "image/x-icon");

# @app.route('/experimental.html', methods=['GET'])
# def experimental():
#     with open('/home/yuriom/csc/experimental.html', 'r') as file:
#         lines = file.read() #.replace('\n', '')
#         file.close()
#     return lines

@app.route('/CA.html', methods=['GET'])
def CA():
    with open('/home/yuriom/csc/index.html', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return lines

@app.route('/CB.html', methods=['GET'])
def CB():
    with open('/home/yuriom/csc/CB.html', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return lines

@app.route('/CB.js', methods=['GET'])
def CB_js():
    with open('/home/yuriom/csc/CB.js', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, "text/javascript");

@app.route('/EA.html', methods=['GET'])
def EA():
    with open('/home/yuriom/csc/EA.html', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return lines

@app.route('/EA.js', methods=['GET'])
def EA_js():
    with open('/home/yuriom/csc/EA.js', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, "text/javascript");

@app.route('/EAdmin.html', methods=['GET'])
def EAdmin():
    with open('/home/yuriom/csc/EAdmin.html', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return lines

@app.route('/EAdmin.js', methods=['GET'])
def EAdmin_js():
    with open('/home/yuriom/csc/EAdmin.js', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, "text/javascript");

@app.route('/main.js', methods=['GET'])
def main_js():
    with open('/home/yuriom/csc/main.js', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, "text/javascript");

@app.route('/experimental.js', methods=['GET'])
def experimental_js():
    with open('/home/yuriom/csc/experimental.js', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, "text/javascript");

@app.route('/stylesheet.css', methods=['GET'])
def stylesheet():
    with open('/home/yuriom/csc/stylesheet.css', 'r') as file:
        lines = file.read() #.replace('\n', '')
        file.close()
    return Response(lines, mimetype="text/css")

@app.route('/sound/pop0.mp3', methods=['GET'])
def sound_pop0():
    with open('/home/yuriom/csc/sound/pop0.mp3', 'rb') as file:
        lines = file.read()
        file.close()
    return Response(lines, mimetype="audio/mp3")

@app.route('/sound/golf0.mp3', methods=['GET'])
def sound_golf0():
    with open('/home/yuriom/csc/sound/golf0.mp3', 'rb') as file:
        lines = file.read()
        file.close()
    return Response(lines, mimetype="audio/mp3")

@app.route('/sound/golf1.mp3', methods=['GET'])
def sound_golf1():
    with open('/home/yuriom/csc/sound/golf1.mp3', 'rb') as file:
        lines = file.read()
        file.close()
    return Response(lines, mimetype="audio/mp3")

@app.route('/api/distance', methods=['GET'])
def api_distance():
    if ('first' in request.args) and ('second' in request.args):
        first = request.args['first']
        second = request.args['second']
        return jsonify({
            'first': first,
            'second': second,
            'distance': distance(first, second)})
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/distance_matrix', methods=['POST'])
def api_distance_matrix():
    c = mysql.connection.cursor()
    words = request.json['words']
    words_vector = {}
    result = []
    for w in words:
        if (c.execute("SELECT * FROM words WHERE word=%s", (w,))) > 0:
            d = c.fetchone()
        if d is not None:
            vector = []
            for i in range (0, 300):
                vector.append(d[i + 1])
            words_vector[w] = vector
    for w1 in words:
        row = [w1]
        for w2 in words:
            if w1 == w2:
                row.append(1)
            else:
                row.append(dot(words_vector[w1], words_vector[w2])/(norm(words_vector[w1])*norm(words_vector[w2])))
        result.append(row)
    return jsonify(result)

@app.route('/api/get_noun_adj', methods=['GET'])
def get_noun_adj():
    if ('text' in request.args):
        noun_adj = {'noun' : [], 'adj' : []}
        text = request.args['text']
        tokens = nltk.word_tokenize(text)
        for word, pos in nltk.pos_tag(tokens):
            if (pos in ['JJ', 'JJR', 'JJS']):
                noun_adj['adj'].append(word)
            elif (pos in ['NN', 'NNP', 'NNS', 'NNPS']):
                noun_adj['noun'].append(word)
        return jsonify(noun_adj)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/save_log', methods=['POST'])
def save_log():
    log = request.form['log']
    filename = 'EA_' + str(randrange(100000, 1000000))
    with open('/home/yuriom/csc/logs/' + filename + '.log', 'w') as f:
        f.write(log)
    return jsonify(filename)

def convert(word, from_pos, to_pos):

    # Just to make it a bit more readable
    WN_NOUN = 'n'
    WN_VERB = 'v'
    WN_ADJECTIVE = 'a'
    WN_ADJECTIVE_SATELLITE = 's'
    WN_ADVERB = 'r'

    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return None

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w:-w[1])

    if len(result) != 0:
        return result[0][0]
    else:
        return None

def ranking(candidates):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(cur_dir, 'SingleWord_U_score.csv')
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['word', 'U_score'])
    y = df['U_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    params = {
        'silent': 1,
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.1,
        'tree_method': 'exact',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'predictor': 'cpu_predictor'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params['max_depth'] = 6
    params['eta'] =0.05
    model = xgb.train(params=params,
                      dtrain=dtrain,
                      num_boost_round=1000,
                      early_stopping_rounds=5,
                      evals=[(dtest, 'test')])
    group_prediction = []
    for group in candidates:
        group_df = pd.DataFrame(group['vector'], columns =X_train.columns)
        prediction = list(model.predict(xgb.DMatrix(group_df), ntree_limit=model.best_ntree_limit))
        prediction.insert(0, 5)
        tuple_group = list(zip(group['words'], prediction))
        tuple_group.sort(key=lambda tup: tup[1], reverse = True)
        # new_tg = []
        # for tg in tuple_group:
        #     new_tg.append([tg[0], str(tg[1])])
        # group_prediction.append(new_tg)
        group_prediction.append([tg[0] for tg in tuple_group])
    return group_prediction

def get_vector_matrix(word_matrix):
    cur = mysql.connection.cursor()
    new_matrix = []
    for group in word_matrix:
        format_strings = ','.join(['%s'] * (len(group) - 1))
        cur.execute("SELECT * FROM words WHERE word IN (%s)" % format_strings, tuple(group[1:]))
        vectors = cur.fetchall()
        tmp_group = {
            'words' : [group[0]],
            'vector' : []
        }
        for vector in vectors:
            tmp_group['words'].append(list(vector)[0])
            tmp_group['vector'].append(list(vector)[1:])
        new_matrix.append(tmp_group)
    return new_matrix

@app.route('/api/get_all_related_words_as_adj', methods=['GET'])
def api_get_all_related_words_as_adj():
    words = request.args['word'].split(',')
    final_words = []
    for word in words:
        related_words = [word]
        results = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500").json()
        for edge in results['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']
            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] != "Antonym") and (edge['rel']['label'] != "DistinctFrom"):
                if not "-" in related_word and not " " in related_word and not "_" in related_word:
                    if check_pos(related_word, 'ADJ'):
                        related_words.append(related_word)
                    else:
                        c_res = convert(related_word, 'n', 'a')
                        if c_res is not None:
                            related_words.append(c_res)
        final_words.append(list(dict.fromkeys(related_words)))
    vector_matrix = get_vector_matrix(final_words)
    candidates_word1 = ranking(vector_matrix)
    return jsonify(candidates_word1)

def filter_xboost(words):
    cur = mysql.connection.cursor()

    format_strings = ','.join(['%s'] * len(words))
    cur.execute("SELECT * FROM words WHERE word IN (%s)" % format_strings, tuple(words))
    vectors = cur.fetchall()
    new_matrix = {
        'words' : [],
        'vector' : []
    }
    for vector in vectors:
        new_matrix['words'].append(list(vector)[0])
        new_matrix['vector'].append(list(vector)[1:])

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(cur_dir, 'SingleWord_U_score.csv')
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['word', 'U_score'])
    y = df['U_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    params = {
        'silent': 1,
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.1,
        'tree_method': 'exact',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'predictor': 'cpu_predictor'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params['max_depth'] = 6
    params['eta'] =0.05
    model = xgb.train(params=params,
                      dtrain=dtrain,
                      num_boost_round=1000,
                      early_stopping_rounds=5,
                      evals=[(dtest, 'test')])
    group_prediction = []

    group_df = pd.DataFrame(new_matrix['vector'], columns =X_train.columns)
    prediction = list(model.predict(xgb.DMatrix(group_df), ntree_limit=model.best_ntree_limit))
    tuple_group = list(zip(new_matrix['words'], prediction))
    group_prediction = list(filter(lambda tg: tg[1] >= 1.7, tuple_group))

    return [tg[0] for tg in group_prediction]

@app.route('/api/get_all_combinations_w1_w2', methods=['GET'])
def api_get_all_combinations_w1_w2():
    score_list = [
        [0.15, 0.2, 5.17],
        [-0.05, 0, 4.85],
        [0.05, 0.1, 4.77],
        [0.1, 0.15, 4.75],
        [0.4, 0.45, 4.73],
        [0.2, 0.25, 4.63],
        [0.65, 10, 4.57],
        [-0.1, -0.05, 4.55],
        [0.3, 0.35, 4.55],
        [0, 0.05, 4.54],
        [-10, -0.1, 4.43],
        [0.25, 0.3, 4.30],
        [0.5, 0.55, 4.26],
        [0.35, 0.4, 4.23],
        [0.55, 0.6, 4.06],
        [0.45, 0.5, 3.68],
        [0.6, 0.65, 3.48]
    ]
    words1 = request.args['word'].split(',')
    low = float(request.args['low'])
    high = float(request.args['high'])
    combinations = {}
    related_nouns = []

    for word in words1:
        results = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500").json()
        for edge in results['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']
            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] != "Antonym") and (edge['rel']['label'] != "DistinctFrom"):
                if check_pos(related_word, 'NOUN'):
                    related_nouns.append(related_word)
                else:
                    c_res = convert(related_word, 'a', 'n')
                    if c_res is not None:
                        related_nouns.append(c_res)
    related_nouns = list(dict.fromkeys(related_nouns))

    related_nouns = filter_xboost(related_nouns)

    for word in words1:
        combinations[word] = []
        tmp = []
        for noun in related_nouns:
            dis = distance(word, noun)
            if dis is not None and low <= dis and dis < high:
                for score_item in score_list:
                    if score_item[0] <= dis and dis <= score_item[1]:
                        tmp.append([word + '-' + noun, score_item[2], dis])
                        break
        tmp.sort(key=lambda y: (-y[1], y[2]))
        combinations[word] = [t[0] for t in tmp]
    return jsonify(combinations)

@app.route('/api/get_antonyms_w1_w2', methods=['GET'])
def api_get_antonyms_w1_w2():

    word1 = request.args['word1']
    word2 = request.args['word2']
    candi_word1 = []
    candi_word2 = []

    sym_results = requests.get("https://api.conceptnet.io/c/en/" + word1 + "?filter=/c/en&limit=500").json()
    for sym_edge in sym_results['edges']:
        sym_node_related_word = sym_edge['end'] if sym_edge['start']['label'] == word1 else sym_edge['start']
        sym_related_word = sym_node_related_word['label']
        if (word1 != sym_related_word) and ('language' in sym_node_related_word) and (sym_node_related_word['language'] == 'en') and (sym_edge['rel']['label'] == "RelatedTo" or sym_edge['rel']['label'] == "SimilarTo"):
            results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + sym_related_word + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
            for edge in results['edges']:
                node_related_word = edge['end'] if edge['start']['label'] == sym_related_word else edge['start']
                related_word = node_related_word['label']
                if (sym_related_word != related_word) and (word1 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
                    if check_pos(related_word, 'ADJ'):
                        candi_word1.append(related_word)
                    else:
                        c_res = convert(related_word, 'n', 'a')
                        if (c_res is not None) and (c_res != word1):
                            candi_word1.append(c_res)
    candi_word1 = list(dict.fromkeys(candi_word1))

    sym_results = requests.get("https://api.conceptnet.io/c/en/" + word2 + "?filter=/c/en&limit=500").json()
    for sym_edge in sym_results['edges']:
        sym_node_related_word = sym_edge['end'] if sym_edge['start']['label'] == word2 else sym_edge['start']
        sym_related_word = sym_node_related_word['label']
        if (word2 != sym_related_word) and ('language' in sym_node_related_word) and (sym_node_related_word['language'] == 'en') and (sym_edge['rel']['label'] == "RelatedTo" or sym_edge['rel']['label'] == "SimilarTo"):
            results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + sym_related_word + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
            for edge in results['edges']:
                node_related_word = edge['end'] if edge['start']['label'] == sym_related_word else edge['start']
                related_word = node_related_word['label']
                if (sym_related_word != related_word) and (word2 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
                    if check_pos(related_word, 'NOUN'):
                        candi_word2.append(related_word)
                    else:
                        c_res = convert(related_word, 'a', 'n')
                        if (c_res is not None) and (c_res != word2):
                            candi_word2.append(c_res)
    candi_word2 = list(dict.fromkeys(candi_word2))


    # results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + word1 + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
    # for edge in results['edges']:
    #     node_related_word = edge['end'] if edge['start']['label'] == word1 else edge['start']
    #     related_word = node_related_word['label']
    #     if (word1 != related_word) and (word1 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
    #         if check_pos(related_word, 'ADJ'):
    #             candi_word1.append(related_word)
    #         else:
    #             c_res = convert(related_word, 'n', 'a')
    #             if (c_res is not None) and (c_res != word1):
    #                 candi_word1.append(c_res)
    # candi_word1 = list(dict.fromkeys(candi_word1))



    # results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + word2 + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
    # for edge in results['edges']:
    #     node_related_word = edge['end'] if edge['start']['label'] == word2 else edge['start']
    #     related_word = node_related_word['label']
    #     if (word2 != related_word) and (word2 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
    #         if check_pos(related_word, 'NOUN'):
    #             candi_word2.append(related_word)
    #         else:
    #             c_res = convert(related_word, 'a', 'n')
    #             if (c_res is not None) and (c_res != word2):
    #                 candi_word2.append(c_res)
    # candi_word2 = list(dict.fromkeys(candi_word2))


    return jsonify({
            word1: candi_word1,
            word2: candi_word2
        })

@app.route('/api/noun_adj_convertor', methods=['GET'])
def api_noun_adj_convertor():
    if ('word' in request.args and 'from' in request.args and 'to' in request.args):
        words = request.args['word'].split(',')
        from_pos = request.args['from']
        to_pos = request.args['to']

        final_result = []

        for word in words:
            c_res = convert(word, from_pos, to_pos)
            if c_res is not None:
                final_result.append(c_res)

        return jsonify(final_result)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/vector', methods=['GET'])
def api_vector():
    if ('word' in request.args):
        word = request.args['word']
        c = mysql.connection.cursor()
        d0 = 0
        if (c.execute("SELECT * FROM words WHERE word=%s", (word,))) > 0:
            d0 = c.fetchone()
        return jsonify(d0)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/supersense_noun', methods=['GET'])
def api_supersense_noun():
    if ('word' in request.args):
        word = request.args['word']
        ss_obj = manager.SuperSense()
        classes = []
        result = {
            'Tops' : 0,
            'act' : 0,
            'animal' : 0,
            'artifact' : 0,
            'attribute' : 0,
            'body' : 0,
            'cognition' : 0,
            'communication' : 0,
            'event' : 0,
            'feeling' : 0,
            'food' : 0,
            'group' : 0,
            'location' : 0,
            'motive' : 0,
            'object' : 0,
            'person' : 0,
            'phenomenon' : 0,
            'plant' : 0,
            'possession' : 0,
            'process' : 0,
            'quantity' : 0,
            'relation' : 0,
            'shape' : 0,
            'state' : 0,
            'substance' : 0,
            'time' : 0
        }
        SuperSense_total = ss_obj.sense_info_list_for_lemma_pos[(word, 'n')]
        for sense_info in SuperSense_total:
            these_classes = ss_obj.get_classes_for_synset_pos(sense_info.synset, 'n')
            if these_classes is not None:
                classes.extend(list(map(lambda x: x.replace('noun.', ''), these_classes)))
        if len(classes) != 0:
            if not isinstance(classes[0],list):
                sorted_classes = list(set(classes))
        for c_item in sorted_classes:
            result[c_item] = classes.count(c_item) / len(SuperSense_total)
        return jsonify(result)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/adjective-frequency', methods=['GET'])
def api_frequency():
    if ('word' in request.args):
        word = request.args['word'] + "-j"

        c = mysql.connection.cursor()
        d = None
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word,)) > 0:
            d = c.fetchone()
        return jsonify({
            'word': request.args['word'],
            'frequency': d[1] if d != None else None,
            'relative-frequency': d[2] if d != None else None})
    else:
        return "Error: Please specify appropriate fields."

def check_pos(word, pos):
    doc = nlp(word)
    return doc[0].pos_ == pos

@app.route('/api/list-related-adjectives', methods=['GET'])
def api_list_related_adjectives():
    if ('word' in request.args):
        word = request.args['word']
        relation = None
        if ("relation" in request.args):
            relation = request.args['relation']
        limit = 500
        if ("limit" in request.args):
            limit = int(request.args['limit'])
        results = []
        related_words = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en" + "&limit=" + str(limit) + ("" if relation == None else ("&filter=/r/" + relation))).json()

        for edge in related_words['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']

            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (relation == None or relation == edge['rel']['label']):
                c = mysql.connection.cursor()
                d = None
                if c.execute("SELECT * FROM frequencies WHERE word=%s", (related_word + '-j',)) > 0:
                    d = c.fetchone()
                    if d[1] != None and d[2] != None:
                        results.append({
                            'word': related_word,
                            'frequency': d[1] if d != None else None,
                            'relative-frequency': d[2] if d != None else None,
                            'distance': distance(word, related_word)})

        return jsonify(results)
    else:
        return "Error: Please specify appropriate fields."


@app.route('/api/list-antonyms-of-adjective', methods=['GET'])
def api_list_antonyms_of_adjective():
    if ('word' in request.args):
        word = request.args['word']
        limit = 10
        if ("limit" in request.args):
            limit = int(request.args['limit'])
        done = []
        results = []
        start = datetime.datetime.now()

        related_words = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500").json()

        for edge in related_words['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']

            if ((datetime.datetime.now() - start).seconds < 4) and len(done) < limit and (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and ("Antonym" != edge['rel']['label']):
                antonyms = requests.get("https://api.conceptnet.io/c/en/" + related_word + "?filter=/c/en&filter=/r/Antonym&limit=2000").json()

                for edge in antonyms['edges']:
                    node = edge['end'] if edge['start']['label'] == word else edge['start']
                    antonym = node['label']

                    if ((datetime.datetime.now() - start).seconds < 4) and len(done) < limit and (not (antonym in done)) and (antonym != related_word) and ('language' in node) and (node['language'] == 'en') and ("Antonym" == edge['rel']['label']):
                        c = mysql.connection.cursor()
                        d = None
                        if c.execute("SELECT * FROM frequencies WHERE word=%s", (antonym + '-j',)) > 0:
                            d = c.fetchone()
                            if d[1] != None and d[2] != None:
                                done.append(antonym)
                                results.append({
                                    'word': antonym,
                                    'frequency': d[1] if d != None else None,
                                    'relative-frequency': d[2] if d != None else None,
                                    'distance': distance(word, antonym)})

        return jsonify(results)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/relative-frequency_distance', methods=['GET'])
def api_relative_frequency_distance():
    if ('word1' in request.args):
        word1 = request.args['word1']
    if ('word2' in request.args):
        word2 = request.args['word2']

    if word1 != None and word2 != None:
        result = {
            'distance': distance(word1, word2)
        }
        c = mysql.connection.cursor()
        d = None
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word1 + '-j',)) > 0:
            d = c.fetchone()
            if d[1] != None and d[2] != None:
                result['word1'] = d[2] if d != None else None
        d = None
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word2 + '-n',)) > 0:
            d = c.fetchone()
            if d[1] != None and d[2] != None:
                result['word2'] = d[2] if d != None else None
        return jsonify(result)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/is-adjective', methods=['GET'])
def api_is_adjective():
    if ('word' in request.args):
        word = request.args['word']
        c = mysql.connection.cursor()
        result = False
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word + '-j',)) > 0:
            result = True
        return jsonify({'word': word, 'result': result})
    else:
        return "Error: Please specify appropriate fields."

@app.teardown_appcontext
def close_connection(exception):
   db = getattr(g, '_database', None)
   if db is not None:
      db.close()

#app.run(host='0.0.0.0', port=80)
