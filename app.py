from flask import Flask,render_template,request
app = Flask(__name__)

import pickle,gensim
import numpy as np
from gensim.models import Word2Vec
model = pickle.load(open('model.pkl','rb'))
def generate_rep(x,vectors):
    vocab_tokens = [word for word in x if word in vectors.index_to_key]
    return np.mean(vectors.__getitem__(vocab_tokens), axis=0).tolist()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/result',methods=['post'])
def result():
    review = [i for i in request.form.values()]
    text = review[0].split()
    vec = Word2Vec(list([text]),vector_size=50,min_count=1)
    rep = generate_rep(text,vec.wv)
    out = model.predict([rep])
    return render_template('result.html',review=out)
if __name__ == '__main__':
    app.run(debug=True)