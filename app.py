from flask import Flask
from flask import request
from flask_cors import CORS
import json
from bert import *

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/answer', methods=['POST'])
def answer():
    data = request.json
    answer = get_answer(data['question'], data['text'])
    print(data['question'], answer)
    return json.dumps({'answer': answer})
