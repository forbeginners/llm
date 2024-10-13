from flask import Flask, request, jsonify
from plannerMaker import rag_chain

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/recommend', methods=['POST'])
def recommend():
    # request_body
    payload = request.get_json()

    planner = rag_chain.invoke({
        'name': payload['planner_name'],
        'desc': payload['planner_desc'],
        'target_period': payload['target_period'],
        'repeating_days': payload['repeating_days'],
    })

    print(planner)
    # return planner

    return jsonify({
        "todo": planner.todos
    }), 200


if __name__ == '__main__':
    app.run()
