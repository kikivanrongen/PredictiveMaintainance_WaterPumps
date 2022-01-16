import pickle as pkl

from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Predictions(Resource):

    # send GET request
    def get(self):
        
        # retrieve data from folder
        result_file = 'results/predictions.pkl'
        predictions = pkl.load(open(result_file, 'rb'))
        data = {'predictions': predictions.tolist()}

        return {'data': data}, 200

if __name__ == '__main__':
    api.add_resource(Predictions, '/predictions')
    app.run()
