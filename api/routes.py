from flask import Blueprint, request, jsonify
from api.schemas import PredictionRequest, BatchPredictionRequest

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/v1/predict', methods=['POST'])
def predict_passenger():
    """API endpoint for single passenger prediction"""
    try:
        data = PredictionRequest(**request.json)
        # Process prediction logic here
        return jsonify({'message': 'Prediction endpoint'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400