from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from comparitive import job_recommendation
from resumeParser import resume_parser

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

@app.route('/parse-resume', methods=['POST'])
def parse_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    return(resume_parser(request.files['file']))

@app.route('/recommend-jobs', methods=['POST'])
def recommend_jobs():
    sessionId = request.args.get('sessionId')
    return job_recommendation(sessionId)

@app.route('/recommend-candidates/<int:jobId>', methods=['POST'])
def recommend_candidates(jobId):
    return(candidate_recommendation(jobId))
if __name__ == '__main__':
    app.run(host='localhost', debug=True)
