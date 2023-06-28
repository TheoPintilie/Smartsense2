from flask import Flask, jsonify, request
import main

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Retrieve uploaded file from the request
    file = request.files['audio']

    # Save the file locally (optional)
#    file_path = 'uploads/' + file.filename
#    file.save(file_path)

    # Call the analysis function
    results = main.detect(file)  # Assuming main takes the file path as input

    # Return the analysis result as JSON response
    return jsonify(result=results)

if __name__ == '__main__':
    app.run()