import os
import csv
import pymysql
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'




@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'csv_file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['csv_file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        # 파일 저장
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 데이터베이스에 저장
        if insert_csv_to_db(file_path):
            return jsonify({'message': 'CSV uploaded and data inserted successfully!'}), 200
        else:
            pass
            # return jsonify({'message': 'Failed to insert data into database'}), 500
    else:
        return jsonify({'message': 'Invalid file format. Please upload a .csv file.'}), 400


def get_db_connection():
    return pymysql.connect(
        host='192.168.0.35',
        user='root',
        password='1234',
        db='db_aiproject',
        port=3306
    )


def insert_csv_to_db(file_path):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_data = csv.reader(file)
                next(csv_data)  # 헤더 건너뛰기
                for row in csv_data:
                    # 빈 값을 NULL로 변환
                    row = [None if cell == '' else cell for cell in row]
                    cursor.execute("""
                        INSERT INTO solar_data 
                        (datetime, temperature, precipitation, windspeed, winddirection,
                         humidity, airpressure, irradiance, radiation, amountofcloud, 
                         groundtemperature, generation)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, row)
        connection.commit()
    except Exception as e:
        pass
        # print(f"Database error: {e}")
    finally:
        connection.close()


def preprocess_csv(file_path):
    clean_rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_data = csv.reader(file)
        headers = next(csv_data)  # 헤더 읽기
        for row in csv_data:
            clean_row = [None if cell == '' else cell for cell in row]  # 빈 값을 None으로 변환
            clean_rows.append(clean_row)
    return clean_rows


def fetch_24hr_data():
    """MySQL에서 24시간 데이터를 가져옵니다."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                SELECT datetime, generation
                FROM solar_data
                ORDER BY datetime DESC
                LIMIT 24;
            """
            cursor.execute(query)
            rows = cursor.fetchall()  # 데이터 가져오기
        # 데이터 변환: [{"datetime": "2022-04-01 00:00:00", "generation": 123.45}, ...]
        print('Success to getting data from database!')
        return [{"datetime": str(row[0]), "generation": row[1]} for row in rows]
    except Exception as e:
        print(f"Database error: {e}")
        return []
    finally:
        connection.close()

@app.route('/')
def index():
    data = fetch_24hr_data()
    gen = []
    for di in data:
        gen.append(di['generation'])
    for i in range(24):
        gen.insert(0, 0)
        print(gen)
    maximum = round(max(gen), 2)
    total = round(sum(gen),2)
    return render_template('index.html', chart_data=data, maximum_data=maximum, total_data=total)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)