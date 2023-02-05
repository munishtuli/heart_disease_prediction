# Web Application
# load libraries
import flask
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import shap

# initialize app
app = Flask(__name__)

# Load model
model_lr = pickle.load(open('model_lr.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form.get("age"))
    sex = str(request.form.get("sex"))
    cp = str(request.form.get("cp"))
    trestbps = float(request.form.get("trestbps"))
    chol = float(request.form.get("chol"))
    fbs = str(request.form.get("fbs"))
    restecg = str(request.form.get("restecg"))
    thalach = float(request.form.get("thalach"))
    exang = str(request.form.get("exang"))
    oldpeak = float(request.form.get("oldpeak"))
    slope = str(request.form.get("slope"))

    # new data
    x_new = pd.DataFrame(np.zeros((1, 11)), columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Sex',
                                                     'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina',
                                                     'ST_Slope'])
    x_new.iloc[0, 0] = age
    x_new.iloc[0, 1] = trestbps
    x_new.iloc[0, 2] = chol
    x_new.iloc[0, 3] = thalach
    x_new.iloc[0, 4] = oldpeak
    x_new.iloc[0, 5] = sex
    x_new.iloc[0, 6] = cp
    x_new.iloc[0, 7] = fbs
    x_new.iloc[0, 8] = restecg
    x_new.iloc[0, 9] = exang
    x_new.iloc[0, 10] = slope

    x_new.FastingBS = x_new.FastingBS.astype('int32').astype('object')

    num = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    nom = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    x_new_num, x_new_nom = x_new[num], x_new[nom]

    ohe = pickle.load(open('ohe.pkl', 'rb'))
    x_new_nom_ohe = ohe.transform(x_new_nom)
    col = ['Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
           'FastingBS_0',
           'FastingBS_1', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N',
           'ExerciseAngina_Y',
           'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
    x_new_nom_ohe = pd.DataFrame(x_new_nom_ohe, columns=col)

    x_new_prep = pd.concat([x_new_num, x_new_nom_ohe], axis=1)
    y_pred_lr = model_lr.predict(x_new_prep)
    y_pred_prob_lr = model_lr.predict_proba(x_new_prep)

    # Explain Model:SHAP
    explainer_lr = pickle.load(open('explainer_lr.pkl', 'rb'))
    shap_values_lr_new = explainer_lr.shap_values(x_new_prep)

    p = shap.force_plot(explainer_lr.expected_value, shap_values_lr_new, x_new_prep, matplotlib=True, show=False)
    plt.savefig('tmp.png')

    # Process outputs for display
    output = y_pred_lr[0]
    output_prob = np.round(y_pred_prob_lr * 100, 1).flatten()
    classes = ['No Heart Disease', 'Heart Disease']
    output_class = {classes[0]: output_prob[0], classes[1]: output_prob[1]}

    # Embedded Plot
    import base64
    from io import BytesIO
    from matplotlib.figure import Figure
    import io
    from PIL import Image
    # Image1
    # Generate the embedded figure **without using pyplot**.
    fig1 = Figure(figsize=(10, 2))
    ax = fig1.subplots()
    ax.bar(classes, output_prob, color='green', width=0.2)
    ax.set_ylabel('Prediction Probability (%)')
    ax.set_xlabel('Heart Disease')
    # Save it to a temporary buffer.
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png")
    # Embed the result in the html output.
    data1 = base64.b64encode(buf1.getbuffer()).decode("ascii")

    # Image2

    def get_encoded_img(image_path):
        img = Image.open(image_path, mode='r')
        img = img.resize((1000, 150), Image.LANCZOS)  # Resize image
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
        return my_encoded_img

    data2 = get_encoded_img('tmp.png')
    return render_template('index.html', img_data1=data1, img_data2=data2,
                           prediction_text2='Prediction Probability(%): {}'.format(output_class))


# run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
