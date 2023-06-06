import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 页面内容设置
# 页面名称
st.set_page_config(page_title="Readmission", layout="wide")
# 标题
st.title('The machine-learning based model to predict Readmission')
# 文本
st.write('This is a web app to predict the prob of Readmission based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction.')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

def option_name(x):
    if x == 0:
        return "no"
    if x == 1:
        return "yes"
@st.cache
def predict_quality(model, df):
    y_pred = model.predict(df, prediction_type='Probability')
    return y_pred[:, 1]

# 导入模型
model = joblib.load('catb.pkl')##导入相应的模型
st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
Age = st.sidebar.selectbox(label='Age', options=[1,2,3,4,5,6], index=1)#label里面是标签，可以随意更改
Gender = st.sidebar.selectbox(label='Gender', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Smokeing = st.sidebar.selectbox(label='Smoke', options=[0,1], format_func=lambda x: option_name(x), index=0)
Drinking = st.sidebar.selectbox(label='Drink', options=[0,1], format_func=lambda x: option_name(x), index=0)
Diabetes = st.sidebar.selectbox(label='Diabetes', options=[0,1], format_func=lambda x: option_name(x), index=0)
Heart_disease = st.sidebar.selectbox(label='Heart.disease', options=[0,1], format_func=lambda x: option_name(x), index=0)
BMI = st.sidebar.number_input(label='BMI', min_value=15.0,
                                  max_value=40.0,
                                  value=15.0,
                                  step=0.1)

WBC = st.sidebar.number_input(label='WBC', min_value=2.0,
                                  max_value=28.0,
                                  value=2.0,
                                  step=0.1)
RBC = st.sidebar.number_input(label='RBC', min_value=2.0,
                                  max_value=7.8,
                                  value=2.0,
                                  step=0.1)

PLT = st.sidebar.number_input(label='PLT', min_value=53.0,
                                  max_value=737.0,
                                  value=53.0,
                                  step=2.0)
HGB = st.sidebar.number_input(label='HGB', min_value=58.0,
                                  max_value=197.0,
                                  value=60.0,
                                  step=1.0)

HCT = st.sidebar.number_input(label='HCT', min_value=17.7,
                                  max_value=58.5,
                                  value=18.0,
                                  step=2.2)

Neu = st.sidebar.number_input(label='Neu', min_value=1.0,
                                  max_value=25.0,
                                  value=2.0,
                                  step=1.0)
PT = st.sidebar.number_input(label='PT', min_value=8.9,
                                  max_value=154.0,
                                  value=2.0,
                                  step=1.0)
Fib = st.sidebar.number_input(label='Fib', min_value=0.8,
                                  max_value=8.4,
                                  value=0.8,
                                  step=0.2)
ALB = st.sidebar.number_input(label='ALB', min_value=23.3,
                                  max_value=68.7,
                                  value=24.0,
                                  step=1.0)
AST = st.sidebar.number_input(label='AST', min_value=2.6,
                                  max_value=333.0,
                                  value=5.0,
                                  step=5.0)
Neu = st.sidebar.number_input(label='Neu.', min_value=3.3,
                                  max_value=98.1,
                                  value=5.5,
                                  step=0.1)

AST = st.sidebar.number_input(label='AST', min_value=4.4,
                                  max_value=1500.0,
                                  value=22.0,
                                  step=1.0)


features = {'Age': Age, 'Gender': Gender,
            'Smokeing': Smokeing, 'Drinking': Drinking,
            'Diabetes': Diabetes, 'Heart.disease': Heart_disease,
            'WBC': WBC, 'RBC': RBC,
            'PLT': PLT, 'HGB': HGB,
            'HCT': HCT, 'Neu': Neu,
            'PT': PT, 'Fib': Fib,
            'ALB': ALB, 'AST': AST
            }##'Age'引号里面才是数据集对应的输入特征名

features_df = pd.DataFrame([features])
#显示输入的特征
st.table(features_df)

from skimage import io,data

#显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of Readmission:")
    st.success(round(prediction[0], 4))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap.force_plot(explainer.expected_value, shap_values[0], features_df, matplotlib=True, show=False)
    plt.subplots_adjust(top=0.67,
                        bottom=0.0,
                        left=0.1,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig('test_shap.png')
    st.image('test_shap.png', caption='Individual prediction explaination', use_column_width=True)
