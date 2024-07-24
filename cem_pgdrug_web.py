import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from openbabel import pybel
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from openbabel import pybel
import warnings
from sklearn.exceptions import ConvergenceWarning
from rdkit import RDLogger
import os
import plotly.express as px

# Suppress various warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# 加载保存的最佳模型和指纹类型
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_model.joblib')
best_model = joblib.load(model_path)
best_fp = 'FCFP2'  # 假设ECFP4是最佳指纹类型
ACTIVITY_THRESHOLD = 0.5

def calculate_fingerprint(smiles, fp_type):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if fp_type == 'FP2':
        mol = pybel.readstring('smi', str(smiles))
        fp = mol.calcfp(fp_type)
        bitlist = fp.bits
        rawfp = np.zeros(1025)
        for b in bitlist:
            rawfp[b] = 1
        return list(rawfp)
    elif fp_type == 'FP3':
        mol = pybel.readstring('smi', str(smiles))
        fp = mol.calcfp(fp_type)
        bitlist = fp.bits
        rawfp = np.zeros(56)
        for b in bitlist:
            rawfp[b] = 1
        return list(rawfp)
    elif fp_type == 'FP4':
        mol = pybel.readstring('smi', str(smiles))
        fp = mol.calcfp(fp_type)
        bitlist = fp.bits
        rawfp = np.zeros(308)
        for b in bitlist:
            rawfp[b] = 1
        return list(rawfp)        
    #elif fp_type == 'Estate':
    #    return EStateFingerprinter.FingerprintMol(mol)
    elif fp_type == 'MACCS':
        return AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == 'ECFP2':
        return GetMorganFingerprintAsBitVect(mol, 1, nBits=1024)
    elif fp_type == 'ECFP4':
        return GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    elif fp_type == 'ECFP6':
        return GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    elif fp_type == 'FCFP2':
        return GetMorganFingerprintAsBitVect(mol, 1, nBits=1024, useFeatures=True)
    elif fp_type == 'FCFP4':
        return GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
    elif fp_type == 'FCFP6':
        return GetMorganFingerprintAsBitVect(mol, 3, nBits=1024, useFeatures=True)
    #elif fp_type == 'DLFP':
    #    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def classify_compound(smiles):
    url = "https://npclassifier.ucsd.edu/classify"
    response = requests.get(url, params={"smiles": smiles})
    if response.status_code == 200:
        data = response.json()
        data["smiles"] = smiles
        if data["pathway_results"] and data["pathway_results"][0]:
            superclass = data["pathway_results"][0]
            if superclass == "Shikimates and Phenylpropanoids":
                superclass = "Phenylpropanoids"
            elif superclass == "Amino acids and Peptides":
                superclass = "Peptides"
            data["superclass"] = superclass
            print(data["superclass"])
            return data["superclass"]
    return "Other"

def predict_activity(smiles, best_model):
    X_all = np.array(calculate_fingerprint(smiles, best_fp))
    #valid_indices = [i for i, x in enumerate(X_all) if x is not None]
    #X_all = np.array([X_all[i] for i in valid_indices])
    if X_all is not None:
        probability = best_model.predict_proba([X_all])[0][1]
        is_active = probability >= ACTIVITY_THRESHOLD
        return is_active, probability
    return None

st.title('抗毛滴虫药物分子指纹机器学习模型预测机器')

input_type = st.radio("选择输入类型", ('SMILES序列', 'CSV文件'))

if input_type == 'SMILES序列':
    smiles = st.text_input('输入SMILES序列')
    if smiles:
        is_active, probability = predict_activity(smiles, best_model)
        if is_active is not None:
            st.write(f"预测结果: {'活性' if is_active else '非活性'}")
            st.write(f"活性概率: {probability:.2f}")
        else:
            st.write("无法预测该化合物")

else:
    uploaded_file = st.file_uploader("上传CSV文件", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        results = []
        for _, row in data.iterrows():
            smiles = row['SMILES']
            is_active, probability = predict_activity(smiles, best_model)
            results.append({
                'ID': row['ID'],
                'SMILES': smiles,
                'Prediction': '活性' if is_active else '非活性' if is_active is not None else 'N/A',
                'Activity_Probability': f"{probability:.2f}" if probability is not None else 'N/A'
            })
        
        results_df = pd.DataFrame(results)
        st.write(results_df)
        
        # 提供下载链接
        csv = results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载结果为CSV",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv",
        )

        # 简单的结果统计
        st.subheader("预测结果统计")
        prediction_counts = results_df['Prediction'].value_counts()
        st.write(prediction_counts)

        # 可视化预测结果分布
        #import matplotlib.pyplot as plt
        #import matplotlib as mpl
        #from matplotlib.font_manager import FontProperties

        #fig, ax = plt.subplots()
        #results_df['Activity_Probability'] = pd.to_numeric(results_df['Activity_Probability'], errors='coerce')
        #results_df['Activity_Probability'].hist(bins=20, ax=ax)
        #ax.set_xlabel('活性概率')
        #ax.set_ylabel('频数')
        #ax.set_title('活性概率分布')
        #st.pyplot(fig)

        # 绘图代码
        fig = px.histogram(results_df, x="Activity_Probability", nbins=20, 
                        labels={"Activity_Probability": "活性概率", "count": "频数"},
                        title="活性概率分布")
        st.plotly_chart(fig)
