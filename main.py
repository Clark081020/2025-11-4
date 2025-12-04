import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# pip install koreanize-matplotlib 을 실행했다면
import koreanize_matplotlib
import numpy as np # 예시 데이터 생성을 위해 추가

# 1. 앱 기본 설정
st.set_page_config(layout="wide")
st.title("랜덤 포레스트 분류 모델 웹 앱 🌳")
st.markdown("---")

# 2. 데이터 로딩 (실제 앱에서는 st.file_uploader를 사용하여 사용자가 파일을 올리도록 할 수 있습니다.)
@st.cache_data
def load_data():
    # 실제 데이터를 대신하여 예시 데이터(Iris dataset과 유사)를 생성합니다.
    data = {
        'Feature_A': np.random.rand(150) * 5,
        'Feature_B': np.random.rand(150) * 4,
        'Feature_C': np.random.rand(150) * 6,
        'Target': np.random.randint(0, 3, 150) # 3개의 클래스
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# 3. 사이드바 - 사용자 입력 및 모델 설정
with st.sidebar:
    st.header("⚙️ 모델 설정 및 예측")

    # A. 모델 설정 파라미터
    n_estimators = st.slider('결정 나무 개수 (n_estimators)', 10, 200, 100)
    max_depth = st.slider('최대 깊이 (max_depth)', 2, 10, 5)

    # B. 예측을 위한 사용자 입력값 (4개의 특성 columns을 가정)
    st.subheader("새로운 데이터 입력")
    input_a = st.number_input('특성 A 값', min_value=0.0, max_value=10.0, value=df['Feature_A'].mean())
    input_b = st.number_input('특성 B 값', min_value=0.0, max_value=10.0, value=df['Feature_B'].mean())
    input_c = st.number_input('특성 C 값', min_value=0.0, max_value=10.0, value=df['Feature_C'].mean())

# 4. 데이터 전처리 및 모델 훈련
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 훈련 및 캐싱 (파라미터가 바뀌지 않으면 재훈련 방지)
@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train, n_estimators, max_depth)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 5. 메인 화면 출력
st.header("📊 분석 결과 및 모델 성능")

col1, col2 = st.columns(2)

with col1:
    st.subheader("모델 성능 평가")
    st.metric(label="Accuracy (정확도)", value=f"{accuracy:.4f}")
    
    st.text("Classification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

with col2:
    st.subheader("혼동 행렬 (Confusion Matrix) 시각화")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('혼동 행렬', fontsize=15)
    fig.colorbar(cax)
    
    # 숫자 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], va='center', ha='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
            
    ax.set_xlabel('예측 클래스', fontsize=12)
    ax.set_ylabel('실제 클래스', fontsize=12)
    
    st.pyplot(fig) # Streamlit에 Matplotlib 그래프 표시

st.markdown("---")

# 6. 예측 결과 표시
st.header("🎯 새로운 데이터에 대한 예측")

# 사용자 입력값으로 DataFrame 생성
new_data = pd.DataFrame({
    'Feature_A': [input_a],
    'Feature_B': [input_b],
    'Feature_C': [input_c]
})

# 예측
prediction = model.predict(new_data)[0]
prediction_proba = model.predict_proba(new_data)[0]

# 예측 결과 출력
st.success(f"입력값: A={input_a:.2f}, B={input_b:.2f}, C={input_c:.2f}")
st.success(f"**모델의 예측 클래스:** **{prediction}**")

# 확률 시각화
st.subheader("클래스별 예측 확률")
proba_df = pd.DataFrame({
    'Class': y.unique(),
    'Probability': prediction_proba
}).sort_values('Class')

st.bar_chart(proba_df.set_index('Class'))
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# koreanize_matplotlib 대신 사용할 코드
# 폰트 파일을 프로젝트 폴더에 저장했다고 가정
fontpath = 'NanumGothic.ttf' 
font_name = fm.FontProperties(fname=fontpath, size=10).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
