# Random Forest 빠른 하이퍼파라미터 튜닝 (Random Search) - 개선 버전
# 
# 개선사항:
# - 향상된 파일 경로 처리
# - 에러 처리 및 디버깅 정보 추가
# - 더 자세한 출력 정보

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import time
import warnings
import os
warnings.filterwarnings('ignore')

print("=== Random Forest Random Search - 개선 버전 ===\n")

# 데이터 로드 (현재 디렉토리에서 파일 찾기)
print("1. 데이터 로드 중...")
try:
    # 현재 디렉토리에서 파일 찾기
    if os.path.exists('data/processed/df_selected_05.pkl'):
        file_path = 'data/processed/df_selected_05.pkl'
    elif os.path.exists('../data/processed/df_selected_05.pkl'):
        file_path = '../data/processed/df_selected_05.pkl'
    elif os.path.exists('df_selected_05.pkl'):
        file_path = 'df_selected_05.pkl'
    else:
        raise FileNotFoundError("df_selected_05.pkl 파일을 찾을 수 없습니다.")
    
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"✓ 데이터 로드 성공: {file_path}")
    
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"디렉토리 내 파일: {os.listdir('.')}")
    raise

# X, y 분리
print("\n2. 데이터 전처리 중...")
if 'SalePrice' not in df.columns:
    raise ValueError("SalePrice 컬럼이 데이터에 없습니다!")
    
y = df['SalePrice']
X = df.drop('SalePrice', axis=1)

print(f"데이터 형태: X {X.shape}, y {y.shape}")
print(f"특성 목록: {list(X.columns)}")

# 교차 검증 설정
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Search를 위한 파라미터 분포 정의
print("\n3. 파라미터 분포 설정 중...")
param_distributions = {
    'n_estimators': randint(100, 1000),           # 100~999 사이 정수
    'max_depth': [None] + list(range(5, 51, 5)),  # None 또는 5~50 (5 간격)
    'min_samples_split': randint(2, 21),          # 2~20 사이 정수
    'min_samples_leaf': randint(1, 11),           # 1~10 사이 정수
    'max_features': ['sqrt', 'log2'] + [0.3, 0.5, 0.7, 0.9, 1.0],  # 다양한 특성 비율
    'max_samples': uniform(0.6, 0.4),             # 0.6~1.0 사이 실수
    'bootstrap': [True, False]                     # 부트스트랩 사용 여부
}

print("=== 파라미터 분포 ===")
for param, distribution in param_distributions.items():
    print(f"{param}: {distribution}")
    
print(f"\n총 가능한 조합: 매우 많음 (연속 분포 포함)")
print(f"Random Search로 100회 시도 예정")

# 기준선 모델 성능
print("\n4. 기준선 모델 평가 중...")
base_model = RandomForestRegressor(random_state=42)
base_scores = cross_val_score(base_model, X, y, cv=kfold, scoring='r2')
baseline_r2 = base_scores.mean()

print(f"기준선 R² 점수: {baseline_r2:.4f}")

# Random Search 설정
print("\n5. Random Search 실행 중...")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    rf, 
    param_distributions,
    n_iter=100,          # 100회 시도
    cv=kfold,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("\n=== Random Search 시작 ===")
start_time = time.time()
random_search.fit(X, y)
end_time = time.time()

print(f"\nRandom Search 완료!")
print(f"소요 시간: {end_time - start_time:.2f}초")
print(f"최적 R² 점수: {random_search.best_score_:.4f}")
print(f"성능 향상: {random_search.best_score_ - baseline_r2:.4f}")
print(f"향상률: {((random_search.best_score_ - baseline_r2) / baseline_r2 * 100):.2f}%")

# 최적 파라미터 분석
print("\n6. 최적 파라미터 분석 중...")
print("=== 최적 하이퍼파라미터 ===")
best_params = random_search.best_params_
for param, value in best_params.items():
    print(f"{param}: {value}")

# 상위 10개 결과 분석
results_df = pd.DataFrame(random_search.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[
    ['mean_test_score', 'std_test_score'] + [f'param_{p}' for p in param_distributions.keys()]
]

print("\n=== 상위 10개 결과 ===")
print(top_10.to_string(index=False))

# 최종 모델 검증
final_model = RandomForestRegressor(**best_params, random_state=42)
final_scores = cross_val_score(final_model, X, y, cv=kfold, scoring='r2')

print(f"\n=== 최종 검증 결과 ===")
print(f"R² 평균: {final_scores.mean():.4f} (±{final_scores.std():.4f})")
print(f"개별 폴드 점수: {final_scores}")

# 모델 저장
print("\n7. 모델 저장 중...")
final_model.fit(X, y)

# 모델 저장 경로 결정
try:
    model_filename = 'models/quick_tuned_rf_model_fixed.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(final_model, f)
    
    # 저장된 파일 크기 확인
    file_size = os.path.getsize(model_filename) / (1024 * 1024)  # MB
    print(f"✓ 모델이 '{model_filename}'로 저장되었습니다. (파일 크기: {file_size:.1f}MB)")
    
except Exception as e:
    print(f"❌ 모델 저장 실패: {e}")

# 특성 중요도
print("\n8. 특성 중요도 분석 중...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 특성 중요도 (Top 5) ===")
print(feature_importance.head().to_string(index=False))

# 모든 특성 중요도를 확인하고 싶은 경우
print(f"\n총 {len(feature_importance)}개 특성의 중요도가 계산되었습니다.")
print("전체 특성 중요도 합계:", feature_importance['importance'].sum())

print("\n=== 전체 특성 중요도 ===")
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:15s}: {row['importance']:.6f}")

print("\n🎉 모든 작업이 완료되었습니다!") 

