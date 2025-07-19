# Kaggle House Prices Competition

이 프로젝트는 Kaggle의 House Prices Competition을 위한 데이터 분석 환경입니다.

## 설치된 패키지

- numpy: 수치 계산
- pandas: 데이터 분석
- matplotlib: 데이터 시각화
- seaborn: 통계 시각화
- scikit-learn: 머신러닝
- jupyter: Jupyter 노트북 환경
- kaggle: Kaggle API 클라이언트

## 환경 설정

### 1. 가상환경 활성화
```bash
source venv/bin/activate
```

### 2. Kaggle API 키 설정

Kaggle API를 사용하려면 API 키가 필요합니다:

1. [Kaggle 웹사이트](https://www.kaggle.com)에 로그인
2. 계정 설정 > API 섹션으로 이동
3. "Create New API Token" 클릭하여 `kaggle.json` 파일 다운로드
4. 다운로드한 파일을 다음 위치에 저장:
   ```bash
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Jupyter 노트북 실행

```bash
source venv/bin/activate
jupyter notebook
```

또는 JupyterLab:
```bash
source venv/bin/activate
jupyter lab
```

## Kaggle CLI 사용법

### 대회 데이터 다운로드
```bash
kaggle competitions download -c home-data-for-ml-course
```

### 데이터 압축 해제
```bash
unzip home-data-for-ml-course.zip
```

### 제출 파일 업로드
```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "My submission message"
```

## 프로젝트 구조

```
kaggle-house-prices-competition/
├── venv/                    # 가상환경
├── requirements.txt         # 패키지 목록
├── README.md               # 이 파일
├── data/                   # 데이터 파일들 (다운로드 후)
├── notebooks/              # Jupyter 노트북들
└── submissions/            # 제출 파일들
```

## 다음 단계

1. Kaggle API 키 설정 완료
2. 대회 데이터 다운로드
3. 데이터 탐색 및 분석 시작
4. 모델 개발 및 평가
5. 예측 결과 제출

## 유용한 명령어

- 가상환경 활성화: `source venv/bin/activate`
- 가상환경 비활성화: `deactivate`
- 패키지 설치: `pip install package_name`
- 패키지 목록 업데이트: `pip freeze > requirements.txt`
- Jupyter 커널 확인: `jupyter kernelspec list` 