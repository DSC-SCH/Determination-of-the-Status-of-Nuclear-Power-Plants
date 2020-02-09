## Dacon Nuclear Competition


#### pre-installed
- tensorflow == 2.0
- scikit-learn == 0.21
- numpy
- pandas
- matplotlib
- seaborn
- keras == 2.3 
- Xgboost
- lightgbm

### 대회 주제
- 안전한 원자력발전을 위해 한국수력원자력에서 제공하는 모의 운전 및 실제 데이터를 기반으로 하는 알고리즘 개발
 
 

### 데이터 설명
 - train데이터와 test데이터, train_label데이터가 각각 주어짐
      - 모사데이터와 실제데이터가 존재하고 실제데이터는 문자열이 포함되어 있지만 모사데이터는 그렇지 않음.
      - 모사데이터는 0-15초, 실제데이터는 10초에 발전소 상태가 변함.
      
 - train데이터
    1. 각 파일마다 10분간 초 단위(0-599)로 V0000-V5120 컬럼명을 가진 비식별화된 feature들이 존재.
    2. train데이터에 해당하는 train_label데이터의 label값을 파일명 기준으로 따로 합쳐주는 과정 필요
    
 - test데이터
    1. 각 파알미다 1분간 초 단위(0-60)로 V0000-V5120 컬럼명을 가진 비식별화된 feature들이 존재.
    

### 접근법
  1. 모사데이터와 실제데이터 분류
       
  2. 문자열 및 결측치처리
       - 모든 문자열과 결측치를 0으로 가정
       - on/off 등 문자열에 라벨인코딩 적용
       
  3. 적정 event_time조정
       - 실제 데이터의 event_time인 10초 사용
       - 모사 데이터의 event_time인 0~15초를 랜덤하게 사용
       
  4. 정규화 및 차원축소
       - 정규화 적용
           - 이상치에 영향을 최소하하는 RobustScaler적용
       - PCA를 이용하여 차원 축소 
       
  5. 모델생성
       - Dacon코드인 랜덤포레스트 적용
       - Xgboost 적용
       - SVM/OcSVM 적용
       - LSTM-autoencoder 적용
       - CNN 적용

  6. 모델 재생성
 
 
 ### 실행결과 1차
  1. 실제데이터는 train data에서 1개, test data에서 2개 존재
       
  2. 문자열 및 결측치처리
       - 모든 문자열과 결측치를 0으로 가정
       - on/off 등 문자열에 라벨인코딩 적용
       
  3. 적정 event_time조정
       - event_time을 0-15초 사이로 랜덤하게 적용했을 때보다 실제데이터의 event_time인 10초 사용했을 때 logloss값이 더 낮음.
       
  4. 정규화 및 차원축소
       - 정규화 적용
           - 일반적인 정규화 모델 StandardScaler적용
           - 이상치에 영향을 최소하하는 RobustScaler적용
       - PCA 적용 결과, 
         차원 축소 전후의 동일 모델 중 적용 전의 logloss값이 더 낮음. 
       
  5. 모델생성
       - Dacon코드인 랜덤포레스트 적용
       - Xgboost 적용
       - SVM/OcSVM 적용
       - LSTM-autoencoder 적용
       - CNN 적용

  6. 모델 재생성
  
  
 
   
