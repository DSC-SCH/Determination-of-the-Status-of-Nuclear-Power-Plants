# fpp2의 usmelec 예제로 실습하기
# usmelec: 1973년 1월부터 2013년 6월까지의 월별 전력량
library(forecast)
library(TSA)
library(usmelec)

# data확인
usmelec

# 시계열 데이터인가?
is.ts(usmelec)    # TRUE반환

# 1. 평활법을 이용한 여러모형 적용해보기
# 단순지수평활법 적용
# 초기값을 최적값으로 사용
usm_ses <- ses(usmelec, initial = 'optimal')
par(mfrow=c(1,2))   # 화면 분할(두개로)
plot(usmelec,main="original")
plot(usmelec,main="ses_fitted")
lines(fitted(usm_ses),col="red",pch=1,lwd=1)

# holt선형추세지수평활법 적용
usm_holt <- holt(usmelec,initial="optimal")
plot(usmelec,main="original")
plot(usmelec,main="holt_fitted")
lines(fitted(usm_holt),col="red",pch=1,lwd=1)

# holt선형추세지수평활법에 급격한 상승 완화
usm_holt_d <- holt(usmelec,initial="optimal",damped=TRUE)
plot(usmelec,main="original")
plot(usmelec,main="holt_damped_fitted")
lines(fitted(usm_holt_d),col="red",pch=1,lwd=1)

# holt-winters 가법모형
usm_hw_add <- hw(usmelec,initial = "optimal",seasonal = "additive")
plot(usmelec,main="original")
plot(usmelec,main="hw_additive")
lines(fitted(usm_hw_add),col="red",pch=1,lwd=1)

# holt-winters 승법모형
usm_hw_multi <- hw(usmelec,initial = "optimal",seasonal = "multiplicative")
plot(usmelec,main="original")
plot(usmelec,"hw_multiplicative")
lines(fitted(usm_hw_multi),col="red",pch=1,lwd=1)
## 가법모형과 승법모형 어떤 모형을 써야할지 헷갈린다면 rmse가 작은 값을 택


par(mfrow=c(1,1))   # 화면에 그림 하나 

# 2. 데이터에 적합한 예측모형 찾기
# 데이터 시각화
plot(usmelec)      # 추세 존재,계절성 존재 -> 'ets' or 'holt-winters'

# train/test data 생성
# window함수를 이용하여 사전평가데이터 사후평가데이터로 분리
usm_train <- window(usmelec,start=c(1973,1),end=c(2011,12))
usm_test <- window(usmelec,start=c(2012,1),end=c(2013,6))

# holt-winters모형 적용
for_hw_add <- hw(usm_train,seasonal = "additive",h=18)
for_hw_multi <- hw(usm_train,seasonal = "multiplicative",h=18)

# ets모형 적용
fit_ets <- ets(usm_train, model='ZZZ')
for_ets <- forecast(fit_ets,h=18)

# 모델 평가
accuracy(for_hw_add, usm_test)     # traning data set rmse = 8.19 \ test data set rmse = 8.61 
accuracy(for_hw_multi, usm_test)    # traning data set rmse = 7.61 \ test data set rmse = 8.43 
accuracy(for_ets,usm_test)           # traning data set rmse = 7.52 \ test data set rmse = 8.34 
summary(for_ets)                      # alpha = 0.27로 낮은 값을 보이고 test data set rmse가 가장 낮으며 
                                        #해당 rmse와 traning data set rmse 차이가 적음

# 실제값과 예측값 비교
plot(usm_test) 
lines(for_ets$mean,col='red')  # 예측값은 조금 다르지만 비슷한 추세를 따르고 있음

# 예측오차 평가
a <- usm_test-for_ets$mean  # 예측오차: 실제값-예측값
ggtsdisplay(a)  # ACF(자기상관함수) 및 PACF(편자기상관함수)가 신뢰구간 안에 포함되어 있음
