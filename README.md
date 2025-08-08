# Trajectory-prediction-Transformer-for-Autonomous-driving
I reconfigurated Individual TF, BERT, Quantized TF, Quantized BERT.  Original version is not the complete code so i fixed it to customize the TF that i want.

이 코드는 4가지 종류의 TF로 구성되어 있습니다. TF, BERT, Quantized TF, Quantized BERT
원본 버전은 완전한 코드가 아니므로 원하는 TF로 재구성해서 완성했습니다. 일반 TF는 가장 기본적으로 알려진 TF를 따릅니다. 

code는 두가지 version으로 사용가능하다. 원본 코드가 사람궤적 예측용 코드였으므로 코드를 돌릴수 있도록 같은 형식의 data로 변환하는 코드를 추가하였다.
따라서 -eth ver (사람 궤적용 ver) 과 -vehicle ver (차량궤적용 ver) 두가지가 존재한다.
차량 궤적용으로 사용하려면 사용하려는 데이터를 같은 형식으로 변환해주는 코드로 조금만 수정하면 된다. (data_transform.py파일 수정 필요)
기본적으로 내부에 포함된 data는 참고로 argoverse trajectory forecasting 학습 data를 가공한것이다. 

*컴퓨터 상태에 따라 tensorboard에 loss가 기록되지 않을 가능성이 있다. 

[[[[[[[[[[[[[code 1 사용방법 -eth ver.]]]]]]]]]]]]]


<TF 사용하는방법>
1. 필요한 프로그램 설치
2. dataset 설정
무조건 다음과 같은 형식으로 설정되어 있어야 함
-dataset
	-dataset_name(원하는 이름)
		-train_folder
		-test_folder
		-validation_forder(optional)
		-clusters.mat(For quantizedTF)

dataset 안에는 작성자가 생성해놓은 여러 data파일들이 있음 (clusters.mat파일은 삭제할것.)

3. train_individual.py 실행
train을 하기 위해서는 여러 파라미터 값을 넣고 위 파일을 실행한다. 
ex) eth data 파일의 학습을 진행하기 위해서는 명령창에 다음과 같이 입력할것

CUDA_VISIBLE_DEVICES=0 python train_individualTF.py --dataset_name eth --name eth --max_epoch 240 --batch_size 100 --name eth_train --factor 1

4. test_individualTF.py 실행
individual TF 를 test하는 코드

ex)python test_individualTF.py --dataset_name eth --epoch 99 --name eth_train

—-------------------------------------------------------------------------------------------------------------------------
<Quantized TF 사용하는방법>
5. QuantizedTF 생성하기
GPU를 사용해 컴퓨터 계산 시간을 줄이기 위해서는 data의 형식 변환이 필요하다.
dataset의 각 dataset_name 폴더 안에는 clusters.mat파일을 생성해야하는데 이는 kmeans.py 파일을 실행하여 생성한다.
명령어는 다음과 같다.
ex) eth data 파일의  cluster_mat file을 생성하기 위해서는 명령창에 다음과 같이 입력할것
CUDA_VISIBLE_DEVICES=0 python kmeans.py --dataset_name eth

생성후에는 적절한 dataset 폴더에 집어넣어준다. (수동으로 해야함)
eth_1000_200000_scaleTrue_rotTrue 라는 파일이 생기는데 안에 있는 cluster파일만 datasets에 옮겨주면 된다.

6. Quantized 된 파일 train하기
ClassifyTF.py 를 실행하여 train시킨다.
ex)실행 명령어는 다음과 같다.

CUDA_VISIBLE_DEVICES=0 python train_quantizedTF.py --dataset_name eth --name eth --batch_size 1024

7. test와 평가를 하기
test_class.py 파일을 실행하여 test와 평가 과정을 진행한다. 
ex) 
CUDA_VISIBLE_DEVICES=0 python test_quantizedTF.py --dataset_name eth --name eth --batch_size 1024 --epoch 00030 --num_samples 20

8. visualization
training loss, validation loss, mad and fad for the test 값 그래프는 tensorboard에서 확인 가능하다.
아래 명령어를 입력하도록 한다.
ex) 
tensorboard --logdir logs
python visualize_trajectory.py

—-------------------------------------------------------------------------------------------------------------------------
<BERT 실행 방법>
1. 필요한 프로그램 설치
2. dataset 설정
무조건 다음과 같은 형식으로 설정되어 있어야 함
-dataset
	-dataset_name(원하는 이름)
		-train_folder
		-test_folder
		-validation_forder(optional)
		-clusters.mat(For quantizedTF)

dataset 안에는 작성자가 생성해놓은 여러 data파일들이 있음 (clusters.mat파일은 삭제할것.)

3. BERT 기반 (Non-Quantized) 모델 학습
실행 파일: BERT.py
ex)
CUDA_VISIBLE_DEVICES=0 python BERT.py --dataset_name eth --name eth_BERT --max_epoch 100 --batch_size 256

4. BERT 기반 (Non-Quantized) 모델 test &평가지표 출력
실행 파일: test_BERT.py
ex)
CUDA_VISIBLE_DEVICES=0 python test_BERT.py --dataset_folder datasets --dataset_name eth --name eth_BERT --epoch 00099

—-------------------------------------------------------------------------------------------------------------------------
<Quantized BERT 실행 방법>

5. Quantized data 생성
기본TF에서 생성했다면 안해도 됨
ex)
CUDA_VISIBLE_DEVICES=0 python kmeans.py --dataset_name eth

6. Quantized BERT 모델 학습
실행 전 요구사항: clusters.mat 생성-> datasets 파일의 eth 파일안에 넣을 것
실행 파일: BERT_Quantized.py

ex)
CUDA_VISIBLE_DEVICES=0 python BERT_Quantized.py --dataset_name eth --name eth_QBERT --epochs 100 --batch_size 256

7. Quantized BERT 모델 test와 평가지표 출력

ex) CUDA_VISIBLE_DEVICES=0 python test_quantized_BERT.py --dataset_folder datasets --dataset_name eth --name eth_QBERT --epoch 00099


8.  visualization
training loss, validation loss, mad and fad for the test 값 그래프는 tensorboard에서 확인 가능하다.
아래 명령어를 입력하도록 한다.
ex) 
tensorboard --logdir logs

[[[[[[[[[[[[[code 1 사용방법 -vehicle ver.]]]]]]]]]]]]]

<TF 사용하는방법>
1. 필요한 프로그램 설치
2. dataset 설정
무조건 다음과 같은 형식으로 설정되어 있어야 함
-dataset
	-dataset_name(원하는 이름)
		-train_folder
		-test_folder
		-validation_forder(optional)
		-clusters.mat(For quantizedTF)

dataset 안에는 작성자가 생성해놓은 여러 data파일들이 있음 (clusters.mat파일은 삭제할것.)

3. train_individual.py 실행
train을 하기 위해서는 여러 파라미터 값을 넣고 위 파일을 실행한다. 
(test individual.py는 존재 X)
ex) eth data 파일의 학습을 진행하기 위해서는 명령창에 다음과 같이 입력할것

CUDA_VISIBLE_DEVICES=0 python train_individualTF.py --dataset_name vehicle --name vehicle --max_epoch 240 --batch_size 100 --name vehicle_train --factor 1

4. test_individualTF.py 실행
individual TF 를 test하는 코드

ex)python test_individualTF.py --dataset_name vehicle --epoch 99 --name vehicle_train

—-------------------------------------------------------------------------------------------------------------------------

<Quantized TF 사용하는방법>
5. QuantizedTF 생성하기
GPU를 사용해 컴퓨터 계산 시간을 줄이기 위해서는 data의 형식 변환이 필요하다.
dataset의 각 dataset_name 폴더 안에는 clusters.mat파일을 생성해야하는데 이는 kmeans.py 파일을 실행하여 생성한다.
명령어는 다음과 같다.
ex) eth data 파일의  cluster_mat file을 생성하기 위해서는 명령창에 다음과 같이 입력할것
CUDA_VISIBLE_DEVICES=0 python kmeans.py --dataset_name vehicle

생성후에는 적절한 dataset 폴더에 집어넣어준다. (수동으로 해야함) 
vehicle_1000_200000_scaleTrue_rotTrue 라는 파일이 생기는데 안에 있는 cluster파일만 datasets에 옮겨주면 된다.

6. Quantized 된 파일 train하기
ClassifyTF.py 를 실행하여 train시킨다.
ex)실행 명령어는 다음과 같다.

CUDA_VISIBLE_DEVICES=0 python train_quantizedTF.py --dataset_name vehicle --name vehicle --batch_size 1024

7. test와 평가를 하기
test_class.py 파일을 실행하여 test와 평가 과정을 진행한다. 
ex) 
CUDA_VISIBLE_DEVICES=0 python test_quantizedTF.py --dataset_name vehicle --name vehicle --batch_size 1024 --epoch 00030 --num_samples 20

8. visualization
training loss, validation loss, mad and fad for the test 값 그래프는 tensorboard에서 확인 가능하다.
아래 명령어를 입력하도록 한다.
ex) 
tensorboard --logdir logs
python visualize_trajectory.py

—-------------------------------------------------------------------------------------------------------------------------
<BERT 실행 방법>
1. 필요한 프로그램 설치
2. dataset 설정
무조건 다음과 같은 형식으로 설정되어 있어야 함
-dataset
	-dataset_name(원하는 이름)
		-train_folder
		-test_folder
		-validation_forder(optional)
		-clusters.mat(For quantizedTF)

dataset 안에는 작성자가 생성해놓은 여러 data파일들이 있음 (clusters.mat파일은 삭제할것.)

3. BERT 기반 (Non-Quantized) 모델 학습
실행 파일: BERT.py
ex)
CUDA_VISIBLE_DEVICES=0 python BERT.py --dataset_name vehicle --name vehicle_BERT --max_epoch 100 --batch_size 256

4. BERT 기반 (Non-Quantized) 모델 test &평가지표 출력
실행 파일: test_BERT.py
ex)
CUDA_VISIBLE_DEVICES=0 python test_BERT.py --dataset_folder datasets --dataset_name vehicle --name vehicle_BERT --epoch 00099

—-------------------------------------------------------------------------------------------------------------------------
<Quantized BERT 실행 방법>

5. Quantized data 생성
기본TF에서 생성했다면 안해도 됨
ex)
CUDA_VISIBLE_DEVICES=0 python kmeans.py --dataset_name vehicle

6. Quantized BERT 모델 학습
실행 전 요구사항: clusters.mat 생성-> datasets 파일의 eth 파일안에 넣을 것
실행 파일: BERT_Quantized.py

ex)
CUDA_VISIBLE_DEVICES=0 python BERT_Quantized.py --dataset_name vehicle --name vehicle_QBERT --epochs 100 --batch_size 256

7. Quantized BERT 모델 test와 평가지표 출력

ex) CUDA_VISIBLE_DEVICES=0 python test_quantized_BERT.py --dataset_folder datasets --dataset_name vehicle --name vehicle_QBERT --epoch 00099


8.  visualization
training loss, validation loss, mad and fad for the test 값 그래프는 tensorboard에서 확인 가능하다.
아래 명령어를 입력하도록 한다.
ex) 
tensorboard --logdir logs
