# Mobile-Manipulator-Simulation
반도체 패키징 공정에서 그리퍼 성능 개선을 고려한 Mobile Manipulator 디스패칭 최적화 시뮬레이션 소스 코드입니다.

## 핵심 모듈 설명
* `data_loader.py`: 설비 제원 및 시나리오 데이터를 불러오는 모듈
* `graph_model.py`: NetworkX 기반의 공정 레이아웃 방향 그래프 및 교통 제약(교행, 일방, 회피 구간) 모델링 모듈
* `simulation.py`: SimPy 기반의 이산사건 시뮬레이션(DES) 메인 실행 엔진 및 디스패칭 룰(FIFO, NN, LQF, RTT, ATCS) 적용 모듈

*(기업 보안 규정에 따라 실제 공정 레이아웃 수치 및 원본 데이터셋은 비공개 또는 가상의 더미 데이터로 대체 처리되었습니다.)*
