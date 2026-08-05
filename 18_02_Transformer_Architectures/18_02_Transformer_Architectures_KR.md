**Volume 18 Physical AI Foundation Models**

# 2. Transformer Architectures

## 2.1 Transformer Architecture Self Attention MHA FFN [w/Code]

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

트랜스포머(Transformer) 아키텍처는 현대의 대부분의 파운데이션 모델(Foundation Model)을 구성하는 계산적 기반(Computational Foundation)이다. 이는 대규모 언어 모델(Large Language Model), 비전 트랜스포머(Vision Transformer), 멀티모달 시스템(Multimodal System), 월드 모델(World Model), 비전-언어-행동 모델(Vision-Language-Action Model), 그리고 점점 더 많은 로봇 학습(Robot Learning) 아키텍처를 지원한다. 트랜스포머가 중요한 이유는 긴 시퀀스(Long Sequence)에 걸친 관계를 효과적으로 표현하면서도 현대의 가속기(Accelerator)에서 높은 병렬 처리(Parallel Processing)가 가능하기 때문이다. 정보를 순차적으로 처리하는 순환 신경망(Recurrent Neural Network)과 달리, 트랜스포머는 입력의 여러 요소를 동시에 분석하고 각 요소가 다른 모든 요소와 얼마나 강하게 상호작용해야 하는지를 결정할 수 있다.

트랜스포머는 입력을 하나의 분리할 수 없는 객체로 취급하지 않는다. 먼저 입력을 토큰(Token)의 시퀀스로 변환한다. 자연어에서는 토큰이 단어(Word), 서브워드(Subword), 문장 부호(Punctuation Mark), 또는 개별 문자(Character)에 해당할 수 있다. 컴퓨터 비전에서는 하나의 이미지를 여러 개의 패치(Patch)로 나누어 각각을 시각 토큰(Visual Token)으로 사용할 수 있다. 로보틱스에서는 토큰이 카메라 특징(Camera Feature), 포인트 클라우드(Point Cloud) 영역, 고유감각 상태(Proprioceptive State), 행동(Action), 언어 명령(Language Instruction), 시간 단계(Time Step), 또는 압축된 멀티모달 관측(Compressed Multimodal Observation)을 나타낼 수 있다. 토큰화(Tokenization)가 완료되면 각 요소는 네트워크가 처리할 수 있는 벡터 임베딩(Vector Embedding)으로 변환된다.

트랜스포머는 이러한 임베딩에 여러 개의 계층(Layer)을 반복적으로 적용한다. 각 계층은 일반적으로 어텐션(Attention) 서브레이어(Sub-layer)와 위치별 피드포워드 네트워크(Position-wise Feed-Forward Network)를 포함한다. 잔차 연결(Residual Connection)과 정규화(Normalization)는 안정적인 최적화를 지원하고 네트워크를 통과하는 동안 정보가 유지되도록 돕는다. 이러한 계층을 여러 번 쌓음으로써 모델은 낮은 수준의 토큰 표현(Low-Level Token Representation)을 점점 더 문맥적(Contextual)이고 추상적인 표현(Abstract Representation)으로 변환한다.

트랜스포머를 정의하는 핵심 연산은 자기 주의(Self-Attention)이다. 자기 주의는 각 토큰이 동일한 시퀀스 안의 다른 토큰들을 살펴보고 현재 자신의 표현을 위해 어떤 토큰이 중요한지를 결정할 수 있도록 한다. 이는 고정된 합성곱 필터(Convolution Filter)나 순환 은닉 상태(Recurrent Hidden State)와 근본적으로 다르다. 두 토큰 사이의 관계는 그들의 내용(Content)에 따라 동적으로 계산된다. 따라서 동일한 단어, 이미지 패치, 또는 로봇 상태도 주변 문맥에 따라 서로 다른 방식으로 상호작용할 수 있다.

예를 들어 "bank"라는 단어가 포함된 문장을 생각해 보자. 이것이 금융기관(Financial Institution)을 의미하는지 강둑(River Bank)을 의미하는지를 판단하기 위해 모델은 "money", "loan", "water", "shore"와 같은 주변 단어를 함께 살펴보아야 한다. 자기 주의는 "bank"를 나타내는 토큰이 문맥상 관련성이 높은 단어에 더 큰 중요도를 부여하도록 만든다. 로봇 작업에서는 "grasp"를 나타내는 토큰이 목표 물체(Target Object), 그리퍼 상태(Gripper State), 작업 지시(Task Instruction), 그리고 주변 장애물(Nearby Obstacle)에 동시에 주의를 기울여야 할 수도 있다. 어텐션 패턴(Attention Pattern)은 현재 장면(Scene)과 작업(Task)에 따라 계속 변화한다.

자기 주의 과정은 모든 입력 토큰을 질의(Query), 키(Key), 값(Value)이라는 세 개의 벡터로 변환하는 것에서 시작된다. 이 벡터들은 학습 가능한 선형 투영(Learned Linear Projection)을 통해 생성된다. 세 벡터는 모두 동일한 입력 임베딩에서 시작하지만 서로 다른 기능적 역할을 수행한다.

질의(Query) 벡터는 현재 토큰이 무엇을 찾고 있는지를 나타낸다. 키(Key) 벡터는 다른 토큰이 어떤 특징으로 검색될 수 있는지를 나타낸다. 값(Value) 벡터는 해당 토큰이 중요하다고 판단되었을 때 실제로 전달될 정보를 포함한다. 따라서 어텐션은 내용 기반 검색(Content-Addressable Retrieval)과 유사하게 동작한다. 하나의 질의(Query)가 여러 개의 키(Key)와 비교되고, 그 결과 얻어진 유사도(Similarity)에 따라 값(Value)들이 결합된다.

각 토큰에 대해 질의(Query)는 허용된 어텐션 범위 안에 있는 모든 토큰의 키(Key)와 비교된다. 가장 일반적으로 사용되는 유사도 척도는 내적(Dot Product)이다. 내적 값이 크면 질의와 키가 강하게 일치함을 의미하고, 값이 작으면 관련성이 낮음을 의미한다. 계산된 점수는 키 벡터 차원의 제곱근(Square Root)으로 나누어진다. 이러한 스케일링(Scaling)은 벡터 차원이 커질수록 내적 값이 지나치게 커지는 것을 방지하여, 이후 소프트맥스(Softmax) 연산에서 그래디언트(Gradient)가 불안정해지는 문제를 줄여준다.

스케일링된 점수는 소프트맥스 함수(Softmax Function)를 통과하면서 정규화된 어텐션 가중치(Attention Weight)로 변환된다. 각 질의에 대한 가중치의 합은 항상 1이 된다. 이후 모델은 값(Value) 벡터들의 가중합(Weighted Sum)을 계산한다. 그 결과는 현재 토큰의 문맥 의존 표현(Context-Sensitive Representation)이 된다. 이제 토큰은 자신의 원래 정보뿐 아니라 전체 시퀀스에서 선택된 정보까지 함께 포함하게 된다.

이 과정은 질의(Query), 키(Key), 값(Value) 행렬에 대한 어텐션 연산으로 간결하게 표현할 수 있다. 질의와 키의 곱(Query-Key Product)은 관계 점수(Relationship Score)를 생성하고, 스케일링은 수치 크기를 조절하며, 소프트맥스는 정규화된 가중치를 생성하고, 마지막으로 값(Value) 행렬과의 곱은 문맥 정보를 집계(Aggregation)한다. 비록 수식은 매우 간단하지만 실제로는 토큰 사이의 매우 유연한 정보 교환(Learned Communication)을 구현한다.

질의(Query), 키(Key), 값(Value)이 모두 동일한 시퀀스에서 생성되면 이를 자기 주의(Self-Attention)라고 한다. 반대로 질의는 하나의 시퀀스에서 오고 키와 값은 다른 시퀀스에서 오면 이를 교차 어텐션(Cross-Attention)이라고 한다. 자기 주의는 언어(Language), 이미지(Image), 로봇 궤적(Robot Trajectory), 센서 스트림(Sensor Stream) 내부의 관계를 학습하며, 교차 어텐션은 언어 명령(Language Instruction)과 시각 관측(Visual Observation), 행동 토큰(Action Token)과 환경 특징(Environment Feature)처럼 서로 다른 정보원을 연결한다.

허용되는 어텐션 범위는 모델 구조에 따라 달라진다. 인코더(Encoder) 모델은 일반적으로 양방향 어텐션(Bidirectional Attention)을 사용하여 모든 토큰이 앞뒤의 모든 토큰을 참조할 수 있도록 한다. 이는 입력 전체가 이미 주어져 있고 목적이 이해(Understanding)나 표현 학습(Representation Learning)인 경우에 적합하다. 반면 디코더 전용(Decoder-Only) 언어 모델은 인과적 어텐션(Causal Attention)을 사용하여 토큰이 미래의 토큰을 볼 수 없도록 제한한다. 각 위치는 자기 자신과 이전 토큰만 참조할 수 있다.

인과적 마스킹(Causal Masking)은 다음 토큰 예측(Next Token Prediction)에 필수적이다. 이러한 제약이 없다면 모델은 학습 과정에서 미래의 정답을 미리 볼 수 있기 때문이다. 마스크(Mask)는 소프트맥스 이전의 어텐션 점수에 적용되어 미래 위치에는 매우 작은 값을 부여한다. 결과적으로 해당 위치의 어텐션 가중치는 사실상 0이 된다. 추론(Inference) 과정에서는 모델이 한 번에 하나의 토큰만 생성하면서 이전 토큰들을 문맥(Context)으로 반복적으로 사용한다.

로보틱스에서는 과거와 현재의 관측으로 미래 행동을 예측할 때 인과적 어텐션이 매우 중요하다. 로봇 정책(Robot Policy)은 아직 발생하지 않은 미래 센서 정보를 사용할 수 없기 때문이다. 반면 오프라인 궤적 분석(Offline Trajectory Analysis), 표현 학습, 데이터 복원(Data Reconstruction), 행동 평가에서는 전체 궤적이 이미 존재하므로 양방향 어텐션을 사용할 수 있다. 따라서 어텐션 마스크(Attention Mask)의 선택은 작업의 시간적 요구사항과 운영 방식에 따라 달라진다.

하나의 어텐션 연산만으로도 유용한 관계를 학습할 수 있지만, 모든 토큰 간 관계를 하나의 유사도 공간(Similarity Space)에만 표현해야 한다는 한계가 있다. 다중 헤드 어텐션(Multi-Head Attention)은 여러 개의 어텐션 연산을 병렬로 수행함으로써 이러한 문제를 해결한다. 각 헤드(Head)는 독립적인 질의, 키, 값 투영을 가지므로 서로 다른 유형의 관계를 학습할 수 있다.

어떤 헤드는 문장의 지역적인 문법 구조(Local Syntactic Dependency)에 집중할 수 있고, 다른 헤드는 장거리 의미 관계(Long-Range Semantic Relationship)를 학습할 수 있다. 컴퓨터 비전에서는 하나의 헤드가 객체 경계(Object Boundary)를 추적하고 다른 헤드는 동일한 객체에 속한 멀리 떨어진 영역을 연결할 수 있다. 로보틱스에서는 서로 다른 헤드가 목표 물체(Target Object), 충돌 위험(Collision Risk), 시간 의존성(Temporal Dependency), 언어 접지(Language Grounding), 운동학 제약(Kinematic Constraint), 인간 움직임(Human Motion) 등을 각각 담당할 수 있다.

각 헤드의 출력은 연결(Concatenation)된 뒤 또 하나의 선형 투영(Linear Projection)을 통과한다. 이를 통해 여러 관계를 동시에 포함하는 통합 표현(Unified Representation)이 생성된다. 다중 헤드 어텐션은 각각의 헤드가 전체 임베딩 차원을 모두 사용하지 않더라도 여러 관계를 동시에 표현할 수 있도록 모델의 표현력을 크게 향상시킨다.

헤드의 개수는 중요한 아키텍처 설계 요소이다. 헤드 수가 많아질수록 다양한 관계를 표현할 수 있지만, 전체 임베딩 차원이 고정되어 있다면 각 헤드가 담당하는 차원은 줄어든다. 헤드가 너무 적으면 다양한 관계를 충분히 표현하지 못하고, 너무 많으면 중복되거나 거의 사용되지 않는 헤드가 생길 수 있다. 따라서 현대의 트랜스포머는 헤드 수, 헤드 차원, 메모리 사용량, 계산 효율성, 하드웨어 호환성 사이의 균형을 고려하여 설계된다.

모든 어텐션 헤드가 명확하게 해석 가능한 역할을 가지는 것은 아니다. 일부 헤드는 매우 특화되어 있지만, 일부는 거의 중복된 기능을 수행하기도 한다. 실제 연구에서는 일부 헤드를 제거해도 성능 저하가 거의 없는 경우가 보고되었다. 그럼에도 다중 헤드 어텐션은 모델이 여러 개의 독립적인 정보 통로를 통해 관계를 조직할 수 있도록 해주기 때문에 여전히 매우 효과적이다.

자기 주의(Self-Attention)는 장거리 의존성(Long-Range Dependency)을 효과적으로 학습할 수 있다. 하나의 토큰이 단 하나의 계층만 거쳐도 멀리 떨어진 다른 토큰과 직접 연결될 수 있기 때문이다. 순환 신경망은 정보를 전달하기 위해 여러 순차 단계를 거쳐야 하며, 이 과정에서 정보가 손실될 위험이 있다. 합성곱 신경망(Convolutional Network)은 먼 위치를 연결하기 위해 많은 계층이나 큰 커널(Kernel)이 필요하다. 자기 주의는 이러한 문제를 해결하여 긴 문서, 장시간 대화, 비디오, 로봇 궤적 처리에 매우 적합하다.

그러나 이러한 유연성에는 계산 비용이 따른다. 표준 자기 주의는 모든 토큰을 모든 다른 토큰과 비교하기 때문에 계산량과 메모리 사용량이 시퀀스 길이의 제곱에 비례하여 증가한다. 토큰 수가 두 배가 되면 어텐션 행렬(Attention Matrix)의 크기는 약 네 배가 된다. 따라서 긴 문서, 고해상도 이미지, 긴 비디오, 밀집 포인트 클라우드(Dense Point Cloud), 장기 로봇 데이터(Long-Horizon Robot Data)를 처리할 때 큰 제약이 된다.

이 문제를 해결하기 위해 다양한 효율적 어텐션(Efficient Attention) 기법이 개발되었다. 희소 어텐션(Sparse Attention)은 일부 지역 또는 전역 영역만 참조하도록 제한한다. 윈도우 어텐션(Windowed Attention)은 시퀀스나 이미지를 작은 영역으로 나누어 처리한다. 선형 어텐션(Linear Attention)은 전체 어텐션 행렬을 명시적으로 계산하지 않도록 연산을 재구성한다. 플래시 어텐션(Flash Attention)은 GPU의 메모리 접근 방식을 최적화하여 동일한 어텐션을 훨씬 효율적으로 계산한다. 검색(Retrieval) 메커니즘은 일부 정보를 외부 메모리에 저장하고 필요한 정보만 선택적으로 불러온다.

피지컬 AI에서는 멀티모달 로봇 데이터가 매우 긴 시퀀스를 생성하므로 효율적 어텐션이 특히 중요하다. 하나의 로봇은 여러 대의 카메라, LiDAR, 관절 상태(Joint State), 힘 센서(Force Sensor), 작업 명령(Task Instruction), 지도(Map), 최근 행동(Action)을 동시에 처리해야 한다. 모든 원시 데이터를 고해상도 토큰으로 변환하는 것은 현실적으로 계산량이 너무 크다. 따라서 실제 시스템은 모달리티별 인코더(Modality-Specific Encoder), 시간적 서브샘플링(Temporal Subsampling), 공간 압축(Spatial Compression), 잠재 토큰(Latent Token), 선택적 어텐션(Selective Attention)을 함께 사용한다.

어텐션만으로는 하나의 트랜스포머 계층을 완성할 수 없다. 토큰들이 어텐션을 통해 정보를 교환한 후에는 각각의 토큰이 피드포워드 네트워크(Feed-Forward Network, FFN)를 통과한다. FFN은 모든 토큰 위치에 동일한 파라미터를 사용하여 독립적으로 적용된다. FFN은 토큰 사이의 정보를 직접 섞지는 않으며, 이미 문맥 정보가 반영된 각 토큰 내부의 특징을 변환하는 역할을 수행한다.

일반적인 피드포워드 네트워크는 두 개의 선형 변환(Linear Transformation)과 그 사이의 비선형 활성화 함수(Nonlinear Activation Function)로 구성된다. 첫 번째 선형 변환은 임베딩 차원을 더 큰 중간 차원(Intermediate Dimension)으로 확장한다. 활성화 함수는 비선형성을 제공하여 복잡한 함수를 표현할 수 있도록 만든다. 마지막 선형 변환은 다시 원래의 모델 차원으로 축소한다.

중간 차원은 일반적으로 임베딩 차원의 몇 배 크기로 설정된다. 이러한 확장은 모델이 더 넓은 특징 공간(Feature Space)에서 정보를 재구성하고 변환할 수 있도록 한다. 어텐션 서브레이어는 어떤 정보를 가져올 것인지를 결정하고, 피드포워드 네트워크는 가져온 정보를 어떻게 처리할 것인지를 결정한다. 두 구성 요소는 서로 보완적인 역할을 수행한다.

초기의 트랜스포머는 주로 ReLU(Rectified Linear Unit)를 활성화 함수로 사용하였다. 이후에는 GELU(Gaussian Error Linear Unit), GLU(Gated Linear Unit), SwiGLU, GEGLU와 같은 게이트 기반 활성화 함수(Gated Activation Function)가 널리 사용되었다. 게이트형 피드포워드 네트워크는 하나의 투영 결과를 이용하여 다른 투영 결과의 통과량을 조절한다. 이러한 방식은 최적화와 모델 품질을 향상시키는 경우가 많지만, 파라미터 수와 계산량에도 영향을 준다.

피드포워드 네트워크는 트랜스포머 전체 파라미터의 상당 부분을 차지한다. 많은 모델에서는 FFN의 파라미터 수가 어텐션 계층보다 더 많다. 따라서 FFN은 부분적으로 분산 지식 저장소(Distributed Knowledge Store)의 역할도 수행한다. 일부 뉴런(Neuron)은 특정 개념이나 패턴에 강하게 반응하지만, 대부분의 지식은 하나의 뉴런이 아니라 많은 파라미터에 분산되어 저장된다.

FFN은 위치별(Position-wise) 연산이므로 계산 효율도 높다. 모든 토큰을 하나의 대규모 행렬 곱(Matrix Multiplication)으로 병렬 처리할 수 있으며, 이는 GPU와 AI 가속기에서 매우 효율적으로 실행된다. 동일한 연산이 모든 토큰에 적용되지만, 어텐션 이후 각 토큰이 서로 다른 문맥 정보를 가지고 있으므로 최종 출력은 서로 다르게 된다.

트랜스포머 계층은 일반적으로 어텐션과 FFN을 잔차 연결(Residual Connection)로 결합한다. 잔차 연결은 서브레이어의 입력을 출력에 더하는 방식이다. 각 계층이 전체 표현을 새롭게 생성할 필요 없이 필요한 수정(Modification)만 학습하면 되므로 그래디언트 흐름(Gradient Flow)이 개선되고 매우 깊은 네트워크도 안정적으로 학습할 수 있다.

정규화(Normalization) 역시 매우 중요하다. 계층 정규화(Layer Normalization)는 각 토큰 내부의 특징을 정규화하여 학습 과정에서 활성값(Activation Scale)의 안정성을 유지한다. 일반적으로 두 가지 구조가 사용된다. 후정규화(Post-Normalization)는 잔차 연결 이후 정규화를 수행하며, 전정규화(Pre-Normalization)는 어텐션 또는 FFN에 들어가기 전에 먼저 정규화를 수행한다. 매우 깊은 트랜스포머에서는 전정규화가 더 안정적인 학습을 제공하는 경우가 많다.

이러한 구성 요소의 순서는 최적화에도 영향을 미친다. 대표적인 전정규화 블록은 먼저 입력을 정규화하고, 다중 헤드 어텐션을 수행한 뒤 잔차 연결을 적용한다. 이후 다시 정규화를 수행하고 FFN을 통과한 다음 또 한 번 잔차 연결을 수행한다. 이러한 구조를 여러 번 반복하면 트랜스포머의 핵심 백본(Backbone)이 형성된다.

트랜스포머는 자기 주의만으로는 순서를 이해하지 못하기 때문에 위치 정보(Positional Information)가 반드시 필요하다. 위치 인코딩(Positional Encoding)이 없다면 동일한 토큰 집합은 순서와 관계없이 거의 동일한 어텐션 관계를 형성하게 된다. 언어, 비디오, 제어 궤적(Control Trajectory), 공간 데이터는 모두 순서와 위치에 크게 의존한다.

절대 위치 임베딩(Absolute Positional Embedding)은 각 위치마다 고유한 표현을 부여한다. 사인·코사인 위치 인코딩(Sinusoidal Encoding)은 수학적 함수를 이용하여 위치를 구분한다. 상대 위치 표현(Relative Positional Representation)은 절대 위치보다 토큰 사이의 거리를 표현한다. 회전 위치 임베딩(Rotary Positional Embedding, RoPE)은 질의와 키를 위치에 따라 회전시키는 방식으로 최근의 디코더 전용 언어 모델에서 널리 사용되고 있다.

비전과 피지컬 AI에서는 위치 인코딩이 2차원 이미지 좌표(Image Coordinate), 3차원 공간 위치(3D Location), 카메라 시점(Camera Viewpoint), 시간 정보(Time Stamp), 관절 번호(Joint Index), 센서 식별자(Sensor Identity), 로봇 좌표계(Robot Coordinate Frame)를 동시에 표현해야 할 수도 있다. 하나의 멀티모달 로봇 트랜스포머는 여러 종류의 위치 정보를 함께 사용해야 하며, 정보가 어디에서 왔고, 언제 관측되었으며, 어떤 센서에서 생성되었고, 로봇의 어느 부분과 관련되는지를 동시에 구분해야 한다.

인코더 전용(Encoder-Only), 디코더 전용(Decoder-Only), 인코더-디코더(Encoder-Decoder) 구조는 동일한 핵심 구성 요소를 서로 다른 방식으로 배치한 것이다. 인코더 전용 모델은 양방향 자기 주의를 사용하여 입력 전체를 처리하고 문맥 표현(Contextual Representation)을 생성한다. 이러한 구조는 분류(Classification), 검색(Retrieval), 인식(Perception), 표현 학습에 적합하다.

디코더 전용 모델은 인과적 자기 주의를 사용하여 자기회귀(Auto-Regressive) 방식으로 시퀀스를 생성한다. 많은 작업을 조건부 생성(Conditional Generation)으로 표현할 수 있기 때문에 대규모 언어 모델에서는 디코더 전용 구조가 가장 널리 사용된다. 로보틱스에서도 디코더 전용 트랜스포머는 관측과 언어 명령을 입력받아 행동 토큰(Action Token)이나 행동 청크(Action Chunk)를 생성할 수 있다.

인코더-디코더 모델은 서로 분리된 두 개의 스택(Stack)을 가진다. 인코더는 입력을 처리하고, 디코더는 출력을 생성한다. 디코더는 이전에 생성한 토큰에 대해 인과적 자기 주의를 수행하고, 동시에 교차 어텐션(Cross-Attention)을 이용하여 인코더의 표현을 참조한다. 이러한 구조는 번역(Translation), 멀티모달 생성(Multimodal Generation), 그리고 관측과 행동을 분리하여 처리하는 로봇 제어 시스템에 적합하다.

교차 어텐션은 비전-언어-행동 모델(Vision-Language-Action Model)에서 특히 중요하다. 시각 인코더(Visual Encoder)는 이미지를 시각 토큰으로 변환하고, 언어 인코더 또는 디코더는 작업 지시를 처리한다. 교차 어텐션은 행동 생성 과정이 두 모달리티의 정보를 동시에 참조하도록 만든다. 예를 들어 "빨간 컵을 집어라(Pick up the red cup)"라는 명령에서는 "빨간 컵"이라는 언어 표현과 실제 이미지 속 해당 영역을 연결한 후 적절한 행동 시퀀스를 생성해야 한다.

트랜스포머 학습은 대규모 그래디언트 기반 최적화(Gradient-Based Optimization)를 이용한다. 순전파(Forward Pass)에서는 임베딩이 어텐션과 FFN을 통과하여 출력 확률이나 잠재 표현(Latent Representation)을 생성한다. 손실 함수(Loss Function)는 출력과 학습 목표를 비교한다. 역전파(Backpropagation)는 모든 투영 행렬, 정규화 파라미터, 임베딩, FFN 가중치에 대한 그래디언트를 계산하며, 최적화기(Optimizer)는 이후 손실을 줄이도록 파라미터를 업데이트한다.

트랜스포머는 대부분의 핵심 연산이 행렬 곱(Matrix Multiplication)이기 때문에 매우 잘 확장된다. 이러한 연산은 데이터 병렬화(Data Parallelism), 텐서 병렬화(Tensor Parallelism), 파이프라인 병렬화(Pipeline Parallelism), 시퀀스 병렬화(Sequence Parallelism), 전문가 병렬화(Expert Parallelism)를 이용하여 여러 가속기에 쉽게 분산할 수 있다. 혼합 정밀도(Mixed Precision) 연산은 처리량을 높이고 메모리 사용량을 줄인다. 이러한 하드웨어 친화성이 트랜스포머가 기존의 시퀀스 모델을 대체하게 된 중요한 이유 중 하나이다.

그러나 대규모 트랜스포머는 막대한 메모리를 요구한다. 학습 과정에서는 모델 파라미터뿐 아니라 그래디언트, 최적화기 상태(Optimizer State), 중간 활성값(Intermediate Activation)도 저장해야 한다. 긴 시퀀스에서는 어텐션 행렬이 특히 큰 메모리를 차지한다. 따라서 그래디언트 체크포인팅(Gradient Checkpointing), 활성값 재계산(Activation Recomputation), 파라미터 샤딩(Parameter Sharding), 최적화기 상태 분할(Optimizer-State Partitioning), 플래시 어텐션(Flash Attention) 등이 대규모 학습에서 필수적으로 사용된다.

추론(Inference)에서는 또 다른 제약이 존재한다. 자기회귀 디코더에서는 이전 단계에서 계산된 키(Key)와 값(Value)을 키-값 캐시(Key-Value Cache)에 저장할 수 있다. 이후 새로운 토큰을 생성할 때는 전체 시퀀스를 다시 계산하지 않고 저장된 정보를 재사용한다. 이는 계산량을 크게 줄여주지만, 컨텍스트가 길어질수록 메모리 사용량은 증가한다.

로봇에서는 키-값 캐시를 이용하여 최근 센서 정보와 행동 이력을 유지할 수 있지만, 계속 변화하는 관측 데이터 때문에 추가적인 문제가 발생한다. 오래된 토큰은 더 이상 중요하지 않을 수 있으며, 새로운 고주파 센서 데이터는 캐시를 매우 빠르게 증가시킨다. 따라서 실제 시스템에서는 슬라이딩 윈도우(Sliding Window), 메모리 압축(Memory Compression), 이벤트 기반 토큰 선택(Event-Based Token Selection), 계층적 메모리(Hierarchical Memory)가 필요할 수 있다.

기본적인 트랜스포머 구조는 매우 강력하지만, 이것만으로 실제 세계를 이해하거나 안전한 행동을 보장하는 것은 아니다. 자기 주의는 데이터 안의 통계적 관계만 학습한다. 학습 데이터에 실제 물리적 상호작용이 충분하지 않다면 모델은 움직임을 언어적으로 설명할 수는 있어도 실제 동역학(Dynamics)은 이해하지 못할 수 있다. 따라서 피지컬 AI는 관측, 행동, 결과를 함께 연결하는 데이터셋과 학습 목표가 필요하다.

로봇 트랜스포머는 언어 목표(Language Goal), 시각 장면(Visual Scene), 고유감각 피드백(Proprioceptive Feedback), 이전 행동을 모두 어텐션으로 통합할 수 있다. FFN은 이러한 통합 표현을 계획(Planning)과 제어(Control)에 적합한 특징으로 변환한다. 최종 출력 계층(Output Head)은 이산 행동 토큰(Discrete Action Token), 연속 제어값(Continuous Control), 궤적 웨이포인트(Trajectory Waypoint), 가치 추정(Value Estimate), 미래 잠재 상태(Future Latent State)를 예측할 수 있다.

계층적 로봇 구조(Hierarchical Robot Architecture)에서는 대형 트랜스포머를 상위 의사결정 계층에 배치할 수 있다. 이 모델은 작업 지시를 이해하고, 물체를 인식하며, 기술(Skill)을 선택하고, 행동 청크를 생성한다. 이후 하위 제어기가 이를 정확한 모터 명령으로 변환한다. 이러한 구조는 트랜스포머의 높은 표현 능력을 활용하면서도 결정론적 실시간 제어와 안전 메커니즘을 유지할 수 있도록 한다.

트랜스포머는 교차 체화 학습(Cross-Embodiment Learning)도 지원할 수 있다. 서로 다른 로봇 플랫폼의 관측과 행동을 체화 메타데이터(Embodiment Metadata)가 포함된 토큰으로 표현할 수 있다. 자기 주의는 공통 작업 구조를 찾고, FFN은 이를 공통 잠재 공간(Common Latent Space)으로 변환한다. 이후 어댑터(Adapter)나 행동 디코더(Action Decoder)가 이를 각 로봇의 제어 공간(Control Space)으로 변환할 수 있다.

그러나 트랜스포머에도 한계는 존재한다. 내부 동작은 해석하기 어렵고, 어텐션 가중치가 항상 신뢰할 수 있는 설명을 제공하는 것은 아니다. 또한 대규모 모델은 방대한 데이터와 계산 자원을 요구하며, 잘못된 상관관계를 암기하거나 학습 분포 밖 환경(Out-of-Distribution Environment)에서 예측 불가능하게 동작하거나 과도한 확신을 가진 출력을 생성할 수 있다. 실제 물리 시스템에서는 이러한 위험을 줄이기 위해 불확실성 추정(Uncertainty Estimation), 독립적인 안전 제어기(Independent Safety Controller), 런타임 모니터링(Runtime Monitoring), 검증된 비상 동작(Fallback Behavior)이 반드시 필요하다.

트랜스포머는 현재도 계속 발전하고 있다. 효율적 어텐션(Efficient Attention), 전문가 혼합 FFN(Mixture-of-Experts Feed-Forward Network), 멀티모달 토큰화(Multimodal Tokenization), 상태공간(State-Space) 하이브리드, 순환 메모리(Recurrent Memory), 하드웨어 최적화 아키텍처(Hardware-Aware Architecture)가 계속 연구되고 있다. 그러나 기본 구조는 여전히 동일하다. 토큰은 자기 주의를 통해 정보를 교환하고, 다중 헤드 어텐션은 다양한 관계를 학습하며, 피드포워드 네트워크는 문맥 정보를 비선형적으로 변환하고, 잔차 연결과 정규화는 깊고 안정적인 학습을 가능하게 한다.

이러한 구성 요소를 이해하는 것은 피지컬 AI 파운데이션 모델(Physical AI Foundation Model)을 설계하는 데 필수적이다. 자기 주의는 시간, 센서, 물체, 언어 명령, 행동 사이의 관계를 어떻게 연결할 것인지를 결정한다. 다중 헤드 어텐션은 여러 종류의 관계를 동시에 표현할 수 있도록 해준다. 피드포워드 네트워크는 이러한 관계를 유용한 내부 지식으로 변환하는 비선형 표현 능력을 제공한다. 이 세 가지 요소는 함께 트랜스포머 기반 시스템이 문맥을 이해하고, 여러 모달리티를 통합하며, 시퀀스를 예측하고, 실제 행동을 생성하는 핵심 계산 메커니즘을 구성한다.

## 2.2 Vision Transformer ViT and Variants DeiT MAE [w/Code]

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

비전 트랜스포머(Vision Transformer, ViT)는 현대 컴퓨터 비전(Computer Vision) 분야에서 가장 중요한 아키텍처 변화 가운데 하나이다. 오랫동안 합성곱 신경망(Convolutional Neural Network, CNN)은 이미지 분류(Image Classification), 객체 검출(Object Detection), 의미 분할(Semantic Segmentation), 그리고 시각 표현 학습(Visual Representation Learning)을 지배해 왔다. 이러한 성공은 지역성(Locality), 이동 불변성(Translation Equivariance), 계층적 특징 추출(Hierarchical Feature Extraction)과 같은 강력한 귀납적 편향(Inductive Bias)에 기반하고 있었다. 그러나 트랜스포머(Transformer)는 이미지 이해 역시 시퀀스 모델링(Sequence Modeling) 문제로 표현할 수 있음을 보여주었다. 이미지를 여러 개의 패치(Patch)로 나누고 이를 토큰(Token)으로 취급함으로써, 원래 자연어 처리를 위해 개발된 자기 주의(Self-Attention) 메커니즘을 시각 정보에도 적용할 수 있게 된 것이다.

ViT의 핵심 아이디어는 매우 단순하다. 입력 이미지를 일정한 크기의 패치로 나눈 후, 각 패치를 펼쳐(Flatten) 하나의 벡터(Vector)로 만들고 학습 가능한 임베딩 공간(Embedding Space)으로 투영한다. 이렇게 생성된 패치 임베딩(Patch Embedding)은 언어 모델(Language Model)의 토큰 시퀀스(Token Sequence)와 동일한 형태를 갖는다. 이후 위치 정보(Positional Information)를 추가하여 각 패치가 원래 이미지의 어느 위치에서 왔는지를 구분할 수 있도록 한다. 이렇게 만들어진 시퀀스는 다중 헤드 자기 주의(Multi-Head Self-Attention), 피드포워드 네트워크(Feed-Forward Network), 정규화(Normalization), 잔차 연결(Residual Connection)로 구성된 표준 트랜스포머 인코더(Transformer Encoder)를 통과한다.

예를 들어 하나의 이미지가 224×224 픽셀이고 이를 16×16 픽셀 크기의 패치로 분할한다고 가정해 보자. 이 경우 가로와 세로 각각 14개의 패치가 생성되므로 전체적으로 196개의 시각 토큰(Visual Token)이 만들어진다. 각 패치는 이미지의 작은 사각형 영역에 해당하는 픽셀 값을 포함한다. 이를 펼친 뒤 선형 투영(Linear Projection)을 수행하면 트랜스포머 전체에서 사용하는 동일한 차원의 벡터가 생성된다. 결과적으로 트랜스포머 내부에서는 이미지를 더 이상 2차원 픽셀 배열(Two-Dimensional Pixel Array)로 처리하지 않고, 정렬된 시각 토큰 시퀀스로 처리하게 된다.

일반적으로 시퀀스의 맨 앞에는 특별한 분류 토큰(Classification Token)이 추가된다. 이 토큰은 이미지의 특정 영역을 나타내는 것이 아니라, 자기 주의를 수행하면서 모든 패치 토큰과 정보를 교환하고 이미지 전체의 정보를 점진적으로 통합하는 역할을 한다. 트랜스포머 인코더를 통과한 뒤 분류 토큰의 최종 표현은 이미지 분류와 같은 작업을 수행하기 위한 예측 헤드(Prediction Head)에 전달된다. 일부 구조에서는 별도의 분류 토큰 대신 모든 패치 토큰에 대한 전역 평균 풀링(Global Average Pooling)을 사용하기도 한다.

위치 임베딩(Positional Embedding)은 매우 중요한 요소이다. 자기 주의만으로는 패치들의 공간적 배치(Spatial Arrangement)를 이해할 수 없기 때문이다. 위치 정보가 없다면 두 개의 패치 토큰을 서로 바꾸어도 모델은 이를 본질적으로 동일한 입력으로 해석하게 된다. 원래의 ViT는 학습 가능한 절대 위치 임베딩(Learned Absolute Positional Embedding)을 사용하였지만, 이후에는 상대 위치(Relative Position), 회전 위치 임베딩(Rotary Positional Embedding), 2차원 위치 편향(Two-Dimensional Positional Bias), 계층적 공간 인코딩(Hierarchical Spatial Encoding) 등 다양한 방식이 제안되었다. 이러한 방법들은 이미지 영역 간의 기하학적 관계(Geometric Relationship)를 보다 효과적으로 유지하도록 도와준다.

트랜스포머 인코더는 모든 패치 토큰 사이에서 자기 주의를 수행한다. 따라서 하나의 패치는 이미지 안의 다른 모든 패치와 직접 정보를 교환할 수 있다. 이것은 ViT와 기존 합성곱 신경망의 가장 큰 차이점 가운데 하나이다. 합성곱 계층은 처음에는 매우 작은 지역 영역(Local Neighborhood)만 관찰하며, 멀리 떨어진 정보를 통합하기 위해 여러 계층을 거쳐야 한다. 반면 ViT는 하나의 어텐션 계층만으로도 서로 멀리 떨어진 두 패치 사이의 관계를 직접 학습할 수 있다.

이러한 전역 상호작용(Global Interaction)은 객체의 구조와 장면의 문맥을 이해하는 데 매우 유용하다. 예를 들어 자동차 바퀴의 일부를 포함한 패치는 멀리 떨어진 자동차 차체 영역과 직접 연결될 수 있다. 사람을 포함한 영역은 다른 위치에 있는 물체와 연결될 수 있다. 로봇의 시각 인식에서는 그리퍼(Gripper)가 보이는 패치가 목표 물체(Target Object), 주변 장애물(Obstacle), 작업 테이블(Work Surface), 또는 사람의 손(Human Hand)과 직접 관계를 형성할 수 있다. 자기 주의는 이러한 관계를 고정된 합성곱 커널(Convolution Kernel)이 아니라 현재 장면에 따라 동적으로 계산한다.

다중 헤드 자기 주의(Multi-Head Self-Attention)는 여러 종류의 시각적 관계를 동시에 표현할 수 있도록 한다. 하나의 헤드는 국부적인 질감(Local Texture)에 집중하고, 다른 헤드는 객체의 경계(Object Boundary)를 추적하며, 또 다른 헤드는 반복되는 패턴(Repeated Pattern)이나 장면 전체의 구조(Large-Scale Scene Structure)를 학습할 수 있다. 각 헤드의 출력은 결합되어 하나의 통합 표현(Unified Representation)으로 변환된다. 계층이 깊어질수록 모델은 점점 더 풍부하고 추상적인 시각 표현을 형성하게 된다.

각 트랜스포머 계층의 피드포워드 네트워크(Feed-Forward Network)는 각 토큰을 독립적으로 비선형 변환한다. 자기 주의는 어떤 패치들이 서로 정보를 교환해야 하는지를 결정하고, 피드포워드 네트워크는 교환된 정보를 어떻게 변환할 것인지를 결정한다. 잔차 연결은 이전의 표현을 유지하며 안정적인 학습을 지원하고, 계층 정규화(Layer Normalization)는 활성값(Activation)의 크기를 안정적으로 유지한다.

ViT는 충분히 큰 데이터셋으로 학습할 경우 트랜스포머가 뛰어난 이미지 인식 성능을 달성할 수 있음을 보여주었다. 여기서 중요한 점은 "충분히 큰 데이터셋"이라는 조건이다. 합성곱 신경망과 비교하면 ViT는 이미지의 지역성(Locality)과 이동 불변성에 대한 내재적 가정을 상대적으로 적게 가지고 있다. 합성곱 구조는 인접한 픽셀들이 서로 관련되어 있으며 동일한 패턴이 이미지의 어느 위치에서도 나타날 수 있다고 가정한다. 반면 ViT는 이러한 관계를 대부분 데이터로부터 직접 학습해야 한다.

그 결과 초기의 Vision Transformer는 강력한 합성곱 신경망을 능가하기 위해 매우 큰 규모의 사전학습(Pretraining)을 필요로 하는 경우가 많았다. 학습 데이터가 적을 경우에는 샘플 효율성(Sample Efficiency)이 낮을 수도 있었다. 그러나 충분한 데이터가 제공되면 이러한 제약은 오히려 장점으로 바뀌었다. ViT는 전역적인 관계를 학습하고, 모델 규모가 커질수록 성능이 잘 확장되며, 다양한 하위 작업(Downstream Task)으로 효과적으로 전이될 수 있었다.

ViT의 확장 특성(Scaling Behavior)은 트랜스포머를 범용 시각 백본(General Visual Backbone)으로 자리 잡게 만들었다. 모델을 크게 만들고, 데이터셋을 확대하며, 학습 시간을 늘릴수록 성능은 꾸준히 향상되었다. 사전학습된 ViT는 이미지 분류(Image Classification), 객체 검출(Object Detection), 의미 분할(Semantic Segmentation), 인스턴스 분할(Instance Segmentation), 자세 추정(Pose Estimation), 시각 질의응답(Visual Question Answering), 그리고 멀티모달 학습(Multimodal Learning) 등 다양한 작업으로 쉽게 미세조정(Fine-Tuning)될 수 있었다. 패치 기반 표현(Patch-Based Representation)은 언어 트랜스포머(Language Transformer)와도 자연스럽게 연결되었다.

그러나 원래의 ViT에도 한계는 존재하였다. 전역 자기 주의(Global Self-Attention)는 토큰 수의 제곱에 비례하여 계산량이 증가한다. 이미지 해상도가 증가하면 패치 수 역시 급격히 증가한다. 패치 크기를 줄이면 공간 해상도는 향상되지만 토큰 수와 계산량이 크게 증가한다. 예를 들어 패치 크기를 16×16에서 8×8로 줄이면 동일한 이미지에 대해 토큰 수는 약 네 배가 된다.

이러한 상충 관계는 밀집(Dense) 시각 작업에서 특히 중요하다. 이미지 분류는 전역적인 레이블만 필요하므로 비교적 큰 패치도 허용된다. 그러나 객체 검출이나 의미 분할은 매우 세밀한 공간 정보를 필요로 한다. 로봇은 작은 물체, 좁은 이동 공간, 케이블의 가장자리, 공구 끝부분, 파지점(Grasp Point), 안전과 관련된 장애물 등을 정확하게 인식해야 한다. 따라서 실제 Vision Transformer는 계층적 처리(Hierarchical Processing), 지역 윈도우(Local Window), 다중 해상도 특징 피라미드(Multi-Scale Feature Pyramid), 적응형 해상도(Adaptive Resolution), 또는 합성곱-트랜스포머 하이브리드 구조(Hybrid Convolution-Transformer Architecture)를 함께 사용한다.

데이터 효율적 이미지 트랜스포머(Data-efficient Image Transformer), 즉 DeiT는 ViT의 가장 큰 약점 가운데 하나였던 대규모 사설 데이터셋에 대한 의존성을 해결하기 위해 개발되었다. DeiT는 보다 접근 가능한 규모의 데이터셋만으로도 Vision Transformer를 효과적으로 학습시킬 수 있음을 보여주었다. 핵심 기여는 새로운 트랜스포머 구조를 제안한 것이 아니라, 훨씬 더 효율적인 학습 방법론(Training Methodology)을 제시한 것이었다.

DeiT는 데이터 증강(Data Augmentation), 정규화(Regularization), 최적화(Optimization), 그리고 지식 증류(Knowledge Distillation)를 적극적으로 활용하였다. 랜덤 증강(Random Augmentation), 믹스업(Mixup), 컷믹스(CutMix), 반복 증강(Repeated Augmentation), 확률적 깊이(Stochastic Depth), 레이블 스무딩(Label Smoothing), 세밀하게 조정된 학습 스케줄(Learning Schedule) 등을 통해 과적합을 줄이고 일반화 성능을 향상시켰다. 이를 통해 트랜스포머의 성능은 단순히 모델 구조나 데이터 크기뿐 아니라 전체 학습 전략(Training Recipe)에 크게 의존한다는 사실이 입증되었다.

DeiT의 가장 독창적인 요소 가운데 하나는 증류 토큰(Distillation Token)이다. 전통적인 지식 증류는 큰 교사 모델(Teacher Model)의 출력 확률이나 중간 특징을 작은 학생 모델(Student Model)이 모방하도록 학습시킨다. DeiT는 여기에 별도의 증류 토큰을 추가하여, 분류 토큰(Classification Token)은 실제 정답(Label)을 학습하고 증류 토큰은 교사 모델의 예측을 학습하도록 하였다.

증류 토큰은 다른 토큰들과 마찬가지로 자기 주의에 참여하며 모든 이미지 패치로부터 정보를 수집한다. 그러나 이 토큰의 학습 목표는 실제 정답이 아니라 교사 모델의 출력이다. 이를 통해 트랜스포머 학생 모델은 사람이 제공한 레이블뿐 아니라 강력한 교사 모델의 지식도 동시에 흡수할 수 있다. 실제로 교사 모델은 대부분 성능이 우수한 합성곱 신경망이었다.

이러한 구조는 매우 중요한 의미를 가진다. 합성곱 신경망과 트랜스포머는 서로 다른 시각 특징을 학습하고 서로 다른 오류를 발생시키기 때문이다. 합성곱 신경망으로부터 증류된 트랜스포머는 지역 특징(Local Feature)에 대한 강점을 유지하면서도 전역적인 관계를 학습하는 트랜스포머의 장점을 동시에 활용할 수 있다. 따라서 DeiT는 서로 다른 아키텍처를 경쟁 관계가 아니라 상호 보완적인 구조로 결합한 대표적인 사례가 되었다.

지식 증류는 엣지 시스템(Edge System) 배포에서도 매우 중요한 역할을 한다. 대형 교사 모델은 오프라인 학습에 사용하고, 보다 작은 학생 모델은 로봇, 자율주행 차량, 드론, 임베디드 장치(Embedded Device)에 배포할 수 있다. 학생 모델은 교사 모델의 대부분의 시각 지식을 유지하면서도 메모리 사용량, 지연 시간(Latency), 에너지 소비(Energy Consumption)를 크게 줄일 수 있다.

ViT와 DeiT는 지도학습(Supervised Learning) 기반의 뛰어난 성능을 보여주었지만, 여전히 클래스 레이블(Class Label)이나 교사 모델의 출력이 필요하였다. 마스크드 오토인코더(Masked Autoencoder, MAE)는 이를 넘어 자기지도학습(Self-Supervised Learning)을 이용한 새로운 시각 사전학습 방법을 제안하였다. MAE는 언어 모델에서 사용된 마스크드 모델링(Masked Modeling)의 개념을 이미지에 적용하였다.

마스크드 언어 모델링(Masked Language Modeling)에서는 문장의 일부를 가리고 이를 복원하도록 학습한다. MAE는 동일한 원리를 이미지 패치에 적용한다. 이미지의 상당 부분을 제거한 뒤 모델이 보이지 않는 영역을 복원하도록 학습한다. 학습 목표는 입력 이미지 자체에서 생성되므로 사람이 직접 라벨을 만들 필요가 없다. 따라서 방대한 양의 비라벨 이미지(Unlabeled Image)를 그대로 학습에 사용할 수 있다.

MAE의 가장 큰 특징은 비대칭 인코더-디코더(Asymmetric Encoder-Decoder) 구조이다. 인코더는 보이는 패치만 처리하고, 제거된 패치에 대해서는 계산하지 않는다. 대부분의 패치가 제거되므로 인코더는 매우 짧은 시퀀스만 처리하면 된다. 이후 작은 디코더(Decoder)가 보이는 패치의 표현과 마스크 토큰(Mask Token)을 함께 입력받아 전체 이미지를 복원한다.

이러한 비대칭 구조는 계산 효율을 크게 향상시킨다. 예를 들어 전체 패치의 75%를 제거하였다면 가장 비용이 큰 인코더는 남은 25%의 패치만 처리하면 된다. 디코더는 비교적 작은 구조이며 사전학습 과정에서만 사용된다. 사전학습이 끝나면 디코더는 제거되고 인코더만 다양한 하위 작업의 시각 백본으로 사용된다.

높은 마스킹 비율(High Masking Ratio)은 매우 중요한 요소이다. 인접한 픽셀과 패치는 매우 높은 중복성을 가지므로 일부만 가리면 단순한 보간(Interpolation)만으로도 쉽게 복원할 수 있다. 그러나 이미지 대부분을 가리면 모델은 제한된 정보만으로 객체의 형태(Object Shape), 장면 구조(Scene Layout), 질감(Texture), 문맥(Context)을 추론해야 한다. 이를 통해 인코더는 보다 의미 있는 시각 표현을 학습하게 된다.

복원의 목표는 원시 픽셀 값(Raw Pixel Value)일 수도 있고 정규화된 패치 표현(Normalized Patch Representation)일 수도 있다. 겉으로 보기에는 단순한 픽셀 복원처럼 보이지만, 대부분의 이미지가 제거된 상태에서 이를 성공적으로 복원하려면 객체와 장면을 깊이 이해해야 한다. 이러한 과정에서 학습된 표현은 이미지 분류, 객체 검출, 의미 분할 등 다양한 시각 작업으로 매우 잘 전이된다.

MAE는 대조학습(Contrastive Learning)과도 차이가 있다. 대조학습은 서로 관련된 이미지 표현은 가깝게 만들고 관련 없는 표현은 멀어지도록 학습한다. 반면 MAE는 제거된 정보를 복원하는 데 초점을 둔다. 대조학습은 다양한 증강(Augmentation)에 대한 불변성(Invariance)을 강조하는 반면, MAE는 이미지 내부 구조 자체를 모델링한다. 실제 시스템에서는 이 두 가지 방법을 함께 사용하거나 순차적으로 적용하기도 한다.

피지컬 AI에서는 마스크드 시각 모델링(Masked Visual Modeling)이 특히 중요한 의미를 가진다. 로봇은 끊임없이 카메라 영상(Camera Image), 깊이 영상(Depth Image), 열화상(Thermal Image), 점유 지도(Occupancy Map), 포인트 클라우드(Point Cloud)를 수집하지만 대부분은 라벨이 없다. 모든 프레임을 사람이 직접 라벨링하는 것은 현실적으로 불가능하다. MAE는 이러한 방대한 센서 데이터를 자기지도학습 자원으로 활용할 수 있게 만든다.

로봇의 시각 모델은 제거된 이미지 패치, 가려진 깊이 영역, 누락된 포인트 클라우드, 일시적으로 사용할 수 없는 센서 채널 등을 복원하도록 학습될 수 있다. 이를 통해 가림(Occlusion), 센서 장애(Sensor Dropout), 강한 빛(Glare), 그림자(Shadow), 모션 블러(Motion Blur), 부분 관측(Partial Observation)에 더욱 강인한 표현을 학습할 수 있다. 또한 단순한 지역 질감(Local Texture)에 의존하기보다 장면 전체의 구조(Scene Structure)를 추론하도록 만든다.

ViT, DeiT, MAE의 관계는 시각 트랜스포머의 발전 과정을 잘 보여준다. ViT는 이미지를 토큰 시퀀스로 표현하여 트랜스포머로 처리할 수 있음을 보여주었다. DeiT는 효율적인 학습 전략과 지식 증류를 통해 데이터 효율성을 향상시켰다. MAE는 마스크 복원을 이용한 대규모 자기지도학습이 가능함을 입증하였다.

세 모델은 모두 동일한 패치 토큰 기반(Patch Token Backbone)을 사용하지만, 학습 목표와 활용 방식은 서로 다르다. ViT는 기본적인 시각 트랜스포머 구조이자 지도학습 기반의 시각 백본이다. DeiT는 데이터 효율성을 높이기 위한 지도학습 및 지식 증류 기반 학습 전략이다. MAE는 전이 가능한 시각 인코더를 학습하기 위한 자기지도 사전학습 프레임워크(Self-Supervised Pretraining Framework)이다.

실제 시스템에서는 이들 방법을 별도로 사용하지 않는다. 하나의 시각 인코더는 먼저 MAE를 이용하여 대규모 비라벨 이미지로 사전학습을 수행하고, 이후 DeiT 방식으로 지도학습 미세조정을 수행할 수 있다. 마지막으로 지식 증류를 통해 엣지 장치에 적합한 작은 모델로 압축할 수 있다. 이는 자기지도 표현 학습에서 지도학습 적응, 그리고 효율적인 배포까지 이어지는 하나의 완전한 파이프라인을 구성한다.

Vision Transformer는 멀티모달 파운데이션 모델(Multimodal Foundation Model)과도 자연스럽게 결합된다. 시각 패치 토큰은 언어 모델(Language Model)의 표현 공간으로 투영될 수 있으며, 교차 어텐션(Cross-Attention)이나 통합 자기 주의(Unified Self-Attention)를 통해 이미지 영역과 단어, 작업 지시, 질문, 행동이 연결된다. 이러한 구조는 비전-언어(Vision-Language) 및 비전-언어-행동(Vision-Language-Action) 모델의 핵심이 된다.

로봇 파운데이션 모델(Robot Foundation Model)에서는 시각 인코더가 ViT 또는 그 계층적 변형(Hierarchical Variant)이 될 수 있다. 시각 인코더는 카메라 영상을 시각 토큰으로 변환하고, 이러한 토큰은 언어 토큰(Language Token), 고유감각 상태(Proprioceptive State), 행동 이력(Action History), 작업 문맥(Task Context)과 융합된다. 이후 정책 디코더(Policy Decoder)가 행동(Action), 궤적(Trajectory), 또는 행동 청크(Action Chunk)를 생성한다.

패치 크기와 시각 해상도는 로보틱스에서 매우 중요한 설계 요소이다. 큰 패치는 계산량을 줄이지만 작은 안전 관련 물체를 놓칠 수 있다. 작은 패치는 더 많은 정보를 유지하지만 토큰 수와 지연 시간을 증가시킨다. 이를 해결하기 위한 방법 가운데 하나가 다중 해상도 처리(Multi-Resolution Processing)이다. 낮은 해상도의 전역 영상은 전체 장면을 이해하고, 일부 중요한 영역은 높은 해상도로 처리하여 목표 물체, 공구, 장애물, 사람의 손 등을 정밀하게 분석한다.

시간 정보(Temporal Information)는 또 다른 과제이다. 기본 ViT는 하나의 이미지만 처리한다. 그러나 로봇은 시간에 따른 변화와 움직임을 이해해야 한다. 비디오 트랜스포머(Video Transformer)는 여러 프레임의 패치를 함께 토큰화하여 시공간 토큰(Spatiotemporal Token)을 생성한다. 하지만 토큰 수는 매우 빠르게 증가한다. 따라서 시간적 서브샘플링(Temporal Subsampling), 튜블릿 임베딩(Tubelet Embedding), 분해형 어텐션(Factorized Attention), 순환 메모리(Recurrent Memory), 잠재 압축(Latent Compression) 등이 계산량을 줄이기 위해 사용된다.

깊이 정보와 3차원 정보도 함께 활용될 수 있다. RGB 영상은 깊이 지도(Depth Map), 스테레오 특징(Stereo Feature), LiDAR 투영(LiDAR Projection), 포인트 클라우드 토큰(Point Cloud Token)과 결합될 수 있다. 모달리티 임베딩(Modality Embedding)은 각 토큰이 어떤 센서에서 생성되었는지를 표시하며, 기하학적 위치 인코딩(Geometric Positional Encoding)은 카메라 보정(Camera Calibration)과 공간 관계를 유지한다. 이를 통해 트랜스포머는 단순한 외형뿐 아니라 거리와 형태까지 함께 이해할 수 있다.

ViT 기반 모델은 자율주행과 로봇 내비게이션에서 자유 공간(Free Space), 랜드마크(Landmark), 장애물, 의미 영역(Semantic Region)을 인식할 수 있다. 조작(Manipulation)에서는 물체를 인식하고, 어포던스(Affordance)를 추정하며, 파지점을 찾고, 접촉 영역(Contact Region)을 추적할 수 있다. 검사 로봇(Inspection Robot)에서는 표면 결함(Surface Defect), 열 패턴(Thermal Pattern), 구조 부품(Structural Component), 이상 현상(Anomaly)을 분석할 수 있다.

그러나 트랜스포머의 어텐션을 곧바로 완전한 시각적 설명으로 해석해서는 안 된다. 어텐션 맵(Attention Map)은 어떤 패치가 표현에 영향을 주는지를 보여줄 수는 있지만, 항상 정확한 인과적 설명(Causal Explanation)을 제공하는 것은 아니다. 모델은 배경, 질감, 데이터셋의 편향과 같은 잘못된 단서를 사용할 수도 있다. 따라서 시각화(Visualization), 가림 실험(Occlusion Test), 개념 분석(Concept Analysis), 반사실적 평가(Counterfactual Evaluation)를 통해 모델을 지속적으로 검증해야 한다.

강건성(Robustness) 역시 중요한 문제이다. Vision Transformer는 조명(Lighting), 카메라 특성(Camera Property), 날씨(Weather), 시점(Viewpoint), 배경(Background), 물체 외형(Object Appearance)이 바뀌면 성능이 저하될 수 있다. 자기지도학습은 이러한 일반화 성능을 향상시키지만 문제를 완전히 해결하지는 못한다. 산업 및 로봇 환경에서는 블러(Blur), 노이즈(Noise), 가림, 압축(Compression), 센서 장애, 새로운 환경에 대한 충분한 검증이 필요하다.

지연 시간(Latency)과 메모리(Memory)도 실제 배포에서 중요한 제약이 된다. 큰 ViT는 특히 높은 해상도에서 많은 계산량을 요구한다. 양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation), 토큰 병합(Token Merging), 조기 종료(Early Exiting), 효율적 어텐션(Efficient Attention)은 이러한 비용을 줄이는 데 사용된다. 엣지 시스템에서는 작은 DeiT 기반 모델을 사용하고, 보다 큰 MAE 사전학습 인코더는 데이터센터나 오프라인 학습 환경에서 활용할 수 있다.

미래의 시각 트랜스포머는 적응형 계산(Adaptive Computation) 방향으로 발전하고 있다. 모든 이미지 패치를 동일하게 처리하는 대신, 중요한 영역만 더 많은 계산을 수행하는 방식이다. 배경은 압축하고, 불확실하거나 작업과 관련된 영역에는 더 많은 계산 자원을 할당한다. 로보틱스에서는 이는 능동 인식(Active Perception)과 유사하다. 현재 목표, 안전 위험, 예상 정보 가치(Expected Information Value)에 따라 계산 자원을 동적으로 배분하는 것이다.

또 다른 발전 방향은 월드 모델(World Model)과의 통합이다. ViT 인코더는 현재 관측을 잠재 토큰(Latent Token)으로 변환하고, 시간 모델(Temporal Model)은 행동 이후 이 토큰이 어떻게 변화할지를 예측한다. 마스크드 모델링도 단순한 이미지 복원을 넘어 미래 상태(Future State)의 예측으로 확장될 수 있다. 이를 통해 정적인 시각 인식과 행동 기반 예측(Action-Conditioned Prediction)이 하나의 시스템으로 연결된다.

따라서 ViT, DeiT, MAE는 단순히 서로 다른 모델 이름이 아니라 피지컬 AI를 위한 세 가지 핵심 원리를 나타낸다. ViT는 유연한 전역 시각 추론(Global Visual Reasoning)을 제공하고, DeiT는 효율적인 학습과 교사 모델 기반 압축의 중요성을 보여주며, MAE는 비라벨 센서 데이터로부터 대규모 자기지도학습을 수행할 수 있는 방법을 제공한다.

이 세 가지 기술은 함께 로봇의 시각 지능을 위한 강력한 기반을 형성한다. 피지컬 AI 시스템은 MAE를 이용하여 방대한 로봇 영상을 자기지도학습하고, 지식 증류를 통해 엣지 환경에 적합한 모델을 생성하며, 트랜스포머 어텐션을 이용하여 시각 정보와 언어, 로봇 상태, 메모리, 행동을 통합할 수 있다. 이러한 기술들의 결합은 단순한 이미지 분류 시스템을 넘어 추론과 실제 물리적 상호작용을 지원하는 범용 시각 지능(General-Purpose Visual Intelligence)으로 발전하는 기반이 된다.

## 2.3 Efficient Attention Linear Flash Attention [w/Code]

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

트랜스포머(Transformer) 아키텍처의 성공은 자기 주의(Self-Attention) 메커니즘의 효과성과 밀접하게 연결되어 있다. 자기 주의는 모든 토큰(Token)이 다른 모든 토큰과 직접 상호작용할 수 있도록 하여, 모델이 장거리 의존성(Long-Range Dependency)을 학습하고, 문맥 정보(Contextual Information)를 통합하며, 매우 풍부한 내부 표현(Internal Representation)을 구축할 수 있게 한다. 그러나 이러한 유연성에는 상당한 계산 비용이 따른다. 트랜스포머 모델의 파라미터 수(Parameter Count), 컨텍스트 길이(Context Length), 이미지 해상도(Image Resolution), 비디오 길이(Video Duration), 멀티모달 복잡성(Multimodal Complexity)이 계속 증가함에 따라 어텐션 연산 자체가 점점 핵심 계산 병목(Computational Bottleneck)이 되고 있다. 효율적 어텐션 메커니즘은 표준 어텐션(Standard Attention)의 표현력을 최대한 유지하면서 이러한 한계를 극복하기 위해 개발되었다.

기존 자기 주의의 계산 복잡도는 시퀀스 길이의 제곱에 비례하여 증가한다. 시퀀스에 N개의 토큰이 포함되어 있다면 어텐션 행렬(Attention Matrix)은 N×N개의 유사도 점수(Similarity Score)를 포함한다. 모든 토큰이 다른 모든 토큰과 비교되기 때문에 계산량과 메모리 사용량 모두 이차적으로 증가한다. 짧은 시퀀스에서는 이러한 비용을 감당할 수 있지만, 긴 문서, 고해상도 이미지, 장시간 비디오, 대규모 포인트 클라우드(Point Cloud), 그리고 로봇 센서 이력(Robot Sensor History)에서는 이러한 제곱 관계가 빠르게 비현실적인 수준에 도달한다.

그 이유를 이해하려면 어텐션 계산 자체를 살펴볼 필요가 있다. 질의(Query) 벡터와 키(Key) 벡터를 곱하여 모든 토큰 쌍 사이의 유사도 점수를 생성한다. 이 점수는 소프트맥스(Softmax)를 통해 정규화된 뒤 값(Value) 벡터와 곱해져 문맥 표현(Contextual Representation)을 만든다. 각 개별 연산은 비교적 단순하지만 토큰 수가 증가하면 어텐션 행렬의 크기는 매우 커진다. 1,000개의 토큰을 가진 시퀀스는 백만 개의 쌍별 어텐션 점수를 요구한다. 10,000개의 토큰을 가진 시퀀스는 1억 개의 점수를 요구한다. 따라서 메모리 사용량과 계산량은 급격히 증가한다.

파운데이션 모델(Foundation Model)은 점점 더 긴 컨텍스트를 요구하고 있다. 현대의 언어 모델(Language Model)은 책, 소프트웨어 저장소(Software Repository), 과학 문서, 대화, 법률 기록, 그리고 수만 개에서 수십만 개의 토큰을 포함하는 검색 증강 지식베이스(Retrieval-Augmented Knowledge Base)를 처리한다. 비전 트랜스포머(Vision Transformer)는 더욱 세밀한 이미지를 처리하고, 비디오 트랜스포머(Video Transformer)는 수백 또는 수천 개의 프레임을 처리한다. 피지컬 AI(Physical AI) 시스템은 카메라 스트림(Camera Stream), LiDAR 관측(LiDAR Observation), 로봇 상태(Robot State), 힘 센서(Force Sensor), 언어 명령(Language Instruction), 지도(Map), 행동 이력(Action History), 월드 모델(World Model)을 긴 멀티모달 시퀀스 하나로 결합한다. 이러한 조건에서는 표준 어텐션의 계산 비용이 빠르게 감당하기 어려운 수준으로 증가한다.

따라서 핵심 문제는 어텐션이 유용한가가 아니라, 모델의 능력을 크게 저하시키지 않으면서 어텐션의 계산 효율을 어떻게 개선할 것인가이다. 연구자들은 이 문제에 여러 방향으로 접근하였다. 일부 방법은 토큰 간 비교 횟수를 줄인다. 다른 방법은 전체 어텐션 행렬을 명시적으로 생성하지 않도록 계산 순서를 재구성한다. 일부는 하드웨어의 특성을 활용하고, 다른 일부는 어텐션 연산을 수학적으로 근사한다. 이러한 접근법을 종합적으로 효율적 어텐션(Efficient Attention)이라고 부른다.

대표적인 범주 가운데 하나는 희소 어텐션(Sparse Attention)이다. 모든 토큰이 모든 토큰을 참조하도록 하는 대신, 희소 어텐션은 상호작용을 신중하게 선택된 일부 토큰으로 제한한다. 하나의 토큰은 가까운 이웃 토큰, 일정 간격으로 배치된 위치, 전역 요약 토큰(Global Summary Token), 또는 학습된 중요 영역에만 주의를 기울일 수 있다. 자연 신호에는 지역성(Locality)이 존재하는 경우가 많기 때문에 어텐션을 제한해도 대부분의 유용한 정보를 유지하면서 계산량을 크게 줄일 수 있다.

윈도우 기반 어텐션(Window-Based Attention)은 시퀀스나 이미지를 작은 지역 영역으로 나누어 처리하는 방식이다. 토큰은 자신이 속한 윈도우 안에서만 상호작용하며, 이를 통해 쌍별 비교 횟수가 크게 줄어든다. 이후의 계층적 아키텍처(Hierarchical Architecture)는 윈도우 이동(Shifting), 병합(Merging), 또는 전역 계층(Global Layer)을 통해 서로 다른 윈도우 사이의 정보를 교환할 수 있다. 이 접근법은 인접한 이미지 패치(Image Patch) 사이에 강한 관계가 존재하는 비전 트랜스포머 분야에서 특히 성공적으로 사용되었다.

또 다른 방법군은 전체 어텐션을 수학적으로 근사하는 방식이다. 모든 질의와 키의 유사도를 명시적으로 계산하는 대신, 어텐션 수식을 시퀀스 길이의 제곱이 아니라 선형적으로 증가하는 연산으로 재구성한다. 이러한 방식들을 통틀어 선형 어텐션(Linear Attention)이라고 한다.

선형 어텐션은 소프트맥스 어텐션 수식 안의 행렬 곱셈 순서를 적절한 수학적 변환을 통해 바꿀 수 있다는 점에서 출발한다. 전체 어텐션 행렬을 생성하는 대신, 질의와 키에 특징 매핑(Feature Mapping)을 적용하여 더 작은 중간 연산으로 계산을 재구성한다. 그 결과 계산 복잡도는 토큰 수의 제곱이 아니라 대략 토큰 수에 비례하여 증가하게 된다.

이러한 감소는 긴 컨텍스트를 다룰 때 매우 큰 의미를 가진다. 시퀀스 길이가 두 배가 되면 표준 어텐션의 계산량은 대략 네 배 증가한다. 반면 이상적인 선형 어텐션은 약 두 배만 증가한다. 따라서 컨텍스트 길이가 계속 커질수록 계산 절감 효과는 더욱 커진다.

선형 어텐션은 구현 방식에 따라 여러 형태로 나뉜다. 일부 방법은 소프트맥스를 다른 커널 함수(Kernel Function)로 대체한다. 다른 방법은 지수형 유사도(Exponential Similarity)를 무작위 특징 투영(Random Feature Projection)으로 근사한다. 일부는 정규화 성질을 유지하는 양의 특징 매핑(Positive Feature Mapping)을 사용하고, 다른 일부는 시퀀스를 점진적으로 처리하는 순환식 표현(Recurrent Formulation)을 도출한다. 수학적 세부사항은 다르지만 모두 전체 어텐션 행렬을 명시적으로 만들지 않는다는 공통 목표를 가진다.

커널 기반 선형 어텐션(Kernel-Based Linear Attention)은 대표적인 접근법이다. 소프트맥스를 통해 지수 유사도를 직접 계산하는 대신 질의와 키 벡터를 더 높은 차원의 특징 표현으로 변환한다. 이 변환 공간에서의 내적(Inner Product)은 원하는 어텐션 관계를 근사한다. 변환된 특징은 효율적으로 누적될 수 있기 때문에 계산량은 시퀀스 길이에 선형적으로 비례한다.

무작위 특징 방식(Random Feature Method)은 무작위 투영을 이용하여 커널 함수를 근사한다. 정확한 유사도를 계산하는 대신 확률적 특징 표현(Stochastic Feature Representation)을 통해 이를 추정한다. 일반적으로 무작위 특징의 수를 늘리면 근사 품질이 향상되지만 계산 비용도 증가한다. 따라서 효율성과 정확도 사이에 또 다른 상충 관계(Trade-off)가 발생한다.

일부 선형 어텐션 모델은 명시적인 쌍별 상호작용 대신 순환 상태 업데이트(Recurrent State Update)를 사용한다. 이전의 모든 토큰 관계를 저장하는 대신 새로운 토큰이 입력될 때마다 갱신되는 압축된 요약 상태(Compressed Summary State)를 유지한다. 이를 통해 시퀀스 길이와 관계없이 일정한 메모리 사용량으로 스트리밍 추론(Streaming Inference)을 수행할 수 있다. 이러한 방식은 연속적인 센서 처리와 온라인 로보틱스(Online Robotics)에 특히 적합하다.

그러나 선형 어텐션에도 문제가 있다. 정확한 소프트맥스 어텐션은 여러 유용한 수학적 성질을 가지지만 근사 방식은 이를 완전히 보존하지 못할 수 있다. 긴 시퀀스에서는 근사 오차가 누적되어 모델 품질이 저하될 수 있다. 일부 선형 방식은 특정 작업에서는 잘 동작하지만 정밀한 전역 관계를 요구하는 복잡한 추론에서는 성능이 낮아질 수 있다. 따라서 효율적 어텐션은 계산 절감과 표현 충실도(Representational Fidelity) 사이의 균형을 고려해야 한다.

또 다른 주요 연구 방향은 어텐션을 수학적으로 근사하는 것이 아니라, 현대 하드웨어에서 정확한 어텐션을 훨씬 효율적으로 구현하는 것이다. 플래시어텐션(FlashAttention)은 이 분야에서 가장 영향력 있는 기술 가운데 하나이다. 플래시어텐션은 어텐션의 수학적 정의를 바꾸지 않고 계산 순서를 재구성하여 메모리 이동(Memory Movement)을 최소화하면서도 정확한 수치 결과를 유지한다.

플래시어텐션의 핵심 동기는 하드웨어 구조에서 나온다. 현대 GPU는 매우 높은 계산 처리량(Computational Throughput)을 제공하지만 실제 성능은 산술 연산보다 메모리 대역폭(Memory Bandwidth)에 의해 제한되는 경우가 많다. 표준 어텐션은 큰 중간 행렬을 고대역폭 메모리(High-Bandwidth Memory)와 온칩 메모리(On-Chip Memory) 사이에서 반복적으로 이동시킨다. 이러한 메모리 전송은 상당한 시간과 에너지를 소비하며 전체 실행 시간의 주요 부분을 차지할 수 있다.

플래시어텐션은 행렬 연산을 작은 타일(Tile) 단위로 나누어 이 병목을 해결한다. 전체 어텐션 행렬을 메모리에 생성하지 않고, 고속 온칩 공유 메모리(Shared Memory)에 들어갈 수 있는 작은 블록으로 계산한다. 각 블록은 불러오기, 처리, 정규화, 누적 과정을 거친 뒤 다음 블록으로 이동한다. 따라서 중간 어텐션 행렬을 외부 메모리에 완전히 저장할 필요가 없다.

이 방식은 기존 소프트맥스 어텐션과 동일한 결과를 유지하면서도 메모리 트래픽(Memory Traffic)을 크게 줄인다. 성능 향상은 수학적 알고리즘을 변경한 것이 아니라 하드웨어에서 계산이 수행되는 방식을 변경한 데서 나온다. 따라서 플래시어텐션은 모델 정확도를 희생하지 않으면서 처리량을 높이고, 메모리 사용량을 줄이며, 확장성을 개선한다.

플래시어텐션의 중요한 혁신 가운데 하나는 수치적으로 안정적인 온라인 소프트맥스(Online Softmax) 계산이다. 기존 구현은 모든 어텐션 점수를 계산한 뒤 소프트맥스 정규화를 수행한다. 플래시어텐션은 블록 단위로 정규화를 점진적으로 수행하면서도 전체 계산과 동일한 수치 결과를 유지한다. 이를 통해 근사 오차 없이 블록 기반 처리가 가능해진다.

플래시어텐션은 알고리즘 효율성과 하드웨어 효율성이 모두 중요하다는 사실을 보여준다. 수학적으로 동일한 알고리즘이라도 메모리 접근 패턴(Memory Access Pattern), 캐시 활용(Cache Utilization), 동기화 오버헤드(Synchronization Overhead), 커널 융합(Kernel Fusion)에 따라 실제 성능은 크게 달라질 수 있다. 모델 규모가 증가할수록 하드웨어 인식 최적화(Hardware-Aware Optimization)의 중요성은 더욱 커진다.

후속 버전에서는 추가적인 개선이 이루어졌다. 플래시어텐션-2(FlashAttention-2)는 GPU 스레드(Thread)와 스트리밍 멀티프로세서(Streaming Multiprocessor) 사이의 작업 분배를 개선하여 활용률을 높이고 동기화 오버헤드를 줄였다. 더 효율적인 스케줄링(Scheduling)을 통해 기본 어텐션 연산을 바꾸지 않고도 대형 모델에서 더 높은 실효 처리량(Effective Throughput)을 달성할 수 있다.

최근의 구현은 가변 시퀀스 길이(Variable Sequence Length), 인과 마스킹(Causal Masking), 그룹드 쿼리 어텐션(Grouped Query Attention), 멀티쿼리 어텐션(Multi-Query Attention), 혼합 정밀도(Mixed Precision), 대규모 하드웨어 구성을 더욱 효율적으로 지원하도록 발전하고 있다. 플래시어텐션은 GPU 아키텍처와 함께 지속적으로 진화하며, 머신러닝 알고리즘과 컴퓨터 아키텍처 사이의 긴밀한 관계를 보여준다.

그룹드 쿼리 어텐션(Grouped Query Attention, GQA)은 또 다른 효율성 개선 방법이다. 기존의 다중 헤드 어텐션(Multi-Head Attention)은 각 어텐션 헤드마다 독립적인 질의, 키, 값 벡터를 사용한다. 멀티쿼리 어텐션(Multi-Query Attention)은 여러 질의 헤드가 동일한 키와 값을 공유하여 추론 중 메모리 요구량을 줄인다. 그룹드 쿼리 어텐션은 여러 질의 헤드를 그룹으로 나누고 각 그룹이 키와 값을 공유하도록 하여 두 방식 사이의 절충안을 제공한다. 최근의 많은 대규모 언어 모델은 모델 품질을 대부분 유지하면서 키-값 캐시(Key-Value Cache) 크기를 크게 줄일 수 있기 때문에 GQA를 사용한다.

배포된 파운데이션 모델은 학습보다 추론에서 더 많은 총 계산량을 사용할 수 있기 때문에 추론 효율성(Inference Efficiency)이 특히 중요하다. 수백만 명의 사용자가 요청을 생성하거나 수천 대의 로봇이 지속적으로 어텐션 계산을 수행하면 추론 지연과 메모리 사용량은 운영 비용에 직접적인 영향을 준다.

키-값 캐싱(Key-Value Caching)은 자기회귀 추론(Auto-Regressive Inference)을 효율적으로 만드는 핵심 기술이다. 텍스트 생성 과정에서 이전에 계산된 키와 값 벡터는 변하지 않는다. 새로운 토큰을 생성할 때마다 전체 프리픽스(Prefix)를 다시 계산하는 대신 이전 키와 값을 메모리에 저장한다. 새로운 토큰은 저장된 표현과만 어텐션을 계산한다. 이는 중복 계산을 줄이지만 컨텍스트 길이가 증가할수록 메모리 사용량이 커진다.

효율적 어텐션 기술은 키-값 캐싱과 자연스럽게 결합된다. 플래시어텐션은 캐시된 표현을 사용하는 계산을 가속한다. 그룹드 쿼리 어텐션은 캐시 크기를 줄인다. 양자화(Quantization)는 캐시 텐서(Tensor)를 압축한다. 슬라이딩 윈도우 어텐션(Sliding-Window Attention)은 오래된 토큰을 제거하여 캐시 증가를 제한한다. 메모리 압축(Memory Compression)은 오래된 문맥을 작은 잠재 표현(Latent Representation)으로 요약한다. 이러한 방법들을 함께 사용하면 실제 하드웨어에서 긴 컨텍스트 추론이 가능해진다.

슬라이딩 윈도우 어텐션은 또 다른 유용한 절충 방식이다. 전체 시퀀스를 무제한으로 참조하는 대신 각 토큰은 최근의 일정 범위에 있는 이전 토큰만 참조한다. 이는 가까운 문맥이 대개 더 중요하다는 관찰에 기반한다. 장거리 정보는 요약 토큰, 순환 메모리(Recurrent Memory), 검색 시스템(Retrieval System), 계층적 표현(Hierarchical Representation)을 통해 유지할 수 있다.

검색 증강 어텐션(Retrieval-Augmented Attention)은 이를 보완하는 방법이다. 이전 토큰을 모두 현재 어텐션 윈도우 안에 저장하는 대신 외부 메모리 시스템이 필요한 정보만 검색한다. 이는 장기 기억(Long-Term Memory)과 즉각적인 작업 기억(Working Memory)을 분리하는 효과를 가진다. 대규모 언어 모델은 고정된 트랜스포머 컨텍스트를 넘어 유용한 문맥을 확장하기 위해 검색과 어텐션을 점점 더 많이 결합하고 있다.

계층적 어텐션(Hierarchical Attention)은 여러 공간적 또는 시간적 해상도에서 정보를 표현하여 계산량을 줄인다. 하위 계층은 세밀한 지역 정보를 처리하고, 상위 계층은 점점 더 넓은 영역을 요약한다. 비전 트랜스포머는 이미지 패치를 점차 더 큰 토큰으로 병합하고, 비디오 트랜스포머는 시간 정보도 유사하게 압축한다. 계층적 처리는 전역 구조를 유지하면서 시퀀스 길이를 줄인다.

토큰 가지치기(Token Pruning)는 또 다른 적응형 전략이다. 추론 과정에서 일부 토큰은 최종 예측에 거의 기여하지 않는다. 토큰 가지치기는 중요하지 않은 토큰을 식별하고 동적으로 제거하여 중요한 영역에 계산을 집중한다. 이미지에서는 배경 패치를 제거할 수 있고, 언어에서는 중복된 토큰의 처리를 줄일 수 있다. 로보틱스에서는 정적인 환경 영역보다 움직이는 물체나 사람과의 상호작용에 더 많은 계산을 할당할 수 있다.

토큰 병합(Token Merging)은 유사한 원리를 따른다. 토큰을 완전히 제거하는 대신 서로 유사한 인접 토큰을 더 적은 수의 대표 토큰으로 결합한다. 이는 시퀀스 길이를 줄이면서도 더 많은 정보를 유지할 수 있다. 적응형 토큰화(Adaptive Tokenization)는 장면의 복잡도에 따라 토큰을 지속적으로 병합하거나 분리할 수 있다.

피지컬 AI에서는 여러 센서 모달리티가 동시에 작동하기 때문에 효율적 어텐션이 특별히 중요하다. 이동 로봇은 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), LiDAR 스캔, 관성 측정(Inertial Measurement), 힘 센서, 휠 오도메트리(Wheel Odometry), 언어 명령, 의미 지도(Semantic Map), 위치 추정(Localization Estimate), 과거 궤적을 동시에 처리할 수 있다. 모든 측정값을 하나의 트랜스포머 시퀀스로 단순 연결하면 계산량이 빠르게 비현실적인 수준에 도달한다.

따라서 실제 로봇 아키텍처는 계산을 계층적으로 분산한다. 지역 인식 모듈(Local Perception Module)이 원시 센서 데이터를 압축된 잠재 표현으로 변환한 뒤 트랜스포머 통합 단계로 전달한다. 어텐션은 원시 센서 데이터가 아니라 정보량이 높은 잠재 토큰에 주로 적용된다. 시간적 서브샘플링(Temporal Subsampling)은 모든 관측을 처리하는 대신 핵심 프레임(Key Frame)이나 중요한 이벤트(Event)를 선택하여 시퀀스 길이를 추가로 줄인다.

이벤트 기반 어텐션(Event-Driven Attention)은 로보틱스에서 특히 유망한 방향이다. 고정된 주기로 모든 어텐션을 갱신하는 대신 환경에 의미 있는 변화가 발생했을 때만 계산을 증가시킨다. 정적인 장면에서는 재계산을 줄이고, 빠르게 변화하는 상황에서는 더 많은 계산 자원을 할당한다. 이러한 적응형 배분은 계산 효율과 에너지 소비를 동시에 개선한다.

교차 모달 어텐션(Cross-Modal Attention)은 추가적인 복잡성을 가져온다. 언어 명령은 시각 관측에 주의를 기울이고, 로봇 상태는 환경 특징을 참조하며, 계획된 행동은 예측된 미래 상태와 상호작용한다. 효율적인 구현은 각 모달리티 내부의 자기 주의뿐 아니라 모달리티 사이의 상호작용도 함께 최적화해야 한다. 선택적 라우팅(Selective Routing), 모달리티별 압축(Modality-Specific Compression), 계층적 융합(Hierarchical Fusion)은 이러한 계산 요구를 관리하는 데 사용된다.

비디오 트랜스포머는 여러 효율적 어텐션 개념을 동시에 보여주는 대표적인 예이다. 모든 프레임의 모든 픽셀 패치를 개별 토큰으로 처리하는 단순한 트랜스포머는 계산 비용이 지나치게 크다. 실제 아키텍처에서는 인접한 시간과 공간 영역을 결합하는 튜블릿 임베딩(Tubelet Embedding), 공간과 시간 계산을 분리하는 분해형 어텐션(Factorized Attention), 시간 풀링(Temporal Pooling), 계층적 프레임 표현(Hierarchical Frame Representation), 잠재 비디오 압축(Latent Video Compression)을 사용한다. 이러한 전략은 현실적인 계산 비용 안에서 움직임 이해 능력을 유지하도록 한다.

대규모 멀티모달 파운데이션 모델은 여러 효율적 어텐션 기술을 동시에 사용하는 경우가 많다. 플래시어텐션은 정확한 계산을 가속한다. 그룹드 쿼리 어텐션은 추론 메모리를 줄인다. 희소 어텐션은 불필요한 상호작용을 제한한다. 슬라이딩 윈도우는 컨텍스트 증가를 통제한다. 검색은 장기 기억을 확장한다. 양자화는 수치 정밀도를 낮춰 메모리와 연산 비용을 줄인다. 계층적 토큰화는 고차원 센서 입력을 압축한다. 하나의 최적화만으로 모든 문제를 해결할 수는 없지만, 이러한 방법을 결합하면 대규모 모델을 실제로 배포할 수 있다.

하드웨어 발전도 효율적 어텐션 연구에 큰 영향을 준다. GPU의 계산 처리량은 메모리 대역폭보다 빠르게 증가하고 있다. 전용 텐서 프로세서(Tensor Processor), AI 가속기(AI Accelerator), 새로운 메모리 아키텍처가 등장하면서 성능 특성도 계속 변화한다. 효율적 알고리즘은 이제 하드웨어를 단순한 실행 플랫폼으로 보지 않고 하드웨어 능력과 계산을 공동 설계(Co-Design)하는 방향으로 발전하고 있다.

컴파일러(Compiler) 기술 또한 중요한 역할을 한다. 현대의 머신러닝 컴파일러는 여러 연산을 융합하고, 메모리 할당을 최적화하며, 커널을 효율적으로 스케줄링하고, 하드웨어별 구현 코드를 생성한다. 동일한 트랜스포머 모델이라도 컴파일러 최적화 수준에 따라 처리량이 크게 달라질 수 있다. 따라서 효율적 어텐션은 수학적 알고리즘을 넘어 소프트웨어 시스템 엔지니어링(Software Systems Engineering)의 문제이기도 하다.

학습 효율성과 추론 효율성은 서로 관련되어 있지만 동일한 목표는 아니다. 일부 방법은 학습 시 메모리 요구량을 줄여 더 큰 배치(Batch)나 더 긴 컨텍스트를 가능하게 한다. 다른 방법은 자기회귀 생성을 주로 가속한다. 로봇 파운데이션 모델은 오프라인에서 대규모 사전학습을 수행하면서 동시에 온라인에서는 낮은 지연의 제어가 필요하므로 두 측면을 모두 최적화해야 한다.

효율적 어텐션을 평가할 때는 처리량만 측정해서는 안 된다. 정확도(Accuracy), 강건성(Robustness), 수치 안정성(Numerical Stability), 확장성(Scalability), 하드웨어 이식성(Hardware Portability), 구현 복잡성(Implementation Complexity), 에너지 효율(Energy Efficiency)을 함께 고려해야 한다. 이론적으로는 효율적이지만 실제 하드웨어에서 최적화하기 어려운 어텐션은 실질적인 가치가 제한될 수 있다. 반대로 하나의 하드웨어에서만 동작하는 고도로 최적화된 커널은 이식성이 낮을 수 있다.

미래의 효율적 어텐션은 고정된 계산 패턴보다 적응형 계산(Adaptive Computation)으로 발전할 가능성이 높다. 모델은 어떤 토큰이 전역 어텐션을 필요로 하는지, 어떤 토큰은 지역 처리만으로 충분한지, 어떤 토큰은 압축하거나 무시해도 되는지를 동적으로 결정하게 될 것이다. 계산 자원은 시퀀스 위치보다 정보량(Information Content)에 따라 할당될 가능성이 높다.

피지컬 AI에서 적응형 어텐션은 지능형 행동과 자연스럽게 연결된다. 비어 있는 복도를 주행하는 로봇은 상대적으로 적은 시각 추론만 필요하지만, 사람 근처에서 깨지기 쉬운 물체를 조작하는 로봇은 훨씬 정밀한 분석이 필요하다. 효율적 어텐션은 환경의 복잡성이 추가 계산을 정당화하는 경우에만 계산량을 증가시킬 수 있도록 한다.

궁극적으로 효율적 어텐션은 순수한 알고리즘 설계에서 수학, 소프트웨어, 하드웨어, 응용 요구사항을 통합적으로 최적화하는 방향으로 전환되고 있다. 선형 어텐션은 수학적 재구성을 통해 점근적 복잡도(Asymptotic Complexity)를 줄일 수 있음을 보여준다. 플래시어텐션은 정확한 알고리즘도 하드웨어 인식 구현을 통해 훨씬 빠르게 실행될 수 있음을 보여준다. 두 기술은 미래의 파운데이션 모델이 단순히 더 큰 네트워크뿐 아니라 계산 자원을 더욱 정교하게 배분하는 방법에 의존하게 될 것임을 보여준다.

차세대 파운데이션 모델과 피지컬 AI 시스템에서 효율적 어텐션은 더 이상 단순한 최적화 기술이 아니다. 이는 긴 컨텍스트 추론(Long-Context Reasoning), 멀티모달 인식(Multimodal Perception), 대규모 월드 모델링(Large-Scale World Modeling), 실시간 로봇 지능(Real-Time Robotic Intelligence)을 계산적으로 가능하게 만드는 핵심 기반 기술이다. 시퀀스 길이, 센서 다양성, 모델 능력이 계속 증가함에 따라 효율적 어텐션 메커니즘은 확장 가능한 인공지능(Scalable Artificial Intelligence)을 지원하는 중심 기술로 남게 될 것이다.

## 2.4 Decoder Only GPT Architecture for Action Seq [w/Code]

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

GPT 계열과 일반적으로 연관되는 디코더 전용 트랜스포머(Decoder-Only Transformer) 아키텍처는 현대 파운데이션 모델(Foundation Model)에서 가장 영향력 있는 구조 가운데 하나가 되었다. 이 구조의 본래 강점은 이전에 관측한 모든 토큰(Token)을 바탕으로 다음 토큰을 예측하는 자기회귀 언어 생성(Autoregressive Language Generation)에 있었다. 그러나 동일한 원리는 언어 이외의 영역으로도 확장될 수 있다. 로봇의 관측(Observation), 목표(Goal), 상태(State), 행동(Action)을 토큰으로 표현한다면, 디코더 전용 트랜스포머는 언어 모델이 문장을 생성하는 것과 유사한 방식으로 행동 시퀀스(Action Sequence)를 생성할 수 있다.

이러한 접근법은 많은 로봇 작업이 순차적 의사결정 문제(Sequential Decision Problem)로 표현될 수 있다는 점에서 피지컬 AI(Physical AI)에 중요하다. 로봇은 관측을 수신하고, 작업을 해석하며, 이전 행동을 고려한 뒤 다음 행동을 선택한다. 행동은 이산 명령(Discrete Command), 토큰으로 변환된 연속 제어값(Continuous Control Value), 웨이포인트(Waypoint), 모션 프리미티브(Motion Primitive), 또는 일정 시간 동안 지속되는 행동 청크(Action Chunk)가 될 수 있다. 이러한 요소를 순서가 있는 시퀀스로 모델링하면 디코더 전용 트랜스포머를 통해 미래 행동을 예측하는 통합 아키텍처(Unified Architecture)를 구성할 수 있다.

디코더 전용 트랜스포머는 반복되는 트랜스포머 디코더 블록(Transformer Decoder Block)으로 구성된다. 각 블록은 일반적으로 인과적 자기 주의(Causal Self-Attention), 피드포워드 네트워크(Feed-Forward Network), 잔차 연결(Residual Connection), 정규화(Normalization)를 포함한다. 인코더-디코더(Encoder-Decoder) 구조와 달리 별도의 인코더 스택(Encoder Stack)은 존재하지 않는다. 모든 입력 정보와 출력 정보가 하나의 토큰 시퀀스에 배치되며, 모델은 다음 토큰을 자기회귀 방식으로 예측한다.

언어 모델에서는 시퀀스가 단어 또는 서브워드 토큰(Subword Token)으로 구성된다. 행동 생성 모델에서는 시퀀스가 작업 지시(Task Instruction), 이미지에서 추출된 토큰(Image-Derived Token), 로봇 상태(Robot State), 이전 행동(Previous Action), 보상(Reward), 목표(Goal), 미래 행동 토큰(Future Action Token)으로 구성될 수 있다. 모델은 사용 가능한 문맥(Context)을 포함하는 프리픽스(Prefix)를 입력받고, 이후 내용을 토큰 단위 또는 행동 청크 단위로 생성한다.

인과적 자기 주의(Causal Self-Attention)는 이 아키텍처를 정의하는 핵심 메커니즘이다. 각 토큰은 자기 자신과 이전에 등장한 토큰만 참조할 수 있다. 학습이나 추론 과정에서 미래 토큰에는 접근할 수 없다. 이러한 인과적 제약은 모델이 실제 실행 시점에 이용 가능했을 정보만으로 미래 행동을 예측하도록 보장한다.

로봇이 특정 물체를 집으라는 명령을 받았다고 가정해 보자. 문맥에는 목표를 설명하는 언어 토큰(Language Token), 장면을 표현하는 시각 토큰(Visual Token), 로봇 상태를 나타내는 고유감각 토큰(Proprioceptive Token), 이전 움직임을 나타내는 행동 토큰(Action Token)이 포함될 수 있다. 디코더 전용 모델은 이 전체 이력(History)에 대해 어텐션을 수행한 뒤 다음 행동 토큰을 예측한다. 생성된 토큰은 다음 행동을 예측하기 위한 문맥의 일부로 다시 포함된다.

이 과정은 모델이 시퀀스 종료 토큰(End-of-Sequence Token), 작업 완료 토큰(Task-Completion Token)을 생성하거나 미리 정해진 수의 행동을 생성할 때까지 계속된다. 이러한 방식으로 로봇 행동은 조건부 시퀀스 생성(Conditional Sequence Generation) 문제로 처리된다. 모델은 작업마다 별도로 설계된 계획기(Planner)를 반드시 필요로 하지 않는다. 대신 학습 데이터에 포함된 지시, 관측, 상태, 행동 사이의 통계적 관계를 바탕으로 계획 행동(Planning Behavior)을 학습한다.

디코더 전용 구조는 여러 실용적인 장점을 제공한다. 개념적으로 단순하고, 확장성이 높으며, 대규모 언어 모델에 사용되는 동일한 학습 인프라와 호환된다. 전체 문제를 다음 토큰 예측(Next-Token Prediction)으로 표현할 수 있다. 서로 다른 유형의 입력은 모두 토큰으로 변환되어 하나의 시퀀스에 배치되며, 모델은 동일한 자기 주의 메커니즘을 이용해 사용 가능한 모든 문맥을 통합한다.

이러한 통합된 표현 방식은 아키텍처의 분절을 줄여준다. 전통적인 로봇 시스템은 언어 이해(Language Understanding), 인식(Perception), 계획(Planning), 행동 선택(Behavior Selection), 행동 예측(Action Prediction)을 서로 다른 모듈로 구성하는 경우가 많았다. 디코더 전용 행동 모델은 이러한 기능을 하나의 자기회귀 백본(Autoregressive Backbone) 안에서 결합할 수 있다. 다만 저수준 제어기(Low-Level Controller)와 안전 모듈(Safety Module)은 여전히 별도로 유지될 수 있다.

첫 번째 단계는 토큰화(Tokenization)이다. 언어 지시는 일반적인 텍스트 토크나이저(Text Tokenizer)를 이용해 토큰화된다. 이미지는 비전 트랜스포머(Vision Transformer)나 다른 시각 인코더(Visual Encoder)를 이용해 시각 토큰으로 변환할 수 있다. 포인트 클라우드(Point Cloud)는 학습된 영역 토큰(Region Token)으로 표현할 수 있다. 고유감각 상태는 임베딩(Embedding)으로 투영할 수 있으며, 행동은 이산화된 토큰 또는 연속 출력 헤드(Continuous Output Head)를 통해 표현할 수 있다.

이러한 토큰의 배치 순서는 중요하다. 하나의 가능한 시퀀스는 작업 토큰(Task Token)으로 시작하여 관측 토큰(Observation Token), 상태 토큰(State Token), 이전 행동 토큰(Previous Action Token) 순서로 이어질 수 있다. 또 다른 설계에서는 시간에 따라 관측과 행동을 교차 배치할 수 있다. 하나의 궤적(Trajectory)은 각 시간 단계마다 관측, 상태, 행동 정보를 포함하는 블록을 반복하는 구조로 표현할 수 있다.

교차 배치된 궤적 표현(Interleaved Trajectory Representation)은 시간적 관계를 보존한다. 모델은 특정 관측 뒤에 특정 행동이 수행되었고, 그 행동이 이후 환경 변화를 유발했다는 관계를 학습할 수 있다. 이는 행동 조건부 동역학(Action-Conditioned Dynamics)과 암묵적 계획(Implicit Planning)을 학습하는 데 유용하다.

특수 토큰(Special Token)은 모달리티(Modality), 시간 단계(Time Step), 로봇 식별자(Robot Identity), 센서 종류(Sensor Type), 에피소드 경계(Episode Boundary), 작업 완료(Task Completion)를 표시할 수 있다. 이러한 토큰은 모델이 서로 다른 유형의 정보를 구분하도록 돕는다. 여러 로봇을 함께 학습하는 교차 로봇 학습(Cross-Robot Training)에서는 체화 토큰(Embodiment Token)을 이용해 로봇 플랫폼, 관절 구성(Joint Configuration), 행동 공간(Action Space)을 나타낼 수 있다.

위치 인코딩(Positional Encoding)도 필수적이다. 모델은 토큰의 순서뿐 아니라 어떤 토큰이 동일한 시간 단계와 모달리티에 속하는지도 알아야 한다. 로봇 시퀀스는 시간 위치(Temporal Position), 공간 위치(Spatial Position), 센서 식별자(Sensor Identity), 행동 인덱스(Action Index)를 동시에 표현해야 할 수 있다. 학습형 임베딩(Learned Embedding), 회전 위치 인코딩(Rotary Positional Encoding), 상대 위치 편향(Relative Position Bias), 구조화된 시간 임베딩(Structured Time Embedding)을 결합하여 이러한 관계를 표현할 수 있다.

인과적 어텐션 마스크(Causal Attention Mask)는 어떤 토큰이 어떤 정보에 접근할 수 있는지를 결정한다. 단순한 자기회귀 시퀀스에서는 각 토큰이 이전의 모든 토큰에 어텐션할 수 있다. 그러나 특정 관측이 여러 미래 행동에 계속 보이도록 해야 하거나 서로 다른 모달리티마다 접근 규칙이 달라야 하는 경우에는 더 구조화된 마스크를 사용할 수 있다.

예를 들어 현재 관측의 시각 토큰은 다음 행동 토큰 전체에 공개할 수 있지만 미래 관측은 숨겨야 한다. 언어 지시는 에피소드 전체 동안 계속 참조할 수 있다. 이전 행동 토큰은 이후 모든 토큰에 공개할 수 있다. 잘 설계된 마스크는 인과성을 보존하면서도 멀티모달 문맥을 효과적으로 사용할 수 있도록 한다.

행동 표현(Action Representation)은 가장 중요한 설계 요소 가운데 하나이다. 언어 토큰은 본질적으로 이산적이지만 로봇 행동은 대부분 연속적이다. 매니퓰레이터(Manipulator)는 관절 위치(Joint Position), 관절 속도(Joint Velocity), 그리퍼 명령(Gripper Command), 엔드이펙터 자세(End-Effector Pose)를 필요로 할 수 있다. 이동 로봇(Mobile Robot)은 선속도(Linear Velocity), 각속도(Angular Velocity), 조향각(Steering Angle), 웨이포인트 좌표(Waypoint Coordinate)를 필요로 할 수 있다.

하나의 방법은 이산화(Discretization)이다. 연속 행동값을 여러 구간(Bin)으로 나누고 각 구간에 하나의 토큰을 할당한다. 모델은 원하는 행동값에 해당하는 토큰을 예측한다. 이 방식은 행동 예측을 범주형 교차 엔트로피 손실(Categorical Cross-Entropy Loss)을 사용하는 표준 언어 모델 학습과 호환되게 만든다.

이산화는 학습을 단순화하지만 양자화 오차(Quantization Error)를 발생시킨다. 구간을 세밀하게 나누면 정밀도는 향상되지만 어휘 크기(Vocabulary Size)가 증가하고 학습 난도가 높아진다. 구간을 거칠게 나누면 복잡도는 감소하지만 제어가 불연속적이거나 부정확해질 수 있다. 비균일 구간화(Nonuniform Binning)는 중요한 행동 범위에 더 높은 해상도를 배정할 수 있다.

또 다른 방법은 하이브리드 모델(Hybrid Model)을 사용하는 것이다. 트랜스포머는 문맥 표현(Contextual Representation)을 생성하고, 연속 출력 헤드는 실수형 행동값을 예측한다. 디코더는 시퀀스 수준에서는 자기회귀 방식으로 동작하지만 최종 행동 출력은 회귀(Regression), 가우시안 분포(Gaussian Distribution), 혼합 모델(Mixture Model), 확산 과정(Diffusion Process), 플로우 기반 정책(Flow-Based Policy)으로 모델링할 수 있다.

행동 시퀀스는 한 번에 하나의 시간 단계씩 생성할 수 있다. 그러나 단일 단계 생성은 느릴 수 있고 오류가 누적되기 쉽다. 행동 청킹(Action Chunking)은 여러 미래 행동을 한꺼번에 예측하여 이러한 문제를 완화한다. 하나의 청크는 여러 제어 단계를 포함하는 짧은 궤적이 될 수 있다.

행동 청킹은 독립적인 명령 대신 일관된 국소 움직임(Local Motion)을 생성하므로 시간적 일관성(Temporal Consistency)을 향상시킨다. 또한 자기회귀 디코딩 단계 수를 줄여 지연 시간(Latency)을 감소시킨다. 그러나 청크가 길수록 정책이 새로운 관측에 반응하는 빈도는 낮아진다.

이는 반응성(Responsiveness)과 시간적 일관성 사이의 상충 관계를 만든다. 짧은 청크는 환경 변화에 빠르게 적응하지만 더 많은 추론을 요구한다. 긴 청크는 계산 효율이 높고 움직임이 부드럽지만 장면이 갑자기 변하면 부적절해질 수 있다. 이동 지평선 실행(Receding-Horizon Execution)은 하나의 절충안이다. 청크 전체를 생성하되 일부만 실행하고, 새로운 관측으로 다시 계획하는 방식이다.

디코더 전용 행동 모델의 학습에는 일반적으로 교사 강제(Teacher Forcing)가 사용된다. 학습 과정에서 모델은 실제 이전 행동을 입력받고 다음 행동을 예측한다. 인과적 마스크 아래에서 모든 예측 위치를 병렬로 계산할 수 있으므로 최적화가 효율적이다.

그러나 배포 과정에서는 실제 행동이 아니라 모델이 이전에 생성한 행동을 다시 입력으로 사용한다. 따라서 오류가 누적될 수 있다. 학습과 추론 사이의 이러한 차이를 노출 편향(Exposure Bias)이라고 한다. 행동 복제(Behavioral Cloning) 모델은 작은 오류 하나가 로봇을 학습 데이터에 포함되지 않은 상태로 이동시킬 수 있기 때문에 특히 취약하다.

노출 편향을 줄이기 위한 여러 방법이 있다. 학습 데이터에 복구 시연(Recovery Demonstration), 의도적인 교란(Perturbation), 실패 사례(Failure Case)를 포함할 수 있다. 스케줄드 샘플링(Scheduled Sampling)은 학습 과정에서 실제 행동을 점차 모델 생성 행동으로 대체한다. 온라인 데이터 수집(Online Data Collection)은 현재 정책이 생성한 궤적을 데이터셋에 추가한다. 강화학습(Reinforcement Learning)과 선호 최적화(Preference Optimization)는 단순한 모방을 넘어 정책을 추가로 개선할 수 있다.

디코더 전용 아키텍처는 긴 행동 시퀀스의 장거리 의존성(Long-Range Dependency)도 모델링할 수 있다. 로봇은 수십 초 전에 받은 지시를 기억하거나, 일시적으로 가려진 물체를 추적하거나, 장기 목표(Long-Term Goal)를 향한 진행 상태를 유지해야 할 수 있다. 자기 주의는 행동 생성이 먼 과거의 문맥에 직접 의존할 수 있도록 한다.

그러나 긴 로봇 궤적은 큰 계산 비용과 메모리 비용을 발생시킨다. 표준 자기 주의의 비용은 시퀀스 길이의 제곱에 비례한다. 고주파 센서 스트림(High-Frequency Sensor Stream)은 매우 큰 컨텍스트를 생성할 수 있다. 따라서 효율적 어텐션(Efficient Attention), 슬라이딩 윈도우(Sliding Window), 메모리 압축(Memory Compression), 검색(Retrieval), 계층적 시간 모델링(Hierarchical Temporal Modeling)이 중요하다.

일반적인 해결 방법은 상위 수준과 하위 수준의 시간 척도를 분리하는 것이다. 디코더 전용 트랜스포머는 중간 정도의 주기로 상위 행동(High-Level Action) 또는 행동 청크를 생성한다. 하위 제어기(Low-Level Controller)는 훨씬 높은 주기로 정밀한 모터 명령을 실행한다. 이를 통해 트랜스포머가 모든 서보 수준 갱신(Servo-Level Update)을 직접 처리하지 않아도 된다.

예를 들어 상위 모델은 초당 수 회의 주기로 엔드이펙터 웨이포인트를 생성하고, 전통적인 제어기는 이를 수백 또는 수천 헤르츠로 보간(Interpolation)하고 추종할 수 있다. 이러한 구조는 파운데이션 모델이 의미 해석(Semantic Interpretation)과 장기 계획(Long-Horizon Planning)을 담당하면서도 실시간 안정성(Real-Time Stability)을 유지하도록 한다.

디코더 전용 행동 모델은 리턴(Return), 보상(Reward), 원하는 결과(Desired Outcome)를 조건으로 사용할 수도 있다. 오프라인 강화학습(Offline Reinforcement Learning)에서는 하나의 궤적을 리턴 투 고(Return-to-Go), 상태(State), 행동(Action) 토큰을 포함하는 시퀀스로 표현할 수 있다. 모델은 원하는 미래 보상을 조건으로 행동을 생성하도록 학습한다.

이 방식은 강화학습을 시퀀스 모델링(Sequence Modeling) 문제로 변환한다. 시간차 학습(Temporal-Difference Learning)을 이용해 가치 함수(Value Function)와 정책을 명시적으로 학습하는 대신, 모델은 과거 궤적과 원하는 성능에 따라 행동을 예측한다. 대규모 오프라인 데이터셋이 있을 경우 효과적일 수 있지만 데이터셋의 품질과 범위에 크게 의존한다.

목표 조건(Goal Conditioning)은 언어, 목표 이미지(Target Image), 좌표(Coordinate), 기호형 작업 토큰(Symbolic Task Token), 원하는 종료 상태(Desired Terminal State)로 표현할 수 있다. 모델은 제공된 목표에 따라 서로 다른 행동 시퀀스를 생성하도록 학습한다. 하나의 아키텍처로 여러 행동을 표현할 수 있기 때문에 멀티태스크 학습(Multitask Learning)을 지원한다.

멀티모달 조건화(Multimodal Conditioning)는 로봇 파운데이션 모델에서 특히 중요하다. 시각 관측은 객체와 공간 구조에 대한 정보를 제공한다. 언어는 의도된 작업을 지정한다. 고유감각(Proprioception)은 로봇의 신체 상태를 나타낸다. 이전 행동은 시간적 문맥을 제공한다. 디코더 전용 트랜스포머는 인과적 자기 주의를 통해 이러한 정보원을 결합한다.

시각 정보를 삽입하는 방법에는 여러 가지가 있다. 시각 인코더는 행동 토큰보다 앞에 배치되는 이미지 토큰 시퀀스를 생성할 수 있다. 시각 인코더는 고정(Frozen)하거나, 미세조정(Fine-Tuning)하거나, 전체 모델과 공동 학습할 수 있다. 투영 계층(Projection Layer)은 시각 특징 차원을 트랜스포머 임베딩 공간과 정렬한다.

또 다른 전략은 궤적 전체에 시각 토큰을 반복적으로 삽입하는 것이다. 각 시간 단계에 새로운 관측 토큰 그룹을 배치하고 그 뒤에 행동 토큰을 배치한다. 이 방식은 실행 중 계속 갱신되는 인식을 제공하지만 시퀀스 길이를 크게 증가시킨다.

따라서 시각 토큰 압축(Visual Token Compression)이 중요하다. 하나의 원시 이미지는 수백 또는 수천 개의 패치 토큰(Patch Token)을 생성할 수 있다. 잠재 리샘플링 모듈(Latent Resampling Module), 풀링(Pooling), 토큰 병합(Token Merging), 영역 선택(Region Selection), 객체 중심 표현(Object-Centric Representation)을 이용하여 더 적은 수의 정보성 토큰으로 줄일 수 있다.

객체 중심 토큰(Object-Centric Token)은 조작 작업에서 특히 유용하다. 모든 이미지 패치를 동일하게 표현하는 대신 물체(Object), 그리퍼(Gripper), 손(Hand), 도구(Tool), 장애물(Obstacle)을 서로 다른 개체(Entity)로 인코딩할 수 있다. 디코더는 작업과 관련된 관계에 더 효율적으로 어텐션할 수 있다.

디코더 전용 아키텍처는 도구 토큰(Tool Token)과 스킬 토큰(Skill Token)도 지원한다. 로봇은 이동(Navigate), 도달(Reach), 파지(Grasp), 들어 올리기(Lift), 열기(Open), 배치(Place), 검사(Inspect)와 같은 저수준 스킬 라이브러리(Skill Library)를 가질 수 있다. 트랜스포머는 직접 모터 명령을 생성하는 대신 스킬 토큰을 생성하고, 각 스킬 토큰은 전문 제어기(Specialized Controller)를 호출할 수 있다.

이러한 하이브리드 접근법은 학습된 상위 계획과 신뢰성 높은 저수준 실행을 결합한다. 생성된 시퀀스가 의미 있는 행동에 대응하므로 해석 가능성(Interpretability)도 향상된다. 그러나 스킬 라이브러리의 존재와 품질에 의존한다는 한계가 있다.

직접 행동 생성(Direct Action Generation)은 미리 정의된 스킬에 제한되지 않으므로 더 높은 유연성을 제공한다. 새로운 궤적을 생성하고 새로운 상황에 적응할 수 있다. 그러나 직접 제어는 안전 위험과 데이터 요구량을 증가시킨다. 실제 시스템에서는 일반적인 작업에는 스킬 수준 계획을 사용하고, 정밀 조작에는 연속 행동 생성을 사용하는 방식으로 두 접근법을 결합할 수 있다.

교차 체화 학습(Cross-Embodiment Learning)은 또 다른 중요한 응용 분야이다. 하나의 디코더 전용 모델을 여러 로봇의 궤적으로 학습할 수 있다. 체화 토큰은 플랫폼을 식별하고, 행동 토크나이저(Action Tokenizer)는 서로 다른 제어 공간을 공통 표현으로 변환한다.

공통 잠재 행동 공간(Common Latent Action Space)은 전진, 회전, 도달, 그리퍼 닫기, 힘 적용과 같은 의도(Intent)를 표현할 수 있다. 이후 로봇별 디코더(Robot-Specific Decoder)가 이러한 잠재 행동을 하드웨어 명령으로 변환한다. 이를 통해 서로 다른 로봇 형태(Robot Morphology) 사이에서 지식을 전이할 수 있다.

그러나 교차 체화 모델링은 어렵다. 동일한 행동 토큰이라도 서로 다른 플랫폼에서는 물리적 의미가 달라질 수 있다. 소형 이동 로봇, 대형 실외 자율이동로봇(Outdoor AMR), 사족보행 로봇(Quadruped Robot), 매니퓰레이터는 서로 다른 동역학(Dynamics), 제약조건(Constraint), 제어 주기(Control Frequency)를 가진다. 모델은 공유 가능한 작업 구조와 체화별 실행 방식을 구분해야 한다.

로봇 식별 임베딩(Robot Identity Embedding), 운동학 설명(Kinematic Description), 관절 마스크(Joint Mask), 능력 벡터(Capability Vector), 정규화된 좌표계(Normalized Coordinate System)가 도움이 될 수 있다. 또한 지식 전이가 발생하려면 학습 데이터에 작업과 환경의 충분한 중첩(Overlap)이 존재해야 한다.

디코더 전용 모델은 월드 모델링(World Modeling) 능력을 암묵적으로 포함할 수도 있다. 행동뿐 아니라 관측 또는 상태 토큰을 함께 예측하면 모델은 환경이 시간에 따라 어떻게 변화하는지를 학습한다. 이후 후보 행동에 따른 미래 궤적을 상상하여 생성할 수 있다.

통합 시퀀스는 관측, 행동, 다음 관측, 보상 토큰을 포함할 수 있다. 모델은 궤적 전체의 결합 분포(Joint Distribution)를 학습한다. 계획 과정에서는 여러 행동 시퀀스와 예상 결과를 샘플링한 뒤 가장 바람직한 궤적을 선택할 수 있다.

이는 통합 자기회귀 모델을 사용하는 모델 기반 계획(Model-Based Planning)과 유사하다. 동일한 트랜스포머가 프롬프트(Prompt)와 디코딩 방식에 따라 정책(Policy), 동역학 모델(Dynamics Model), 궤적 생성기(Trajectory Generator)로 동작할 수 있다.

그러나 자기회귀 롤아웃(Autoregressive Rollout)은 예측 오차를 누적시킬 수 있다. 예측된 상태에서 작은 오류가 발생하면 이후 예측에도 영향을 주어 궤적이 현실에서 점차 벗어날 수 있다. 따라서 실제 실행에서는 새로운 관측을 이용해 자주 재계획(Replanning)해야 한다.

디코딩 전략(Decoding Strategy)은 행동 품질에 큰 영향을 미친다. 탐욕 디코딩(Greedy Decoding)은 각 단계에서 가장 확률이 높은 행동 토큰을 선택한다. 결정론적이고 효율적이지만 지나치게 보수적인 행동을 생성할 수 있다. 샘플링(Sampling)은 다양성을 제공하지만 안전하지 않은 행동을 생성할 수 있다. 빔 탐색(Beam Search)은 여러 시퀀스를 탐색하지만 지연 시간을 증가시킨다.

로봇 제어에서는 제한 없는 샘플링이 일반적으로 적합하지 않다. 온도 제한(Temperature Limit), 상위 k 필터링(Top-k Filtering), 행동 마스크(Action Mask), 실행 가능성 제약(Feasibility Constraint), 안전 필터(Safety Filter)를 이용해 후보 행동을 제한할 수 있다. 계획기는 여러 행동 시퀀스를 생성한 뒤 가치 모델(Value Model), 월드 모델, 안전 검사기(Safety Checker)를 이용해 평가할 수 있다.

제약 디코딩(Constrained Decoding)은 로봇의 물리적 한계를 강제할 수 있다. 유효하지 않은 관절 명령, 도달할 수 없는 위치, 과도한 속도, 위험 영역을 행동 선택 전에 제거할 수 있다. 이는 학습된 생성과 결정론적 규칙(Deterministic Rule)을 결합하는 방식이다.

불확실성 추정(Uncertainty Estimation)도 중요하다. 모델은 행동 토큰에 대한 확률 분포를 생성할 수 있다. 높은 엔트로피(Entropy)는 불확실성을 나타낼 수 있다. 여러 샘플 궤적 사이의 차이는 모델의 의견 불일치(Disagreement)를 보여줄 수 있다. 외부 신뢰도 모델(Confidence Model)은 현재 상황이 학습 분포와 얼마나 유사한지를 평가할 수 있다.

불확실성이 높을 때 로봇은 속도를 낮추거나, 인간의 도움을 요청하거나, 안전 제어기로 전환하거나, 추가 관측을 수집하거나, 완전히 정지할 수 있다. 따라서 디코더 전용 정책은 제약 없이 직접 제어하도록 두기보다 더 큰 안전 아키텍처(Safety Architecture) 안에 포함되어야 한다.

추론 지연 시간(Inference Latency)은 중요한 실용적 문제이다. 자기회귀 생성은 각 토큰이 이전 출력에 의존하므로 순차적으로 진행된다. 긴 행동 시퀀스는 느릴 수 있다. 행동 청킹, 병렬 행동 헤드(Parallel Action Head), 추측 디코딩(Speculative Decoding), 캐싱(Caching), 모델 압축(Model Compression)을 통해 비용을 줄일 수 있다.

키-값 캐시(Key-Value Cache)는 이전 토큰의 어텐션 키와 값을 저장하여 반복 계산을 방지한다. 생성 속도는 향상되지만 메모리 사용량은 증가한다. 긴 로봇 에피소드에서는 캐시 크기가 무제한으로 증가하지 않도록 슬라이딩 윈도우나 압축 메모리가 필요하다.

양자화(Quantization)는 낮은 정밀도의 수치 형식을 이용해 모델과 캐시 메모리를 줄인다. 가지치기(Pruning)는 중요하지 않은 파라미터를 제거한다. 지식 증류(Knowledge Distillation)는 대형 모델의 행동을 작은 학생 모델(Student Model)에 이전한다. 저랭크 적응(Low-Rank Adaptation)은 전체 모델을 다시 학습하지 않고 특정 로봇에 맞게 특화할 수 있다.

대형 디코더 전용 행동 모델은 데이터센터에서 학습한 뒤 엣지에 배포 가능한 정책으로 증류할 수 있다. 대형 모델은 폭넓은 일반화와 추론을 제공하고, 소형 모델은 지연 시간과 전력 제약을 만족한다.

학습 데이터는 모델의 최종 행동을 결정한다. 시연 데이터는 다양한 물체, 환경, 카메라 시점, 로봇 상태, 작업 지시, 실패 복구를 포함해야 한다. 반복되는 성공 궤적만으로는 문제가 발생했을 때 어떻게 대응해야 하는지를 학습하기 어렵다.

부정 사례(Negative Example)와 중단된 시연(Interrupted Demonstration)도 중요하다. 모델은 안전한 행동과 위험한 행동을 구분하고, 파지 실패(Failed Grasp)를 감지하고, 위치 추정 오류(Localization Error)에서 복구하며, 예상치 못한 장애물에 대응하도록 학습할 수 있다. 선호 데이터(Preference Data)를 추가하면 여러 성공 행동 중 어떤 것이 더 부드럽고, 안전하며, 효율적인지도 학습할 수 있다.

데이터 동기화(Data Synchronization)는 필수적이다. 시각 관측, 로봇 상태, 행동이 시간적으로 정확히 대응해야 한다. 작은 시간 오차도 잘못된 인과 관계(Causal Relationship)를 학습하게 만들 수 있다. 따라서 하드웨어 타임스탬프(Hardware Timestamp), 동기화된 시계(Synchronized Clock), 보정된 센서 파이프라인(Calibrated Sensor Pipeline)이 고품질 행동 시퀀스 학습에 중요하다.

손실 함수(Loss Function)는 여러 목표를 결합할 수 있다. 주요 항은 행동 토큰을 예측한다. 추가 항은 미래 상태(Future State), 작업 완료(Task Completion), 보상(Reward), 불확실성(Uncertainty), 접촉 이벤트(Contact Event), 성공 확률(Success Probability)을 예측할 수 있다. 다중 목적 학습(Multi-Objective Training)은 표현 품질을 향상시킬 수 있지만 각 손실 항의 가중치를 신중히 조정해야 한다.

평가는 단순한 토큰 예측 정확도를 넘어야 한다. 모델이 낮은 행동 토큰 손실을 달성하더라도 실제 물리적 행동은 불안정할 수 있다. 실제 평가는 작업 성공률(Task Success Rate), 충돌 빈도(Collision Frequency), 제어의 부드러움(Control Smoothness), 복구 능력(Recovery Capability), 지연 시간, 에너지 사용량(Energy Use), 일반화(Generalization), 안전성을 측정해야 한다.

개방 루프 평가(Open-Loop Evaluation)는 예측 행동을 기록된 행동과 비교한다. 폐루프 평가(Closed-Loop Evaluation)는 모델을 시뮬레이션 또는 실제 환경에서 실행한다. 작은 예측 오류가 이후 관측을 바꿀 수 있기 때문에 폐루프 테스트가 더 많은 정보를 제공한다. 시뮬레이션은 규모와 안전성을 제공하고, 실제 환경 시험은 현실 전이(Real-World Transfer)를 검증한다.

디코더 전용 아키텍처에도 한계는 존재한다. 많은 데이터를 필요로 하고, 계산 비용이 높으며, 분포 변화(Distribution Shift)에 민감할 수 있다. 자기회귀 오류 누적은 여전히 중요한 문제이다. 긴 컨텍스트는 메모리 비용을 증가시키며, 다음 토큰 예측만으로 최적 계획이 보장되는 것은 아니다.

그럼에도 개념적 단순성과 높은 확장성 때문에 매우 매력적인 구조이다. 동일한 아키텍처에서 언어, 인식, 상태, 행동, 보상을 하나의 통합 시퀀스로 모델링할 수 있다. 폭넓은 멀티모달 데이터셋으로 사전학습하고, 로봇 시연으로 미세조정하며, 선호 최적화를 통해 정렬하고, 범용 정책(Generalist Policy)으로 배포할 수 있다.

피지컬 AI에서 디코더 전용 GPT 방식 아키텍처는 파운데이션 모델과 순차 제어(Sequential Control)를 연결하는 다리 역할을 한다. 로봇 행동을 생성 모델링(Generative Modeling) 문제로 전환한다. 지시는 조건 토큰(Conditioning Token)이 되고, 관측은 문맥이 되며, 행동은 출력이 되고, 궤적은 체화 지능(Embodied Intelligence)의 학습 언어가 된다.

가장 효과적인 구현은 트랜스포머 하나에만 의존하지 않을 것이다. 디코더 전용 시퀀스 생성과 시각 인코더(Visual Encoder), 월드 모델(World Model), 메모리 시스템(Memory System), 안전 제약(Safety Constraint), 저수준 제어기, 런타임 모니터링(Runtime Monitoring)을 함께 결합해야 한다. 이러한 계층적 시스템 안에서 디코더 전용 트랜스포머는 멀티모달 문맥을 일관된 행동 시퀀스로 변환하는 중앙 의사결정 엔진(Central Decision Engine)의 역할을 수행할 수 있다.

## 2.5 Encoder Decoder Architecture for Robot Control [w/Code]

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

인코더-디코더 아키텍처(Encoder-Decoder Architecture)는 복잡한 관측(Observation)을 의미 있는 출력(Output)으로 변환하기 위한 가장 중요한 구조적 패턴 가운데 하나이다. 자연어 처리(Natural Language Processing)에서는 원래 하나의 언어로 된 입력 시퀀스를 다른 언어의 출력 시퀀스로 변환하는 데 사용되었다. 로보틱스(Robotics)에서는 동일한 개념을 훨씬 더 넓은 문제로 확장할 수 있다. 즉, 멀티모달 관측(Multimodal Observation), 작업 목표(Task Goal), 환경 문맥(Environmental Context), 로봇 상태(Robot State)를 행동(Action), 궤적(Trajectory), 제어 명령(Control Command)으로 변환하는 것이다. 인코더(Encoder)는 현재 상황을 해석하고, 디코더(Decoder)는 그 해석으로부터 이어져야 할 행동을 생성한다.

이러한 분리는 인식(Perception)과 제어(Control)가 서로 다른 계산적 역할을 수행하기 때문에 유용하다. 인코더는 카메라(Camera), LiDAR, 깊이 센서(Depth Sensor), 고유감각(Proprioception), 힘 센서(Force Sensor), 지도(Map), 언어 명령(Language Instruction)의 정보를 조직해야 한다. 디코더는 인코딩된 표현을 시간적으로 일관된 행동 시퀀스(Temporally Coherent Action Sequence)로 변환해야 한다. 이처럼 역할을 분리하면 입력 이해와 출력 생성을 위해 서로 다른 토큰화(Tokenization), 어텐션 마스크(Attention Mask), 시간 해상도(Temporal Resolution), 최적화 목표(Optimization Objective)를 사용할 수 있다.

일반적인 로봇 인코더는 여러 종류의 정보를 입력받는다. 시각 관측은 비전 트랜스포머(Vision Transformer)를 통해 패치 토큰(Patch Token)으로 변환될 수 있다. 포인트 클라우드(Point Cloud)는 지역 영역으로 분할된 뒤 기하학적 토큰(Geometric Token)으로 인코딩될 수 있다. 관절 위치(Joint Position), 속도(Velocity), 토크(Torque), 그리퍼 상태(Gripper State)는 고유감각 임베딩(Proprioceptive Embedding)으로 투영할 수 있다. 언어 명령은 텍스트 토크나이저(Text Tokenizer)를 이용해 토큰화할 수 있다. 내비게이션 목표(Navigation Goal), 객체 자세(Object Pose), 지도, 안전 제약(Safety Constraint)은 추가적인 구조화 토큰(Structured Token)으로 표현할 수 있다.

이러한 입력들은 본래 동일한 형식으로 표현되지 않는다. 카메라 이미지는 공간적으로 조밀하고, 언어는 순차적이며, 고유감각은 저차원이지만 고주파로 변화하고, 포인트 클라우드는 3차원 공간의 순서 없는 집합(Unordered Set)이다. 따라서 인코더는 서로 이질적인 정보를 하나의 공통 잠재 표현(Common Latent Representation) 안에서 정렬해야 한다. 일반적으로 모달리티별 프런트엔드 네트워크(Modality-Specific Front-End Network)가 각 정보원을 먼저 처리하고, 이후 생성된 특징을 결합한다.

인코더는 일반적으로 양방향 자기 주의(Bidirectional Self-Attention)를 사용하는 트랜스포머 블록(Transformer Block)으로 구성된다. 현재의 전체 관측이 이미 사용 가능하므로 각 입력 토큰은 관련된 다른 모든 입력 토큰에 어텐션할 수 있다. "빨간 용기(red container)"를 설명하는 언어 토큰은 빨간 물체가 포함된 시각 영역에 어텐션할 수 있다. 그리퍼 상태 토큰은 목표 물체 토큰에 어텐션할 수 있다. 내비게이션 목표는 장애물, 위치 추정값(Localization Estimate), 의미 지도(Semantic Map)의 영역을 참조할 수 있다.

양방향 어텐션은 현재 관측 안에서 미래 생성 제약을 유지할 필요가 없기 때문에 인식에 유리하다. 사용 가능한 모든 정보를 동시에 통합할 수 있다. 이를 통해 인코더는 각 토큰의 의미가 전체 장면에 따라 결정되는 문맥 표현(Contextual Representation)을 구성할 수 있다.

디코더는 인코딩된 표현을 받아 출력 시퀀스를 생성한다. 로봇 제어에서 출력은 행동 토큰(Action Token), 엔드이펙터 자세(End-Effector Pose), 관절 명령(Joint Command), 웨이포인트(Waypoint), 스킬 식별자(Skill Identifier), 힘 목표(Force Target), 행동 청크(Action Chunk)로 구성될 수 있다. 디코더는 일반적으로 이전에 생성한 출력에 대해 인과적 자기 주의(Causal Self-Attention)를 수행하고, 인코더 표현에 대해 교차 어텐션(Cross-Attention)을 수행한다.

인과적 자기 주의는 각 출력 단계가 이전에 생성된 행동에만 의존하도록 만든다. 교차 어텐션은 디코더가 필요할 때마다 인코딩된 관측으로부터 정보를 검색할 수 있도록 한다. 이 두 가지 어텐션은 서로 다른 의존성을 처리한다. 자기 주의는 행동 시퀀스 내부의 시간 관계를 모델링하고, 교차 어텐션은 각 행동을 환경과 작업의 관련 부분에 연결한다.

예를 들어 로봇이 병을 집어 트레이 위에 놓으라는 지시를 받았다고 가정해 보자. 인코더는 언어 명령, 병, 트레이, 그리퍼 자세, 주변 장애물, 현재 관절 구성을 표현할 수 있다. 디코딩 초기에 행동 질의(Action Query)는 병과 그리퍼에 강하게 어텐션할 수 있다. 이후의 행동 질의는 들어 올리는 경로의 자유 공간(Free Space)에 어텐션할 수 있다. 마지막 행동 질의는 트레이의 위치와 방향에 어텐션할 수 있다.

따라서 교차 어텐션은 동적 정보 검색 메커니즘(Dynamic Information-Retrieval Mechanism)으로 작동한다. 디코더는 입력의 모든 세부사항을 하나의 고정 벡터로 압축할 필요가 없다. 각 행동 단계에서 현재 실행 단계에 가장 관련된 인코더 토큰을 선택적으로 참조할 수 있다. 이것은 전체 입력을 하나의 병목 표현(Bottleneck Representation)에 의존했던 초기 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델보다 트랜스포머 기반 인코더-디코더 구조가 가지는 중요한 장점이다.

인코더 출력은 토큰 수준의 구조(Token-Level Structure)를 유지할 수 있다. 시각 영역, 객체 표현, 센서 상태, 명령 토큰은 개별적으로 접근 가능한 상태로 남는다. 디코더는 작업 진행 단계에 따라 서로 다른 토큰 집합에 어텐션할 수 있다. 이러한 특성은 서로 다른 환경 특징이 시간에 따라 중요해지는 장기 로봇 작업(Long-Horizon Robot Task)에 적합하다.

인코더 표현의 선택은 제어 품질에 큰 영향을 준다. 조밀한 시각 인코더(Dense Visual Encoder)는 세밀한 공간 정보를 유지할 수 있지만 실시간 디코딩에 지나치게 많은 토큰을 생성할 수 있다. 지나치게 압축된 인코더는 효율적일 수 있지만 작은 객체, 좁은 통로, 정밀한 파지 기하(Grasp Geometry)를 잃을 수 있다. 따라서 로봇 시스템은 표현의 세밀함과 계산 비용 사이의 균형을 맞추어야 한다.

객체 중심 인코딩(Object-Centric Encoding)은 하나의 해결책이 될 수 있다. 모든 이미지 패치를 디코더에 전달하는 대신 인식 시스템이 객체, 표면(Surface), 장애물(Obstacle), 도구(Tool), 사람(Human), 로봇 부품(Robot Part)을 나타내는 토큰을 생성한다. 이러한 토큰에는 범주(Category), 자세(Pose), 크기(Size), 속도(Velocity), 불확실성(Uncertainty), 어포던스(Affordance) 정보가 포함될 수 있다. 이후 디코더는 압축된 장면 표현(Scene Representation)을 기반으로 추론한다.

그러나 객체 중심 인코딩은 신뢰할 수 있는 검출(Detection)과 분할(Segmentation)에 의존한다. 인식 시스템이 고정된 범주를 너무 일찍 확정하면 미지의 객체나 모호한 영역이 누락될 수 있다. 조밀한 잠재 인코딩(Dense Latent Encoding)은 더 유연하지만 해석 가능성이 낮고 계산 비용이 높다. 하이브리드 인코더(Hybrid Encoder)는 소수의 객체 토큰과 선택된 조밀한 시각 특징을 함께 유지할 수 있다.

공간 표현(Spatial Representation)은 로봇 제어에서 특히 중요하다. 언어 트랜스포머는 주로 1차원 토큰 순서를 모델링하지만, 로봇은 3차원 기하학적 공간에서 동작한다. 따라서 위치 인코딩(Positional Encoding)은 이미지 좌표(Image Coordinate), 깊이(Depth), 월드 좌표(World Coordinate), 카메라 자세(Camera Pose), 로봇 좌표계(Robot-Frame Coordinate), 시간 정보(Time Stamp)를 포함할 수 있다. 상대 기하 인코딩(Relative Geometric Encoding)은 객체, 센서, 로봇 링크(Robot Link) 사이의 거리와 방향을 표현할 수 있다.

인코더는 운동학적 구조(Kinematic Structure)도 포함할 수 있다. 관절 토큰은 로봇의 물리적 체인(Physical Chain)에 따라 연결할 수 있다. 어텐션 마스크나 그래프 기반 편향(Graph-Based Bias)은 인접 링크 사이의 관계를 강조할 수 있다. 엔드이펙터 토큰에는 순기구학(Forward Kinematics) 정보가 포함될 수 있다. 이를 통해 모델은 물리적으로 연결된 로봇 구성요소와 서로 관련 없는 수치형 상태 변수를 구분할 수 있다.

여러 관측 프레임을 인코딩하여 시간 정보를 포함할 수도 있다. 단일 이미지만으로는 객체 속도(Object Velocity), 접촉 변화(Contact Change), 움직임 의도(Motion Intent)를 알기 어렵다. 최근 관측 시퀀스를 사용하면 인코더는 시간에 따른 동역학을 추정하고 사건을 추적할 수 있다. 그러나 시간축이 추가되면 토큰 수가 크게 증가한다.

시간 압축(Temporal Compression)은 이러한 비용을 줄일 수 있다. 인코더는 각 프레임을 독립적으로 처리한 뒤 프레임 수준 특징(Frame-Level Feature)을 집계할 수 있다. 분해형 어텐션(Factorized Attention)은 공간 처리와 시간 처리를 분리할 수 있다. 순환 메모리(Recurrent Memory)는 오래된 관측을 요약할 수 있다. 이벤트 기반 선택(Event-Based Selection)은 의미 있는 변화가 발생한 프레임만 유지할 수 있다.

디코더는 여러 형태의 행동을 생성할 수 있다. 이산 행동 토큰(Discrete Action Token)은 표준 범주형 예측(Categorical Prediction)을 사용할 수 있어 편리하다. 연속 행동(Continuous Action)은 정밀 제어에 더 자연스럽다. 따라서 디코더는 잠재 행동 토큰(Latent Action Token)을 생성하고, 이후 연속 제어 헤드(Continuous Control Head)가 이를 실제 제어값으로 변환하는 구조를 사용할 수 있다.

조작 작업에서는 디코더가 엔드이펙터 위치, 방향(Orientation), 그리퍼 상태, 힘(Force), 관절 목표를 예측할 수 있다. 이동 로봇에서는 선속도(Linear Velocity), 각속도(Angular Velocity), 곡률 명령(Curvature Command), 지역 웨이포인트(Local Waypoint)를 생성할 수 있다. 사족보행 로봇(Quadruped Robot)에서는 몸체 속도 목표(Body Velocity Target), 발 디딤 위치(Footstep Location), 보행 파라미터(Gait Parameter)를 생성할 수 있다. 무인항공기(UAV)에서는 자세 목표(Attitude Target), 추력 명령(Thrust Command), 궤적 구간(Trajectory Segment)을 생성할 수 있다.

직접적인 저수준 제어(Direct Low-Level Control)도 가능하지만 항상 바람직한 것은 아니다. 트랜스포머 추론은 모터 제어 루프(Motor-Control Loop)가 요구하는 엄격한 시간성과 결정론(Determinism)을 만족하지 못할 수 있다. 일반적인 구조는 디코더가 상위 수준 또는 중간 수준 제어를 담당하고, 전통적인 제어기(Conventional Controller)가 빠른 서보 실행(Servo Execution)을 담당하는 방식이다.

디코더는 개별 제어 명령 대신 행동 청크(Action Chunk)를 생성할 수 있다. 행동 청크는 짧은 미래 행동 시퀀스를 포함한다. 이를 통해 자기회귀 디코딩 비용을 줄이고 국소 움직임의 일관성을 향상시킬 수 있다. 로봇은 청크의 일부를 실행한 뒤 새로운 예측을 요청할 수 있다.

이동 지평선 제어(Receding-Horizon Control)는 인코더-디코더 구조와 특히 잘 맞는다. 각 제어 주기마다 인코더는 최신 관측을 처리하고, 디코더는 미래 행동 시퀀스를 생성한다. 전체 행동 가운데 첫 부분만 실행한다. 이후 시스템은 실행 결과로 변화한 상태를 다시 관측하고 환경을 재인코딩한 뒤 수정된 계획을 디코딩한다.

이러한 접근법은 학습된 시퀀스 생성과 지속적인 피드백(Continuous Feedback)을 결합한다. 모델이 실제 세계의 새로운 관측을 반복적으로 조건으로 사용하기 때문에 오류 누적을 줄일 수 있다. 움직이는 장애물, 파지 실패, 인간 개입(Human Intervention), 예상하지 못한 환경 변화에도 대응할 수 있다.

인코더-디코더 아키텍처는 계층적 제어(Hierarchical Control)도 표현할 수 있다. 인코더는 공유 월드 표현(Shared World Representation)을 생성하고, 서로 다른 디코더가 서로 다른 수준의 출력을 생성할 수 있다. 하나의 디코더는 기호형 스킬(Symbolic Skill)을 생성하고, 다른 디코더는 궤적을 생성하며, 또 다른 디코더는 작업 성공 가능성이나 위험을 추정할 수 있다.

상위 디코더(High-Level Decoder)는 접근(Approach), 파지(Grasp), 들어 올리기(Lift), 검사(Inspect), 배치(Place)와 같은 명령을 출력할 수 있다. 궤적 디코더(Trajectory Decoder)는 이러한 명령을 웨이포인트로 변환할 수 있다. 이후 저수준 제어기가 웨이포인트를 모터 명령으로 변환할 수 있다. 이러한 계층 구조는 모듈성(Modularity)을 높이고 안전 검증(Safety Validation)을 단순화할 수 있다.

여러 디코더가 하나의 인코더를 공유할 수도 있다. 하나의 인식 인코더는 행동 생성, 가치 추정(Value Estimation), 불확실성 예측(Uncertainty Prediction), 객체 추적(Object Tracking), 미래 상태 예측(Future-State Forecasting)을 동시에 지원할 수 있다. 멀티태스크 학습(Multi-Task Learning)은 인코더가 여러 목표에 유용한 정보를 유지하도록 강제하여 표현 품질을 향상시킬 수 있다.

예를 들어 하나의 디코더는 행동을 예측하고, 다른 디코더는 미래 관측을 예측하며, 또 다른 디코더는 충돌 확률(Collision Probability)을 추정할 수 있다. 공유 인코더는 제어, 동역학, 위험 사이의 관계를 학습한다. 그러나 멀티태스크 학습에서는 하나의 목표가 그래디언트를 지배하거나 다른 목표에 해로운 특징을 학습할 수 있으므로 신중한 균형 조정이 필요하다.

인코더-디코더 로봇 모델은 지도 모방학습(Supervised Imitation Learning)을 통해 학습할 수 있다. 인코더는 기록된 관측과 목표를 입력받고, 디코더는 전문가 행동(Expert Action)을 예측한다. 학습 중에는 교사 강제(Teacher Forcing)를 통해 실제 이전 행동을 제공하므로 모든 디코더 위치를 효율적으로 최적화할 수 있다.

배포 단계에서는 디코더가 자신의 이전 출력을 다시 사용한다. 이로 인해 노출 편향(Exposure Bias)이 발생한다. 모델이 작은 오류를 만들면 이후 행동이 시연 데이터에 거의 없던 상태를 조건으로 생성될 수 있다. 복구 데이터(Recovery Data), 교란 학습(Perturbation Training), 스케줄드 샘플링(Scheduled Sampling), 온라인 미세조정(Online Fine-Tuning), 강화학습(Reinforcement Learning)은 이러한 문제를 줄일 수 있다.

이 아키텍처는 오프라인 강화학습(Offline Reinforcement Learning)과도 호환된다. 보상(Reward), 리턴 투 고(Return-to-Go), 원하는 결과(Desired Outcome)를 인코더 입력에 포함할 수 있다. 이후 디코더는 목표 성능 수준을 조건으로 행동을 생성한다. 또는 가치 디코더(Value Decoder)가 정책 디코더가 생성한 후보 행동 시퀀스를 평가할 수 있다.

동일한 작업을 서로 다른 품질로 수행하는 여러 궤적이 있을 경우 선호 최적화(Preference Optimization)를 적용할 수 있다. 인간 작업자는 더 안전하고, 부드럽고, 빠르며, 에너지 효율적인 시퀀스를 선호할 수 있다. 디코더는 거부된 궤적보다 선호된 궤적의 확률을 높이는 방향으로 미세조정될 수 있다.

피지컬 AI에서 안전성은 학습된 선호에만 의존해서는 안 된다. 강제되어야 하는 제약은 신경망 외부 또는 병렬 구조에서 처리해야 한다. 관절 제한(Joint Limit), 속도 제한(Speed Limit), 충돌 구역(Collision Zone), 힘 임계값(Force Threshold), 비상 정지 조건(Emergency-Stop Condition)은 결정론적 안전 구성요소(Deterministic Safety Component)가 검사해야 한다.

디코더가 안전 관련 인코더 토큰을 입력받더라도 독립적인 런타임 안전 계층(Runtime Safety Layer)이 실행 전에 출력을 검증해야 한다. 유효하지 않거나 위험한 행동은 거부하거나, 안전 영역(Safe Region)으로 투영하거나, 대체 명령(Fallback Command)으로 교체할 수 있다. 이는 학습 모델이 행동을 제안하고 안전 시스템이 이를 승인하는 계층적 아키텍처를 구성한다.

교차 어텐션 패턴(Cross-Attention Pattern)은 제한적인 해석 가능성(Interpretability)을 제공할 수 있다. 엔지니어는 생성된 행동에 어떤 시각 또는 상태 토큰이 영향을 주었는지 확인할 수 있다. 디코더가 파지 명령을 생성한 경우 올바른 객체와 그리퍼 영역에 집중했는지를 어텐션을 통해 확인할 수 있다.

그러나 어텐션 가중치를 완전한 인과적 설명(Causal Explanation)으로 간주해서는 안 된다. 모델은 어텐션 맵으로 명확히 드러나지 않는 분산 특징(Distributed Feature)에 의존할 수 있다. 따라서 해석 가능성 평가에는 반사실적 테스트(Counterfactual Testing), 입력 마스킹(Input Masking), 특징 기여도 분석(Feature Attribution), 궤적 재생(Trajectory Replay), 실패 분석(Failure Analysis)이 포함되어야 한다.

인코더-디코더 구조는 관측 처리와 행동 생성을 자연스럽게 분리하기 때문에 멀티모달 로봇 제어에 특히 유용하다. 비전, 언어, 고유감각, 오디오(Audio), 촉각(Touch), 지도를 서로 다른 인코더가 처리한 뒤 출력을 결합할 수 있다. 디코더는 교차 어텐션을 통해 융합된 표현에 접근할 수 있다.

일부 시스템은 하나의 공유 멀티모달 인코더(Shared Multimodal Encoder)를 사용한다. 다른 시스템은 여러 전문 인코더(Specialized Encoder)를 사용한 뒤 융합 모듈(Fusion Module)을 적용한다. 공유 인코더는 강한 교차 모달 정렬(Cross-Modal Alignment)을 유도할 수 있고, 전문 인코더는 모달리티별 귀납적 편향을 유지하며 학습 난도를 낮출 수 있다.

초기 융합(Early Fusion)은 깊은 처리 이전에 토큰을 결합한다. 이는 모달리티 사이의 풍부한 상호작용을 가능하게 하지만 계산 비용이 높다. 후기 융합(Late Fusion)은 각 모달리티를 개별 처리한 뒤 상위 특징을 결합한다. 더 효율적이지만 저수준 교차 모달 관계를 놓칠 수 있다. 중간 융합(Intermediate Fusion)은 현실적인 절충안이 되는 경우가 많다.

예를 들어 카메라와 깊이 특징은 기하학적으로 정렬되어 있기 때문에 초기에 융합할 수 있다. 언어는 이후 교차 어텐션을 통해 융합할 수 있다. 고유감각은 작업 지시보다 높은 주기로 변하므로 반복적으로 삽입할 수 있다. 최적의 융합 전략은 로봇과 작업에 따라 달라진다.

교차 체화 학습(Cross-Embodiment Learning)에서도 인코더-디코더 구조를 사용할 수 있다. 공유 인코더는 작업과 환경을 체화 독립 공간(Embodiment-Independent Space)으로 표현한다. 이후 로봇별 디코더(Robot-Specific Decoder)가 각 플랫폼에 적합한 행동을 생성한다.

이러한 분리는 인식과 작업 의미(Task Semantics)가 모터 명령보다 더 쉽게 전이될 수 있기 때문에 매력적이다. 매니퓰레이터, 휴머노이드(Humanoid), 이동 로봇, 사족보행 로봇은 서로 다른 행동 시퀀스를 필요로 하지만 동일한 객체를 보고 동일한 명령을 이해할 수 있다.

체화별 토큰(Embodiment-Specific Token)은 로봇의 능력(Capability), 크기(Dimension), 센서 배치(Sensor Placement), 운동학(Kinematics), 적재하중(Payload), 제어 한계(Control Limit)를 인코더에 제공할 수 있다. 디코더는 공유된 장면 표현을 현재의 로봇 몸체에 맞게 해석한다.

반대로 각 로봇이 전문 인코더를 사용하고, 정규화된 잠재 행동 공간(Normalized Latent Action Space)에서 공통 디코더를 공유할 수도 있다. 이는 작업 구조보다 센서 구성의 차이가 더 클 때 유용할 수 있다. 어떤 구성요소가 플랫폼 간에 일반화될 것으로 기대되는지에 따라 설계 선택이 달라진다.

인코더-디코더 아키텍처는 월드 모델 기반 제어(World-Model-Based Control)에도 적합하다. 인코더는 현재 관측을 잠재 상태(Latent State)로 변환한다. 예측 디코더(Predictive Decoder)는 후보 행동을 조건으로 미래 잠재 상태를 생성한다. 또 다른 디코더는 행동 자체를 생성할 수 있다.

모델은 여러 가능한 미래를 시뮬레이션하고 가장 바람직한 결과를 생성하는 행동 시퀀스를 선택할 수 있다. 모든 후보 궤적을 실제로 실행하지 않고도 계획을 수행할 수 있다. 그러나 예측 오차가 누적될 수 있으므로 상상된 궤적은 실제 관측을 통해 반복적으로 보정해야 한다.

잠재 상태 예측은 전체 해상도 이미지를 복원하는 것보다 효율적인 경우가 많다. 모델은 객체 위치, 자유 공간, 접촉(Contact), 작업 진행 상태(Task Progress)와 같이 제어에 필요한 정보만 유지하면 된다. 그러나 지나치게 압축된 잠재 공간은 예상하지 못한 위험이나 이후 중요해지는 세부사항을 누락할 수 있다.

강건한 시스템은 예측 잠재 표현과 선택적 복원(Selective Reconstruction)을 결합할 수 있다. 월드 모델은 압축된 미래 상태를 예측하고, 보조 디코더(Auxiliary Decoder)는 핵심 시각, 기하학, 접촉 특징을 복원한다. 이를 통해 효율성과 진단 가능성(Diagnostic Visibility)을 동시에 확보할 수 있다.

학습 데이터 품질은 매우 중요하다. 인코더는 다양한 환경, 센서 조건, 객체 외형, 로봇 상태를 경험해야 한다. 디코더는 성공 행동뿐 아니라 복구, 실패 직전 상황(Near Failure), 불확실성에 대한 안전한 대응을 학습해야 한다. 이상적인 시연만 포함된 데이터셋은 취약한 행동을 만들 수 있다.

인코더-디코더 구조는 관측과 행동 사이의 관계를 학습하기 때문에 동기화(Synchronization)가 특히 중요하다. 카메라 프레임, 관절 상태, 힘 센서 값, 명령은 정확한 시간에 정렬되어야 한다. 타임스탬프가 일관되지 않으면 디코더가 잘못된 물리 상태와 행동을 연결하여 학습할 수 있다.

보정(Calibration)도 중요하다. 카메라 내부 파라미터(Camera Intrinsics), 외부 파라미터(Extrinsics), 로봇 좌표계, 센서 변환(Sensor Transformation)은 알려져 있거나 일관되게 표현되어야 한다. 기하학적 불일치는 모델의 학습 부담을 증가시키고 로봇과 환경 사이의 전이 성능을 저하시킬 수 있다.

인코더-디코더 모델은 작업별 미세조정 전에 사전학습(Pretraining)할 수 있다. 시각 인코더는 마스크드 이미지 모델링(Masked Image Modeling)으로 사전학습할 수 있다. 언어 구성요소는 언어 파운데이션 모델(Language Foundation Model)에서 초기화할 수 있다. 고유감각과 행동 표현은 궤적 복원(Trajectory Reconstruction)이나 다음 상태 예측(Next-State Prediction)을 통해 사전학습할 수 있다.

멀티모달 사전학습(Multimodal Pretraining)은 전체 정책 학습 전에 언어, 비전, 상태, 행동을 정렬할 수 있다. 예를 들어 모델은 지시와 시연을 매칭하거나, 하나의 궤적이 목표를 만족하는지 예측하거나, 누락된 센서 채널을 복원하거나, 미래 상태를 예측하도록 학습할 수 있다.

사전학습 이후 지도 미세조정(Supervised Fine-Tuning)은 작업별 제어를 가르친다. 선호 최적화는 행동 품질을 향상시킬 수 있다. 시뮬레이션 기반 강화학습은 장기 성능을 개선할 수 있다. 이후 제한적인 실제 환경 적응(Real-World Adaptation)을 통해 남아 있는 시뮬레이션-현실 차이(Simulation-to-Reality Gap)를 보정할 수 있다.

이 아키텍처는 파라미터 효율적 적응(Parameter-Efficient Adaptation)도 지원한다. 대형 공유 인코더는 대부분 고정한 채 작은 어댑터(Adapter)나 저랭크 모듈(Low-Rank Module)을 통해 새로운 로봇에 맞게 특화할 수 있다. 전체 파운데이션 모델을 수정하지 않고 로봇별 디코더만 학습할 수도 있다.

이 방식은 학습 비용을 줄이고 파국적 망각(Catastrophic Forgetting)을 방지하는 데 도움이 된다. 또한 공통 인식 및 추론 백본이 여러 하드웨어 변형을 지원할 수 있으므로 플릿 배포(Fleet Deployment)를 단순화한다.

추론 효율성(Inference Efficiency)은 여전히 중요한 과제이다. 인코더는 수백 또는 수천 개의 센서 토큰을 처리할 수 있고, 디코더는 여러 행동 단계를 생성해야 한다. 고해상도 비전과 긴 관측 이력은 지연 시간과 메모리 사용량을 증가시킨다.

효율적 어텐션(Efficient Attention), 토큰 가지치기(Token Pruning), 잠재 리샘플링(Latent Resampling), 혼합 정밀도(Mixed Precision), 양자화(Quantization), 지식 증류(Knowledge Distillation)는 이러한 비용을 줄일 수 있다. 인코더는 천천히 변하는 장면 정보는 낮은 빈도로 갱신하고, 고주파 상태 토큰은 더 자주 갱신할 수 있다.

캐싱(Caching)도 도움이 된다. 언어 명령과 정적인 지도 특징은 매 제어 주기마다 다시 인코딩할 필요가 없다. 이들의 표현은 캐시에 저장하고, 동적인 카메라와 고유감각 입력만 갱신할 수 있다. 교차 어텐션은 캐시된 정보와 현재 정보를 함께 사용할 수 있다.

비동기 처리(Asynchronous Processing)는 성능을 더 향상시킬 수 있다. 시각 인코더는 카메라 주기로 실행하고, 상태 인코더는 제어 주기로 실행하며, 언어 인코더는 새로운 명령이 들어올 때만 실행할 수 있다. 융합 계층은 가장 최근의 표현을 결합한다.

그러나 비동기 동작은 시간 문제를 만든다. 디코더는 각 표현이 얼마나 오래된 것인지를 알아야 한다. 타임스탬프 임베딩(Timestamp Embedding)과 최신성 표시(Freshness Indicator)는 오래된 관측을 현재 정보로 잘못 처리하는 문제를 줄일 수 있다.

평가는 표현 품질과 폐루프 제어 성능(Closed-Loop Control Performance)을 모두 포함해야 한다. 인코더 벤치마크는 객체 인식(Object Recognition), 언어 접지(Language Grounding), 공간 추론(Spatial Reasoning), 상태 추정(State Estimation)을 평가할 수 있다. 디코더 평가는 행동 오차(Action Error), 궤적 부드러움(Trajectory Smoothness), 시퀀스 가능도(Sequence Likelihood)를 평가할 수 있다.

개방 루프 예측(Open-Loop Prediction)보다 폐루프 평가가 더 중요하다. 모델은 기록된 행동을 정확히 모방하더라도 자신의 행동이 미래 관측을 바꾸는 상황에서는 제대로 동작하지 못할 수 있다. 따라서 시뮬레이션과 실제 실행을 통해 작업 성공률(Success Rate), 충돌 빈도(Collision Frequency), 완료 시간(Completion Time), 에너지 사용량(Energy Use), 복구 행동(Recovery Behavior), 불확실성 처리(Uncertainty Handling), 인간 개입 빈도(Human Intervention Frequency)를 평가해야 한다.

분포 밖 평가(Out-of-Distribution Evaluation)는 필수적이다. 로봇은 익숙하지 않은 객체, 조명, 배경, 시점, 레이아웃(Layout), 적재하중, 센서 노이즈에서 시험되어야 한다. 인코더는 의미 있는 표현을 생성해야 하고, 디코더는 상황이 익숙하지 않을 때 과도하게 확신하는 위험한 행동을 피해야 한다.

불확실성은 두 단계 모두에서 모델링할 수 있다. 인코더는 객체, 자세, 장면 표현에 대한 신뢰도를 추정할 수 있다. 디코더는 행동 시퀀스에 대한 불확실성을 추정할 수 있다. 높은 불확실성은 추가 센싱(Additional Sensing), 속도 감소(Slower Execution), 인간 지원(Human Assistance), 보수적 대체 정책(Conservative Fallback Policy)을 유도할 수 있다.

인코더-디코더 아키텍처에는 중요한 한계도 있다. 분리된 두 스택은 디코더 전용 모델보다 파라미터 수와 지연 시간을 증가시킬 수 있다. 교차 어텐션은 추가적인 계산 비용을 발생시킨다. 여러 모달리티를 공동 학습하는 것은 어렵고, 모델은 물리적 인과 관계 대신 지름길 패턴(Shortcut)을 학습할 수 있다.

인코딩과 디코딩의 분리는 정보 병목(Information Bottleneck)을 만들 수도 있다. 인코더가 이후 중요해지는 정보를 버리면 디코더는 이를 복원할 수 없다. 신중한 토큰 보존(Token Preservation), 스킵 연결(Skip Connection), 다중 스케일 특징(Multi-Scale Feature), 반복적 재인코딩(Recurrent Re-Encoding)은 이러한 위험을 줄일 수 있다.

이러한 한계에도 불구하고 인코더-디코더 아키텍처는 문제의 논리적 구조를 그대로 반영하기 때문에 로봇 제어에 매우 매력적이다. 로봇은 먼저 세계를 해석하고, 이후 행동을 생성해야 한다. 인코더는 작업, 환경, 로봇 몸체의 문맥 표현을 제공한다. 디코더는 이 표현을 시간적으로 조직된 행동 시퀀스로 변환한다.

피지컬 AI 파운데이션 모델(Physical AI Foundation Model)에서 인코더-디코더 아키텍처는 멀티모달 인식과 체화 행동(Embodied Action)을 연결하는 유연한 다리를 제공한다. 언어 조건부 제어(Language-Conditioned Control), 객체 중심 조작(Object-Centric Manipulation), 내비게이션(Navigation), 교차 체화 전이(Cross-Embodiment Transfer), 월드 모델 계획(World-Model Planning), 행동 청킹(Action Chunking), 계층적 실행(Hierarchical Execution)을 지원할 수 있다.

가장 효과적인 사용 방식은 계층적 로봇 시스템 안에 배치하는 것이다. 인코더는 인식과 문맥을 결합하고, 디코더는 목표나 궤적을 생성하며, 전통적인 제어기는 정밀한 움직임을 실행하고, 독립적인 안전 모듈은 전체 과정을 감독한다. 이러한 구조에서 인코더-디코더 트랜스포머는 물리 세계에 대한 이해를 통제되고 목적 지향적인 로봇 행동으로 변환하는 중심 번역 메커니즘(Central Translation Mechanism)의 역할을 수행한다.

## 2.6 State Space Models Mamba for Sequence Robot Data [w/Code]

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

상태공간 모델(State Space Model)은 긴 시퀀스 데이터(Long Sequential Data)를 처리하기 위한 트랜스포머(Transformer) 아키텍처의 중요한 대안으로 부상하고 있다. 이러한 모델은 카메라 특징(Camera Feature), 관절 상태(Joint State), 힘 측정값(Force Measurement), LiDAR 관측, 내비게이션 상태(Navigation State), 행동(Action), 임무 이벤트(Mission Event), 환경 변화(Environmental Change)가 연속적으로 유입되는 로보틱스(Robotics)에서 특히 높은 가치를 가진다. 짧고 독립적인 입력을 처리하는 많은 디지털 작업과 달리, 로봇 지능(Robot Intelligence)은 새로운 관측에 빠르게 대응하면서도 장시간에 걸쳐 유용한 정보를 유지해야 한다. 상태공간 모델은 압축된 내부 상태(Compact Internal State)를 통해 이러한 시간적 변화(Temporal Evolution)를 표현하는 수학적 프레임워크(Mathematical Framework)를 제공한다.

상태공간 모델은 시간에 따라 변화하는 내부 상태를 이용하여 동적 시스템(Dynamic System)을 설명한다. 이 상태는 이전 입력에서 얻은 정보를 요약하고, 시스템이 이후 입력에 어떻게 반응할지를 결정한다. 각 시간 단계(Time Step)에서 모델은 관측값을 입력받고, 은닉 상태(Hidden State)를 갱신하며, 출력값을 생성한다. 이러한 구조는 로봇이 해결해야 하는 기본적인 문제와 매우 유사하다. 로봇은 세계를 관측하고, 환경과 자신의 상태에 대한 믿음(Belief)을 갱신한 뒤, 행동이나 예측을 생성한다.

고전적인 선형 상태공간 표현(Linear State Space Formulation)에서는 한 시간 단계의 은닉 상태가 상태 전이 함수(State Transition Function)를 통해 다음 시간 단계의 상태로 변환된다. 현재 입력은 이 전이에 영향을 주며, 출력 함수(Output Function)는 은닉 상태를 관측 가능한 예측값으로 변환한다. 기본 방정식이 선형일 수는 있지만, 현대의 상태공간 모델은 학습 가능한 파라미터(Learnable Parameter), 비선형 변환(Nonlinear Transformation), 선택적 메커니즘(Selective Mechanism), 심층 신경망(Deep Neural Architecture)을 도입한다.

이러한 표현 방식의 핵심 장점은 전체 과거 이력을 명시적으로 계속 보관할 필요가 없다는 것이다. 과거 정보는 현재 상태에 압축된다. 스트리밍 추론(Streaming Inference)에서 메모리 사용량이 전체 시퀀스 길이에 비례하여 증가하지 않을 수 있기 때문에, 상태공간 모델은 장시간 시퀀스에 특히 적합하다.

로봇 데이터는 본질적으로 이러한 순차 구조에 잘 맞는다. 이동형 로봇(Mobile Robot)은 오도메트리(Odometry), 관성 측정값(Inertial Measurement), 위치 추정값(Localization Estimate), 장애물 정보, 임무 명령(Mission Command), 배터리 상태, 액추에이터 피드백(Actuator Feedback)을 지속적으로 입력받는다. 매니퓰레이터(Manipulator)는 관절 위치, 속도, 토크(Torque), 카메라 관측, 힘 신호, 그리퍼 상태(Gripper State), 접촉 이벤트(Contact Event)를 수신한다. 피지컬 AI 모델(Physical AI Model)은 각 관측을 독립적으로 처리하는 것이 아니라 시간에 따라 이러한 신호를 결합해야 한다.

단일 센서 프레임(Sensor Frame)만으로는 올바른 행동을 결정하기 어려운 경우가 많다. 하나의 이미지는 물체를 보여줄 수 있지만 그 물체가 움직이는지는 알려주지 못할 수 있다. 하나의 힘 측정값은 접촉을 나타낼 수 있지만 힘이 증가하고 있는지 감소하고 있는지는 알려주지 못한다. 관절 위치는 현재 자세를 보여주지만 움직임의 방향은 알려주지 않는다. 시간 모델링(Temporal Modeling)은 시스템이 변화 추세, 상태 전환(Transition), 인과 관계(Causal Relationship)를 해석할 수 있게 한다.

순환 신경망(Recurrent Neural Network)은 이러한 문제를 다루기 위한 초기 신경망 접근법이었다. 순환 신경망은 은닉 상태를 순차적으로 갱신하며, 이론적으로는 이전 관측의 정보를 유지할 수 있다. 장단기 메모리(Long Short-Term Memory, LSTM)와 게이트 순환 유닛(Gated Recurrent Unit, GRU)은 어떤 정보를 저장하고, 갱신하고, 잊을지를 제어하는 게이트(Gate)를 도입하였다.

그러나 기존 순환 모델은 매우 긴 의존성(Long Dependency)을 처리하는 데 어려움을 겪는 경우가 많다. 순차적인 계산 방식은 병렬 학습을 제한하며, 긴 시퀀스에서는 그래디언트(Gradient)가 소실되거나 불안정해질 수 있다. 트랜스포머는 모든 토큰 사이의 직접적인 상호작용을 가능하게 하는 자기 주의(Self-Attention)를 통해 이러한 한계를 상당 부분 해결하였다.

자기 주의는 유연한 전역 문맥(Global Context)과 높은 병렬 학습 능력을 지원하기 때문에 파운데이션 모델의 지배적인 메커니즘이 되었다. 그러나 계산량과 메모리 사용량은 일반적으로 시퀀스 길이의 제곱에 비례하여 증가한다. 이는 수시간의 로봇 운용 기록, 고주파 텔레메트리(High-Frequency Telemetry), 장시간 비디오 스트림, 다중 센서 이력(Multi-Sensor History)을 처리할 때 큰 비용을 발생시킨다.

상태공간 모델은 이와 다른 절충안을 제공한다. 시퀀스 길이에 대략 선형적으로 증가하는 계산량으로 데이터를 처리할 수 있다. 온라인 추론에서는 새로운 입력이 들어올 때마다 압축된 상태를 갱신할 수 있다. 이는 지속적으로 동작하는 로봇에 낮고 예측 가능한 지연 시간(Latency)을 제공할 수 있다.

현대의 구조화 상태공간 모델(Structured State Space Model)은 순환 구조의 효율성과 합성곱(Convolution) 및 병렬 시퀀스 모델의 학습 가능성을 결합하기 위해 개발되었다. 이러한 모델은 연속시간 동적 시스템(Continuous-Time Dynamic System)을 기반으로 하며, 디지털 계산을 위해 이산화(Discretization)된다. 신중하게 설계된 상태 전이 구조는 모든 토큰 쌍 사이의 상호작용을 직접 계산하지 않고도 장거리 의존성을 학습할 수 있게 한다.

은닉 상태는 시퀀스에 대한 학습된 메모리(Learned Memory)로 이해할 수 있다. 모든 과거 토큰을 저장하는 대신, 모델은 과거 이력을 압축된 표현으로 변환한다. 상태의 서로 다른 차원(Dimension)은 느린 변화 추세, 최근 변화, 주기적 행동(Periodic Behavior), 작업 진행 상태(Task Progress), 잠재 환경 조건(Latent Environmental Condition)을 나타낼 수 있다.

로보틱스에서는 관련 변수들이 서로 다른 속도로 변화하기 때문에 이러한 압축 메모리가 매우 유용하다. 모터 전류는 수 밀리초 안에 변할 수 있다. 객체 움직임은 수초에 걸쳐 전개될 수 있다. 배터리 상태와 임무 진행은 수분 또는 수시간에 걸쳐 변화할 수 있다. 상태공간 모델은 하나의 아키텍처 안에서 여러 시간 척도(Temporal Timescale)를 학습할 수 있다.

맘바(Mamba)는 상태공간 아키텍처가 입력 내용에 따라 달라지는 시퀀스를 더 효과적으로 처리하도록 설계된 현대적인 선택적 상태공간 모델(Selective State Space Model)이다. 기존의 선형 상태공간 모델은 학습이 끝난 이후 상태 전이 파라미터가 대체로 고정된다. 모든 입력을 동일한 기본 동역학으로 처리하기 때문에 계산 효율은 높지만, 어떤 정보를 기억하고 어떤 정보를 무시할지 입력별로 결정하는 능력은 제한될 수 있다.

맘바는 입력 의존적 선택성(Input-Dependent Selectivity)을 도입한다. 모델은 현재 입력에 따라 상태 갱신 방식의 일부를 변경할 수 있다. 실질적으로는 중요한 이벤트를 유지하고, 관련 없는 반복 정보를 버리며, 시퀀스 내용에 따라 시간적 동작을 조정하도록 학습할 수 있다.

이러한 선택성은 로봇 데이터에서 매우 중요하다. 대부분의 로봇 센서 스트림에는 많은 중복성이 존재한다. 빈 복도를 바라보는 카메라는 거의 동일한 프레임을 연속해서 생성할 수 있다. 로봇이 대기하는 동안 관절 상태는 거의 변하지 않을 수 있다. 정적인 환경에서는 LiDAR 스캔도 조금씩만 변한다. 모든 관측을 동일하게 처리하고 기억하는 것은 계산 자원과 메모리를 낭비한다.

반면 매우 짧은 특정 이벤트가 결정적으로 중요할 수 있다. 갑작스러운 힘의 급증은 충돌을 나타낼 수 있다. 잠깐 나타난 시각 정보가 사람이 이동 경로에 진입했음을 보여줄 수 있다. 일시적인 위치 추정 불일치는 지도 기반 위치 추정 실패를 나타낼 수 있다. 새로운 임무 명령은 로봇의 목표 자체를 변경할 수 있다. 선택적 상태공간 모델은 이러한 사건을 일반적인 반복 관측과 다르게 처리하도록 학습할 수 있다.

맘바의 선택적 메커니즘은 상태 갱신에 사용되는 파라미터가 입력 토큰에 따라 달라지도록 한다. 이는 동적 필터링(Dynamic Filtering)의 한 형태를 만든다. 모델은 새로운 정보가 상태에 얼마나 강하게 영향을 줄지, 이전 메모리의 어떤 부분을 유지할지, 상태가 출력에 어떻게 기여할지를 결정할 수 있다.

이는 순환 신경망의 게이팅(Gating)과 유사하지만, 맘바는 현대 가속기(Accelerator)에서 효율적으로 구현할 수 있는 구조화 상태공간 계산을 중심으로 설계되었다. 전체 시퀀스를 대상으로 병렬 학습을 수행하면서도 스트리밍 추론에서는 순환 방식으로 실행할 수 있다.

이러한 이중 형태(Dual Form)는 특히 유용하다. 학습 과정에서는 GPU 처리량을 활용하여 시퀀스를 효율적으로 병렬 처리할 수 있다. 배포 단계에서는 로봇이 압축된 순환 상태를 유지하면서 관측을 한 번에 하나씩 처리할 수 있다. 학습 방식과 추론 방식은 서로 다른 계산 형태를 사용하지만 동일한 학습 동역학을 표현한다.

맘바는 일반적으로 선형 시퀀스 복잡도(Linear Sequence Complexity)를 가진다고 설명된다. 이는 시퀀스 길이가 두 배가 될 때 주요 처리 연산도 대략 두 배가 된다는 의미이다. 완전 자기 주의(Full Self-Attention)처럼 약 네 배로 증가하지 않는다. 실제 이점은 구현 방식, 하드웨어, 상태 차원, 주변 모델 구성요소에 따라 달라지지만, 긴 시퀀스에서는 이러한 확장성 차이가 중요해진다.

메모리 효율성도 중요하다. 트랜스포머는 자기회귀 또는 스트리밍 추론 과정에서 이전 토큰의 키(Key)와 값(Value) 표현을 저장해야 할 수 있다. 이러한 키-값 캐시(Key-Value Cache)는 컨텍스트 길이에 따라 계속 증가한다. 상태공간 모델은 고정된 크기의 상태를 갱신하므로 메모리 요구량이 더 안정적으로 유지될 수 있다.

장시간 운용되는 이동 로봇에서는 모든 이전 관측을 저장하지 않고도 시간 문맥을 유지할 수 있다는 의미이다. 로봇은 내부 상태를 통해 최근 움직임, 작업 진행 상태, 장애물, 센서 상태에 대한 정보를 유지할 수 있다.

그러나 압축에는 상충 관계(Trade-Off)가 존재한다. 고정 크기의 상태는 무한히 긴 과거의 모든 세부사항을 보존할 수 없다. 모델은 어떤 정보가 중요한지를 학습해야 한다. 중요한 정보가 덮어쓰이거나 부정확하게 압축되면 이후에 복구하지 못할 수 있다.

자기 주의는 이전 토큰에 직접 접근할 수 있지만 순환 상태는 이전 갱신 결과를 압축한 정보만 제공한다. 따라서 선택적 메모리 설계(Selective Memory Design)가 맘바 기반 시스템의 성능을 결정하는 핵심 요소가 된다.

로봇 시퀀스는 멀티모달(Multimodal)이기도 하다. 하나의 시간 단계에는 시각 토큰, 관절 상태, 힘 값, 행동 명령, 지도 정보, 언어 문맥이 함께 포함될 수 있다. 이러한 모달리티는 크기, 주기, 구조, 중요성이 크게 다르다. 맘바 기반 로봇 모델은 이들을 하나의 시퀀스로 어떻게 구성할지, 또는 서로 다른 상태공간 모듈로 어떻게 처리할지를 결정해야 한다.

하나의 접근법은 초기 융합(Early Fusion)이다. 모든 모달리티 특징을 공통 임베딩 공간(Shared Embedding Space)으로 투영하고, 하나의 시간 순서 시퀀스에 배치한다. 모달리티 임베딩(Modality Embedding)은 각 토큰이 비전, 고유감각, 행동, 힘, 언어 가운데 무엇을 나타내는지를 표시한다. 이후 상태공간 모델이 모든 입력 사이의 시간 관계를 학습한다.

이 구조는 개념적으로 단순하며 직접적인 교차 모달 의존성(Cross-Modal Dependency)을 학습할 수 있다. 힘의 급증을 직전의 그리퍼 명령과 현재 이미지에 연결할 수 있다. 위치 추정 변화는 휠 오도메트리와 시각 랜드마크(Visual Landmark)에 연결할 수 있다.

그러나 초기 융합은 매우 긴 시퀀스를 만들 수 있다. 한 장의 카메라 영상은 수백 개의 시각 패치 토큰을 생성하지만 고유감각 상태는 몇 개의 값만 필요할 수 있다. 모든 모달리티를 전체 해상도로 표현하면 시각 정보가 시퀀스와 계산 비용을 지배할 수 있다.

더 실용적인 전략은 모달리티별 인코더(Modality-Specific Encoder)를 사용하는 것이다. 비전 트랜스포머, 합성곱 신경망(Convolutional Neural Network), 시각 상태공간 모델(Visual State Space Model)이 이미지를 소수의 잠재 토큰(Latent Token)으로 압축한다. 포인트 클라우드 인코더는 기하 데이터를 처리하고, 상태 인코더는 고유감각을 처리한다. 이후 압축된 표현이 공통 시간 맘바 모듈로 전달된다.

이 방식은 공간 처리(Spatial Processing)와 시간 처리를 분리한다. 시각 인코더는 각 프레임에 무엇이 존재하는지를 판단하고, 상태공간 모델은 중요한 특징이 시간에 따라 어떻게 변화하는지를 추적한다. 이러한 분리는 효율성을 높이고 서로 다른 센서에 맞게 아키텍처를 적응시키기 쉽게 만든다.

또 다른 설계는 각 모달리티마다 별도의 상태공간 스트림(State Space Stream)을 사용하는 것이다. 비전, 고유감각, 힘, 행동 이력을 독립적으로 처리한 뒤 상위 계층에서 융합한다. 이는 모달리티별 시간적 특성을 보존한다.

예를 들어 힘 센싱은 짧고 고주파인 상태를 요구할 수 있지만, 임무 문맥은 장기 기억을 필요로 한다. 별도의 모듈은 서로 다른 샘플링 속도와 상태 차원으로 동작할 수 있다. 이후 융합 계층(Fusion Layer)이 계획이나 행동 생성을 위해 이들의 출력을 결합한다.

다중 주기 처리(Multi-Rate Processing)는 로보틱스에서 매우 중요하다. 센서는 거의 항상 서로 다른 주기로 동작한다. 카메라는 30Hz 또는 60Hz, LiDAR는 10Hz 또는 20Hz, 모터 엔코더는 수백 또는 수천 Hz로 작동할 수 있으며, 언어 명령은 사용자가 지시할 때만 들어온다.

단순한 시퀀스 모델은 모든 데이터를 하나의 공통 주기로 리샘플링(Resampling)할 수 있지만, 이 과정에서 데이터를 낭비하거나 중요한 시간 정보를 잃을 수 있다. 상태공간 모델은 각 스트림이 자신의 상태를 유지하고 새로운 정보가 들어올 때만 갱신되는 비동기 또는 계층형 업데이트를 지원할 수 있다.

타임스탬프 임베딩(Timestamp Embedding)과 경과 시간(Elapsed Time) 정보를 포함하면 모델이 불규칙한 샘플링 간격을 이해할 수 있다. 이는 통신 지연, 패킷 손실, 계산 부하 변화로 관측 간 시간 간격이 달라지는 경우에 중요하다.

연속시간 상태공간 표현(Continuous-Time State Space Formulation)은 불규칙한 로봇 데이터에 특히 매력적이다. 상태 전이는 고정된 샘플링 주기를 가정하지 않고 측정 간 경과 시간에 따라 달라질 수 있기 때문이다.

실제로 많은 신경 상태공간 모델은 이산시간으로 구현되지만, 시간 간격 조건화(Time-Delta Conditioning)를 통해 이러한 능력을 근사할 수 있다. 모델은 10밀리초 간격의 두 관측과 1초 간격의 두 관측을 구분할 수 있다.

맘바는 시간 압축(Temporal Compression)에도 사용할 수 있다. 고주파 데이터는 단기 상태로 처리하고, 선택된 요약 정보만 느린 장기 모델로 전달할 수 있다. 이는 로봇 제어 계층 구조와 유사하다.

빠른 모듈은 모터 전류, 관절 움직임, 접촉을 감시할 수 있다. 느린 모듈은 물체 상호작용과 지역 작업 진행 상태를 추적할 수 있다. 임무 수준 모델은 검사 지점 순서, 배터리 관리, 장기 목표를 유지할 수 있다. 상태공간 모델은 이러한 계층 가운데 하나 또는 여러 계층에 사용될 수 있다.

시퀀스 길이는 로봇 비디오에서 특히 큰 문제이다. 30fps로 한 시간 동안 수행한 임무는 10만 개가 넘는 프레임을 포함한다. 각 프레임을 여러 패치 토큰으로 분할하면 전체 시퀀스는 매우 커진다.

시각 상태공간 아키텍처는 완전한 시공간 어텐션(Spatiotemporal Attention) 대신 상태공간 연산을 이용해 이미지와 비디오를 처리함으로써 이러한 부담을 줄일 수 있다. 공간 스캔(Spatial Scanning) 방식은 2차원 이미지를 정렬된 시퀀스로 변환하고, 시간 모듈은 프레임 간 변화를 추적한다.

그러나 이미지 구조는 언어의 순서 구조와 다르다. 2차원 이미지에는 하나의 자연스러운 시퀀스 방향이 존재하지 않는다. 따라서 비주얼 맘바(Visual Mamba) 변형은 높이와 너비 방향의 관계를 유지하기 위해 여러 스캔 방향(Multiple Scan Direction), 계층적 처리(Hierarchical Processing), 지역 윈도우(Local Window), 구조화된 공간 상태 갱신을 사용한다.

로봇 비디오 아키텍처는 공간적 세부사항과 시간적 연속성을 모두 유지해야 한다. 작은 장애물은 몇 개의 픽셀만 차지할 수 있지만 안전에는 매우 중요할 수 있다. 지나친 공간 압축은 이러한 정보를 제거할 수 있다.

다중 스케일 처리(Multi-Scale Processing)가 이를 보완할 수 있다. 낮은 해상도의 전역 특징은 장면 문맥을 제공하고, 선택된 고해상도 영역은 객체, 사람의 손, 공구, 장애물, 결함 영역을 보존한다. 시간 상태 모델은 작업과 관련된 영역에 더 많은 용량을 할당할 수 있다.

맘바는 고유감각 및 행동 시퀀스에도 적합하다. 이러한 신호는 이미지보다 차원이 낮지만 고주파·저지연 처리를 요구하는 경우가 많다. 관절 상태, 속도, 토크, 모터 전류, 제어 명령은 강한 동역학을 가진 구조화된 시계열(Time Series)을 형성한다.

상태공간 모델은 움직임 패턴을 학습하고, 제어 응답의 변화를 식별하며, 잠재 물리 상태(Latent Physical Condition)를 추정할 수 있다. 시간 패턴을 통해 휠 슬립, 액추에이터 열화(Actuator Degradation), 진동(Oscillation), 비정상 마찰, 적재물 변화를 감지할 수 있다.

조작 작업에서는 힘과 토크 이력이 접촉 상태를 보여줄 수 있다. 모델은 자유 공간 이동, 초기 접촉, 안정적인 파지, 미끄러짐, 걸림(Obstruction), 과부하를 구분할 수 있다. 선택적 메모리는 단순 평균화 과정에서 사라질 수 있는 짧은 접촉 전환을 보존할 수 있다.

자율이동로봇에서는 맘바 기반 시간 모델이 오도메트리, IMU, 위치 추정 잔차(Localization Residual), 휠 전류, LiDAR 특징, 내비게이션 명령을 처리할 수 있다. 정상 주행, 경사로, 휠 슬립, 충돌, 이동 차단, 위치 추정 드리프트와 관련된 패턴을 학습할 수 있다.

그러나 충분한 검증 없이 기존의 위치 추정이나 제어 알고리즘을 대체해서는 안 된다. 대신 학습 기반 잔차 추정(Learned Residual Estimation), 이상 탐지, 문맥 메모리(Context Memory), 상위 행동 예측에 활용할 수 있다.

상태공간 모델은 스트리밍 이상 탐지(Streaming Anomaly Detection)에 자연스럽게 적합하다. 시스템은 예상되는 시간적 행동을 상태로 유지하고 그로부터 벗어나는 변화를 식별한다. 상태가 최근 동역학을 요약하기 때문에 하나의 샘플에서는 보이지 않는 이상도 감지할 수 있다.

산업용 검사 로봇에서는 비정상 진동, 불안정한 센서 위치, 반복되는 도킹 오류(Docking Error), 카메라 가림, 비정상 모터 부하, 일관되지 않은 검사 소요 시간 등을 탐지할 수 있다. 이상 감지는 감속, 재보정(Recalibration), 유지보수, 인간 검토를 유발할 수 있다.

맘바는 행동 생성에도 사용할 수 있다. 모델은 관측과 목표 시퀀스를 입력받아 미래 행동을 생성할 수 있다. 아키텍처가 인과적(Causal)이므로 각 행동은 이전 관측과 행동에서 축적된 상태에 의존할 수 있다.

행동은 한 단계씩 생성할 수 있지만, 이는 지연 시간과 오류 누적을 발생시킬 수 있다. 행동 청킹(Action Chunking)을 사용하면 여러 미래 제어 단계를 함께 출력할 수 있다. 맘바 백본은 긴 과거 문맥을 인코딩하고, 행동 헤드(Action Head)는 지역 궤적(Local Trajectory)을 예측할 수 있다.

고정 크기의 순환 상태는 무한히 증가하는 어텐션 캐시가 필요하지 않기 때문에 장기 임무의 행동 생성에 적합하다. 모델은 내부 상태를 계속 갱신하면서 반복적으로 행동 청크를 생성할 수 있다.

그러나 순환 기반 행동 생성만으로는 매우 오래전에 발생한 특정 이벤트를 정확히 검색하기 어려울 수 있다. 로봇이 정확한 명령, 객체 식별자(Object Identifier), 이전 관측을 다시 기억해야 한다면 압축 상태만으로는 부족할 수 있다.

외부 메모리(External Memory)와 검색(Retrieval)이 이러한 한계를 보완할 수 있다. 맘바 상태는 연속적인 시간 동역학을 처리하고, 구조화된 메모리는 개별 이벤트, 임무 기록, 지도, 객체 ID, 작업자 명령을 저장한다. 필요할 때 모델이 관련 정보를 검색할 수 있다.

이러한 하이브리드 설계는 암묵적 메모리(Implicit Memory)와 명시적 메모리(Explicit Memory)의 차이를 반영한다. 상태공간 모델은 암묵적인 시간 문맥을 제공한다. 데이터베이스, 의미 지도, 임무 상태 머신(Mission-State Machine), 벡터 저장소(Vector Store)는 명시적인 장기 기억을 제공한다.

트랜스포머와 맘바를 항상 서로 배타적인 구조로 볼 필요는 없다. 직접적인 토큰 간 상호작용이 중요한 곳에는 어텐션을 사용하고, 장기 스트리밍 효율성이 더 중요한 곳에는 상태공간 계층을 사용하는 하이브리드 아키텍처를 구성할 수 있다.

예를 들어 비전-언어 모듈은 현재 시각 객체와 명령을 정렬하기 위해 트랜스포머 어텐션을 사용할 수 있다. 맘바 모듈은 이러한 객체와 로봇 상태를 시간에 따라 추적할 수 있다. 행동 디코더는 구조화된 메모리에서 작업 관련 세부사항을 검색하기 위해 교차 어텐션(Cross-Attention)을 사용할 수 있다.

또 다른 하이브리드 설계는 어텐션 계층과 상태공간 계층을 교대로 배치하는 것이다. 어텐션은 지역 문맥 안에서 유연한 전역 상호작용을 지원하고, 맘바는 더 긴 시간에 걸쳐 정보를 압축한다. 이를 통해 정확한 검색과 효율적인 시간 모델링을 동시에 유지할 수 있다.

최적의 아키텍처는 작업에 따라 달라진다. 풍부한 시각 관계를 포함하는 짧은 조작 에피소드에서는 어텐션이 더 유리할 수 있다. 장시간 내비게이션 임무와 텔레메트리 스트림에서는 상태공간 메모리가 더 유리할 수 있다. 범용 피지컬 AI 시스템은 두 구조를 모두 필요로 할 가능성이 높다.

맘바 기반 로봇 모델을 학습하려면 세심하게 구성된 시퀀스가 필요하다. 데이터는 동기화된 관측, 로봇 상태, 행동, 타임스탬프, 작업 목표, 결과를 포함해야 한다. 상태 전이를 직접 학습하기 때문에 시간적 오류는 특히 치명적일 수 있다.

행동이 지연된 관측과 연결되면 모델은 잘못된 동역학을 추론할 수 있다. 센서 시계가 드리프트하면 힘, 움직임, 시각 변화 사이의 거짓 관계를 학습할 수 있다. 따라서 하드웨어 타임스탬프와 보정된 데이터 파이프라인이 필수적이다.

시퀀스 분할(Sequence Segmentation)도 중요한 문제이다. 로봇 데이터셋은 일반적으로 에피소드(Episode) 단위로 나뉘지만, 이 경계가 항상 의미 있는 상태 초기화와 일치하는 것은 아니다. 임의의 파일 경계마다 은닉 상태를 초기화하면 유용한 문맥이 사라질 수 있다.

반대로 관련 없는 에피소드 사이에 상태를 유지하면 학습이 오염될 수 있다. 데이터셋은 작업 시작, 실패, 중단, 충전, 작업자 인계(Operator Takeover), 환경 변화를 구분해야 한다. 특수 경계 토큰(Boundary Token)이나 초기화 신호(Reset Signal)를 이용해 모델 상태를 제어할 수 있다.

학습 데이터에는 성공한 궤적과 실패한 궤적이 모두 포함되어야 한다. 이상적인 시연만 본 모델은 정상적인 진행은 학습할 수 있지만, 실행이 계획에서 벗어나는 상황은 인식하지 못할 수 있다. 실패와 복구 시퀀스는 문제가 발생했을 때 예상 동역학이 어떻게 변하는지를 가르친다.

행동 생성 모델을 학습할 때 교사 강제(Teacher Forcing)를 사용할 수 있다. 모델은 올바른 이전 행동을 입력받으며 이후 행동을 예측한다. 배포 단계에서는 모델 자신이 생성한 행동을 입력으로 사용하므로 노출 편향(Exposure Bias)이 발생한다.

온라인 데이터 수집(Online Data Collection), 교란 학습(Perturbation Training), 스케줄드 샘플링(Scheduled Sampling), 폐루프 미세조정(Closed-Loop Fine-Tuning)은 이러한 문제를 줄일 수 있다. 모델은 자신의 실수로 발생한 상태를 경험하고 복구 방법을 학습해야 한다.

로봇 데이터는 풍부하지만 라벨이 부족하기 때문에 자기지도 학습(Self-Supervised Learning) 목표가 유용하다. 맘바 모델은 미래 상태를 예측하고, 가려진 시간 구간(Masked Time Segment)을 복원하고, 시간 순서를 판별하고, 누락된 센서 채널을 추정하거나, 동기화된 데이터와 어긋난 데이터를 구분하도록 학습할 수 있다.

마스크드 시간 모델링(Masked Temporal Modeling)은 주변 문맥을 이용해 누락된 관측을 추론하도록 한다. 이를 통해 움직임, 작업 진행, 센서 관계에 대한 표현을 학습할 수 있다. 미래 상태 예측(Future-State Prediction)은 동역학에 대한 이해를 촉진한다.

대조학습 목표(Contrastive Objective)는 시간에 따라 서로 다른 모달리티를 정렬할 수 있다. 시각 관측과 이에 대응하는 고유감각 상태는 관련 없는 데이터 쌍보다 가까운 표현을 가져야 한다. 행동 조건부 학습 목표는 제어 명령이 미래 관측에 어떤 영향을 주는지를 학습시킬 수 있다.

월드 모델 학습(World-Model Training)도 중요한 활용 분야이다. 맘바는 과거 문맥을 인코딩하고 후보 행동에 따른 미래 잠재 상태를 예측할 수 있다. 효율적인 순환 구조는 긴 롤아웃(Rollout)을 가능하게 하지만 예측 오류는 여전히 누적될 수 있다.

전체 이미지를 복원하는 대신 제어에 필요한 특징을 예측하도록 학습할 수 있다. 객체 자세, 자유 공간, 접촉 상태, 위치 추정 신뢰도, 작업 진행 상태가 사실적인 이미지 생성보다 더 유용할 수 있다.

불확실성 추정(Uncertainty Estimation)은 여전히 필요하다. 결정론적 순환 상태는 입력이 익숙하지 않은 상황에서도 확신에 찬 예측을 생성할 수 있다. 앙상블(Ensemble), 확률적 출력 헤드(Probabilistic Output Head), 잠재 분포(Latent Distribution), 다중 궤적 샘플을 이용해 불확실성을 표현할 수 있다.

모델은 압축된 상태가 더 이상 환경을 신뢰성 있게 표현하지 못하는 시점을 인식해야 한다. 센서 고장, 갑작스러운 환경 변화, 장시간 중단, 이전에 보지 못한 동역학은 상태를 무효화할 수 있다.

따라서 상태 초기화와 복구 메커니즘이 중요하다. 위치 추정 복구, 공구 교체, 로봇 재시작, 임무 전환 이후 상태의 일부를 초기화할 수 있다. 반면 작업 명령이나 장기 임무 메모리와 같은 일부 정보는 유지되어야 한다.

하나의 구분되지 않은 상태는 선택적 초기화를 어렵게 만든다. 구조화된 아키텍처는 인식, 체화, 작업 진행, 제어에 각각 별도의 상태를 유지할 수 있다. 각 상태는 서로 다른 이벤트에 따라 초기화되거나 갱신될 수 있다.

은닉 상태가 많은 학습 차원에 분산되어 있기 때문에 해석 가능성(Interpretability)은 어렵다. 엔지니어는 상태가 시간 정보를 포함한다는 사실은 알 수 있지만, 어떤 차원이 특정 물리 개념과 대응하는지는 알기 어렵다.

보조 예측 헤드(Auxiliary Prediction Head)는 관측 가능성(Observability)을 향상시킬 수 있다. 상태를 이용해 객체 상태, 접촉 상태, 작업 단계, 위치 추정 신뢰도, 실패 확률을 예측할 수 있다. 이러한 출력은 모델이 어떤 정보를 유지하고 있는지 보여주는 단서가 된다.

프로빙 실험(Probing Experiment)은 상태가 과거 이벤트에 대한 정보를 포함하는지를 시험할 수 있다. 예를 들어 모델이 오랜 관측 이후에도 임무 명령, 지나간 검사 지점, 일시적 장애물을 기억하는지를 측정할 수 있다.

반사실적 시험(Counterfactual Testing)은 이전 입력 하나를 변경했을 때 이후 예측이 어떻게 달라지는지를 확인할 수 있다. 이를 통해 모델이 피상적 상관관계가 아니라 물리적으로 관련된 이력에 의존하는지를 평가할 수 있다.

강건성 평가는 누락 샘플(Missing Sample), 불규칙 시간 간격, 센서 노이즈, 드리프트, 가림, 장시간 운용을 포함해야 한다. 짧고 깨끗한 에피소드에서는 잘 동작하는 모델도 수시간 동안 상태 오차가 누적되면 실패할 수 있다.

장기 상태 드리프트(Long-Horizon State Drift)는 중요한 문제이다. 작은 갱신 오류가 누적되어 내부 표현을 점진적으로 왜곡할 수 있다. 명시적 위치 추정, 디지털 트윈(Digital Twin), 알려진 랜드마크, 작업 상태 데이터베이스를 이용한 주기적 접지(Grounding)가 이러한 드리프트를 보정할 수 있다.

로봇은 이러한 초기화 기회를 자주 가질 수 있다. 도킹 스테이션(Docking Station), 검사 마커(Inspection Marker), 에이프릴태그(AprilTag), 지도화된 랜드마크, 알려진 자세(Known Pose), 작업자 확인이 기준점을 제공할 수 있다. 학습된 상태는 이러한 명시적 보정을 통합해야 한다.

안전 검증에서는 상태에 오래되거나 잘못된 정보가 포함될 가능성을 고려해야 한다. 현재 센서와 메모리가 충돌할 때 로봇은 메모리에만 의존해서는 안 된다. 최근의 높은 신뢰도 관측과 결정론적 안전 센서가 우선되어야 한다.

맘바 기반 모델은 계층형 안전 아키텍처 안에서 동작해야 한다. 문맥을 추정하고, 행동을 예측하고, 이상을 탐지할 수 있지만, 독립적인 시스템이 속도 제한, 충돌 회피, 힘 제한, 비상 정지, 기능 안전 요구사항을 강제해야 한다.

순환 상태 갱신은 각 새로운 토큰이나 시간 단계에 거의 일정한 계산량을 요구하므로 지연 시간의 예측 가능성(Latency Predictability)이 주요 배포 장점 가운데 하나이다. 이는 제한된 실행 시간을 요구하는 실시간 시스템에 유용하다.

그러나 전체 파이프라인은 여전히 비용이 큰 비전 인코더, 행동 디코더, 검색 시스템, 계획기를 포함할 수 있다. 평가는 상태공간 계층만이 아니라 전체 종단 간 지연 시간(End-to-End Latency)을 측정해야 한다.

하드웨어 최적화(Hardware Optimization)도 중요하다. 맘바의 실제 성능은 선택적 스캔(Selective Scan), 메모리 접근, 커널 융합(Kernel Fusion), 수치 정밀도의 효율적인 구현에 달려 있다. 이론적으로 선형인 모델도 대상 가속기를 충분히 활용하지 못하면 빠르지 않을 수 있다.

엣지 배포에서는 양자화(Quantization), 가지치기(Pruning), 상태 차원 축소, 지식 증류(Knowledge Distillation), 하드웨어별 커널을 사용할 수 있다. 데이터센터 GPU가 아니라 실제 로봇 컴퓨팅 플랫폼에서 모델을 평가해야 한다.

에너지 효율(Energy Efficiency)은 이동형 로봇에서 큰 장점이 될 수 있다. 압축된 시간 모델은 긴 어텐션 캐시를 유지하는 대형 트랜스포머보다 적은 메모리 이동과 전력으로 연속 텔레메트리를 처리할 수 있다.

실용적인 아키텍처는 가벼운 맘바 모델을 지속적으로 실행하고, 복잡한 의미 추론이 필요할 때만 대형 멀티모달 트랜스포머를 호출할 수 있다. 상태공간 모델은 일상 운용을 감시하고 상위 지능이 필요한 시점을 탐지한다.

이러한 적응형 아키텍처(Adaptive Architecture)는 평균 계산 비용을 줄인다. 빈 복도, 안정된 조작, 정기 순찰은 경량 모델이 처리할 수 있다. 새로운 객체, 상충되는 지시, 예상하지 못한 실패는 더 높은 능력의 추론 시스템을 호출할 수 있다.

교차 체화 학습(Cross-Embodiment Learning)도 잠재적인 활용 분야이다. 공유 상태공간 백본은 서로 다른 로봇의 정규화된 이력을 처리하고, 체화 임베딩(Embodiment Embedding)은 형태, 센서, 액추에이터, 행동 공간을 설명할 수 있다.

로봇별 입력 어댑터(Input Adapter)는 원시 데이터를 공통 잠재 표현으로 변환한다. 로봇별 출력 헤드는 공유 상태를 적절한 행동으로 변환한다. 시간 백본은 접근, 정지, 접촉, 막힘, 작업 완료와 같은 공통 패턴을 학습한다.

그러나 동역학 차이는 여전히 크다. 동일한 관측 변화도 대형 AMR, 휴머노이드, 매니퓰레이터에서 서로 다른 의미를 가질 수 있다. 체화 조건부 전이(Embodiment-Conditioned Transition)를 사용하면 상태 갱신이 현재 로봇 몸체에 따라 달라질 수 있다.

공구와 적재물 변화도 모델 조건으로 포함해야 한다. 무거운 검사 장비를 탑재한 로봇은 빈 플랫폼과 다른 동역학을 가진다. 긴 프로브를 장착한 매니퓰레이터는 충돌 영역과 접촉 응답이 달라진다.

모델 상태는 적재물, 공구 식별자, 배터리 상태, 액추에이터 온도, 상태 진단(Health Diagnostic)을 포함할 수 있다. 이러한 변수는 시간 시퀀스 모델링과 동적 체화 인식(Dynamic Embodiment Awareness)을 연결한다.

플릿 학습(Fleet Learning)은 대규모 시퀀스 데이터를 제공할 수 있다. 여러 로봇의 로그에는 반복 임무, 실패, 복구, 환경 변화, 유지보수 이벤트가 포함된다. 맘바 기반 모델은 이러한 장기 운용 패턴을 학습하는 데 적합할 수 있다.

그러나 플릿 데이터는 이질적이다. 소프트웨어 버전, 센서 구성, 보정, 제어 설정, 운용 조건이 서로 다르다. 모델이 시스템 변경과 환경 행동을 혼동하지 않도록 메타데이터를 보존해야 한다.

모델 업데이트에는 회귀 시험(Regression Testing)이 필요하다. 새로운 맘바 모델이 장기 예측은 향상시키지만 빠른 충돌 감지나 행동 반응성을 저하시킬 수 있다. 평가는 여러 시간 척도를 모두 포함해야 한다.

상태공간 로봇 모델의 벤치마크는 시퀀스 길이, 스트리밍 지연 시간, 메모리 사용량, 시간 예측, 이벤트 보존(Event Retention), 복구, 폐루프 작업 성능을 포함해야 한다. 오프라인 손실만 비교해서는 충분하지 않다.

이벤트 보존 시험은 모델이 반복적인 일상 데이터가 오랫동안 이어진 이후에도 드물지만 중요한 정보를 기억하는지 평가할 수 있다. 예를 들어 임무 초기에 특정 구역에 진입해서는 안 된다는 지시가 주어질 수 있다.

선택적 망각(Selective Forgetting)도 평가해야 한다. 모델은 환경이 바뀌면 오래된 장애물 정보를 버려야 하지만 지속적인 임무 제약은 유지해야 한다. 모든 것을 기억하는 것이 항상 바람직한 것은 아니다.

장기 컨텍스트 평가는 짧은 패턴을 단순히 반복하는 대신 실제로 긴 시퀀스를 사용해야 한다. 모델은 작업, 환경, 센서 상태, 행동 결과의 실제 변화를 처리해야 한다.

시간 예측 오류가 미래 관측을 변화시키기 때문에 폐루프 시험이 필요하다. 로그 데이터에서는 우수한 모델도 자신의 행동이 시퀀스에 영향을 미치는 환경에서는 다르게 동작할 수 있다.

시뮬레이션은 확장 가능한 장시간 시험 환경을 제공한다. 로봇은 변화하는 날씨, 교통, 적재물, 센서 장애, 임무 이벤트 아래에서 가상으로 수시간 운용될 수 있다. 하드웨어 위험 없이 은닉 상태의 안정성과 복구를 평가할 수 있다.

그러나 액추에이터 특성, 센서 노이즈, 통신 지연, 물리적 접촉은 시뮬레이션과 실제 환경에서 다르기 때문에 실세계 검증(Real-World Validation)이 필수적이다. 상태공간 모델은 이러한 시간 동역학의 체계적인 차이에 특히 민감할 수 있다.

따라서 피지컬 AI에서 맘바의 역할은 트랜스포머를 완전히 대체하는 범용 해법이 아니라 더 큰 아키텍처의 일부로 이해해야 한다. 맘바의 강점은 효율적인 시퀀스 처리, 스트리밍 운용, 압축된 시간 메모리, 입력 내용에 따른 상태 갱신이다.

반면 압축된 과거 이력에만 접근한다는 점, 먼 과거의 정확한 정보를 검색하기 어렵다는 점, 상태 드리프트가 발생할 수 있다는 점, 세심한 시퀀스 설계가 필요하다는 점은 한계이다. 트랜스포머는 유연한 전역 상호작용을 제공하고, 명시적 메모리는 신뢰성 있는 사건 저장을 제공한다. 전통적인 제어 및 상태 추정은 물리적 보장을 제공한다.

강력한 로봇 아키텍처는 이러한 방법들을 함께 결합할 수 있다. 비전 및 언어 트랜스포머는 현재의 의미적 문맥을 해석한다. 맘바는 장기적인 시간 변화를 추적한다. 데이터베이스와 임무 상태 머신은 명시적 사실을 저장한다. 월드 모델은 결과를 예측한다. 제어기는 안정적인 움직임을 실행한다. 안전 시스템은 물리적 권한을 제한한다.

피지컬 AI 파운데이션 모델에서 이러한 결합은 컨텍스트와 메모리 비용을 무한히 증가시키지 않으면서도 지속적인 운용을 지원할 수 있다. 로봇은 필요할 때 정확한 과거 정보를 검색하면서 최근 동역학에 대한 압축된 이해를 유지할 수 있다.

상태공간 모델은 시퀀스 학습을 더 물리적으로 의미 있는 관점에서 이해하도록 한다. 로봇 운용을 서로 독립된 거대한 토큰 집합으로 취급하는 대신, 로봇과 환경을 계속 변화하는 잠재 상태를 가진 동적 시스템으로 모델링한다.

이러한 관점은 제어 이론(Control Theory), 필터링(Filtering), 시스템 식별, 월드 모델링과 자연스럽게 연결된다. 현대 파운데이션 모델을 전통적인 로보틱스와 분리하기보다 학습된 표현과 알려진 물리 구조를 함께 결합할 기회를 제공한다.

맘바는 메모리 동역학이 현재 입력에 따라 달라지도록 하여 이러한 관점을 확장한다. 중요한 이벤트는 상태에 강하게 영향을 줄 수 있고, 반복적인 관측은 걸러낼 수 있다. 이러한 선택성은 길고 반복적이며 이벤트 중심적인 로봇 데이터에 잘 맞는다.

로보틱스에서 맘바의 장기적 가치는 안전에 중요한 정보를 보존하고, 서로 다른 체화에 적응하며, 멀티모달 입력을 통합하고, 실제 운용 시간 범위에서 안정성을 유지할 수 있는지에 달려 있다. 이러한 질문은 언어 또는 비전 벤치마크만으로 결론을 내릴 수 없으며 체계적인 검증이 필요하다.

순차 로봇 데이터에서 상태공간 모델은 확장 가능한 시간 지능(Scalable Temporal Intelligence)으로 가는 경로를 제공한다. 긴 센서 이력을 처리하고, 스트리밍 메모리를 유지하며, 동적 변화를 감지하고, 미래 상태를 예측하며, 통제된 계산 비용으로 행동 생성을 지원할 수 있다.

가장 큰 기여는 모든 어텐션 계층을 대체하는 것이 아닐 수 있다. 로봇 파운데이션 모델이 지속적이고 효율적이며 빠르게 반응하면서 단기 물리 동역학과 장기 임무 문맥을 함께 처리할 수 있도록 만드는 데 있을 수 있다.

성공적인 피지컬 AI 시스템은 시간적 연속성이 중요한 곳에는 상태공간 메모리를 사용하고, 유연한 관계 추론이 필요한 곳에는 어텐션을 사용하며, 정확한 기억이 필요한 곳에는 명시적 메모리를 사용하고, 물리적 보장이 필요한 곳에는 결정론적 제어를 사용할 것이다. 이러한 계층형 아키텍처 안에서 맘바는 연속적인 로봇 경험을 변화하는 물리 세계에 대한 압축되고, 적응 가능하며, 행동으로 연결할 수 있는 표현으로 변환하는 핵심 시간 백본(Central Temporal Backbone)의 역할을 수행할 수 있다.

## 2.7 Cross Attention for Multi Modal Robot Input [w/Code]

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

교차 어텐션(Cross-Attention)은 피지컬 AI(Physical AI) 파운데이션 모델(Foundation Model)에서 서로 다른 종류의 정보를 통합하기 위한 가장 중요한 아키텍처 메커니즘 가운데 하나이다. 로봇은 하나의 통일된 형식으로만 정보를 받지 않는다. RGB 이미지, 깊이 영상(Depth Map), LiDAR 스캔, 관절 위치(Joint Position), 모터 전류(Motor Current), 힘 측정값(Force Measurement), 촉각 신호(Tactile Signal), 음성(Audio), 자연어 명령(Language Instruction), 내비게이션 목표(Navigation Goal), 디지털 트윈(Digital Twin) 데이터, 이전 행동(Previous Action) 등을 동시에 처리해야 한다. 이러한 입력들은 차원(Dimensionality), 시간 주기(Temporal Frequency), 좌표계(Coordinate System), 불확실성(Uncertainty), 의미(Semantic Meaning)가 모두 다르다. 교차 어텐션은 하나의 표현(Representation)이 다른 표현으로부터 필요한 정보를 선택적으로 검색하도록 함으로써, 모든 신호를 하나의 구분되지 않는 시퀀스로 강제로 합치지 않고도 다양한 모달리티(Modality)를 통합할 수 있게 한다.

교차 어텐션의 기본 개념은 자기 어텐션(Self-Attention)과 유사하지만, 쿼리(Query), 키(Key), 값(Value) 벡터의 출처가 서로 다르다는 점에서 차이가 있다. 자기 어텐션에서는 세 벡터 모두 동일한 시퀀스에서 생성된다. 각 토큰(Token)은 동일한 시퀀스 안의 다른 토큰과 비교하여 관련 정보를 집계한다. 반면 교차 어텐션에서는 쿼리 벡터는 하나의 모달리티 또는 처리 스트림에서 생성되고, 키와 값 벡터는 다른 모달리티에서 생성된다. 이를 통해 하나의 표현이 다른 표현으로부터 어떤 정보가 필요한지를 질의할 수 있다.

예를 들어 "왼쪽 위의 밸브를 검사하라(inspect the upper-left valve)"라는 언어 토큰(Language Token)이 쿼리 역할을 할 수 있다. 이미지 영역을 나타내는 시각 토큰(Visual Token)은 키와 값 역할을 수행한다. 교차 어텐션을 통해 언어 표현은 지정된 밸브가 존재할 가능성이 높은 이미지 영역에 더 높은 가중치를 부여할 수 있다. 결과적으로 출력은 현재 장면에서 상징적 명령(Symbolic Instruction)과 실제 물리적 목표를 연결하는 언어 조건부 시각 표현(Language-Conditioned Visual Representation)이 된다.

동일한 메커니즘은 반대 방향으로도 동작할 수 있다. 시각 객체 토큰(Visual Object Token)이 언어 토큰에 질의하여 현재 작업과 관련된 특징만 선택할 수 있다. 로봇은 수많은 객체를 인식할 수 있지만 현재 명령과 관련된 객체는 일부뿐이다. 교차 어텐션은 모든 객체를 동일하게 취급하지 않고 작업 문맥(Task Context)에 따라 시각 표현을 선택적으로 필터링한다.

이러한 선택적 상호작용은 멀티모달 접지(Multimodal Grounding)의 핵심이다. 로봇은 언어와 영상을 단순히 병렬 처리해서는 안 된다. 단어를 실제 객체와 연결하고, 행동을 몸 상태와 연결하며, 힘 신호를 접촉 지점과 연결하고, 지도 위치를 현재 관측과 연결해야 한다. 교차 어텐션은 이러한 연결을 학습 가능한 방식으로 구축하는 메커니즘을 제공한다.

간단한 멀티모달 아키텍처는 입력 종류마다 별도의 인코더(Encoder)를 가질 수 있다. 비전 인코더(Vision Encoder)는 이미지를 패치 토큰(Patch Token) 또는 객체 토큰으로 변환한다. 언어 인코더(Language Encoder)는 명령을 텍스트 임베딩(Text Embedding)으로 변환한다. 고유감각 인코더(Proprioceptive Encoder)는 관절 위치, 속도, 토크(Torque), 배터리 상태, 액추에이터 상태를 표현한다. 기하 인코더(Geometric Encoder)는 LiDAR 또는 포인트 클라우드(Point Cloud)를 처리한다. 시간 인코더(Temporal Encoder)는 최근 행동과 관측을 요약한다.

교차 어텐션 계층(Cross-Attention Layer)은 이러한 인코딩된 스트림(Stream)이 서로 정보를 교환하도록 한다. 모델은 모든 원시 입력(Raw Input)을 처음부터 하나로 합칠 필요가 없다. 각 모달리티는 자신의 구조와 귀납적 편향(Inductive Bias)을 유지한 채 먼저 특징을 추출할 수 있으며, 이후에 필요한 수준에서 융합(Fusion)이 이루어진다.

이러한 분리는 멀티모달 로봇 데이터가 매우 이질적(Heterogeneous)이기 때문에 중요하다. 이미지는 밀집된 공간 정보(Dense Spatial Information)를 포함한다. 언어는 상징적이고 순차적인 정보를 포함한다. 고유감각은 저차원이지만 매우 빠르게 변화한다. 힘과 촉각은 이벤트 중심(Event-Driven)이며 높은 시간 해상도를 요구한다. LiDAR는 희소한 기하 구조를 제공한다. 지도(Map)와 디지털 트윈은 대부분 정적인 정보를 유지한다.

이 모든 입력을 동일한 종류의 토큰으로 취급하면 학습 효율이 떨어지고 각 모달리티 고유의 정보가 약해질 수 있다. 교차 어텐션은 이러한 전문화된 표현들 사이에서 제어된 융합을 가능하게 한다.

교차 어텐션의 방향(Direction)도 중요하다. 한 가지 설계에서는 소수의 작업 쿼리(Task Query)나 행동 쿼리(Action Query)가 대규모 센서 토큰 집합에 주의를 기울인다. 출력 토큰 수가 적기 때문에 효율적이다. 이러한 쿼리는 미래 행동, 객체 슬롯(Object Slot), 궤적 지점(Trajectory Point), 잠재 제어 상태(Latent Control State)를 나타낼 수 있다.

예를 들어 행동 디코더(Action Decoder)는 행동 청크(Action Chunk)의 각 미래 제어 단계마다 하나의 쿼리를 가질 수 있다. 이 쿼리는 시각, 언어, 로봇 상태 토큰에 주의를 기울인다. 각 행동 쿼리는 해당 움직임을 생성하기 위해 필요한 정보를 선택적으로 검색한다. 초기 쿼리는 현재 목표와 로봇 자세에 집중하고, 이후 쿼리는 목표 위치나 예상 접촉 상태에 집중할 수 있다.

반대로 센서 토큰이 작업 또는 언어 토큰에 질의하는 구조도 가능하다. 이는 인식을 임무에 맞게 조건화(Conditioning)한다. 비전 인코더는 수백 개의 이미지 토큰을 생성할 수 있지만, 각 토큰은 간결한 명령 표현에 주의를 기울일 수 있다. 결과적으로 작업과 관련된 객체는 강조되고 배경은 억제된다.

양방향 교차 어텐션(Bidirectional Cross-Attention)은 두 방향을 모두 사용한다. 언어가 비전에 주의를 기울이고, 비전도 언어에 주의를 기울인다. 반복적인 상호작용을 통해 두 모달리티는 서로를 점진적으로 개선한다. 언어 표현은 시각적으로 접지되고, 시각 표현은 작업 의존적(Task-Aware)이 된다.

이러한 반복적 정제(Iterative Refinement)는 초기 명령이 모호하거나 유사한 객체가 여러 개 존재할 때 특히 유용하다. 모델은 먼저 후보 객체를 찾고, 이후 공간 언어(Spatial Language)와 작업 문맥을 이용하여 목표를 점진적으로 좁혀갈 수 있다.

교차 어텐션은 비전과 고유감각의 결합에도 중요하다. 로봇은 자신의 몸 상태와 관련하여 시각 정보를 해석해야 한다. 어떤 물체는 보일 수 있지만 도달할 수는 없을 수 있다. 세계 좌표계(World Coordinate)에서는 가능한 자세라도 현재 관절 구성에서는 위험할 수 있다.

시각 토큰은 로봇 상태 토큰에 주의를 기울여 체화 인식(Embodiment Awareness)을 획득할 수 있다. 모델은 화면 가장자리에 있는 물체가 이동 플랫폼(Base)이 먼저 움직여야만 접근 가능하다는 점이나 특정 파지 방향이 현재 손목 자세와 맞지 않는다는 점을 학습할 수 있다.

반대로 로봇 상태 쿼리는 시각 토큰에 주의를 기울일 수 있다. 그리퍼 상태 토큰은 현재 손가락 사이에 있는 물체 정보를 검색할 수 있다. 이동 플랫폼 상태 토큰은 장애물과 경로 경계를 검색할 수 있다. 이는 현재의 물리적 자세를 고려한 인식을 가능하게 한다.

비전과 기하 정보의 결합도 중요한 활용 분야이다. RGB 이미지는 의미와 외관 정보를 제공하고, 깊이, LiDAR, 포인트 클라우드는 거리와 형상을 제공한다. 로봇은 두 가지 모두를 필요로 한다. 이미지는 물체가 상자(Box)임을 알려줄 수 있지만 깊이 데이터는 거리와 실제 형상을 알려준다.

이미지 토큰은 기하 토큰에 질의하여 깊이와 공간 정보를 얻을 수 있다. 기하 토큰은 이미지 특징에 질의하여 의미 라벨이나 질감(Texture)을 획득할 수 있다. 이러한 융합 표현은 물체가 무엇인지, 어디에 있는지, 어떤 형상을 가지는지를 동시에 연결한다.

이를 위해서는 정확한 센서 보정(Calibration)이 필수적이다. 이미지와 LiDAR가 서로 다른 좌표계를 사용한다면 교차 어텐션은 잘못된 대응 관계를 학습할 수 있다. 위치 인코딩(Positional Encoding)은 카메라 좌표, 깊이, 세계 좌표, 센서 ID, 타임스탬프를 포함해야 한다.

센서 좌표계 사이의 명시적 투영(Explicit Projection)은 융합을 더욱 향상시킬 수 있다. LiDAR 점은 이미지 평면으로 투영될 수 있고, 이미지 특징은 3차원 광선(Ray)과 연결될 수 있다. 이후 교차 어텐션은 기하학적으로 의미 있는 대응 후보만을 대상으로 동작하게 된다.

명시적 기하 정보와 학습 기반 어텐션을 결합한 이러한 구조는 완전히 암묵적인 융합보다 일반적으로 더 신뢰성이 높다. 모델은 알려진 센서 보정을 활용하면서도 어떤 특징이 최종 표현에 얼마나 강하게 영향을 줄지를 학습할 수 있다.

힘과 촉각 데이터는 또 다른 형태의 멀티모달 융합을 요구한다. 접촉 전에는 비전과 기하 정보가 중요하지만, 접촉 이후에는 힘, 토크, 모터 전류, 촉각 측정이 더 중요한 정보를 제공한다. 교차 어텐션은 상호작용 단계에 따라 주의를 동적으로 변경할 수 있다.

행동 또는 접촉 쿼리는 처음에는 목표 표면을 설명하는 시각 특징에 집중할 수 있다. 그리퍼가 접근하면서 거리와 고유감각에 더 집중하게 되고, 실제 접촉 시점에는 힘과 촉각 토큰이 지배적인 역할을 한다. 이러한 단계별 융합은 조작(Manipulation)에 매우 중요하다.

모델은 서로 다른 모달리티가 서로 다른 시점에서 신뢰성이 높아진다는 사실을 학습해야 한다. 파지 과정에서는 시야가 가려져 비전이 약해질 수 있지만 촉각은 더욱 중요해진다. 힘 센서는 접촉 여부는 알려주지만 물체의 의미 정보는 제공하지 못한다. 교차 어텐션은 이러한 상호 보완적인 특성을 통합한다.

센서 간 교차 어텐션에서는 시간 정렬(Temporal Alignment)이 매우 중요하다. 카메라 프레임, 힘 샘플, 관절 상태, 행동은 서로 다른 주기로 기록될 수 있다. 현재의 힘 측정이 몇 초 전에 촬영된 이미지에 동일한 수준으로 주의를 기울여서는 안 된다.

타임스탬프 임베딩(Timestamp Embedding), 상대 시간 인코딩(Relative-Time Encoding), 최신성(Freshness) 정보를 포함하면 이를 해결할 수 있다. 각 토큰은 언제 측정되었는지와 얼마나 오래된 데이터인지를 포함한다. 이후 어텐션 점수는 의미적 관련성과 시간적 일관성을 함께 반영할 수 있다.

비동기 교차 어텐션(Asynchronous Cross-Attention)은 서로 다른 주기로 갱신되는 센서를 지원할 수 있다. 하나의 언어 명령은 임무 전체 동안 유지될 수 있으며, 매 제어 주기마다 다시 계산할 필요가 없다. 정적 지도 토큰은 캐시(Cache)될 수 있다. 반면 동적인 이미지와 상태 토큰은 자주 갱신된다. 행동 디코더는 각 스트림의 가장 최신 표현에 주의를 기울인다.

그러나 오래된 정보는 신중히 다루어야 한다. 지도는 복도가 비어 있다고 표시하지만 현재 LiDAR는 장애물을 감지할 수 있다. 모델은 항상 최신의 신뢰할 수 있는 센서 정보를 우선시해야 한다. 토큰이 최신성과 신뢰도를 포함하면 교차 어텐션은 이를 학습할 수 있다.

신뢰도 기반 교차 어텐션(Confidence-Aware Cross-Attention)은 중요한 확장이다. 센서 토큰은 불확실성 추정, 데이터 품질 점수, 센서 건강 상태를 포함할 수 있다. 흐릿한 카메라 영상, 낮은 신뢰도의 위치 추정, 잡음이 큰 힘 측정은 의사결정에 적게 기여해야 한다.

그러나 어텐션 점수 자체가 항상 신뢰도를 의미하는 것은 아니다. 모델은 시각적으로 눈에 띄지만 신뢰성이 낮은 특징에 높은 어텐션을 줄 수도 있다. 따라서 품질 임베딩과 센서 검증 모듈이 함께 필요하다.

교차 어텐션은 지도, 데이터베이스, 디지털 트윈으로부터 구조화된 지식도 통합할 수 있다. 산업 설비를 관찰하는 로봇은 부품 식별자, 기준 자세, 유지보수 이력, 검사 절차, 제한 구역을 포함하는 디지털 트윈 표현에 질의할 수 있다.

실시간 시각 토큰은 현재 상태를 제공하고, 디지털 트윈 토큰은 사전 지식을 제공한다. 교차 어텐션은 실제 관측된 객체와 예상되는 객체를 정렬하여 의미 기반 위치 추정, 검사 계획, 이상 탐지를 지원한다.

그러나 저장된 지식과 실시간 데이터의 관계는 항상 조건부여야 한다. 디지털 트윈은 오래되었을 수 있고, 설비가 이동되거나 변경되었을 수도 있다. 모델은 현재 관측을 오래된 기준 정보에 강제로 맞추어서는 안 된다.

교차 어텐션은 두 정보원의 불일치를 발견할 수 있다. 실시간 센서와 디지털 트윈 사이에 큰 차이가 존재한다면 환경 변화, 보정 오류, 데이터 오류를 의미할 수 있다. 이러한 경우에는 자동 융합이 아니라 검증이 수행되어야 한다.

언어와 지도 사이의 교차 어텐션도 내비게이션에서 매우 중요하다. "압축기실 근처 두 번째 검사 지점으로 이동하라"와 같은 명령은 의미 지도(Semantic Map)를 이해해야 한다. 언어 쿼리는 지도 노드, 방 이름, 랜드마크, 임무 메타데이터를 검색할 수 있다.

이후 접지된 목표는 내비게이션 플래너(Navigation Planner)로 전달된다. 파운데이션 모델이 바퀴 제어 명령을 직접 생성할 필요는 없다. 교차 어텐션은 사람의 의도를 결정론적 경로 계획기가 사용할 수 있는 구조화된 목표로 변환한다.

임무 상태(Mission State)와 인식의 결합도 중요하다. 동일한 환경이라도 작업 단계에 따라 중요 정보가 달라진다. 이동 중에는 자유 공간과 장애물이 중요하지만, 도킹 단계에서는 마커와 정렬 특징이 중요하다. 검사 단계에서는 검사 대상 표면과 영상 품질이 핵심이다.

임무 상태 토큰은 센서 특징에 질의하여 작업 관련 정보만 선택할 수 있다. 반대로 센서 토큰도 임무 상태에 주의를 기울여 작업 단계 인식(Task Phase Awareness)을 갖게 된다. 이를 통해 인식이 문맥과 무관한 독립 과정이 되는 것을 방지한다.

계층형 교차 어텐션(Hierarchical Cross-Attention)은 복잡한 로봇에서 특히 유용하다. 상위 작업 모델은 의미 토큰에 집중하고, 하위 행동 모델은 기하 정보와 상태 특징에 집중한다. 서로 다른 계층은 서로 다른 시간 및 공간 해상도로 동작한다.

임무 수준에서는 방 식별자, 작업 진행 상태, 배터리 상태, 작업자 명령이 필요하다. 궤적 수준에서는 지역 장애물, 현재 자세, 목표 기하 정보가 필요하다. 접촉 수준에서는 힘, 촉각, 관절 피드백이 필요하다.

모든 정보를 하나의 거대한 어텐션 계층으로 처리하는 것은 비효율적이며 검증도 어렵다. 계층형 융합은 각 제어 계층이 자신의 역할에 맞는 정보만 받을 수 있도록 한다.

잠재 쿼리 아키텍처(Latent Query Architecture)는 또 다른 효율적인 방법이다. 모든 센서 토큰이 서로 주의를 기울이도록 하지 않고, 소수의 학습된 잠재 쿼리가 모든 모달리티로부터 정보를 수집한다. 이 잠재 토큰은 압축된 멀티모달 표현이 된다.

이 접근은 수천 개의 이미지 또는 포인트 클라우드 토큰을 처리해야 하는 경우 특히 유용하다. 잠재 쿼리는 정보 병목(Information Bottleneck) 역할을 수행하며 작업 객체, 공간 영역, 행동 문맥, 장면 요약을 표현할 수 있다.

잠재 쿼리의 수와 의미는 고정될 수도 있고 학습될 수도 있다. 일부는 객체에 특화되고, 일부는 로봇 상태에, 일부는 환경 위험에 특화될 수 있다. 그러나 적절한 학습 목표가 없으면 이러한 분화는 자동으로 보장되지 않는다.

객체 쿼리(Object Query) 아키텍처는 조작 작업에서 특히 유용하다. 각 쿼리는 후보 객체 또는 상호작용 대상을 나타낸다. 이미지, 깊이, 언어, 지도 토큰에 주의를 기울여 객체의 식별자, 자세, 어포던스(Affordance), 불확실성, 작업 관련성을 포함하는 표현을 생성한다.

이후 행동 쿼리는 원시 이미지 대신 이러한 객체 표현에 주의를 기울일 수 있다. 이는 계산량을 줄이고 해석 가능성을 높인다. 엔지니어는 어떤 객체 표현이 파지나 배치 행동에 영향을 주었는지 확인할 수 있다.

교차 어텐션은 다중 카메라 시스템(Multi-Camera System)에도 매우 적합하다. 로봇은 전면, 후면, 측면, 손목, 천장, 스테레오 카메라 등을 사용할 수 있다. 각 카메라는 서로 다른 시야와 중첩, 가림을 가진다.

공유 쿼리는 모든 카메라 스트림에 주의를 기울여 가장 유용한 시점을 선택할 수 있다. 카메라 ID, 보정 정보, 타임스탬프, 시야각(Field of View)이 함께 인코딩되어야 한다. 모델은 손목 카메라는 정밀 조작에, 외부 카메라는 전체 문맥에 적합하다는 점을 학습할 수 있다.

실행 단계에 따라 카메라 선택도 달라질 수 있다. 접근 단계에서는 광각 카메라가 중요하지만, 접촉 직전에는 손목 카메라가 더 중요해질 수 있다. 하나의 카메라가 가려질 경우 다른 시점이 이를 보완할 수 있다.

그러나 다중 카메라 교차 어텐션은 시각 토큰 수가 급격히 증가하므로 계산량이 매우 커질 수 있다. 토큰 풀링(Token Pooling), 영역 선택, 객체 수준 압축, 계층형 융합이 필요하다.

LiDAR와 카메라의 융합도 유사한 문제를 가진다. LiDAR는 신뢰성 높은 기하 정보를 제공하지만 의미 정보는 부족하다. 카메라는 풍부한 의미 정보를 제공하지만 조명 변화에 민감하다. 교차 어텐션은 두 정보를 결합하여 장애물 분류, 자유 공간 추정, 내비게이션을 향상시킨다.

레이더(Radar)는 악천후에서도 속도와 장거리 정보를 제공할 수 있다. 오디오는 기계 상태, 경보, 사람의 음성 명령을 제공할 수 있다. 열화상 카메라는 과열된 장비를 발견할 수 있다. 이러한 모달리티도 시간 정렬, 보정, 불확실성이 표현된다면 교차 어텐션에 포함될 수 있다.

교차 어텐션의 가장 큰 장점은 유연성이지만, 동시에 아키텍처 복잡성도 증가시킨다. 이론적으로는 모든 모달리티 쌍마다 별도의 교차 어텐션 블록을 둘 수 있다. 비전, 언어, LiDAR, 힘, 오디오, 지도, 고유감각을 모두 가진 시스템에서는 가능한 상호작용 수가 매우 많다.

완전히 연결된 융합 구조는 계산 비용이 크고 학습도 어렵다. 또한 우연한 상관관계를 학습할 위험도 있다. 설계자는 어떤 모달리티가 직접 상호작용해야 하는지, 어느 계층에서, 어떤 주기로 상호작용할지를 결정해야 한다.

희소 또는 라우팅 기반 교차 어텐션(Sparse or Routed Cross-Attention)은 이러한 비용을 줄일 수 있다. 라우팅 모듈은 현재 작업에 필요한 모달리티만 선택하고 해당 상호작용만 계산한다.

예를 들어 자유 주행에서는 촉각을 무시할 수 있지만 조작 작업에서는 활성화한다. 정지 상태 검사에서는 상세 지도 정보가 필요하지 않을 수 있다. 오디오는 경보나 음성 명령이 감지될 때만 중요해질 수 있다.

동적 라우팅(Dynamic Routing)은 적응형 계산을 가능하게 한다. 환경이나 작업이 복잡할 때만 더 많은 계산을 수행한다. 그러나 라우팅 오류는 중요한 정보를 숨길 수 있으므로 안전에 중요한 센서는 항상 유지되어야 한다.

교차 어텐션은 알려진 물리 구조에 의해 제한될 수도 있다. 왼쪽 그리퍼 행동 쿼리는 왼팔 관절, 주변 영상, 관련 힘 센서에 우선적으로 주의를 기울일 수 있다. 내비게이션 쿼리는 매니퓰레이터보다 이동 플랫폼 상태와 지역 지도 토큰에 더 집중할 수 있다.

구조화된 어텐션 마스크(Structured Attention Mask)는 계산량을 줄이는 동시에 운동학 연결(Kinematic Connectivity), 센서 위치, 공간적 근접성, 임무 권한, 인과 관계를 반영할 수 있다.

그러나 이러한 제약은 지나치게 엄격해서는 안 된다. 매니퓰레이터 움직임이 이동 플랫폼 안정성에 영향을 줄 수도 있고, 적재물이 카메라를 가릴 수도 있다. 따라서 예상하지 못한 상호작용을 표현할 수 있는 유연성도 유지해야 한다.

교차 어텐션은 인코더-디코더(Encoder-Decoder) 구조에서 널리 사용된다. 멀티모달 인코더는 관측과 목표를 표현하고, 행동 디코더는 교차 어텐션을 이용하여 제어 출력을 생성한다. 각 디코더 계층은 현재 행동 단계에 필요한 센서 정보를 검색한다.

이 구조는 자기회귀 행동 생성(Autoregressive Action Generation)을 지원한다. 디코더는 이전 행동에는 인과적 자기 어텐션(Causal Self-Attention)을 사용하고, 관측에는 교차 어텐션을 사용한다. 자기 어텐션은 행동 시퀀스의 시간 일관성을 유지하고, 교차 어텐션은 행동을 현재 환경과 연결한다.

행동 청크 생성에서는 여러 미래 제어 단계를 나타내는 여러 디코더 쿼리를 사용할 수 있다. 이들은 인코딩된 문맥에 공동으로 주의를 기울인다. 여러 행동을 함께 생성하기 때문에 개별 단계 예측보다 더 부드럽고 일관된 움직임을 만들 수 있다.

교차 어텐션은 확산(Diffusion) 및 플로우 기반(Flow-Based) 행동 모델과도 잘 결합된다. 잡음이 포함된 행동 궤적은 멀티모달 문맥에 주의를 기울이면서 점진적으로 정제된다. 각 디노이징(Denoising) 단계에서 행동 표현은 비전, 언어, 로봇 상태로부터 필요한 정보를 검색한다.

이는 여러 행동 궤적이 모두 가능한 경우에 유용하다. 확산 기반 디코더는 결정론적 회귀(Deterministic Regression)보다 다양한 행동 분포를 더 잘 표현할 수 있다.

생성 과정에서는 조건 정보가 항상 최신 상태를 유지해야 한다. 긴 행동 궤적을 생성하는 동안 센서 상태가 오래되면 위험한 결과가 나올 수 있다. 반복적인 재인코딩(Re-Encoding)과 후퇴 지평선(Receding Horizon) 실행이 이러한 위험을 줄인다.

교차 어텐션은 월드 모델(World Model)에도 중요하다. 미래 상태 쿼리는 현재 관측, 행동, 임무 목표에 주의를 기울인다. 이를 통해 환경의 어떤 부분이 앞으로 어떻게 변할지를 예측한다.

월드 모델은 객체 움직임, 접촉, 로봇 자세, 충돌 위험, 작업 진행 상태를 각각 담당하는 별도의 쿼리를 가질 수 있다. 각 쿼리는 현재 멀티모달 문맥에서 서로 다른 정보를 검색한다.

이러한 구조화된 예측은 전체 미래 이미지를 생성하는 것보다 실용적일 수 있다. 충돌 위험 쿼리는 기하 정보, 속도, 행동이 필요하고, 작업 진행 쿼리는 임무 상태, 객체 상태, 이전 행동이 필요하다.

교차 어텐션은 동일한 환경 표현에 여러 후보 행동을 조건으로 적용하여 반사실적 계획(Counterfactual Planning)을 수행할 수도 있다. 각 행동 시퀀스는 환경 표현에 질의하여 예상 결과를 생성한다.

플래너는 작업 성공률, 위험도, 에너지 소비, 불확실성을 비교할 수 있다. 이처럼 교차 어텐션은 인식, 행동, 결과 예측을 연결하는 핵심 인터페이스가 된다.

교차 어텐션 모델의 학습에는 정확하게 정렬된 멀티모달 데이터가 필요하다. 모델은 어떤 언어 명령이 어떤 작업과 대응되는지, 어떤 이미지가 어떤 로봇 상태와 대응되는지, 어떤 행동이 어떤 결과를 만드는지를 정확히 학습해야 한다.

동기화가 잘못되면 교차 어텐션은 잘못된 대응 관계를 학습한다. 힘 신호가 잘못된 이미지 프레임과 연결되거나, 언어 라벨이 다른 작업을 설명하거나, 지도 자세가 오래된 보정을 사용할 수 있다.

따라서 데이터 파이프라인은 타임스탬프, 좌표계, 센서 ID, 보정 버전, 행동 스키마(Action Schema), 작업 경계를 모두 보존해야 한다. 멀티모달 융합의 품질은 이러한 정렬 품질을 넘을 수 없다.

실제 시스템에서는 일부 모달리티가 누락되는 경우도 흔하다. 카메라가 고장나거나 힘 센서가 없거나 지도가 존재하지 않을 수 있다. 따라서 학습에는 모달리티 드롭아웃(Modality Dropout)이 포함되어야 한다.

모달리티가 없는 경우 모델은 단순한 0 벡터를 받는 것이 아니라 해당 입력이 존재하지 않는다는 사실을 명시적으로 알아야 한다. 누락 모달리티 토큰(Missing-Modality Token)이나 마스크(Mask)가 필요하다.

교차 어텐션의 강건성(Robustness)은 특정 모달리티를 제거하거나 지연시키거나 손상시키거나 다른 정보로 대체하여 평가해야 한다. 중요한 정보가 사라질 경우 시스템은 안전한 행동을 유지하면서 신뢰도를 낮출 수 있어야 한다.

모달리티 불균형(Modality Imbalance)도 중요한 학습 문제이다. 대규모 시각 정보는 작은 상태 정보나 힘 정보를 압도할 수 있다. 모델은 가장 예측력이 높은 모달리티만 사용하고 나머지는 무시하는 지름길을 학습할 수 있다.

균형 잡힌 샘플링(Balanced Sampling), 모달리티별 정규화, 보조 손실(Auxiliary Loss), 어텐션 정규화는 모든 입력을 의미 있게 사용하도록 도울 수 있다. 예를 들어 접촉 상태 예측 손실은 모델이 힘과 촉각을 실제로 활용하도록 강제할 수 있다.

교차 모달 대조학습(Cross-Modal Contrastive Learning)은 행동 학습 이전에 표현을 정렬하는 데 도움이 된다. 관련된 이미지, 언어, 상태, 행동은 잠재 공간에서 가깝게 배치되고, 관련 없는 데이터는 멀어지게 된다. 이후 교차 어텐션은 더 좋은 초기 표현에서 시작할 수 있다.

마스크드 멀티모달 모델링(Masked Multimodal Modeling)도 유용하다. 모델은 다른 센서들을 이용하여 누락된 센서 스트림을 복원할 수 있다. 비전과 행동으로 힘을 예측하거나, 상태와 행동으로 영상 변화를 예측하도록 학습하면 물리적 관계를 이해하게 된다.

그러나 예측된 센서 값은 실제 센서를 대체해서는 안 된다. 학습 기반 예측은 표현 학습과 이상 탐지에는 유용하지만, 기능 안전(Function Safety)은 반드시 실제 검증된 센서를 사용해야 한다.

교차 어텐션은 어텐션 가중치를 시각화할 수 있기 때문에 해석 가능성(Interpretability)과도 관련이 있다. 엔지니어는 어떤 이미지 영역, 센서 토큰, 명령 단어가 특정 행동에 영향을 주었는지를 확인할 수 있다.

이러한 시각화는 명백한 오류를 찾는 데 도움이 된다. 파지 행동이 잘못된 객체에 집중한다면 목표 접지(Grounding)에 문제가 있을 가능성이 있다. 내비게이션이 장애물 토큰을 무시한다면 위험한 지름길을 학습했을 수 있다.

그러나 어텐션 가중치만으로는 완전한 설명이 되지 않는다. 정보는 잔차 연결(Residual Connection), 피드포워드 네트워크(Feed-Forward Network), 이전 계층을 통해서도 전달된다. 어텐션이 낮은 토큰도 간접적으로 큰 영향을 줄 수 있다.

따라서 해석 가능성은 어텐션 시각화뿐 아니라 입력 마스킹(Input Masking), 반사실적 교란(Counterfactual Perturbation), 특징 기여도(Feature Attribution), 행동 재생(Action Replay), 행동 시험을 함께 수행해야 한다.

여러 모달리티가 같은 잘못된 이유로 일치하는 경우 교차 어텐션 모델은 과도한 확신을 가질 수도 있다. 예를 들어 지도와 이미지가 모두 오래된 정보를 포함하고 있을 수 있다. 두 정보원이 동일한 오류를 공유한다면 일치 자체는 정답을 의미하지 않는다.

따라서 독립적인 센서 검증과 정보원 품질 모델링(Source Quality Modeling)이 중요하다. 시스템은 동일한 정보를 반복해서 보는 것과 독립적인 여러 증거가 일치하는 것을 구분해야 한다.

안전이 중요한 교차 어텐션에서는 현재의 물리적 증거가 우선되어야 한다. 비상 센서, 충돌 감지기, 관절 한계 검사, 힘 제한은 단순한 학습 토큰으로만 표현되어서는 안 된다.

강제적인 안전 제약(Hard Safety Constraint)은 학습 기반 융합을 우회하거나 덮어쓸 수 있어야 한다. 교차 어텐션은 계획을 위한 문맥을 제공하지만 최종 명령 승인(Command Authorization)은 결정론적 시스템이 담당해야 한다.

지연 시간(Latency)도 중요한 고려사항이다. 많은 쿼리와 센서 토큰 사이의 교차 어텐션은 계산 비용이 매우 크다. 행동 쿼리가 많고 비전 인코더가 수천 개의 토큰을 생성하면 계산량이 급격히 증가한다.

따라서 토큰 압축(Token Compression)이 필수적이다. 이미지는 풀링(Pooling), 객체 추출(Object Extraction), 잠재 리샘플링(Latent Resampling), 영역 선택으로 줄일 수 있다. 포인트 클라우드는 클러스터(Cluster)나 기하 특징으로 요약할 수 있다. 긴 상태 이력은 순환 모델이나 상태공간 모델로 압축할 수 있다.

정적 정보는 캐시 기반 교차 어텐션(Cached Cross-Attention)을 사용할 수 있다. 언어 명령, 로봇 설명, 지도는 여러 제어 주기 동안 변하지 않을 수 있다. 이들의 키와 값 표현은 한 번 계산한 뒤 재사용할 수 있다.

반면 동적인 센서 특징은 계속 갱신되어야 한다. 작업, 공구, 로봇 구성, 환경이 바뀌면 캐시된 토큰은 무효화되어야 한다.

양자화(Quantization), 혼합 정밀도(Mixed Precision), 커널 융합(Kernel Fusion), 효율적 어텐션 구현, 하드웨어 최적화는 실행 비용을 줄일 수 있다. 평가는 반드시 실제 로봇 플랫폼에서 수행해야 한다.

에너지 소비도 중요하다. 항상 전체 멀티모달 융합을 수행하면 상당한 전력을 소비할 수 있다. 적응형 교차 어텐션은 필요할 때만 상세 융합을 수행하여 평균 계산량을 줄일 수 있다.

일반적인 이동 중에는 압축된 비전과 상태 정보만 사용하고, 도킹이나 조작 시에는 고해상도 교차 어텐션을 활성화할 수 있다. 이는 계산량을 작업 난이도에 맞추어 배분하는 방식이다.

교차 어텐션은 교차 체화 학습(Cross-Embodiment Learning)에서도 중요하다. 공유 작업 표현은 서로 다른 체화 설명에 주의를 기울일 수 있다. 로봇별 상태와 센서 토큰은 형태, 운동학(Kinematics), 적재물, 행동 공간을 설명한다.

모델은 "물체에 접근하라"와 같은 공통 개념을 학습하면서도 이동형 로봇, 휴머노이드, 드론에서 서로 다른 행동을 생성할 수 있다. 체화 쿼리는 현재 몸체에 필요한 작업 정보를 검색한다.

로봇별 행동 디코더는 동일한 멀티모달 문맥에 주의를 기울인다. 이를 통해 공통적인 인식과 의미는 공유하고, 플랫폼별 제어는 분리할 수 있다.

그러나 동일한 모달리티라도 로봇마다 의미가 다를 수 있다. 하나의 로봇 관절 상태 벡터를 다른 로봇 운동학으로 해석해서는 안 된다. 센서와 행동 스키마는 명확하게 구분되어야 한다.

공구 교체도 동일한 방식으로 처리할 수 있다. 공구 설명 토큰, 보정 정보, 형상, 기능 정보가 추가 문맥이 된다. 행동 쿼리는 움직임을 생성할 때 현재 공구 표현에 주의를 기울인다.

교차 어텐션은 플릿(Fleet) 수준 추론도 지원할 수 있다. 중앙 모델은 여러 로봇의 요약 정보, 임무 목표, 지도 정보, 자원 상태를 처리한다. 플릿 수준 쿼리는 작업 할당, 충돌 해결, 공유 공간 조정을 수행할 수 있다.

그러나 모든 로봇의 원시 센서 데이터를 중앙에서 직접 융합할 필요는 없다. 각 로봇은 관측을 의미 이벤트, 신뢰도, 작업 상태, 위험 요약으로 압축하고, 중앙에서는 이러한 요약 표현만 교차 어텐션으로 처리할 수 있다.

이는 통신 대역폭을 절약하고 시스템 복잡도를 줄이며, 통신이 끊겨도 로컬 자율성을 유지할 수 있게 한다.

다중 로봇 교차 어텐션에서는 로봇 ID와 시간이 명확하게 표현되어야 한다. 하나의 로봇 위치 정보가 다른 로봇 정보와 혼동되어서는 안 되며, 오래된 메시지도 현재 상태로 취급되어서는 안 된다.

보안(Security)도 중요하다. 악의적이거나 손상된 입력이 전체 멀티모달 표현에 영향을 줄 수 있다. 외부 도구 출력, 언어 명령, 지도 갱신, 외부 데이터베이스는 반드시 검증되어야 한다.

시각 표지판이나 문서에 포함된 프롬프트 인젝션(Prompt Injection)이 모든 텍스트를 신뢰하는 시스템에서는 로봇 행동을 변경할 수도 있다. 따라서 작업자 명령, 환경 텍스트, 데이터베이스 정보, 신뢰할 수 없는 관측을 구분해야 한다.

권한 임베딩(Authority Embedding)이나 보호된 명령 채널(Protected Instruction Channel)은 이를 지원할 수 있다. 중요한 임무 명령은 환경에서 발견된 일반 텍스트와 동일하게 취급되어서는 안 된다.

멀티모달 교차 어텐션의 평가는 단순한 작업 성공률만으로는 충분하지 않다. 접지 정확도(Grounding Accuracy), 모달리티 기여도(Modality Contribution), 센서 누락에 대한 강건성, 시간 정렬, 기하학적 일관성, 지연 시간, 에너지 소비, 안전 개입 등을 함께 평가해야 한다.

목표 선택 시험(Target Selection Test)은 언어가 올바른 객체를 선택하는지를 평가할 수 있다. 체화 시험은 시각 목표가 실제 도달 가능성을 고려하는지를 확인할 수 있다. 접촉 시험은 힘과 촉각이 실제 행동에 적절히 반영되는지를 확인할 수 있다.

제거 실험(Ablation Study)은 특정 모달리티를 제거하여 실제 의존성을 확인할 수 있다. 접촉 작업에서 힘 데이터를 제거해도 성능이 동일하다면 모델은 해당 모달리티를 제대로 사용하지 않는 것이다.

어텐션 품질 지표는 신중히 해석해야 한다. 사람과 동일한 영역에 주의를 기울였다고 해서 반드시 올바른 행동을 생성하는 것은 아니다. 최종 평가는 항상 폐루프 실행이어야 한다.

장기 평가(Long-Horizon Evaluation)도 중요하다. 시간이 지남에 따라 모달리티의 중요성은 변한다. 초기에는 목표를 올바르게 찾더라도 가림이나 작업 전환 이후 이를 잃어버릴 수 있다.

메모리 시스템은 중요한 교차 모달 연결을 유지할 수 있다. 비전과 언어를 통해 식별된 객체 ID는 임무 메모리에 저장된다. 객체가 잠시 보이지 않아도 로봇은 그 정체성과 예상 위치를 유지할 수 있다.

교차 어텐션은 이후 이 저장된 연결을 다시 검색할 수 있다. 이는 압축된 은닉 상태에만 의존하는 것보다 더 신뢰성이 높다.

교차 어텐션과 상태공간 모델(State Space Model)의 결합도 매우 유망한 방향이다. 맘바(Mamba)와 같은 시간 모델은 긴 센서 이력을 요약하고, 교차 어텐션은 현재 행동 쿼리를 과거 상태, 시각 객체, 명시적 메모리와 연결한다.

이러한 하이브리드 구조는 효율적인 스트리밍 메모리와 유연한 멀티모달 검색을 동시에 제공한다. 상태공간 계층은 시간 변화를 처리하고, 교차 어텐션은 서로 다른 모달리티 사이의 관계를 연결한다.

그래프 신경망(Graph Neural Network)과의 결합도 유용하다. 로봇 몸체, 객체, 방, 작업 의존성을 그래프로 표현하고, 교차 어텐션이 비정형 센서 특징과 구조화된 그래프 노드를 연결할 수 있다.

예를 들어 이미지 특징은 장면 그래프(Scene Graph)의 객체 노드에 주의를 기울이고, 행동 쿼리는 장면 그래프와 로봇 운동학 그래프 모두에 주의를 기울일 수 있다. 이를 통해 물리적 관계와 행동 제약을 함께 추론할 수 있다.

미래에는 교차 어텐션이 더욱 적응적으로 발전할 가능성이 있다. 불확실성, 작업 단계, 감지된 이벤트에 따라 새로운 쿼리를 동적으로 생성할 수 있다. 새로운 객체가 나타나면 객체 쿼리가 생성되고, 힘 이상이 발생하면 진단 쿼리가 생성되며, 사람의 질문이 들어오면 설명 쿼리가 생성될 수 있다.

이러한 이벤트 기반 쿼리(Event-Driven Query)는 모든 모달리티를 동일하게 처리하지 않고 의미 있는 변화에만 계산을 집중시킨다. 이는 실제 물리적 상호작용이 이벤트 중심이라는 사실과도 잘 맞는다.

장기적인 목표는 단순히 더 많은 센서를 결합하는 것이 아니다. 각 모달리티가 자신의 신뢰도, 관련성, 시간성, 물리적 의미에 따라 적절히 기여하는 일관된 표현(Coherent Representation)을 구축하는 것이다.

교차 어텐션의 가치는 전문화된 표현들 사이의 선택적 통신(Selective Communication)을 가능하게 한다는 데 있다. 언어는 시각 증거를 검색하고, 행동은 몸 상태와 환경 정보를 검색하며, 힘 신호는 접촉 문맥을 검색하고, 임무 목표는 관련 메모리를 검색할 수 있다.

그러나 성공은 단순한 어텐션 수식만으로 결정되지 않는다. 정확한 보정, 시간 동기화, 모달리티별 인코더, 불확실성 모델링, 계층형 제어, 토큰 압축, 안전 아키텍처, 폐루프 평가가 모두 필요하다.

대규모 교차 어텐션 모델도 잘못된 센서 좌표계, 오래된 데이터, 부족한 안전 검사, 불명확한 행동 의미를 보완할 수는 없다. 멀티모달 지능은 체계적인 시스템 엔지니어링(System Engineering) 위에서 구축되어야 한다.

따라서 피지컬 AI 파운데이션 모델에서 교차 어텐션은 완전한 해결책이라기보다 핵심 통합 메커니즘으로 이해해야 한다. 서로 독립적으로 유지되어야 하지만 긴밀히 협력해야 하는 정보들을 연결하는 역할을 수행한다.

가장 강력한 로봇 아키텍처는 교차 어텐션을 이용하여 의미적 목표를 현재 인식에 접지하고, 인식을 체화와 연결하며, 상호 보완적인 센서를 결합하고, 행동 생성을 가장 관련성이 높은 증거에 조건화할 것이다.

동시에 결정론적 기하학, 제어, 모니터링, 안전 시스템은 이러한 판단을 최종적으로 검증해야 한다. 교차 어텐션은 유연한 멀티모달 추론을 제공하고, 주변의 로봇 아키텍처는 이 추론이 제어 가능하고 신뢰할 수 있으며 물리적으로 의미 있는 행동이 되도록 보장한다.

이러한 역할에서 교차 어텐션은 파운데이션 모델 지능과 체화된 로봇 운용을 연결하는 가장 중요한 다리 가운데 하나가 된다. 이는 비전, 언어, 상태, 기하 정보, 힘, 메모리, 임무 문맥이라는 서로 분리된 정보 흐름을 통합하여, 로봇이 무엇을 인식하고, 이해하며, 다음에 무엇을 해야 하는지를 하나의 일관된 표현으로 변환한다.

## 2.8 Positional Encoding 3D Scene and Temporal [w/Code]

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

위치 인코딩(Positional Encoding)은 트랜스포머(Transformer) 기반 및 토큰 기반(Token-Based) 피지컬 AI(Physical AI) 모델의 핵심 구성요소이다. 어텐션 메커니즘(Attention Mechanism)은 정보가 어디에 존재하는지 또는 사건이 언제 발생했는지를 본질적으로 이해하지 못하기 때문이다. 트랜스포머는 토큰의 내용을 기준으로 서로 비교할 수 있지만, 별도의 구조가 제공되지 않으면 하나의 이미지 패치(Image Patch)가 다른 패치보다 위에 있다는 사실, 객체가 로봇의 3미터 앞에 있다는 사실, 힘 이벤트(Force Event)가 접촉 이후 발생했다는 사실, 두 관측 사이에 수초의 시간 차이가 있다는 사실을 자동으로 알지 못한다. 위치 인코딩은 이러한 공간적·시간적 구조를 모델에 제공한다.

언어 모델(Language Model)에서는 일반적으로 단어가 1차원 시퀀스(One-Dimensional Sequence) 안에서 나타나는 순서를 위치로 표현한다. 그러나 로보틱스(Robotics)는 훨씬 더 복잡한 위치 표현을 요구한다. 로봇 관측에는 2차원 이미지 좌표, 3차원 객체 위치, 회전(Rotation), 센서 자세(Sensor Pose), 관절 구성(Joint Configuration), 지도 위치(Map Location), 궤적 지점(Trajectory Point), 타임스탬프(Timestamp), 경과 시간(Elapsed Time), 움직임 이력(Motion History)이 포함될 수 있다. 이러한 변수들은 서로 다른 좌표계(Coordinate System)에 존재하며 서로 다른 속도로 변화한다. 따라서 피지컬 AI 파운데이션 모델(Physical AI Foundation Model)은 공간, 시간, 체화(Embodiment), 센서 기하학(Sensor Geometry)에 걸쳐 위치를 표현해야 한다.

위치 인코딩의 중요성은 어텐션의 기본적인 특성에서 비롯된다. 토큰의 순서나 기하 구조가 명시적으로 제공되지 않으면 자기 어텐션(Self-Attention)은 순열 불변성(Permutation Invariance)을 가진다. 동일한 토큰 집합의 순서를 바꾸더라도 어텐션 메커니즘은 순서가 아니라 내용만 비교하기 때문에 같은 결과를 생성할 수 있다. 이는 순서가 없는 집합(Unordered Set)을 처리할 때는 유용하지만, 공간 배열(Spatial Arrangement)이 중요한 경우에는 문제가 된다.

예를 들어 로봇이 문(Door), 사람(Person), 벽(Wall)을 나타내는 이미지 토큰을 입력받을 수 있다. 그러나 객체의 정체성만으로는 충분하지 않다. 사람이 문 앞에 있는지, 벽 뒤에 있는지, 로봇 이동 경로 밖에 있는지를 알아야 한다. 마찬가지로 동일한 값을 가진 두 관절 상태 토큰도 서로 다른 관절을 의미할 수 있으므로 물리적 의미가 완전히 다르다. 위치, 식별 정보(Identity), 구조적 문맥(Structural Context)이 함께 인코딩되어야 한다.

가장 단순한 위치 인코딩은 모든 토큰에 위치 벡터(Position Vector)를 할당한 뒤, 이를 토큰의 내용 임베딩(Content Embedding)에 더하거나 연결하는 방식이다. 1차원 시퀀스에서는 위치가 정수 인덱스(Integer Index)일 수 있다. 로봇 시스템에서는 위치 벡터에 이미지 좌표, 깊이(Depth), 월드 위치(World Position), 센서 ID, 방향(Orientation), 타임스탬프, 신뢰도(Confidence)가 포함될 수 있다.

절대 위치 인코딩(Absolute Positional Encoding)은 고정된 기준계 안에서 토큰의 위치를 표현한다. 이미지에서는 수평 및 수직 패치 좌표가 될 수 있다. 지도에서는 전역 x, y, z 좌표가 될 수 있다. 로봇 궤적에서는 시간 단계(Time Step) 또는 절대 타임스탬프를 의미할 수 있다.

절대 인코딩은 위치 자체가 의미를 가질 때 유용하다. 고정된 천장 카메라(Overhead Camera)는 항상 이미지 오른쪽 아래 영역에서 적재 구역(Loading Zone)을 관찰할 수 있다. 알려진 검사 지점(Inspection Station)은 일정한 전역 지도 좌표를 가질 수 있다. 로봇의 각 관절은 운동학 체인(Kinematic Chain) 안에서 고정된 위치를 가진다.

그러나 카메라, 로봇, 환경이 변경되면 절대 위치만 사용한 표현은 일반화 성능이 낮아질 수 있다. 한 카메라에서는 왼쪽 위에 나타나는 객체가 다른 카메라에서는 오른쪽 아래에 나타날 수 있다. 서로 다른 공장에서는 동일한 기계가 다른 전역 좌표에 설치될 수 있다. 따라서 모델은 고정된 위치를 암기하기보다 관계(Relationship)를 이해해야 하는 경우가 많다.

상대 위치 인코딩(Relative Positional Encoding)은 토큰 사이의 변위(Displacement)나 관계를 표현한다. 각 객체가 어디에 있는지만 나타내는 대신, 하나의 객체가 다른 객체보다 위에 있는지, 뒤에 있는지, 가까운지, 어느 방향으로 회전되어 있는지를 인코딩한다. 어텐션 메커니즘에서는 상대 위치를 어텐션 점수(Attention Score)에 추가하거나 키(Key)와 값(Value)을 수정하는 데 사용할 수 있다.

상대 인코딩은 로봇 작업이 관계 중심(Relational)이기 때문에 특히 중요하다. 명령은 모터 위쪽의 밸브, 팔레트 뒤의 상자, 커넥터 왼쪽의 검사 지점, 그리퍼와 가장 가까운 객체를 지칭할 수 있다. 이러한 개념은 고정된 월드 위치가 아니라 객체 간 관계에 의해 결정된다.

상대 위치는 환경 간 전이(Transfer)도 향상시킬 수 있다. 그리퍼와 목표 객체 사이의 관계는 두 객체가 다른 위치로 함께 이동하더라도 의미를 유지한다. 내비게이션 정책(Navigation Policy)은 모든 장애물의 지도 좌표를 암기하지 않고도 로봇과 장애물의 상대 위치를 기준으로 추론할 수 있다.

2차원 위치 인코딩(Two-Dimensional Positional Encoding)은 이미지에 널리 사용된다. 비전 트랜스포머(Vision Transformer)는 이미지를 패치로 나누고 각 패치를 토큰으로 표현한다. 모델은 각 패치에 해당하는 행(Row)과 열(Column)을 알아야 한다. 이를 위해 수평 위치 벡터와 수직 위치 벡터를 결합할 수 있다.

학습형 2차원 임베딩(Learned Two-Dimensional Embedding)은 각 패치 위치에 학습 가능한 벡터를 할당한다. 데이터셋의 공간적 규칙을 잘 학습할 수 있지만 이미지 해상도나 패치 수가 달라지면 어려움을 겪을 수 있다. 새로운 해상도에 맞게 보간(Interpolation)할 수 있지만, 보간된 표현이 원래 기하 구조를 완전히 보존하는 것은 아니다.

사인파 위치 인코딩(Sinusoidal Positional Encoding)은 서로 다른 주파수(Frequency)를 가진 수학 함수를 이용한다. 학습 중 보지 못한 위치에서도 계산할 수 있는 연속적인 구조를 제공한다. 2차원에서는 수평 좌표와 수직 좌표를 각각 별도의 사인파 함수로 표현할 수 있다.

주기적 구조(Periodic Structure)를 이용하면 여러 주파수의 결합을 통해 상대 변위를 추론할 수 있다. 그러나 기본적인 사인파 인코딩은 주로 시퀀스 순서를 위해 개발되었으며, 실제 거리 단위의 기하학을 직접 표현하지는 않는다. 따라서 로봇 인식에서는 깊이 및 카메라 정보가 추가로 필요하다.

회전형 위치 임베딩(Rotary Positional Embedding)은 위치에 따라 쿼리(Query)와 키 벡터에 회전 변환을 적용한다. 그 결과 어텐션 점수에 토큰 사이의 상대 변위가 반영된다. 회전형 방식은 대규모 언어 모델에서 널리 사용되며 이미지, 비디오, 3차원 데이터로 확장할 수 있다.

비전에서는 회전형 인코딩을 수평축과 수직축에 각각 적용할 수 있다. 3차원 장면에서는 x, y, z 좌표 방향으로 확장할 수 있다. 이는 토큰 내용에 위치 벡터를 단순히 더하는 방식보다 상대적인 공간 관계를 구조적으로 표현할 수 있게 한다.

3차원 위치 인코딩(Three-Dimensional Positional Encoding)은 로봇 인식이 깊이 카메라(Depth Camera), 스테레오 비전(Stereo Vision), LiDAR, 포인트 클라우드(Point Cloud), 점유 지도(Occupancy Map), 재구성된 장면(Reconstructed Scene)을 포함할 때 필요하다. 하나의 토큰은 3차원 공간의 점(Point), 복셀(Voxel), 객체(Object), 표면 영역(Surface Region), 장면 노드(Scene Node)를 표현할 수 있다.

가장 직접적인 방법은 데카르트 좌표(Cartesian Coordinate) x, y, z를 임베딩하는 것이다. 원시 좌표를 정규화(Normalization)한 뒤 신경망을 통해 투영할 수 있다. 푸리에 특징(Fourier Feature)은 좌표를 여러 주파수로 변환하여 넓은 공간 변화와 미세한 기하 세부사항을 모두 표현할 수 있게 한다.

좌표 정규화는 신중하게 설계해야 한다. 전역 지도는 수백 미터 범위를 가질 수 있지만 조작 작업은 밀리미터 단위의 정밀도를 요구할 수 있다. 모든 작업에 하나의 스케일만 사용하면 미세한 세부사항을 잃거나 수치 범위가 불안정해질 수 있다.

다중 스케일 좌표 인코딩(Multi-Scale Coordinate Encoding)은 이러한 문제를 해결할 수 있다. 거친 인코딩(Coarse Encoding)은 방(Room), 검사 지점, 지도 수준의 위치를 표현하고, 세밀한 인코딩(Fine Encoding)은 객체의 지역 형상과 접촉 지점을 표현한다. 모델은 전역 내비게이션과 지역 조작을 동시에 추론할 수 있다.

계층적 공간 인코딩(Hierarchical Spatial Encoding)은 이동형 매니퓰레이터(Mobile Manipulator)에 특히 유용하다. 하나의 목표를 건물, 방, 이동 플랫폼, 매니퓰레이터 베이스, 엔드이펙터(End Effector), 공구 중심점(Tool Center Point)을 기준으로 각각 표현할 수 있다. 각 계층은 서로 다른 정보를 제공한다.

로봇이 어떤 부품을 검사하라는 명령을 받으면 먼저 전역 좌표를 이용해 올바른 검사 지점으로 이동할 수 있다. 이후 지역 지도 좌표로 이동 플랫폼을 정렬하고, 로봇 좌표계에서 매니퓰레이터를 대상에 접근시키며, 공구 좌표계에서 접촉을 제어할 수 있다. 하나의 구분되지 않은 위치 표현만으로는 이 모든 단계를 효과적으로 지원하기 어렵다.

따라서 좌표계 식별 정보(Coordinate-Frame Identity)를 명시적으로 인코딩해야 한다. x가 1미터라는 벡터는 어떤 기준 좌표계를 사용하는지 알지 못하면 의미가 없다. 이는 카메라 전방 1미터, 로봇 베이스 전방 1미터, 지도 원점으로부터 1미터, 엔드이펙터 기준 1미터를 의미할 수 있다.

좌표계 인식 위치 인코딩(Frame-Aware Positional Encoding)은 좌표계 ID, 변환 행렬(Transformation Matrix), 센서 자세, 부모-자식 관계(Parent-Child Relationship)를 포함할 수 있다. 모델이 모든 위치를 공통 좌표계로 변환하여 입력받더라도 원래 좌표계의 식별 정보는 센서 기하와 불확실성을 이해하는 데 유용할 수 있다.

모든 관측을 하나의 공통 월드 좌표계로 변환하면 멀티모달 융합이 단순해진다. 그러나 이 방식은 정확한 보정(Calibration)과 위치 추정(Localization)에 의존한다. 로봇 자세 추정이 잘못되면 변환된 모든 관측이 불일치하게 된다. 따라서 가능하다면 변환 신뢰도나 공분산(Covariance)을 함께 제공해야 한다.

지역 좌표계(Local Frame) 표현은 제어에 더 강건할 수 있다. 현재 로봇 자세에 대한 상대 위치로 목표를 표현하면 전역 위치 추정이 조금 변하더라도 의미가 유지된다. 상대 기하학은 반응형 행동(Reactive Behavior)을 지원하고, 전역 좌표는 임무 계획을 지원할 수 있다.

3차원 방향(Three-Dimensional Orientation)은 위치보다 더 복잡한 문제를 추가한다. 위치만으로는 객체의 전체 자세(Pose)를 표현할 수 없다. 로봇은 카메라, 공구, 객체, 관절, 좌표계의 회전을 표현해야 한다. 방향은 오일러각(Euler Angle), 회전 행렬(Rotation Matrix), 축-각 벡터(Axis-Angle Vector), 쿼터니언(Quaternion)으로 표현할 수 있다.

오일러각은 직관적이지만 불연속성과 특이점(Singularity)을 가진다. 각도가 주기적으로 래핑(Wrapping)되는 지점에서는 작은 실제 회전이 큰 수치 변화로 나타날 수 있다. 회전 행렬은 일부 불연속성을 피하지만 중복된 값을 포함하며 직교성(Orthogonality) 제약을 만족해야 한다.

쿼터니언은 짐벌 락(Gimbal Lock) 없이 회전을 간결하게 표현하지만 q와 -q가 동일한 물리적 회전을 나타낸다. 신경망은 이러한 이중 표현(Double Representation)을 주의해서 처리해야 한다. 학습이 쉽고 유효한 회전 행렬로 변환할 수 있기 때문에 6차원 연속 회전 표현(Six-Dimensional Continuous Rotation Representation)이 사용되기도 한다.

방향 임베딩은 위치 임베딩과 결합하여 6자유도 자세(Six-Degree-of-Freedom Pose)를 표현할 수 있다. 조작에서는 목표 객체의 자세와 엔드이펙터 자세를 함께 인코딩할 수 있다. 상대 자세 인코딩(Relative Pose Encoding)은 두 자세를 정렬하기 위해 필요한 변환을 표현한다.

강체 변환(Rigid-Body Transformation)은 수학적으로 특수 유클리드 군(Special Euclidean Group) SE(3)에 속한다. SE(3) 기하학을 존중하는 아키텍처는 일반화 성능을 향상시킬 수 있다. SE(3) 등변 모델(SE(3)-Equivariant Model)은 입력 장면이 이동하거나 회전하면 출력도 그에 맞추어 일관되게 변화한다.

등변성(Equivariance)은 물리 법칙과 로봇 작업이 좌표 변환 이후에도 동일한 경우가 많기 때문에 유용하다. 객체와 로봇을 함께 회전시키면 예측된 파지 자세도 같은 방향으로 회전해야 한다. 모든 방향에서 동일한 상호작용을 다시 학습해야 하는 모델은 비효율적이다.

등변 위치 인코딩(Equivariant Positional Encoding)은 상대 변위 벡터, 거리, 구면 조화 함수(Spherical Harmonic), 군 표현(Group Representation)을 사용할 수 있다. 이러한 방법은 임의의 학습 임베딩보다 기하학적 구조를 명확하게 보존한다.

그러나 엄격한 등변성이 항상 적합한 것은 아니다. 중력(Gravity)은 고정된 수직 방향을 만든다. 카메라 방향은 이미지의 외관에 영향을 준다. 공장 지도에는 고정된 전역 방향과 제한 구역이 존재한다. 아키텍처는 보존해야 하는 대칭성과 절대적으로 유지해야 하는 환경 구조를 구분해야 한다.

거리 인코딩(Distance Encoding)도 핵심 구성요소이다. 토큰 사이의 유클리드 거리(Euclidean Distance)는 어텐션에 직접 영향을 줄 수 있다. 접촉, 충돌, 지역 기하학에서는 가까운 점이 먼 점보다 더 중요할 수 있다.

방사 기저 함수(Radial Basis Function)는 거리를 부드러운 임베딩으로 변환할 수 있다. 여러 거리 스케일을 사용하면 미세한 지역 구조와 넓은 장면 관계를 함께 표현할 수 있다. 공간적 근접성에 따라 어텐션을 제한하거나 편향시킬 수도 있다.

지역 공간 어텐션(Local Spatial Attention)은 대규모 포인트 클라우드의 계산량을 줄인다. 하나의 점이나 객체 토큰은 장면의 모든 토큰이 아니라 주로 주변 이웃에 주의를 기울인다. 이는 많은 물리적 상호작용의 지역성을 반영한다.

그러나 장거리 관계도 중요하다. 멀리 있는 이동 차량이 내비게이션에 영향을 줄 수 있고, 먼 임무 목표가 현재 계획을 결정할 수 있다. 계층적 어텐션(Hierarchical Attention)은 지역 기하 추론과 전역 의미 문맥을 결합할 수 있다.

포인트 클라우드는 본질적으로 순서가 없는 데이터이다. 미리 정의된 시퀀스 순서가 존재하지 않는다. 3차원 위치 인코딩은 어텐션이 점 사이의 관계를 추론하는 데 필요한 구조를 제공한다. 점 좌표, 표면 법선(Surface Normal), 반사 강도(Intensity), 타임스탬프, 센서 원점(Sensor Origin)을 함께 포함할 수 있다.

LiDAR 포인트 클라우드는 스캔 특유의 시간 구조도 가진다. 센서가 회전하는 동안 점들이 조금씩 다른 시간에 수집된다. 로봇이나 환경이 움직이는 경우 전체 스캔을 동시에 취득한 것처럼 처리하면 왜곡이 발생할 수 있다.

시간 인식 점 인코딩(Time-Aware Point Encoding)은 각 점의 취득 시간을 포함할 수 있다. 모션 보정(Motion Compensation)을 통해 점들을 하나의 공통 시간 기준으로 변환하면서도 잔여 시간 정보를 모델에 유지할 수 있다.

복셀 기반 표현(Voxel-Based Representation)은 공간을 규칙적인 3차원 셀로 분할한다. 각 복셀은 이산 공간 인덱스를 가지며 점유 상태(Occupancy), 의미 정보(Semantic Information), 특징을 포함할 수 있다. 3차원 그리드 위치 인코딩은 이미지 패치 인코딩과 유사한 방식으로 적용할 수 있다.

복셀 그리드는 합성곱과 어텐션을 단순화하지만, 높은 해상도에서는 매우 큰 메모리를 소비할 수 있다. 희소 복셀(Sparse Voxel) 방식은 점유된 영역만 처리한다. 다중 해상도 복셀 계층(Multi-Resolution Voxel Hierarchy)은 넓은 환경을 표현하면서도 지역적인 세부사항을 보존할 수 있다.

객체 중심 표현(Object-Centric Representation)은 공간 복잡성을 더 줄일 수 있다. 모든 점을 인코딩하는 대신 감지된 객체나 부품을 토큰으로 표현한다. 각 토큰은 클래스(Class), 위치, 방향, 크기(Size), 속도(Velocity), 불확실성, 어포던스(Affordance)를 포함할 수 있다.

객체 수준 위치 인코딩은 관계 추론과 장기 기억에 유용하다. 개별 이미지 패치가 변하더라도 로봇은 기계 부품의 정체성과 자세를 계속 추적할 수 있다. 그러나 미지의 구조나 부분적으로 관측된 대상에서는 객체 추출이 실패할 수 있다.

하이브리드 표현은 객체 토큰과 선택된 밀집 기하 특징을 함께 유지할 수 있다. 객체 토큰은 의미 계획을 지원하고, 밀집 특징은 조작과 검사에 필요한 정밀한 지역 기하학을 보존한다.

장면 그래프(Scene Graph)는 또 다른 구조화 표현이다. 객체, 표면, 방, 로봇 링크, 작업 요소를 그래프 노드(Graph Node)로 표현한다. 엣지(Edge)는 위쪽, 연결됨, 내부, 지지됨, 도달 가능함과 같은 공간적·기능적·인과적 관계를 나타낸다.

그래프 위치 인코딩(Graph Positional Encoding)은 노드 ID, 그래프 거리(Graph Distance), 엣지 유형(Edge Type), 기하학적 변위를 사용할 수 있다. 트랜스포머 또는 그래프 신경망(Graph Neural Network)은 이러한 구조화된 장면에서 추론할 수 있다.

산업 검사에서는 장면 그래프가 기계와 패널, 밸브, 용접부, 커넥터, 검사 지점을 연결할 수 있다. 로봇은 언어 명령을 해당 그래프 노드에 연결하고, 다시 그 노드를 실시간 센서 데이터에 접지할 수 있다.

시간 위치 인코딩(Temporal Positional Encoding)은 공간 인코딩만큼 중요하다. 로봇 시스템은 관측과 행동의 시퀀스를 통해 동작한다. 모델은 무엇이 발생했는지만이 아니라 언제 발생했으며 어떤 순서였는지를 알아야 한다.

가장 단순한 시간 인코딩은 정수 시간 단계 인덱스를 할당한다. 데이터가 일정한 주기로 샘플링된다면 효과적이다. 그러나 로봇 센서는 서로 다른 주기로 동작하며 관측이 지연되거나 누락되거나 불규칙하게 수집될 수 있다.

절대 타임스탬프는 실제 시간 정보를 보존한다. 하나의 토큰은 임무 시작 이후 경과 시간, 전역 시각(Global Clock Time), 센서 취득 시각(Sensor Acquisition Time)을 포함할 수 있다. 절대 시간은 로깅(Logging), 일정 관리(Scheduling), 다중 시스템 동기화에 유용하다.

상대 시간 인코딩(Relative Temporal Encoding)은 토큰 사이의 지연 시간(Delay)을 표현한다. 이는 동역학을 이해할 때 더 중요할 수 있다. 두 관측이 10밀리초 간격인지 1초 간격인지에 따라 물리적 의미가 크게 달라질 수 있지만, 시퀀스 인덱스상으로는 모두 한 단계 차이일 수 있다.

경과 시간 임베딩(Elapsed-Time Embedding)을 사용하면 불규칙 샘플링을 처리할 수 있다. 상태 간 전이는 실제 시간 간격에 따라 달라질 수 있다. 이는 상태공간 모델(State Space Model), 이벤트 카메라(Event-Based Sensor), 패킷 손실, 비동기 멀티모달 융합에서 특히 유용하다.

시간 푸리에 특징(Temporal Fourier Feature)은 연속 시간을 여러 주파수로 표현할 수 있다. 높은 주파수는 진동, 접촉과 같은 빠른 변화를 표현하고, 낮은 주파수는 임무 진행이나 배터리 추세와 같은 느린 변화를 표현한다.

로봇 시스템에는 여러 시간 척도가 존재한다. 모터 제어는 킬로헤르츠(kHz) 수준으로 동작할 수 있고, 힘 제어는 수백 헤르츠, 카메라는 수십 헤르츠, 임무 계획은 수초 또는 수분 간격으로 갱신될 수 있다.

하나의 시간 인코딩만으로는 모든 시간 척도를 효과적으로 표현하지 못할 수 있다. 계층적 시간 인코딩(Hierarchical Temporal Encoding)은 빠른 제어 시간, 지역 작업 시간, 임무 시간을 분리할 수 있다. 하나의 토큰이 여러 시간 좌표를 동시에 가질 수도 있다.

예를 들어 관절 상태 토큰은 정확한 센서 타임스탬프, 현재 행동 시작 이후 경과 시간, 현재 스킬 시작 이후 경과 시간, 임무 시작 이후 경과 시간을 함께 포함할 수 있다. 이러한 값은 서로 다른 종류의 시간 문맥을 제공한다.

단계 인코딩(Phase Encoding)도 유용하다. 조작 시퀀스는 접근(Approach), 접촉(Contact), 파지(Grasp), 들어 올리기(Lift), 운반(Transport), 배치(Placement) 단계로 구성될 수 있다. 절대 시간은 비슷하더라도 서로 다른 작업에서는 기능적 역할이 다를 수 있다.

작업 단계 토큰(Task-Phase Token)은 타임스탬프를 보완할 수 있다. 모델은 자유 공간 이동과 접촉 단계에서 힘 신호를 다르게 해석해야 한다는 사실을 학습할 수 있다. 임무 상태 정보는 의미적인 시간 좌표를 제공한다.

주기적 행동(Periodic Behavior)에는 특별한 표현이 필요하다. 보행, 바퀴 회전, 진동, 반복적인 산업 동작에는 주기가 존재한다. 사인파 시간 특징은 이러한 주기 안의 위상(Phase)을 표현할 수 있다.

사족보행 로봇(Quadruped)에서는 보행 위상이 어떤 발이 지면에 닿아야 하는지를 결정한다. 회전 기계에서는 진동 위상이 비정상 패턴을 찾는 데 도움을 줄 수 있다. 이동 로봇에서는 바퀴 회전 패턴을 통해 슬립(Slip)이나 기계적 결함을 감지할 수 있다.

행동 생성에서는 시간 순서가 인과적(Causal)으로 유지되어야 한다. 배포 시점에 현재 행동을 생성하면서 미래 관측을 사용해서는 안 된다. 인과적 어텐션 마스크(Causal Attention Mask)는 이후 시간 단계의 정보가 이전 예측에 영향을 주지 못하도록 한다.

오프라인 표현 학습에서는 전체 시퀀스를 사용할 수 있으므로 양방향 시간 어텐션(Bidirectional Temporal Attention)이 유용할 수 있다. 모델은 과거와 미래 문맥을 함께 이용하여 누락된 구간을 복원할 수 있다. 그러나 실제 배포 모델은 미래 정보를 필요로 하지 않도록 설계되어야 한다.

오프라인 처리와 온라인 처리의 차이를 명확히 해야 한다. 모델이 오프라인 평가에서는 좋은 성능을 보이지만 미래 프레임에 의존한다면 실시간 제어에는 사용할 수 없다.

비디오 위치 인코딩(Video Positional Encoding)은 공간 구조와 시간 구조를 함께 결합한다. 각 시각 토큰은 이미지 좌표와 프레임 시간을 가진다. 가장 단순한 표현은 수평 위치, 수직 위치, 시간에 각각 별도의 임베딩을 사용하는 것이다.

분해형 위치 인코딩(Factorized Positional Encoding)은 공간과 시간을 별도로 처리하여 복잡성을 줄인다. 공간 어텐션은 각 프레임 내부의 관계를 처리하고, 시간 어텐션은 프레임 사이의 특징 변화를 추적한다.

결합 시공간 인코딩(Joint Spatiotemporal Encoding)은 각 비디오 패치를 시간, 높이, 너비로 구성된 3차원 그리드의 토큰으로 처리한다. 유연한 상호작용이 가능하지만 시퀀스 길이가 매우 커질 수 있다.

로보틱스에서는 비디오 그리드 정보만으로는 충분하지 않다. 카메라 움직임, 로봇 자세, 깊이, 센서 좌표계가 시각 변화의 의미에 영향을 준다. 이미지에서 픽셀이 이동했다고 해서 반드시 객체가 움직인 것은 아니다. 카메라가 움직였거나 둘 다 움직였을 수 있다.

자기운동 인식 위치 인코딩(Egomotion-Aware Positional Encoding)은 각 프레임과 연관된 로봇 또는 카메라 자세를 포함한다. 시각 토큰을 공통 좌표계에서 비교할 수 있어 장면의 움직임과 센서 자체의 움직임을 구분하는 데 도움을 준다.

손목 카메라(Wrist Camera)는 로봇 팔과 함께 계속 움직인다. 모델은 각 프레임의 카메라 순운동학 변환(Forward-Kinematic Transform)을 알아야 한다. 그렇지 않으면 물리적으로 정지한 객체도 계속 움직이는 것처럼 보일 수 있다.

자세 조건부 시각 토큰(Pose-Conditioned Visual Token)은 패치 좌표, 카메라 내부 파라미터(Camera Intrinsics), 외부 파라미터(Camera Extrinsics), 깊이, 타임스탬프를 결합할 수 있다. 이는 각 시각 관측이 어디에서 출발하여 어떤 방향을 바라보는지를 표현하는 기하학적 광선(Ray) 정보를 제공한다.

광선 기반 위치 인코딩(Ray-Based Positional Encoding)은 다중 카메라 융합에 유용하다. 시각 특징을 카메라 중심에서 3차원 공간으로 향하는 광선과 연결할 수 있다. 서로 다른 카메라의 토큰을 단순한 이미지 좌표가 아니라 기하학적 광선 관계로 비교할 수 있다.

깊이 정보가 존재하면 광선을 3차원 점으로 변환할 수 있다. 깊이가 불확실하면 광선을 따라 존재할 수 있는 위치 분포를 표현할 수 있다.

불확실성(Uncertainty)도 위치 인코딩의 일부가 되어야 한다. 로봇 위치, 객체 자세, 깊이 추정, 타임스탬프는 완벽하게 알려져 있지 않다. 위치 추정에는 공분산이 있을 수 있고, 객체 감지에는 자세 불확실성이 있으며, 네트워크 지연으로 관측의 실제 나이가 불확실할 수 있다.

추정 위치만 입력하면 모델은 불확실한 값을 정확한 값으로 취급할 수 있다. 신뢰도, 공분산, 품질 임베딩(Quality Embedding)을 함께 제공하면 어텐션이 정보의 신뢰성을 고려할 수 있다.

불확실성은 공간 어텐션에 영향을 줄 수 있다. 평균 좌표는 다르더라도 불확실성 영역이 겹치는 두 객체는 동일한 객체일 가능성이 있다. 내비게이션 시스템은 장애물 위치가 불확실하면 더 큰 안전 여유를 적용할 수 있다.

시간 불확실성도 중요하다. 센서 타임스탬프는 정확하더라도 처리 지연이나 통신 지연 때문에 관측의 실질적인 최신성이 불확실할 수 있다. 최신성 표시(Freshness Indicator)는 모델이 해당 관측을 얼마나 신뢰할지 결정하는 데 도움을 준다.

위치 인코딩은 로봇 자체의 구조도 표현해야 한다. 관절 토큰은 운동학 체인에서의 식별자와 위치를 가져야 한다. 관절 값을 임의의 시퀀스 순서로만 배치하면 모델이 물리 구조가 아니라 임의의 인덱스 순서에 의존할 수 있다.

운동학 위치 인코딩(Kinematic Positional Encoding)은 부모 관절, 자식 링크, 관절 유형, 회전축(Axis), 링크 길이, 그래프 깊이(Graph Depth)를 포함할 수 있다. 어텐션은 이러한 물리적 연결 관계를 반영할 수 있다.

로봇 링크 사이의 상대 변환은 단순한 시퀀스 인덱스보다 더 의미 있는 구조를 제공한다. 어깨 관절과 손목 관절은 토큰 목록에서는 멀리 떨어져 있을 수 있지만 운동학적으로 강하게 연결되어 있다.

그래프 거리를 어텐션 편향(Attention Bias)으로 추가할 수 있다. 인접한 링크는 더 강하게 상호작용하고, 전체 신체 조정을 위해 전역 어텐션도 유지할 수 있다.

교차 체화 학습(Cross-Embodiment Learning)에서는 형태 인식 위치 인코딩(Morphology-Aware Positional Encoding)이 필수적이다. 서로 다른 로봇은 관절 수와 링크 배열이 다르다. 하나의 관절 인덱스에 고정된 학습형 임베딩은 다른 로봇으로 잘 전이되지 않는다.

구조화된 형태 설명자(Morphology Descriptor)를 사용하면 하나의 공통 모델이 다양한 몸체를 처리할 수 있다. 각 액추에이터나 센서 토큰은 고정 벡터에서의 위치가 아니라 물리적 역할과 연결 관계를 포함한다.

행동 공간(Action Space)에도 위치 구조가 필요하다. 하나의 행동 토큰은 특정 관절, 엔드이펙터 축, 미래 궤적 단계, 스킬 단계를 나타낼 수 있다. 모델은 각 출력이 어떤 액추에이터와 어떤 미래 시점을 제어하는지를 알아야 한다.

미래 행동 쿼리(Future-Action Query)는 시간 오프셋(Temporal Offset)을 포함할 수 있다. 다음 제어 단계의 쿼리와 1초 뒤 제어 단계의 쿼리는 다르게 표현되어야 한다. 상대 미래 시간 인코딩을 사용하면 행동 디코더가 일관된 궤적을 생성할 수 있다.

궤적 위치 인코딩(Trajectory Positional Encoding)은 웨이포인트 순서, 경과 시간, 경로 길이(Arc Length), 기준 좌표계를 표현할 수 있다. 이동 로봇에서는 곡률(Curvature)과 헤딩(Heading) 관계가 중요하다. 매니퓰레이터에서는 상대 엔드이펙터 자세와 관절 구성이 일관되어야 한다.

확산(Diffusion) 및 플로우 기반 행동 모델(Flow-Based Action Model)도 시간 위치를 필요로 한다. 각 행동 지점은 예측 궤적에서 실행될 물리 시간을 가지며, 디노이징 과정에는 별도의 확산 시간(Diffusion Time)이 존재한다. 모델은 물리적 궤적 시간과 생성 과정의 시간을 구분해야 한다.

이 두 시간 차원은 별도로 인코딩되어야 한다. 물리 시간은 로봇이 행동을 언제 실행할지를 나타낸다. 확산 시간은 반복적인 궤적 정제 과정에서 현재 단계가 어디인지를 나타낸다.

임무 메모리(Mission Memory)는 또 다른 위치 영역을 만든다. 저장된 사건은 시간, 위치, 객체 ID, 작업 단계, 인과 관계에 따라 인덱싱될 수 있다. 로봇은 이전에 방문한 특정 검사 지점의 결과를 검색해야 할 수 있다.

시공간 메모리 인코딩(Spatiotemporal Memory Encoding)은 사건의 지도 위치, 타임스탬프, 임무 단계, 의미 식별 정보를 함께 결합할 수 있다. 검색은 내용과 문맥을 동시에 사용할 수 있다.

작업자가 "이상 진동이 감지되었던 위치로 돌아가라"고 명령할 수 있다. 모델은 사건 설명을 저장된 공간 좌표와 시간 정보에 연결해야 한다. 이를 위해 메모리 토큰은 공간과 시간 이력에 모두 접지되어야 한다.

디지털 트윈(Digital Twin)은 기준 위치 구조(Nominal Positional Structure)를 제공한다. 설비 부품은 설계된 자세와 계층적 관계를 가진다. 실시간 센서 관측은 이러한 저장 위치와 정렬되어야 한다.

실시간 데이터와 기준 데이터의 위치 차이는 이동, 손상, 보정 오류, 오래된 디지털 트윈 정보를 나타낼 수 있다. 모델은 두 값 중 하나를 삭제하지 않고 모두 보존해야 한다.

절대 디지털 트윈 좌표는 계획에 유용하고, 상대 잔차 인코딩(Relative Residual Encoding)은 예상 위치로부터의 차이를 표현한다. 검사 모델은 이러한 잔차에 집중하여 이상을 탐지할 수 있다.

위치 인코딩은 일반화 성능에도 영향을 준다. 모델은 고정 카메라 위치, 지도 방향, 이미지 영역에 과적합될 수 있다. 데이터 증강(Data Augmentation)과 좌표 무작위화(Coordinate Randomization)는 이러한 의존성을 줄일 수 있다.

무작위 이동, 회전, 카메라 자세, 시간 오프셋은 모델이 상대 구조를 학습하도록 유도한다. 그러나 증강은 물리적 의미를 보존해야 한다. 중력이 존재하는 환경에서 임의의 수직 회전은 적절하지 않을 수 있다.

정준화(Canonicalization)도 하나의 전략이다. 객체나 지역 장면을 인코딩하기 전에 표준 좌표계로 변환할 수 있다. 파지 모델은 모든 객체를 추정된 주축(Principal Axis)에 맞추어 표현할 수 있다.

정준화는 학습을 단순화하지만 신뢰할 수 있는 자세 추정에 의존한다. 대칭 객체는 유일한 정준 방향을 가지지 않을 수 있다. 정준화 오류는 제어까지 전달될 수 있다.

아키텍처는 정준 좌표와 원래 좌표를 함께 사용할 수 있다. 정준 좌표는 일반화를 지원하고, 원래 좌표는 실제 실행에 필요한 정보를 보존한다.

위치 인코딩을 학습하려면 다양한 데이터가 필요하다. 로봇이 하나의 시점에서만 작업을 관찰하면 모델은 시점을 작업 정의의 일부로 학습할 수 있다. 의미와 위치를 분리하려면 여러 사이트, 여러 카메라, 여러 체화에서 수집한 데이터가 필요하다.

자기지도 학습(Self-Supervised Learning) 목표는 공간 및 시간 표현을 향상시킬 수 있다. 모델은 프레임 사이의 상대 자세를 예측하고, 마스킹된 좌표를 복원하며, 센서 스트림이 동기화되었는지를 판단하고, 미래 객체 위치를 예측할 수 있다.

대조학습(Contrastive Learning)은 서로 다른 시점에서 관찰된 동일한 객체를 정렬할 수 있다. 같은 물리적 위치를 나타내는 토큰은 이미지 외형이 달라도 유사한 표현을 가져야 한다.

순환 일관성(Cycle Consistency)은 기하 관계를 검증할 수 있다. 카메라 A에서 카메라 B로 변환한 뒤 다시 카메라 A로 돌아오면 원래 위치에 가깝게 복원되어야 한다.

시간 순서 예측(Temporal-Order Prediction)은 모델이 시퀀스 방향을 이해하는지를 시험할 수 있다. 그러나 모션 블러(Motion Blur)와 같은 단순 단서에 의존하는 지름길이 발생할 수 있으므로 과제를 신중히 설계해야 한다.

위치 인코딩 오류는 심각한 물리적 실패를 만들 수 있다. 모델은 올바른 객체를 선택했지만 잘못된 좌표계를 사용할 수 있다. 로봇 기준 왼쪽과 작업자 기준 왼쪽을 혼동하거나, 오래된 장애물 위치를 현재 정보로 처리할 수 있다.

따라서 평가는 의미 인식과 기하학적 접지를 분리해야 한다. 위치 오차, 방향 오차, 좌표계 선택 정확도, 시간 정렬, 궤적 일관성, 폐루프 작업 성공률을 측정해야 한다.

교차 좌표계 시험(Cross-Frame Test)은 카메라 장착 위치, 로봇 자세, 지도 방향을 변경할 수 있다. 실제 문제의 의미가 변하지 않는 좌표 변환 이후에도 모델은 작업 성능을 유지해야 한다.

시간 시험에서는 불규칙 샘플링, 프레임 누락, 지연 관측, 다양한 제어 주기를 포함할 수 있다. 모델은 단순한 시퀀스 인덱스가 아니라 실제 경과 시간에 따라 반응해야 한다.

장기 임무 시험에서는 시간 표현이 장시간 동안 안정적으로 유지되는지를 확인해야 한다. 시계 드리프트(Clock Drift), 누적 위치 추정 오류, 메모리 압축은 위치 문맥을 점차 왜곡할 수 있다.

명시적 동기화와 주기적 공간 앵커링(Spatial Anchoring)은 드리프트를 줄일 수 있다. 로봇은 지도화된 랜드마크, 도킹 스테이션(Docking Station), GNSS, 에이프릴태그(AprilTag), 보정 타깃(Calibration Target), 디지털 트윈 기준점을 사용하여 위치를 다시 확립할 수 있다.

모델은 유효한 지역 문맥을 잃지 않으면서 이러한 보정을 통합해야 한다. 갑작스러운 위치 추정 갱신으로 전역 좌표계가 이동하더라도 지역 객체 관계는 유지될 수 있다.

해석 가능성(Interpretability)도 중요하다. 엔지니어는 어떤 좌표계, 타임스탬프, 상대 관계가 행동에 영향을 주었는지를 확인할 수 있어야 한다. 구조화된 중간 출력에는 목표 자세, 기준 좌표계, 관측의 최신성, 불확실성이 포함될 수 있다.

어텐션 시각화는 궤적 쿼리가 공간적으로 관련된 토큰에 집중하는지를 보여줄 수 있지만 완전한 증거로 간주해서는 안 된다. 반사실적 위치 변경(Counterfactual Position Change)이 더 강한 증거를 제공한다.

다른 모든 입력을 고정한 상태에서 목표 객체의 위치를 변경하면 생성 행동도 일관되게 변해야 한다. 타임스탬프를 변경하여 정보가 오래되었음을 나타내면 모델은 해당 관측에 대한 의존도를 낮추어야 한다.

강건한 위치 인코딩은 정보 누락도 처리해야 한다. 깊이 정보가 없거나, 전역 위치 추정이 실패하거나, 센서 변환이 불확실할 수 있다. 모델은 오해를 일으키는 기본값을 삽입하지 않고 좌표 정보가 없음을 명시적으로 표현해야 한다.

대체 행동(Fallback Behavior)은 상대 이미지 기하, 지역 오도메트리(Local Odometry), 능동적 인식(Active Perception)을 사용할 수 있다. 로봇은 더 나은 시점을 확보하거나 위치 추정을 다시 초기화한 후 작업을 계속할 수 있다.

안전 시스템은 학습된 위치 인코딩에만 의존해서는 안 된다. 충돌 검사(Collision Checking), 관절 한계, 힘 제한, 비상 정지는 검증된 기하학과 실시간 측정값을 사용해야 한다.

파운데이션 모델은 위치 표현을 의미 계획과 행동 생성에 사용할 수 있지만, 결정론적 모듈(Deterministic Module)은 생성된 좌표와 궤적이 실제로 유효한지 검증해야 한다.

지연 시간과 계산 비용도 위치 인코딩 설계에 영향을 준다. 긴 시퀀스나 밀집 포인트 클라우드에서 모든 토큰 쌍의 상대 위치 테이블을 계산하면 비용이 매우 클 수 있다. 분해(Factorization), 희소 이웃(Sparse Neighborhood), 저랭크 편향(Low-Rank Bias), 계층적 처리가 비용을 줄일 수 있다.

정적인 위치 정보는 캐시(Cache)할 수 있다. 로봇 형태, 카메라 보정, 지도 구조, 디지털 트윈 기하 정보는 여러 제어 주기 동안 변하지 않을 수 있다. 동적인 객체 자세와 로봇 상태는 계속 갱신해야 한다.

센서가 재보정되거나, 공구가 변경되거나, 로봇 구성이 수정되면 캐시된 기하 정보를 무효화해야 한다. 위치 정보는 올바른 물리 시스템 버전과 연결되어 있을 때만 의미를 가진다.

엣지 배포(Edge Deployment)에서는 위치 특징의 계산 효율성도 중요하다. 복잡한 등변 연산은 기하학적 이점을 제공하지만 실행 시간과 구현 복잡성을 높일 수 있다. 설계는 작업이 요구하는 정밀도와 지연 시간을 고려해야 한다.

상위 임무 모델은 방 수준의 거친 위치만 필요할 수 있지만 조작 제어기는 정밀한 6자유도 자세를 요구한다. 가장 복잡한 위치 표현을 모든 계층에 동일하게 사용할 필요는 없다.

따라서 계층적 위치 인코딩은 피지컬 AI에서 가장 실용적인 전략 중 하나이다. 전역 위치, 지역 기하, 체화 구조, 궤적 시간, 임무 이력을 분리하면서도 이들 사이의 변환 관계를 유지한다.

전역 수준에서는 건물, 방, 검사 지점, 제한 구역을 표현한다. 지역 수준에서는 장애물, 목표, 로봇 자세를 표현한다. 조작 수준에서는 객체 표면, 파지점, 접촉 좌표계, 공구 자세를 표현한다. 시간 수준에서는 센서 시간, 행동 단계, 임무 진행 상태를 표현한다.

교차 어텐션(Cross-Attention)은 이러한 계층을 연결할 수 있다. 임무 쿼리는 전역 지도 토큰에 주의를 기울이고, 내비게이션 쿼리는 지역 기하 정보에 주의를 기울이며, 파지 쿼리는 객체와 엔드이펙터 자세에 주의를 기울일 수 있다. 복구 쿼리는 과거 사건의 위치와 시간 정보를 검색할 수 있다.

상태공간 모델은 연속적인 은닉 상태 갱신을 통해 시간 위치를 유지할 수 있다. 트랜스포머는 상대 위치 인코딩을 이용하여 유연한 관계 추론을 수행할 수 있다. 그래프 모델은 운동학 및 장면 구조를 보존할 수 있다. 등변 네트워크(Equivariant Network)는 기하 일관성을 유지할 수 있다.

이러한 접근은 상호 보완적이다. 모든 로봇 모달리티와 작업에 적합한 하나의 위치 표현은 존재하지 않는다.

장기적인 목표는 피지컬 AI 모델이 모든 관련 객체가 어디에 있는지, 어떤 방향을 향하는지, 그 추정값이 얼마나 신뢰할 수 있는지, 언제 관측되었는지, 어떻게 이동했는지, 로봇 몸체와 임무에 대해 어떤 관계를 가지는지를 이해할 수 있는 통합 시공간 표현(Unified Spatiotemporal Representation)을 구축하는 것이다.

이러한 표현은 카메라 위치, 로봇 플랫폼, 공장, 작업 시간의 차이를 넘어 일반화되어야 한다. 밀리미터 수준의 지역 정밀도를 유지하면서 건물 규모의 내비게이션과 수시간 규모의 기억을 동시에 지원해야 한다.

이를 위해서는 표준 위치 벡터를 토큰에 더하는 것만으로는 충분하지 않다. 좌표계 규율(Coordinate-Frame Discipline), 보정된 변환, 상대 기하학, 방향 인식 인코딩, 시간 동기화, 불확실성 모델링, 계층형 아키텍처가 모두 필요하다.

따라서 위치 인코딩은 단순한 구현 세부사항이 아니다. 이는 순서가 없는 멀티모달 토큰을 구조화된 물리 세계 표현으로 변환하는 핵심 메커니즘이다.

피지컬 AI 파운데이션 모델에서 공간적·시간적 위치는 의미 지능(Semantic Intelligence)이 실제 물리적 의미를 가질 수 있는지를 결정한다. 모델이 밸브를 인식하고, 검사 명령을 이해하고, 그럴듯한 계획을 생성하더라도 어떤 밸브인지, 어디에 있는지, 로봇이 상대적으로 어디에 위치하는지, 관측이 여전히 최신인지 알지 못하면 신뢰성 있게 행동할 수 없다.

가장 강력한 로봇 아키텍처는 안정된 기준을 위한 절대 좌표, 전이 가능한 관계를 위한 상대 인코딩, 기하 일관성을 위한 등변 구조, 동역학을 위한 시간 인코딩, 운영 신뢰성을 위한 명시적인 좌표계 및 불확실성 정보를 결합할 것이다.

이러한 아키텍처 안에서 위치 인코딩은 인식, 메모리, 체화, 계획, 제어를 연결하는 공통 언어(Common Language)가 된다. 이를 통해 로봇은 자신이 보고 기억하는 정보를 일관된 3차원 및 시간적 세계로 구성하고, 그 안에서 안전하고 목적 지향적인 행동을 생성할 수 있다.

## 2.9 Mixture of Experts MoE for Scalable Robot Model [w/Code]

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

전문가 혼합(Mixture of Experts, MoE)은 모든 계산 과정에서 모든 파라미터(Parameter)를 활성화하지 않고도 모델의 전체 용량(Model Capacity)을 확장하기 위해 설계된 아키텍처 접근법이다. 이러한 특성은 매우 다양한 작업, 센서 모달리티(Sensor Modality), 로봇 몸체(Robot Embodiment), 환경, 제어 요구사항을 처리해야 하는 확장형 로봇 파운데이션 모델에서 특히 중요하다. 하나의 로봇 모델은 언어를 이해하고, 영상을 해석하며, 포인트 클라우드(Point Cloud)를 처리하고, 시간 상태(Temporal State)를 추적하며, 임무를 계획하고, 객체를 조작하고, 시설을 자율 주행하며, 실패를 감지하고, 전문 도구를 조정해야 할 수 있다. 하나의 밀집형 모델(Dense Model)이 이러한 모든 기능을 수행하도록 학습하면 계산 비용이 크게 증가하고 작업 사이의 간섭(Task Interference)이 발생할 수 있다. MoE는 네트워크의 일부를 여러 개의 전문화된 전문가 모듈(Expert Module)로 분리하고, 학습된 라우터(Router)가 각 입력을 처리할 전문가를 선택하도록 하여 이러한 문제를 해결한다.

기존의 밀집형 신경망(Dense Neural Network)에서는 모든 입력이 동일한 파라미터를 통과한다. 트랜스포머(Transformer)에 대형 피드포워드 네트워크(Feed-Forward Network)가 포함되어 있다면, 토큰이 언어, 비전, 힘 측정값, 내비게이션 상태, 로봇 행동 중 무엇을 나타내든 모두 동일한 피드포워드 네트워크를 활성화한다. 파라미터 수를 증가시키면 모델의 용량뿐 아니라 각 토큰을 처리하는 계산량도 함께 증가한다.

MoE 계층은 하나의 밀집 구성요소를 여러 전문가로 대체한다. 각 전문가는 일반적으로 동일한 입력 및 출력 차원을 가지지만 서로 독립적으로 학습되는 파라미터를 가진 신경망 하위 모듈이다. 라우팅 메커니즘(Routing Mechanism)은 각 토큰 또는 특징을 평가한 뒤 하나 이상의 전문가에게 할당한다. 선택된 전문가만 활성화되고 나머지 전문가는 해당 계산에 참여하지 않는다.

이를 통해 희소 조건부 계산(Sparse Conditional Computation)이 가능해진다. 전체 파라미터 수는 크게 증가할 수 있지만, 하나의 토큰을 처리할 때 실제로 활성화되는 계산량은 제한적으로 유지된다. 따라서 로봇 모델은 매 시간 단계마다 모든 기능을 계산하지 않고도 다양한 전문 능력을 포함할 수 있다.

기본적인 MoE 구조는 일반적으로 라우터, 여러 전문가, 전문가 출력 결합 방식으로 구성된다. 라우터는 토큰 표현(Token Representation)을 입력받아 사용 가능한 전문가들에 대한 점수(Score)를 생성한다. 상위 k개 선택(Top-k Selection) 규칙은 가장 높은 점수를 가진 전문가를 선택한다. 선택된 전문가의 출력은 가중 결합되어 최종 결과를 생성한다.

상위 1개 라우팅(Top-1 Routing)을 사용하면 각 토큰은 하나의 전문가만 통과한다. 이는 계산량을 최소화하지만 전문화 결정을 매우 강하게 만든다. 상위 2개 또는 상위 k개 라우팅은 여러 전문가가 함께 기여하도록 하여 유연성을 높이지만 추가적인 계산과 통신 비용을 발생시킨다.

로봇 파운데이션 모델에서는 전문가의 전문화가 여러 차원에서 나타날 수 있다. 전문가는 모달리티, 작업, 체화, 환경, 시간 척도(Temporal Scale), 제어 단계(Control Phase), 안전 기능(Safety Function)에 따라 특화될 수 있다. 하나의 전문가는 시각 객체 이해에 강하고, 다른 전문가는 힘 신호 해석에 강하며, 또 다른 전문가는 이동 로봇 내비게이션이나 조작 계획에 특화될 수 있다.

전문화는 더 미세한 형태로도 나타날 수 있다. 일부 전문가는 정적인 공간 구조를 처리하고, 다른 전문가는 빠른 시간 변화를 처리할 수 있다. 일부는 익숙한 일상 조건을 담당하고, 다른 일부는 이상 상황(Anomaly)이나 불확실한 관측을 처리할 수 있다. 일부 전문가는 일반적인 의미 지식(Semantic Knowledge)을 학습하고, 다른 전문가는 로봇별 동역학(Robot-Specific Dynamics)을 학습할 수 있다.

이러한 표현 및 계산 분할 능력은 피지컬 AI(Physical AI)가 서로 부분적으로 관련되지만 동일하지는 않은 많은 문제를 포함하기 때문에 중요하다. 내비게이션과 조작은 기하학, 장애물 회피, 불확실성, 행동 결과라는 공통 개념을 공유하지만 센서 패턴과 제어 공간은 서로 다르다. 밀집형 모델은 공통 지식과 특화 능력을 동시에 보존하기 어려울 수 있다.

MoE는 하나의 절충안을 제공한다. 공유 계층(Shared Layer)은 언어, 비전, 작업 목표, 물리 문맥에 대한 공통 표현을 학습할 수 있다. 전문가 계층은 서로 다른 운영 영역에 특화될 수 있다. 이를 통해 지식 전이를 유지하면서도 작업 사이의 파괴적 간섭(Destructive Interference)을 줄일 수 있다.

작업 간섭은 범용 로봇 학습(Generalist Robot Learning)의 주요 문제이다. 정교한 조작 능력을 향상시키기 위해 학습한 모델이 이동 내비게이션 성능을 잃을 수 있다. 하나의 로봇 체화에 맞게 미세조정(Fine-Tuning)하면 다른 로봇에서의 행동이 변할 수 있다. 시각적으로 단순한 내비게이션 데이터가 대량으로 존재하면, 양은 적지만 접촉이 풍부하고 복잡한 조작 데이터의 학습을 압도할 수 있다.

전문가는 이러한 업데이트의 일부를 분리할 수 있다. 특정 작업의 데이터는 주로 그 작업에 선택되는 전문가를 학습시키며, 다른 전문가는 상대적으로 적은 영향을 받는다. 이는 파국적 망각(Catastrophic Forgetting)과 부정적 전이(Negative Transfer)를 줄이지만 완전히 제거하지는 않는다.

모달리티별 라우팅(Modality-Specific Routing)은 자연스러운 활용 방식이다. 시각 패치 토큰, 언어 토큰, 고유감각 토큰(Proprioceptive Token), 힘 측정값, 지도 토큰, 행동 토큰을 서로 다른 전문가로 보낼 수 있다. 비전 전문가는 공간적 외형 패턴을 학습하고, 고유감각 전문가는 관절 및 액추에이터 동역학을 학습할 수 있다.

그러나 엄격한 모달리티 분리는 한계를 가진다. 로봇 지능은 교차 모달 관계(Cross-Modal Relationship)에 의존하기 때문이다. 힘 이벤트는 시각적 접촉 지점과 이전 행동을 함께 고려해야 해석할 수 있다. 언어 명령은 기하학과 체화에 연결되어야 한다.

더 나은 설계는 모달리티별 전문가와 공유 멀티모달 전문가를 함께 포함하는 것이다. 일부 전문가는 전문적인 처리 능력을 유지하고, 다른 전문가는 여러 모달리티를 통합한다. 라우터는 토큰이 국부적인 모달리티 처리만 필요한지, 교차 모달 추론이 필요한지를 판단할 수 있다.

예를 들어 일상적인 시각 배경 토큰은 효율적인 비전 전문가가 처리할 수 있다. 작업과 관련된 객체 토큰은 언어, 기하학, 로봇 상태까지 고려하는 멀티모달 전문가로 보낼 수 있다. 이를 통해 비용이 큰 교차 모달 추론을 중요한 정보에만 사용하게 된다.

작업 기반 전문화(Task-Based Specialization)도 가능하다. 전문가는 내비게이션, 파지, 검사, 도킹(Docking), 보행(Locomotion), 인간 상호작용, 복구(Recovery)에 특화될 수 있다. 작업 토큰이나 임무 토큰(Mission Token)이 라우팅에 영향을 주어 현재 목표에 적합한 전문가를 활성화할 수 있다.

이동형 검사 임무 중에는 로봇이 검사 지점 사이를 이동하는 동안 내비게이션 전문가가 주로 사용될 수 있다. 도킹 지점에 접근하면 정밀 정렬 전문가가 더 많이 활성화될 수 있다. 검사 단계에서는 영상 품질과 이상 분석 전문가가 더 많은 토큰을 처리할 수 있다. 실패가 발생하면 복구 전문가가 활성화될 수 있다.

이러한 전문가 사용의 동적 변화는 로봇 운용 과정에서 계산 요구가 계속 달라진다는 사실을 반영한다. 로봇이 빈 복도를 이동할 때 정교한 조작 추론을 수행할 필요는 없으며, 안정적인 접촉 검사 중에는 장거리 내비게이션 추론이 필요하지 않다.

체화별 전문가(Embodiment-Specific Expert)는 교차 로봇 파운데이션 모델에서 특히 중요하다. 서로 다른 로봇은 형태(Morphology), 센서, 액추에이터, 적재 능력, 동역학, 행동 공간이 서로 다르다. 하나의 전문가는 바퀴형 이동 로봇에, 다른 전문가는 매니퓰레이터(Manipulator)에, 또 다른 전문가는 휴머노이드(Humanoid)나 비행 로봇에 특화될 수 있다.

공유 백본(Shared Backbone)은 일반적인 작업과 장면 표현을 학습하고, 체화 전문가는 이러한 표현을 몸체별 추론으로 변환할 수 있다. 로봇 ID, 형태 토큰(Morphology Token), 행동 스키마(Action Schema), 하드웨어 메타데이터가 라우터의 판단 조건으로 사용될 수 있다.

동일한 체화 안에서도 더 세부적인 전문화가 가능하다. 중량 자율이동로봇(Heavy Autonomous Mobile Robot)은 경량 서비스 플랫폼과 제동, 회전, 적재물 거동이 다르다. 라우터는 총질량, 무게중심(Center of Gravity), 공구 구성(Tool Configuration), 운용 모드에 따라 서로 다른 전문가를 선택할 수 있다.

공구 변경도 라우팅에 영향을 줄 수 있다. 그리퍼를 사용하는 매니퓰레이터는 프로브(Probe), 흡착 장치(Suction Tool), 카메라 붐(Camera Boom)을 사용하는 경우와 행동 해석이 다르다. 공구별 전문가는 접촉 특성, 보정 정보, 형상, 성공 조건을 학습할 수 있다.

로봇이 공장, 창고, 병원, 가정, 야외 시설, 농업 환경에서 운용될 경우 환경별 전문화(Environment Specialization)도 나타날 수 있다. 이러한 환경은 시각 패턴, 바닥 재질, 장애물, 인간 행동, 운영 규칙이 서로 다르다.

환경 전문가는 지역적 성능을 향상시킬 수 있지만 지나친 전문화는 일반화를 저하시킬 수 있다. 공장 전문가는 고정된 레이아웃이나 조명에 과적합될 수 있다. 따라서 전이 가능한 물리적·의미적 구조를 학습하는 공유 전문가를 유지해야 한다.

계층형 MoE(Hierarchical MoE)는 전역 지식과 지역 전문성을 균형 있게 결합할 수 있다. 상위 전문가는 내비게이션이나 조작과 같은 큰 작업 영역에 특화되고, 하위 전문가는 특정 로봇, 센서, 환경에 특화될 수 있다.

하나의 토큰은 먼저 넓은 작업 범주로 라우팅되고, 이후 더 세부적인 전문가로 전달될 수 있다. 예를 들어 조작 토큰은 조작 전문가 그룹으로 이동한 뒤 힘 제어, 파지 계획, 공구 사용 전문가로 다시 분류될 수 있다.

계층형 라우팅은 한 번에 비교해야 하는 전문가 수를 줄일 수 있다. 전문화 구조를 더 쉽게 해석할 수 있다는 장점도 있다. 그러나 상위 단계에서 라우팅이 잘못되면 올바른 하위 전문가가 입력을 받지 못할 수 있다.

로봇 모델은 여러 시간 척도에서 동작한다. 고주파 센서 스트림은 모터 및 접촉 동역학을 설명한다. 중간 길이의 시퀀스는 파지, 도킹, 출입문 통과와 같은 스킬을 설명한다. 장기 문맥은 임무 진행, 배터리 관리, 작업 이력을 설명한다.

시간 척도별 전문가(Temporal-Scale Expert)는 이러한 서로 다른 패턴을 처리할 수 있다. 빠른 전문가는 짧은 지역 동역학에 특화되고, 느린 전문가는 장기 문맥을 유지할 수 있다. 상태공간 모델(State Space Model), 순환 모듈(Recurrent Module), 트랜스포머, 시간 합성곱(Temporal Convolution)을 서로 다른 전문가 유형으로 사용할 수 있다.

모든 전문가가 동일한 내부 아키텍처를 가질 필요는 없다. 이질적 MoE(Heterogeneous MoE)는 비전 트랜스포머 전문가, 맘바(Mamba) 기반 시간 전문가, 로봇 형태를 위한 그래프 전문가(Graph Expert), 포인트 클라우드를 위한 기하 전문가(Geometric Expert)를 함께 포함할 수 있다.

이러한 아키텍처 다양성은 서로 다른 데이터 구조에 더 자연스럽게 대응할 수 있다. 언어 및 시각 의미 작업에는 어텐션(Attention)이 적합하고, 연속 텔레메트리에는 상태공간 모델링이 적합하며, 운동학 추론에는 그래프 계산이 적합할 수 있다.

그러나 이질적인 전문가는 라우팅과 시스템 최적화를 복잡하게 만든다. 전문가마다 지연 시간, 메모리 사용량, 출력 특성, 하드웨어 요구조건이 다를 수 있다. 라우터는 의미적 관련성뿐 아니라 실행 비용도 고려해야 한다.

따라서 비용 인식 라우팅(Cost-Aware Routing)은 로보틱스에서 중요하다. 로봇은 제한된 온보드 계산 자원, 에너지, 메모리, 열 용량을 가진다. 가장 정확한 전문가라도 필요한 시간 안에 결과를 제공하지 못한다면 적절한 선택이 아닐 수 있다.

라우터는 지연 시간 예산(Latency Budget), 현재 프로세서 부하, 배터리 수준, 통신 가능 여부, 작업 위험도를 고려할 수 있다. 반복적인 관측은 경량 전문가로 보내고, 복잡하거나 불확실한 상황에서는 대형 전문가를 활성화할 수 있다.

이를 통해 적응형 계산(Adaptive Computation)이 가능해진다. 모델은 작업이 복잡할 때 더 많은 자원을 사용하고 상황이 예측 가능할 때는 적은 자원을 사용한다. 이는 엣지 기반 피지컬 AI에 매우 바람직한 특성이다.

지도화된 빈 복도를 이동하는 로봇은 소형 지역 내비게이션 전문가를 사용할 수 있다. 혼잡한 교차로, 예상하지 못한 장애물, 위치 추정 불일치가 발생하면 대형 멀티모달 추론 전문가를 활성화할 수 있다. 어려운 조작 작업은 일반적인 픽앤플레이스(Pick-and-Place) 작업보다 더 많은 전문가를 사용할 수 있다.

클라우드-엣지 MoE(Cloud-Edge MoE) 아키텍처는 전문가를 서로 다른 계산 위치에 분산할 수 있다. 자주 사용되고 지연 시간에 민감한 전문가는 로봇에 유지하고, 더 큰 전문가는 중요도가 낮은 추론을 위해 엣지 서버나 클라우드에서 실행할 수 있다.

예를 들어 로컬 전문가는 장애물 인식, 이동 문맥, 대체 행동(Fallback Behavior)을 처리할 수 있다. 원격 전문가는 복잡한 기술 문서를 분석하고, 대규모 장면 추론을 수행하거나, 임무 수준 대안을 생성할 수 있다.

라우터는 네트워크 지연과 서비스 가용성을 고려해야 한다. 원격 전문가는 비상 정지나 고주파 제어에 절대로 필수 구성요소가 되어서는 안 된다. 통신이 끊겨도 로봇은 로컬 전문가만으로 안전하게 동작해야 한다.

전문가 배치(Expert Placement)는 능력, 지연 시간, 대역폭, 개인정보 보호, 에너지, 결함 허용성을 함께 고려하는 최적화 문제가 된다. 동일한 논리적 MoE 모델도 로봇과 운용 사이트에 따라 다르게 배포될 수 있다.

희소 계산이 전체 메모리를 자동으로 줄여주는 것은 아니다. 일부 전문가만 활성화되더라도 전체 전문가 파라미터를 저장해야 할 수 있다. 따라서 대형 MoE 모델은 온보드 컴퓨터의 메모리 용량을 초과할 수 있다.

전문가 분할(Expert Partitioning), 양자화(Quantization), 오프로딩(Offloading), 선택적 로딩(Selective Loading)은 이러한 문제를 완화할 수 있다. 자주 사용하는 전문가만 GPU 메모리에 유지하고, 나머지는 느린 메모리나 원격 시스템에 저장할 수 있다.

동적 전문가 로딩(Dynamic Expert Loading)은 추가 지연을 만든다. 시스템은 곧 필요할 가능성이 높은 전문가를 예측하여 미리 불러와야 한다. 임무 문맥과 작업 단계는 이러한 예측에 활용될 수 있다.

이동 로봇이 검사 지점에 접근하면 관련 도킹 및 검사 전문가를 미리 로드할 수 있다. 검사가 완료되면 해당 전문가를 제거하고 내비게이션 전문가를 다시 배치할 수 있다.

전문가 캐싱(Expert Caching)도 하나의 방법이다. 로봇 작업은 시간적 연속성을 가지므로 최근에 사용한 전문가는 다시 필요할 가능성이 높다. 정상 운용에서는 갑작스러운 전문가 전환이 자주 발생하지 않을 수 있다.

라우터는 MoE 시스템의 핵심 구성요소이다. 라우팅이 부정확하면 모델 용량을 낭비하고, 학습을 불안정하게 만들며, 중요한 토큰을 적절하지 않은 전문가에게 전달할 수 있다. 따라서 라우터 설계는 전문가 설계만큼 중요하다.

기본적인 라우터는 선형 투영(Linear Projection)과 전문가별 소프트맥스(Softmax)를 사용할 수 있다. 더 복잡한 라우터는 작업 토큰, 로봇 ID, 모달리티, 신뢰도, 센서 상태, 시간, 시스템 부하를 함께 고려할 수 있다.

문맥 기반 라우팅(Contextual Routing)은 개별 토큰이 아니라 토큰 그룹을 평가할 수 있다. 하나의 이미지 프레임, 센서 패킷, 객체 표현, 행동 청크를 함께 라우팅할 수 있다. 이는 일관성을 유지하고 통신 비용을 줄일 수 있다.

토큰 수준 라우팅(Token-Level Routing)은 유연하지만 관련 정보가 서로 다른 전문가로 분산될 수 있다. 같은 객체의 서로 다른 이미지 패치가 관계없는 전문가에게 보내지면 표현이 분절될 수 있다.

객체 수준 또는 시퀀스 수준 라우팅(Object-Level or Sequence-Level Routing)은 이러한 일관성을 유지할 수 있다. 객체 토큰과 이에 연결된 기하 정보, 언어 참조, 이력을 동일한 전문가 그룹으로 전달할 수 있다.

서로 다른 라우팅 단위가 동시에 사용될 수도 있다. 저수준 시각 토큰은 토큰 수준 라우팅을 사용하고, 작업 및 행동 표현은 시퀀스 수준 라우팅을 사용할 수 있다. 적절한 단위는 모델 계층과 기능에 따라 달라진다.

부하 균형(Load Balancing)은 학습 과정의 주요 과제이다. 별도의 제약이 없으면 라우터가 대부분의 토큰을 소수의 전문가에게 보낼 수 있다. 인기 있는 전문가는 과부하되고, 나머지 전문가는 충분한 데이터를 받지 못해 의미 있는 전문화를 이루지 못한다.

부하 균형 손실(Load-Balancing Loss)은 전문가 사용이 지나치게 일부에 집중되는 것을 방지한다. 전문가 용량 제한(Capacity Limit)을 사용해 각 전문가가 처리할 수 있는 토큰 수를 제한할 수도 있다.

그러나 완전히 균등한 라우팅이 항상 바람직한 것은 아니다. 로봇 데이터 자체가 실제로 불균형할 수 있기 때문이다. 내비게이션 토큰은 드문 실패 복구 토큰보다 훨씬 많을 수 있다. 무조건 균등하게 사용하도록 강제하면 부적절한 전문가에게 데이터가 전달될 수 있다.

목표는 계산 효율성과 의미 있는 전문화 사이의 균형을 찾는 것이다. 드물게 사용되더라도 안전에 중요한 전문가는 일반 운용 중 사용 빈도가 낮더라도 별도의 집중 학습이 필요하다.

전문가 용량은 하나의 배치에서 전문가가 처리할 수 있는 토큰 수를 결정한다. 용량이 초과되면 추가 토큰을 버리거나, 다른 전문가로 재라우팅하거나, 대체 전문가가 처리할 수 있다. 피지컬 AI에서는 중요한 센서나 안전 정보가 버려질 수 있으므로 토큰 드롭(Token Dropping)이 특히 위험하다.

중요한 토큰은 보호된 라우팅(Protected Routing)을 사용해야 한다. 비상 신호, 충돌 표시, 임무 제약, 높은 불확실성 이벤트는 전문가 과부하 때문에 제거되어서는 안 된다.

우선순위 기반 라우팅(Priority-Aware Routing)은 중요한 토큰을 위한 용량을 미리 확보할 수 있다. 중복된 저우선순위 시각 토큰은 압축하거나 지연 처리하고, 안전 관련 토큰은 즉시 처리할 수 있다.

공유 밀집 대체 경로(Shared Dense Fallback Path)도 유용하다. 라우팅이 실패하거나 전문가 용량이 부족한 경우에도 토큰이 범용 네트워크를 통과할 수 있다. 이는 희소 라우팅 실패로 인한 치명적인 결과를 줄인다.

라우터 자체도 학습 중 불안정해질 수 있다. 표현이 조금만 변해도 토큰이 갑자기 다른 전문가로 이동할 수 있다. 이는 최적화를 불안정하게 만들고 행동 일관성을 저하시킬 수 있다.

라우터 정규화(Router Regularization), 노이즈 주입(Noise Injection), 온도 조절(Temperature Control), 전문가 드롭아웃(Expert Dropout), 점진적 전문화(Gradual Specialization)는 안정성을 향상시킬 수 있다. 일부 시스템은 초기에는 균등한 라우팅으로 학습한 뒤 점차 강한 전문화를 허용한다.

전문가 초기화도 중요하다. 모든 전문가가 동일하게 초기화되고 유사한 데이터를 받으면 서로 중복된 기능을 학습할 수 있다. 무작위 초기화는 다양성을 만들지만 전문화 방향이 임의적으로 결정될 수 있다.

전문가를 이미 역할이 알려진 사전학습 모듈로 초기화할 수도 있다. 비전 전문가는 비전 파운데이션 모델에서 시작하고, 시간 전문가는 시퀀스 모델에서 시작할 수 있다. 이를 통해 라우터는 학습 초기부터 의미 있는 선택지를 가질 수 있다.

반대로 모든 전문가를 공통 초기화에서 시작한 뒤 데이터에 따라 전문화시킬 수도 있다. 이 방식은 전문가 간 호환성을 높일 수 있지만 라우팅 목표를 세심하게 설계해야 한다.

전문가 다양성(Expert Diversity)은 당연히 생긴다고 가정해서는 안 되며 실제로 측정해야 한다. 두 전문가는 서로 다른 토큰을 받더라도 거의 동일한 기능을 학습할 수 있다. 표현 유사도, 출력 행동, 작업 성능, 라우팅 패턴을 분석하여 진정한 전문화가 발생했는지 확인해야 한다.

일부 중복성은 결함 허용성(Fault Tolerance)을 위해 유용하다. 하나의 전문가만 중요 상황을 처리할 수 있다면 해당 전문가의 실패가 단일 실패 지점(Single Point of Failure)이 된다. 부분적으로 겹치는 여러 전문가는 대체 경로를 제공할 수 있다.

중복 전문가 라우팅은 불확실성 추정에도 활용할 수 있다. 여러 전문가가 동일한 중요 토큰을 독립적으로 처리하고 출력 차이가 클 경우 분포 변화(Distribution Shift)나 불확실성을 나타낼 수 있다.

이는 앙상블(Ensemble)과 유사하지만 MoE 전문가는 하나의 모델 안에서 공동으로 학습되므로 완전히 독립적이지 않다. 따라서 라우터 신뢰도와 전문가 불일치는 신중히 해석해야 한다.

안전에 중요한 결정에서는 상위 1개 라우팅보다 여러 전문가를 사용하는 상위 k개 라우팅이 적합할 수 있다. 내비게이션 결정은 기하, 의미, 안전 전문가의 결과를 함께 결합할 수 있다. 이들이 크게 불일치하면 로봇이 감속하거나 추가 센싱을 요청할 수 있다.

MoE는 불확실성 기반 에스컬레이션(Uncertainty-Aware Escalation)도 지원할 수 있다. 경량 전문가는 먼저 일상적인 입력을 처리하고, 신뢰도가 낮으면 추가 전문가나 더 큰 추론 모델을 활성화한다.

이러한 단계형 계산(Staged Computation)은 평균 비용을 줄이면서 어려운 상황에 필요한 모델 용량을 유지한다. 이는 의료 분류나 계층형 로봇 안전 아키텍처와 유사하다.

첫 단계는 상황이 익숙하고 위험이 낮은지를 판단할 수 있다. 이후 단계에서는 더 깊은 멀티모달 추론, 월드 모델 시뮬레이션(World-Model Simulation), 인간 지원을 수행할 수 있다.

라우터는 모델 신뢰도만을 맹목적으로 사용해서는 안 된다. 신경망 신뢰도는 잘못 보정될 수 있다. 센서 품질, 분포 외 탐지(Out-of-Distribution Detection), 전문가 불일치, 물리적 제약 검사를 함께 고려해야 한다.

전문가 전문화는 도구 사용(Tool Use)과도 연결할 수 있다. 하나의 전문가는 모션 플래너(Motion Planner) 호출에, 다른 전문가는 디지털 트윈 질의에, 또 다른 전문가는 검사 알고리즘이나 유지보수 데이터베이스 사용에 특화될 수 있다.

상위 라우터는 현재 하위 문제에 적합한 내부 추론 전문가와 외부 도구를 선택한다. 파운데이션 모델은 내부 전문가와 외부 엔지니어링 서비스의 오케스트레이터(Orchestrator)가 된다.

예를 들어 의미 전문가는 작업자 명령을 해석할 수 있다. 지도 전문가는 목적지를 접지하고, 계획 전문가는 내비게이션 플래너를 호출하며, 모니터링 전문가는 임무 진행을 평가하고, 보고서 전문가는 결과를 요약할 수 있다.

이러한 아키텍처는 모듈형 로봇 소프트웨어와 유사하지만 학습 기반 라우팅과 공유 표현을 사용한다. 모듈성의 장점을 유지하면서 유연한 조정을 지원할 수 있다.

그러나 학습 기반 라우팅이 명시적 권한 경계를 대체해서는 안 된다. 라우터가 높은 점수를 부여했다고 해서 모델 전문가가 고위험 액추에이터 인터페이스를 호출할 수 있어서는 안 된다. 도구 권한과 물리적 권한은 독립적으로 제한되어야 한다.

MoE는 멀티모달 융합에도 활용할 수 있다. 하나의 공유 피드포워드 네트워크가 모든 융합 토큰을 처리하는 대신, 전문가는 시각-언어 정렬, 비전-기하 융합, 상태-행동 추론, 힘-접촉 해석에 특화될 수 있다.

교차 어텐션(Cross-Attention) 계층이 융합 표현을 생성하고, MoE 계층은 그 내용에 따라 서로 다른 전문가가 이를 처리하도록 할 수 있다. 객체 관련 토큰은 의미-기하 전문가를 활성화하고, 행동 토큰은 체화-제어 전문가를 활성화할 수 있다.

반대 구조도 가능하다. 모달리티별 전문가가 먼저 입력을 처리하고, 이후 교차 어텐션이 이들의 출력을 연결할 수 있다. 전문화가 멀티모달 통합 이전에 발생하는지 이후에 발생하는지는 아키텍처 설계에 따라 달라진다.

초기 전문가 라우팅은 원시 데이터를 조기에 필터링하여 계산을 줄일 수 있다. 후기 전문가 라우팅은 더 풍부한 문맥을 이용해 적절한 전문가를 선택할 수 있다. 하이브리드 시스템은 여러 계층에서 라우팅을 사용할 수 있다.

MoE는 비전-언어-행동 모델(Vision-Language-Action Model)과도 결합할 수 있다. 비전-언어 백본은 MoE 피드포워드 계층을 포함하고, 행동 디코더는 작업 또는 체화별 전문가를 가질 수 있다.

언어와 비전 토큰은 일반 의미 전문가를 활성화하고, 행동 토큰은 조작, 내비게이션, 보행 전문가를 활성화할 수 있다. 로봇 ID와 행동 스키마는 라우팅 판단의 조건이 된다.

행동 생성에서는 전문가 전환이 시간적으로 일관되어야 한다. 연속된 행동 토큰이 서로 매우 다른 전문가로 이동하면 궤적이 불연속적이거나 불안정해질 수 있다.

시퀀스 일관성 손실(Sequence-Consistency Loss)은 인접한 행동 단계가 호환 가능한 전문가를 사용하도록 유도할 수 있다. 행동 청크 전체를 독립된 토큰이 아니라 하나의 단위로 라우팅할 수도 있다.

게이팅 일정(Gating Schedule)은 하나의 스킬 동안 전문가 전환을 제한할 수 있다. 로봇이 파지나 도킹 동작을 시작하면 의미 있는 작업 단계 전환이 발생하기 전까지 동일한 전문가 집합을 유지할 수 있다.

이는 각 전문가가 하나의 제어 모드(Control Mode)를 나타내는 하이브리드 동적 시스템(Hybrid Dynamical System)과 유사하다. 라우터는 모드 전환을 결정한다. 이러한 전환은 안정적이고 물리적으로 의미 있어야 한다.

따라서 MoE는 전통적인 행동 트리(Behavior Tree), 스킬 라이브러리(Skill Library), 유한상태기계(Finite-State Machine), 이득 스케줄 제어(Gain-Scheduled Control)와 개념적으로 연결된다. 차이점은 라우팅 경계와 전문가 표현이 완전히 프로그래밍되지 않고 데이터로부터 학습된다는 것이다.

학습 전문가와 명시적 스킬 상태를 결합하면 신뢰성을 높일 수 있다. 임무 시스템은 허용된 스킬 단계를 정의하고, 라우터는 해당 단계에서 유효한 전문가 가운데 선택한다.

예를 들어 최종 도킹 정렬 단계에서는 도킹 관련 전문가만 사용할 수 있도록 제한할 수 있다. 이는 관계없는 전문가가 충돌하는 행동을 생성하는 것을 방지한다.

전문가 라우팅은 물리적 제약도 고려할 수 있다. 현재 로봇, 공구, 적재물, 제어 주기를 지원하지 않는 전문가는 선택 후보에서 제외할 수 있다.

능력 메타데이터(Capability Metadata)는 각 전문가의 유효 범위를 설명할 수 있다. 라우터가 전문가를 제안하고, 결정론적 호환성 계층(Deterministic Compatibility Layer)이 실제 사용 가능 여부를 검증한다.

학습 데이터의 품질은 전문가 형성에 큰 영향을 준다. 작업 레이블, 체화 메타데이터, 센서 ID, 실패 상태가 누락되면 라우터는 우연한 상관관계를 기준으로 전문화할 수 있다.

전문가가 하나의 물리적 스킬이 아니라 특정 카메라 시점에 전문화될 수 있다. 다른 전문가는 로봇 행동이 아니라 데이터셋 출처에 특화될 수 있다. 이러한 전문화는 일반화 성능을 떨어뜨린다.

따라서 균형 잡히고 잘 문서화된 데이터셋이 필수적이다. 메타데이터에는 로봇 유형, 행동 스키마, 작업, 환경, 공구, 적재물, 센서 구성, 제어 주기, 결과가 포함되어야 한다.

전문가 역할을 알고 있는 경우 라우팅 감독(Routing Supervision)을 사용할 수 있다. 조작 토큰이 조작 전문가를 사용하도록 유도할 수 있다. 그러나 지나친 감독은 모델이 유용한 공유 구조를 스스로 발견하는 것을 방해할 수 있다.

약한 감독(Weak Supervision)은 넓은 전문가 그룹만 지정하고 더 세부적인 전문화는 데이터에서 나타나도록 할 수 있다. 일부 전문가는 범용으로 유지하고, 다른 전문가는 특정 영역으로 유도할 수 있다.

자기지도 사전학습(Self-Supervised Pretraining)에도 MoE를 사용할 수 있다. 전문가는 서로 다른 센서 채널, 시간 척도, 마스킹 영역, 미래 상태 예측에 특화될 수 있다.

시각 전문가는 외형을 복원하고, 기하 전문가는 깊이를 예측하며, 동역학 전문가는 미래 상태를 예측하고, 의미 전문가는 언어와 이미지를 정렬할 수 있다. 행동 학습 이전에 공유 표현을 구축할 수 있다.

행동 미세조정 단계에서는 이러한 전문화를 다시 활용할 수 있다. 그러나 표현 학습과 제어 학습의 요구사항이 다르기 때문에 전문가 적응이 필요할 수 있다.

지속학습(Continual Learning)에서도 MoE는 유용할 수 있다. 기존 전문가를 유지하면서 새로운 로봇, 작업, 환경을 위한 새 전문가를 추가할 수 있다. 이를 통해 기존 능력에 대한 간섭을 줄일 수 있다.

새로운 검사 공구는 전용 어댑터(Adapter)나 전문가를 추가할 수 있다. 라우터는 해당 공구를 사용할 때 이 전문가를 선택하며, 공유 백본과 기존 전문가는 대부분 유지된다.

확장형 MoE 아키텍처는 장기적인 플릿(Fleet) 발전을 지원할 수 있다. 새로운 로봇 유형이 추가될 때 전체 시스템을 다시 학습하기보다 전문화된 전문가를 추가하여 모델을 확장할 수 있다.

그러나 통제되지 않은 확장은 복잡성을 증가시킨다. 전문가가 지나치게 많아지면 저장 공간, 라우팅 비용, 검증 부담, 배포 복잡성이 커진다. 따라서 전문가 가지치기(Expert Pruning), 병합(Merging), 폐기(Retirement)가 필요하다.

유사한 행동을 하는 전문가를 병합할 수 있다. 거의 사용되지 않거나 오래된 전문가는 제거할 수 있다. 전문화가 안정된 이후 새로운 전문가를 더 작은 모듈로 증류(Distillation)할 수도 있다.

수명주기 관리(Lifecycle Management)는 각 전문가를 버전이 관리되는 구성요소로 취급해야 한다. 하나의 전문가 변경은 일부 작업에만 영향을 줄 수 있지만, 라우터 행동 변화는 훨씬 넓은 영향을 만들 수 있다. 회귀 시험(Regression Testing)은 전문가 성능과 라우팅 결정 모두를 포함해야 한다.

새로운 전문가는 이전에 다른 전문가가 안전하게 처리하던 토큰을 예상하지 못하게 가져갈 수 있다. 새 전문가가 목표 작업에서는 우수하더라도 전문가 경쟁 구조를 바꾸어 관계없는 행동을 저하시킬 수 있다.

따라서 라우터와 전문가 버전은 함께 기록해야 한다. 배포 로그는 어떤 전문가가 중요 토큰을 처리하고 어떤 전문가가 행동을 생성했는지를 추적할 수 있어야 한다.

해석 가능성(Interpretability)은 MoE의 잠재적인 장점이다. 전문가 사용 패턴은 모델 행동에 대한 거친 설명을 제공할 수 있다. 엔지니어는 이동 중 내비게이션 전문가가 활성화되고, 영상 취득 중 검사 전문가가 활성화되는지를 확인할 수 있다.

이는 완전한 밀집형 네트워크보다 이해하기 쉽지만 전문가 이름이 실제 기능을 보장하는 것은 아니다. 내비게이션 전문가로 이름 붙인 모듈이 실제로는 데이터셋 특유의 시각 패턴만 학습했을 수도 있다.

따라서 행동 프로빙(Behavioral Probing)이 필요하다. 통제된 입력에서 전문가를 시험하여 실제로 어떤 정보와 작업을 처리하는지 확인해야 한다.

라우팅 시각화(Routing Visualization)는 어떤 전문가가 모달리티, 객체, 행동, 시간 구간을 처리하는지 보여줄 수 있다. 갑작스러운 전문가 전환은 불확실성이나 불안정을 나타낼 수 있다.

전문가 수준 제거 실험(Expert-Level Ablation)은 각 전문가의 중요성을 평가할 수 있다. 하나의 전문가를 제거한 뒤 성능 변화를 관찰하면 전문화가 실제로 의미 있는지, 대체 경로가 존재하는지를 확인할 수 있다.

MoE는 새로운 실패 양상도 만든다. 라우터가 중요한 데이터를 잘못된 전문가에게 보낼 수 있다. 전문가가 과부하될 수 있다. 장치 간 통신이 끊길 수 있다. 전문가 출력이 서로 불일치할 수 있다. 악의적인 입력이 라우팅을 조작할 수도 있다.

안전 분석(Safety Analysis)은 이러한 가능성을 포함해야 한다. 시스템은 라우터 불확실성, 전문가 장애, 용량 초과, 비정상적인 전문가 전환을 감지할 수 있어야 한다.

안전한 대체 모델(Safe Fallback Model)은 MoE 시스템이 실패했을 때 최소 기능을 유지할 수 있다. 이 모델은 능력은 더 낮더라도 충분히 검증되어 있어야 한다.

이동형 로봇의 대체 행동은 정지, 알려진 안전 위치로 복귀, 기존 내비게이션 스택 사용이 될 수 있다. 조작 로봇은 힘을 안전하게 해제하거나 공구를 후퇴시킬 수 있다.

MoE는 충분한 검증 없이 하드 실시간 제어(Hard Real-Time Control) 안에 직접 배치해서는 안 된다. 희소 전문가 디스패치(Sparse Expert Dispatch)는 특히 서로 다른 하드웨어에서 전문가가 실행될 경우 지연 시간을 변동시킬 수 있다.

고주파 모터 제어 루프는 예측 가능한 실행 시간을 요구한다. MoE는 인식, 의미 추론, 스킬 선택, 중기 행동 생성, 임무 계획에 더 자연스럽게 적합하다.

시간에 민감한 행동 정책에서 MoE를 사용한다면 전문가 배치와 라우팅이 제한된 최악 지연 시간을 보장해야 한다. 평균 처리량보다 최악의 실행 경로가 더 중요하다.

통신 오버헤드(Communication Overhead)는 분산 MoE 학습과 추론에서 큰 문제이다. 토큰은 선택된 전문가에게 전송되어야 하며, 전문가는 서로 다른 장치에 존재할 수 있다. 이러한 전체 대 전체 통신(All-to-All Communication)이 주요 병목이 될 수 있다.

데이터센터에서는 고속 인터커넥트(High-Speed Interconnect)가 분산 전문가를 지원할 수 있다. 그러나 로봇에서는 통신 자원이 훨씬 제한적이다. 전문가 병렬화(Expert Parallelism)는 실제 하드웨어 토폴로지에 맞게 설계해야 한다.

지역 그룹화(Local Grouping)는 통신 비용을 줄일 수 있다. 자주 협력하는 전문가는 동일한 가속기에 배치할 수 있다. 임무 수준 전문가와 인식 전문가는 서로 다른 프로세서에서 실행할 수 있다.

토큰 배칭(Token Batching)은 학습 중 처리량을 높이지만, 실시간 추론에서는 배치 크기가 작기 때문에 같은 이점을 얻기 어렵다. 대규모 학습 배치에서 측정된 MoE 효율이 로봇 배포에서도 그대로 유지된다고 가정해서는 안 된다.

따라서 엣지 지향 MoE(Edge-Oriented MoE)는 현실적인 배치 크기, 시퀀스 길이, 센서 주기, 제어 마감 시간을 기준으로 평가해야 한다. 라우팅 오버헤드가 크면 희소 모델이 밀집 모델보다 느릴 수도 있다.

양자화는 전문가 메모리와 계산량을 줄일 수 있다. 민감도에 따라 전문가별로 서로 다른 정밀도를 사용할 수도 있다. 의미 전문가는 낮은 정밀도를 허용할 수 있지만, 기하 또는 행동 전문가는 더 높은 정밀도를 요구할 수 있다.

전문가별 압축(Expert-Specific Compression)도 장점이 될 수 있다. 자주 사용하는 전문가에 더 많은 최적화 자원을 투자할 수 있다. 드문 전문가는 압축된 형태로 저장하고 필요할 때만 활성화할 수 있다.

지식 증류(Knowledge Distillation)는 대형 MoE 교사 모델을 배포용 소형 밀집 모델이나 희소 학생 모델로 변환할 수 있다. 교사 모델은 학습 중 광범위한 능력을 제공하고, 학생 모델은 특정 로봇에 맞게 최적화된다.

또는 개별 전문가를 로봇별 정책으로 증류할 수 있다. 대형 범용 MoE 모델이 특화된 엣지 모델의 지식 원천이 될 수 있다.

이는 실용적인 개발 절차를 만든다. 대형 모델에서 광범위한 교차 체화 학습을 수행하고, 실제 제품에는 해당 로봇과 작업에 필요한 전문가나 증류된 행동만 배포할 수 있다.

MoE 평가는 단순한 종합 정확도를 넘어야 한다. 전문가 사용률, 라우팅 안정성, 부하 균형, 지연 시간, 메모리, 에너지, 전문화, 대체 행동, 폐루프 작업 성능을 함께 평가해야 한다.

작업별 평가는 어떤 전문가가 사용되는지, 작업 단계에 따라 전문가 선택이 적절히 변하는지를 확인해야 한다. 교차 체화 시험에서는 라우팅이 로봇 ID와 물리적 제약을 존중하는지 검증해야 한다.

강건성 시험(Robustness Test)은 모달리티 손상, 환경 변화, 적재물 변경, 센서 지연, 새로운 작업을 포함해야 한다. 라우터는 부적절한 전문화를 자신 있게 선택하기보다 불확실성을 높이거나 범용 전문가를 활성화해야 한다.

전문가 과부하도 직접 시험해야 한다. 많은 입력이 동시에 하나의 전문가를 요구할 때도 중요 토큰을 보존하고 안전한 행동을 유지해야 한다.

장기 평가는 라우팅 드리프트(Routing Drift)를 확인해야 한다. 모델은 임무 초기에는 적절한 전문가를 사용하지만 내부 표현이 변하면서 점차 다른 전문가로 이동할 수 있다. 전문가 사용은 실제 운영 문맥과 일관되어야 한다.

폐루프 시험(Closed-Loop Testing)은 필수적이다. 라우팅 결정은 행동에 영향을 주고, 행동은 이후 관측과 다음 라우팅에 영향을 준다. 작은 오류가 피드백 루프(Feedback Loop)를 만들 수 있다.

시뮬레이션은 많은 작업과 실패 조합을 생성할 수 있지만, 실제 로봇 시험도 필요하다. 실행 지연, 센서 노이즈, 통신, 물리적 동역학이 전문가 선택에 영향을 주기 때문이다.

MoE는 플릿 수준 전문화(Fleet-Level Specialization)도 지원할 수 있다. 전문가는 서로 다른 현장이나 로봇 그룹의 데이터로 학습될 수 있다. 전역 라우터(Global Router)는 공유 전문가 또는 사이트별 전문가를 선택한다.

이를 통해 모델을 완전히 분리하지 않고 지역 적응(Local Adaptation)을 수행할 수 있다. 공통 지식은 공유 전문가에 유지하고, 사이트 전문가는 지역 레이아웃, 용어, 장비, 교통 패턴을 학습할 수 있다.

개인정보 보호형 학습(Privacy-Preserving Training)은 특정 전문가를 로컬에 유지할 수 있다. 기업은 일반 백본을 공유하면서 기밀 공장 데이터로 비공개 전문가를 학습할 수 있다.

연합학습(Federated Learning)이나 분산학습을 통해 모든 원시 데이터를 중앙에 모으지 않고도 전문가를 업데이트할 수 있다. 그러나 라우팅 호환성과 모델 일관성이 어려운 문제가 된다.

사이트별 전문가는 지역적으로는 잘 동작하더라도 업데이트된 전역 라우터와 결합되면 예상하지 못한 행동을 만들 수 있다. 버전 관리와 검증이 필수적이다.

MoE와 모듈형 로보틱스(Modular Robotics)의 관계는 전략적으로 중요하다. 전통적인 로봇 시스템은 인식, 계획, 제어, 안전, 임무 관리를 명확한 모듈로 분리한다. 엔드투엔드(End-to-End) 파운데이션 모델은 이러한 경계를 가로지르는 넓은 매핑을 학습하려 한다.

MoE는 두 방식 사이의 중간 경로를 제공한다. 하나의 공유 학습 모델을 유지하면서 내부 전문화를 허용한다. 그러나 학습된 전문가는 인증된 소프트웨어 모듈과 동일하게 취급해서는 안 된다. 전문화 경계와 행동은 확률적이며 데이터에 따라 변한다.

가장 신뢰성 높은 구조는 학습 기반 MoE 추론과 명시적 시스템 모듈을 결합하는 방식일 수 있다. 전문가는 장면을 해석하고, 스킬을 선택하고, 결과를 예측하고, 후보 행동을 생성한다. 결정론적 플래너, 제어기, 안전 모니터는 이를 검증하고 실행한다.

MoE 내비게이션 전문가는 경로 전략을 제안할 수 있지만 모션 플래너가 충돌 기하를 검사한다. 조작 전문가는 파지를 제안할 수 있지만 제어기는 힘과 관절 한계를 강제한다. 복구 전문가는 재시도를 제안할 수 있지만 임무 관리자가 자율 복구 허용 여부를 결정한다.

이러한 계층형 아키텍처에서 MoE는 제한 없는 물리적 권한을 받지 않으면서 인지 용량(Cognitive Capacity)을 확장한다.

피지컬 AI에서 MoE의 장기적인 가능성은 확장 가능한 전문화(Scalable Specialization)에 있다. 로봇 파운데이션 모델은 모든 계산에서 전체 모델을 사용하지 않고도 더 많은 작업, 체화, 모달리티, 환경을 포함하도록 성장할 수 있다.

언어 명령 이해에는 언어 전문가를, 장면 해석에는 비전 전문가를, 공간 추론에는 기하 전문가를, 긴 시퀀스에는 시간 전문가를, 로봇별 제약에는 체화 전문가를, 개별 스킬에는 행동 전문가를 활성화할 수 있다.

라우터는 어떤 센서, 작업, 체화, 시점에 어떤 종류의 지능을 적용할지를 결정하는 학습 기반 계산 정책(Learned Computational Policy)이 된다. 이러한 판단은 입력 내용, 작업, 불확실성, 하드웨어, 운영 위험에 따라 달라질 수 있다.

핵심 과제는 전문화가 안정적이고, 해석 가능하며, 효율적이고, 안전하게 유지되도록 하는 것이다. 희소 활성화만으로 로보틱스 성능이 자동으로 향상되는 것은 아니다. 부정확한 라우팅, 전문가 불균형, 불안정한 지연 시간, 부족한 메타데이터, 안전하지 않은 대체 행동은 모델 용량 증가의 이점을 상쇄할 수 있다.

성공적인 MoE 로봇 모델은 신중하게 설계된 라우팅, 보호된 안전 경로, 균형 잡힌 학습 데이터, 전문가 수명주기 관리, 하드웨어 인식 배포(Hardware-Aware Deployment), 광범위한 폐루프 검증을 필요로 한다.

또한 학습 기반 조건부 계산과 결정론적 물리 제약을 명확히 분리해야 한다. 전문가는 추천하고, 예측하고, 계획할 수 있지만 안전 시스템이 최종 권한을 유지해야 한다.

확장 가능한 로봇 파운데이션 모델에서 전문가 혼합은 광범위한 공유 지능과 전문적인 물리 능력을 결합하는 강력한 방법을 제공한다. 하나의 모델 계열이 계산량을 통제하면서 여러 로봇과 작업을 지원하도록 만들 수 있다.

그 가장 큰 가치는 단순히 파라미터 수를 늘리는 데 있지 않을 수 있다. 올바른 종류의 추론을 올바른 센서, 작업, 체화, 순간에 할당하는 능력을 피지컬 AI 시스템에 제공하는 데 있을 수 있다.

계층형 로봇 아키텍처 안에서 MoE는 일반 지식과 전문 행동을 연결하는 메커니즘이 될 수 있다. 공유 표현은 지식 전이를 제공하고, 전문가는 깊이 있는 능력을 제공하며, 라우팅은 적응성을 제공하고, 결정론적 시스템은 운영 신뢰성을 제공한다.

이러한 역할을 통해 전문가 혼합은 로봇 파운데이션 모델이 단순히 더 커지는 방식이 아니라, 더 체계적이고, 선택적이며, 자원 인식적(Resource-Aware)이고, 지능형 로봇이 마주하는 다양한 물리 조건에 더 잘 대응하는 방향으로 확장되는 미래를 지원할 수 있다.

## 2.10 Transformer Inference Optimization KV Cache Quant [w/Code]

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

트랜스포머 추론 최적화(Transformer Inference Optimization)는 피지컬 AI 시스템에 파운데이션 모델(Foundation Model)을 배포하기 위해 필수적이다. 대형 트랜스포머 모델은 언어 이해(Language Understanding), 멀티모달 인식(Multimodal Perception), 작업 계획(Task Planning), 행동 생성(Action Generation), 로봇 조정(Robot Coordination)에서 높은 성능을 제공할 수 있지만, 계산량과 메모리 요구사항이 로봇 탑재 컴퓨터의 가용 자원을 초과할 수 있다. 데이터센터 GPU에서는 잘 동작하는 모델도 자율이동로봇(Autonomous Mobile Robot), 매니퓰레이터(Manipulator), 드론(Drone), 휴머노이드(Humanoid)에 배포하면 지나치게 느리거나, 전력을 많이 소비하거나, 메모리 사용량이 너무 커질 수 있다. 따라서 추론 최적화는 트랜스포머가 연구용 시제품에 머무를지, 실제 로봇 시스템의 실용적 구성요소가 될지를 결정한다.

피지컬 AI는 많은 클라우드 기반 애플리케이션보다 더 엄격한 요구조건을 가진다. 로봇은 정해진 시간 안에 반응해야 하고, 제한된 전력으로 동작해야 하며, 열을 관리해야 하고, 네트워크가 끊겨도 계속 작동해야 하며, 위치 추정(Localization), 인식(Perception), 계획(Planning), 제어(Control), 로깅(Logging), 안전 프로세스와 계산 자원을 공유해야 한다. 평균 처리량(Throughput)만으로는 충분하지 않다. 컨텍스트 길이가 증가하거나, 여러 센서 스트림이 동시에 입력되거나, 계산량이 많은 상황에 진입하더라도 예측 가능한 지연 시간(Latency)을 유지해야 한다.

트랜스포머 추론은 여러 계산 단계로 구성된다. 입력 데이터는 토큰(Token)이나 임베딩(Embedding)으로 변환되고, 어텐션(Attention) 및 피드포워드 계층(Feed-Forward Layer)을 통과한 뒤, 최종적으로 언어, 계획, 행동, 궤적(Trajectory), 구조화된 출력(Structured Output)으로 디코딩된다. 자기회귀 모델(Autoregressive Model)에서는 토큰이 하나씩 순차적으로 생성된다. 새 토큰은 이전에 처리된 전체 문맥에 의존한다. 최적화가 없다면 모델은 매 생성 단계마다 이전 토큰의 표현을 모두 다시 계산해야 한다.

키-값 캐시(Key-Value Cache), 일반적으로 KV 캐시(KV Cache)라고 불리는 메커니즘은 이러한 반복 계산을 피하기 위해 도입되었다. 트랜스포머 어텐션 계층에서 각 토큰은 쿼리(Query), 키(Key), 값(Value) 벡터를 생성한다. 자기회귀 생성 과정에서는 이전 토큰의 키와 값이 변하지 않는다. 따라서 이를 저장해 두었다가 다음 토큰을 생성할 때 재사용할 수 있다.

새로운 디코딩 단계마다 모델은 가장 최근 토큰의 쿼리, 키, 값을 계산한다. 새로운 쿼리는 이전에 저장된 모든 키에 어텐션을 적용하고, 대응하는 값들을 결합하여 어텐션 출력을 생성한다. 이후 새 키와 값이 캐시에 추가된다. 이를 통해 반복 계산량이 크게 줄어들고 점진적 생성(Incremental Generation)이 훨씬 빨라진다.

KV 캐시는 대형 자기회귀 모델이 실용적인 속도로 출력을 생성할 수 있게 하는 주요 요소 중 하나이다. 그러나 상당한 메모리 비용을 발생시킨다. 캐시 크기는 트랜스포머 계층 수, 어텐션 헤드(Attention Head) 수, 은닉 차원(Hidden Dimension), 배치 크기(Batch Size), 컨텍스트 길이에 따라 증가한다. 대형 모델이 긴 로봇 임무 이력을 처리하면 모델 가중치가 이미 압축되어 있더라도 KV 캐시만으로 수 기가바이트의 메모리가 필요할 수 있다.

이는 피지컬 AI에서 중요한 상충관계(Trade-Off)를 만든다. 긴 컨텍스트는 모델이 명령, 최근 관측, 작업 진행 상태, 행동 이력, 감지된 객체, 실패 정보를 기억하도록 한다. 그러나 모든 이전 토큰을 KV 캐시에 유지하면 메모리 사용량이 증가하고 각 디코딩 단계의 어텐션 계산량도 커진다.

따라서 로봇 파운데이션 모델은 모든 원시 관측을 하나의 지속적으로 증가하는 자기회귀 컨텍스트에 넣어서는 안 된다. 고주파 카메라 프레임, LiDAR 특징, 관절 상태, 텔레메트리(Telemetry)는 매우 긴 토큰 시퀀스를 생성할 수 있다. 아무리 최적화된 KV 캐시라도 무한히 확장될 수는 없다.

계층형 메모리(Hierarchical Memory)가 더 실용적이다. 활성 트랜스포머 컨텍스트에는 현재 작업과 관련된 최근 정보만 포함해야 한다. 오래된 관측은 요약(Summary), 객체 메모리(Object Memory), 임무 상태(Mission State), 의미 지도(Semantic Map), 외부 데이터베이스로 압축할 수 있다. 이후 다시 필요할 수 있는 정확한 이벤트는 구조화 메모리에 저장한 뒤 필요할 때 검색할 수 있다.

이 접근은 작업 메모리(Working Memory)와 장기 메모리(Long-Term Memory)를 분리한다. KV 캐시는 현재의 추론 및 행동 구간을 위한 빠른 작업 메모리 역할을 한다. 외부 메모리는 모든 과거 토큰을 어텐션 컨텍스트에 유지하지 않고도 장시간 임무 정보를 보존한다.

KV 캐시 최적화는 각 토큰에 저장하는 데이터의 양을 줄이는 것에서 시작된다. 다중 헤드 어텐션(Multi-Head Attention)은 일반적으로 모든 어텐션 헤드마다 별도의 키와 값 벡터를 저장한다. 다중 쿼리 어텐션(Multi-Query Attention)은 여러 쿼리 헤드를 사용하지만 하나의 키 헤드와 값 헤드를 공유한다. 그룹 쿼리 어텐션(Grouped-Query Attention)은 여러 쿼리 헤드 그룹이 키와 값 헤드를 공유하도록 하여 중간적인 절충안을 제공한다.

이러한 방식은 KV 캐시 크기와 메모리 대역폭을 크게 줄인다. 특히 그룹 쿼리 어텐션은 완전히 공유된 다중 쿼리 어텐션보다 표현 유연성을 더 많이 유지하면서도 일반 다중 헤드 어텐션보다 훨씬 적은 메모리를 사용한다.

로봇 배포에서는 그룹 쿼리 어텐션이 지연 시간과 메모리 효율성을 모두 향상시킬 수 있다. 컨텍스트가 길거나 여러 모델 인스턴스가 동일한 가속기(Accelerator)에서 실행될 때 효과가 크다. 다만 기존의 일반 다중 헤드 어텐션 모델을 그룹 쿼리 어텐션 구조로 바꾸려면 일반적으로 아키텍처 학습이나 세심한 적응 과정이 필요하다.

KV 캐시 양자화(KV-Cache Quantization)는 또 다른 최적화 방법이다. 키와 값을 전체 16비트 정밀도로 저장하지 않고 8비트, 4비트 또는 혼합 정밀도(Mixed Precision) 형식으로 표현할 수 있다. 이를 통해 메모리 소비와 어텐션 계산 시 캐시를 읽는 데 필요한 대역폭을 줄일 수 있다.

캐시 양자화는 가중치 양자화(Weight Quantization)와 다르다. 가중치 양자화는 고정된 모델 파라미터를 압축하고, KV 캐시 양자화는 추론 중 생성되는 동적 활성 메모리(Dynamic Activation Memory)를 압축한다. 긴 컨텍스트 배포에서는 두 방식이 모두 필요할 수 있다.

키와 값은 수치 민감도가 다를 수 있다. 일부 차원은 이상치(Outlier)를 포함하거나 더 높은 정밀도를 필요로 할 수 있다. 전체 캐시에 하나의 전역 스케일(Global Scale)을 적용하는 것보다 채널별(Per-Channel), 헤드별(Per-Head), 토큰별(Per-Token) 스케일을 사용하는 것이 정확도 보존에 더 유리할 수 있다.

최근 토큰은 높은 정밀도로 저장하고 오래된 토큰은 낮은 정밀도로 저장하는 혼합 방식도 가능하다. 최근 관측은 현재 로봇 행동에 더 중요할 수 있고, 오래된 문맥은 어느 정도의 근사 오차를 허용할 수 있다. 이는 시간적 정밀도 계층(Temporal Precision Hierarchy)을 형성한다.

캐시 제거(Cache Eviction)도 중요하다. 컨텍스트가 미리 정의된 한계에 도달하면 어떤 토큰을 제거할지 결정해야 한다. 가장 단순한 슬라이딩 윈도우(Sliding Window)는 가장 오래된 토큰을 버리고 최근 문맥만 유지한다.

슬라이딩 윈도우 어텐션은 지역적 시간 의존성을 처리하는 데 효과적이지만, 중요한 장기 명령을 삭제할 수 있다. 로봇은 임무 시작 시 받은 안전 제한을 수시간 후에도 따라야 할 수 있다. 단순히 오래되었다는 이유만으로 정보를 제거하면 이러한 제약이 사라질 수 있다.

어텐션 싱크(Attention Sink) 방식은 일부 초기 토큰이나 특수 토큰을 유지하면서 일반적인 과거 이력만 제거한다. 영구 토큰(Persistent Token)은 시스템 명령, 임무 목표, 로봇 ID, 안전 규칙, 작업 제약을 표현할 수 있다. 최근 토큰은 이동하는 윈도우 안에 유지되고, 안정적인 전역 문맥은 계속 사용할 수 있다.

중요도 기반 제거(Importance-Based Eviction)는 어텐션 점수, 학습된 메모리 정책, 의미적 중요도, 작업 메타데이터에 따라 관련성이 높다고 판단된 토큰을 보존한다. 예를 들어 성공한 도킹 기준, 이상 이벤트, 작업자 수정 사항, 해결되지 않은 작업은 반복적인 센서 설명보다 오래 유지할 수 있다.

그러나 학습된 중요도 판단은 실패할 수 있다. 안전에 중요한 정보가 중요한지 여부를 모델 판단에만 맡겨서는 안 된다. 지속적인 임무 제약과 안전 정책은 보호된 명시적 메모리에 저장해야 한다.

페이지형 KV 캐시 관리(Paged KV-Cache Management)는 가상 메모리와 유사하게 캐시 블록을 구성한다. 각 시퀀스마다 하나의 큰 연속 메모리 버퍼를 할당하는 대신 작은 캐시 블록을 동적으로 할당하고 재사용한다. 이를 통해 서로 다른 컨텍스트 길이를 가진 여러 요청이나 로봇 에이전트를 처리할 때 메모리 단편화(Fragmentation)를 줄이고 활용률을 높일 수 있다.

페이지형 어텐션(Paged Attention)은 서버 환경, 플릿(Fleet) 시스템, 여러 로봇 요청을 처리하는 엣지 서버에서 특히 유용하다. 동시성(Concurrency)을 높이고 배칭(Batching)을 효율적으로 수행할 수 있다. 단일 로봇에서도 임무 계획, 작업자 대화, 행동 생성을 별도 모델 프로세스가 처리하는 경우 유용할 수 있다.

프리픽스 캐싱(Prefix Caching)은 여러 요청에서 공통으로 사용하는 컨텍스트의 KV 상태를 재사용한다. 로봇 ID, 시스템 명령, 형태 설명(Morphology Description), 공구 사양, 지도, 임무 정책은 여러 추론 주기 동안 변하지 않을 수 있다. 이러한 키와 값 표현은 한 번 계산한 뒤 반복적으로 재사용할 수 있다.

이를 통해 프리필(Prefill) 계산을 줄일 수 있다. 로봇이 동일한 시스템 프롬프트와 체화 설명을 포함하는 비슷한 컨텍스트를 반복적으로 전송한다면 프리픽스 캐싱은 지연 시간을 크게 줄일 수 있다.

캐시 무효화(Cache Invalidation)는 신중하게 관리해야 한다. 로봇 공구, 센서 보정, 임무, 소프트웨어 버전, 안전 정책이 변경되면 저장된 프리픽스는 더 이상 유효하지 않다. 토큰 텍스트가 비슷해 보여도 오래된 캐시를 재사용하면 숨겨진 불일치가 발생할 수 있다.

트랜스포머 추론의 프리필 단계와 디코딩 단계는 서로 다른 성능 특성을 가진다. 프리필은 초기 입력 시퀀스를 처리하며 모든 입력 토큰을 알고 있기 때문에 높은 병렬성을 활용할 수 있다. 디코딩은 한 번에 하나 또는 소수의 새로운 토큰을 생성하며, 주로 메모리 대역폭과 KV 캐시 접근 성능에 의해 제한된다.

피지컬 AI 작업은 이 두 단계를 반복적으로 오갈 수 있다. 로봇은 새로운 이미지, 포인트 클라우드 요약, 임무 갱신을 받아 큰 프리필 작업을 수행한 뒤 짧은 행동 생성 단계를 실행할 수 있다. 따라서 효율적인 배포는 긴 텍스트 생성의 초당 토큰 수만 최적화할 것이 아니라 프리필과 디코딩을 모두 최적화해야 한다.

청크형 프리필(Chunked Prefill)은 긴 입력 컨텍스트를 더 작은 블록으로 나눈다. 이를 통해 최대 메모리 사용량을 줄이고, 스케줄링을 개선하며, 하나의 대형 요청이 다른 작업을 장시간 차단하는 것을 막을 수 있다. 로봇에서는 큰 의미 컨텍스트를 인코딩하는 동안에도 긴급 인식이나 안전 프로세스가 계속 실행되도록 할 수 있다.

그러나 시간 순서는 일관되게 유지되어야 한다. 센서 데이터를 임의로 분할하여 행동과 대응 관측을 분리하거나 동기화를 파괴해서는 안 된다.

연속 배칭(Continuous Batching)은 서로 다른 추론 요청의 토큰을 하나의 GPU 연산에 결합한다. 서버 처리량을 크게 향상시킬 수 있지만, 로봇 추론은 일반적으로 배치가 작고 지연 시간에 민감하다. 큰 배치는 요청 대기 시간을 증가시킬 수 있다.

플릿 엣지 서버는 여러 로봇의 요청을 함께 처리할 수 있기 때문에 연속 배칭의 이점을 크게 얻을 수 있다. 반면 온보드 모델은 하나의 요청을 즉시 처리하는 지연 시간 최적화를 우선해야 한다.

따라서 스케줄링 정책은 배포 환경을 반영해야 한다. 클라우드 서비스는 처리량을 최대화할 수 있지만, 자율 로봇은 마감 시간(Deadline) 준수를 우선해야 한다. 우선순위가 높은 장애물 또는 고장 질의가 낮은 우선순위의 보고서 생성 때문에 지연되어서는 안 된다.

가중치 양자화는 트랜스포머 추론 비용을 줄이는 가장 널리 사용되는 방법이다. 16비트 또는 32비트로 학습된 모델 가중치를 더 적은 비트로 표현할 수 있다. 일반적인 형식은 8비트 정수, 4비트 정수, 여러 저정밀 부동소수점 형식을 포함한다.

양자화는 모델 저장 공간, 메모리 대역폭, 에너지 소비를 줄인다. 4비트 표현은 이론적으로 16비트 표현의 약 4분의 1 수준까지 가중치 메모리를 줄일 수 있지만, 스케일, 메타데이터, 구현 오버헤드 때문에 실제 절감률은 이보다 낮다.

가중치 전용 양자화(Weight-Only Quantization)는 가중치는 낮은 정밀도로 저장하고 활성값(Activation)은 16비트 또는 다른 고정밀 형식으로 유지한다. 활성값은 동적 범위와 이상치를 포함할 수 있어 양자화가 더 어렵기 때문에 비교적 배포가 쉽다.

가중치 전용 8비트 양자화는 일반적으로 성능 손실이 작다. 4비트 가중치 양자화는 더 큰 압축률을 제공하지만 세심한 보정, 그룹화, 이상치 처리가 필요하다.

그룹별 양자화(Group-Wise Quantization)는 가중치 행렬을 여러 그룹으로 나누고 각 그룹에 별도의 스케일을 적용한다. 작은 그룹은 지역적인 수치 범위에 더 잘 적응하므로 일반적으로 정확도 유지에 유리하지만 스케일 메타데이터가 더 많이 필요하다.

채널별 양자화(Per-Channel Quantization)는 개별 출력 채널에 스케일을 할당한다. 서로 다른 채널의 값 분포가 크게 다를 때 텐서별 양자화(Per-Tensor Quantization)보다 정확도가 높다.

대칭 양자화(Symmetric Quantization)는 0을 중심으로 동일한 양수와 음수 범위를 사용한다. 비대칭 양자화(Asymmetric Quantization)는 영점(Zero-Point)을 추가하여 중심이 0이 아닌 분포를 표현할 수 있다. 대칭 방식은 일반적으로 단순하고 빠르며, 비대칭 방식은 일부 활성값 분포에 더 잘 맞을 수 있다.

학습 후 양자화(Post-Training Quantization)는 모델 학습 이후에 양자화를 적용한다. 보정 데이터셋(Calibration Dataset)을 이용하여 적절한 범위와 스케일을 추정한다. 전체 재학습이 필요하지 않기 때문에 비교적 비용이 적다.

보정 데이터셋의 품질은 매우 중요하다. 텍스트만 사용해 보정한 로봇 모델은 이미지 임베딩, 행동 토큰, 힘 특징, 비정상적인 임무 문맥을 처리할 때 성능이 떨어질 수 있다. 보정 데이터는 실제 배포에서 예상되는 모달리티, 시퀀스 길이, 로봇 작업, 운용 조건을 반영해야 한다.

피지컬 AI 보정에는 정상 상황과 어려운 상황이 모두 포함되어야 한다. 긴 컨텍스트, 드문 센서 이벤트, 강한 힘 신호, 저조도 시각 특징, 실패 복구, 교차 모달 융합은 일반적인 예제에서는 나타나지 않는 활성값 범위를 만들 수 있다.

양자화 인식 학습(Quantization-Aware Training)은 학습 또는 미세조정 과정에서 저정밀 연산을 모사한다. 모델은 양자화 이후에도 강건한 파라미터를 학습한다. 특히 매우 낮은 정밀도에서는 학습 후 양자화보다 높은 정확도를 제공하는 경우가 많다.

그러나 양자화 인식 학습은 추가 데이터, 계산량, 엔지니어링을 요구한다. 대형 파운데이션 모델의 전체 재학습은 현실적으로 어려울 수 있다. 파라미터 효율적 미세조정(Parameter-Efficient Fine-Tuning)이나 저랭크 적응(Low-Rank Adaptation)을 양자화 인식 방법과 결합하여 비용을 줄일 수 있다.

활성값 양자화(Activation Quantization)는 가중치 양자화보다 어렵다. 활성값은 현재 입력에 따라 달라지고 동적으로 이상치가 나타날 수 있기 때문이다. 8비트 활성값 양자화는 실용적일 수 있지만 4비트 활성값 양자화는 훨씬 더 어렵다.

스무딩 기반 스케일링(Smooth Scaling)은 가중치와 활성값 사이에 수치 범위를 재분배한다. 큰 활성값의 일부를 가중치 쪽으로 흡수하여 활성값을 더 쉽게 양자화할 수 있게 한다. 이를 통해 효율적인 정수 행렬 곱셈이 가능해질 수 있다.

활성값 스무딩의 효과는 모델 아키텍처와 데이터에 따라 달라진다. 멀티모달 로봇 모델은 비전, 언어, 고유감각(Proprioception), 행동 계층에서 활성값 분포가 매우 다를 수 있다. 하나의 전역 보정 전략만으로는 충분하지 않을 수 있다.

계층별 혼합 정밀도(Layer-Wise Mixed Precision)는 민감한 구성요소를 높은 정밀도로 유지하고 덜 민감한 계층은 더 강하게 양자화한다. 임베딩 계층, 어텐션 출력 투영, 정규화 계층, 행동 헤드는 서로 다른 처리가 필요할 수 있다.

로봇 행동 생성에서는 최종 출력 계층이 특히 민감할 수 있다. 언어 생성에서 작은 수치 오차는 표현만 바꿀 수 있지만, 연속 제어 출력의 작은 오차는 위치, 힘, 속도를 바꿀 수 있다.

따라서 행동 디코더와 안전 관련 예측 헤드는 대부분의 백본이 4비트 가중치를 사용하더라도 16비트 정밀도로 유지할 수 있다. 작은 메모리 증가로 물리적 신뢰성을 크게 향상시킬 수 있다.

양자화는 모델 퍼플렉서티(Perplexity)나 임베딩 유사도만이 아니라 작업 수준 지표로 평가해야 한다. 오프라인 예측 손실의 작은 변화가 폐루프 로봇 성공률을 크게 낮출 수 있다.

시험에는 행동 부드러움(Action Smoothness), 궤적 오차, 충돌 여유(Collision Margin), 힘 최대값, 접지 정확도(Grounding Accuracy), 불확실성 보정, 실패 복구를 포함해야 한다. 텍스트 품질은 유지하지만 기하학 정밀도가 저하되는 양자화 모델은 사용할 수 없을 수 있다.

양자화는 신뢰도 추정에도 영향을 줄 수 있다. 수치 압축 이후 모델이 이전보다 더 확신하거나 덜 확신할 수 있다. 불확실성이 로봇의 자율 행동 여부나 도움 요청 여부를 결정한다면 다시 보정해야 한다.

모델 압축은 체계적인 편향을 만들 수도 있다. 일반적인 입력보다 드문 작업, 긴 컨텍스트, 비정상적인 센서 조합에서 더 큰 성능 저하가 발생할 수 있다. 따라서 꼬리 사례(Tail Case)와 분포 변화(Distribution Shift)를 포함하여 평가해야 한다.

저랭크 분해(Low-Rank Factorization)는 또 다른 추론 최적화 방법이다. 대형 가중치 행렬을 더 작은 행렬의 곱으로 근사할 수 있다. 원래 행렬에 충분한 저랭크 구조가 존재하면 파라미터 수와 행렬 곱셈 비용을 줄일 수 있다.

저랭크 적응 모듈은 일반적으로 미세조정에 사용되지만 모듈형 배포에도 유용하다. 공유 양자화 기반 모델은 고정하고, 로봇별 저랭크 어댑터(Adapter)만 추가하여 특화할 수 있다.

서로 다른 어댑터는 다양한 체화, 공구, 공장, 작업을 지원할 수 있다. 현재 활성 어댑터만 로드하면 되므로 로봇마다 전체 모델을 별도로 유지하는 것보다 갱신 및 저장 비용이 적다.

어댑터 융합(Adapter Fusion)은 여러 특화를 결합할 수 있지만 계산량이 증가하고 간섭이 발생할 수 있다. 라우팅 메커니즘은 로봇 ID와 임무 단계에 따라 사용할 어댑터를 선택할 수 있다.

프루닝(Pruning)은 불필요하다고 판단되는 가중치, 채널, 어텐션 헤드, 전체 계층을 제거한다. 비구조적 프루닝(Unstructured Pruning)은 개별 값을 제거하여 희소 가중치 행렬을 만든다. 파라미터 수는 줄어들지만 하드웨어가 희소 연산을 효율적으로 지원하지 않으면 실제 추론 속도는 향상되지 않을 수 있다.

구조적 프루닝(Structured Pruning)은 전체 채널, 뉴런, 헤드, 블록을 제거한다. 표준 하드웨어에서 가속하기 쉬운 더 작은 밀집 행렬을 만든다. 그러나 구조 전체를 제거하기 때문에 정확도 손실이 더 클 수 있다.

어텐션 헤드 프루닝(Attention-Head Pruning)은 중복된 헤드를 제거할 수 있다. 일부 헤드는 목표 로봇 작업에 거의 영향을 주지 않을 수 있다. 그러나 평균적으로 중요하지 않아 보이는 헤드가 드문 접지 또는 안전 상황에서 핵심 역할을 할 수도 있다.

피드포워드 계층은 트랜스포머 파라미터와 계산량의 큰 비중을 차지한다. 너비 프루닝(Width Pruning)이나 뉴런 프루닝으로 비용을 줄일 수 있다. 이후 성능 회복을 위해 지식 증류(Knowledge Distillation)가 필요할 수 있다.

계층 제거(Layer Dropping)는 전체 트랜스포머 블록을 제거한다. 조기 종료(Early Exit) 아키텍처는 입력이 단순한 경우 중간 계층에서 계산을 종료하고, 복잡한 입력만 더 깊은 계층까지 처리한다.

적응형 깊이(Adaptive Depth)는 로봇에 매력적이다. 익숙한 장면이나 단순한 명령은 적은 계산만 필요할 수 있다. 모호하거나 위험한 상황에서는 전체 모델 깊이를 사용할 수 있다.

조기 종료의 신뢰도는 세심하게 보정해야 한다. 얕은 계층의 출력이 잘못되었음에도 높은 확신을 보이면 위험하다. 센서 품질, 분포 외 탐지, 물리적 제약 검사도 조기 종료 결정에 영향을 주어야 한다.

추측 디코딩(Speculative Decoding)은 작은 초안 모델(Draft Model)이 여러 미래 토큰을 제안하고, 큰 목표 모델(Target Model)이 이 토큰들을 병렬로 검증하는 방식이다. 승인된 토큰은 비용이 큰 목표 모델의 순차 호출 횟수를 줄인다.

이 기술은 텍스트와 구조화 토큰 생성에 유용하다. 로봇 계획에서는 작은 행동 또는 계획 모델이 시퀀스를 제안하고, 더 큰 모델이 이를 검증할 수 있다.

그러나 검증은 물리적 일관성을 포함해야 한다. 언어 토큰을 승인하는 것과 로봇 운동에 영향을 주는 행동을 승인하는 것은 다르다. 검증된 행동 청크도 기하학, 동역학, 안전 검사를 통과해야 한다.

초안 모델은 목표 모델 및 배포 도메인과 정렬되어야 한다. 범용 소형 언어 모델은 유창한 텍스트를 생성하더라도 부적절한 로봇 행동을 제안할 수 있다.

병렬 디코딩(Parallel Decoding)은 여러 후보 계획 또는 행동 시퀀스를 동시에 생성할 수도 있다. 월드 모델, 플래너, 안전 평가기가 후보를 순위화할 수 있다. 계산량은 증가하지만 의사결정 품질은 향상될 수 있다.

빔 서치(Beam Search)는 시퀀스 생성에 널리 사용되지만 계산 비용이 크고 다양성을 줄일 수 있다. 물리적 행동에서는 가장 확률이 높은 시퀀스가 반드시 가장 안전하거나 효율적인 것은 아니다. 후보 평가에는 우도뿐 아니라 작업 비용과 위험도 포함되어야 한다.

행동 청킹(Action Chunking)은 추론 호출 횟수를 줄인다. 매 제어 단계마다 하나의 행동을 생성하는 대신 짧은 미래 행동 시퀀스를 한 번에 생성한다. 이를 통해 처리량과 시간적 일관성이 향상된다.

그러나 로봇은 긴 행동 청크를 그대로 실행해서는 안 된다. 후퇴 지평선 제어(Receding-Horizon Control)는 시퀀스 일부만 실행한 뒤 새로운 관측을 받고 갱신된 행동 청크를 생성한다.

청크 길이도 상충관계를 가진다. 긴 청크는 추론 빈도를 줄이지만 환경 변화에 느리게 반응한다. 짧은 청크는 반응성을 높이지만 계산량이 증가한다.

동적 청크 길이(Dynamic Chunk Length)는 불확실성에 따라 조절할 수 있다. 예측 가능한 빈 복도에서는 긴 이동 시퀀스를 생성할 수 있다. 사람, 장애물, 접촉에 가까워지면 짧은 시퀀스를 생성하고 더 자주 재계획해야 한다.

토큰 축소(Token Reduction)는 멀티모달 트랜스포머에서 특히 중요하다. 고해상도 이미지 한 장이 수천 개의 패치 토큰을 생성할 수 있다. 여러 카메라, 비디오 프레임, 포인트 클라우드를 함께 처리하면 어텐션과 캐시 메모리를 빠르게 소모한다.

시각 토큰 풀링(Visual Token Pooling)은 인접한 패치 특징을 압축한다. 객체 중심 토큰화(Object-Centric Tokenization)는 밀집 이미지를 더 적은 객체 또는 영역 토큰으로 변환한다. 학습형 리샘플러(Learned Resampler)는 잠재 쿼리(Latent Query)를 사용하여 관련 시각 정보를 모은다.

작업 조건부 토큰 프루닝(Task-Conditioned Token Pruning)은 현재 임무에 영향을 줄 가능성이 낮은 이미지 영역을 제거한다. 도킹 중에는 정렬 마커와 플랫폼 경계를 유지하고 멀리 있는 배경 영역을 제거할 수 있다.

토큰 프루닝은 작은 안전 관련 객체를 보존해야 한다. 얇은 케이블, 사람의 손, 낮은 장애물은 이미지에서 작은 영역만 차지하지만 안전 행동에 큰 영향을 준다. 단순한 시각적 중요도만을 기준으로 제거하면 이러한 특징을 잃을 수 있다.

다중 해상도 토큰 처리(Multi-Resolution Token Processing)는 저해상도 전역 장면을 유지하면서 선택된 영역의 고해상도 토큰을 보존한다. 이를 통해 전체 문맥과 지역 정밀도를 함께 확보한다.

시간 토큰 압축(Temporal Token Compression)은 반복되는 프레임이나 안정된 로봇 상태를 요약한다. 정지한 장면에서는 동일한 관측을 모두 컨텍스트에 유지할 필요가 없다. 상태공간 모델(State Space Model), 순환 인코더(Recurrent Encoder), 시간 풀링(Temporal Pooling)이 압축된 이력 표현을 만들 수 있다.

이벤트 기반 토큰화(Event-Driven Tokenization)는 의미 있는 변화가 발생할 때만 토큰을 생성한다. 새로운 장애물, 힘의 급증, 임무 갱신, 센서 고장이 이벤트 토큰을 생성한다. 변화가 없는 반복 텔레메트리는 압축된 상태에 유지할 수 있다.

이 접근은 대부분의 데이터가 반복적이고 드문 이벤트가 중요한 로봇 운용 특성과 잘 맞는다. 트랜스포머 부담도 줄여준다.

플래시 어텐션(Flash Attention)과 관련 최적화 어텐션 커널은 메모리 이동량을 줄이고 계산 효율을 높인다. 일반 어텐션은 전체 어텐션 점수 행렬을 메모리에 저장할 수 있어 긴 시퀀스에서 매우 비싸다. 타일 기반 알고리즘(Tiled Algorithm)은 어텐션을 작은 블록 단위로 계산하고 불필요한 중간 결과를 저장하지 않는다.

이러한 방법은 일반적으로 모델 출력 자체를 크게 바꾸지 않으면서 속도와 메모리 효율을 향상시킨다. 프리필과 학습에서 특히 효과가 크지만 최적화된 디코딩 커널도 추론 성능을 향상시킬 수 있다.

커널 융합(Kernel Fusion)은 정규화, 투영, 활성화 함수, 잔차 연결과 같은 여러 연산을 더 적은 GPU 커널로 결합한다. 이를 통해 메모리 읽기·쓰기와 커널 실행 오버헤드를 줄인다.

소형 엣지 작업에서는 실제 산술 연산보다 실행 오버헤드가 더 큰 비중을 차지할 수 있다. 따라서 이론적인 연산량이 같더라도 융합 커널은 지연 시간을 줄일 수 있다.

연산자 호환성(Operator Compatibility)도 중요하다. 모델이 사용자 정의 어텐션, 회전형 위치 임베딩(Rotary Positional Embedding), MoE 라우팅, 특수 멀티모달 계층을 사용하면 최적화 런타임이 이를 지원하지 않을 수 있다. 이 경우 느린 구현으로 대체될 수 있다.

따라서 최적화는 학습 이후가 아니라 모델 설계 단계부터 고려해야 한다. 지원되는 연산만 사용한 아키텍처는 컴파일과 가속이 더 쉽다.

그래프 컴파일(Graph Compilation)은 모델을 최적화된 실행 그래프로 변환한다. 상수 폴딩(Constant Folding), 연산자 융합, 메모리 계획, 하드웨어별 커널 선택을 통해 오버헤드를 줄일 수 있다. 정적 형상(Static Shape)은 컴파일을 단순화하지만 로봇 입력은 토큰 수와 시퀀스 길이가 달라질 수 있다.

형상 버킷팅(Shape Bucketing)은 제한된 수의 입력 크기를 미리 정의한다. 센서 데이터는 가장 가까운 지원 형상에 맞게 패딩(Padding)하거나 압축한다. 이를 통해 일정한 유연성을 유지하면서 컴파일 최적화의 이점을 얻는다.

동적 형상(Dynamic Shape)은 더 높은 일반성을 제공하지만 성능이 저하될 수 있다. 배포 아키텍처는 어떤 차원이 실제로 가변적이어야 하는지를 구분해야 한다.

메모리 할당도 예측 가능해야 한다. 빈번한 동적 메모리 할당은 단편화와 지연 시간 급증을 만들 수 있다. 실시간 시스템에서는 사전 할당 버퍼와 고정 최대 용량이 유용하다.

CPU와 GPU 사이의 데이터 전송도 병목이 될 수 있다. 카메라 전처리, 토큰화, 모델 추론, 제어 출력이 서로 다른 프로세서에서 수행될 수 있다. 반복적인 데이터 복사는 성능을 저하시킬 뿐 아니라 지연 시간 변동을 키운다.

제로 카피 메모리(Zero-Copy Memory), 고정 메모리(Pinned Buffer), 통합 메모리(Unified Memory), 하드웨어 가속 전처리는 전송 오버헤드를 줄일 수 있다. 최적의 방법은 로봇 컴퓨팅 플랫폼에 따라 달라진다.

센서 전처리도 성능 분석에 포함해야 한다. 이미지 리사이징, 정규화, 포인트 클라우드 변환, 압축, 좌표 변환은 상당한 시간을 소비할 수 있다.

모델 추론이 20밀리초라고 해도 전처리와 통신에 80밀리초가 더 필요하다면 전체 시스템은 20밀리초 시스템이 아니다.

파이프라인 병렬화(Pipeline Parallelism)는 센서 전처리, 추론, 계획, 행동 실행을 겹쳐 수행할 수 있다. 하나의 프레임을 처리하는 동안 다음 프레임을 취득하고 이전 행동을 실행할 수 있다.

그러나 세심한 동기화가 필요하다. 과도한 파이프라이닝은 실질적인 관측 나이(Observation Age)를 증가시킬 수 있다. 시스템이 최신 프레임에 느리게 반응하는 것이 아니라 오래된 프레임에 빠르게 반응하게 될 수 있다.

따라서 모델 단독 지연 시간보다 관측-행동 지연 시간(Observation-to-Action Latency)이 더 중요하다. 이는 물리 센서 데이터가 취득된 시점부터 액추에이터 명령이 준비될 때까지의 시간을 의미한다.

정보의 나이(Age of Information)도 유용한 지표이다. 로봇이 의사결정을 사용할 때 기반 관측이 얼마나 오래된 것인지 나타낸다. 버퍼링이 많다면 계산 시간이 짧아도 정보는 오래되었을 수 있다.

열 관리(Thermal Management)도 추론 최적화의 일부이다. 로봇 컴퓨터는 밀폐된 플랫폼에서 동작하고 다른 전자 장치와 냉각 시스템을 공유할 수 있다. 장시간 모델 추론은 열 스로틀링(Thermal Throttling)을 일으켜 긴 임무 동안 지연 시간을 증가시킬 수 있다.

따라서 벤치마크는 짧은 실행뿐 아니라 장시간 운용도 포함해야 한다. 실제 임무 시간 동안 클럭 주파수, 소비전력, 온도, 지연 시간을 모니터링해야 한다.

동적 주파수 조절(Dynamic Frequency Scaling)은 계산 요구가 낮을 때 에너지를 절약하고 어려운 상황에서는 성능을 높일 수 있다. 그러나 주파수 변화는 지연 시간 변동을 만들 수 있다.

추론 스케줄러는 임무 단계에 따라 계산 자원을 배분할 수 있다. 일상적인 내비게이션에서는 더 작은 모델이나 낮은 갱신 주기가 충분할 수 있다. 조작이나 사람과의 상호작용에서는 더 많은 자원을 활성화할 수 있다.

모델 캐스케이딩(Model Cascading)도 유용하다. 작은 로컬 모델이 일반적인 사례를 처리하고, 신뢰도가 낮거나 작업이 익숙하지 않거나 추가 추론이 필요한 경우에만 더 큰 모델을 호출한다.

이 방식은 평균 에너지와 지연 시간을 줄일 수 있다. 또한 큰 모델을 사용할 수 없는 경우에도 점진적 성능 저하(Graceful Degradation)를 지원한다.

작은 모델은 언제 상위 모델 호출이 필요한지를 판단할 수 있어야 한다. 보정되지 않은 로컬 모델이 자신의 능력을 넘어서는 상황을 높은 확신으로 처리하면 위험하다.

캐스케이드는 불확실성, 센서 품질, 새로움(Novelty), 작업 위험, 모델 간 불일치 등 여러 신호를 사용해야 한다. 중요한 조건에서는 항상 더 강력한 경로를 사용하도록 설정할 수도 있다.

클라우드 오프로딩(Cloud Offloading)은 더 큰 모델을 사용할 수 있게 하지만 지연 시간, 개인정보 보호, 비용, 가용성 문제를 만든다. 로봇 시스템은 비상 행동이나 핵심 제어를 원격 추론에 의존해서는 안 된다.

하이브리드 아키텍처는 즉각적인 로컬 행동을 수행하면서 복잡한 임무 추론에 대해 원격 지원을 요청할 수 있다. 원격 응답은 이후 계획에 영향을 줄 수 있지만 안전한 로컬 운용을 차단해서는 안 된다.

클라우드로 전송하는 데이터는 최소화하고 보호해야 한다. 원시 비디오 전체를 보내는 대신 선택된 이미지, 의미 요약, 익명화된 특징을 전송할 수 있다.

시스템은 원격 출력도 검증해야 한다. 클라우드 모델은 현재의 정확한 물리 상태를 알지 못할 수 있다. 모든 원격 권고는 최신 로컬 센서 데이터에 다시 접지되어야 한다.

추론 최적화는 결정론적 안전 경계(Deterministic Safety Boundary)를 유지해야 한다. 양자화되거나 프루닝되거나 가속된 파운데이션 모델도 확률적 구성요소이다. 모델은 제어된 인터페이스 안에서 목표, 스킬, 궤적, 행동 청크를 제안해야 한다.

런타임 안전 계층은 충돌, 속도, 가속도, 힘, 작업 공간, 제한 구역, 로봇 상태를 검사해야 한다. 지연 시간을 줄이기 위해 이러한 안전 검사를 제거해서는 안 된다.

안전 처리 자체는 결정론적 알고리즘과 필요할 경우 전용 하드웨어를 사용하여 별도로 최적화해야 한다. 비상 정지는 대형 트랜스포머의 결과를 기다릴 수 없다.

중복된 저수준 안전 센서는 파운데이션 모델과 독립적으로 동작해야 한다. 모델은 행동을 개선할 수 있지만 피해를 막는 유일한 메커니즘이 되어서는 안 된다.

최적화 이후에는 버전 관리가 더 복잡해진다. 양자화 모델, 프루닝 모델, 컴파일된 엔진, 원본 체크포인트(Checkpoint)는 서로 다르게 동작할 수 있다. 모두 별도의 모델 아티팩트(Artifact)로 취급해야 한다.

배포 패키지는 기반 모델, 양자화 방식, 보정 데이터셋, 정밀도 설정, 컴파일러 버전, 런타임 라이브러리, 대상 하드웨어, 최적화 옵션을 기록해야 한다.

모델 가중치가 같더라도 추론 런타임을 변경하면 수치적 행동이 달라질 수 있다. 소프트웨어나 드라이버 업데이트 이후에는 회귀 시험(Regression Testing)이 필요하다.

KV 캐시 형식과 관리 정책도 버전 관리해야 한다. 캐시 양자화, 제거 정책, 프리픽스 재사용, 메모리 제한은 긴 컨텍스트에서 모델 행동에 영향을 줄 수 있다.

재현성(Reproducibility)을 위해서는 결정론적 시험 시퀀스, 고정된 전처리, 기록된 하드웨어 조건이 필요하다. 일부 저정밀 커널은 작은 비결정적 차이를 만들 수 있다.

언어 출력에서는 작은 수치 변화가 허용될 수 있지만, 로봇 제어에서는 작은 차이가 다른 행동 분기를 만들 수 있다. 따라서 폐루프 검증은 행동 안정성을 측정해야 한다.

벤치마크는 실제 로봇 작업을 반영해야 한다. 긴 텍스트 생성을 중심으로 한 일반 언어 모델 시험은 멀티모달 로봇 추론을 제대로 반영하지 못한다.

현실적인 벤치마크는 이미지 인코딩, 짧은 언어 명령, 상태 토큰 갱신, 긴 임무 메모리, 행동 청크 생성, 도구 호출, 반복 제어 주기를 포함할 수 있다.

지표에는 최초 출력 시간(Time to First Output), 단계별 디코딩 지연 시간, 프리필 지연 시간, KV 캐시 메모리, 전체 모델 메모리, 전력, 온도, 관측 나이, 마감 시간 초과, 작업 성공률이 포함되어야 한다.

백분위 지연 시간(Percentile Latency)도 중요하다. 중앙값은 낮지만 드문 긴 지연이 제어 요구를 위반할 수 있다. 95번째, 99번째 백분위와 최악 지연 시간을 보고해야 한다.

성능은 다른 로봇 프로세스와 자원을 경쟁하는 상황에서 측정해야 한다. 위치 추정, 지도 작성(Mapping), 카메라 드라이버, 로깅, 사용자 인터페이스가 같은 자원을 사용한다.

모델만 단독으로 벤치마크한 결과는 완성된 로봇 안에서 다르게 나타날 수 있다. 자원 경쟁, 메모리 압박, 열 부하를 포함해야 한다.

양자화 평가는 여러 정밀도 구성을 비교해야 한다. 완전한 4비트 모델은 최대 압축을 제공할 수 있지만 4비트와 8비트를 혼합한 시스템이 더 좋은 균형을 제공할 수 있다.

올바른 선택은 작업에 따라 달라진다. 상위 수준의 보고서 생성은 정밀 도킹이나 조작 계획보다 더 강한 압축을 허용할 수 있다.

동일한 파운데이션 모델이라도 서로 다른 로봇 제품에서 다른 최적화 버전을 사용할 수 있다. 고성능 플랫폼은 더 큰 혼합 정밀도 모델을 사용하고, 저비용 플랫폼은 증류 및 양자화된 학생 모델(Student Model)을 사용할 수 있다.

그러나 모델은 호환 가능한 인터페이스를 제공해야 한다. 공유 행동 스키마, 도구 호출, 메모리 형식, 안전 계약(Safety Contract)은 제품군 유지보수를 단순화한다.

지식 증류는 이러한 배포 변형을 생성하는 강력한 방법이다. 대형 교사 모델(Teacher Model)은 목표 분포, 추론 과정, 계획, 행동 후보, 잠재 표현을 생성한다. 작은 학생 모델은 필요한 행동을 모방하도록 학습한다.

학생 모델은 모든 영역에서 교사 모델과 동일할 필요가 없다. 특정 로봇에 필요한 작업만 특화할 수 있다. 이는 범용 모델을 특화 없이 압축하는 것보다 더 나은 엣지 성능을 제공할 수 있다.

증류 데이터에는 일상적인 성공 사례뿐 아니라 실패 사례와 불확실한 조건도 포함해야 한다. 그렇지 않으면 학생 모델은 정상 행동은 모방하지만 상위 모델 호출이 필요한 시점을 인식하지 못할 수 있다.

배포 모델은 줄어든 내부 용량을 보완하기 위해 검색(Retrieval)과 도구를 사용할 수 있다. 모든 지도, 절차, 기술 지식을 가중치 안에 저장하지 않고 필요한 정보만 외부에서 검색할 수 있다.

이는 모델 크기를 줄이고 재학습 없이 정보를 갱신할 수 있게 한다. 그러나 검색 및 도구 호출은 지연 시간을 추가하고 실패 처리 메커니즘을 요구한다.

자주 사용하는 정보는 로컬에 캐시할 수 있다. 드물게 사용하거나 큰 문서는 외부 저장소에 둘 수 있다. 런타임은 즉각적인 행동에 필요한 데이터와 지연 검색이 가능한 데이터를 구분해야 한다.

따라서 트랜스포머 추론 최적화는 하나의 기술이 아니라 여러 방법을 조정하는 설계 과정이다. KV 캐싱은 반복 어텐션 계산을 줄인다. 캐시 압축과 제거는 긴 컨텍스트 메모리를 제어한다. 양자화는 모델과 활성값 비용을 줄인다. 프루닝, 증류, 토큰 축소, 적응형 깊이는 불필요한 계산을 줄인다. 최적화된 커널과 그래프 컴파일은 하드웨어 효율을 향상시킨다.

이러한 기술은 서로 영향을 준다. 강하게 양자화된 모델은 메모리 효율은 높지만 시각 토큰 수가 너무 많으면 여전히 느릴 수 있다. 프루닝된 모델은 빠르지만 드문 이벤트 처리 능력을 잃을 수 있다. 큰 KV 캐시는 가중치 압축의 메모리 이점을 상쇄할 수 있다.

따라서 최적화는 시스템 수준에서 평가해야 한다. 목표는 가장 작은 모델, 가장 높은 토큰 처리량, 가장 낮은 평균 지연 시간 자체가 아니다. 실제 계산, 전력, 열, 메모리, 시간, 안전 제한 안에서 신뢰성 있는 로봇 지능을 구현하는 것이 목표이다.

피지컬 AI에서 가장 효과적인 아키텍처는 일반적으로 계층형이다. 소형 모델은 연속적인 센서 및 상태 정보를 처리한다. 더 큰 트랜스포머는 의미 추론과 어려운 의사결정에만 호출된다. 외부 메모리는 장기 문맥을 보존한다. 결정론적 플래너와 제어기는 검증된 행동을 실행한다. 안전 시스템은 최종 권한을 유지한다.

이러한 아키텍처에서 KV 캐시는 로봇의 전체 이력이 아니라 현재 활성 추론 컨텍스트만 유지해야 한다. 양자화는 수치 허용 범위가 큰 영역에서는 적극적으로 적용하고, 기하학, 행동, 안전에 민감한 영역에서는 보수적으로 적용해야 한다. 토큰 압축은 중복 정보를 제거하되 작지만 중요한 물리적 증거를 삭제해서는 안 된다.

추론 스케줄링은 임무 단계와 불확실성에 따라 적응해야 한다. 일상 운용은 효율적인 로컬 계산을 사용한다. 복잡하거나 익숙하지 않은 상황에서는 더 많은 모델 깊이, 더 많은 전문가, 더 긴 컨텍스트, 추가 센서, 원격 지원을 사용할 수 있다.

장기적인 목표는 단순히 대형 트랜스포머를 더 빠르게 실행하는 것이 아니다. 파운데이션 모델의 능력을 물리 기계에서 지속적으로 동작할 수 있는 예측 가능하고, 에너지 효율적이며, 실시간적인 지능으로 변환하는 것이다.

성공적으로 최적화된 트랜스포머는 로봇 플랫폼의 제한 안에서 의미 이해, 멀티모달 접지, 시간 문맥, 계획 품질, 불확실성 인식을 유지해야 한다. 출력은 의사결정 마감 시간 전에 도착해야 하며, 긴 컨텍스트, 센서 부하, 장시간 운용에서도 행동이 안정적이어야 한다.

따라서 트랜스포머 추론 최적화는 모델 아키텍처와 로봇 제품 엔지니어링을 연결하는 다리이다. 신경망 설계를 하드웨어 선정, 메모리 계획, 소프트웨어 스케줄링, 열 관리, 안전 검증, 수명주기 제어와 연결한다.

피지컬 AI 파운데이션 모델에서 KV 캐시 설계와 양자화는 단순한 구현 세부사항이 아니다. 이는 로봇이 얼마나 많은 문맥을 유지할 수 있는지, 얼마나 빠르게 반응할 수 있는지, 얼마나 오래 운용할 수 있는지, 어느 수준의 지능을 로컬에 배포할 수 있는지를 직접 결정한다.

가장 강력한 시스템은 압축된 모델 가중치, 효율적 어텐션, 관리되는 KV 메모리, 멀티모달 토큰 축소, 적응형 계산, 엄격한 폐루프 시험을 결합할 것이다. 이러한 결합을 통해 트랜스포머 모델은 데이터센터 시연을 넘어 실제 로봇 시스템에서 실용적이고, 통제 가능하며, 신뢰성 있는 구성요소가 될 수 있다.
