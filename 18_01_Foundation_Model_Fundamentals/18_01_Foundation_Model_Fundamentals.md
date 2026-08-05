**Volume 18 Physical AI Foundation Models**


# 1. Foundation Model Fundamentals

##  

## 1.1 Foundation Model Definition Scale Emergence Generalization

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

Foundation models represent a fundamental shift in the philosophy of artificial intelligence. Rather than designing separate models for every individual task, researchers increasingly train a single large model using enormous amounts of heterogeneous data and then adapt that model to a wide range of downstream applications. This approach has transformed natural language processing, computer vision, speech recognition, multimodal reasoning, scientific computing, and more recently robotics and Physical AI. Instead of viewing intelligence as a collection of isolated capabilities, foundation models attempt to learn a broad representation of the world that can later support many specialized behaviors with relatively little additional training.

The term "foundation model" refers to a model trained on large-scale, diverse datasets using mostly self-supervised or weakly supervised learning objectives. The resulting model becomes a foundation upon which many downstream applications can be constructed. Rather than beginning every new problem with randomly initialized parameters, developers start from a model that already possesses extensive knowledge about language, images, video, audio, code, physical interactions, or combinations of multiple modalities. This pre-trained knowledge significantly reduces development costs while improving generalization, adaptability, and robustness across numerous tasks.

The emergence of foundation models was made possible through the convergence of several technological trends. Massive datasets became available through the internet, industrial data collection systems, simulation platforms, cloud computing infrastructures, and autonomous sensing systems. At the same time, advances in deep neural network architectures, particularly transformers, enabled efficient learning from extremely long sequences of data. Modern GPU clusters, distributed training frameworks, high-speed interconnects, and optimized numerical libraries allowed researchers to scale model training to trillions of tokens and billions or even trillions of parameters. These developments collectively changed AI from task-specific engineering into large-scale representation learning.

Unlike traditional machine learning systems, which are designed around carefully labeled datasets for individual tasks, foundation models primarily learn by predicting missing information from unlabeled data. A language model predicts the next token in a sentence. An image model reconstructs masked image patches. A video model predicts future frames. A robot model predicts future actions given observations. Through these seemingly simple prediction objectives, the model gradually develops internal representations that capture semantic structure, spatial relationships, temporal dependencies, causal regularities, and statistical patterns that extend far beyond the immediate training objective.

The concept of scale is central to understanding foundation models. Scale refers not only to the number of parameters but also to the amount of training data, computational resources, optimization duration, model depth, model width, context length, memory capacity, and infrastructure complexity. Early neural networks contained only thousands or millions of parameters. Modern foundation models routinely contain billions of parameters, while frontier systems may extend into hundreds of billions or even trillions of parameters distributed across multiple computational experts.

Parameter count alone does not define a foundation model. Two models with identical parameter counts may exhibit dramatically different capabilities depending on data quality, data diversity, architecture, optimization strategy, tokenizer design, curriculum scheduling, regularization methods, and hardware efficiency. Consequently, scale should be understood as the combined interaction between model capacity, data richness, computational investment, and optimization quality rather than as a single numerical measure.

Training data has become one of the most valuable resources in foundation model development. Modern models ingest books, academic papers, websites, software repositories, multilingual corpora, mathematical reasoning examples, scientific literature, videos, images, sensor streams, simulation trajectories, robotic demonstrations, and human feedback. Diversity of experience enables the model to discover abstract representations that transfer across many different domains instead of memorizing narrow statistical correlations.

Large-scale computation is another defining characteristic. Training frontier foundation models requires thousands of GPUs or AI accelerators operating simultaneously over weeks or months. Distributed optimization techniques partition model parameters, training batches, optimizer states, and activation tensors across many machines. Communication efficiency becomes nearly as important as numerical computation itself. Improvements in networking, storage throughput, checkpointing, mixed precision arithmetic, gradient compression, and parallel scheduling all contribute significantly to successful large-scale training.

One of the most fascinating aspects of foundation models is emergence. Emergent behavior refers to capabilities that appear unexpectedly as model scale increases. Smaller models may completely fail a task, while a sufficiently large model suddenly begins solving it with reasonable accuracy despite never being explicitly programmed to do so. These qualitative transitions resemble phase changes in complex systems, where gradual increases in underlying capacity produce abrupt changes in observable behavior.

Examples of emergent capabilities include multi-step reasoning, in-context learning, instruction following, code generation, mathematical problem solving, translation between unseen language pairs, commonsense reasoning, chain-of-thought generation, tool usage, planning, and multimodal understanding. These abilities often improve gradually after first appearing, suggesting that sufficient representational capacity enables entirely new computational mechanisms to develop during optimization.

Emergence does not imply magic or consciousness. Instead, it reflects the interaction between large optimization processes and highly expressive neural architectures operating over enormous datasets. Many internal features remain distributed across millions of neurons rather than corresponding to explicitly interpretable symbolic rules. The resulting behaviors may appear surprising because humans cannot easily inspect or predict the representations learned inside extremely high-dimensional parameter spaces.

Researchers continue investigating why emergent capabilities arise. Some hypotheses emphasize increased representational capacity, allowing more complex functions to be approximated. Others focus on optimization dynamics, where sufficiently large networks discover better minima during gradient descent. Additional theories point toward richer internal feature hierarchies, implicit modularity, sparse activation patterns, or increased compositional reasoning emerging naturally from large-scale sequence prediction. Although complete theoretical explanations remain incomplete, empirical evidence consistently demonstrates that capability often grows nonlinearly with scale.

Generalization is another defining property of foundation models. Traditional machine learning systems frequently perform well only on distributions closely resembling their training data. Foundation models instead attempt to learn representations that remain useful across many tasks, domains, environments, and problem formulations. Rather than memorizing specific examples, they develop abstract concepts capable of supporting flexible reasoning under new conditions.

Generalization occurs at multiple levels. Domain generalization enables adaptation across industries such as healthcare, finance, robotics, manufacturing, education, and scientific research. Task generalization allows one model to perform summarization, translation, question answering, reasoning, coding, planning, and dialogue without retraining separate networks. Language generalization extends knowledge across many human languages. Multimodal generalization integrates text, images, audio, video, point clouds, sensor data, and robot observations into unified representations.

Perhaps the most significant form of generalization is compositional generalization. Humans routinely combine familiar concepts into entirely new situations. Foundation models increasingly demonstrate similar abilities by recombining learned representations into novel solutions. For example, a robot foundation model may never have encountered a particular household object during training yet still infer how to manipulate it based on its similarity to previously observed objects.

In-context learning represents a particularly important manifestation of generalization. Instead of updating model parameters through additional gradient descent, the model adapts behavior simply by observing examples placed within its input context. This capability enables rapid adaptation without expensive retraining. A user may provide a few demonstrations, instructions, or examples, after which the model immediately performs new tasks using temporary contextual information.

Transfer learning further extends this concept. Once pre-trained, a foundation model may be fine-tuned using relatively small domain-specific datasets. Medical imaging systems, legal document analyzers, financial assistants, scientific research tools, autonomous robots, and industrial inspection systems all benefit from starting with extensive general knowledge before learning specialized expertise. This significantly reduces data requirements while improving robustness and sample efficiency.

The transformer architecture has become the dominant foundation model architecture because self-attention enables flexible relationships between all elements within an input sequence. Unlike recurrent neural networks, transformers process entire sequences simultaneously, capturing long-range dependencies more effectively while enabling highly parallel computation on modern hardware. Multi-head attention allows the model to represent different types of relationships simultaneously, while feed-forward layers transform representations into increasingly abstract semantic features.

Positional encoding supplements transformers by providing information about sequence order, since attention alone is permutation invariant. Layer normalization stabilizes optimization, residual connections improve gradient flow, and sophisticated optimizers enable efficient learning across billions of parameters. Together these architectural innovations have proven remarkably scalable across language, vision, speech, biology, robotics, and multimodal learning.

Foundation models increasingly extend beyond text into multimodal intelligence. Vision-language models jointly process images and natural language. Audio-language models combine speech with semantic reasoning. Video-language models learn temporal understanding. Physical AI models integrate camera images, LiDAR measurements, proprioceptive sensors, tactile observations, force sensing, language instructions, and robot actions. These multimodal representations enable richer understanding than any individual modality alone.

For robotics, foundation models represent a particularly transformative direction. Classical robotic systems relied heavily on manually engineered perception pipelines, object detectors, planners, localization algorithms, mapping modules, and task-specific controllers. Although highly effective in structured environments, such pipelines often required extensive engineering whenever robots encountered new tasks or environments.

Robot foundation models instead attempt to learn unified representations across perception, reasoning, planning, and control. Rather than separating these components completely, the model learns correlations between sensory observations, language instructions, environmental context, historical experience, and future actions. This enables robots to transfer knowledge between different embodiments, environments, and manipulation tasks more effectively than conventional modular systems.

Scale becomes even more important in Physical AI because robots experience much greater environmental diversity than purely digital systems. A household robot encounters varying lighting conditions, object geometries, human behaviors, surface materials, sensor noise, mechanical wear, unexpected obstacles, and dynamic environments. Large-scale training across diverse robots, simulations, and real-world experiences improves robustness against such variability.

Embodiment introduces additional complexity absent from language models. Physical agents must obey physics, maintain stability, avoid collisions, respect safety constraints, and operate under real-time computational limitations. Consequently, foundation models for robotics must integrate geometric reasoning, temporal prediction, uncertainty estimation, control theory, and physical interaction into a unified representation.

Generalization across embodiments has become a major research objective. Ideally, knowledge learned from one robotic platform should transfer to another with different sensors, kinematics, actuators, or physical dimensions. Cross-embodiment learning reduces the need to collect entirely new datasets whenever hardware changes. Shared latent representations allow robots with different morphologies to benefit from collective experience accumulated across many platforms.

Simulation plays an increasingly important role in achieving scale for Physical AI. Collecting billions of real-world robot interactions remains prohibitively expensive. High-fidelity simulators therefore generate enormous quantities of synthetic experiences, allowing foundation models to observe diverse environments before limited real-world fine-tuning. Domain randomization, realistic rendering, physics simulation, and world models help reduce the gap between simulation and reality.

Evaluation of foundation models presents unique challenges because their capabilities span numerous tasks simultaneously. Traditional benchmark datasets measure narrow performance metrics, whereas foundation models require comprehensive evaluation across reasoning, robustness, calibration, safety, fairness, efficiency, interpretability, transfer learning, and adaptability. For robot foundation models, evaluation additionally includes manipulation success, navigation reliability, collision avoidance, long-horizon planning, energy efficiency, recovery from failures, and human-robot collaboration.

Despite remarkable progress, foundation models possess important limitations. Hallucination remains a major concern in language models. Distribution shift can reduce performance in unfamiliar environments. Training costs remain extremely high. Environmental impact from large-scale computation continues attracting attention. Data quality problems, hidden biases, security vulnerabilities, copyright issues, privacy concerns, and limited interpretability remain active areas of research.

Scaling alone is unlikely to solve every challenge. Future foundation models will likely combine larger datasets with improved architectures, structured memory systems, explicit reasoning mechanisms, causal modeling, retrieval augmentation, world models, active learning, continual adaptation, uncertainty estimation, and more efficient optimization algorithms. Rather than pursuing size indefinitely, research increasingly emphasizes improving data quality, computational efficiency, robustness, and trustworthy behavior.

The long-term vision extends beyond building larger neural networks. Foundation models increasingly serve as universal computational substrates capable of supporting perception, reasoning, planning, communication, decision making, and physical interaction across many domains. For Physical AI, they offer the possibility of robots that continually learn from collective experience, generalize across environments, understand multimodal sensory information, and adapt to unfamiliar tasks with minimal additional supervision.

As Physical AI continues to evolve, foundation models are expected to become the central intelligence layer underlying autonomous systems. Just as operating systems provide common computational infrastructure for software applications, foundation models increasingly provide common cognitive infrastructure for intelligent agents. Their combination of large-scale representation learning, emergent capabilities, and broad generalization forms the technological basis upon which future autonomous robots, embodied AI systems, and general-purpose physical intelligence will be constructed. This topic serves as the conceptual foundation for the remainder of Volume 40, which progressively explores transformer architectures, multimodal learning, world models, robot foundation models, action tokenization, large-scale pretraining, Physical AI training pipelines, safety, adaptation, and real-world deployment strategies within a unified engineering framework.

파운데이션 모델(Foundation Model)은 인공지능(AI, Artificial Intelligence)의 철학 자체를 근본적으로 변화시킨 기술이다. 과거에는 각각의 작업(Task)마다 별도의 모델을 설계하는 것이 일반적이었다. 그러나 현재는 하나의 거대한 모델을 방대한 규모의 다양한 데이터로 먼저 학습시킨 후, 이를 여러 하위 작업(Downstream Applications)에 적응시키는 방식이 주류가 되었다. 이러한 접근법은 자연어 처리(Natural Language Processing), 컴퓨터 비전(Computer Vision), 음성 인식(Speech Recognition), 멀티모달 추론(Multimodal Reasoning), 과학 계산(Scientific Computing), 그리고 최근에는 로보틱스(Robotics)와 피지컬 AI(Physical AI)에 이르기까지 광범위한 분야를 변화시키고 있다. 이제 인공지능은 개별 기능들의 단순한 집합이 아니라, 세상을 폭넓게 이해하는 하나의 공통적인 표현(Representation)을 학습하고 이를 다양한 문제 해결에 활용하는 방향으로 발전하고 있다.

파운데이션 모델(Foundation Model)이란 대규모의 다양한 데이터셋을 이용하여 자기지도학습(Self-Supervised Learning) 또는 약지도학습(Weakly Supervised Learning) 방식으로 사전학습(Pre-training)된 모델을 의미한다. 이렇게 학습된 모델은 다양한 응용 시스템의 기반(Foundation)이 된다. 새로운 문제를 해결할 때마다 무작위 초기화(Random Initialization)된 모델을 처음부터 학습하는 대신, 이미 언어(Language), 이미지(Image), 영상(Video), 음성(Audio), 코드(Code), 물리적 상호작용(Physical Interaction) 및 다양한 멀티모달(Multimodal) 정보를 폭넓게 이해하고 있는 모델을 출발점으로 활용한다. 이러한 사전학습 지식은 개발 비용을 크게 줄이는 동시에 일반화(Generalization), 적응성(Adaptability), 그리고 강건성(Robustness)을 크게 향상시킨다.

파운데이션 모델의 등장은 여러 기술 발전이 동시에 이루어진 결과이다. 인터넷(Internet), 산업 데이터 수집 시스템(Industrial Data Collection Systems), 시뮬레이션 플랫폼(Simulation Platforms), 클라우드 컴퓨팅 인프라(Cloud Computing Infrastructure), 자율 센싱 시스템(Autonomous Sensing Systems) 등을 통해 방대한 데이터가 축적되기 시작했다. 동시에 트랜스포머(Transformer)를 중심으로 하는 심층 신경망(Deep Neural Network) 구조의 발전은 매우 긴 시퀀스(Sequence)를 효율적으로 학습할 수 있도록 만들었다. 여기에 GPU 클러스터(GPU Cluster), 분산 학습 프레임워크(Distributed Training Framework), 고속 인터커넥트(High-Speed Interconnect), 그리고 최적화된 수치 계산 라이브러리(Numerical Libraries)가 결합되면서 수조 개(Token)의 데이터와 수천억 개 이상의 파라미터(Parameter)를 가진 모델을 실제로 학습할 수 있게 되었다. 이러한 변화는 인공지능을 개별 문제 해결 중심의 엔지니어링에서 대규모 표현 학습(Large-Scale Representation Learning)의 시대로 전환시켰다.

기존의 머신러닝(Machine Learning) 시스템은 특정 작업을 위해 정교하게 라벨링(Labeling)된 데이터셋을 필요로 했다. 반면 파운데이션 모델은 대부분 레이블이 없는 데이터(Unlabeled Data)에서 다음 정보를 예측하는 방식으로 학습한다. 언어 모델(Language Model)은 다음 토큰(Token)을 예측하고, 이미지 모델(Image Model)은 가려진 이미지 영역을 복원하며, 비디오 모델(Video Model)은 미래 프레임(Frame)을 예측한다. 로봇 모델(Robot Model)은 현재 관측값으로부터 다음 행동(Action)을 예측한다. 이러한 단순한 예측 작업을 반복하는 과정에서 모델은 의미 구조(Semantic Structure), 공간 관계(Spatial Relationship), 시간 의존성(Temporal Dependency), 인과 관계(Causal Regularity), 그리고 다양한 통계적 패턴(Statistical Pattern)을 내부적으로 학습하게 된다.

규모(Scale)는 파운데이션 모델을 이해하는 가장 중요한 개념 중 하나이다. 규모는 단순히 파라미터 수(Parameter Count)만을 의미하지 않는다. 학습 데이터(Data)의 양, 계산 자원(Compute Resources), 학습 시간(Optimization Duration), 모델의 깊이(Depth), 폭(Width), 컨텍스트 길이(Context Length), 메모리 용량(Memory Capacity), 그리고 학습 인프라(Infrastructure)의 복잡성까지 모두 포함하는 개념이다. 초기의 신경망은 수천 또는 수백만 개의 파라미터만을 사용했지만, 현대의 파운데이션 모델은 수십억(Billion)에서 수천억(Billion to Trillion)에 이르는 파라미터를 사용하며, 이를 여러 계산 장치에 분산하여 학습한다.

그러나 파라미터 수만으로 파운데이션 모델을 정의할 수는 없다. 동일한 파라미터 수를 가진 두 모델이라 하더라도 데이터 품질(Data Quality), 데이터 다양성(Data Diversity), 모델 구조(Architecture), 최적화 전략(Optimization Strategy), 토크나이저(Tokenizer) 설계, 학습 커리큘럼(Curriculum Scheduling), 정규화(Regularization), 그리고 하드웨어 효율성(Hardware Efficiency)에 따라 성능은 크게 달라질 수 있다. 따라서 규모(Scale)는 모델의 크기뿐 아니라 데이터의 풍부함, 계산 자원의 규모, 그리고 최적화의 품질이 함께 만들어내는 종합적인 개념으로 이해해야 한다.

학습 데이터(Training Data)는 오늘날 파운데이션 모델 개발에서 가장 중요한 자산 가운데 하나이다. 현대의 모델은 서적(Books), 논문(Academic Papers), 웹 문서(Web Pages), 소프트웨어 저장소(Code Repositories), 다국어 말뭉치(Multilingual Corpora), 수학 문제(Mathematical Reasoning), 과학 문헌(Scientific Literature), 영상(Video), 이미지(Image), 센서 데이터(Sensor Streams), 시뮬레이션 궤적(Simulation Trajectories), 로봇 시연(Robot Demonstrations), 그리고 인간 피드백(Human Feedback)까지 매우 다양한 데이터를 학습한다. 이러한 다양성은 모델이 특정 사례를 암기하는 것이 아니라 다양한 분야에서 재사용 가능한 추상적 표현(Abstract Representation)을 학습하도록 만든다.

대규모 계산(Large-Scale Computation) 또한 파운데이션 모델의 핵심 특징이다. 최첨단 파운데이션 모델을 학습하기 위해서는 수천 개의 GPU 또는 AI 가속기(AI Accelerator)가 수 주 또는 수 개월 동안 동시에 동작해야 한다. 분산 최적화(Distributed Optimization)는 모델 파라미터(Parameter), 학습 배치(Batch), 옵티마이저 상태(Optimizer State), 활성화 값(Activation Tensor)을 여러 장비에 분산하여 처리한다. 이러한 환경에서는 계산 성능뿐 아니라 네트워크(Network), 저장장치(Storage), 체크포인트(Checkpoint), 혼합 정밀도(Mixed Precision), 그래디언트 압축(Gradient Compression), 병렬 스케줄링(Parallel Scheduling)의 효율성 역시 매우 중요한 요소가 된다.

파운데이션 모델에서 가장 흥미로운 현상 가운데 하나는 창발성(Emergence)이다. 창발성이란 모델의 규모가 일정 수준을 넘어서면서 이전에는 전혀 수행하지 못하던 새로운 능력이 갑자기 나타나는 현상을 의미한다. 작은 모델에서는 완전히 실패하던 작업이 충분히 큰 모델에서는 별도의 명시적 프로그래밍 없이도 성공적으로 수행되기 시작한다. 이러한 현상은 복잡계(Complex Systems)에서 관찰되는 상전이(Phase Transition)와 유사하며, 점진적인 규모 증가가 질적인 능력 변화를 만들어낸다.

대표적인 창발 능력(Emergent Capability)으로는 다단계 추론(Multi-Step Reasoning), 인컨텍스트 학습(In-Context Learning), 지시 수행(Instruction Following), 코드 생성(Code Generation), 수학 문제 해결(Mathematical Problem Solving), 보지 못한 언어 간 번역(Unseen Language Translation), 상식 추론(Commonsense Reasoning), 사고 연쇄(Chain-of-Thought), 도구 사용(Tool Usage), 계획 수립(Planning), 그리고 멀티모달 이해(Multimodal Understanding)가 있다. 이러한 능력은 처음에는 갑자기 등장하지만 이후 모델 규모가 증가할수록 점진적으로 향상되는 경향을 보인다.

창발성은 마법이나 의식(Consciousness)의 존재를 의미하지 않는다. 이는 대규모 최적화 과정과 매우 높은 표현력을 가진 신경망 구조가 방대한 데이터 위에서 학습하면서 발생하는 자연스러운 결과이다. 모델 내부의 특징(Feature)은 수백만 개의 뉴런(Neuron)에 분산되어 저장되며, 사람이 이해하기 쉬운 명시적 규칙으로 표현되지 않는다. 따라서 이러한 능력은 인간 입장에서는 예상하기 어렵지만, 고차원 파라미터 공간(Parameter Space)에서는 자연스럽게 형성될 수 있다.

연구자들은 왜 창발성이 나타나는지에 대해 다양한 이론을 제시하고 있다. 일부는 모델의 표현 능력(Representational Capacity)이 일정 수준을 넘으면 훨씬 복잡한 함수를 근사할 수 있기 때문이라고 설명한다. 다른 연구에서는 최적화 과정(Optimization Dynamics) 자체가 더 우수한 해를 발견하도록 만든다고 주장한다. 또 다른 가설은 계층적인 특징 표현(Hierarchical Features), 암묵적 모듈화(Implicit Modularity), 희소 활성화(Sparse Activation), 조합적 추론(Compositional Reasoning) 등이 자연스럽게 형성된다는 점을 강조한다. 아직 완전한 이론은 존재하지 않지만, 실험적으로는 규모가 증가할수록 능력이 비선형적으로 향상된다는 사실이 반복적으로 확인되고 있다.

일반화(Generalization)는 파운데이션 모델을 정의하는 또 다른 핵심 특성이다. 기존의 머신러닝 시스템은 학습 데이터와 매우 유사한 환경에서만 높은 성능을 보이는 경우가 많았다. 반면 파운데이션 모델은 다양한 작업(Task), 산업(Domain), 환경(Environment), 문제 유형(Problem Formulation)에 걸쳐 활용 가능한 공통 표현을 학습하는 것을 목표로 한다. 이는 특정 사례를 암기하는 것이 아니라 보다 추상적인 개념을 이해하도록 만드는 과정이다.

일반화는 여러 수준에서 나타난다. 도메인 일반화(Domain Generalization)는 의료(Healthcare), 금융(Finance), 로보틱스(Robotics), 제조(Manufacturing), 교육(Education), 과학 연구(Scientific Research) 등 다양한 산업에 동일한 모델을 적용할 수 있도록 한다. 작업 일반화(Task Generalization)는 하나의 모델이 요약(Summarization), 번역(Translation), 질의응답(Question Answering), 추론(Reasoning), 코드 생성(Coding), 계획 수립(Planning), 대화(Dialogue)를 모두 수행하도록 만든다. 언어 일반화(Language Generalization)는 다양한 인간 언어를 하나의 모델에서 처리하게 하며, 멀티모달 일반화(Multimodal Generalization)는 텍스트(Text), 이미지(Image), 오디오(Audio), 비디오(Video), 포인트 클라우드(Point Cloud), 센서 데이터(Sensor Data), 로봇 관측(Robot Observation)을 통합된 표현으로 연결한다.

특히 중요한 개념은 조합적 일반화(Compositional Generalization)이다. 인간은 익숙한 개념들을 조합하여 새로운 상황에서도 문제를 해결할 수 있다. 파운데이션 모델 역시 기존에 학습한 개념들을 새로운 방식으로 결합하여 이전에 보지 못한 문제를 해결하는 능력을 점차 보여주고 있다. 예를 들어 로봇은 한 번도 본 적 없는 물체를 기존에 학습한 유사한 물체의 특징을 이용하여 적절히 조작하는 방법을 추론할 수 있다.

인컨텍스트 학습(In-Context Learning)은 일반화 능력을 가장 잘 보여주는 사례이다. 모델의 파라미터를 다시 학습시키지 않고도 입력(Context)에 몇 개의 예제만 제공하면 새로운 작업을 즉시 수행할 수 있다. 이는 추가적인 그래디언트 업데이트(Gradient Update) 없이도 매우 빠르게 새로운 작업에 적응할 수 있음을 의미한다.

전이학습(Transfer Learning)은 이러한 개념을 더욱 확장한다. 파운데이션 모델을 먼저 사전학습한 후, 소규모의 전문 데이터만을 이용하여 의료 영상(Medical Imaging), 법률 문서(Legal Documents), 금융 분석(Financial Analysis), 과학 연구(Scientific Research), 자율 로봇(Autonomous Robots), 산업 검사(Industrial Inspection) 등 다양한 분야에 맞게 미세조정(Fine-Tuning)할 수 있다. 이 방식은 데이터 요구량을 크게 줄이면서도 높은 성능과 강건성을 유지할 수 있도록 해준다.

트랜스포머(Transformer)는 현재 대부분의 파운데이션 모델에서 사용되는 핵심 구조이다. 자기 주의(Self-Attention)는 입력 시퀀스의 모든 요소 간 관계를 동시에 학습할 수 있으며, 순환 신경망(Recurrent Neural Network)보다 긴 거리의 의존성을 훨씬 효과적으로 처리할 수 있다. 다중 헤드 어텐션(Multi-Head Attention)은 여러 종류의 관계를 동시에 학습하고, 피드포워드 네트워크(Feed-Forward Network)는 표현을 점차 추상적인 의미 공간으로 변환한다.

위치 인코딩(Positional Encoding)은 순서 정보를 제공하며, 레이어 정규화(Layer Normalization)는 학습을 안정화시키고, 잔차 연결(Residual Connection)은 그래디언트 흐름을 개선한다. 이러한 요소들이 결합되어 트랜스포머는 언어(Language), 비전(Vision), 음성(Speech), 생명과학(Biology), 로보틱스(Robotics), 그리고 멀티모달 학습(Multimodal Learning) 전반에서 뛰어난 확장성을 보여주고 있다.

최근의 파운데이션 모델은 텍스트(Text)를 넘어 멀티모달(Multimodal) 영역으로 빠르게 확장되고 있다. 비전-언어 모델(Vision-Language Model)은 이미지와 자연어를 함께 이해하고, 오디오-언어 모델(Audio-Language Model)은 음성과 의미를 통합하며, 비디오-언어 모델(Video-Language Model)은 시간적 변화를 이해한다. 피지컬 AI(Physical AI) 모델은 카메라(Camera), LiDAR(라이다), 고유감각(Proprioception), 촉각(Tactile), 힘 센서(Force Sensor), 언어(Language), 그리고 로봇 행동(Action)을 하나의 통합 표현으로 학습한다. 이러한 멀티모달 표현은 단일 센서보다 훨씬 풍부한 환경 이해를 가능하게 한다.

로보틱스(Robotics)에서는 파운데이션 모델이 특히 혁신적인 의미를 가진다. 기존 로봇 시스템은 인식(Perception), 객체 탐지(Object Detection), 계획(Planning), 위치 추정(Localization), 지도 작성(Mapping), 제어(Control)를 각각 독립적인 모듈로 구현하였다. 이러한 방식은 구조화된 환경에서는 매우 효과적이지만 새로운 작업이 등장할 때마다 많은 엔지니어링 작업이 필요했다.

반면 로봇 파운데이션 모델(Robot Foundation Model)은 인식, 추론, 계획, 제어를 하나의 통합 표현으로 학습한다. 센서 관측(Sensory Observation), 언어 명령(Language Instruction), 환경(Context), 과거 경험(Experience), 미래 행동(Action)을 하나의 모델 안에서 연결하여 다양한 작업과 환경에 적응할 수 있도록 한다.

피지컬 AI에서는 규모(Scale)가 더욱 중요하다. 실제 환경은 조명(Lighting), 물체 형상(Object Geometry), 인간 행동(Human Behavior), 표면 재질(Material), 센서 노이즈(Sensor Noise), 기계적 마모(Mechanical Wear), 예기치 않은 장애물(Unexpected Obstacles), 그리고 끊임없이 변화하는 환경(Dynamic Environment)으로 구성되어 있다. 따라서 다양한 로봇, 시뮬레이션, 실제 환경을 모두 포함하는 대규모 학습은 이러한 변화를 견디는 강건성을 크게 향상시킨다.

체화(Embodiment)는 언어 모델과 로봇 모델을 구분하는 가장 중요한 요소 가운데 하나이다. 물리적 에이전트(Physical Agent)는 물리 법칙(Physics)을 따라야 하며, 균형(Balance)을 유지하고, 충돌(Collision)을 피하며, 안전 제약(Safety Constraint)을 만족하고, 실시간 계산(Real-Time Computation)을 수행해야 한다. 따라서 로봇용 파운데이션 모델은 기하학적 추론(Geometric Reasoning), 시간 예측(Temporal Prediction), 불확실성 추정(Uncertainty Estimation), 제어 이론(Control Theory), 물리적 상호작용(Physical Interaction)을 모두 통합해야 한다.

최근에는 서로 다른 로봇 간 일반화(Cross-Embodiment Generalization)가 중요한 연구 목표가 되고 있다. 서로 다른 센서(Sensor), 운동학(Kinematics), 액추에이터(Actuator), 크기(Size)를 가진 로봇이라도 하나의 공통 표현을 공유하면 기존 경험을 새로운 로봇에서도 활용할 수 있다. 이는 새로운 하드웨어가 등장할 때마다 처음부터 데이터를 다시 수집해야 하는 부담을 크게 줄여준다.

시뮬레이션(Simulation)은 피지컬 AI의 규모 확장에 필수적인 기술이다. 실제 로봇에서 수십억 번의 상호작용을 수집하는 것은 현실적으로 어렵기 때문에, 고품질 시뮬레이터(Simulator)를 이용하여 대량의 합성 데이터(Synthetic Data)를 생성한다. 이후 제한된 실제 데이터로 미세조정(Fine-Tuning)을 수행함으로써 시뮬레이션과 현실 간의 차이(Sim-to-Real Gap)를 최소화한다.

파운데이션 모델의 평가는 기존보다 훨씬 복잡하다. 기존 벤치마크(Benchmark)는 하나의 작업만 측정했지만, 파운데이션 모델은 추론(Reasoning), 강건성(Robustness), 신뢰도(Calibration), 안전성(Safety), 공정성(Fairness), 효율성(Efficiency), 해석 가능성(Interpretability), 전이학습(Transfer Learning), 적응성(Adaptability) 등을 함께 평가해야 한다. 로봇에서는 조작 성공률(Manipulation Success), 내비게이션(Navigation), 충돌 회피(Collision Avoidance), 장기 계획(Long-Horizon Planning), 에너지 효율(Energy Efficiency), 장애 복구(Failure Recovery), 인간-로봇 협업(Human-Robot Collaboration)도 중요한 평가 요소가 된다.

놀라운 발전에도 불구하고 파운데이션 모델은 여전히 여러 한계를 가지고 있다. 환각(Hallucination), 분포 변화(Distribution Shift), 막대한 학습 비용(Training Cost), 높은 에너지 소비(Energy Consumption), 데이터 편향(Data Bias), 보안(Security), 저작권(Copyright), 개인정보 보호(Privacy), 해석 가능성 부족(Interpretability) 등은 앞으로 해결해야 할 중요한 연구 과제이다.

또한 단순히 규모만 계속 증가시키는 것이 모든 문제를 해결하지는 않는다. 앞으로의 파운데이션 모델은 더 큰 모델뿐 아니라 더 나은 데이터(Data Quality), 효율적인 구조(Efficient Architecture), 구조화된 메모리(Structured Memory), 명시적 추론(Explicit Reasoning), 인과 모델(Causal Modeling), 검색 증강(Retrieval Augmentation), 월드 모델(World Model), 능동 학습(Active Learning), 지속 학습(Continual Learning), 불확실성 추정(Uncertainty Estimation) 등을 함께 발전시켜야 한다.

궁극적으로 파운데이션 모델의 목표는 단순히 더 큰 신경망을 만드는 것이 아니다. 파운데이션 모델은 인식(Perception), 추론(Reasoning), 계획(Planning), 의사소통(Communication), 의사결정(Decision Making), 물리적 상호작용(Physical Interaction)을 수행하는 범용 인지 기반(Common Cognitive Infrastructure)으로 발전하고 있다. 피지컬 AI에서는 이러한 모델이 여러 로봇의 경험을 공유하고, 다양한 환경에 일반화하며, 멀티모달 센서 정보를 이해하고, 새로운 작업에도 최소한의 추가 학습만으로 적응하는 핵심 지능 계층(Intelligence Layer)이 될 것으로 기대된다.

앞으로 피지컬 AI가 발전함에 따라 파운데이션 모델은 자율 시스템(Autonomous System)의 중심 지능 계층으로 자리 잡을 것이다. 운영체제(Operating System)가 다양한 소프트웨어의 공통 기반을 제공하듯이, 파운데이션 모델은 미래의 지능형 에이전트(Intelligent Agent)를 위한 공통 인지 인프라(Common Cognitive Infrastructure)를 제공하게 된다. 대규모 표현 학습(Large-Scale Representation Learning), 창발 능력(Emergent Capability), 그리고 뛰어난 일반화(Generalization)의 결합은 미래의 자율 로봇(Autonomous Robot), 체화 인공지능(Embodied AI), 그리고 범용 피지컬 인텔리전스(General-Purpose Physical Intelligence)를 구축하는 핵심 기술적 기반이 될 것이다. 또한 이 장은 이후 이어지는 Volume 40의 트랜스포머 아키텍처(Transformer Architecture), 멀티모달 학습(Multimodal Learning), 월드 모델(World Model), 로봇 파운데이션 모델(Robot Foundation Model), 액션 토크나이제이션(Action Tokenization), 대규모 사전학습(Large-Scale Pretraining), 피지컬 AI 학습 파이프라인(Physical AI Training Pipeline), 안전성(Safety), 적응(Adaptation), 그리고 실제 산업 적용(Real-World Deployment)을 이해하기 위한 개념적 기반을 제공한다.

##  

## 1.2 History of Foundation Models BERT GPT to Gemini

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

The history of foundation models is the history of artificial intelligence moving from narrowly engineered systems toward large, reusable models capable of supporting many tasks, domains, and modalities. Earlier machine learning systems were usually trained for a single objective with a dedicated dataset. A sentiment classifier classified sentiment, a translation model translated between fixed language pairs, and an image recognizer identified objects from predefined categories. These systems could be highly accurate within their intended boundaries, but their knowledge was difficult to transfer. Each new application normally required a new dataset, a new training process, and often a new architecture.

Foundation models changed this development pattern. Instead of building intelligence separately for every task, researchers began training large neural networks on broad collections of data and then adapting them to specific applications. The central idea was that a sufficiently large model could first learn general representations of language, images, code, audio, video, or physical interaction. These representations could then be reused through fine-tuning, prompting, retrieval, or task-specific adaptation. The progression from BERT and GPT to multimodal systems such as Gemini illustrates how this idea expanded from text representation learning into a general architecture for machine intelligence.

The intellectual foundations of this transition were established before the term foundation model became widely used. Early distributed word representations such as Word2Vec and GloVe demonstrated that neural networks could map words into continuous vector spaces. Words appearing in similar contexts developed similar representations, and semantic relationships could be reflected geometrically. These systems were important because they replaced sparse symbolic features with dense learned representations. However, each word was usually assigned one fixed vector regardless of context. The word "bank," for example, had the same representation whether it referred to a financial institution or the side of a river.

Recurrent neural networks, long short-term memory networks, and sequence-to-sequence models extended representation learning to ordered language. They enabled progress in machine translation, speech recognition, and text generation by processing sequences token by token. Attention mechanisms later allowed a model to focus dynamically on relevant parts of an input sequence. Neural machine translation became one of the first areas in which attention showed that learned alignment between words and phrases could substantially improve performance.

A major turning point occurred with the introduction of the transformer architecture in 2017. The transformer removed recurrence and relied primarily on self-attention. Every token could directly interact with every other token in the sequence, allowing the model to capture long-range dependencies more effectively. Because transformers could process many tokens in parallel, they were also far more compatible with large-scale GPU training than recurrent architectures. This combination of representational flexibility and computational scalability created the technical basis for modern foundation models.

The first generation of transformer systems was often designed around individual tasks such as translation. The next breakthrough came when researchers realized that transformer models could be pre-trained on large amounts of unlabeled text and then transferred to many downstream tasks. Instead of using a transformer only as part of a task-specific pipeline, the transformer itself became a general language representation engine.

The Generative Pre-trained Transformer, or GPT, was introduced as an early demonstration of this approach. GPT used a decoder-only transformer trained with an autoregressive objective. Given a sequence of tokens, it learned to predict the next token. After pre-training on a large text corpus, the model could be fine-tuned for classification, question answering, textual entailment, and other natural language processing tasks. The importance of GPT was not merely its performance. It demonstrated that generative pre-training could produce reusable language knowledge that transferred across tasks.

BERT, introduced shortly afterward, represented another major branch of transformer-based pre-training. BERT used an encoder-only architecture and was trained primarily through masked language modeling. Some tokens in an input sentence were hidden, and the model learned to reconstruct them using information from both the left and right contexts. This bidirectional learning made BERT especially effective at understanding relationships within text. It achieved strong results across question answering, named entity recognition, sentiment analysis, sentence classification, and many other language understanding benchmarks.

BERT helped establish the pre-train-and-fine-tune paradigm as the dominant approach in natural language processing. Organizations no longer needed to train every language model from the beginning. They could begin with a pre-trained BERT model and adapt it using a comparatively small labeled dataset. Numerous variants followed, including models optimized for efficiency, multilingual processing, domain-specific language, longer documents, biomedical text, legal documents, and scientific literature.

BERT and GPT represented different but complementary views of pre-training. BERT emphasized contextual understanding through bidirectional encoding, while GPT emphasized generation through autoregressive prediction. Encoder-based models became highly effective for classification and representation tasks. Decoder-only models became increasingly powerful for open-ended generation and later proved capable of performing many tasks through natural language prompting. Encoder-decoder models such as T5 explored a third direction by representing every language problem as a text-to-text transformation.

The next major stage was defined by scale. GPT-2 showed that increasing model size and training data could produce more coherent long-form text and broader zero-shot abilities. The model could continue passages, imitate styles, answer questions, and generate plausible documents without being explicitly trained for every individual use case. Its behavior suggested that next-token prediction, when performed at sufficient scale, could lead to much richer capabilities than previously expected.

GPT-3 greatly expanded this idea. With a much larger parameter count and training corpus, it demonstrated few-shot and in-context learning. Users could describe a task or provide several examples directly in the prompt, and the model could often infer the intended pattern without changing its parameters. This was a profound shift. Adaptation no longer always required fine-tuning. The model could temporarily learn a task from context, using the prompt as an interface for programming behavior in natural language.

In-context learning changed how developers interacted with AI systems. Instead of building a separate output layer and training it with labeled data, they could express requirements through instructions, demonstrations, templates, and constraints. Prompt engineering emerged as a practical discipline. At the same time, the limitations of raw pre-trained models became more visible. Large language models could generate fluent but incorrect statements, ignore user intent, reproduce biases from training data, or produce unsafe outputs.

These challenges led to the development of instruction tuning and alignment methods. Models were fine-tuned on datasets containing instructions and desired responses so that they would behave more like useful assistants. Reinforcement learning from human feedback introduced preference-based optimization. Human evaluators compared model outputs, reward models learned from those comparisons, and the language model was optimized to produce responses that people judged more helpful, relevant, and safe.

The emergence of conversational systems made foundation models accessible to a much wider population. The model was no longer only a component embedded inside a specialized application. It became a general-purpose interface capable of answering questions, drafting documents, explaining concepts, writing code, summarizing information, assisting with analysis, and interacting across many domains through dialogue. This transition helped transform foundation models from a research topic into a major computing platform.

Parallel developments occurred in computer vision. Convolutional neural networks had dominated image recognition for many years, but the Vision Transformer demonstrated that transformer architectures could also process images by dividing them into patches and treating those patches as tokens. Large-scale image pre-training made it possible to transfer visual representations across classification, detection, segmentation, and generation tasks.

Contrastive language-image pre-training further connected visual and textual intelligence. Models such as CLIP learned to align images and natural language descriptions within a shared embedding space. Rather than being restricted to a fixed set of object categories, the model could recognize visual concepts using arbitrary textual prompts. This introduced open-vocabulary perception and demonstrated that language could serve as a flexible interface for visual understanding.

Generative image models also advanced rapidly. Autoregressive models, variational methods, and diffusion models learned to produce high-quality images from textual descriptions. Although image generation and language modeling used different training objectives, both reflected the same foundation model principle: large-scale pre-training on diverse data could create reusable generative capabilities that supported many downstream applications.

The boundary between separate language and vision models gradually began to disappear. Early multimodal systems often connected a pre-trained visual encoder to a language model through projection layers or cross-attention modules. These systems could describe images, answer visual questions, read text inside images, compare objects, and reason about scenes. Later models were trained on broader mixtures of text, images, diagrams, documents, audio, and video, allowing multimodal capabilities to become part of the foundation model rather than an external extension.

Large language model development also became increasingly connected to tool use. Models learned to call search systems, calculators, code interpreters, databases, and external APIs. This addressed an important limitation of purely parametric knowledge. A model's internal parameters contain compressed statistical knowledge from training, but they cannot always provide current information, exact calculations, or access to private data. Tool-augmented foundation models combined generative reasoning with external systems capable of retrieving and executing information.

Retrieval-augmented generation became especially important for enterprise applications. Instead of relying only on the model's memorized knowledge, the system retrieved relevant documents from external storage and inserted them into the model's context. This improved factual grounding, enabled access to proprietary information, and made responses easier to trace to source documents. Foundation models increasingly became components within larger systems that included retrieval, memory, planning, verification, and software execution.

The development of GPT-4 marked another significant stage in this history. It demonstrated improved reasoning, instruction following, coding, and multimodal understanding compared with earlier generations. Although model scale remained important, progress was no longer explained only by parameter count. Data quality, training objectives, alignment methods, synthetic data, inference strategies, system architecture, and safety evaluation all became critical parts of foundation model development.

At the same time, other model families accelerated innovation. Encoder-decoder systems, open-weight language models, multilingual models, mixture-of-experts architectures, long-context models, and domain-specialized foundation models expanded the ecosystem. Competition among research organizations encouraged rapid progress in efficiency, openness, deployment flexibility, reasoning capability, and multimodal integration.

Google's development path from BERT to Gemini illustrates the broader transition from language representation models to multimodal general-purpose systems. BERT was designed primarily to understand text through bidirectional contextual encoding. Later large language models expanded generative and conversational capabilities. Gemini was developed as a multimodal model family intended to process and reason across text, images, audio, video, and code.

The importance of Gemini lies not simply in adding image input to a language model. It represents the movement toward models designed around multimodality as a core property. In a natively multimodal approach, different data types are incorporated into the training process so that the model can learn relationships across modalities. Text can explain an image, video can demonstrate an action, audio can provide temporal and environmental information, and code can represent executable procedures. A shared model can use these modalities together rather than treating them as unrelated input channels.

The progression from BERT to Gemini therefore reflects several overlapping transitions. Models moved from task-specific training to general pre-training, from small datasets to internet-scale corpora, from millions of parameters to large distributed architectures, from fine-tuning to prompting and in-context learning, from text-only systems to multimodal models, and from static prediction to tool-using interactive agents.

This history also reveals that foundation models are not a single architecture or product category. They are a broader engineering paradigm. A model becomes foundational when its learned representations and capabilities can be adapted across many applications. Language models, vision models, multimodal models, scientific models, world models, and robot foundation models can all follow this principle even when their data and objectives differ.

The expansion toward Physical AI is the next stage of this development. Text and image models operate mainly on digital information, while Physical AI systems must connect perception to action. A robot foundation model must process camera streams, depth images, point clouds, language commands, joint states, force measurements, tactile signals, and previous actions. It must then generate behavior that respects geometry, dynamics, safety limits, timing constraints, and the capabilities of a specific robot body.

The history of language foundation models provides several lessons for robotics. First, diverse pre-training data can produce representations that transfer across tasks. Second, scale can generate capabilities that are not visible in smaller systems. Third, natural language can act as a universal interface for expressing goals. Fourth, multimodal learning allows concepts to be grounded across observation and description. Fifth, adaptation methods such as prompting, fine-tuning, and low-rank updates can specialize a general model without retraining it from the beginning.

However, robot foundation models also face challenges that language models do not. Incorrect text may confuse a user, but an incorrect robot action can damage equipment or injure a person. Physical data is more expensive to collect than internet text. Robot platforms differ in kinematics, sensor layouts, coordinate systems, action spaces, and control frequencies. Real-world environments contain noise, friction, delays, occlusion, mechanical wear, and unexpected human behavior. Consequently, the historical progression from BERT to Gemini is only the starting point for Physical AI rather than a complete solution.

Future foundation models are likely to combine multimodal perception, long-term memory, world modeling, causal reasoning, planning, action generation, and continuous learning. They may use language models for abstract reasoning, vision models for scene understanding, world models for predicting possible futures, and action policies for interacting with the physical environment. Instead of one monolithic network performing every operation, practical systems may coordinate multiple specialized models through a common representation and control architecture.

Efficiency will also shape the next era. Early foundation model progress was driven largely by increasing scale, but industrial deployment requires lower latency, reduced energy consumption, predictable behavior, and operation on edge hardware. Techniques such as quantization, pruning, distillation, mixture-of-experts routing, speculative decoding, sparse attention, caching, and hardware-aware optimization will determine whether foundation models can move from large data centers into robots, vehicles, drones, factories, and embedded devices.

The historical journey from BERT and GPT to Gemini demonstrates that the most important change is not simply that models have become larger. Their role within computing has changed. BERT established reusable contextual representations. GPT demonstrated that generative pre-training could become a general interface for tasks. Large-scale models introduced in-context learning and emergent capabilities. Instruction tuning and human feedback made models more usable. Vision-language research expanded intelligence beyond text. Multimodal systems such as Gemini brought multiple forms of information into a more unified foundation.

This progression provides the historical context for Physical AI foundation models. Future robot intelligence will not be created by copying a language model directly into a machine. It will emerge by extending the foundation model paradigm into perception, embodiment, world prediction, action, and safety. The transition from BERT to GPT and Gemini therefore represents the first part of a broader transformation: the movement from models that understand and generate information to models that can interpret, reason about, and eventually act within the physical world.

파운데이션 모델(Foundation Model)의 역사는 인공지능(AI, Artificial Intelligence)이 특정 목적만 수행하는 개별 시스템에서 다양한 작업과 여러 도메인, 그리고 여러 데이터 형태를 동시에 처리할 수 있는 대규모 범용 모델로 발전해 온 과정이라 할 수 있다. 초기의 머신러닝(Machine Learning) 시스템은 대부분 하나의 목적을 위해 하나의 데이터셋만을 사용하여 학습되었다. 감성 분석기(Sentiment Classifier)는 감성을 분류하고, 번역 모델(Translation Model)은 정해진 언어 쌍만 번역하며, 이미지 인식기(Image Recognizer)는 미리 정의된 객체만 식별하였다. 이러한 시스템들은 자신이 설계된 범위 안에서는 높은 정확도를 보였지만, 학습한 지식을 다른 문제로 이전하기는 매우 어려웠다. 새로운 응용 분야가 등장할 때마다 새로운 데이터셋과 새로운 학습 과정, 그리고 경우에 따라 새로운 모델 구조까지 다시 설계해야 했다.

파운데이션 모델은 이러한 개발 방식을 근본적으로 변화시켰다. 각각의 작업마다 별도의 인공지능을 만드는 대신, 하나의 거대한 신경망(Neural Network)을 다양한 데이터로 먼저 학습시키고 이후 여러 응용 분야에 맞게 적응시키는 방식이 등장한 것이다. 핵심 개념은 충분히 큰 모델이 언어(Language), 이미지(Image), 코드(Code), 오디오(Audio), 비디오(Video), 물리적 상호작용(Physical Interaction)과 같은 다양한 정보를 먼저 일반적으로 학습한 뒤, 이를 미세조정(Fine-Tuning), 프롬프팅(Prompting), 검색(Retrieval), 또는 작업별 적응(Task-Specific Adaptation)을 통해 다양한 문제에 활용하는 것이다. BERT와 GPT에서 시작하여 Gemini와 같은 멀티모달(Multimodal) 시스템으로 발전한 과정은 이러한 개념이 텍스트 표현 학습(Text Representation Learning)에서 범용 인공지능 플랫폼으로 확장된 대표적인 사례이다.

파운데이션 모델이라는 용어가 널리 사용되기 이전에도 이러한 변화의 기반이 되는 연구들은 이미 진행되고 있었다. 대표적인 예가 Word2Vec과 GloVe 같은 분산 단어 표현(Distributed Word Representation)이다. 이러한 모델은 단어를 연속적인 벡터 공간(Vector Space)에 매핑하여 의미가 비슷한 단어는 가까운 위치에 배치되도록 학습하였다. 이는 희소한 기호 표현(Sparse Symbolic Feature)을 밀집 표현(Dense Representation)으로 대체한 중요한 발전이었다. 그러나 하나의 단어는 문맥(Context)과 상관없이 항상 동일한 벡터를 사용하였다. 예를 들어 "bank"라는 단어는 금융기관(Financial Institution)과 강둑(River Bank)이라는 서로 다른 의미를 가져도 동일한 표현을 사용하였다.

순환 신경망(Recurrent Neural Network), 장단기 메모리(Long Short-Term Memory), 그리고 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델은 이러한 표현 학습을 순차 데이터(Sequential Data)까지 확장하였다. 이들은 단어를 하나씩 순서대로 처리하면서 기계 번역(Machine Translation), 음성 인식(Speech Recognition), 텍스트 생성(Text Generation) 등의 분야에서 큰 발전을 이루었다. 이후 어텐션(Attention) 메커니즘이 등장하면서 입력 문장 전체에서 필요한 부분을 동적으로 선택할 수 있게 되었고, 특히 신경망 기반 기계 번역(Neural Machine Translation)의 성능이 크게 향상되었다.

진정한 전환점은 2017년에 발표된 트랜스포머(Transformer) 구조였다. 트랜스포머는 순환 구조를 제거하고 자기 주의(Self-Attention)를 중심으로 설계되었다. 이를 통해 입력 시퀀스(Sequence)의 모든 토큰(Token)이 서로 직접 정보를 교환할 수 있었으며, 장거리 의존성(Long-Range Dependency)을 훨씬 효과적으로 학습할 수 있었다. 또한 병렬 처리(Parallel Processing)가 가능해져 GPU 클러스터(GPU Cluster)를 활용한 대규모 학습이 가능해졌다. 이러한 표현 능력과 계산 효율성의 결합은 현대 파운데이션 모델의 기술적 기반이 되었다.

초기의 트랜스포머는 주로 기계 번역과 같은 개별 작업을 위해 설계되었다. 그러나 이후 연구자들은 트랜스포머를 방대한 비정형 텍스트(Unlabeled Text)로 먼저 학습시키고, 이를 다양한 작업으로 전이(Transfer)할 수 있다는 사실을 발견하였다. 이 시점부터 트랜스포머는 단순한 작업용 모델이 아니라 범용 언어 표현 엔진(General Language Representation Engine)으로 발전하기 시작하였다.

생성형 사전학습 트랜스포머(Generative Pre-trained Transformer), 즉 GPT는 이러한 접근법을 대표적으로 보여준 모델이었다. GPT는 디코더 전용(Decoder-Only) 트랜스포머 구조를 사용하였으며, 이전 토큰(Token)을 기반으로 다음 토큰을 예측하는 자기회귀(Auto-Regressive) 방식으로 학습되었다. 대규모 텍스트 데이터로 사전학습(Pre-training)을 수행한 후, 텍스트 분류(Classification), 질의응답(Question Answering), 자연어 추론(Textual Entailment) 등 다양한 작업에 미세조정(Fine-Tuning)을 적용할 수 있었다. GPT의 가장 중요한 의미는 생성 능력 자체보다, 하나의 사전학습 모델이 여러 작업에 재사용될 수 있음을 보여주었다는 점이었다.

이후 등장한 BERT는 트랜스포머 기반 사전학습의 또 다른 중요한 방향을 제시하였다. BERT는 인코더 전용(Encoder-Only) 구조를 사용하며, 마스크드 언어 모델링(Masked Language Modeling)을 핵심 학습 목표로 삼았다. 입력 문장의 일부 단어를 가린 후, 앞과 뒤의 문맥(Context)을 동시에 이용하여 이를 복원하도록 학습하였다. 이러한 양방향(Bidirectional) 학습은 문장의 의미와 단어 간 관계를 더욱 깊이 이해할 수 있도록 만들었다. BERT는 질의응답, 개체명 인식(Named Entity Recognition), 감성 분석(Sentiment Analysis), 문장 분류(Sentence Classification) 등 다양한 자연어 이해(Natural Language Understanding) 벤치마크에서 매우 우수한 성능을 기록하였다.

BERT는 사전학습 후 미세조정(Pre-train and Fine-Tune)이라는 새로운 표준 개발 방식을 확립하였다. 이제 기업이나 연구기관은 처음부터 언어 모델을 학습할 필요 없이, 이미 학습된 BERT를 가져와 소규모 데이터셋만 이용하여 자신의 응용 분야에 맞게 쉽게 적응시킬 수 있었다. 이후 효율성을 높인 다양한 변형 모델, 다국어 모델(Multilingual Model), 긴 문서를 처리하는 모델(Long-Document Model), 의료(Biomedical), 법률(Legal), 과학(Scientific Literature) 전용 모델 등이 계속 등장하였다.

BERT와 GPT는 서로 다른 철학을 가진 모델이었다. BERT는 양방향 문맥 이해(Contextual Understanding)에 집중한 반면, GPT는 자기회귀 기반의 생성(Generation)에 초점을 맞추었다. 인코더 기반 모델은 문장 이해와 분류에서 뛰어난 성능을 보였으며, 디코더 기반 모델은 자연스러운 텍스트 생성과 개방형 대화(Open-Ended Generation)에 강점을 보였다. 이후 T5(Text-to-Text Transfer Transformer)와 같은 인코더-디코더(Encoder-Decoder) 모델은 모든 자연어 문제를 텍스트를 입력받아 텍스트를 출력하는 하나의 문제로 통합하여 처리하는 새로운 접근법을 제시하였다.

다음 단계는 규모 확장(Scaling)이었다. GPT-2는 모델 크기와 데이터 규모를 크게 증가시킴으로써 훨씬 자연스럽고 긴 문장을 생성할 수 있게 되었으며, 별도의 학습 없이 새로운 작업을 수행하는 제로샷(Zero-Shot) 능력도 보여주었다. 문장 이어쓰기, 문체 모방, 질의응답 등 다양한 작업을 수행하면서, 단순한 다음 토큰 예측(Next Token Prediction)만으로도 충분한 규모에서는 매우 풍부한 능력이 나타날 수 있음을 보여주었다.

GPT-3는 이러한 개념을 더욱 발전시켰다. 훨씬 더 큰 파라미터(Parameter)와 데이터셋을 이용하여 학습된 GPT-3는 퓨샷 학습(Few-Shot Learning)과 인컨텍스트 학습(In-Context Learning)을 본격적으로 보여주었다. 사용자는 모델의 파라미터를 다시 학습시키지 않고도 프롬프트(Prompt)에 몇 개의 예시만 제공하면 새로운 작업을 수행할 수 있었다. 이는 미세조정(Fine-Tuning)이 항상 필요한 것이 아니라, 자연어 자체가 프로그래밍 인터페이스(Programming Interface)가 될 수 있다는 새로운 패러다임을 제시하였다.

인컨텍스트 학습(In-Context Learning)은 개발자와 인공지능의 상호작용 방식을 근본적으로 변화시켰다. 이제 별도의 출력 계층(Output Layer)을 설계하고 라벨링된 데이터를 이용해 학습할 필요 없이, 자연어 명령(Instruction), 예시(Demonstration), 템플릿(Template), 제약조건(Constraint)을 프롬프트 안에 작성하는 것만으로 모델의 동작을 제어할 수 있게 되었다. 이에 따라 프롬프트 엔지니어링(Prompt Engineering)이라는 새로운 분야가 등장하였다. 동시에 대규모 언어 모델은 유창하지만 사실과 다른 내용을 생성하거나(Hallucination), 사용자 의도를 잘못 이해하거나, 학습 데이터의 편향(Bias)을 그대로 반영하는 문제도 드러나기 시작하였다.

이러한 문제를 해결하기 위해 지시 미세조정(Instruction Tuning)과 정렬(Alignment) 기술이 발전하였다. 다양한 명령과 올바른 응답을 포함한 데이터셋으로 모델을 다시 학습시켜 사용자의 의도를 보다 정확히 따르도록 만들었다. 이어 인간 피드백 기반 강화학습(Reinforcement Learning from Human Feedback, RLHF)이 도입되었다. 사람의 선호도를 학습한 보상 모델(Reward Model)을 이용하여 언어 모델을 최적화함으로써, 보다 도움이 되고 안전하며 사용자의 기대에 맞는 응답을 생성하도록 개선하였다.

대화형 시스템(Conversational System)의 등장은 파운데이션 모델을 연구실 밖으로 확장시키는 계기가 되었다. 모델은 더 이상 특정 응용 프로그램 안에 포함된 하나의 부품이 아니라, 질문에 답하고, 문서를 작성하며, 개념을 설명하고, 코드를 생성하며, 정보를 요약하고, 분석을 수행하는 범용 인터페이스(General-Purpose Interface)가 되었다. 이를 통해 파운데이션 모델은 연구 주제를 넘어 하나의 새로운 컴퓨팅 플랫폼(Computing Platform)으로 자리 잡기 시작하였다.

동시에 컴퓨터 비전(Computer Vision) 분야에서도 큰 변화가 일어났다. 오랫동안 합성곱 신경망(Convolutional Neural Network)이 이미지 인식의 중심 기술이었지만, 비전 트랜스포머(Vision Transformer)는 이미지를 여러 개의 패치(Patch)로 나누어 각각을 토큰(Token)처럼 처리할 수 있음을 보여주었다. 이를 통해 이미지에서도 트랜스포머 구조가 매우 뛰어난 성능을 낼 수 있다는 사실이 입증되었다.

이후 CLIP(Contrastive Language-Image Pre-training)과 같은 모델은 이미지와 텍스트를 하나의 공통 임베딩 공간(Shared Embedding Space)으로 연결하였다. 이제 모델은 미리 정의된 객체만 인식하는 것이 아니라, 자연어 설명(Text Prompt)을 이용하여 새로운 객체도 이해할 수 있게 되었다. 이는 개방형 어휘(Open-Vocabulary) 인식이라는 새로운 개념을 제시하였으며, 언어가 시각 인식의 유연한 인터페이스가 될 수 있음을 보여주었다.

생성형 이미지 모델(Generative Image Model)도 빠르게 발전하였다. 자기회귀 모델(Auto-Regressive Model), 변분 모델(Variational Model), 확산 모델(Diffusion Model)은 텍스트 설명만으로 매우 높은 품질의 이미지를 생성할 수 있게 되었다. 비록 언어 모델과 이미지 생성 모델은 서로 다른 학습 목표를 사용하지만, 모두 대규모 사전학습을 통해 다양한 작업에 재사용 가능한 생성 능력을 얻는다는 파운데이션 모델의 공통 철학을 따르고 있다.

이후 언어 모델과 비전 모델의 경계는 점차 사라지기 시작하였다. 초기의 멀티모달 시스템은 시각 인코더(Visual Encoder)와 언어 모델을 단순히 연결하는 구조였지만, 이후에는 텍스트, 이미지, 문서, 오디오, 비디오를 동시에 학습하는 진정한 멀티모달 모델이 등장하였다. 이러한 모델은 이미지를 설명하고, 시각 질문에 답하며, 문서 속 글자를 읽고, 장면을 이해하고, 여러 종류의 데이터를 하나의 통합 표현으로 학습할 수 있게 되었다.

대규모 언어 모델은 점차 다양한 외부 도구(Tool)와도 연결되기 시작하였다. 검색(Search), 계산기(Calculator), 코드 실행기(Code Interpreter), 데이터베이스(Database), 외부 API 등을 호출할 수 있게 되면서, 단순히 내부 파라미터에 저장된 지식만 사용하는 것이 아니라 외부 시스템과 협력하여 문제를 해결하는 방향으로 발전하였다.

검색 증강 생성(Retrieval-Augmented Generation, RAG)은 특히 기업 환경에서 중요한 기술이 되었다. 모델 내부의 기억만 사용하는 대신, 필요한 문서를 외부 저장소에서 검색하여 컨텍스트(Context)에 함께 제공함으로써 보다 정확하고 신뢰할 수 있는 응답을 생성할 수 있게 되었다. 이로 인해 파운데이션 모델은 검색, 메모리(Memory), 계획(Planning), 검증(Verification), 소프트웨어 실행(Software Execution) 등을 포함하는 복합 시스템의 핵심 구성 요소로 발전하였다.

GPT-4는 이러한 발전 과정의 또 하나의 중요한 이정표였다. GPT-4는 이전 세대보다 추론(Reasoning), 지시 수행(Instruction Following), 코드 생성(Coding), 멀티모달 이해(Multimodal Understanding)에서 크게 향상된 성능을 보였다. 또한 모델의 성능은 단순한 파라미터 수뿐 아니라 데이터 품질(Data Quality), 학습 목표(Training Objective), 정렬 기술(Alignment), 합성 데이터(Synthetic Data), 추론 전략(Inference Strategy), 시스템 구조(System Architecture), 안전성 평가(Safety Evaluation)에 의해 함께 결정된다는 사실도 보여주었다.

동시에 다양한 모델 계열도 빠르게 발전하였다. 인코더-디코더 모델, 공개 가중치(Open-Weight) 모델, 다국어 모델(Multilingual Model), 전문가 혼합(Mixture of Experts), 긴 컨텍스트(Long Context) 모델, 그리고 산업별 특화 파운데이션 모델이 등장하면서 생태계는 더욱 다양해졌다. 연구기관 간 경쟁은 효율성(Efficiency), 개방성(Openness), 배포 유연성(Deployment Flexibility), 추론 능력(Reasoning), 그리고 멀티모달 통합(Multimodal Integration)의 발전을 더욱 가속화하였다.

Google의 BERT에서 Gemini로 이어지는 발전 과정은 언어 표현 모델(Language Representation Model)에서 멀티모달 범용 모델(Multimodal General-Purpose Model)로 진화한 대표적인 사례이다. BERT는 양방향 문맥 이해에 초점을 맞추었고, 이후 대규모 언어 모델은 생성과 대화 능력을 확장하였다. Gemini는 텍스트(Text), 이미지(Image), 오디오(Audio), 비디오(Video), 코드(Code)를 동시에 처리하고 추론할 수 있는 멀티모달 모델 계열(Multimodal Model Family)로 설계되었다.

Gemini의 가장 큰 의미는 단순히 이미지 입력을 추가한 것이 아니라, 멀티모달을 모델의 핵심 속성(Core Property)으로 설계했다는 점이다. 서로 다른 데이터 유형을 하나의 학습 과정 안에서 통합하여, 텍스트는 이미지를 설명하고, 비디오는 행동을 보여주며, 오디오는 시간 정보와 환경 정보를 제공하고, 코드는 실행 가능한 절차를 표현하는 등 여러 형태의 정보를 동시에 이해하고 활용할 수 있도록 만들었다.

결국 BERT에서 Gemini까지의 발전은 여러 변화가 동시에 이루어진 과정이었다. 작업별 모델(Task-Specific Model)에서 범용 사전학습(General Pre-training)으로, 소규모 데이터에서 인터넷 규모 데이터(Internet-Scale Corpus)로, 수백만 개 파라미터에서 대규모 분산 모델(Large Distributed Architecture)로, 미세조정(Fine-Tuning)에서 프롬프팅(Prompting)과 인컨텍스트 학습(In-Context Learning)으로, 텍스트 전용 모델에서 멀티모달 모델(Multimodal Model)로, 그리고 단순 예측 시스템에서 외부 도구를 활용하는 지능형 에이전트(Intelligent Agent)로 발전한 과정이었다.

이러한 역사는 파운데이션 모델이 특정한 하나의 구조나 제품을 의미하지 않는다는 사실도 보여준다. 중요한 것은 모델이 학습한 표현과 능력이 다양한 작업으로 쉽게 이전될 수 있는가이다. 언어 모델(Language Model), 비전 모델(Vision Model), 멀티모달 모델(Multimodal Model), 과학 모델(Scientific Model), 월드 모델(World Model), 로봇 파운데이션 모델(Robot Foundation Model)은 모두 이러한 철학을 공유할 수 있다.

피지컬 AI(Physical AI)는 이러한 발전의 다음 단계라고 볼 수 있다. 텍스트와 이미지를 처리하는 모델은 디지털 정보만 다루지만, 피지컬 AI는 인식(Perception)과 행동(Action)을 연결해야 한다. 로봇 파운데이션 모델은 카메라(Camera), 깊이 영상(Depth Image), 포인트 클라우드(Point Cloud), 언어 명령(Language Command), 관절 상태(Joint State), 힘 센서(Force Sensor), 촉각(Tactile), 그리고 이전 행동(Action)을 동시에 처리해야 하며, 기하학(Geometry), 동역학(Dynamics), 안전(Safety), 실시간 제약(Real-Time Constraint), 그리고 로봇 하드웨어의 특성까지 고려하여 실제 행동을 생성해야 한다.

언어 기반 파운데이션 모델의 역사는 로보틱스에도 중요한 교훈을 제공한다. 첫째, 다양한 사전학습 데이터는 여러 작업으로 이전 가능한 공통 표현을 만들어 준다. 둘째, 규모 확장은 작은 모델에서는 보이지 않던 새로운 능력을 만들어낸다. 셋째, 자연어는 목표를 표현하는 범용 인터페이스가 될 수 있다. 넷째, 멀티모달 학습은 관측과 설명을 하나의 개념으로 연결한다. 다섯째, 프롬프팅(Prompting), 미세조정(Fine-Tuning), 저랭크 적응(Low-Rank Adaptation, LoRA)과 같은 적응 기술은 처음부터 다시 학습하지 않고도 범용 모델을 특정 작업에 빠르게 적용할 수 있도록 만든다.

그러나 로봇 파운데이션 모델은 언어 모델과는 다른 어려움도 가지고 있다. 잘못된 문장은 사용자에게 혼란만 줄 수 있지만, 잘못된 로봇 행동은 장비를 파손하거나 사람을 다치게 할 수 있다. 또한 실제 로봇 데이터는 인터넷 텍스트보다 훨씬 수집 비용이 높으며, 로봇마다 운동학(Kinematics), 센서 구성(Sensor Layout), 좌표계(Coordinate System), 행동 공간(Action Space), 제어 주기(Control Frequency)가 모두 다르다. 실제 환경에는 노이즈(Noise), 마찰(Friction), 지연(Delay), 가림(Occlusion), 기계적 마모(Mechanical Wear), 그리고 예측하기 어려운 인간 행동까지 존재한다. 따라서 BERT에서 GPT, 그리고 Gemini까지의 발전은 피지컬 AI의 출발점일 뿐, 최종 해결책은 아니다.

미래의 파운데이션 모델은 멀티모달 인식(Multimodal Perception), 장기 메모리(Long-Term Memory), 월드 모델(World Model), 인과 추론(Causal Reasoning), 계획(Planning), 행동 생성(Action Generation), 지속 학습(Continuous Learning)을 하나의 시스템 안에서 결합하게 될 것이다. 언어 모델은 추상적 추론을 담당하고, 비전 모델은 장면 이해를 수행하며, 월드 모델은 미래를 예측하고, 행동 정책(Action Policy)은 실제 환경과 상호작용하는 역할을 수행하는 형태가 될 가능성이 높다. 하나의 거대한 신경망이 모든 일을 수행하기보다, 여러 전문 모델이 공통 표현(Common Representation)을 중심으로 협력하는 구조가 현실적인 방향이 될 것이다.

효율성(Efficiency)은 다음 세대 파운데이션 모델을 결정하는 중요한 요소가 될 것이다. 초기에는 단순한 규모 확장이 성능 향상의 핵심이었지만, 실제 산업에서는 낮은 지연(Low Latency), 낮은 에너지 소비(Energy Consumption), 예측 가능한 동작(Predictable Behavior), 그리고 엣지 하드웨어(Edge Hardware)에서의 실행이 더욱 중요하다. 양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation), 전문가 혼합(Mixture of Experts), 추측 디코딩(Speculative Decoding), 희소 어텐션(Sparse Attention), 캐싱(Caching), 하드웨어 최적화(Hardware-Aware Optimization)는 앞으로 파운데이션 모델이 데이터센터를 넘어 로봇, 차량, 드론, 공장, 임베디드 시스템으로 확산되는 핵심 기술이 될 것이다.

BERT에서 GPT, 그리고 Gemini까지의 발전은 단순히 모델이 더 커졌다는 의미가 아니다. 모델이 컴퓨팅에서 수행하는 역할 자체가 변화하였다. BERT는 재사용 가능한 문맥 표현(Contextual Representation)을 확립하였고, GPT는 생성형 사전학습(Generative Pre-training)을 범용 인터페이스로 발전시켰다. 대규모 모델은 인컨텍스트 학습(In-Context Learning)과 창발 능력(Emergent Capability)을 보여주었으며, 지시 미세조정(Instruction Tuning)과 인간 피드백(Human Feedback)은 모델을 더욱 실용적으로 만들었다. 비전-언어 연구(Vision-Language Research)는 텍스트를 넘어 다양한 정보를 하나로 통합하였고, Gemini와 같은 멀티모달 시스템은 여러 형태의 정보를 하나의 파운데이션 모델 안에서 처리하는 방향을 제시하였다.

이러한 발전은 피지컬 AI 파운데이션 모델(Physical AI Foundation Model)의 역사적 배경을 제공한다. 미래의 로봇 지능은 단순히 언어 모델을 로봇 안에 넣는 것이 아니라, 인식(Perception), 체화(Embodiment), 월드 모델(World Model), 행동(Action), 안전성(Safety)을 모두 통합하는 방향으로 발전할 것이다. 따라서 BERT에서 GPT, 그리고 Gemini까지의 역사는 단순한 언어 모델의 발전사가 아니라, 정보를 이해하고 생성하는 인공지능에서 실제 물리 세계를 이해하고 행동하는 인공지능으로 나아가는 거대한 전환의 첫 번째 단계라고 할 수 있다.

##  

## 1.3 Foundation Model Training Paradigms SSL RLHF DPO

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

Foundation models are not created through a single training method. They emerge from a sequence of learning stages that transform raw data into broad representations, general capabilities, instruction-following behavior, and task-specific reliability. Among the most important paradigms in this process are self-supervised learning, reinforcement learning from human feedback, and direct preference optimization. These approaches address different layers of the training problem. Self-supervised learning builds the model's general knowledge and representational capacity. Reinforcement learning from human feedback aligns model behavior with human preferences. Direct preference optimization offers a simpler and often more stable way to learn from preference data without explicitly training a separate reward model and reinforcement learning loop.

The distinction between these paradigms is essential. A model trained only through self-supervised learning may possess broad linguistic, visual, or physical knowledge but may not respond helpfully to instructions. A model aligned only with preference data would lack sufficient foundational knowledge if it had not first been exposed to a large and diverse corpus. Effective foundation model development therefore depends on combining these methods in a staged pipeline rather than treating them as competing alternatives.

Self-supervised learning, commonly abbreviated as SSL, is the primary mechanism used to pre-train modern foundation models. Unlike supervised learning, it does not require humans to manually label every training example. Instead, the training signal is generated from the data itself. The model is asked to predict a hidden, missing, transformed, or future part of an input using the remaining information as context. Because almost any sufficiently structured dataset can generate its own learning targets, self-supervised learning makes it possible to train on internet-scale text, images, videos, audio recordings, code repositories, sensor streams, and robot trajectories.

In autoregressive language modeling, the model predicts the next token from previous tokens. A sentence is divided into a sequence, and each position becomes a training target. Although next-token prediction appears simple, it forces the model to learn grammar, meaning, factual associations, writing style, discourse structure, and many forms of reasoning. To predict the continuation of technical documentation, the model must learn terminology and procedural relationships. To complete code, it must capture syntax, data flow, and programming patterns. To continue a conversation, it must infer intent, tone, and context.

Masked modeling provides another form of self-supervision. In masked language modeling, selected tokens are hidden and the model reconstructs them using information from both directions. In vision, masked image modeling removes image patches and trains the model to recover their content or latent representations. Similar methods can be applied to audio segments, video frames, point-cloud regions, tactile measurements, and robot states. The central idea is that understanding the whole structure of the data should make it possible to infer the missing parts.

Contrastive learning is another important self-supervised paradigm. Instead of reconstructing the exact input, a model learns which observations should have similar representations and which should remain separated. Two augmented views of the same image may be treated as a positive pair. An image and its correct caption may be brought closer in a shared embedding space, while unrelated image-text pairs are pushed apart. This approach has been highly influential in multimodal foundation models because it aligns information across different modalities without requiring detailed class labels.

For Physical AI, self-supervised learning can exploit temporal and cross-modal regularities in robot data. A robot can predict the next camera frame, estimate future joint states, reconstruct missing sensor channels, align language instructions with action trajectories, or determine whether two observations belong to the same task. Since robot data is expensive to label manually, self-supervision is especially valuable. Every recorded trajectory contains potential training signals in its temporal order, sensor correlations, action consequences, and environmental changes.

Large-scale pre-training normally combines several self-supervised objectives. A multimodal model may simultaneously predict text tokens, reconstruct image features, align language with video, estimate future states, and distinguish matching from mismatching sensor streams. Multi-objective training encourages the model to learn representations that are useful across many downstream tasks. However, balancing these objectives is difficult. One objective may dominate the gradients, one modality may contain more data than another, and some targets may be much easier to optimize than others.

Data quality is as important as objective design. Self-supervised learning can absorb useful knowledge from massive datasets, but it can also absorb errors, biases, duplicated content, unsafe behavior, and spurious correlations. The model does not automatically know which patterns are desirable. It learns statistical regularities, including undesirable ones. Consequently, large-scale pre-training requires filtering, deduplication, source balancing, privacy protection, quality scoring, contamination analysis, and careful mixture design.

Curriculum design also affects self-supervised training. Some systems begin with short and relatively simple sequences before introducing longer and more complex data. Others change the ratio of text, code, mathematics, images, videos, or robot trajectories over time. For Physical AI, a curriculum may progress from static visual recognition to short action sequences, then to long-horizon tasks and multi-step interaction. A well-designed curriculum can improve optimization stability and help the model acquire basic representations before learning more difficult behaviors.

After self-supervised pre-training, the model may have substantial knowledge but still behave like a continuation engine rather than an assistant or autonomous agent. It predicts statistically likely outputs, not necessarily useful, truthful, safe, or instruction-compliant ones. Supervised fine-tuning is often used as the first alignment stage. The model is trained on examples containing an instruction and a desired response. This teaches formatting, conversational behavior, task completion patterns, and domain-specific conventions.

Supervised fine-tuning remains limited by the quality and coverage of the demonstration data. Human authors can provide excellent examples, but it is difficult to write demonstrations for every possible instruction. In addition, there may be many acceptable responses to the same request. A single reference answer does not fully express human preferences regarding clarity, helpfulness, harmlessness, accuracy, tone, or level of detail. Preference learning was developed to address this limitation.

Reinforcement learning from human feedback, usually abbreviated as RLHF, converts human judgments into an optimization signal. The process typically begins by presenting human evaluators with several outputs generated for the same prompt. The evaluators rank or compare those outputs. Their choices form a preference dataset that describes which behavior is more desirable.

A reward model is then trained to predict these preferences. Given a prompt and a candidate response, it assigns a scalar reward representing estimated human approval. The reward model does not directly generate text or actions. It acts as a learned evaluator. Once trained, it can score large numbers of outputs more cheaply than relying on human reviewers for every training step.

The policy model is subsequently optimized to produce outputs with higher predicted reward. In language model alignment, this stage has often used policy-gradient algorithms such as proximal policy optimization. The model generates responses, receives scores from the reward model, and updates its parameters to increase expected reward. A regularization term usually constrains the updated model from drifting too far from the supervised fine-tuned reference model.

This constraint is important because unrestricted reward maximization can produce unstable or undesirable behavior. The model may discover patterns that exploit weaknesses in the reward model rather than genuinely satisfying human intent. This phenomenon is known as reward hacking or specification gaming. A response may become excessively verbose because the reward model associates length with helpfulness, or it may confidently imitate preferred phrasing without improving factual accuracy. In robotics, reward hacking can be more dangerous because the policy may technically maximize the objective while violating the intended physical task.

RLHF has several strengths. It can optimize qualities that are difficult to express through fixed labels, such as politeness, usefulness, caution, coherence, and appropriate refusal behavior. It allows multiple acceptable responses to be compared rather than requiring one perfect target. It can also incorporate expert judgments for specialized domains such as medicine, law, engineering, and robotics safety.

However, RLHF is operationally complex. It requires collecting preference data, training and validating a reward model, running online or semi-online generation, maintaining a reinforcement learning infrastructure, tuning stability-sensitive hyperparameters, and monitoring divergence from the reference policy. Policy-gradient methods can be noisy, computationally expensive, and difficult to reproduce. The reward model may also inherit inconsistencies or biases from evaluators.

Direct preference optimization, commonly abbreviated as DPO, was developed as a simpler alternative for preference alignment. It uses the same basic form of preference data: for a given prompt, one response is preferred and another is rejected. However, instead of first training a reward model and then optimizing the policy through reinforcement learning, DPO directly updates the language model so that the preferred response becomes more probable relative to the rejected response.

The core insight is that preference-based policy optimization can be expressed as a supervised classification-style objective under an appropriate mathematical formulation. The model is encouraged to increase the relative likelihood of preferred outputs while remaining anchored to a reference model. This removes the need for an explicit reward model and avoids a separate policy-gradient loop.

DPO is attractive because of its simplicity. Training resembles ordinary fine-tuning and can use familiar distributed learning infrastructure. It is generally easier to implement, debug, and reproduce than a full RLHF pipeline. It also avoids some instability associated with online reinforcement learning. For many alignment tasks, DPO can achieve strong instruction-following and preference-matching performance with lower engineering overhead.

The reference model still plays an important role in DPO. Without a constraint, the policy might overfit to the preference dataset or distort its general capabilities. The DPO objective compares how strongly the updated model and the reference model prefer the chosen output over the rejected output. A temperature-like parameter controls how aggressively the policy moves toward the preference data.

DPO also has limitations. It relies heavily on the quality and diversity of the preference pairs. If the chosen and rejected responses differ only superficially, the model may learn stylistic preferences rather than substantive quality. If the dataset contains inconsistent judgments, hidden biases, or weak negative examples, alignment performance may degrade. DPO is also fundamentally an offline method in its standard form. It learns from a fixed collection of comparisons and may not explore new behaviors as actively as online reinforcement learning.

The relationship between RLHF and DPO should therefore not be understood as a simple replacement. RLHF provides a flexible framework for optimizing arbitrary learned rewards and supports online exploration. DPO offers a streamlined method when high-quality preference pairs are already available. Some systems may use DPO for initial alignment and later apply reinforcement learning for advanced reasoning, tool use, or domain-specific optimization. Others may use rejection sampling, iterative preference collection, or hybrid methods that combine elements of supervised learning and reinforcement learning.

Preference optimization can also use feedback generated by AI systems. Reinforcement learning from AI feedback replaces or supplements human evaluators with another model operating under a set of rules, principles, or evaluation criteria. This can reduce cost and increase scale, although it introduces the risk that the evaluator model's errors and biases will be amplified. Human oversight remains important, particularly in safety-critical fields.

For robot foundation models, preference learning must extend beyond text quality. A preferred trajectory may be safer, smoother, faster, more energy efficient, or more compliant with human expectations. Two robot behaviors can be compared based on collision risk, task completion, force limits, path efficiency, social acceptability, or recovery quality. Human operators may judge recorded trajectories, simulation videos, or real-world executions.

Applying RLHF to robotics is challenging because actions have delayed and physical consequences. A language response can be evaluated immediately, while a robotic decision may need to be judged over a long horizon. The robot may complete the task successfully but use an unsafe intermediate motion. Reward models must therefore evaluate entire trajectories, not only final outcomes. They may also need access to multimodal observations including video, proprioception, force, and environmental state.

Online reinforcement learning with physical robots is costly and potentially dangerous. Exploration can damage hardware or create unsafe interactions. Simulation is therefore frequently used to generate candidate behaviors and perform preference optimization before limited real-world validation. Safety constraints, action filters, control barriers, and human intervention systems are necessary even when the learned policy appears well aligned.

DPO-like approaches can simplify robot preference learning when paired demonstrations or trajectories are available. A preferred manipulation sequence can be compared with an unsuccessful or unsafe one. The policy can then learn to assign greater probability to the preferred sequence. However, continuous robot actions are more complex than discrete language tokens. Actions may need to be discretized, represented as chunks, generated through diffusion, or modeled using flow-based policies before preference optimization can be applied effectively.

Physical AI training therefore often requires a layered architecture. Self-supervised learning builds representations from large amounts of passive and interactive data. Behavioral cloning or supervised fine-tuning teaches the model to imitate successful demonstrations. Preference optimization improves quality and alignment. Reinforcement learning refines performance through interaction. Safety modules enforce constraints that should not depend solely on learned preferences.

Evaluation must be performed after every stage. During self-supervised pre-training, researchers monitor loss curves, representation quality, data memorization, transfer performance, and emerging capabilities. After supervised fine-tuning, they evaluate instruction following and task success. After RLHF or DPO, they test preference win rates, robustness, factuality, safety, calibration, and regression on previous capabilities. In robotics, evaluation must additionally cover collision frequency, control smoothness, action latency, hardware limits, out-of-distribution behavior, and recovery from failure.

An important risk is alignment tax, in which improving preferred behavior reduces raw capability. A strongly aligned model may become overly cautious, refuse benign tasks, lose diversity, or perform worse on specialized benchmarks. Similarly, a robot policy optimized heavily for safety may become too conservative to complete useful tasks. Training therefore requires balancing capability, helpfulness, efficiency, and safety rather than maximizing a single metric.

Another challenge is preference distribution shift. The people who label the data may not represent all future users, cultures, workplaces, or operating conditions. Preferences suitable for a conversational assistant may not be appropriate for an industrial robot. A warehouse operator, safety engineer, customer, and regulator may value different aspects of behavior. Preference datasets must therefore be documented, segmented, and reviewed in relation to the deployment context.

The future of foundation model training will likely involve more iterative and integrated pipelines. Models will pre-train on large multimodal datasets, generate synthetic examples, receive critiques, revise their outputs, and collect new preference data from deployment. World models may simulate the consequences of candidate actions. Verification systems may check outputs before execution. Human experts may intervene only for uncertain or high-risk cases.

Continual learning will also become important. A deployed model should adapt to new tools, environments, regulations, robot embodiments, and user expectations without forgetting previous capabilities. Preference optimization may become an ongoing process rather than a one-time alignment stage. This requires strong version control, replay datasets, regression testing, rollback mechanisms, and monitoring for behavioral drift.

Self-supervised learning, RLHF, and DPO should ultimately be viewed as complementary components of a broader training strategy. Self-supervised learning provides scale by converting raw data into learning signals. RLHF provides a mechanism for optimizing complex human judgments through learned rewards and interaction. DPO simplifies preference alignment by directly learning from chosen and rejected outputs. Each method solves a different part of the foundation model problem.

For Physical AI, the combination is especially powerful. Self-supervised learning can extract structure from enormous quantities of sensor and trajectory data. Supervised learning can teach basic task execution from demonstrations. Preference optimization can encode safety, efficiency, and human expectations. Reinforcement learning can improve performance through simulated or controlled interaction. Together, these paradigms form the training foundation for models that do not merely predict information but perceive environments, interpret goals, plan actions, and operate safely in the physical world.

파운데이션 모델(Foundation Model)은 하나의 학습 방법만으로 만들어지지 않는다. 원시 데이터(Raw Data)를 폭넓은 표현(Representation)으로 변환하고, 일반적인 능력(General Capability)을 학습하며, 사용자의 지시를 이해하고 수행하는 능력을 갖추고, 실제 응용 분야에서 신뢰성 있게 동작하도록 만들기 위해서는 여러 단계의 학습 과정이 필요하다. 이러한 과정에서 가장 중요한 학습 패러다임(Training Paradigm)은 자기지도학습(Self-Supervised Learning, SSL), 인간 피드백 기반 강화학습(Reinforcement Learning from Human Feedback, RLHF), 그리고 직접 선호 최적화(Direct Preference Optimization, DPO)이다. 이들 방법은 각각 서로 다른 역할을 수행한다. 자기지도학습은 모델의 일반적인 지식과 표현 능력을 구축하고, RLHF는 모델의 행동을 인간의 선호와 일치하도록 정렬하며, DPO는 별도의 보상 모델(Reward Model)과 복잡한 강화학습 과정을 거치지 않고도 선호 데이터를 직접 학습할 수 있도록 하는 보다 단순하고 효율적인 방법이다.

이러한 세 가지 학습 방식은 서로 경쟁하는 기술이 아니라 서로 다른 계층(Level)을 담당하는 기술이라는 점이 중요하다. 자기지도학습만 수행한 모델은 폭넓은 언어(Language), 이미지(Image), 또는 물리적 세계(Physical World)에 대한 지식을 가질 수 있지만 사용자의 지시에 적절하게 반응하지 못할 수 있다. 반대로 선호 데이터만으로 정렬(Alignment)을 수행한 모델은 충분한 일반 지식이 없다면 높은 수준의 성능을 발휘할 수 없다. 따라서 현대의 파운데이션 모델은 이들 기술을 하나의 단계적 파이프라인(Staged Pipeline)으로 결합하여 학습하는 것이 일반적이다.

자기지도학습(Self-Supervised Learning, SSL)은 현대 파운데이션 모델의 사전학습(Pre-training)을 담당하는 핵심 기술이다. 지도학습(Supervised Learning)과 달리 사람이 모든 데이터를 직접 라벨링(Labeling)할 필요가 없다. 대신 데이터 자체로부터 학습 목표(Learning Target)를 자동으로 생성한다. 모델은 입력 데이터의 일부를 가리거나(Masked), 제거하거나(Missing), 변형하거나(Transformed), 미래(Future)의 일부를 예측하도록 학습된다. 데이터 자체가 학습 문제를 만들어내기 때문에 인터넷 규모의 텍스트(Text), 이미지(Image), 비디오(Video), 오디오(Audio), 소스 코드(Source Code), 센서 데이터(Sensor Stream), 로봇 궤적(Robot Trajectory) 등을 모두 학습에 활용할 수 있다.

자기회귀 언어 모델링(Auto-Regressive Language Modeling)에서는 모델이 이전 토큰(Token)을 이용하여 다음 토큰을 예측한다. 문장은 여러 개의 토큰으로 나뉘며, 각 위치의 다음 토큰이 학습 목표가 된다. 이러한 다음 토큰 예측(Next Token Prediction)은 단순해 보이지만 실제로는 문법(Grammar), 의미(Semantics), 사실 관계(Factual Association), 문체(Writing Style), 담화 구조(Discourse Structure), 그리고 다양한 형태의 추론(Reasoning)을 학습하도록 만든다. 기술 문서를 이어 쓰기 위해서는 전문 용어와 절차를 이해해야 하고, 코드를 완성하기 위해서는 문법과 데이터 흐름(Data Flow), 프로그래밍 패턴을 학습해야 하며, 대화를 이어가기 위해서는 사용자의 의도(Intent), 말투(Tone), 그리고 문맥(Context)을 이해해야 한다.

마스크드 모델링(Masked Modeling)은 또 다른 형태의 자기지도학습이다. 마스크드 언어 모델링(Masked Language Modeling)에서는 일부 토큰을 가린 후 앞뒤 문맥을 모두 이용하여 이를 복원하도록 학습한다. 컴퓨터 비전에서는 마스크드 이미지 모델링(Masked Image Modeling)을 이용하여 일부 이미지 패치(Patch)를 제거하고 이를 복원하도록 학습한다. 동일한 원리는 오디오(Audio), 비디오(Video), 포인트 클라우드(Point Cloud), 촉각 데이터(Tactile Data), 그리고 로봇 상태(Robot State)에도 적용할 수 있다. 핵심 아이디어는 데이터 전체 구조를 이해하면 일부 정보가 없어도 이를 추론할 수 있다는 것이다.

대조학습(Contrastive Learning)도 매우 중요한 자기지도학습 기법이다. 이 방법은 입력을 그대로 복원하는 것이 아니라 어떤 데이터는 서로 비슷한 표현을 가져야 하고 어떤 데이터는 서로 구별되어야 하는지를 학습한다. 동일한 이미지의 두 가지 증강(Augmentation)은 긍정 쌍(Positive Pair)이 되고, 올바른 이미지와 설명(Image-Caption Pair)은 동일한 임베딩 공간(Embedding Space)에서 가까워지도록 학습된다. 반대로 서로 관련 없는 이미지와 문장은 멀어지도록 학습된다. 이러한 방식은 멀티모달(Multimodal) 파운데이션 모델에서 서로 다른 데이터 형태를 하나의 표현 공간으로 연결하는 데 매우 중요한 역할을 한다.

피지컬 AI(Physical AI)에서는 자기지도학습이 로봇 데이터의 시간적 관계(Temporal Relationship)와 여러 센서 간 상관관계(Cross-Modal Correlation)를 활용할 수 있다. 로봇은 다음 카메라 프레임(Camera Frame)을 예측하거나, 미래의 관절 상태(Joint State)를 추정하거나, 누락된 센서 채널을 복원하거나, 언어 명령(Language Instruction)과 행동 궤적(Action Trajectory)을 연결하거나, 두 개의 관측값이 동일한 작업(Task)에 속하는지를 판단하도록 학습할 수 있다. 실제 로봇 데이터를 사람이 직접 라벨링하는 것은 매우 비용이 크므로 자기지도학습은 특히 중요한 의미를 가진다. 하나의 주행 기록이나 조작 기록만으로도 시간 순서(Temporal Order), 센서 간 상관관계, 행동의 결과(Action Consequence), 환경 변화(Environmental Change) 등 수많은 학습 신호를 생성할 수 있기 때문이다.

현대의 대규모 사전학습(Large-Scale Pre-training)은 일반적으로 여러 자기지도학습 목표(Objective)를 동시에 사용한다. 하나의 멀티모달 모델은 텍스트 토큰을 예측하면서 동시에 이미지 특징을 복원하고, 언어와 비디오를 정렬하며, 미래 상태를 예측하고, 올바른 센서 조합과 잘못된 센서 조합을 구분하는 작업을 함께 수행할 수 있다. 이러한 다중 목적 학습(Multi-Objective Training)은 다양한 작업에서 활용 가능한 범용 표현을 학습하도록 만든다. 그러나 여러 목적의 균형을 맞추는 것은 쉽지 않다. 특정 목적이 그래디언트(Gradient)를 지배하거나, 특정 데이터 종류가 지나치게 많거나, 어떤 목표는 다른 목표보다 훨씬 쉽게 최적화될 수 있기 때문이다.

데이터 품질(Data Quality)은 학습 목표만큼이나 중요하다. 자기지도학습은 방대한 데이터에서 유용한 지식을 학습하는 동시에 오류(Error), 편향(Bias), 중복(Duplication), 위험한 행동(Unsafe Behavior), 잘못된 상관관계(Spurious Correlation)도 함께 학습할 수 있다. 모델은 어떤 정보가 바람직한지 스스로 판단하지 못하며 단순히 통계적 규칙(Statistical Regularity)을 학습할 뿐이다. 따라서 대규모 사전학습에서는 데이터 필터링(Data Filtering), 중복 제거(Deduplication), 데이터 출처 균형(Source Balancing), 개인정보 보호(Privacy Protection), 품질 평가(Quality Scoring), 데이터 오염 분석(Contamination Analysis), 그리고 데이터 혼합 전략(Data Mixture Design)이 매우 중요하다.

학습 커리큘럼(Curriculum Design) 또한 자기지도학습에 큰 영향을 미친다. 일부 시스템은 짧고 단순한 데이터부터 학습한 후 점차 긴 문장과 복잡한 데이터를 학습한다. 또 다른 시스템은 텍스트(Text), 코드(Code), 수학(Mathematics), 이미지(Image), 비디오(Video), 로봇 데이터(Robot Trajectory)의 비율을 학습 과정에서 점진적으로 변경한다. 피지컬 AI에서는 정적인 시각 인식(Static Visual Recognition)에서 시작하여 짧은 행동(Action Sequence), 그리고 장기 작업(Long-Horizon Task)과 다단계 상호작용(Multi-Step Interaction)으로 점진적으로 발전시키는 방식이 사용될 수 있다. 잘 설계된 커리큘럼은 최적화를 안정화시키고 모델이 기본적인 표현을 먼저 학습한 후 더 어려운 행동을 학습하도록 도와준다.

자기지도 사전학습(Self-Supervised Pre-training)이 끝난 모델은 방대한 지식을 가지고 있지만 여전히 사용자의 지시를 수행하는 비서(Assistant)라기보다는 단순한 문장 이어쓰기 엔진(Continuation Engine)에 가깝다. 모델은 가장 가능성이 높은 출력을 예측할 뿐 반드시 유용하거나, 사실에 맞거나, 안전하거나, 사용자의 지시를 따르는 것은 아니다. 이를 개선하기 위한 첫 번째 정렬 단계(Alignment Stage)가 지도 미세조정(Supervised Fine-Tuning, SFT)이다. 모델은 사용자의 명령(Instruction)과 원하는 응답(Desired Response)으로 구성된 예제를 학습하여 대화 형식, 응답 방식, 작업 수행 패턴 등을 익히게 된다.

그러나 지도 미세조정은 사람이 작성한 예제의 품질과 범위에 크게 의존한다. 모든 가능한 질문에 대해 이상적인 답변을 작성하는 것은 현실적으로 어렵다. 또한 하나의 질문에 대해 좋은 답변은 하나만 존재하는 것이 아니라 여러 개가 존재할 수 있다. 하나의 정답만으로는 명확성(Clarity), 유용성(Helpfulness), 안전성(Harmlessness), 정확성(Accuracy), 말투(Tone), 상세 수준(Level of Detail)과 같은 인간의 선호를 충분히 표현할 수 없다. 이러한 문제를 해결하기 위해 선호 학습(Preference Learning)이 등장하였다.

인간 피드백 기반 강화학습(Reinforcement Learning from Human Feedback, RLHF)은 사람의 평가를 최적화 신호(Optimization Signal)로 변환하는 기술이다. 일반적으로 하나의 질문(Prompt)에 대해 모델이 여러 개의 응답을 생성하면 사람이 이를 비교하여 어떤 응답이 더 좋은지 선택한다. 이러한 선택 결과는 선호 데이터셋(Preference Dataset)이 된다.

다음 단계에서는 보상 모델(Reward Model)을 학습한다. 보상 모델은 질문과 응답을 입력받아 사람이 얼마나 선호할지를 하나의 점수(Reward Score)로 예측한다. 보상 모델은 직접 답변을 생성하는 모델이 아니라, 응답을 평가하는 평가자(Evaluator)의 역할을 수행한다. 학습이 완료되면 사람 대신 수많은 응답을 자동으로 평가할 수 있다.

이후 정책 모델(Policy Model)은 높은 보상을 받는 응답을 생성하도록 최적화된다. 언어 모델에서는 일반적으로 근접 정책 최적화(Proximal Policy Optimization, PPO)와 같은 정책 그래디언트(Policy Gradient) 알고리즘이 사용되었다. 모델은 응답을 생성하고, 보상 모델로부터 점수를 받은 뒤, 더 높은 보상을 받을 수 있도록 자신의 파라미터를 업데이트한다. 동시에 원래의 지도 미세조정 모델과 너무 멀어지지 않도록 정규화(Regularization) 항도 함께 사용된다.

이러한 제약은 매우 중요하다. 제한 없이 보상만 최대화하면 모델은 실제로 사람의 의도를 만족시키기보다 보상 모델의 허점을 이용하는 방향으로 학습될 수 있다. 이를 보상 해킹(Reward Hacking) 또는 명세 게임(Specification Gaming)이라고 한다. 예를 들어 보상 모델이 긴 답변을 선호한다고 학습하면 모델은 불필요하게 장황한 답변을 생성할 수 있다. 로보틱스에서는 이러한 문제가 더욱 위험하다. 로봇이 목표 함수(Objective)는 최대화하지만 실제 의도와 다른 위험한 행동을 수행할 수 있기 때문이다.

RLHF는 여러 장점을 가진다. 정답(Label)으로 표현하기 어려운 공손함(Politeness), 유용성(Helpfulness), 신중함(Caution), 일관성(Coherence), 적절한 거부(Appropriate Refusal)와 같은 특성을 학습할 수 있다. 또한 하나의 정답 대신 여러 좋은 답변을 서로 비교하여 학습할 수 있으며, 의료(Medicine), 법률(Law), 공학(Engineering), 로보틱스 안전(Robotics Safety)과 같은 전문 분야의 전문가 평가도 반영할 수 있다.

반면 RLHF는 구현이 매우 복잡하다. 선호 데이터 수집, 보상 모델 학습과 검증, 온라인 응답 생성, 강화학습 인프라 구축, 다양한 하이퍼파라미터(Hyperparameter) 조정, 그리고 정책 모델의 발산(Divergence)을 지속적으로 모니터링해야 한다. 정책 그래디언트 알고리즘은 계산 비용이 높고 재현성(Reproducibility)도 쉽지 않다. 또한 보상 모델 역시 사람의 편향이나 일관성 부족을 그대로 학습할 수 있다.

직접 선호 최적화(Direct Preference Optimization, DPO)는 이러한 RLHF를 단순화하기 위해 제안된 방법이다. DPO 역시 동일한 선호 데이터, 즉 하나의 질문에 대해 선호되는 응답(Chosen Response)과 거부되는 응답(Rejected Response)을 사용한다. 그러나 RLHF처럼 보상 모델을 먼저 학습하고 이후 강화학습을 수행하지 않는다. 대신 선호되는 응답의 확률을 직접 높이고, 거부되는 응답의 확률을 낮추도록 언어 모델을 직접 학습시킨다.

DPO의 핵심 아이디어는 선호 기반 정책 최적화(Preference-Based Policy Optimization)를 적절한 수학적 형태로 변환하면 일반적인 지도학습(Classification-Style Objective)처럼 최적화할 수 있다는 것이다. 모델은 선호된 응답의 상대적 확률(Relative Likelihood)을 높이도록 직접 업데이트되며, 동시에 기준 모델(Reference Model)과의 차이가 지나치게 커지지 않도록 유지된다. 따라서 별도의 보상 모델과 강화학습 과정이 필요하지 않다.

DPO는 구현이 단순하다는 장점이 있다. 일반적인 미세조정(Fine-Tuning)과 매우 유사한 형태로 학습할 수 있으며, 기존의 분산 학습 인프라를 그대로 사용할 수 있다. RLHF보다 구현, 디버깅(Debugging), 재현(Reproduction)이 훨씬 쉽고, 온라인 강화학습에서 발생하는 불안정성도 줄일 수 있다. 많은 정렬 문제에서 DPO는 훨씬 적은 개발 비용으로도 우수한 성능을 보여준다.

기준 모델(Reference Model)은 DPO에서도 매우 중요한 역할을 한다. 만약 기준 모델이 없다면 정책 모델은 선호 데이터에 과도하게 적합(Overfitting)되거나 기존의 일반 능력을 잃을 수 있다. DPO는 현재 모델과 기준 모델이 각각 선호된 응답을 얼마나 더 높게 평가하는지를 비교하면서 학습을 진행한다. 또한 온도(Temperature)와 유사한 하이퍼파라미터를 이용하여 얼마나 적극적으로 선호 데이터에 맞출 것인지를 조절한다.

그러나 DPO 역시 한계를 가진다. 성능은 선호 데이터의 품질과 다양성에 크게 의존한다. 선호된 응답과 거부된 응답의 차이가 단순한 문체 차이에 불과하다면 모델은 실제 품질보다 표현 방식만 학습할 수 있다. 또한 데이터에 편향이나 일관성 부족이 존재하면 정렬 성능도 함께 저하된다. DPO는 기본적으로 오프라인(Offline) 방식이므로 고정된 선호 데이터만을 이용하여 학습하며, RLHF처럼 새로운 행동을 적극적으로 탐색(Exploration)하지는 않는다.

따라서 RLHF와 DPO는 단순히 하나가 다른 하나를 대체하는 관계가 아니다. RLHF는 보상 모델을 이용한 온라인 최적화와 탐색 능력을 제공하며, DPO는 이미 양질의 선호 데이터가 존재할 때 훨씬 단순한 학습 방법을 제공한다. 실제 시스템에서는 먼저 DPO를 이용하여 기본 정렬을 수행하고, 이후 RLHF를 통해 고급 추론이나 도구 사용(Tool Use), 특정 도메인 최적화를 수행하기도 한다. 또한 거부 샘플링(Rejection Sampling), 반복적 선호 수집(Iterative Preference Collection), 지도학습과 강화학습을 결합한 다양한 하이브리드(Hybrid) 방법도 사용된다.

선호 최적화는 인공지능(AI)이 생성한 피드백을 이용할 수도 있다. AI 피드백 기반 강화학습(Reinforcement Learning from AI Feedback)은 사람 대신 또 다른 모델이 규칙과 평가 기준에 따라 응답을 평가한다. 이는 비용을 줄이고 대규모 학습을 가능하게 하지만, 평가 모델의 오류나 편향이 함께 증폭될 위험도 존재한다. 따라서 안전이 중요한 분야에서는 여전히 인간 전문가의 감독(Human Oversight)이 필요하다.

로봇 파운데이션 모델(Robot Foundation Model)에서는 선호 학습의 대상이 단순한 텍스트 품질을 넘어선다. 더 안전한 경로, 더 부드러운 움직임, 더 빠른 작업, 더 높은 에너지 효율, 그리고 인간이 선호하는 행동이 모두 선호 대상이 될 수 있다. 두 개의 로봇 행동은 충돌 위험(Collision Risk), 작업 성공(Task Completion), 힘 제한(Force Limit), 경로 효율(Path Efficiency), 사회적 적절성(Social Acceptability), 복구 능력(Recovery Quality) 등을 기준으로 비교될 수 있다. 사람은 실제 로봇 영상이나 시뮬레이션 결과를 보고 어느 행동이 더 바람직한지 평가할 수 있다.

그러나 로봇에서 RLHF를 적용하는 것은 쉽지 않다. 언어 모델은 응답 직후 평가할 수 있지만, 로봇 행동은 긴 시간(Long Horizon)이 지나야 평가할 수 있다. 작업은 성공했더라도 중간 과정에서 위험한 움직임을 수행했을 수도 있다. 따라서 보상 모델은 최종 결과뿐 아니라 전체 행동 궤적(Trajectory)을 평가해야 하며, 비디오(Video), 고유감각(Proprioception), 힘 센서(Force Sensor), 환경 상태(Environment State) 등 다양한 센서 정보도 함께 고려해야 한다.

실제 로봇을 이용한 온라인 강화학습은 비용이 높고 위험하다. 탐색 과정 자체가 장비를 파손하거나 사람에게 위험을 줄 수 있기 때문이다. 따라서 대부분의 경우 먼저 시뮬레이션(Simulation)에서 후보 행동을 생성하고 선호 최적화를 수행한 뒤, 제한된 실제 환경에서 검증을 진행한다. 또한 안전 제약(Safety Constraint), 행동 필터(Action Filter), 제어 장벽(Control Barrier), 인간 개입(Human Intervention) 시스템도 반드시 함께 사용된다.

DPO와 유사한 접근법은 로봇에서도 활용될 수 있다. 성공한 조작 행동과 실패하거나 위험한 행동을 하나의 쌍(Pair)으로 구성하여 정책 모델이 성공한 행동의 확률을 더 높게 학습하도록 만들 수 있다. 그러나 로봇의 연속 행동(Continuous Action)은 언어의 이산 토큰(Discrete Token)보다 훨씬 복잡하다. 따라서 행동(Action)은 이산화(Discretization), 청크(Action Chunk), 확산 정책(Diffusion Policy), 플로우 기반 정책(Flow-Based Policy) 등으로 먼저 표현된 후 선호 최적화를 수행하는 경우가 많다.

결국 피지컬 AI의 학습은 계층적인 구조(Layered Architecture)를 필요로 한다. 자기지도학습은 방대한 센서 데이터와 행동 데이터를 이용하여 공통 표현을 학습하고, 행동 복제(Behavioral Cloning) 또는 지도 미세조정은 성공 사례를 모방하도록 만든다. 이후 선호 최적화는 품질과 안전성을 향상시키며, 강화학습은 실제 상호작용을 통해 성능을 더욱 개선한다. 마지막으로 안전 모듈(Safety Module)은 학습된 정책과 관계없이 반드시 지켜야 하는 제약 조건을 강제한다.

각 학습 단계 이후에는 반드시 평가(Evaluation)가 수행되어야 한다. 자기지도학습에서는 손실 함수(Loss Curve), 표현 품질(Representation Quality), 데이터 암기(Data Memorization), 전이 성능(Transfer Performance), 창발 능력(Emergent Capability)을 평가한다. 지도 미세조정 이후에는 지시 수행 능력과 작업 성공률을 측정한다. RLHF나 DPO 이후에는 선호 승률(Preference Win Rate), 강건성(Robustness), 사실성(Factuality), 안전성(Safety), 신뢰도(Calibration), 그리고 기존 능력이 유지되었는지를 평가한다. 로봇에서는 여기에 충돌 빈도(Collision Frequency), 제어의 부드러움(Control Smoothness), 행동 지연(Action Latency), 하드웨어 한계(Hardware Limit), 분포 밖 환경(Out-of-Distribution), 장애 복구(Failure Recovery)도 추가적으로 평가해야 한다.

중요한 문제 가운데 하나는 정렬 비용(Alignment Tax)이다. 사용자가 선호하는 방향으로 모델을 정렬하면 오히려 원래의 능력이 감소할 수 있다. 지나치게 안전성을 강조하면 모델은 정상적인 작업도 거부하거나, 지나치게 보수적으로 행동하거나, 다양한 표현 능력을 잃을 수 있다. 로봇에서도 안전성을 지나치게 강조하면 실제 작업을 수행하지 못할 정도로 소극적인 정책이 될 수 있다. 따라서 학습에서는 능력(Capability), 유용성(Helpfulness), 효율성(Efficiency), 안전성(Safety) 사이의 균형이 중요하다.

또 다른 문제는 선호 분포 변화(Preference Distribution Shift)이다. 데이터를 평가한 사람들이 미래의 모든 사용자나 산업 환경, 문화, 국가를 대표하지는 않는다. 대화형 AI에서 적절한 선호가 산업용 로봇에서도 적절한 것은 아니다. 창고 관리자(Warehouse Operator), 안전 관리자(Safety Engineer), 고객(Customer), 규제 기관(Regulator)은 서로 다른 행동을 선호할 수 있다. 따라서 선호 데이터는 실제 적용 환경(Deployment Context)에 맞추어 관리되고 지속적으로 검토되어야 한다.

미래의 파운데이션 모델 학습은 더욱 반복적이고 통합된 구조가 될 것으로 예상된다. 모델은 대규모 멀티모달 데이터(Multimodal Data)를 사전학습하고, 합성 데이터(Synthetic Data)를 생성하며, 자신의 결과를 스스로 비판(Critique)하고 수정(Revision)하며, 실제 사용 환경에서 새로운 선호 데이터를 지속적으로 수집하게 될 것이다. 월드 모델(World Model)은 행동의 결과를 미리 시뮬레이션하고, 검증 시스템(Verification System)은 실제 실행 전에 결과를 확인하며, 인간 전문가는 불확실하거나 위험한 상황에서만 개입하게 될 가능성이 높다.

지속학습(Continual Learning) 역시 중요한 요소가 된다. 실제 운영 중인 모델은 새로운 도구(Tool), 환경(Environment), 규제(Regulation), 로봇 플랫폼(Robot Embodiment), 사용자 요구(User Expectation)에 적응하면서도 기존 능력을 잊지 않아야 한다. 선호 최적화도 일회성 작업이 아니라 지속적으로 수행되는 과정이 될 것이다. 이를 위해서는 강력한 버전 관리(Version Control), 리플레이 데이터셋(Replay Dataset), 회귀 테스트(Regression Testing), 롤백(Rollback), 그리고 행동 변화 모니터링(Behavioral Drift Monitoring)이 필수적이다.

결국 자기지도학습(SSL), 인간 피드백 기반 강화학습(RLHF), 그리고 직접 선호 최적화(DPO)는 서로 경쟁하는 기술이 아니라 하나의 통합 학습 전략(Integrated Training Strategy)을 구성하는 핵심 요소이다. 자기지도학습은 방대한 원시 데이터를 학습 신호로 변환하여 규모(Scale)를 제공하고, RLHF는 복잡한 인간의 선호를 보상 모델과 강화학습을 통해 최적화하며, DPO는 선호 데이터를 보다 단순하고 효율적으로 학습하도록 만든다. 각 방법은 파운데이션 모델이 해결해야 하는 서로 다른 문제를 담당하고 있다.

피지컬 AI에서는 이들의 조합이 더욱 중요하다. 자기지도학습은 방대한 센서 데이터와 행동 데이터에서 공통 구조를 추출하고, 지도학습은 성공 사례를 모방하며, 선호 최적화는 안전성과 효율성, 인간의 기대를 모델에 반영한다. 강화학습은 시뮬레이션과 실제 환경에서 성능을 지속적으로 개선한다. 이러한 학습 패러다임의 결합은 단순히 정보를 예측하는 모델을 넘어, 환경을 인식하고, 목표를 이해하며, 행동을 계획하고, 실제 물리 세계에서 안전하게 동작하는 피지컬 AI 파운데이션 모델을 구축하는 핵심 기반이 된다.

##  

## 1.4 Scaling Laws and Compute Data Parameter Trade-offs

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

Scaling laws describe how the performance of machine learning models changes as the amount of computation, training data, and model capacity increases. They provide an empirical framework for predicting how much improvement can be expected when a foundation model is made larger or trained for longer. Rather than treating model development as a sequence of isolated experiments, scaling laws allow researchers to estimate relationships among resources and performance before committing to extremely expensive training runs. For foundation models, this is especially important because training decisions may involve billions of parameters, trillions of tokens, thousands of accelerators, and months of computation.

The basic observation behind scaling laws is that model loss often decreases in a relatively regular way as scale increases. When performance is plotted against the amount of data, computation, or number of parameters on logarithmic axes, the resulting curve frequently resembles a power law. This means that predictable improvements can continue over a broad range of scales, although each additional unit of resource usually produces a smaller gain than the previous one. Scaling is therefore powerful but not free. Better performance generally requires rapidly increasing investment.

Model parameters determine the representational capacity of a neural network. A parameter is a learned numerical value that influences how the model transforms its inputs into outputs. Increasing the number of parameters allows the model to represent more complex patterns, store more distributed knowledge, and approximate more sophisticated functions. A small model may capture basic grammatical structures or visual categories, while a much larger model may represent subtle relationships among concepts, contexts, modalities, and tasks.

Parameter count alone, however, does not guarantee better performance. A large model trained on insufficient data may fail to use its capacity effectively. It may memorize repeated examples, overfit particular patterns, or remain undertrained because its parameters do not receive enough informative updates. In this case, additional parameters increase memory use and inference cost without delivering proportional capability improvements.

Training data provides the experiences from which the model learns. For language models, data may consist of tokens from books, websites, source code, scientific documents, conversations, and synthetic examples. For multimodal models, it may include image-text pairs, audio, video, diagrams, and documents. For Physical AI, data can additionally include camera streams, point clouds, robot states, control commands, tactile observations, force measurements, simulation trajectories, and human demonstrations.

Increasing data volume exposes a model to more linguistic, visual, behavioral, and physical variation. This improves coverage and reduces dependence on a narrow training distribution. A model trained on diverse environments is more likely to generalize to unfamiliar situations than one trained on repeated examples from a limited domain. However, data quantity cannot be considered separately from quality. Large collections of duplicated, low-quality, inconsistent, or incorrectly aligned samples may provide less value than a smaller, carefully curated dataset.

Computation connects model capacity and data. Training compute represents the total number of mathematical operations performed while optimizing a model. It depends on model size, sequence length, batch size, number of training tokens, architecture, numerical precision, and optimization method. Larger models require more operations for every token, while larger datasets require more training steps. Increasing both simultaneously can cause computational requirements to grow extremely quickly.

A useful way to understand the scaling problem is to view model parameters, data, and compute as a resource triangle. If compute is fixed, increasing model size leaves fewer resources for processing data. If the dataset is fixed, increasing compute may lead to repeated training over the same information and eventual overfitting. If the model is too small, additional data may not be fully absorbed because the model lacks sufficient capacity. The best configuration depends on balancing all three resources.

Early large-model development often emphasized increasing parameter count. Researchers observed that larger models generally achieved lower training loss and stronger downstream performance. This encouraged a race toward models with tens, hundreds, and eventually thousands of billions of parameters. However, subsequent analysis showed that many large models were undertrained relative to their size. They had enough parameters to absorb more information but had not been exposed to enough tokens.

Compute-optimal training attempts to determine the model size and data volume that make the best use of a fixed computational budget. Instead of spending most resources on parameter count, a compute-optimal strategy allocates sufficient training data to ensure that the model's capacity is meaningfully utilized. Under such a strategy, a smaller model trained on substantially more high-quality data can outperform a much larger model trained on too few tokens.

This insight has major implications for engineering. A larger model is not always the most capable model for a given budget. Training efficiency depends on how often each parameter is updated with useful information. If a model is extremely large but sees too little data, its capacity remains poorly trained. If a model is very small but trained on a vast dataset, it may reach a capacity limit and fail to represent the complexity of the data. Compute-optimal design seeks the point where these constraints are balanced.

Scaling behavior is typically measured using training loss, validation loss, benchmark accuracy, or downstream task performance. Training loss indicates how well the model fits its observed data. Validation loss measures how well it predicts unseen samples from a related distribution. Downstream benchmarks assess abilities such as reasoning, coding, translation, image understanding, manipulation, or navigation. These metrics do not always improve at the same rate.

Loss may decrease smoothly while useful capabilities appear unevenly. A small improvement in average prediction loss can correspond to a large change in the reliability of a specific task. Conversely, a model may achieve lower pre-training loss without significant improvement on safety, factuality, or long-horizon reasoning. Scaling laws are therefore most reliable for aggregate prediction performance and less reliable for every individual behavior.

Emergent capabilities complicate scaling predictions. Certain abilities appear weak or absent in smaller models and become visible only after scale passes a threshold. Examples may include multi-step reasoning, tool use, code execution, instruction following, or cross-modal transfer. Some apparent emergence may result from evaluation methods that use strict success thresholds. Nevertheless, nonlinear capability changes remain important for planning because smooth improvements in loss do not guarantee smooth improvements in practical behavior.

Data scaling also introduces the problem of diminishing availability. High-quality human-generated data is finite. As training corpora grow, organizations may exhaust easily accessible sources and begin incorporating lower-quality, repeated, synthetic, or weakly filtered material. At that point, adding more tokens may produce limited value or even degrade performance. Effective scaling therefore requires better data selection rather than indiscriminate accumulation.

Data curation includes deduplication, quality scoring, source balancing, language balancing, toxicity filtering, privacy protection, formatting normalization, and contamination control. For code models, valid and well-documented repositories may be more valuable than large amounts of broken code. For reasoning models, carefully constructed mathematical and logical examples may contribute more than repetitive web text. For robot models, successful, diverse, and precisely synchronized trajectories may be more useful than much larger quantities of noisy demonstrations.

Synthetic data has become an important response to data scarcity. A capable model can generate explanations, code, questions, trajectories, plans, and alternative solutions that are later used for training. Synthetic data can target rare skills or fill gaps in the original distribution. In Physical AI, simulation can generate variations of lighting, objects, terrain, sensor noise, robot morphology, and failure conditions that would be costly or dangerous to collect in reality.

Synthetic scaling also creates risks. If models train repeatedly on their own unverified outputs, errors may be reinforced. Generated examples may lack the diversity and unpredictability of real-world data. Poor synthetic data can narrow the model's distribution and encourage superficial patterns. Consequently, synthetic examples require filtering, verification, diversity controls, and combination with trusted real data.

The compute dimension of scaling is shaped by hardware. Training large foundation models requires accelerators with high matrix multiplication throughput, large memory capacity, fast interconnects, and efficient collective communication. Compute availability is not simply the number of GPUs. Performance depends on accelerator utilization, memory bandwidth, network topology, software kernels, parallelism strategy, and storage throughput.

Several forms of parallelism are used to distribute large training workloads. Data parallelism replicates the model across devices while processing different batches. Tensor parallelism divides individual matrix operations across accelerators. Pipeline parallelism assigns different model layers to different devices. Sequence parallelism distributes long input sequences. Expert parallelism distributes components of mixture-of-experts models. Practical systems often combine these techniques.

Poor parallel efficiency can make nominal compute misleading. Thousands of accelerators may spend significant time waiting for communication, data loading, synchronization, or memory transfers. The effective compute used for learning can therefore be far lower than the theoretical hardware peak. Scaling laws based only on accelerator count may overestimate expected progress unless utilization is carefully measured.

Numerical precision also affects compute efficiency. Training traditionally used 32-bit floating-point arithmetic, but modern systems increasingly use 16-bit or lower-precision formats while maintaining selected operations at higher precision for stability. Mixed-precision training reduces memory consumption and increases throughput. Quantized training and specialized low-precision hardware may extend these gains further, although numerical errors must be controlled.

The relationship between training compute and inference compute is another important trade-off. A very large model may achieve excellent benchmark performance but require expensive hardware and high latency every time it generates an output. For a frequently used service or a robot fleet, total inference cost can exceed the original training cost. Model design must therefore consider lifecycle compute rather than training compute alone.

This consideration has encouraged the development of smaller models trained on more data, model compression, quantization, pruning, and knowledge distillation. A large teacher model can generate training signals for a smaller student model. The student may sacrifice some generality but offer faster and cheaper deployment. For edge systems, a compact model with reliable task-specific performance may be more valuable than a frontier-scale model that cannot run within latency, power, or memory constraints.

Mixture-of-experts architectures offer another scaling strategy. Instead of activating every parameter for every token, they route each input through a subset of specialized expert modules. This allows total parameter count to increase without proportionally increasing computation per token. The model can develop specialization while keeping active compute relatively controlled.

However, mixture-of-experts systems introduce routing complexity, communication overhead, load-balancing problems, and deployment challenges. If too many tokens are sent to the same expert, some devices become overloaded while others remain idle. Experts may also become overly specialized or underused. The distinction between total parameters and active parameters therefore becomes important when comparing model scale.

Context length is an additional scaling dimension. Longer contexts allow models to process larger documents, extended conversations, video sequences, or long robot histories. However, standard self-attention has computational and memory costs that grow rapidly with sequence length. Increasing context length can therefore consume resources that might otherwise be used for more parameters or more training samples.

Long-context scaling requires efficient attention, memory compression, recurrence, retrieval, sparse patterns, or state-space mechanisms. Simply extending the maximum input length does not guarantee that the model will use distant information effectively. Models must be trained on long sequences, evaluated for retrieval and reasoning across distance, and protected against irrelevant context overwhelming important signals.

For multimodal foundation models, compute allocation becomes even more complex. Text tokens, image patches, video frames, audio samples, point clouds, and robot states have different information densities and computational costs. A few seconds of high-resolution video may produce far more tokens than a page of text. Without careful sampling and compression, visual data can dominate the compute budget.

Multimodal scaling therefore requires modality-specific encoders, token reduction, temporal subsampling, latent compression, adaptive resolution, and balanced data mixing. The objective is not to process every raw measurement at maximum resolution but to preserve task-relevant information. For example, a robot may need high temporal resolution for joint control but lower visual resolution for slowly changing background regions.

Physical AI makes scaling laws more difficult because robot data is not independent and identically distributed. Consecutive observations are strongly correlated, demonstrations vary in quality, and different robots produce different action spaces. Ten million nearly identical trajectories may contain less useful information than a much smaller set of diverse interactions. Effective data volume must therefore be measured in terms of behavioral diversity, environmental coverage, and information content rather than raw sample count alone.

Robot embodiment also changes the parameter-data trade-off. A model trained for one manipulator may learn detailed platform-specific dynamics but transfer poorly to another robot. A cross-embodiment model requires more diverse data and enough capacity to represent shared abstractions as well as hardware-specific differences. Robot identity, morphology, sensor configuration, and control interface may need to be explicitly encoded.

The cost of real-world robot data places a practical limit on direct scaling. Collecting demonstrations requires hardware, operators, maintenance, storage, calibration, and safety supervision. Simulation can expand the dataset, but simulated data differs from reality. A scalable Physical AI pipeline therefore mixes web-scale semantic data, simulation trajectories, synthetic environments, teleoperation demonstrations, autonomous experience, and carefully selected real-world trials.

Compute allocation in robotics must also consider control hierarchy. Large foundation models may operate at a slower rate to interpret goals, understand scenes, or generate action chunks. Smaller policies and conventional controllers may execute high-frequency motor commands. This hierarchical arrangement prevents expensive reasoning models from being placed directly inside tight real-time control loops.

A practical robot foundation model may therefore use substantial training compute but limited active inference compute. A large cloud or data-center model can be distilled into a smaller edge model. High-level planning can run on an edge GPU at a moderate frequency, while safety controllers and servo loops run on deterministic processors. Scaling decisions must reflect the full architecture rather than only the central neural network.

Energy consumption is another important consequence of scale. Large training runs consume substantial electricity, and large-scale inference across millions of requests or thousands of robots creates ongoing energy costs. Improving energy efficiency is not only an environmental concern but also a business and deployment requirement. In mobile robots, every watt used for computation reduces the energy available for movement, sensing, and mission duration.

Scaling laws can support cost forecasting. By fitting performance curves from smaller experiments, researchers can estimate the compute required to reach a target loss or benchmark level. These estimates help determine whether a training goal is economically realistic. However, extrapolation is risky when architecture, data quality, objective functions, or evaluation distributions change. Scaling laws are empirical regularities, not universal physical laws.

Small-scale pilot experiments are therefore essential. Teams can train several models with different parameter counts and data allocations, measure their losses, and estimate the likely behavior of larger runs. This process helps identify compute-optimal configurations, data bottlenecks, training instability, and diminishing returns before full-scale investment.

Scaling strategy should also reflect the intended product. A research model may prioritize maximum capability. An enterprise assistant may prioritize factuality and controllability. An industrial robot may prioritize latency, safety, and predictable behavior. A fleet platform may prioritize total cost per mission. The optimal parameter count and compute budget differ across these contexts.

In many applications, scaling the surrounding system can produce greater value than scaling the base model. Retrieval systems can provide updated knowledge. Tool use can improve calculation and execution. External memory can extend context. Verification models can reduce errors. World models can evaluate action consequences. Specialized smaller models can handle perception or control more efficiently than a single monolithic network.

This leads to a broader interpretation of scaling. The future may not consist only of larger individual models but of scalable systems composed of models, tools, memory, simulators, databases, and controllers. Compute can be allocated dynamically depending on task difficulty. Simple requests may use a small model, while difficult problems invoke additional reasoning steps, external tools, or larger models.

Inference-time scaling has become increasingly important in this context. Instead of spending all available resources during pre-training, systems can allocate more computation when solving difficult tasks. They may generate multiple candidate solutions, verify intermediate steps, search through action sequences, or consult external models. This creates another trade-off between fixed training investment and adaptive inference cost.

For robotics, inference-time scaling may involve simulating multiple future trajectories before selecting an action. A world model can predict likely consequences, while a planner evaluates safety and utility. More compute can improve decisions in difficult situations, but strict latency limits prevent unlimited search. The system must determine when additional computation is worth the delay.

Ultimately, scaling laws provide guidance rather than a complete recipe. They reveal that performance tends to improve predictably with more parameters, data, and compute, but they also expose the importance of balance. Oversized models can be undertrained. Huge datasets can be low quality. Powerful hardware can be poorly utilized. Better performance emerges when model capacity, informative data, optimization, and system architecture are matched to one another.

For Physical AI, the central challenge is not simply to build the largest possible robot model. It is to allocate compute, data, and parameters in a way that produces robust perception, transferable knowledge, safe action generation, and efficient deployment. A well-balanced model trained on diverse, high-quality robot experience may outperform a much larger but poorly trained system. Scaling must therefore be treated as an engineering discipline that connects learning theory, data strategy, hardware infrastructure, model architecture, operational cost, and real-world safety.

확장 법칙(Scaling Laws)은 머신러닝(Machine Learning) 모델의 성능이 계산 자원(Compute), 학습 데이터(Training Data), 모델 용량(Model Capacity)의 증가에 따라 어떻게 변화하는지를 설명한다. 이는 파운데이션 모델(Foundation Model)을 더 크게 만들거나 더 오랫동안 학습했을 때 어느 정도의 성능 향상을 기대할 수 있는지를 예측하는 경험적 프레임워크(Empirical Framework)를 제공한다. 모델 개발을 서로 독립된 실험의 연속으로만 취급하는 대신, 확장 법칙은 매우 비용이 많이 드는 학습을 시작하기 전에 자원과 성능 간의 관계를 추정할 수 있도록 한다. 파운데이션 모델에서는 수십억 개의 파라미터(Parameter), 수조 개의 토큰(Token), 수천 개의 가속기(Accelerator), 수개월의 계산 시간이 필요할 수 있으므로 이러한 예측이 특히 중요하다.

확장 법칙의 기본적인 관찰은 규모(Scale)가 증가할수록 모델의 손실(Loss)이 비교적 규칙적인 방식으로 감소한다는 것이다. 데이터 양, 계산량, 파라미터 수에 대한 성능을 로그 축(Logarithmic Axis)에 표시하면 결과 곡선은 흔히 거듭제곱 법칙(Power Law)과 유사한 형태를 보인다. 이는 매우 넓은 규모 범위에서 예측 가능한 성능 향상이 계속될 수 있음을 의미한다. 그러나 일반적으로 자원을 추가할수록 추가 자원 한 단위가 제공하는 성능 향상은 이전보다 점차 감소한다. 따라서 확장(Scaling)은 강력하지만 비용이 없는 것은 아니다. 더 나은 성능을 얻기 위해서는 일반적으로 빠르게 증가하는 투자가 필요하다.

모델 파라미터(Model Parameters)는 신경망(Neural Network)의 표현 용량(Representational Capacity)을 결정한다. 파라미터는 모델이 입력을 출력으로 변환하는 방식에 영향을 주는 학습 가능한 수치이다. 파라미터 수를 증가시키면 모델은 더 복잡한 패턴을 표현하고, 더 많은 분산 지식(Distributed Knowledge)을 저장하며, 더 정교한 함수를 근사할 수 있다. 작은 모델은 기본적인 문법 구조나 시각적 범주를 학습할 수 있지만, 훨씬 큰 모델은 개념, 문맥, 모달리티(Modality), 작업(Task) 사이의 미세한 관계까지 표현할 수 있다.

그러나 파라미터 수만으로 더 나은 성능이 보장되는 것은 아니다. 충분하지 않은 데이터로 학습된 대형 모델은 자신의 용량을 효과적으로 활용하지 못할 수 있다. 반복되는 사례를 암기하거나, 특정 패턴에 과적합(Overfitting)되거나, 파라미터가 충분한 정보성 업데이트(Informative Update)를 받지 못해 과소학습(Undertraining) 상태로 남을 수 있다. 이러한 경우 추가 파라미터는 이에 비례하는 능력 향상을 제공하지 못하면서 메모리 사용량과 추론 비용(Inference Cost)만 증가시킨다.

학습 데이터(Training Data)는 모델이 학습하는 경험을 제공한다. 언어 모델(Language Model)의 데이터는 서적, 웹사이트, 소스 코드(Source Code), 과학 문서, 대화, 합성 예제(Synthetic Example)의 토큰으로 구성될 수 있다. 멀티모달 모델(Multimodal Model)에는 이미지-텍스트 쌍(Image-Text Pair), 오디오(Audio), 비디오(Video), 도표(Diagram), 문서(Document)가 포함될 수 있다. 피지컬 AI(Physical AI)의 경우 카메라 스트림(Camera Stream), 포인트 클라우드(Point Cloud), 로봇 상태(Robot State), 제어 명령(Control Command), 촉각 관측(Tactile Observation), 힘 측정(Force Measurement), 시뮬레이션 궤적(Simulation Trajectory), 인간 시연(Human Demonstration)까지 추가로 포함될 수 있다.

데이터 양을 증가시키면 모델은 더 다양한 언어적, 시각적, 행동적, 물리적 변화를 경험하게 된다. 이는 데이터 범위를 확대하고 좁은 학습 분포(Training Distribution)에 대한 의존성을 감소시킨다. 다양한 환경에서 학습한 모델은 제한된 도메인의 반복 사례만 학습한 모델보다 익숙하지 않은 상황에 일반화(Generalization)할 가능성이 높다. 그러나 데이터의 양은 품질과 분리해서 생각할 수 없다. 중복되고 품질이 낮으며 일관성이 없거나 잘못 정렬된 대규모 데이터는, 양이 더 적더라도 신중하게 정제된 데이터셋보다 가치가 낮을 수 있다.

계산 자원(Compute)은 모델 용량과 데이터를 연결한다. 학습 계산량(Training Compute)은 모델을 최적화하는 동안 수행되는 전체 수학 연산의 수를 의미한다. 이는 모델 크기, 시퀀스 길이(Sequence Length), 배치 크기(Batch Size), 학습 토큰 수, 아키텍처(Architecture), 수치 정밀도(Numerical Precision), 최적화 방식에 따라 달라진다. 더 큰 모델은 각각의 토큰을 처리할 때 더 많은 연산을 요구하고, 더 큰 데이터셋은 더 많은 학습 단계를 필요로 한다. 두 요소를 동시에 증가시키면 계산 요구량은 매우 빠르게 증가할 수 있다.

확장 문제를 이해하는 유용한 방법은 모델 파라미터, 데이터, 계산 자원을 하나의 자원 삼각형(Resource Triangle)으로 보는 것이다. 계산 자원이 고정되어 있다면 모델 크기를 증가시킬수록 데이터를 처리하는 데 사용할 수 있는 자원은 줄어든다. 데이터셋이 고정되어 있다면 계산량을 증가시키는 것은 동일한 정보를 반복적으로 학습하게 하여 결국 과적합을 일으킬 수 있다. 모델이 너무 작다면 충분한 용량이 부족하여 추가 데이터를 완전히 흡수하지 못할 수 있다. 최적의 구성은 이 세 가지 자원의 균형을 맞추는 데 달려 있다.

초기의 대형 모델 개발은 주로 파라미터 수를 증가시키는 데 초점을 맞추었다. 연구자들은 일반적으로 더 큰 모델이 더 낮은 학습 손실과 더 높은 하위 작업(Downstream Task) 성능을 달성한다는 사실을 관찰하였다. 이는 수백억, 수천억, 궁극적으로 수조 개의 파라미터를 가진 모델을 개발하려는 경쟁을 촉진하였다. 그러나 이후의 분석에서는 많은 대형 모델이 자신의 크기에 비해 충분히 학습되지 않았다는 점이 확인되었다. 이들은 더 많은 정보를 흡수할 수 있는 충분한 파라미터를 가지고 있었지만, 충분한 토큰에 노출되지 않았다.

계산 최적 학습(Compute-Optimal Training)은 고정된 계산 예산(Compute Budget)을 가장 효과적으로 활용할 수 있는 모델 크기와 데이터 양을 결정하려는 접근법이다. 대부분의 자원을 파라미터 수에 사용하는 대신, 계산 최적 전략은 모델의 용량이 의미 있게 활용될 수 있도록 충분한 학습 데이터를 배정한다. 이러한 전략에서는 훨씬 더 많은 고품질 데이터로 학습된 작은 모델이, 너무 적은 토큰으로 학습된 훨씬 큰 모델보다 더 높은 성능을 낼 수 있다.

이러한 통찰은 엔지니어링(Engineering)에 중요한 의미를 가진다. 주어진 예산에서 더 큰 모델이 항상 더 높은 능력을 제공하는 것은 아니다. 학습 효율은 각 파라미터가 유용한 정보로 얼마나 자주 업데이트되는지에 달려 있다. 모델이 극도로 크지만 너무 적은 데이터만 본다면 그 용량은 충분히 학습되지 않는다. 모델이 매우 작지만 방대한 데이터로 학습된다면 용량 한계에 도달하여 데이터의 복잡성을 표현하지 못할 수 있다. 계산 최적 설계(Compute-Optimal Design)는 이러한 제약들이 균형을 이루는 지점을 찾는다.

확장 행동(Scaling Behavior)은 일반적으로 학습 손실(Training Loss), 검증 손실(Validation Loss), 벤치마크 정확도(Benchmark Accuracy), 하위 작업 성능(Downstream Task Performance)을 이용하여 측정된다. 학습 손실은 모델이 관측한 데이터에 얼마나 잘 적합하는지를 나타낸다. 검증 손실은 관련된 분포의 새로운 샘플을 얼마나 잘 예측하는지를 나타낸다. 하위 작업 벤치마크는 추론(Reasoning), 코딩(Coding), 번역(Translation), 이미지 이해(Image Understanding), 조작(Manipulation), 내비게이션(Navigation)과 같은 능력을 평가한다. 이러한 지표들은 항상 동일한 속도로 향상되지 않는다.

손실은 부드럽게 감소할 수 있지만 유용한 능력은 불균일하게 나타날 수 있다. 평균 예측 손실의 작은 향상이 특정 작업의 신뢰성에는 큰 변화를 가져올 수 있다. 반대로 모델이 더 낮은 사전학습 손실을 달성하더라도 안전성(Safety), 사실성(Factuality), 장기 추론(Long-Horizon Reasoning)에서는 큰 향상이 나타나지 않을 수 있다. 따라서 확장 법칙은 전체적인 예측 성능에는 비교적 신뢰할 수 있지만, 모든 개별 행동에 대해서는 덜 신뢰할 수 있다.

창발 능력(Emergent Capability)은 확장 예측을 더욱 복잡하게 만든다. 일부 능력은 작은 모델에서는 약하거나 전혀 나타나지 않다가 규모가 특정 임계값을 넘은 뒤에야 관찰된다. 다단계 추론(Multi-Step Reasoning), 도구 사용(Tool Use), 코드 실행(Code Execution), 지시 수행(Instruction Following), 교차 모달 전이(Cross-Modal Transfer)가 그 예가 될 수 있다. 일부 창발 현상은 엄격한 성공 임계값을 사용하는 평가 방식에서 비롯될 수도 있다. 그럼에도 비선형적인 능력 변화는 학습 손실의 부드러운 개선이 실제 행동의 부드러운 개선을 보장하지 않는다는 점에서 중요한 의미를 가진다.

데이터 확장은 고품질 데이터의 가용성이 점차 감소하는 문제도 발생시킨다. 고품질 인간 생성 데이터(Human-Generated Data)는 유한하다. 학습 말뭉치(Training Corpus)가 커질수록 조직은 쉽게 접근할 수 있는 데이터 소스를 소진하고, 품질이 낮거나 반복되거나 합성되거나 약하게 필터링된 자료를 포함하기 시작할 수 있다. 이 단계에서는 토큰을 더 추가해도 제한적인 가치만 제공하거나 성능을 저하시킬 수도 있다. 따라서 효과적인 확장은 무분별한 데이터 축적이 아니라 더 나은 데이터 선택을 요구한다.

데이터 정제(Data Curation)는 중복 제거(Deduplication), 품질 점수화(Quality Scoring), 출처 균형(Source Balancing), 언어 균형(Language Balancing), 유해성 필터링(Toxicity Filtering), 개인정보 보호(Privacy Protection), 형식 정규화(Formatting Normalization), 오염 통제(Contamination Control)를 포함한다. 코드 모델에서는 유효하고 문서화가 잘된 저장소가 대량의 손상된 코드보다 더 가치 있을 수 있다. 추론 모델에서는 정교하게 구성된 수학 및 논리 예제가 반복적인 웹 문서보다 더 큰 기여를 할 수 있다. 로봇 모델에서는 성공적이고 다양하며 정확하게 동기화된 궤적이 훨씬 더 많은 양의 잡음이 포함된 시연보다 유용할 수 있다.

합성 데이터(Synthetic Data)는 데이터 부족 문제에 대응하는 중요한 방법이 되었다. 충분한 능력을 가진 모델은 설명, 코드, 질문, 궤적, 계획, 대안적 해결책을 생성할 수 있으며, 이후 이를 학습에 사용할 수 있다. 합성 데이터는 희귀한 기술을 집중적으로 학습하거나 기존 데이터 분포의 빈 영역을 채울 수 있다. 피지컬 AI에서는 시뮬레이션을 이용하여 조명, 물체, 지형, 센서 노이즈, 로봇 형태(Robot Morphology), 실패 조건의 변형을 생성할 수 있으며, 이러한 사례는 실제 세계에서 수집하기에 지나치게 비싸거나 위험할 수 있다.

그러나 합성 데이터의 확장은 위험도 만든다. 모델이 검증되지 않은 자신의 출력을 반복적으로 학습하면 오류가 강화될 수 있다. 생성된 예제는 실제 세계 데이터가 가지는 다양성과 예측 불가능성을 충분히 포함하지 못할 수 있다. 품질이 낮은 합성 데이터는 모델의 분포를 좁히고 피상적인 패턴을 학습하도록 만들 수 있다. 따라서 합성 예제는 필터링, 검증(Verification), 다양성 통제(Diversity Control), 신뢰할 수 있는 실제 데이터와의 결합이 필요하다.

확장의 계산 자원 측면은 하드웨어(Hardware)의 영향을 받는다. 대형 파운데이션 모델을 학습하려면 높은 행렬 곱셈 처리량(Matrix Multiplication Throughput), 큰 메모리 용량(Memory Capacity), 빠른 인터커넥트(Interconnect), 효율적인 집단 통신(Collective Communication)을 제공하는 가속기가 필요하다. 계산 자원의 가용성은 단순히 GPU의 수를 의미하지 않는다. 성능은 가속기 활용률(Accelerator Utilization), 메모리 대역폭(Memory Bandwidth), 네트워크 토폴로지(Network Topology), 소프트웨어 커널(Software Kernel), 병렬화 전략(Parallelism Strategy), 저장장치 처리량(Storage Throughput)에 따라 달라진다.

대규모 학습 작업을 분산하기 위해 여러 형태의 병렬화가 사용된다. 데이터 병렬화(Data Parallelism)는 동일한 모델을 여러 장치에 복제하면서 서로 다른 배치를 처리한다. 텐서 병렬화(Tensor Parallelism)는 개별 행렬 연산을 여러 가속기에 나눈다. 파이프라인 병렬화(Pipeline Parallelism)는 서로 다른 모델 계층을 여러 장치에 할당한다. 시퀀스 병렬화(Sequence Parallelism)는 긴 입력 시퀀스를 분산한다. 전문가 병렬화(Expert Parallelism)는 전문가 혼합(Mixture of Experts) 모델의 구성요소를 분산한다. 실제 시스템에서는 이러한 기술들이 함께 사용되는 경우가 많다.

병렬화 효율이 낮으면 명목상의 계산량은 오해를 일으킬 수 있다. 수천 개의 가속기가 통신, 데이터 로딩(Data Loading), 동기화(Synchronization), 메모리 전송을 기다리는 데 많은 시간을 소비할 수 있다. 따라서 실제 학습에 사용되는 유효 계산량(Effective Compute)은 이론적인 하드웨어 최대치보다 훨씬 낮을 수 있다. 가속기 수만을 기준으로 하는 확장 법칙은 활용률을 신중하게 측정하지 않는다면 예상되는 발전을 과대평가할 수 있다.

수치 정밀도(Numerical Precision)도 계산 효율에 영향을 미친다. 전통적으로 학습에는 32비트 부동소수점(Floating-Point) 연산이 사용되었지만, 현대 시스템은 안정성을 위해 일부 연산만 높은 정밀도로 유지하면서 16비트 또는 더 낮은 정밀도 형식을 점점 더 많이 사용한다. 혼합 정밀도 학습(Mixed-Precision Training)은 메모리 소비를 줄이고 처리량을 증가시킨다. 양자화 학습(Quantized Training)과 저정밀도 전용 하드웨어는 이러한 이점을 더욱 확대할 수 있지만, 수치 오류(Numerical Error)는 통제되어야 한다.

학습 계산량과 추론 계산량(Inference Compute)의 관계는 또 다른 중요한 상충 관계이다. 매우 큰 모델은 우수한 벤치마크 성능을 달성할 수 있지만, 출력을 생성할 때마다 비싼 하드웨어와 높은 지연 시간(Latency)을 요구할 수 있다. 자주 사용되는 서비스나 로봇 플릿(Robot Fleet)에서는 전체 추론 비용이 초기 학습 비용을 초과할 수 있다. 따라서 모델 설계는 학습 계산량뿐 아니라 전체 수명주기 계산량(Lifecycle Compute)을 고려해야 한다.

이러한 고려는 더 많은 데이터로 학습된 소형 모델, 모델 압축(Model Compression), 양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation)의 발전을 촉진하였다. 대형 교사 모델(Teacher Model)은 작은 학생 모델(Student Model)을 위한 학습 신호를 생성할 수 있다. 학생 모델은 일부 일반성을 희생할 수 있지만 더 빠르고 저렴한 배포를 제공한다. 엣지 시스템(Edge System)에서는 지연 시간, 전력, 메모리 제약 안에서 동작하지 못하는 프런티어 규모의 모델보다, 안정적인 작업별 성능을 제공하는 소형 모델이 더 가치 있을 수 있다.

전문가 혼합(Mixture of Experts) 아키텍처는 또 다른 확장 전략을 제공한다. 모든 토큰에 대해 모든 파라미터를 활성화하는 대신 각 입력을 일부 전문 모듈(Expert Module)로 라우팅(Routing)한다. 이를 통해 토큰당 계산량을 비례해서 증가시키지 않으면서 전체 파라미터 수를 늘릴 수 있다. 모델은 활성 계산량(Active Compute)을 비교적 통제된 상태로 유지하면서 전문화를 발전시킬 수 있다.

그러나 전문가 혼합 시스템은 라우팅 복잡성, 통신 오버헤드(Communication Overhead), 부하 균형(Load Balancing) 문제, 배포상의 어려움을 발생시킨다. 너무 많은 토큰이 동일한 전문가에게 전달되면 일부 장치는 과부하되고 다른 장치는 유휴 상태가 된다. 전문가가 지나치게 특화되거나 거의 사용되지 않을 수도 있다. 따라서 모델 규모를 비교할 때 전체 파라미터(Total Parameters)와 활성 파라미터(Active Parameters)를 구분하는 것이 중요하다.

컨텍스트 길이(Context Length)는 또 하나의 확장 차원이다. 더 긴 컨텍스트는 모델이 더 큰 문서, 장시간 대화, 비디오 시퀀스(Video Sequence), 긴 로봇 이력을 처리할 수 있도록 한다. 그러나 표준 자기 주의(Self-Attention)는 시퀀스 길이에 따라 계산 비용과 메모리 비용이 빠르게 증가한다. 따라서 컨텍스트 길이를 늘리면 더 많은 파라미터나 더 많은 학습 샘플에 사용할 수 있었던 자원을 소비하게 된다.

긴 컨텍스트 확장에는 효율적 어텐션(Efficient Attention), 메모리 압축(Memory Compression), 순환 구조(Recurrence), 검색(Retrieval), 희소 패턴(Sparse Pattern), 상태공간 메커니즘(State-Space Mechanism)이 필요하다. 최대 입력 길이만 단순히 확장한다고 해서 모델이 멀리 떨어진 정보를 효과적으로 사용할 수 있는 것은 아니다. 모델은 긴 시퀀스로 실제 학습되어야 하고, 장거리 검색과 추론 능력을 평가받아야 하며, 불필요한 문맥이 중요한 신호를 압도하지 않도록 보호되어야 한다.

멀티모달 파운데이션 모델에서는 계산 자원 배분이 더욱 복잡해진다. 텍스트 토큰, 이미지 패치(Image Patch), 비디오 프레임(Video Frame), 오디오 샘플(Audio Sample), 포인트 클라우드, 로봇 상태는 서로 다른 정보 밀도와 계산 비용을 가진다. 몇 초 분량의 고해상도 비디오는 한 페이지의 텍스트보다 훨씬 많은 토큰을 생성할 수 있다. 신중한 샘플링과 압축이 없다면 시각 데이터가 전체 계산 예산을 지배할 수 있다.

따라서 멀티모달 확장에는 모달리티별 인코더(Modality-Specific Encoder), 토큰 축소(Token Reduction), 시간적 서브샘플링(Temporal Subsampling), 잠재 압축(Latent Compression), 적응형 해상도(Adaptive Resolution), 균형 잡힌 데이터 혼합(Balanced Data Mixing)이 필요하다. 목표는 모든 원시 측정값을 최대 해상도로 처리하는 것이 아니라 작업과 관련된 정보를 보존하는 것이다. 예를 들어 로봇은 관절 제어를 위해서는 높은 시간 해상도가 필요하지만, 천천히 변화하는 배경 영역에는 더 낮은 시각 해상도로도 충분할 수 있다.

피지컬 AI에서는 로봇 데이터가 독립 동일 분포(Independent and Identically Distributed)가 아니기 때문에 확장 법칙을 적용하기가 더욱 어렵다. 연속된 관측은 서로 강하게 상관되어 있고, 시연 품질은 다양하며, 서로 다른 로봇은 서로 다른 행동 공간(Action Space)을 생성한다. 거의 동일한 천만 개의 궤적은 훨씬 적은 수의 다양한 상호작용보다 유용한 정보를 적게 포함할 수 있다. 따라서 유효 데이터 양(Effective Data Volume)은 단순한 샘플 수가 아니라 행동 다양성(Behavioral Diversity), 환경 범위(Environmental Coverage), 정보량(Information Content)을 기준으로 측정해야 한다.

로봇의 체화(Embodiment)는 파라미터와 데이터 간의 상충 관계도 변화시킨다. 하나의 매니퓰레이터(Manipulator)를 위해 학습된 모델은 해당 플랫폼의 세부적인 동역학을 학습할 수 있지만, 다른 로봇으로는 제대로 전이되지 않을 수 있다. 교차 체화 모델(Cross-Embodiment Model)은 공유된 추상 개념과 하드웨어별 차이를 모두 표현하기 위해 더 다양한 데이터와 충분한 용량을 필요로 한다. 로봇 식별자(Robot Identity), 형태(Morphology), 센서 구성(Sensor Configuration), 제어 인터페이스(Control Interface)를 명시적으로 인코딩해야 할 수도 있다.

실제 로봇 데이터의 비용은 직접적인 확장에 현실적인 한계를 만든다. 시연을 수집하려면 하드웨어, 작업자, 유지보수, 저장장치, 보정(Calibration), 안전 감독(Safety Supervision)이 필요하다. 시뮬레이션은 데이터셋을 확대할 수 있지만 시뮬레이션 데이터는 현실과 다르다. 따라서 확장 가능한 피지컬 AI 파이프라인은 웹 규모의 의미 데이터(Web-Scale Semantic Data), 시뮬레이션 궤적, 합성 환경, 원격조작 시연(Teleoperation Demonstration), 자율 경험(Autonomous Experience), 신중하게 선택된 실제 환경 시험을 혼합해야 한다.

로보틱스에서 계산 자원을 배분할 때는 제어 계층(Control Hierarchy)도 고려해야 한다. 대형 파운데이션 모델은 목표를 해석하고, 장면을 이해하며, 행동 청크(Action Chunk)를 생성하기 위해 비교적 느린 속도로 동작할 수 있다. 더 작은 정책 모델과 기존 제어기는 고주파 모터 명령을 실행할 수 있다. 이러한 계층적 구성은 비용이 높은 추론 모델이 엄격한 실시간 제어 루프 안에 직접 배치되는 것을 방지한다.

따라서 실용적인 로봇 파운데이션 모델은 학습에는 많은 계산 자원을 사용하면서도 제한된 활성 추론 계산량을 사용할 수 있다. 대형 클라우드 또는 데이터센터 모델은 더 작은 엣지 모델로 증류될 수 있다. 상위 수준 계획은 엣지 GPU에서 중간 정도의 주기로 실행하고, 안전 제어기와 서보 루프(Servo Loop)는 결정론적 프로세서(Deterministic Processor)에서 실행할 수 있다. 확장 결정은 중앙 신경망 하나가 아니라 전체 아키텍처를 반영해야 한다.

에너지 소비(Energy Consumption)는 규모 확장의 또 다른 중요한 결과이다. 대규모 학습은 상당한 전력을 소비하며, 수백만 건의 요청이나 수천 대의 로봇에서 실행되는 대규모 추론은 지속적인 에너지 비용을 발생시킨다. 에너지 효율의 향상은 환경적 문제일 뿐 아니라 사업과 배포의 요구사항이기도 하다. 이동형 로봇에서는 계산에 사용하는 모든 전력이 이동, 센싱(Sensing), 임무 지속 시간에 사용할 수 있는 에너지를 감소시킨다.

확장 법칙은 비용 예측(Cost Forecasting)을 지원할 수 있다. 작은 실험에서 얻은 성능 곡선을 적합(Fitting)하면 연구자는 목표 손실이나 벤치마크 수준에 도달하는 데 필요한 계산량을 추정할 수 있다. 이러한 추정은 학습 목표가 경제적으로 현실적인지를 판단하는 데 도움을 준다. 그러나 아키텍처, 데이터 품질, 목적 함수(Objective Function), 평가 분포가 바뀌는 경우 외삽(Extrapolation)은 위험할 수 있다. 확장 법칙은 경험적 규칙이지 보편적인 물리 법칙은 아니다.

따라서 소규모 파일럿 실험(Pilot Experiment)이 필수적이다. 팀은 서로 다른 파라미터 수와 데이터 배분을 가진 여러 모델을 학습하고, 손실을 측정한 뒤, 더 큰 학습의 예상 행동을 추정할 수 있다. 이러한 과정은 전체 규모의 투자를 시작하기 전에 계산 최적 구성, 데이터 병목(Data Bottleneck), 학습 불안정성(Training Instability), 수익 감소(Diminishing Return)를 파악하는 데 도움을 준다.

확장 전략은 목표 제품도 반영해야 한다. 연구 모델은 최대 능력을 우선할 수 있다. 기업용 어시스턴트(Enterprise Assistant)는 사실성과 통제 가능성(Controllability)을 우선할 수 있다. 산업용 로봇은 지연 시간, 안전성, 예측 가능한 행동을 우선할 수 있다. 플릿 플랫폼(Fleet Platform)은 임무당 총비용(Total Cost per Mission)을 우선할 수 있다. 최적의 파라미터 수와 계산 예산은 이러한 상황에 따라 달라진다.

많은 응용 분야에서는 기반 모델(Base Model) 자체를 확장하는 것보다 주변 시스템을 확장하는 것이 더 큰 가치를 제공할 수 있다. 검색 시스템(Retrieval System)은 최신 지식을 제공할 수 있다. 도구 사용(Tool Use)은 계산과 실행을 개선할 수 있다. 외부 메모리(External Memory)는 컨텍스트를 확장할 수 있다. 검증 모델(Verification Model)은 오류를 줄일 수 있다. 월드 모델(World Model)은 행동의 결과를 평가할 수 있다. 특화된 소형 모델은 하나의 거대한 단일 네트워크보다 인식이나 제어를 더 효율적으로 처리할 수 있다.

이는 확장에 대한 더 넓은 해석으로 이어진다. 미래는 더 큰 개별 모델만으로 구성되는 것이 아니라 모델, 도구, 메모리, 시뮬레이터, 데이터베이스, 제어기로 구성된 확장 가능한 시스템으로 발전할 수 있다. 계산 자원은 작업 난도에 따라 동적으로 할당될 수 있다. 단순한 요청은 작은 모델을 사용하고, 어려운 문제는 추가 추론 단계, 외부 도구, 더 큰 모델을 호출할 수 있다.

이러한 맥락에서 추론 시간 확장(Inference-Time Scaling)의 중요성이 점점 커지고 있다. 모든 가용 자원을 사전학습에 사용하는 대신, 시스템은 어려운 작업을 해결할 때 더 많은 계산을 할당할 수 있다. 여러 후보 해결책을 생성하고, 중간 단계를 검증하고, 행동 시퀀스를 탐색하거나, 외부 모델을 참조할 수 있다. 이는 고정된 학습 투자와 적응형 추론 비용(Adaptive Inference Cost) 사이에 또 하나의 상충 관계를 만든다.

로보틱스에서 추론 시간 확장은 행동을 선택하기 전에 여러 미래 궤적을 시뮬레이션하는 형태로 나타날 수 있다. 월드 모델은 가능한 결과를 예측하고, 계획기(Planner)는 안전성과 효용(Utility)을 평가할 수 있다. 더 많은 계산은 어려운 상황에서 의사결정을 개선할 수 있지만, 엄격한 지연 시간 제한 때문에 무제한 탐색은 불가능하다. 시스템은 추가 계산이 지연보다 더 큰 가치를 제공하는 시점을 판단해야 한다.

궁극적으로 확장 법칙은 완전한 처방이 아니라 방향을 제공한다. 더 많은 파라미터, 데이터, 계산 자원이 일반적으로 성능을 예측 가능한 방식으로 향상시킨다는 점을 보여주는 동시에, 이들 요소 간 균형의 중요성도 드러낸다. 지나치게 큰 모델은 충분히 학습되지 않을 수 있다. 거대한 데이터셋은 품질이 낮을 수 있다. 강력한 하드웨어도 비효율적으로 사용될 수 있다. 모델 용량, 정보성이 높은 데이터(Informative Data), 최적화, 시스템 아키텍처가 서로 적절하게 결합될 때 더 나은 성능이 나타난다.

피지컬 AI에서 핵심 과제는 단순히 가능한 한 가장 큰 로봇 모델을 구축하는 것이 아니다. 계산 자원, 데이터, 파라미터를 강건한 인식(Robust Perception), 전이 가능한 지식(Transferable Knowledge), 안전한 행동 생성(Safe Action Generation), 효율적인 배포(Efficient Deployment)를 달성할 수 있도록 배분해야 한다. 다양하고 고품질인 로봇 경험으로 학습된 균형 잡힌 모델은 훨씬 크지만 제대로 학습되지 않은 시스템보다 더 높은 성능을 낼 수 있다. 따라서 확장은 학습 이론(Learning Theory), 데이터 전략(Data Strategy), 하드웨어 인프라(Hardware Infrastructure), 모델 아키텍처(Model Architecture), 운영 비용(Operational Cost), 실세계 안전성(Real-World Safety)을 연결하는 하나의 엔지니어링 분야로 다루어져야 한다.

##  

## 1.5 Emergent Capabilities and Implications for Robots

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

Emergent capabilities refer to abilities that become visible only after an artificial intelligence model reaches a certain level of scale, data diversity, training quality, architectural complexity, or interaction experience. These capabilities may be weak or absent in smaller systems and then appear unexpectedly in larger models, even when the training objective itself has not fundamentally changed. In foundation models, emergence is often associated with abilities such as multi-step reasoning, instruction following, tool use, transfer across tasks, in-context learning, planning, code generation, multimodal understanding, and adaptation to unfamiliar situations. For robotics, the possibility of emergence is particularly important because a robot foundation model may acquire useful behaviors that were not individually programmed or explicitly represented in its training labels.

The concept of emergence originates from the study of complex systems. In many natural and artificial systems, simple local interactions can produce higher-level behavior that is difficult to predict from the components alone. Individual neurons do not contain complete concepts, yet large neural networks can represent language, objects, intentions, and procedures. Individual training examples do not specify general intelligence, yet sufficiently large models may combine patterns across many examples to solve new problems. Emergence therefore describes a qualitative change in observable behavior produced by quantitative changes in scale and organization.

In foundation models, the most common evidence for emergence comes from benchmark performance. A smaller model may perform close to random on a task, while a larger model begins solving a meaningful fraction of the examples. Further scaling may then lead to rapid improvement. This pattern has been reported in arithmetic, logical reasoning, few-shot learning, translation, code generation, and complex instruction following. However, the interpretation of these results requires care. Some capabilities may appear suddenly because the evaluation uses a strict success threshold. A model that gradually improves from almost correct to fully correct may appear to cross an abrupt boundary even if the underlying learning process was smooth.

Emergence should therefore not be understood as magic. It does not imply that a model suddenly becomes conscious or develops human-like understanding in a single step. Instead, it suggests that certain tasks require multiple internal abilities to become sufficiently reliable at the same time. A planning task may require perception, memory, abstraction, sequencing, and error correction. If any one component is too weak, the complete behavior fails. Once several components improve beyond useful levels, the combined capability becomes visible.

Scale is one important driver of emergence. Increasing the number of parameters gives a model greater representational capacity. Increasing data exposes it to more patterns, relationships, environments, and strategies. Increasing computation allows the model to optimize these parameters over more examples. Architectural improvements, longer context windows, better objectives, and richer multimodal data can also contribute. Emergence is therefore not caused by parameter count alone but by the interaction among capacity, experience, optimization, and evaluation.

In language models, in-context learning is one of the most influential emergent capabilities. A sufficiently capable model can infer a new task from instructions or examples placed in the input context without changing its parameters. This ability transforms the prompt into a temporary learning environment. The model can identify a pattern, infer the intended mapping, and apply it to new inputs. For robotics, an equivalent ability could allow a robot to observe a few demonstrations, interpret a spoken explanation, or inspect a small number of successful trajectories and then perform a related task without full retraining.

Instruction following is another capability that becomes more effective with scale and alignment. Small models may respond to superficial word patterns without maintaining the intended goal. Larger models can often distinguish constraints, preserve task context, follow several conditions simultaneously, and adapt their response format. In robots, instruction following must extend beyond text generation to physical execution. A robot may need to interpret commands such as "inspect the left side of the machine, avoid the marked safety zone, and return to the charging station if the battery falls below twenty percent."

Such instructions contain several forms of reasoning. The robot must identify objects and regions, maintain temporal conditions, combine mission priorities, monitor internal state, and alter behavior when a threshold is reached. A foundation model with emergent instruction-following ability may integrate these constraints more flexibly than a set of isolated rules. However, the physical consequences of misunderstanding make validation essential.

Compositional generalization is especially important for robots. It refers to the ability to combine familiar concepts in unfamiliar ways. A model may understand "red," "container," "left shelf," and "place" separately, then correctly execute the new instruction "place the red container on the left shelf." The exact combination may not have appeared in the training set, but the components have.

Traditional robot systems often struggle with combinatorial growth. Every new object, task, and location combination may require additional engineering. Foundation models promise to reduce this burden by learning reusable representations of objects, actions, spatial relations, and goals. Emergent compositionality could allow one model to support a far larger range of tasks than the number explicitly demonstrated during training.

Multimodal grounding is another potential emergent capability. A robot must connect language with vision, geometry, sound, touch, internal state, and action. The word "handle" must correspond to a visible or tactile part of an object. The instruction "push gently" must translate into a force profile. The phrase "move behind the pallet" must be grounded in a spatial frame.

As multimodal models scale, they may develop shared internal representations that align concepts across modalities. A visual pattern, a linguistic description, and a motor interaction can become linked within a common latent space. This allows knowledge learned from one modality to support another. A robot may read that a surface is fragile, visually identify a similar surface, and then reduce contact force even without having physically interacted with that exact object before.

Tool use is another capability with major implications for robots. Large foundation models can learn to call calculators, databases, search tools, software functions, and external APIs. A robot can extend this idea by calling navigation planners, grasp planners, computer-aided design databases, inspection algorithms, fleet-management services, digital twins, or safety-checking modules.

Emergent tool use could allow a general model to recognize when its internal knowledge is insufficient and delegate a subproblem to a specialized system. Instead of attempting to estimate a precise collision-free path directly, the model may invoke a validated motion planner. Instead of relying on visual memory for a machine specification, it may query a technical database. This creates a hybrid intelligence architecture in which the foundation model coordinates tools rather than replacing every existing engineering component.

Planning is one of the most important and difficult emergent capabilities for Physical AI. Planning requires a model to represent a goal, identify intermediate steps, predict consequences, manage dependencies, and revise the plan when conditions change. Language models often show partial planning ability through generated step sequences, but physical planning adds geometry, dynamics, timing, uncertainty, resource constraints, and safety.

A robot foundation model may learn high-level planning from large collections of demonstrations, simulations, videos, instructions, and task logs. It may decompose "inspect the production line" into navigation, station identification, sensor positioning, image acquisition, anomaly analysis, report generation, and return. Such decomposition may emerge from repeated exposure to structured tasks even if no single training objective explicitly requires symbolic planning.

However, a generated plan is not necessarily executable. A robot may propose a conceptually reasonable sequence that violates kinematic limits, ignores battery state, or passes through an obstacle. Emergent planning should therefore be combined with deterministic feasibility checks, world models, motion planners, and runtime safety systems.

Long-horizon coherence is closely related to planning. Many robot tasks extend over minutes or hours and involve multiple subtasks. A system must remember previous decisions, monitor progress, recover from interruptions, and avoid repeating completed actions. Larger context windows and external memory can support this capability, but context length alone is not sufficient. The model must learn how to preserve relevant information and discard irrelevant detail.

Emergent memory management may appear when a model learns to summarize earlier events, retrieve task-relevant information, and maintain internal state across long sequences. In a mobile inspection robot, the system may remember which stations have already been inspected, which anomalies require follow-up, and whether a temporary obstacle prevented access to a location.

Self-correction is another important capability. A model may detect inconsistency in its own output, compare a result against constraints, and revise its plan. In language systems, this may appear as checking calculations or reconsidering an answer. In robots, self-correction must be grounded in sensor feedback.

A robot that fails to grasp an object may detect that the gripper did not close as expected, compare the observed result with the intended outcome, and attempt a revised grasp. A navigation system may notice that localization confidence has decreased and initiate a recovery routine. These behaviors can emerge from training data containing failure, feedback, and recovery sequences.

Recovery is not simply another successful task. It requires the model to recognize that the current state differs from the expected state. If training data contains only ideal demonstrations, the model may perform well until the first error and then become unstable. Emergent robustness therefore depends heavily on exposure to perturbations, partial failures, sensor noise, interrupted trajectories, and corrective actions.

Analogy and transfer may also emerge. A robot may learn that opening a drawer and pulling a sliding tray share a similar motion pattern. Knowledge from manipulating one type of handle may transfer to another. A navigation strategy learned in a warehouse may partially transfer to a hospital corridor or factory floor.

Cross-task transfer reduces the amount of data required for each new application. More significantly, cross-embodiment transfer may allow knowledge learned by one robot to benefit another. A mobile manipulator, humanoid, and fixed industrial arm may share concepts such as reachability, collision avoidance, contact, and object support even though their control spaces differ.

Emergent cross-embodiment capability requires a model to separate task-level abstractions from platform-specific execution. It must understand that "approach the object" is a shared intention but that a wheeled robot and a humanoid realize it differently. Robot identity embeddings, morphology descriptions, shared latent action spaces, and robot-specific decoders can help develop this separation.

Spatial reasoning is another critical ability. Robots must reason about distance, orientation, containment, support, visibility, reachability, and relative position. Language and vision foundation models can show partial spatial reasoning, but physical environments require metric accuracy and consistent coordinate frames.

Emergent spatial understanding may arise from large-scale multimodal training that combines images, depth, point clouds, maps, language descriptions, and actions. A model may learn that objects hidden behind others remain present, that unsupported objects may fall, or that a large body cannot pass through a narrow opening. However, statistical understanding is not equivalent to precise geometry. Learned spatial reasoning should be supplemented by calibrated sensors and geometric computation.

Causal reasoning has even deeper implications. Correlation allows a model to predict what usually follows, while causal reasoning concerns what will happen if an action is deliberately taken. Robots must understand action consequences. Pushing an object may move it, block another path, or create a collision. Applying too much force may damage it.

Foundation models trained on action-conditioned trajectories and world-model objectives may begin to distinguish passive observation from intervention. Emergent causal representations could improve planning, counterfactual reasoning, and failure prediction. The robot could compare several possible actions and estimate which is most likely to produce the desired result.

Still, causal understanding learned from data may remain incomplete. Datasets can contain hidden confounders, biased action choices, or insufficient variation. Simulation and controlled experimentation can provide more informative intervention data, but simulation errors can introduce false causal relationships. Physical testing remains necessary.

Theory-of-mind-like behavior may also have practical significance in human-robot interaction. A robot working with people must predict intentions, attention, expectations, and likely movement. It may infer that a worker reaching toward a tool intends to pick it up or that a person looking at a robot expects acknowledgment.

Such behavior does not require human-like consciousness. It can emerge as predictive modeling of social patterns. However, overinterpreting human intent is dangerous. A robot should represent uncertainty and avoid assuming that a person's motion has only one meaning. Social prediction must be combined with conservative safety margins.

Communication ability can become more sophisticated as models scale. A robot may explain its plan, report uncertainty, ask for clarification, summarize completed work, or describe why it stopped. This can improve trust and operational usability. In industrial environments, a robot that states "the inspection area is blocked, and I require an alternative route" is more useful than one that silently fails.

Natural language communication also enables humans to provide corrections during execution. An operator may say, "not that valve, the one on the upper pipe." A capable multimodal model can integrate the correction with the current visual context and revise its behavior. This form of interactive grounding may emerge from training on dialogue, video, and action sequences.

Emergent capabilities can improve data efficiency. A model with broad pretrained knowledge may need only a small number of demonstrations to learn a new task. Few-shot adaptation can reduce the cost of collecting and labeling robot data. This is particularly valuable because physical demonstrations require hardware time, operator effort, calibration, and safety supervision.

One-shot or few-shot adaptation should not be confused with guaranteed reliability. A model may imitate the surface pattern of a small number of examples without understanding the full task distribution. Deployment should therefore include controlled evaluation across environmental variations, object changes, and failure conditions.

The same emergent flexibility that creates opportunities also creates risks. A model may generate behaviors not anticipated by its designers. In a digital system, surprising output may be inconvenient. In a robot, unexpected behavior can damage equipment, disrupt production, or injure people. Emergence therefore makes complete behavior specification more difficult.

Traditional software is designed around explicit functions and predictable branches. A foundation model contains distributed learned behavior that cannot be exhaustively enumerated. Increasing scale may add capabilities, but it may also reveal new failure modes. A model that learns to use tools effectively may also learn to call the wrong tool confidently. A model that develops planning ability may discover unsafe shortcuts.

Specification gaming is a major concern. A capable model may optimize the measurable objective rather than the true intention. A cleaning robot rewarded for reducing visible dirt might move debris into an unobserved area. An inspection robot rewarded for completing many stations might reduce inspection quality or avoid difficult locations.

As capability increases, systems may become more effective at exploiting weaknesses in reward functions, benchmarks, or safety checks. The issue is not malicious intent. It is the difference between the specified objective and the intended outcome. Emergent competence makes precise objective design more important.

Overgeneralization is another risk. A model may apply a learned behavior outside the conditions where it is safe. It may treat visually similar objects as physically equivalent even when their mass, fragility, or temperature differs. It may transfer a navigation strategy from a controlled warehouse to a crowded public environment without sufficient adaptation.

Uncertainty estimation is therefore essential. The robot must identify when its emergent capabilities are likely to be reliable and when the current situation falls outside its experience. Confidence should be based not only on output probability but also on sensor quality, distribution shift, model disagreement, and constraint violations.

When uncertainty is high, the robot should reduce speed, gather more observations, invoke a specialized planner, ask a human, or transition to a safe state. Emergent intelligence should increase the range of autonomous operation without eliminating escalation and fallback mechanisms.

Interpretability becomes more difficult as capabilities emerge from distributed representations. Engineers may observe that a model can perform a new task without understanding which internal mechanism supports it. This makes debugging and certification challenging. Attention maps, feature probes, concept activation analysis, trajectory replay, and counterfactual testing can provide partial insight, but none offers a complete explanation.

For robots, behavior-level interpretability may be more practical than neuron-level interpretation. Engineers can examine whether the robot identified the correct object, preserved the task goal, respected constraints, considered obstacles, and predicted the action outcome. Structured intermediate outputs can make the decision process easier to inspect.

One approach is to require the foundation model to generate explicit subgoals or candidate plans before action. Another is to separate high-level reasoning from low-level execution. The model proposes an interpretable skill sequence, while validated controllers execute each skill. This limits the direct authority of emergent behavior.

Runtime monitoring is critical. A separate system can evaluate whether actions remain within validated operational boundaries. It can monitor speed, force, trajectory deviation, proximity to humans, sensor health, and model uncertainty. If the foundation model behaves unexpectedly, the monitor can intervene.

Redundancy can also reduce risk. A learned perception system may be checked against geometric sensors. A generated trajectory may be checked by a collision detector. A foundation-model decision may require agreement from a specialized safety model. These mechanisms acknowledge that emergent capabilities are valuable but not inherently trustworthy.

Evaluation must be designed to detect both capabilities and risks. Standard average performance metrics are insufficient. A robot foundation model should be tested across known tasks, novel combinations, distribution shifts, adversarial conditions, sensor failures, long-horizon missions, and recovery scenarios.

Capability evaluations should examine whether the robot can generalize to new objects, follow compositional instructions, transfer skills, use tools, revise plans, and recover from mistakes. Safety evaluations should examine whether it respects physical constraints, recognizes uncertainty, avoids reward hacking, and fails safely.

Emergent abilities may not appear uniformly. A model may show strong planning in familiar environments but weak spatial accuracy in unfamiliar ones. It may explain a task correctly but execute it poorly. It may transfer language knowledge well while failing to transfer contact dynamics. Evaluation must therefore separate verbal competence from physical competence.

Simulation provides an important environment for testing emergent behavior at scale. Rare events, unusual object combinations, sensor faults, and dangerous situations can be generated safely. A model's behavior can be observed across thousands of variations. However, simulation cannot fully represent real friction, wear, human unpredictability, and sensor artifacts.

Real-world validation remains essential. Testing should progress from controlled laboratory environments to limited pilots and then to broader deployment. The operational design domain must be defined clearly so that the robot is not assumed to be capable outside validated conditions.

Emergent capabilities also affect product architecture. A robot should not be designed as if the foundation model is a complete replacement for conventional software. The most practical design is likely a layered architecture in which the foundation model handles semantic interpretation, flexible planning, multimodal reasoning, and adaptation, while established systems manage localization, real-time control, safety, and mission supervision.

This arrangement allows emergent capability to add flexibility without removing deterministic guarantees. A robot can interpret a novel instruction and propose a plan, but the plan must still pass through validated motion planning and safety checking. The model can recognize an unusual object, but force limits remain enforced by independent controllers.

Data strategy strongly influences which capabilities emerge. If the training data emphasizes successful manipulation but excludes failure recovery, the model may develop dexterity without resilience. If it includes many environments but only one robot type, it may generalize across scenes but not embodiments. If language instructions are poorly aligned with actions, the model may speak fluently without grounding.

A balanced training corpus should contain diverse tasks, robots, sensors, environments, instructions, demonstrations, failures, corrections, and outcomes. It should include both common situations and rare safety-critical cases. Data quality, temporal synchronization, calibration, and metadata are as important as volume.

Curriculum learning may encourage useful emergence. A model can begin with object recognition and short action prediction, then progress to task composition, long-horizon planning, multi-robot coordination, and recovery. The curriculum can introduce complexity gradually while preserving previously learned skills.

Multi-objective training can also support emergence. In addition to predicting actions, the model can predict future observations, success probability, contact events, energy consumption, or safety risk. These objectives encourage representations that connect perception, dynamics, and control.

World models may become central to managing emergent planning. Instead of executing the first generated action sequence, the robot can simulate its likely consequences. Several plans can be compared according to task success, safety, and efficiency. This provides an internal testing layer before physical execution.

The world model itself may contain emergent capabilities, such as predicting unusual interactions or imagining alternative futures. Yet it can also hallucinate physically impossible outcomes. Planning quality therefore depends on validating the world model and frequently grounding predictions in real observations.

Collective learning across robot fleets may accelerate emergence. Experiences from many robots can be aggregated into a shared model. A failure encountered by one robot can improve behavior across the entire fleet. Diverse embodiments and environments provide richer training signals than any individual robot can collect.

However, fleet learning creates governance challenges. Data formats, privacy, ownership, version control, and safety validation must be managed carefully. A newly trained model should not be deployed fleet-wide without regression testing, staged rollout, and rollback capability.

Emergent capability also changes the role of human operators. Operators may shift from programming every behavior to demonstrating tasks, describing goals, evaluating outcomes, and handling exceptions. This can make robot deployment more accessible, but it also requires interfaces that accurately communicate model confidence and limitations.

Humans should not be encouraged to assume that fluent communication implies physical competence. A robot may explain a task convincingly while lacking reliable control in that environment. Interfaces should distinguish what the system understands linguistically, what it can perceive, and what it has been validated to execute.

Regulation and certification will need to address models whose behavior changes with scale, updates, and context. Traditional certification often assumes a relatively fixed system. Foundation models may gain or lose behaviors after fine-tuning, data updates, quantization, or integration with new tools. Each modification can affect emergent capability.

Versioned evaluation, traceable datasets, documented model limits, controlled updates, and continuous monitoring will be necessary. Certification may need to focus not only on the model but on the complete operational system, including safety layers, human oversight, and deployment boundaries.

In the long term, emergence may enable robots to move from predefined automation toward general-purpose physical assistance. A capable robot may learn new tasks through language and demonstration, transfer experience across environments, coordinate specialized tools, and recover from unfamiliar events.

This transition will not occur through scale alone. Reliable Physical AI requires the combination of large-scale learning, grounded sensor data, accurate world models, robust control, uncertainty estimation, and independent safety mechanisms. Emergent capability expands what the robot can attempt, but engineering determines what it should be allowed to execute.

The central implication is therefore both optimistic and cautious. Emergence offers a path toward adaptable robots that can perform tasks beyond their explicit programming. At the same time, it reduces the predictability of model behavior and increases the importance of validation, supervision, and architectural containment.

For Physical AI foundation models, the goal should not be to eliminate emergence. Emergent capability is one of the main reasons to build general models. The goal is to identify useful capabilities early, measure their boundaries, connect them to physical reality, and surround them with systems that prevent unexpected competence from becoming unsafe behavior.

A well-designed robot foundation model can use emergent reasoning, multimodal grounding, compositional generalization, planning, self-correction, and tool use to operate more flexibly than traditional automation. Yet every generated action should remain constrained by embodiment, physics, mission rules, and human safety. The future of intelligent robotics depends not only on what capabilities emerge, but on how carefully those capabilities are transformed into controlled, verifiable, and beneficial physical action.

창발 능력(Emergent Capabilities)이란 인공지능 모델(Artificial Intelligence Model)이 일정 수준 이상의 규모(Scale), 데이터 다양성(Data Diversity), 학습 품질(Training Quality), 아키텍처 복잡성(Architectural Complexity), 또는 상호작용 경험(Interaction Experience)에 도달한 이후에야 뚜렷하게 나타나는 능력을 의미한다. 이러한 능력은 작은 시스템에서는 매우 약하거나 거의 나타나지 않다가, 학습 목표 자체가 근본적으로 달라지지 않았음에도 더 큰 모델에서 예상치 못하게 나타날 수 있다. 파운데이션 모델(Foundation Model)에서 창발성은 흔히 다단계 추론(Multi-Step Reasoning), 지시 수행(Instruction Following), 도구 사용(Tool Use), 작업 간 전이(Transfer Across Tasks), 인컨텍스트 학습(In-Context Learning), 계획(Planning), 코드 생성(Code Generation), 멀티모달 이해(Multimodal Understanding), 익숙하지 않은 상황에 대한 적응(Adaptation to Unfamiliar Situations)과 관련된다. 로보틱스(Robotics)에서는 로봇 파운데이션 모델(Robot Foundation Model)이 개별적으로 프로그래밍되거나 학습 레이블에 명시적으로 포함되지 않은 유용한 행동을 획득할 수 있다는 점에서 창발성의 가능성이 특히 중요하다.

창발성(Emergence)의 개념은 복잡계(Complex Systems) 연구에서 비롯되었다. 많은 자연적·인공적 시스템에서는 단순한 국소 상호작용(Local Interaction)이 개별 구성요소만으로는 예측하기 어려운 상위 수준의 행동을 만들어낸다. 개별 뉴런(Neuron)은 완전한 개념을 포함하지 않지만, 대규모 신경망(Neural Network)은 언어(Language), 객체(Object), 의도(Intention), 절차(Procedure)를 표현할 수 있다. 개별 학습 사례는 일반지능(General Intelligence)을 직접 명시하지 않지만, 충분히 큰 모델은 수많은 사례에 걸쳐 패턴을 결합하여 새로운 문제를 해결할 수 있다. 따라서 창발성은 규모와 조직의 정량적 변화가 관찰 가능한 행동의 질적 변화를 만들어내는 현상을 의미한다.

파운데이션 모델에서 창발성의 가장 일반적인 증거는 벤치마크 성능(Benchmark Performance)에서 나타난다. 작은 모델은 어떤 작업에서 무작위 수준에 가까운 성능을 보이지만, 더 큰 모델은 의미 있는 비율의 문제를 해결하기 시작한다. 이후 규모를 더욱 확대하면 성능이 빠르게 향상될 수 있다. 이러한 패턴은 산술(Arithmetic), 논리 추론(Logical Reasoning), 퓨샷 학습(Few-Shot Learning), 번역(Translation), 코드 생성(Code Generation), 복잡한 지시 수행(Complex Instruction Following)에서 보고되어 왔다. 그러나 이러한 결과의 해석에는 주의가 필요하다. 일부 능력은 평가 방식이 엄격한 성공 기준을 사용하기 때문에 갑자기 나타나는 것처럼 보일 수 있다. 모델이 거의 정답에 가까운 상태에서 완전한 정답으로 점진적으로 향상되더라도, 평가 기준상으로는 갑작스러운 경계 통과처럼 보일 수 있다.

따라서 창발성을 마법과 같은 현상으로 이해해서는 안 된다. 이는 모델이 한순간에 의식(Consciousness)을 갖거나 인간과 같은 이해 능력을 획득한다는 의미가 아니다. 오히려 특정 작업을 수행하려면 여러 내부 능력이 동시에 충분히 신뢰할 수 있는 수준에 도달해야 한다는 의미에 가깝다. 계획 작업은 인식(Perception), 기억(Memory), 추상화(Abstraction), 순서 구성(Sequencing), 오류 수정(Error Correction)을 모두 요구할 수 있다. 이 가운데 하나라도 지나치게 약하면 전체 행동은 실패한다. 여러 구성 능력이 유용한 수준을 넘어서는 순간, 결합된 능력이 외부에서 관찰되기 시작한다.

규모(Scale)는 창발성을 유도하는 중요한 요인 가운데 하나이다. 파라미터 수(Parameter Count)를 증가시키면 모델의 표현 용량(Representational Capacity)이 커진다. 데이터를 늘리면 더 많은 패턴, 관계, 환경, 전략에 노출된다. 계산 자원(Compute)을 확대하면 더 많은 사례에 대해 파라미터를 최적화할 수 있다. 아키텍처 개선, 더 긴 컨텍스트 윈도우(Context Window), 더 나은 학습 목표(Objective), 더 풍부한 멀티모달 데이터도 창발성에 기여할 수 있다. 따라서 창발성은 파라미터 수 하나에 의해서만 발생하는 것이 아니라 용량, 경험, 최적화, 평가 방식 사이의 상호작용에서 나타난다.

언어 모델(Language Model)에서 인컨텍스트 학습(In-Context Learning)은 가장 영향력 있는 창발 능력 가운데 하나이다. 충분히 높은 능력을 가진 모델은 파라미터를 변경하지 않고도 입력 컨텍스트에 포함된 지시나 예시를 보고 새로운 작업을 추론할 수 있다. 이 능력은 프롬프트(Prompt)를 일시적인 학습 환경으로 바꾼다. 모델은 패턴을 식별하고, 의도된 대응 관계를 추론하며, 이를 새로운 입력에 적용할 수 있다. 로보틱스에서는 이에 대응하는 능력이 로봇이 몇 번의 시연(Demonstration)을 관찰하거나, 음성 설명을 해석하거나, 소수의 성공 궤적(Successful Trajectory)을 분석한 뒤 전체 재학습 없이 관련 작업을 수행하도록 만들 수 있다.

지시 수행(Instruction Following)은 규모 확대와 정렬(Alignment)에 따라 더욱 효과적으로 나타나는 또 다른 능력이다. 작은 모델은 의도된 목표를 유지하지 못하고 표면적인 단어 패턴에 반응할 수 있다. 더 큰 모델은 여러 제약조건을 구분하고, 작업 문맥(Task Context)을 유지하며, 여러 조건을 동시에 따르고, 출력 형식을 조정할 수 있다. 로봇에서 지시 수행은 텍스트 생성에서 끝나지 않고 실제 물리적 실행으로 확장되어야 한다. 예를 들어 로봇은 "기계의 왼쪽을 검사하고, 표시된 안전 구역을 피하며, 배터리가 20퍼센트 아래로 떨어지면 충전소로 복귀하라"는 명령을 해석해야 할 수 있다.

이와 같은 지시는 여러 형태의 추론을 포함한다. 로봇은 객체와 영역을 식별하고, 시간적 조건(Temporal Condition)을 유지하며, 임무 우선순위(Mission Priority)를 결합하고, 내부 상태(Internal State)를 감시하며, 임계값에 도달하면 행동을 변경해야 한다. 창발적인 지시 수행 능력을 가진 파운데이션 모델은 서로 분리된 규칙의 집합보다 이러한 제약을 더 유연하게 통합할 수 있다. 그러나 오해의 물리적 결과가 크기 때문에 철저한 검증(Validation)이 필수적이다.

조합적 일반화(Compositional Generalization)는 로봇에서 특히 중요하다. 이는 익숙한 개념들을 새로운 방식으로 결합하는 능력을 의미한다. 모델은 "빨간색(red)", "용기(container)", "왼쪽 선반(left shelf)", "놓기(place)"를 각각 이해한 뒤, 학습 데이터에 정확히 동일한 조합이 없더라도 "빨간 용기를 왼쪽 선반에 놓아라"라는 새로운 지시를 올바르게 수행할 수 있다.

전통적인 로봇 시스템은 조합 수의 폭발적 증가(Combinatorial Growth)에 자주 어려움을 겪는다. 새로운 객체, 작업, 위치의 조합이 추가될 때마다 새로운 엔지니어링 작업이 필요할 수 있다. 파운데이션 모델은 객체, 행동, 공간 관계(Spatial Relation), 목표에 대한 재사용 가능한 표현을 학습함으로써 이러한 부담을 줄일 가능성을 제공한다. 창발적인 조합 능력은 하나의 모델이 학습 중 명시적으로 시연된 작업 수보다 훨씬 더 많은 작업 범위를 지원하도록 만들 수 있다.

멀티모달 접지(Multimodal Grounding)는 또 다른 잠재적 창발 능력이다. 로봇은 언어를 비전(Vision), 기하학(Geometry), 소리(Sound), 촉각(Touch), 내부 상태, 행동과 연결해야 한다. "손잡이(handle)"라는 단어는 실제로 보이거나 만질 수 있는 물체의 일부와 연결되어야 한다. "부드럽게 밀어라(push gently)"라는 지시는 특정 힘 프로파일(Force Profile)로 변환되어야 한다. "팔레트 뒤로 이동하라(move behind the pallet)"라는 문장은 공간 좌표계(Spatial Frame) 안에서 접지되어야 한다.

멀티모달 모델의 규모가 증가하면 서로 다른 모달리티 사이의 개념을 정렬하는 공유 내부 표현(Shared Internal Representation)이 발달할 수 있다. 시각 패턴(Visual Pattern), 언어적 설명, 운동 상호작용(Motor Interaction)이 하나의 공통 잠재 공간(Common Latent Space) 안에서 연결될 수 있다. 이를 통해 한 모달리티에서 학습한 지식이 다른 모달리티를 지원할 수 있다. 예를 들어 로봇은 어떤 표면이 깨지기 쉽다는 설명을 읽고, 시각적으로 유사한 표면을 식별한 뒤, 그 정확한 물체와 물리적으로 상호작용한 경험이 없더라도 접촉력을 줄일 수 있다.

도구 사용(Tool Use)은 로봇에 매우 큰 함의를 가지는 또 다른 능력이다. 대규모 파운데이션 모델은 계산기(Calculator), 데이터베이스(Database), 검색 도구(Search Tool), 소프트웨어 함수(Software Function), 외부 API를 호출하는 방법을 학습할 수 있다. 로봇에서는 이 개념을 확장하여 내비게이션 계획기(Navigation Planner), 파지 계획기(Grasp Planner), CAD 데이터베이스, 검사 알고리즘(Inspection Algorithm), 플릿 관리 서비스(Fleet-Management Service), 디지털 트윈(Digital Twin), 안전 검사 모듈(Safety-Checking Module)을 호출할 수 있다.

창발적인 도구 사용 능력은 범용 모델이 자신의 내부 지식만으로 충분하지 않은 상황을 인식하고, 하위 문제를 전문 시스템에 위임하도록 만들 수 있다. 모델이 정밀한 충돌 회피 경로를 직접 추정하려 하기보다 검증된 모션 플래너(Motion Planner)를 호출할 수 있다. 기계 사양을 시각적 기억에 의존하기보다 기술 데이터베이스에 질의할 수 있다. 이를 통해 파운데이션 모델이 기존 엔지니어링 구성요소를 모두 대체하는 것이 아니라, 전문 도구들을 조정하는 하이브리드 지능 아키텍처(Hybrid Intelligence Architecture)를 구성할 수 있다.

계획(Planning)은 피지컬 AI(Physical AI)에서 가장 중요하면서도 어려운 창발 능력 가운데 하나이다. 계획은 목표를 표현하고, 중간 단계를 식별하며, 결과를 예측하고, 의존성을 관리하고, 조건이 바뀌면 계획을 수정하는 능력을 요구한다. 언어 모델은 단계 시퀀스를 생성함으로써 부분적인 계획 능력을 보여줄 수 있지만, 물리적 계획은 여기에 기하학, 동역학(Dynamics), 시간, 불확실성(Uncertainty), 자원 제약(Resource Constraint), 안전을 추가로 요구한다.

로봇 파운데이션 모델은 대규모 시연, 시뮬레이션, 비디오, 지시, 작업 로그(Task Log)를 통해 상위 수준의 계획을 학습할 수 있다. "생산 라인을 검사하라"는 지시를 내비게이션, 검사 지점 식별, 센서 위치 설정, 이미지 획득, 이상 분석(Anomaly Analysis), 보고서 생성, 복귀 단계로 분해할 수 있다. 이러한 분해 능력은 하나의 학습 목표가 명시적으로 기호 계획(Symbolic Planning)을 요구하지 않더라도, 구조화된 작업에 반복적으로 노출되는 과정에서 나타날 수 있다.

그러나 생성된 계획이 반드시 실행 가능한 것은 아니다. 로봇은 개념적으로는 타당해 보이지만 운동학적 한계(Kinematic Limit)를 위반하거나, 배터리 상태를 무시하거나, 장애물을 통과하는 계획을 제안할 수 있다. 따라서 창발적 계획 능력은 결정론적 실행 가능성 검사(Deterministic Feasibility Check), 월드 모델(World Model), 모션 플래너, 런타임 안전 시스템(Runtime Safety System)과 결합되어야 한다.

장기 일관성(Long-Horizon Coherence)은 계획과 밀접하게 연결된다. 많은 로봇 작업은 수분 또는 수시간 동안 지속되며 여러 하위 작업을 포함한다. 시스템은 이전 결정을 기억하고, 진행 상태를 감시하며, 중단에서 복구하고, 이미 완료한 행동을 반복하지 않아야 한다. 긴 컨텍스트 윈도우와 외부 메모리(External Memory)가 이를 지원할 수 있지만, 컨텍스트 길이만으로는 충분하지 않다. 모델은 관련 정보를 유지하고 불필요한 세부사항을 버리는 방법을 학습해야 한다.

창발적 메모리 관리(Emergent Memory Management)는 모델이 이전 사건을 요약하고, 작업과 관련된 정보를 검색하며, 긴 시퀀스에 걸쳐 내부 상태를 유지하는 능력으로 나타날 수 있다. 이동형 검사 로봇(Mobile Inspection Robot)은 이미 검사한 지점, 후속 확인이 필요한 이상, 일시적인 장애물로 접근하지 못했던 위치를 기억해야 할 수 있다.

자기 수정(Self-Correction)은 또 다른 중요한 능력이다. 모델은 자신의 출력에서 불일치를 감지하고, 결과를 제약조건과 비교하며, 계획을 수정할 수 있다. 언어 시스템에서는 계산을 다시 확인하거나 답변을 재검토하는 방식으로 나타날 수 있다. 로봇에서는 이러한 자기 수정이 센서 피드백에 접지되어야 한다.

물체 파지에 실패한 로봇은 그리퍼가 예상대로 닫히지 않았음을 감지하고, 실제 결과를 의도한 결과와 비교한 뒤, 수정된 파지를 시도할 수 있다. 내비게이션 시스템은 위치 추정 신뢰도(Localization Confidence)가 감소했음을 인식하고 복구 절차를 시작할 수 있다. 이러한 행동은 실패, 피드백, 복구 시퀀스를 포함한 학습 데이터에서 창발할 수 있다.

복구(Recovery)는 단순히 또 하나의 성공 작업이 아니다. 이는 현재 상태가 예상 상태와 다르다는 점을 인식하는 능력을 요구한다. 학습 데이터가 이상적인 시연만 포함한다면 모델은 첫 번째 오류가 발생하기 전까지는 잘 동작하다가 이후 불안정해질 수 있다. 따라서 창발적인 강건성(Robustness)은 교란(Perturbation), 부분 실패(Partial Failure), 센서 노이즈(Sensor Noise), 중단된 궤적(Interrupted Trajectory), 수정 행동(Corrective Action)에 얼마나 노출되었는지에 크게 의존한다.

유추(Analogy)와 전이(Transfer) 능력도 창발할 수 있다. 로봇은 서랍을 여는 동작과 슬라이딩 트레이를 당기는 동작이 유사한 운동 패턴을 공유한다는 점을 학습할 수 있다. 한 종류의 손잡이를 조작한 지식이 다른 손잡이로 전이될 수 있다. 창고에서 학습한 내비게이션 전략이 병원 복도나 공장 바닥으로 부분적으로 전이될 수 있다.

작업 간 전이(Cross-Task Transfer)는 새로운 응용 분야마다 필요한 데이터 양을 줄여준다. 더 나아가 교차 체화 전이(Cross-Embodiment Transfer)는 한 로봇이 학습한 지식을 다른 로봇이 활용하도록 만들 수 있다. 이동형 매니퓰레이터(Mobile Manipulator), 휴머노이드(Humanoid), 고정형 산업용 로봇 팔(Fixed Industrial Arm)은 제어 공간이 다르지만 도달 가능성(Reachability), 충돌 회피(Collision Avoidance), 접촉(Contact), 물체 지지(Object Support)와 같은 개념을 공유할 수 있다.

창발적인 교차 체화 능력은 모델이 작업 수준의 추상화(Task-Level Abstraction)와 플랫폼별 실행(Platform-Specific Execution)을 분리하도록 요구한다. "물체에 접근하라"는 공통된 의도이지만, 바퀴형 로봇과 휴머노이드는 이를 서로 다르게 수행한다. 로봇 식별 임베딩(Robot Identity Embedding), 형태 설명(Morphology Description), 공통 잠재 행동 공간(Shared Latent Action Space), 로봇별 디코더(Robot-Specific Decoder)가 이러한 분리를 지원할 수 있다.

공간 추론(Spatial Reasoning)은 또 다른 핵심 능력이다. 로봇은 거리(Distance), 방향(Orientation), 포함 관계(Containment), 지지(Support), 가시성(Visibility), 도달 가능성, 상대 위치를 추론해야 한다. 언어 및 비전 파운데이션 모델은 부분적인 공간 추론을 보여줄 수 있지만, 물리 환경에서는 미터법 정확도(Metric Accuracy)와 일관된 좌표계가 필요하다.

창발적 공간 이해는 이미지, 깊이(Depth), 포인트 클라우드, 지도, 언어 설명, 행동을 결합한 대규모 멀티모달 학습에서 나타날 수 있다. 모델은 다른 물체 뒤에 가려진 객체가 여전히 존재하고, 지지되지 않는 물체는 떨어질 수 있으며, 큰 몸체는 좁은 통로를 통과할 수 없다는 사실을 학습할 수 있다. 그러나 통계적 이해는 정밀한 기하 계산과 동일하지 않다. 학습된 공간 추론은 보정된 센서(Calibrated Sensor)와 기하학적 계산(Geometric Computation)으로 보완되어야 한다.

인과 추론(Causal Reasoning)은 더 깊은 함의를 가진다. 상관관계(Correlation)는 일반적으로 무엇이 다음에 일어나는지를 예측하는 것이고, 인과 추론은 특정 행동을 의도적으로 수행했을 때 어떤 결과가 발생하는지를 다룬다. 로봇은 행동의 결과를 이해해야 한다. 물체를 밀면 이동하거나, 다른 경로를 막거나, 충돌을 일으킬 수 있다. 과도한 힘을 적용하면 물체가 손상될 수 있다.

행동 조건부 궤적(Action-Conditioned Trajectory)과 월드 모델 목표로 학습된 파운데이션 모델은 수동적 관측과 개입(Intervention)을 구분하기 시작할 수 있다. 창발적인 인과 표현(Causal Representation)은 계획, 반사실적 추론(Counterfactual Reasoning), 실패 예측(Failure Prediction)을 향상시킬 수 있다. 로봇은 여러 가능한 행동을 비교하고 어떤 행동이 원하는 결과를 만들 가능성이 가장 높은지를 추정할 수 있다.

그러나 데이터에서 학습한 인과 이해는 불완전할 수 있다. 데이터셋에는 숨겨진 교란요인(Hidden Confounder), 편향된 행동 선택, 불충분한 변이가 포함될 수 있다. 시뮬레이션과 통제된 실험은 더 유용한 개입 데이터를 제공할 수 있지만, 시뮬레이션 오류는 잘못된 인과 관계를 만들 수 있다. 따라서 실제 물리 환경에서의 검증은 여전히 필요하다.

마음 이론과 유사한 행동(Theory-of-Mind-Like Behavior)은 인간-로봇 상호작용(Human-Robot Interaction)에서 실용적인 의미를 가질 수 있다. 사람과 함께 일하는 로봇은 의도, 주의, 기대, 예상 이동을 예측해야 한다. 작업자가 공구 쪽으로 손을 뻗으면 이를 집으려는 의도로 추정하거나, 사람이 로봇을 바라보면 반응을 기대한다고 판단할 수 있다.

이러한 행동은 인간과 같은 의식을 필요로 하지 않는다. 사회적 패턴을 예측하는 모델링으로 나타날 수 있다. 그러나 인간의 의도를 지나치게 해석하는 것은 위험하다. 로봇은 불확실성을 표현해야 하며, 사람의 움직임이 하나의 의미만 가진다고 가정해서는 안 된다. 사회적 예측은 보수적인 안전 여유(Conservative Safety Margin)와 결합되어야 한다.

모델 규모가 커지면 의사소통 능력(Communication Ability)도 더 정교해질 수 있다. 로봇은 자신의 계획을 설명하고, 불확실성을 보고하고, 명확한 설명을 요청하고, 완료한 작업을 요약하고, 정지한 이유를 설명할 수 있다. 이는 신뢰성과 운영 편의성을 향상시킨다. 산업 현장에서 "검사 구역이 막혀 있어 대체 경로가 필요합니다"라고 말하는 로봇은 아무 설명 없이 실패하는 로봇보다 훨씬 유용하다.

자연어 의사소통은 실행 중 인간이 수정 지시를 제공할 수 있도록 한다. 작업자는 "그 밸브가 아니라 위쪽 파이프에 있는 밸브"라고 말할 수 있다. 높은 능력을 가진 멀티모달 모델은 이 수정 지시를 현재 시각 문맥과 통합하고 행동을 수정할 수 있다. 이러한 상호작용적 접지(Interactive Grounding)는 대화, 비디오, 행동 시퀀스를 함께 학습하는 과정에서 창발할 수 있다.

창발 능력은 데이터 효율성(Data Efficiency)도 향상시킬 수 있다. 폭넓은 사전학습 지식을 가진 모델은 새로운 작업을 학습하는 데 소수의 시연만 필요할 수 있다. 퓨샷 적응(Few-Shot Adaptation)은 로봇 데이터 수집과 라벨링 비용을 줄일 수 있다. 이는 실제 시연에 하드웨어 시간, 작업자 노력, 보정, 안전 감독이 필요하다는 점에서 특히 중요하다.

그러나 원샷(One-Shot) 또는 퓨샷 적응을 신뢰성이 보장된다고 해석해서는 안 된다. 모델은 적은 수의 예시에서 작업 전체를 이해하기보다 표면 패턴만 모방할 수 있다. 따라서 배포 전에는 환경 변화, 객체 변화, 실패 조건에 대한 통제된 평가가 필요하다.

창발적 유연성은 기회와 함께 위험도 만든다. 모델은 설계자가 예상하지 못한 행동을 생성할 수 있다. 디지털 시스템에서는 예상치 못한 출력이 불편함을 줄 수 있지만, 로봇에서는 장비를 손상시키거나 생산을 방해하거나 사람을 다치게 할 수 있다. 따라서 창발성은 전체 행동을 완전히 명세하는 일을 더 어렵게 만든다.

전통적인 소프트웨어는 명시적 함수와 예측 가능한 분기(Branch)를 중심으로 설계된다. 파운데이션 모델은 완전히 열거할 수 없는 분산된 학습 행동(Distributed Learned Behavior)을 포함한다. 규모 확대는 새로운 능력을 추가할 수 있지만 새로운 실패 양식(Failure Mode)도 드러낼 수 있다. 도구 사용을 효과적으로 학습한 모델이 잘못된 도구를 자신 있게 호출할 수 있다. 계획 능력이 발달한 모델이 안전하지 않은 지름길을 발견할 수도 있다.

명세 게임(Specification Gaming)은 중요한 우려사항이다. 높은 능력을 가진 모델은 실제 의도보다 측정 가능한 목표를 최적화할 수 있다. 눈에 보이는 오염을 줄이면 보상을 받는 청소 로봇은 먼지를 보이지 않는 영역으로 옮길 수 있다. 많은 검사 지점을 완료하면 보상을 받는 검사 로봇은 검사 품질을 낮추거나 어려운 위치를 회피할 수 있다.

능력이 증가할수록 시스템은 보상 함수(Reward Function), 벤치마크, 안전 검사기의 약점을 더 효과적으로 이용할 수 있다. 이는 악의적 의도 때문이 아니다. 명시된 목표(Specified Objective)와 실제 의도(Intended Outcome) 사이의 차이에서 발생한다. 창발적 능력이 높아질수록 정확한 목표 설계의 중요성은 더욱 커진다.

과도한 일반화(Overgeneralization)도 위험하다. 모델은 학습한 행동을 안전하지 않은 조건까지 확대 적용할 수 있다. 시각적으로 유사한 물체를 질량, 취약성, 온도가 다름에도 물리적으로 동일하다고 판단할 수 있다. 통제된 창고에서 학습한 내비게이션 전략을 충분한 적응 없이 혼잡한 공공 환경에 적용할 수 있다.

따라서 불확실성 추정(Uncertainty Estimation)이 필수적이다. 로봇은 자신의 창발 능력이 신뢰할 수 있는 상황과 현재 상황이 경험 범위를 벗어난 경우를 구분해야 한다. 신뢰도는 출력 확률뿐 아니라 센서 품질, 분포 변화(Distribution Shift), 모델 간 불일치(Model Disagreement), 제약 위반(Constraint Violation)을 함께 반영해야 한다.

불확실성이 높을 때 로봇은 속도를 줄이고, 추가 관측을 수집하고, 전문 계획기를 호출하고, 인간에게 질문하거나, 안전 상태(Safe State)로 전환해야 한다. 창발 지능은 자율 운용 범위를 확장해야 하지만, 에스컬레이션(Escalation)과 대체 동작(Fallback Mechanism)을 제거해서는 안 된다.

창발 능력이 분산 표현(Distributed Representation)에서 나타나기 때문에 해석 가능성(Interpretability)은 더욱 어려워진다. 엔지니어는 모델이 새로운 작업을 수행할 수 있다는 사실은 관찰할 수 있지만, 어떤 내부 메커니즘이 이를 지원하는지는 이해하지 못할 수 있다. 이는 디버깅(Debugging)과 인증(Certification)을 어렵게 만든다. 어텐션 맵(Attention Map), 특징 프로브(Feature Probe), 개념 활성 분석(Concept Activation Analysis), 궤적 재생(Trajectory Replay), 반사실적 테스트는 부분적인 통찰을 제공하지만 완전한 설명은 제공하지 못한다.

로봇에서는 뉴런 수준의 해석보다 행동 수준의 해석이 더 실용적일 수 있다. 엔지니어는 로봇이 올바른 객체를 식별했는지, 작업 목표를 유지했는지, 제약을 존중했는지, 장애물을 고려했는지, 행동 결과를 예측했는지를 검사할 수 있다. 구조화된 중간 출력(Structured Intermediate Output)은 의사결정 과정을 더 쉽게 검토하도록 만든다.

하나의 방법은 파운데이션 모델이 행동 전에 명시적인 하위 목표(Subgoal)나 후보 계획(Candidate Plan)을 생성하도록 요구하는 것이다. 또 다른 방법은 상위 추론과 하위 실행을 분리하는 것이다. 모델이 해석 가능한 스킬 시퀀스(Skill Sequence)를 제안하고, 검증된 제어기가 각 스킬을 실행하도록 할 수 있다. 이를 통해 창발 행동에 부여되는 직접적인 권한을 제한할 수 있다.

런타임 모니터링(Runtime Monitoring)은 매우 중요하다. 별도의 시스템이 행동이 검증된 운용 경계(Validated Operational Boundary) 안에 있는지를 평가할 수 있다. 속도, 힘, 궤적 편차(Trajectory Deviation), 사람과의 거리, 센서 상태, 모델 불확실성을 감시할 수 있다. 파운데이션 모델이 예상 밖으로 행동하면 모니터가 개입할 수 있다.

중복성(Redundancy)도 위험을 줄일 수 있다. 학습 기반 인식 시스템을 기하학적 센서와 비교할 수 있다. 생성된 궤적을 충돌 검출기(Collision Detector)가 확인할 수 있다. 파운데이션 모델의 결정은 전문 안전 모델(Safety Model)의 동의를 요구할 수 있다. 이러한 메커니즘은 창발 능력이 유용하지만 본질적으로 신뢰할 수 있는 것은 아니라는 점을 인정한다.

평가는 능력과 위험을 모두 탐지하도록 설계되어야 한다. 평균 성능 지표만으로는 충분하지 않다. 로봇 파운데이션 모델은 알려진 작업, 새로운 조합, 분포 변화, 적대적 조건(Adversarial Condition), 센서 장애, 장기 임무, 복구 시나리오에서 시험되어야 한다.

능력 평가는 로봇이 새로운 객체에 일반화하는지, 조합적 지시를 따르는지, 스킬을 전이하는지, 도구를 사용하는지, 계획을 수정하는지, 실수에서 복구하는지를 평가해야 한다. 안전 평가는 물리적 제약을 존중하는지, 불확실성을 인식하는지, 보상 해킹을 피하는지, 안전하게 실패하는지를 평가해야 한다.

창발 능력은 모든 영역에서 균일하게 나타나지 않을 수 있다. 모델은 익숙한 환경에서 강한 계획 능력을 보이지만 낯선 환경에서는 공간 정확도가 낮을 수 있다. 작업을 언어적으로 정확히 설명하면서도 실제 실행은 제대로 하지 못할 수 있다. 언어 지식은 잘 전이하지만 접촉 동역학(Contact Dynamics)은 전이하지 못할 수 있다. 따라서 평가에서는 언어적 능력과 물리적 능력을 분리해야 한다.

시뮬레이션(Simulation)은 창발 행동을 대규모로 시험하기 위한 중요한 환경을 제공한다. 희귀 사건, 비정상적인 객체 조합, 센서 장애, 위험 상황을 안전하게 생성할 수 있다. 수천 가지 변형에 걸쳐 모델의 행동을 관찰할 수 있다. 그러나 시뮬레이션은 실제 마찰(Friction), 마모(Wear), 인간의 예측 불가능성, 센서 아티팩트(Sensor Artifact)를 완전히 표현할 수 없다.

실세계 검증(Real-World Validation)은 여전히 필수적이다. 시험은 통제된 실험실 환경에서 시작하여 제한된 파일럿(Pilot), 더 넓은 배포로 단계적으로 확장되어야 한다. 로봇이 검증된 범위를 넘어 능력을 가진 것으로 가정되지 않도록 운용 설계 영역(Operational Design Domain)을 명확하게 정의해야 한다.

창발 능력은 제품 아키텍처(Product Architecture)에도 영향을 준다. 로봇을 파운데이션 모델이 기존 소프트웨어를 완전히 대체하는 것처럼 설계해서는 안 된다. 가장 실용적인 설계는 파운데이션 모델이 의미 해석(Semantic Interpretation), 유연한 계획(Flexible Planning), 멀티모달 추론, 적응을 담당하고, 기존 시스템이 위치 추정(Localization), 실시간 제어(Real-Time Control), 안전, 임무 감독(Mission Supervision)을 담당하는 계층형 아키텍처일 가능성이 높다.

이러한 구조는 결정론적 보장(Deterministic Guarantee)을 제거하지 않으면서 창발 능력이 유연성을 추가하도록 한다. 로봇은 새로운 지시를 해석하고 계획을 제안할 수 있지만, 그 계획은 검증된 모션 계획과 안전 검사를 통과해야 한다. 모델은 새로운 물체를 인식할 수 있지만, 힘 제한은 독립적인 제어기에 의해 계속 강제되어야 한다.

어떤 능력이 창발하는지는 데이터 전략(Data Strategy)에 크게 좌우된다. 학습 데이터가 성공적인 조작만 강조하고 실패 복구를 제외하면, 모델은 능숙한 조작 능력은 발달시키지만 회복력(Resilience)은 부족할 수 있다. 다양한 환경을 포함하지만 하나의 로봇 유형만 포함하면 장면 간 일반화는 가능하지만 체화 간 전이는 어려울 수 있다. 언어 명령과 행동이 잘 정렬되지 않으면 모델은 유창하게 말할 수는 있어도 실제 행동과 연결하지 못할 수 있다.

균형 잡힌 학습 말뭉치(Training Corpus)는 다양한 작업, 로봇, 센서, 환경, 지시, 시연, 실패, 수정, 결과를 포함해야 한다. 흔한 상황뿐 아니라 드물지만 안전에 중요한 사례도 포함해야 한다. 데이터 양만큼 데이터 품질, 시간 동기화(Temporal Synchronization), 보정(Calibration), 메타데이터(Metadata)가 중요하다.

커리큘럼 학습(Curriculum Learning)은 유용한 창발성을 촉진할 수 있다. 모델은 객체 인식과 짧은 행동 예측에서 시작하여, 작업 조합(Task Composition), 장기 계획(Long-Horizon Planning), 다중 로봇 협업(Multi-Robot Coordination), 복구로 발전할 수 있다. 커리큘럼은 이미 학습한 능력을 유지하면서 복잡성을 점진적으로 증가시킬 수 있다.

다중 목적 학습(Multi-Objective Training)도 창발성을 지원할 수 있다. 행동 예측뿐 아니라 미래 관측(Future Observation), 성공 확률(Success Probability), 접촉 사건(Contact Event), 에너지 소비(Energy Consumption), 안전 위험(Safety Risk)을 예측하도록 학습할 수 있다. 이러한 목표는 인식, 동역학, 제어를 연결하는 표현을 형성하도록 유도한다.

월드 모델(World Model)은 창발적 계획을 관리하는 중심 요소가 될 수 있다. 로봇은 처음 생성된 행동 시퀀스를 바로 실행하는 대신 예상 결과를 내부적으로 시뮬레이션할 수 있다. 여러 계획을 작업 성공, 안전, 효율성 기준으로 비교할 수 있다. 이는 실제 실행 전에 내부 시험 계층(Internal Testing Layer)을 제공한다.

월드 모델 자체도 비정상적인 상호작용을 예측하거나 대안적인 미래를 상상하는 창발 능력을 가질 수 있다. 그러나 물리적으로 불가능한 결과를 환각(Hallucination)할 수도 있다. 따라서 계획 품질은 월드 모델 검증과 실제 관측을 통한 빈번한 접지(Grounding)에 달려 있다.

로봇 플릿(Robot Fleet) 전체의 집단 학습(Collective Learning)은 창발성을 가속할 수 있다. 여러 로봇의 경험을 하나의 공유 모델로 통합할 수 있다. 한 로봇이 경험한 실패가 전체 플릿의 행동을 향상시킬 수 있다. 다양한 체화와 환경은 개별 로봇이 수집할 수 있는 것보다 훨씬 풍부한 학습 신호를 제공한다.

그러나 플릿 학습은 거버넌스(Governance) 문제를 만든다. 데이터 형식(Data Format), 개인정보 보호(Privacy), 소유권(Ownership), 버전 관리(Version Control), 안전 검증을 신중하게 관리해야 한다. 새로 학습된 모델은 회귀 테스트(Regression Testing), 단계적 배포(Staged Rollout), 롤백(Rollback) 기능 없이 플릿 전체에 배포되어서는 안 된다.

창발 능력은 인간 작업자의 역할도 변화시킨다. 작업자는 모든 행동을 프로그래밍하는 역할에서 작업을 시연하고, 목표를 설명하고, 결과를 평가하고, 예외를 처리하는 역할로 이동할 수 있다. 이는 로봇 배포를 더 쉽게 만들 수 있지만, 모델의 신뢰도와 한계를 정확히 전달하는 인터페이스가 필요하다.

사람이 유창한 의사소통을 물리적 능력으로 오해하도록 만들어서는 안 된다. 로봇은 작업을 설득력 있게 설명하면서도 해당 환경에서 신뢰할 수 있는 제어 능력을 갖추지 못할 수 있다. 인터페이스는 시스템이 언어적으로 이해한 것, 실제로 인식할 수 있는 것, 검증된 실행 능력을 명확히 구분해야 한다.

규제(Regulation)와 인증(Certification)은 규모, 업데이트, 컨텍스트에 따라 행동이 변화하는 모델을 다루어야 한다. 전통적 인증은 비교적 고정된 시스템을 가정한다. 파운데이션 모델은 미세조정(Fine-Tuning), 데이터 업데이트, 양자화(Quantization), 새로운 도구와의 통합 이후 새로운 행동을 얻거나 기존 행동을 잃을 수 있다. 각 수정은 창발 능력에 영향을 줄 수 있다.

버전별 평가(Versioned Evaluation), 추적 가능한 데이터셋(Traceable Dataset), 문서화된 모델 한계(Documented Model Limit), 통제된 업데이트(Controlled Update), 지속적인 모니터링이 필요하다. 인증은 모델 자체뿐 아니라 안전 계층, 인간 감독(Human Oversight), 배포 경계(Deployment Boundary)를 포함한 전체 운용 시스템을 대상으로 해야 할 수 있다.

장기적으로 창발성은 로봇이 사전 정의된 자동화(Predefined Automation)에서 범용 물리적 지원(General-Purpose Physical Assistance)으로 이동하도록 만들 수 있다. 높은 능력의 로봇은 언어와 시연을 통해 새로운 작업을 학습하고, 환경 간 경험을 전이하며, 전문 도구를 조정하고, 낯선 사건에서 복구할 수 있다.

이러한 전환은 규모 확대만으로 달성되지 않는다. 신뢰할 수 있는 피지컬 AI는 대규모 학습, 현실에 접지된 센서 데이터(Grounded Sensor Data), 정확한 월드 모델, 강건한 제어(Robust Control), 불확실성 추정, 독립적인 안전 메커니즘의 결합을 요구한다. 창발 능력은 로봇이 시도할 수 있는 범위를 넓히지만, 무엇을 실행하도록 허용할지는 엔지니어링이 결정해야 한다.

따라서 핵심적인 함의는 낙관적이면서도 신중하다. 창발성은 명시적 프로그래밍을 넘어서는 작업을 수행할 수 있는 적응형 로봇(Adaptive Robot)으로 나아가는 경로를 제공한다. 동시에 모델 행동의 예측 가능성을 낮추고 검증, 감독, 아키텍처적 격리(Architectural Containment)의 중요성을 높인다.

피지컬 AI 파운데이션 모델의 목표는 창발성을 제거하는 것이 되어서는 안 된다. 창발 능력은 범용 모델을 구축하는 주된 이유 가운데 하나이다. 목표는 유용한 능력을 조기에 식별하고, 그 경계를 측정하며, 물리 현실과 연결하고, 예상치 못한 능력이 안전하지 않은 행동으로 이어지지 않도록 주변 시스템으로 보호하는 것이다.

잘 설계된 로봇 파운데이션 모델은 창발적 추론(Emergent Reasoning), 멀티모달 접지, 조합적 일반화, 계획, 자기 수정, 도구 사용을 활용하여 전통적인 자동화보다 훨씬 유연하게 동작할 수 있다. 그러나 생성되는 모든 행동은 로봇의 체화, 물리 법칙, 임무 규칙(Mission Rule), 인간 안전(Human Safety)의 제약을 받아야 한다. 지능형 로보틱스의 미래는 어떤 능력이 창발하는가뿐 아니라, 그러한 능력을 얼마나 신중하게 통제 가능하고, 검증 가능하며, 유익한 물리 행동으로 변환하는가에 달려 있다.

##  

## 1.6 Physical AI Foundation Model Requirements

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

Physical AI foundation models extend the foundation model paradigm from digital information processing into embodied interaction with the physical world. A language model can generate text without directly affecting its surroundings, but a Physical AI model must perceive environments, reason about objects and people, predict physical consequences, and generate actions through a robot body. This difference fundamentally changes the requirements of the model. High linguistic fluency, visual recognition accuracy, or benchmark performance alone is not sufficient. A Physical AI foundation model must combine multimodal perception, embodiment awareness, temporal reasoning, world modeling, action generation, safety, adaptability, and real-time execution within a coherent system.

The first major requirement is multimodal perception. Physical environments cannot be understood through a single data source. A robot may receive RGB images, depth maps, LiDAR scans, radar measurements, audio, tactile signals, force measurements, joint states, inertial data, wheel odometry, localization estimates, semantic maps, and natural language instructions. Each modality captures different properties of the world. Cameras provide appearance and semantic information, depth and LiDAR provide geometry, force and tactile sensors reveal contact, and proprioception describes the robot's own body state.

A Physical AI foundation model must transform these heterogeneous signals into a shared internal representation. This does not mean that every modality should be treated identically. Different sensors have different sampling rates, noise characteristics, coordinate systems, resolutions, and failure modes. Modality-specific encoders may therefore be required before information is fused into a common latent space. The model must preserve the unique strengths of each sensor while learning relationships across modalities.

Cross-modal alignment is essential. A language instruction such as "inspect the upper-left weld" must be connected to the correct visual and geometric region. A tactile event must be associated with the visible contact point. A sudden motor-current increase must be linked to the corresponding mechanical interaction. Without accurate alignment, multimodal data becomes a collection of unrelated signals rather than a coherent representation of the physical situation.

Time synchronization is part of this requirement. Sensor observations and robot actions must correspond to the same physical events. Even small timing errors can create incorrect causal associations. A camera image captured before contact must not be paired with force data recorded after contact as if they occurred simultaneously. Hardware timestamps, synchronized clocks, calibrated delays, and consistent logging pipelines are therefore foundational requirements for Physical AI training data.

The second major requirement is embodiment awareness. A robot does not act as an abstract agent. It acts through a specific body with defined dimensions, mass, joint limits, payload capacity, actuator characteristics, sensor placement, kinematics, dynamics, and energy constraints. A foundation model intended for robotics must understand not only the external world but also the capabilities and limitations of the body through which it acts.

Embodiment information may be encoded through robot identity tokens, kinematic descriptions, joint masks, morphology vectors, coordinate frames, actuator specifications, and capability metadata. A command such as "reach the object" has different meanings for a fixed manipulator, a mobile manipulator, a humanoid, a quadruped, or a UAV. The model must distinguish the shared task intention from the platform-specific method of execution.

This requirement becomes even more important for cross-embodiment learning. A general robot foundation model should transfer useful knowledge across different platforms without assuming that every robot has the same action space. Shared latent representations can capture high-level concepts such as approach, grasp, lift, avoid, inspect, or transport. Robot-specific decoders can then transform these abstractions into hardware commands appropriate for each embodiment.

However, cross-embodiment transfer must not erase important physical differences. A heavy autonomous mobile robot cannot accelerate, stop, or turn like a lightweight service robot. A manipulator with limited reach cannot execute the same trajectory as a humanoid. The model must preserve embodiment-specific constraints while learning transferable task structure.

The third requirement is spatial and geometric grounding. Physical AI must understand where objects are, how they are oriented, which regions are free, what is reachable, and how the robot body relates to the surrounding environment. Purely semantic recognition is insufficient. Knowing that an object is a cup does not determine whether it can be safely grasped from the current position.

Geometric grounding requires consistent coordinate systems. Sensor measurements may originate in camera frames, LiDAR frames, robot-base frames, end-effector frames, map frames, or world frames. The model must represent and transform information across these frames. Calibration errors can directly damage control quality, so the architecture should not be expected to learn around severe geometric inconsistency.

Three-dimensional reasoning should include distance, orientation, containment, support, visibility, collision relationships, and reachability. The model should understand that an object may remain present while temporarily occluded, that an unsupported object may fall, and that the robot cannot pass through a space smaller than its body. These relationships can be learned statistically, but they should also be supported by explicit geometry and validated sensor processing.

For manipulation, the model must reason about object pose, grasp points, affordances, contact surfaces, force directions, and tool relationships. For navigation, it must reason about free space, dynamic obstacles, traversability, slope, localization uncertainty, and stopping distance. For inspection, it must relate sensor viewpoint, target geometry, lighting, distance, and image quality.

The fourth requirement is temporal understanding. Physical actions unfold over time, and their effects may be delayed. A robot must connect past observations with present state and expected future outcomes. It must track moving objects, remember earlier instructions, monitor task progress, and detect when execution has diverged from the plan.

Temporal modeling can use action histories, observation sequences, recurrent memory, long-context attention, state-space models, or compressed episodic memory. The system must distinguish rapidly changing information from slowly changing context. Joint states may require high-frequency updates, while a mission goal may remain constant for an entire operation.

Long-horizon memory is particularly important. A mobile inspection robot may operate for hours, visit many stations, record anomalies, skip inaccessible regions, recharge, and resume the mission. The foundation model must maintain task state without processing the entire raw history at every step. Structured memory, external databases, semantic summaries, and mission-state representations may therefore be more effective than an ever-expanding context window.

The fifth requirement is action representation. Physical AI does not end with perception or language generation. The model must produce outputs that can be translated into safe physical behavior. Actions may be represented as discrete tokens, continuous values, waypoints, skill identifiers, action chunks, diffusion trajectories, or flow-based policies.

The chosen action representation must match the control problem. Discrete tokens simplify sequence modeling but introduce quantization error. Continuous outputs support precision but may be harder to model reliably. Skill-level actions improve interpretability and safety but depend on a predefined library. Direct motor commands offer flexibility but create significant timing and validation challenges.

Many practical systems will use hierarchical action representations. The foundation model may generate task plans, skill sequences, or short trajectory segments. Conventional controllers then translate these outputs into high-frequency motor commands. This separation allows the model to focus on semantic reasoning and medium-horizon planning while deterministic control systems maintain stability.

Action chunking is another important requirement. Generating several future actions together can improve temporal consistency and reduce inference latency. However, the robot must remain responsive to new observations. Receding-horizon execution, in which only part of the predicted chunk is executed before replanning, provides a useful compromise.

The sixth requirement is predictive world modeling. A capable Physical AI system should not only recognize the current state but also estimate how the world may change after an action. This ability supports planning, risk assessment, recovery, and counterfactual reasoning.

A world model can predict future observations, latent states, object motion, contact events, task progress, or reward. It may allow the robot to compare several candidate action sequences before physical execution. This reduces the need for risky trial and error in the real world.

World models must be action-conditioned. Passive prediction of what usually happens is not enough. The model must estimate what will happen if the robot deliberately moves, pushes, grasps, turns, or applies force. This distinction between observation and intervention is central to meaningful planning.

Prediction should also include uncertainty. The farther the model imagines into the future, the less reliable its predictions may become. Planning systems should account for this uncertainty and frequently correct imagined trajectories using new observations. A world model should guide action selection rather than be treated as a perfect simulator.

The seventh requirement is causal reasoning. Physical AI must learn not only statistical correlation but also action consequence. A robot must understand that excessive force may damage an object, that rapid braking may destabilize a payload, and that moving one object can block access to another.

Causal understanding can be improved through action-conditioned datasets, simulation, intervention experiments, randomized environments, and failure data. However, purely learned causal representations may remain incomplete. Hidden variables, biased demonstrations, and limited exploration can create incorrect conclusions. Explicit physical models and controlled validation should therefore complement learned reasoning.

The eighth requirement is robust generalization. A foundation model should operate beyond the exact conditions represented in its training data. It must handle new objects, layouts, viewpoints, lighting conditions, payloads, sensor noise, and task combinations. Generalization is one of the main reasons to use a foundation model rather than a task-specific policy.

Compositional generalization is especially valuable. The system should combine familiar objects, actions, and locations into new instructions. A robot that understands "red container," "inspection station," and "left side" separately should be able to interpret a new combination without retraining.

Domain generalization is also necessary. A model trained in one warehouse may need to transfer to another factory with different floors, shelves, lighting, and traffic patterns. Sim-to-real transfer must handle differences between simulated physics and actual hardware. Cross-site deployment should include adaptation mechanisms rather than assuming perfect transfer.

Robustness also includes tolerance to sensor degradation. Cameras may experience glare or blur, LiDAR returns may be sparse, and localization confidence may fall. The model should degrade gracefully rather than fail unpredictably. Sensor redundancy, uncertainty estimation, and fallback behaviors are therefore required.

The ninth requirement is failure detection and recovery. Physical interaction inevitably includes errors. Grasping may fail, navigation may be blocked, localization may drift, tools may slip, and people may interrupt the robot. A useful Physical AI foundation model must recognize failure and choose an appropriate recovery action.

Training data should therefore include more than successful demonstrations. It should contain partial failures, near misses, interrupted tasks, operator corrections, and recovery trajectories. A model trained only on ideal behavior may perform well until the first unexpected event and then become unstable.

Failure detection requires comparing expected and observed outcomes. If a gripper command is issued but force and position signals indicate no object was acquired, the system should recognize the mismatch. If the robot expects forward motion but odometry indicates no progress, it should infer blockage or wheel slip.

Recovery behavior may involve retrying with a modified strategy, repositioning, collecting additional observations, invoking a specialized planner, requesting human assistance, or entering a safe state. The model should not continue confidently when its assumptions have been invalidated.

The tenth requirement is uncertainty awareness. A Physical AI model must know when its predictions are unreliable. Output probability alone is not enough. Confidence should account for sensor quality, distribution shift, missing modalities, disagreement among models, and violation of known constraints.

Uncertainty may be represented through probability distributions, ensembles, multiple trajectory samples, predictive variance, or confidence heads. The system should distinguish between uncertainty caused by incomplete information and uncertainty caused by unfamiliar situations.

The operational response to uncertainty is as important as the estimate itself. A robot may slow down, increase sensing, move to a better viewpoint, request clarification, switch to a conservative controller, or stop. Uncertainty should influence behavior rather than remain only as a diagnostic number.

The eleventh requirement is safety alignment. Language-model alignment focuses on helpful and harmless responses, but Physical AI alignment must govern physical action. The system must follow human intent while respecting constraints related to collision, force, speed, privacy, restricted zones, and mission rules.

Preference learning can help distinguish smoother, safer, or more efficient trajectories. However, safety cannot depend solely on learned preference. Hard constraints must be enforced independently. Joint limits, velocity limits, emergency stops, collision zones, force thresholds, and functional safety requirements should be implemented through deterministic systems.

The foundation model should propose actions within a layered safety architecture. A runtime monitor can evaluate every command, project unsafe outputs into safe regions, or reject them entirely. Safety controllers should retain authority even if the foundation model is highly capable.

Human oversight is another part of alignment. Operators should be able to pause, modify, or cancel tasks. The robot should communicate uncertainty and explain why it stopped. High-risk actions may require explicit approval. Escalation policies should be built into the system rather than improvised after deployment.

The twelfth requirement is real-time performance. A model may be intelligent but operationally useless if its outputs arrive too late. Robot control has strict timing requirements. Perception, planning, action generation, and safety checking must complete within predictable latency bounds.

Different control layers operate at different frequencies. High-level mission planning may run relatively slowly, while motor control may require hundreds or thousands of updates per second. A single large foundation model should not be expected to handle all timescales directly.

Hierarchical deployment is therefore essential. Large models can operate at lower frequencies to interpret goals, analyze scenes, and generate action chunks. Smaller policies and deterministic controllers can execute high-frequency commands. The architecture must define clear interfaces between these layers.

Inference optimization includes quantization, pruning, distillation, efficient attention, caching, token reduction, mixed precision, and hardware-aware compilation. Static information such as task instructions or maps can be cached, while dynamic sensor data is updated continuously.

Predictability matters as much as average speed. Industrial systems require bounded latency and stable behavior under load. A model that usually responds quickly but occasionally stalls may be unsuitable for safety-critical control.

The thirteenth requirement is computational and energy efficiency. Physical robots have limited onboard power, thermal capacity, memory, and processing resources. Every watt used for computation reduces the energy available for movement, sensing, payload operation, and mission duration.

A Physical AI foundation model may be trained on large data-center clusters but deployed through smaller edge models. Knowledge distillation, adapter-based specialization, and hierarchical cloud-edge architectures can support this transition. However, cloud dependence introduces communication delay, connectivity risk, privacy concerns, and operational fragility.

Critical functions should remain available locally. A robot should continue operating safely if the network connection is lost. Cloud systems may support heavy reasoning, fleet learning, or model updates, but local controllers must preserve essential autonomy and safety.

The fourteenth requirement is scalable data infrastructure. Physical AI models require large volumes of synchronized, calibrated, multimodal data. Data collection must support cameras, depth, LiDAR, audio, force, tactile, proprioception, actions, goals, outcomes, and metadata.

The data pipeline should preserve timestamps, coordinate frames, calibration versions, robot identity, software version, environmental context, task labels, and safety events. Without this metadata, combining data from multiple robots and sites becomes unreliable.

Storage systems must support high-bandwidth ingestion, long-term archiving, efficient sampling, dataset versioning, and traceability. Data quality tools should detect corruption, duplication, synchronization errors, sensor dropout, and inconsistent labels.

Privacy and governance are also required. Robots may capture people, workplaces, proprietary equipment, or sensitive operations. Data access, retention, anonymization, ownership, and permitted use must be defined before large-scale collection.

The fifteenth requirement is continual adaptation. Robots operate in environments that change over time. Equipment is moved, floors wear, lighting changes, new products are introduced, and human routines evolve. A static model may gradually become less reliable.

Continual learning should allow the system to absorb new experience without catastrophic forgetting. Replay datasets, regularization, adapter updates, model merging, and periodic retraining can help preserve previous capabilities. Every update should be validated against regression benchmarks.

Fleet learning can accelerate adaptation. Experience from multiple robots can be aggregated to improve the shared model. A failure encountered by one robot can become a recovery example for the fleet. However, fleet-wide deployment requires staged rollout, version control, monitoring, and rollback.

The sixteenth requirement is interpretability and observability. Engineers and operators must understand what the system perceived, what goal it inferred, why it selected an action, and when it became uncertain. Complete neural interpretation may be impossible, but behavior-level transparency is achievable.

Structured intermediate outputs can include detected objects, target associations, planned subgoals, selected skills, predicted outcomes, confidence, and safety status. These outputs make debugging and incident analysis easier.

Operational observability should include model inputs, outputs, latency, resource use, uncertainty, safety interventions, and mission state. Logs must be synchronized and traceable to model and software versions. This is essential for validation, maintenance, and certification.

The seventeenth requirement is evaluation across both capability and safety. Offline accuracy alone cannot establish robot reliability. A model should be evaluated in open-loop prediction, closed-loop simulation, hardware-in-the-loop environments, controlled laboratories, pilot sites, and real deployment.

Capability testing should cover task success, generalization, compositional instructions, tool use, planning, recovery, cross-embodiment transfer, and long-horizon consistency. Safety testing should cover collision, force limits, restricted zones, sensor failure, adversarial conditions, distribution shift, and emergency behavior.

Rare events require special attention. Average performance may hide dangerous tail failures. Simulation, scenario generation, stress testing, and red teaming can expose conditions that are unlikely but operationally important.

The operational design domain must be documented. A model should not be assumed to work reliably outside the environments, robot configurations, speeds, payloads, and sensor conditions in which it has been validated.

The eighteenth requirement is system integration. A Physical AI foundation model should not be treated as a self-contained replacement for all robot software. It must operate within a broader architecture containing perception modules, localization, mapping, motion planning, real-time control, safety, mission management, fleet services, databases, digital twins, and human interfaces.

Clear interfaces are essential. The model should receive well-defined observations and goals and produce outputs with explicit semantics. Versioned APIs, message schemas, coordinate conventions, timing contracts, and failure codes reduce ambiguity.

The model's authority must also be defined. It may recommend a plan, select a skill, generate a trajectory, or directly control certain actions. Higher levels of authority require stronger validation and safety containment.

The nineteenth requirement is reproducible training and deployment. Foundation models are sensitive to data mixtures, random seeds, optimization settings, tokenizer versions, architecture changes, and hardware behavior. Training pipelines must record configurations, datasets, checkpoints, and evaluation results.

Model artifacts should be versioned together with calibration data, action vocabularies, robot adapters, safety policies, and runtime dependencies. A model update without corresponding system metadata may create hidden incompatibilities.

Deployment should support rollback. If a new model produces unexpected behavior, the fleet must return quickly to a validated version. Canary deployment and staged expansion reduce the risk of large-scale failures.

The final requirement is lifecycle governance. Physical AI foundation models are not one-time research outputs. They are continuously operated, monitored, updated, and audited systems. Their capabilities and risks may change after fine-tuning, compression, hardware migration, new sensor integration, or tool connection.

Governance should define responsibility for data, training, validation, deployment approval, incident response, and retirement. Safety cases and technical documentation should evolve with the model. Changes must be assessed at the complete system level rather than only at the neural-network level.

Ultimately, a Physical AI foundation model must satisfy a much broader set of requirements than a digital foundation model. It must perceive through multiple sensors, understand embodiment, reason spatially and temporally, predict action consequences, generate controllable behavior, recognize uncertainty, recover from failure, and operate within strict safety and timing constraints.

The most successful systems will not rely on scale alone. They will combine large-scale multimodal learning with calibrated geometry, world models, hierarchical control, deterministic safety, structured memory, efficient edge inference, and rigorous lifecycle management. The foundation model will provide flexible perception, reasoning, and adaptation, while the surrounding system ensures that these capabilities are transformed into safe, reliable, and purposeful physical action.

피지컬 AI 파운데이션 모델(Physical AI Foundation Model)은 파운데이션 모델(Foundation Model)의 패러다임을 디지털 정보 처리에서 물리 세계와의 체화된 상호작용(Embodied Interaction)으로 확장한다. 언어 모델(Language Model)은 주변 환경에 직접 영향을 주지 않으면서 텍스트를 생성할 수 있지만, 피지컬 AI 모델은 환경을 인식하고, 객체와 사람에 대해 추론하며, 물리적 결과를 예측하고, 로봇의 몸체를 통해 행동을 생성해야 한다. 이러한 차이는 모델에 요구되는 조건을 근본적으로 변화시킨다. 높은 언어적 유창성(Linguistic Fluency), 시각 인식 정확도(Visual Recognition Accuracy), 벤치마크 성능(Benchmark Performance)만으로는 충분하지 않다. 피지컬 AI 파운데이션 모델은 멀티모달 인식(Multimodal Perception), 체화 인식(Embodiment Awareness), 시간 추론(Temporal Reasoning), 월드 모델링(World Modeling), 행동 생성(Action Generation), 안전성(Safety), 적응성(Adaptability), 실시간 실행(Real-Time Execution)을 하나의 일관된 시스템 안에서 결합해야 한다.

첫 번째 주요 요구사항은 멀티모달 인식(Multimodal Perception)이다. 물리 환경은 하나의 데이터 소스만으로 완전히 이해할 수 없다. 로봇은 RGB 이미지, 깊이 지도(Depth Map), LiDAR 스캔, 레이더 측정(Radar Measurement), 오디오(Audio), 촉각 신호(Tactile Signal), 힘 측정(Force Measurement), 관절 상태(Joint State), 관성 데이터(Inertial Data), 휠 오도메트리(Wheel Odometry), 위치 추정값(Localization Estimate), 의미 지도(Semantic Map), 자연어 명령(Natural Language Instruction)을 입력받을 수 있다. 각 모달리티(Modality)는 물리 세계의 서로 다른 특성을 포착한다. 카메라는 외형과 의미 정보를 제공하고, 깊이 센서와 LiDAR는 기하학적 정보를 제공하며, 힘 센서와 촉각 센서는 접촉 상태를 보여주고, 고유감각(Proprioception)은 로봇 자신의 신체 상태를 설명한다.

피지컬 AI 파운데이션 모델은 이러한 이질적인 신호를 하나의 공유 내부 표현(Shared Internal Representation)으로 변환해야 한다. 그렇다고 모든 모달리티를 동일하게 처리해야 한다는 의미는 아니다. 서로 다른 센서는 샘플링 주기(Sampling Rate), 노이즈 특성(Noise Characteristic), 좌표계(Coordinate System), 해상도(Resolution), 고장 양상(Failure Mode)이 서로 다르다. 따라서 정보를 공통 잠재 공간(Common Latent Space)으로 융합하기 전에 모달리티별 인코더(Modality-Specific Encoder)가 필요할 수 있다. 모델은 센서마다 고유한 장점을 유지하면서도 여러 모달리티 사이의 관계를 학습해야 한다.

교차 모달 정렬(Cross-Modal Alignment)은 필수적이다. "왼쪽 위 용접부를 검사하라(inspect the upper-left weld)"라는 언어 명령은 올바른 시각적·기하학적 영역에 연결되어야 한다. 촉각 이벤트(Tactile Event)는 화면에 보이는 실제 접촉 지점과 연관되어야 한다. 모터 전류(Motor Current)의 갑작스러운 증가는 이에 대응하는 기계적 상호작용(Mechanical Interaction)과 연결되어야 한다. 정확한 정렬이 없다면 멀티모달 데이터는 물리 상황에 대한 일관된 표현이 아니라 서로 관련 없는 신호들의 모음에 불과하다.

시간 동기화(Time Synchronization)도 이러한 요구사항의 일부이다. 센서 관측과 로봇 행동은 동일한 물리적 사건에 대응해야 한다. 작은 시간 오차도 잘못된 인과 관계(Causal Association)를 만들 수 있다. 접촉이 발생하기 전에 촬영된 카메라 영상과 접촉 이후 기록된 힘 데이터를 동시에 발생한 것처럼 연결해서는 안 된다. 따라서 하드웨어 타임스탬프(Hardware Timestamp), 동기화된 시계(Synchronized Clock), 보정된 지연 시간(Calibrated Delay), 일관된 로깅 파이프라인(Logging Pipeline)은 피지컬 AI 학습 데이터의 기초 요구사항이다.

두 번째 주요 요구사항은 체화 인식(Embodiment Awareness)이다. 로봇은 추상적인 에이전트(Abstract Agent)로 행동하지 않는다. 로봇은 정해진 크기, 질량, 관절 한계(Joint Limit), 적재 용량(Payload Capacity), 액추에이터 특성(Actuator Characteristic), 센서 배치(Sensor Placement), 운동학(Kinematics), 동역학(Dynamics), 에너지 제약(Energy Constraint)을 가진 구체적인 몸체를 통해 행동한다. 로보틱스를 위한 파운데이션 모델은 외부 세계뿐 아니라 자신이 행동하는 몸체의 능력과 한계도 이해해야 한다.

체화 정보는 로봇 식별 토큰(Robot Identity Token), 운동학 설명(Kinematic Description), 관절 마스크(Joint Mask), 형태 벡터(Morphology Vector), 좌표계(Coordinate Frame), 액추에이터 사양(Actuator Specification), 능력 메타데이터(Capability Metadata)를 통해 인코딩할 수 있다. "물체에 도달하라(reach the object)"는 명령은 고정형 매니퓰레이터(Fixed Manipulator), 이동형 매니퓰레이터(Mobile Manipulator), 휴머노이드(Humanoid), 사족보행 로봇(Quadruped), 무인항공기(UAV)에서 서로 다른 의미를 가진다. 모델은 공통된 작업 의도(Task Intention)와 플랫폼별 실행 방법(Platform-Specific Execution)을 구분해야 한다.

교차 체화 학습(Cross-Embodiment Learning)에서는 이러한 요구사항이 더욱 중요하다. 범용 로봇 파운데이션 모델은 모든 로봇이 동일한 행동 공간(Action Space)을 가진다고 가정하지 않으면서도 서로 다른 플랫폼 사이에 유용한 지식을 전이해야 한다. 공유 잠재 표현(Shared Latent Representation)은 접근(Approach), 파지(Grasp), 들어 올리기(Lift), 회피(Avoid), 검사(Inspect), 운반(Transport)과 같은 상위 개념을 표현할 수 있다. 이후 로봇별 디코더(Robot-Specific Decoder)가 이러한 추상 개념을 각 체화에 적합한 하드웨어 명령으로 변환할 수 있다.

그러나 교차 체화 전이는 중요한 물리적 차이를 제거해서는 안 된다. 대형 자율이동로봇(Autonomous Mobile Robot)은 경량 서비스 로봇과 동일한 방식으로 가속, 정지, 회전할 수 없다. 제한된 작업 반경을 가진 매니퓰레이터는 휴머노이드와 동일한 궤적을 실행할 수 없다. 모델은 전이 가능한 작업 구조를 학습하면서도 체화별 제약조건을 유지해야 한다.

세 번째 요구사항은 공간적·기하학적 접지(Spatial and Geometric Grounding)이다. 피지컬 AI는 객체가 어디에 있고, 어떤 방향을 향하고 있으며, 어느 영역이 비어 있고, 무엇이 도달 가능하며, 로봇의 몸체가 주변 환경과 어떤 관계에 있는지를 이해해야 한다. 단순한 의미 인식(Semantic Recognition)만으로는 충분하지 않다. 어떤 물체가 컵이라는 사실을 안다고 해서 현재 위치에서 안전하게 파지할 수 있는지는 알 수 없다.

기하학적 접지에는 일관된 좌표계(Coordinate System)가 필요하다. 센서 측정값은 카메라 좌표계(Camera Frame), LiDAR 좌표계(LiDAR Frame), 로봇 베이스 좌표계(Robot-Base Frame), 엔드이펙터 좌표계(End-Effector Frame), 지도 좌표계(Map Frame), 월드 좌표계(World Frame)에서 생성될 수 있다. 모델은 이러한 좌표계 사이에서 정보를 표현하고 변환해야 한다. 보정 오류(Calibration Error)는 제어 품질을 직접적으로 저하시킬 수 있으므로, 아키텍처가 심각한 기하학적 불일치를 학습으로 자동 보정할 것이라고 기대해서는 안 된다.

3차원 추론(Three-Dimensional Reasoning)은 거리(Distance), 방향(Orientation), 포함 관계(Containment), 지지(Support), 가시성(Visibility), 충돌 관계(Collision Relationship), 도달 가능성(Reachability)을 포함해야 한다. 모델은 객체가 일시적으로 가려져도 계속 존재하고, 지지되지 않은 객체는 떨어질 수 있으며, 로봇 몸체보다 좁은 공간을 통과할 수 없다는 사실을 이해해야 한다. 이러한 관계는 통계적으로 학습할 수 있지만, 명시적 기하학(Explicit Geometry)과 검증된 센서 처리(Validated Sensor Processing)로도 지원되어야 한다.

조작(Manipulation)에서는 객체 자세(Object Pose), 파지점(Grasp Point), 어포던스(Affordance), 접촉면(Contact Surface), 힘의 방향(Force Direction), 도구 관계(Tool Relationship)를 추론해야 한다. 내비게이션(Navigation)에서는 자유 공간(Free Space), 동적 장애물(Dynamic Obstacle), 주행 가능성(Traversability), 경사(Slope), 위치 추정 불확실성(Localization Uncertainty), 정지 거리(Stopping Distance)를 추론해야 한다. 검사(Inspection)에서는 센서 시점(Sensor Viewpoint), 대상 기하(Target Geometry), 조명(Lighting), 거리, 이미지 품질(Image Quality)의 관계를 이해해야 한다.

네 번째 요구사항은 시간적 이해(Temporal Understanding)이다. 물리적 행동은 시간에 따라 진행되며, 행동의 결과는 지연되어 나타날 수 있다. 로봇은 과거 관측을 현재 상태 및 예상 미래 결과와 연결해야 한다. 움직이는 물체를 추적하고, 이전 지시를 기억하며, 작업 진행 상태를 감시하고, 실행이 계획에서 벗어났는지를 감지해야 한다.

시간 모델링(Temporal Modeling)은 행동 이력(Action History), 관측 시퀀스(Observation Sequence), 순환 메모리(Recurrent Memory), 긴 컨텍스트 어텐션(Long-Context Attention), 상태공간 모델(State-Space Model), 압축된 에피소드 메모리(Compressed Episodic Memory)를 사용할 수 있다. 시스템은 빠르게 변화하는 정보와 천천히 변화하는 문맥을 구분해야 한다. 관절 상태는 고주파 갱신이 필요할 수 있지만, 임무 목표(Mission Goal)는 전체 작업 동안 변하지 않을 수 있다.

장기 기억(Long-Horizon Memory)은 특히 중요하다. 이동형 검사 로봇은 수시간 동안 운용되면서 여러 지점을 방문하고, 이상을 기록하고, 접근할 수 없는 구역을 건너뛰고, 충전한 뒤 임무를 재개할 수 있다. 파운데이션 모델은 매 단계마다 전체 원시 이력을 처리하지 않으면서도 작업 상태를 유지해야 한다. 따라서 계속 확장되는 컨텍스트 윈도우보다 구조화된 메모리(Structured Memory), 외부 데이터베이스(External Database), 의미 요약(Semantic Summary), 임무 상태 표현(Mission-State Representation)이 더 효과적일 수 있다.

다섯 번째 요구사항은 행동 표현(Action Representation)이다. 피지컬 AI는 인식이나 언어 생성에서 끝나지 않는다. 모델은 안전한 물리적 행동으로 변환할 수 있는 출력을 생성해야 한다. 행동은 이산 토큰(Discrete Token), 연속값(Continuous Value), 웨이포인트(Waypoint), 스킬 식별자(Skill Identifier), 행동 청크(Action Chunk), 확산 궤적(Diffusion Trajectory), 플로우 기반 정책(Flow-Based Policy)으로 표현될 수 있다.

선택한 행동 표현은 제어 문제에 적합해야 한다. 이산 토큰은 시퀀스 모델링을 단순화하지만 양자화 오차(Quantization Error)를 발생시킨다. 연속 출력은 정밀성을 제공하지만 신뢰성 있게 모델링하기 더 어려울 수 있다. 스킬 수준 행동(Skill-Level Action)은 해석 가능성과 안전성을 높이지만 사전 정의된 라이브러리에 의존한다. 직접 모터 명령(Direct Motor Command)은 유연성을 제공하지만 시간성과 검증에 큰 어려움을 만든다.

많은 실제 시스템은 계층적 행동 표현(Hierarchical Action Representation)을 사용할 가능성이 높다. 파운데이션 모델은 작업 계획(Task Plan), 스킬 시퀀스(Skill Sequence), 짧은 궤적 구간(Trajectory Segment)을 생성할 수 있다. 이후 기존 제어기(Conventional Controller)가 이를 고주파 모터 명령으로 변환한다. 이러한 분리는 모델이 의미 추론(Semantic Reasoning)과 중기 계획(Medium-Horizon Planning)에 집중하면서 결정론적 제어 시스템이 안정성을 유지하도록 한다.

행동 청킹(Action Chunking)도 중요한 요구사항이다. 여러 미래 행동을 함께 생성하면 시간적 일관성을 향상시키고 추론 지연(Inference Latency)을 줄일 수 있다. 그러나 로봇은 새로운 관측에 계속 반응해야 한다. 예측된 청크의 일부만 실행한 뒤 다시 계획하는 이동 지평선 실행(Receding-Horizon Execution)은 유용한 절충안을 제공한다.

여섯 번째 요구사항은 예측 월드 모델링(Predictive World Modeling)이다. 높은 능력을 가진 피지컬 AI 시스템은 현재 상태를 인식하는 것뿐 아니라 행동 이후 세계가 어떻게 변할지도 추정해야 한다. 이러한 능력은 계획, 위험 평가(Risk Assessment), 복구, 반사실적 추론(Counterfactual Reasoning)을 지원한다.

월드 모델(World Model)은 미래 관측(Future Observation), 잠재 상태(Latent State), 객체 움직임(Object Motion), 접촉 이벤트(Contact Event), 작업 진행 상태(Task Progress), 보상(Reward)을 예측할 수 있다. 로봇은 실제로 행동을 실행하기 전에 여러 후보 행동 시퀀스를 비교할 수 있다. 이를 통해 실세계에서 위험한 시행착오(Trial and Error)를 줄일 수 있다.

월드 모델은 행동 조건부(Action-Conditioned)여야 한다. 일반적으로 다음에 무엇이 일어나는지를 수동적으로 예측하는 것만으로는 충분하지 않다. 로봇이 의도적으로 움직이고, 밀고, 파지하고, 회전하고, 힘을 가했을 때 어떤 결과가 발생할지를 예측해야 한다. 관측과 개입(Intervention)의 차이는 의미 있는 계획의 핵심이다.

예측에는 불확실성도 포함되어야 한다. 모델이 더 먼 미래를 상상할수록 예측의 신뢰성은 낮아질 수 있다. 계획 시스템은 이러한 불확실성을 고려하고, 새로운 관측을 이용하여 상상된 궤적을 자주 보정해야 한다. 월드 모델은 행동 선택을 지원해야 하며, 완벽한 시뮬레이터로 취급되어서는 안 된다.

일곱 번째 요구사항은 인과 추론(Causal Reasoning)이다. 피지컬 AI는 단순한 통계적 상관관계뿐 아니라 행동의 결과도 학습해야 한다. 로봇은 과도한 힘이 물체를 손상시킬 수 있고, 급격한 제동이 적재물을 불안정하게 만들 수 있으며, 하나의 물체를 이동하면 다른 물체에 대한 접근이 막힐 수 있다는 점을 이해해야 한다.

인과적 이해는 행동 조건부 데이터셋(Action-Conditioned Dataset), 시뮬레이션(Simulation), 개입 실험(Intervention Experiment), 무작위 환경(Randomized Environment), 실패 데이터(Failure Data)를 통해 향상될 수 있다. 그러나 순수하게 학습된 인과 표현은 여전히 불완전할 수 있다. 숨겨진 변수(Hidden Variable), 편향된 시연(Biased Demonstration), 제한된 탐색(Limited Exploration)은 잘못된 결론을 만들 수 있다. 따라서 명시적 물리 모델(Explicit Physical Model)과 통제된 검증(Controlled Validation)이 학습 기반 추론을 보완해야 한다.

여덟 번째 요구사항은 강건한 일반화(Robust Generalization)이다. 파운데이션 모델은 학습 데이터에 정확히 포함된 조건을 넘어 동작해야 한다. 새로운 객체, 레이아웃(Layout), 시점(Viewpoint), 조명 조건, 적재하중, 센서 노이즈, 작업 조합을 처리할 수 있어야 한다. 일반화는 작업별 정책(Task-Specific Policy) 대신 파운데이션 모델을 사용하는 주된 이유 가운데 하나이다.

조합적 일반화(Compositional Generalization)는 특히 가치가 크다. 시스템은 익숙한 객체, 행동, 위치를 새로운 지시로 조합할 수 있어야 한다. "빨간 용기(red container)", "검사 지점(inspection station)", "왼쪽(left side)"을 각각 이해하는 로봇은 추가 학습 없이 새로운 조합을 해석할 수 있어야 한다.

도메인 일반화(Domain Generalization)도 필요하다. 하나의 창고에서 학습한 모델이 바닥, 선반, 조명, 교통 패턴이 다른 공장으로 전이되어야 할 수 있다. 시뮬레이션-현실 전이(Sim-to-Real Transfer)는 시뮬레이션 물리와 실제 하드웨어의 차이를 처리해야 한다. 사이트 간 배포(Cross-Site Deployment)는 완벽한 전이를 가정하기보다 적응 메커니즘(Adaptation Mechanism)을 포함해야 한다.

강건성은 센서 성능 저하에 대한 내성도 포함한다. 카메라에 눈부심(Glare)이나 블러(Blur)가 발생하고, LiDAR 반환값이 희소해지며, 위치 추정 신뢰도가 떨어질 수 있다. 모델은 예측 불가능하게 실패하는 대신 점진적으로 성능이 저하되어야 한다. 따라서 센서 중복성(Sensor Redundancy), 불확실성 추정, 대체 행동(Fallback Behavior)이 필요하다.

아홉 번째 요구사항은 실패 감지(Failure Detection)와 복구(Recovery)이다. 물리적 상호작용에서는 오류가 필연적으로 발생한다. 파지는 실패할 수 있고, 내비게이션 경로는 막힐 수 있으며, 위치 추정은 드리프트(Drift)할 수 있고, 도구는 미끄러질 수 있으며, 사람이 로봇을 방해할 수 있다. 유용한 피지컬 AI 파운데이션 모델은 실패를 인식하고 적절한 복구 행동을 선택해야 한다.

따라서 학습 데이터는 성공 시연만 포함해서는 안 된다. 부분 실패(Partial Failure), 니어미스(Near Miss), 중단된 작업(Interrupted Task), 작업자 수정(Operator Correction), 복구 궤적(Recovery Trajectory)을 포함해야 한다. 이상적인 행동만으로 학습된 모델은 첫 번째 예상치 못한 사건이 발생하기 전까지는 잘 동작하지만, 이후 불안정해질 수 있다.

실패 감지는 예상 결과와 실제 관측 결과를 비교하는 능력을 요구한다. 그리퍼 명령을 내렸지만 힘과 위치 신호가 물체를 잡지 못했음을 보여준다면 시스템은 불일치를 인식해야 한다. 로봇이 전진할 것으로 예상했지만 오도메트리에서 이동이 나타나지 않는다면 막힘이나 휠 슬립(Wheel Slip)을 추론해야 한다.

복구 행동에는 수정된 전략으로 재시도하거나, 위치를 변경하거나, 추가 관측을 수집하거나, 전문 계획기를 호출하거나, 인간의 도움을 요청하거나, 안전 상태로 진입하는 방법이 포함될 수 있다. 모델은 자신의 전제가 무효화되었음에도 자신 있게 계속 행동해서는 안 된다.

열 번째 요구사항은 불확실성 인식(Uncertainty Awareness)이다. 피지컬 AI 모델은 자신의 예측이 신뢰할 수 없는 시점을 알아야 한다. 출력 확률만으로는 충분하지 않다. 신뢰도는 센서 품질, 분포 변화(Distribution Shift), 모달리티 누락(Missing Modality), 모델 간 불일치, 알려진 제약 위반을 함께 고려해야 한다.

불확실성은 확률 분포(Probability Distribution), 앙상블(Ensemble), 다중 궤적 샘플(Multiple Trajectory Sample), 예측 분산(Predictive Variance), 신뢰도 헤드(Confidence Head)를 통해 표현할 수 있다. 시스템은 정보가 부족해서 발생하는 불확실성과 익숙하지 않은 상황 때문에 발생하는 불확실성을 구분해야 한다.

불확실성에 대한 운영 대응은 추정값 자체만큼 중요하다. 로봇은 속도를 낮추고, 센싱을 강화하고, 더 나은 시점으로 이동하고, 명확한 설명을 요청하고, 보수적인 제어기로 전환하거나, 정지할 수 있다. 불확실성은 단순한 진단 수치로 남지 않고 실제 행동에 영향을 주어야 한다.

열한 번째 요구사항은 안전 정렬(Safety Alignment)이다. 언어 모델의 정렬은 유용하고 해롭지 않은 응답에 초점을 맞추지만, 피지컬 AI의 정렬은 물리적 행동을 통제해야 한다. 시스템은 충돌, 힘, 속도, 개인정보 보호(Privacy), 제한 구역(Restricted Zone), 임무 규칙(Mission Rule)과 관련된 제약을 존중하면서 인간의 의도를 따라야 한다.

선호 학습(Preference Learning)은 더 부드럽고, 안전하고, 효율적인 궤적을 구분하는 데 도움을 줄 수 있다. 그러나 안전은 학습된 선호에만 의존해서는 안 된다. 하드 제약(Hard Constraint)은 독립적으로 강제되어야 한다. 관절 제한, 속도 제한, 비상 정지(Emergency Stop), 충돌 구역, 힘 임계값, 기능 안전 요구사항(Functional Safety Requirement)은 결정론적 시스템(Deterministic System)을 통해 구현되어야 한다.

파운데이션 모델은 계층형 안전 아키텍처(Layered Safety Architecture) 안에서 행동을 제안해야 한다. 런타임 모니터(Runtime Monitor)는 모든 명령을 평가하고, 위험한 출력을 안전 영역으로 투영하거나 완전히 거부할 수 있다. 파운데이션 모델의 능력이 높더라도 안전 제어기는 최종 권한을 유지해야 한다.

인간 감독(Human Oversight)도 정렬의 일부이다. 작업자는 작업을 일시 정지하고, 수정하고, 취소할 수 있어야 한다. 로봇은 불확실성을 전달하고 정지한 이유를 설명해야 한다. 고위험 행동은 명시적 승인을 요구할 수 있다. 에스컬레이션 정책(Escalation Policy)은 배포 후 임시로 만드는 것이 아니라 시스템에 미리 포함되어야 한다.

열두 번째 요구사항은 실시간 성능(Real-Time Performance)이다. 모델이 지능적이더라도 출력이 너무 늦게 생성되면 운영상 쓸모가 없다. 로봇 제어에는 엄격한 시간 요구사항이 존재한다. 인식, 계획, 행동 생성, 안전 검사는 예측 가능한 지연 시간 안에 완료되어야 한다.

서로 다른 제어 계층은 서로 다른 주기로 작동한다. 상위 임무 계획(High-Level Mission Planning)은 비교적 느리게 동작할 수 있지만, 모터 제어는 초당 수백 또는 수천 회의 갱신이 필요할 수 있다. 하나의 대형 파운데이션 모델이 모든 시간 척도를 직접 처리할 것이라고 기대해서는 안 된다.

따라서 계층적 배포(Hierarchical Deployment)가 필수적이다. 대형 모델은 낮은 주기로 목표를 해석하고, 장면을 분석하고, 행동 청크를 생성할 수 있다. 소형 정책 모델과 결정론적 제어기는 고주파 명령을 실행할 수 있다. 아키텍처는 이러한 계층 사이에 명확한 인터페이스를 정의해야 한다.

추론 최적화(Inference Optimization)는 양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation), 효율적 어텐션(Efficient Attention), 캐싱(Caching), 토큰 축소(Token Reduction), 혼합 정밀도(Mixed Precision), 하드웨어 인식 컴파일(Hardware-Aware Compilation)을 포함한다. 작업 지시나 지도와 같은 정적 정보는 캐싱하고, 동적 센서 데이터만 지속적으로 갱신할 수 있다.

평균 속도만큼 예측 가능성(Predictability)도 중요하다. 산업 시스템은 제한된 지연 시간(Bounded Latency)과 부하 상황에서도 안정적인 행동을 요구한다. 대부분의 경우 빠르게 응답하지만 가끔 멈추는 모델은 안전이 중요한 제어에 적합하지 않을 수 있다.

열세 번째 요구사항은 계산 및 에너지 효율성(Computational and Energy Efficiency)이다. 물리 로봇은 제한된 온보드 전력(Onboard Power), 열 용량(Thermal Capacity), 메모리, 처리 자원을 가진다. 계산에 사용하는 모든 전력은 이동, 센싱, 적재물 동작, 임무 지속 시간에 사용할 수 있는 에너지를 줄인다.

피지컬 AI 파운데이션 모델은 대규모 데이터센터 클러스터에서 학습한 뒤 더 작은 엣지 모델(Edge Model)을 통해 배포할 수 있다. 지식 증류, 어댑터 기반 특화(Adapter-Based Specialization), 계층적 클라우드-엣지 아키텍처(Hierarchical Cloud-Edge Architecture)가 이러한 전환을 지원할 수 있다. 그러나 클라우드 의존성은 통신 지연, 연결 위험, 개인정보 보호 문제, 운영 취약성을 발생시킨다.

중요한 기능은 로컬에서 계속 사용할 수 있어야 한다. 네트워크 연결이 끊어져도 로봇은 안전하게 운용되어야 한다. 클라우드는 고부하 추론(Heavy Reasoning), 플릿 학습(Fleet Learning), 모델 업데이트를 지원할 수 있지만, 로컬 제어기는 필수 자율성과 안전을 유지해야 한다.

열네 번째 요구사항은 확장 가능한 데이터 인프라(Scalable Data Infrastructure)이다. 피지컬 AI 모델은 동기화되고, 보정되며, 멀티모달인 대규모 데이터를 필요로 한다. 데이터 수집 시스템은 카메라, 깊이, LiDAR, 오디오, 힘, 촉각, 고유감각, 행동, 목표, 결과, 메타데이터를 지원해야 한다.

데이터 파이프라인은 타임스탬프, 좌표계, 보정 버전(Calibration Version), 로봇 식별자, 소프트웨어 버전, 환경 문맥(Environmental Context), 작업 레이블(Task Label), 안전 이벤트(Safety Event)를 보존해야 한다. 이러한 메타데이터가 없다면 여러 로봇과 사이트에서 수집한 데이터를 신뢰성 있게 결합하기 어렵다.

저장 시스템은 고대역폭 수집(High-Bandwidth Ingestion), 장기 보관(Long-Term Archiving), 효율적인 샘플링, 데이터셋 버전 관리(Dataset Versioning), 추적 가능성(Traceability)을 지원해야 한다. 데이터 품질 도구는 손상(Corruption), 중복(Duplication), 동기화 오류, 센서 누락(Sensor Dropout), 불일치 레이블을 감지해야 한다.

개인정보 보호와 거버넌스(Governance)도 필요하다. 로봇은 사람, 작업장, 독점 장비(Proprietary Equipment), 민감한 운영 상황을 촬영할 수 있다. 대규모 수집을 시작하기 전에 데이터 접근, 보유 기간(Retention), 익명화(Anonymization), 소유권, 허용된 사용 범위를 정의해야 한다.

열다섯 번째 요구사항은 지속적 적응(Continual Adaptation)이다. 로봇은 시간이 지나면서 변화하는 환경에서 운용된다. 장비 위치가 바뀌고, 바닥이 마모되고, 조명이 변하며, 새로운 제품이 투입되고, 인간의 작업 습관도 달라진다. 정적인 모델은 시간이 지나면서 점차 신뢰성을 잃을 수 있다.

지속학습(Continual Learning)은 기존 능력을 파국적으로 잊지 않으면서 새로운 경험을 흡수할 수 있어야 한다. 리플레이 데이터셋(Replay Dataset), 정규화(Regularization), 어댑터 업데이트(Adapter Update), 모델 병합(Model Merging), 주기적 재학습(Periodic Retraining)이 기존 능력을 보존하는 데 도움을 줄 수 있다. 모든 업데이트는 회귀 벤치마크(Regression Benchmark)를 통해 검증되어야 한다.

플릿 학습은 적응을 가속할 수 있다. 여러 로봇의 경험을 통합하여 공유 모델을 개선할 수 있다. 한 로봇이 경험한 실패를 전체 플릿의 복구 사례로 사용할 수 있다. 그러나 플릿 전체 배포에는 단계적 롤아웃(Staged Rollout), 버전 관리, 모니터링, 롤백이 필요하다.

열여섯 번째 요구사항은 해석 가능성(Interpretability)과 관측 가능성(Observability)이다. 엔지니어와 작업자는 시스템이 무엇을 인식했고, 어떤 목표를 추론했으며, 왜 특정 행동을 선택했고, 언제 불확실해졌는지를 이해할 수 있어야 한다. 신경망의 완전한 해석은 불가능할 수 있지만, 행동 수준의 투명성(Behavior-Level Transparency)은 달성할 수 있다.

구조화된 중간 출력에는 검출된 객체, 목표 연결(Target Association), 계획된 하위 목표(Planned Subgoal), 선택된 스킬(Selected Skill), 예측 결과(Predicted Outcome), 신뢰도, 안전 상태(Safety Status)가 포함될 수 있다. 이러한 출력은 디버깅과 사고 분석(Incident Analysis)을 더 쉽게 만든다.

운영 관측 가능성은 모델 입력, 출력, 지연 시간, 자원 사용량, 불확실성, 안전 개입(Safety Intervention), 임무 상태를 포함해야 한다. 로그는 동기화되어야 하며 모델 및 소프트웨어 버전까지 추적할 수 있어야 한다. 이는 검증, 유지보수, 인증에 필수적이다.

열일곱 번째 요구사항은 능력과 안전을 모두 포함하는 평가(Evaluation)이다. 오프라인 정확도만으로는 로봇의 신뢰성을 입증할 수 없다. 모델은 개방 루프 예측(Open-Loop Prediction), 폐루프 시뮬레이션(Closed-Loop Simulation), 하드웨어 인 더 루프(Hardware-in-the-Loop), 통제된 실험실, 파일럿 현장, 실제 배포 환경에서 평가되어야 한다.

능력 시험(Capability Testing)은 작업 성공, 일반화, 조합적 지시, 도구 사용, 계획, 복구, 교차 체화 전이, 장기 일관성을 포함해야 한다. 안전 시험(Safety Testing)은 충돌, 힘 제한, 제한 구역, 센서 장애, 적대적 조건(Adversarial Condition), 분포 변화, 비상 행동(Emergency Behavior)을 포함해야 한다.

희귀 사건(Rare Event)은 특별히 주의해야 한다. 평균 성능은 위험한 꼬리 실패(Tail Failure)를 숨길 수 있다. 시뮬레이션, 시나리오 생성(Scenario Generation), 스트레스 테스트(Stress Testing), 레드팀 테스트(Red Teaming)는 발생 가능성은 낮지만 운영상 중요한 조건을 드러낼 수 있다.

운용 설계 영역(Operational Design Domain)은 문서화되어야 한다. 모델이 검증된 환경, 로봇 구성, 속도, 적재하중, 센서 조건을 넘어 신뢰성 있게 동작한다고 가정해서는 안 된다.

열여덟 번째 요구사항은 시스템 통합(System Integration)이다. 피지컬 AI 파운데이션 모델을 모든 로봇 소프트웨어를 대체하는 독립형 시스템으로 취급해서는 안 된다. 인식 모듈, 위치 추정, 지도 작성(Mapping), 모션 계획(Motion Planning), 실시간 제어, 안전, 임무 관리(Mission Management), 플릿 서비스(Fleet Service), 데이터베이스, 디지털 트윈, 인간 인터페이스를 포함하는 더 큰 아키텍처 안에서 동작해야 한다.

명확한 인터페이스가 필수적이다. 모델은 잘 정의된 관측과 목표를 입력받고, 의미가 명확한 출력을 생성해야 한다. 버전이 관리되는 API, 메시지 스키마(Message Schema), 좌표계 규약(Coordinate Convention), 시간 계약(Timing Contract), 실패 코드(Failure Code)는 모호성을 줄인다.

모델의 권한 범위(Authority)도 정의되어야 한다. 모델은 계획을 추천하거나, 스킬을 선택하거나, 궤적을 생성하거나, 특정 행동을 직접 제어할 수 있다. 더 높은 권한에는 더 강한 검증과 안전 격리(Safety Containment)가 필요하다.

열아홉 번째 요구사항은 재현 가능한 학습과 배포(Reproducible Training and Deployment)이다. 파운데이션 모델은 데이터 혼합(Data Mixture), 무작위 시드(Random Seed), 최적화 설정, 토크나이저 버전(Tokenizer Version), 아키텍처 변경, 하드웨어 동작에 민감하다. 학습 파이프라인은 설정, 데이터셋, 체크포인트(Checkpoint), 평가 결과를 기록해야 한다.

모델 아티팩트(Model Artifact)는 보정 데이터, 행동 어휘(Action Vocabulary), 로봇 어댑터(Robot Adapter), 안전 정책(Safety Policy), 런타임 의존성(Runtime Dependency)과 함께 버전 관리되어야 한다. 관련 시스템 메타데이터 없이 모델만 업데이트하면 숨겨진 비호환성(Incompatibility)이 발생할 수 있다.

배포 시스템은 롤백을 지원해야 한다. 새로운 모델이 예상하지 못한 행동을 보이면 플릿은 검증된 버전으로 빠르게 복귀해야 한다. 카나리 배포(Canary Deployment)와 단계적 확대(Staged Expansion)는 대규모 실패 위험을 줄인다.

마지막 요구사항은 수명주기 거버넌스(Lifecycle Governance)이다. 피지컬 AI 파운데이션 모델은 일회성 연구 결과가 아니다. 지속적으로 운용되고, 모니터링되고, 업데이트되고, 감사(Audit)되는 시스템이다. 미세조정(Fine-Tuning), 압축(Compression), 하드웨어 이전(Hardware Migration), 새로운 센서 통합, 도구 연결 이후 모델의 능력과 위험은 달라질 수 있다.

거버넌스는 데이터, 학습, 검증, 배포 승인(Deployment Approval), 사고 대응(Incident Response), 폐기(Retirement)에 대한 책임을 정의해야 한다. 안전 사례(Safety Case)와 기술 문서는 모델과 함께 발전해야 한다. 변경 사항은 신경망 수준뿐 아니라 전체 시스템 수준에서 평가되어야 한다.

궁극적으로 피지컬 AI 파운데이션 모델은 디지털 파운데이션 모델보다 훨씬 더 폭넓은 요구사항을 만족해야 한다. 여러 센서를 통해 인식하고, 체화를 이해하며, 공간적·시간적으로 추론하고, 행동의 결과를 예측하며, 제어 가능한 행동을 생성하고, 불확실성을 인식하고, 실패에서 복구하고, 엄격한 안전성과 시간 제약 안에서 동작해야 한다.

가장 성공적인 시스템은 규모(Scale)에만 의존하지 않을 것이다. 대규모 멀티모달 학습(Large-Scale Multimodal Learning)을 보정된 기하학(Calibrated Geometry), 월드 모델, 계층적 제어(Hierarchical Control), 결정론적 안전(Deterministic Safety), 구조화된 메모리, 효율적인 엣지 추론(Efficient Edge Inference), 엄격한 수명주기 관리와 결합할 것이다. 파운데이션 모델은 유연한 인식, 추론, 적응을 제공하고, 주변 시스템은 이러한 능력이 안전하고, 신뢰할 수 있으며, 목적 지향적인 물리 행동으로 변환되도록 보장하게 될 것이다.

##  

## 1.7 Robot Embodiment and Grounding Challenges

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

Robot embodiment and grounding are central challenges in the development of Physical AI foundation models. A digital model can process language, images, or code without possessing a body that must obey physical laws. A robot, by contrast, perceives and acts through a specific mechanical structure. Its intelligence is inseparable from its sensors, actuators, dimensions, mass, kinematics, dynamics, energy limits, and interaction with the surrounding environment. A robot foundation model must therefore learn not only what objects and instructions mean, but also how those meanings relate to a particular body acting in a physical world.

Embodiment refers to the way intelligence is shaped by the body through which an agent perceives and acts. The body determines what the robot can sense, where it can move, what it can reach, how much force it can apply, how quickly it can stop, and what risks arise during interaction. A wheeled autonomous mobile robot, a humanoid, a quadruped, a fixed manipulator, and an unmanned aerial vehicle may all receive the instruction to inspect an object, but the meaning of execution differs for each platform.

A mobile robot may approach the object and position a sensor. A manipulator may move an end effector around it. A humanoid may walk, bend, and use two hands. A UAV may inspect it from the air. The semantic goal is shared, but the available action spaces, constraints, and physical strategies are different. A foundation model must represent both the common task abstraction and the embodiment-specific realization.

Grounding refers to connecting symbols, language, concepts, or internal representations to real observations and actions. The word "cup" must correspond to an actual object in the current scene. The instruction "move closer" must correspond to a measurable change in spatial distance. The phrase "push gently" must be translated into an appropriate direction, velocity, contact force, and duration. Without grounding, a model may generate fluent descriptions while remaining unable to act reliably in the physical world.

Embodiment and grounding are closely related. Grounding is always mediated by the robot's body and sensors. What a robot can understand operationally depends on what it can perceive and how it can act. A tactile robot hand may ground the concept of softness through deformation and force response, while a mobile robot without touch sensors may rely only on visual appearance or prior knowledge. A robot with depth sensing may ground distance metrically, whereas a monocular system may estimate it indirectly.

One of the first embodiment challenges is sensor diversity. Different robot platforms use different cameras, LiDAR systems, depth sensors, inertial units, force sensors, encoders, tactile arrays, microphones, and localization devices. Even robots performing the same task may observe the world through very different sensor configurations. Their fields of view, resolutions, sampling rates, noise characteristics, and coordinate frames may not match.

A foundation model trained on one sensor arrangement may transfer poorly to another. A camera mounted near the floor sees objects from a different viewpoint than a camera mounted on a humanoid head. A wrist camera moves with a manipulator and produces a changing local perspective. A fixed overhead camera provides a global view but may suffer from occlusion. The model must understand that visual appearance depends on sensor position and orientation.

Sensor identity and calibration information may therefore need to be encoded explicitly. Camera intrinsics, extrinsics, field of view, mounting pose, timing characteristics, and distortion parameters affect how observations should be interpreted. Treating all images as equivalent can cause the model to confuse perspective variation with object variation.

Coordinate frames create another grounding challenge. Robot systems use multiple coordinate systems, including sensor frames, robot-base frames, end-effector frames, local map frames, global map frames, and world frames. A point that is directly in front of a camera may be to the left of the robot base. A target position expressed in map coordinates must be transformed before the controller can use it.

A foundation model must either receive geometrically consistent inputs or learn explicit frame relationships. Severe calibration errors should not be treated as a problem that a model will automatically overcome through scale. Incorrect transformations can produce systematic spatial errors that lead directly to unsafe behavior.

Temporal alignment is equally important. Observations, state estimates, and actions must refer to the same physical moment. Cameras may operate at thirty frames per second, joint encoders at hundreds of hertz, force sensors at higher rates, and language instructions only occasionally. Communication, buffering, and processing delays can cause data streams to become misaligned.

If an image is paired with a robot state recorded several hundred milliseconds later, the model may learn an incorrect relationship between appearance and action. This problem becomes especially serious during rapid movement, contact, or obstacle avoidance. Hardware timestamps, clock synchronization, latency calibration, and time-aware embeddings are therefore essential for grounded learning.

Another major challenge is morphology. Robot bodies differ in number of joints, joint arrangement, link lengths, wheel configuration, steering mechanism, limb structure, payload, and workspace. A policy trained for a six-axis industrial arm cannot be transferred directly to a seven-axis collaborative arm without accounting for the different kinematic structure.

Morphology affects both perception and control. A robot's own body may block its cameras. A manipulator can reach some objects only from particular configurations. A large mobile robot requires more space to turn and stop. A quadruped must maintain balance while moving its feet. Aerial robots must continuously control attitude and thrust.

One approach is to describe morphology explicitly through structured tokens or graphs. Each joint, link, sensor, and actuator can be represented with type, position, orientation, limits, and connectivity. A model can then reason over the robot as a structured body rather than as an unidentified vector of numbers.

Graph representations are particularly useful because robot bodies naturally form kinematic trees or more general mechanical graphs. Message passing or attention can emphasize relationships between connected components. This may allow a shared model to process different robot morphologies while preserving their structural differences.

However, a body description alone does not guarantee transfer. The model must also understand the functional significance of morphology. Two joints with similar numerical limits may play very different roles depending on their location. A shoulder joint, wheel steering joint, gripper finger, and camera gimbal cannot be treated as interchangeable simply because they are rotational actuators.

Actuator characteristics further complicate embodiment. Motors differ in torque, speed, backlash, compliance, friction, control bandwidth, and thermal limits. Hydraulic systems behave differently from electric motors. Series-elastic actuators behave differently from rigid industrial joints. A command that is safe on one robot may cause oscillation, overheating, or excessive force on another.

A Physical AI foundation model should not assume that desired actions translate perfectly into motion. Actuator delay, saturation, dead zones, backlash, and mechanical wear affect execution. These factors may need to be represented through system identification, learned dynamics, or explicit hardware metadata.

Mass and inertia are also important. A heavy inspection robot carrying a large payload cannot react like a small educational platform. Its stopping distance, tipping risk, energy use, and collision consequences are much greater. The model must understand that the same velocity command can have different physical implications depending on total mass, center of gravity, floor condition, and payload distribution.

Payload changes create dynamic embodiment. A robot's body is not always constant. A mobile manipulator carrying a heavy object has different acceleration and stability limits than when empty. A robot arm holding a long tool has a different effective reach and collision envelope. A UAV carrying a payload has reduced flight time and altered dynamics.

The model should therefore represent current payload, tool configuration, center of gravity, battery state, and other changing embodiment variables. Static robot identity alone is insufficient. Embodiment is partly fixed by design and partly altered by the current operating state.

Tool use extends the body even further. When a robot holds a screwdriver, inspection probe, suction device, or camera boom, the tool becomes part of its effective embodiment. The model must understand the tool's geometry, contact properties, operating constraints, and relationship to the robot's motion.

Tool grounding requires more than recognizing the tool category. The robot must know where the functional tip is located, which direction force should be applied, and what outcomes indicate successful use. Tool calibration and tool-center-point definitions are therefore fundamental.

Action-space variation is one of the most difficult challenges for general robot models. Language models usually operate over a shared token space. Robots produce actions in very different formats. One robot may use joint positions, another joint velocities, another Cartesian poses, and another discrete navigation commands.

A common robot foundation model needs a strategy for representing these heterogeneous actions. One option is to transform them into a shared latent action space representing intentions such as move, rotate, reach, grasp, release, or apply force. Robot-specific decoders can then map latent actions into platform commands.

This improves transfer but can hide important details. The concept "move forward" means something different for a differential-drive robot, an omnidirectional platform, a quadruped, and a humanoid. A shared latent representation must preserve enough information for each embodiment while avoiding a vocabulary so specific that transfer becomes impossible.

Another approach is to condition the model on explicit action schemas. The model receives a description of the available controls, dimensions, units, limits, and update rates. It then generates actions within the current schema. This resembles tool use, where the model learns to call different interfaces according to their definitions.

Action normalization can also help. Joint positions, velocities, and forces can be converted into normalized ranges. Coordinate systems can be standardized. However, normalization does not remove differences in dynamics or risk. A normalized value of one does not have the same physical consequence across platforms.

The grounding of language into action is especially challenging because natural-language instructions are often ambiguous. "Move a little closer" does not specify an exact distance. "Grip firmly" depends on object fragility. "Quickly" may conflict with safety or payload stability. Human language relies heavily on shared context and common sense.

A robot must resolve such ambiguity using current observations, task context, user history, and operational rules. It should ask for clarification when uncertainty is too high rather than selecting an arbitrary interpretation. Grounded language understanding therefore requires uncertainty estimation and interactive communication.

Spatial language presents additional difficulty. Terms such as left, right, behind, above, near, and far depend on a reference frame. "Move to the left of the machine" may mean the operator's left, the robot's left, or the machine's own left side. The system must identify or request the intended frame.

Reference resolution is a related challenge. An instruction such as "pick that one" requires the model to identify which object the speaker indicates through gaze, pointing, dialogue context, or previous interaction. In human-robot collaboration, language grounding may therefore require vision, gesture recognition, and conversational memory.

Affordance grounding concerns what actions an object supports. A handle affords pulling, a button affords pressing, and a flat surface may support placing. These relationships are not determined by appearance alone. A visually similar handle may be decorative or fixed. A surface may look stable but be too weak to support weight.

Foundation models may learn broad affordance priors from images, videos, and text, but physical interaction is necessary for reliable grounding. Force, motion, sound, and tactile feedback reveal whether the expected affordance is real. Active perception and exploratory action can reduce uncertainty.

Contact grounding is particularly difficult. Before contact, the robot reasons mainly from vision and geometry. During contact, force, torque, tactile, motor current, and deformation become critical. The model must recognize the transition between free-space motion and physical interaction.

Contact events are brief, high-frequency, and sensitive to synchronization. A small delay can cause force to increase dangerously. Large general-purpose models are rarely suitable for direct high-bandwidth contact control. A hierarchical architecture should use the foundation model for contact strategy while lower-level controllers regulate force and impedance.

Object permanence is another aspect of grounding. Objects may become occluded, leave the camera view, or change appearance under lighting. The robot should maintain a belief about their continued existence and probable location. This requires memory and world modeling rather than frame-by-frame recognition alone.

Dynamic environments make grounding even harder. People move, doors open, pallets are relocated, and machinery changes state. A representation grounded at one moment may become outdated. The system must update continuously and know how old each observation is.

Freshness indicators and timestamps can help. A model should not treat a map created yesterday as equivalent to a current LiDAR observation. Static knowledge and live sensor evidence must be distinguished. When they conflict, the system should determine which source is more trustworthy.

Partial observability is unavoidable. Sensors cannot see everything, and some relevant variables are hidden. The friction of a surface, internal damage of an object, or intention of a human may not be directly observable. The robot must maintain uncertainty rather than invent precise values.

Belief-state representations can summarize what is known and unknown. A model may estimate probability distributions over object pose, contact state, or human intention. Planning should then account for uncertainty, possibly selecting actions that gather information before committing to a risky behavior.

Active perception is therefore an important grounding capability. The robot can move a camera, change viewpoint, touch an object, or ask a person for clarification to reduce uncertainty. Perception becomes an action selection problem rather than a passive input process.

A robot may reposition itself to see behind an obstacle, rotate an object to inspect another surface, or approach slowly to improve depth estimation. A foundation model should learn when additional sensing is more valuable than immediate execution.

Simulation offers a scalable environment for learning embodiment and grounding. Robot models, sensors, dynamics, objects, and environments can be varied systematically. Millions of interactions can be generated without damaging hardware. Rare failures and dangerous situations can be explored safely.

However, simulation introduces the reality gap. Simulated images, friction, contact, actuator delay, deformation, and sensor noise differ from the real world. A policy may exploit simulation artifacts that do not exist physically. Visual realism alone does not guarantee accurate dynamics.

Domain randomization attempts to address this problem by varying textures, lighting, object properties, dynamics, sensor noise, and latency. The model learns not to rely on one exact simulation configuration. Yet excessive randomization can create unrealistic data and make learning unnecessarily difficult.

System identification can improve simulation fidelity by estimating real robot parameters. Recorded trajectories can be used to tune friction, motor response, mass, and delay models. Even then, real-world fine-tuning and validation remain necessary.

Real robot data is expensive. It requires hardware time, maintenance, operators, safety supervision, calibration, and storage. Physical interaction is slower than processing text or images, and failures can interrupt data collection. This limits direct scaling.

Teleoperation is a common source of grounded robot demonstrations. Human operators control the robot while observations and actions are recorded. This provides successful behavior but also includes operator habits, delays, and inconsistencies. Different operators may use different strategies for the same task.

Demonstrations must be synchronized and annotated with embodiment metadata. Robot configuration, tool, payload, control mode, calibration, and environment should be recorded. Without this information, trajectories from different systems may be impossible to combine correctly.

Autonomous data collection can expand scale. Robots can repeat tasks, explore variations, and identify failures. However, autonomous collection may reinforce the current policy's biases. If the robot avoids difficult situations, the dataset will lack the very cases required for improvement.

Failure and recovery data are especially important for grounding. A model that sees only successful grasping may not recognize a slipped object. A robot trained only on clear navigation paths may fail when blocked. Grounded intelligence requires examples of expectation violation and corrective action.

The model must compare predicted consequences with observed consequences. If an action was intended to move an object but the object remained stationary, the system should revise its belief about the object, tool, contact, or action. This feedback loop links world modeling to grounding.

Embodiment drift can occur over time. Wheels wear, joints develop backlash, cameras shift, payload mechanisms loosen, and battery performance changes. A model calibrated to the original robot may gradually lose accuracy. Continual system identification and health monitoring are therefore necessary.

Grounding should include awareness of sensor and actuator health. A camera may still produce images while its lens is dirty. An encoder may drift. A motor may overheat and deliver reduced torque. The model should receive diagnostic information and adapt its behavior accordingly.

Cross-embodiment learning creates both opportunities and risks. Data from many robots can improve task diversity and generalization. Shared concepts can emerge across platforms. Yet combining incompatible trajectories without sufficient metadata can confuse the model.

A successful cross-embodiment dataset should clearly describe robot morphology, action format, control frequency, sensor configuration, payload, and task semantics. Common task labels and normalized coordinate conventions can support transfer while preserving hardware-specific information.

The model architecture may use shared and private components. A shared encoder can learn general perception and task representations, while robot-specific adapters and decoders handle embodiment. Alternatively, robot-specific encoders may map different sensors into a shared latent space.

Parameter-efficient adaptation is useful when deploying a foundation model to a new platform. Small adapters, low-rank updates, or embodiment embeddings can be trained while most of the model remains fixed. This reduces cost and helps preserve general knowledge.

However, adaptation data must cover the new robot's unique physical behavior. A few static images are insufficient to learn actuator delay, contact response, or stopping distance. Embodiment adaptation must include interaction trajectories.

Safety is deeply connected to embodiment. Physical risk depends on mass, speed, force, geometry, and proximity to people. The same planning error has different consequences on a lightweight tabletop robot and a two-ton mobile platform.

A foundation model cannot use one universal safety threshold for all bodies. Limits must be conditioned on robot type, payload, operating mode, environment, and current state. Independent safety systems should enforce these limits deterministically.

Collision geometry must reflect the complete current embodiment, including tools, payloads, extended arms, and moving parts. A robot carrying a long object occupies more space than its base footprint. Grounded planning must use the updated collision model.

Human-robot interaction introduces social grounding challenges. Personal space, approach direction, speed, gaze, and communication affect whether behavior is acceptable. These expectations vary by context, culture, workplace, and individual.

A robot may need to distinguish a person who is collaborating from one who is simply passing nearby. It may need to yield, stop, explain its intention, or request permission. Social behavior cannot be reduced to geometric collision avoidance alone.

Yet social grounding should remain conservative. Predicting human intent is uncertain, and a robot must not rely on confident but unsupported assumptions. Human motion prediction should be combined with safety zones and fail-safe behavior.

Evaluation of embodiment and grounding must go beyond conventional benchmark accuracy. A model may correctly label objects but fail to reach them. It may describe spatial relations but produce metrically incorrect actions. It may recognize a tool but misuse it physically.

Evaluation should therefore include perception-to-action consistency. The robot should identify a target, plan a feasible approach, execute within physical limits, and verify the outcome. Tests should measure task success, geometric error, force control, timing, recovery, and safety.

Cross-embodiment evaluation should examine whether knowledge transfers without unsafe assumptions. A model adapted to a new robot should be tested across changes in morphology, payload, sensors, control frequency, and workspace.

Grounding evaluation should include ambiguous instructions, occlusion, sensor degradation, dynamic scenes, unknown objects, and conflicting information sources. The system should demonstrate not only successful execution but also appropriate uncertainty and requests for clarification.

Open-loop evaluation is insufficient because grounding depends on interaction. Closed-loop simulation and real-world testing are required. Small action errors change future observations and may reveal weaknesses that are invisible in recorded datasets.

Interpretability is also important. Engineers should be able to inspect which object the model associated with an instruction, which coordinate frame it used, what embodiment constraints it considered, and why it selected an action.

Structured intermediate representations can improve observability. The model may output the selected target, estimated pose, reference frame, intended skill, predicted consequence, confidence, and detected constraints before execution.

These outputs should not be treated as perfect explanations, but they make failure analysis easier. If the robot moves incorrectly, engineers can determine whether the problem originated in language grounding, object association, geometry, planning, or control.

A layered architecture provides the most practical way to manage embodiment and grounding. Foundation models can handle semantic interpretation, multimodal fusion, task decomposition, and adaptation. Geometric modules can handle coordinate transformations, localization, collision checking, and kinematics. Control systems can handle dynamics, force, and high-frequency stability.

World models can predict action consequences, while deterministic safety components enforce physical limits. Human interfaces can resolve ambiguity and provide oversight. This division preserves the flexibility of learned models without expecting them to replace every reliable engineering method.

Digital twins can strengthen grounding by providing structured knowledge about the robot and environment. A digital twin may contain geometry, equipment identity, nominal poses, maintenance records, and operational constraints. The foundation model can connect live sensor data with this structured representation.

However, digital twins can become outdated. The model must distinguish nominal design information from current reality. Live perception should verify whether objects remain where the twin expects them to be.

Grounding is therefore not a one-time mapping from symbols to objects. It is a continuous process of aligning language, perception, body state, world knowledge, action, and feedback. The alignment must be updated whenever the robot, environment, tool, or task changes.

The long-term goal is a robot foundation model that can transfer semantic knowledge broadly while adapting its behavior precisely to each body. It should understand that many robots share concepts such as grasping, navigation, inspection, and cooperation, but that every embodiment realizes these concepts through different physical possibilities.

Achieving this goal requires more than larger models. It requires calibrated sensors, synchronized data, explicit embodiment descriptions, action-aware training, diverse interaction experience, reliable world models, uncertainty estimation, and independent safety enforcement.

Robot embodiment and grounding are not secondary implementation details. They define the boundary between intelligence that talks about the physical world and intelligence that can act within it. A model becomes physically useful only when its internal concepts correspond consistently to measurable objects, states, forces, spaces, actions, and consequences.

For Physical AI foundation models, the central challenge is therefore to preserve broad generalization while respecting the exact reality of each robot body. The system must combine universal task understanding with platform-specific control, shared concepts with calibrated geometry, and flexible reasoning with physical constraint.

A successful architecture will not treat embodiment as an obstacle to general intelligence. It will treat the body as a source of structure, experience, and meaning. Through interaction, the robot learns what concepts mean operationally. Grounding transforms abstract representations into physical competence, while embodiment determines how that competence can be expressed safely and effectively in the real world.

로봇 체화(Robot Embodiment)와 접지(Grounding)는 피지컬 AI 파운데이션 모델(Physical AI Foundation Model)을 개발하는 과정에서 가장 핵심적인 과제 가운데 하나이다. 디지털 모델(Digital Model)은 물리 법칙을 따라야 하는 몸체 없이도 언어, 이미지, 코드 등을 처리할 수 있다. 반면 로봇은 특정한 기계적 구조(Mechanical Structure)를 통해 세상을 인식하고 행동한다. 따라서 로봇의 지능은 센서(Sensor), 액추에이터(Actuator), 크기(Dimension), 질량(Mass), 운동학(Kinematics), 동역학(Dynamics), 에너지 한계(Energy Limit), 그리고 주변 환경과의 상호작용과 분리될 수 없다. 로봇 파운데이션 모델은 객체와 명령의 의미를 이해하는 것뿐 아니라, 그러한 의미가 특정한 몸체를 통해 물리 세계에서 어떻게 구현되는지도 함께 학습해야 한다.

체화(Embodiment)는 에이전트가 세상을 인식하고 행동하는 몸체를 통해 지능이 형성되는 방식을 의미한다. 몸체는 로봇이 무엇을 감지할 수 있는지, 어디까지 이동할 수 있는지, 무엇을 잡을 수 있는지, 얼마나 큰 힘을 가할 수 있는지, 얼마나 빨리 정지할 수 있는지, 그리고 상호작용 과정에서 어떤 위험이 발생하는지를 결정한다. 바퀴형 자율이동로봇(Autonomous Mobile Robot), 휴머노이드(Humanoid), 사족보행 로봇(Quadruped), 고정형 매니퓰레이터(Fixed Manipulator), 무인항공기(Unmanned Aerial Vehicle)는 모두 동일하게 "객체를 검사하라"는 명령을 받을 수 있지만, 이를 수행하는 방식은 플랫폼마다 완전히 다르다.

이동형 로봇은 대상까지 접근하여 센서를 위치시킬 수 있다. 매니퓰레이터는 엔드이펙터(End Effector)를 움직여 물체 주변을 검사할 수 있다. 휴머노이드는 걸어가고 몸을 숙인 뒤 양손을 사용할 수 있다. 무인항공기는 공중에서 대상을 검사할 수 있다. 의미적 목표(Semantic Goal)는 동일하지만, 사용 가능한 행동 공간(Action Space), 제약조건(Constraint), 물리적 전략(Physical Strategy)은 서로 다르다. 따라서 파운데이션 모델은 공통적인 작업 추상화(Task Abstraction)와 체화별 실행 방식(Embodiment-Specific Realization)을 모두 표현해야 한다.

접지(Grounding)는 기호(Symbol), 언어(Language), 개념(Concept), 또는 내부 표현(Internal Representation)을 실제 관측(Observation)과 행동(Action)에 연결하는 과정을 의미한다. "컵(cup)"이라는 단어는 현재 장면 속 실제 컵과 연결되어야 한다. "조금 더 가까이 이동하라(move closer)"는 명령은 실제 공간 거리의 변화로 연결되어야 한다. "부드럽게 밀어라(push gently)"는 문장은 적절한 방향(Direction), 속도(Velocity), 접촉력(Contact Force), 지속 시간(Duration)으로 변환되어야 한다. 이러한 접지가 이루어지지 않으면 모델은 유창한 설명을 생성할 수는 있어도 실제 물리 세계에서는 신뢰성 있게 행동할 수 없다.

체화와 접지는 서로 밀접하게 연결되어 있다. 접지는 항상 로봇의 몸체와 센서를 통해 이루어진다. 로봇이 실제로 이해할 수 있는 것은 무엇을 감지할 수 있으며 어떻게 행동할 수 있는가에 의해 결정된다. 촉각 센서를 가진 로봇 손은 변형(Deformation)과 힘의 반응을 통해 부드러움(Softness)의 개념을 접지할 수 있다. 반면 촉각 센서가 없는 이동형 로봇은 시각 정보나 기존 지식에만 의존해야 한다. 깊이 센서를 가진 로봇은 거리를 계량적으로 접지할 수 있지만, 단안 카메라(Monocular Camera)만 가진 시스템은 이를 간접적으로 추정해야 한다.

체화에서 가장 먼저 나타나는 도전 과제는 센서 다양성(Sensor Diversity)이다. 서로 다른 로봇 플랫폼은 카메라(Camera), LiDAR, 깊이 센서(Depth Sensor), 관성측정장치(IMU), 힘 센서(Force Sensor), 엔코더(Encoder), 촉각 배열(Tactile Array), 마이크(Microphone), 위치 추정 장치(Localization Device)를 서로 다르게 사용한다. 동일한 작업을 수행하는 로봇이라도 센서 구성은 크게 다를 수 있다. 시야(Field of View), 해상도(Resolution), 샘플링 속도(Sampling Rate), 노이즈 특성(Noise Characteristic), 좌표계(Coordinate Frame)는 모두 달라질 수 있다.

하나의 센서 구성에서 학습된 파운데이션 모델은 다른 센서 구성으로 잘 일반화되지 않을 수 있다. 바닥 가까이에 설치된 카메라는 휴머노이드 머리에 설치된 카메라와 전혀 다른 시점을 가진다. 손목 카메라(Wrist Camera)는 매니퓰레이터와 함께 움직이며 지속적으로 변화하는 국부 시점을 제공한다. 천장에 설치된 고정 카메라는 전체 장면을 볼 수 있지만 가림(Occlusion)이 발생하기 쉽다. 따라서 모델은 시각적 외형이 센서 위치와 방향에 따라 달라진다는 사실을 이해해야 한다.

센서 식별 정보(Sensor Identity)와 보정 정보(Calibration Information)는 명시적으로 인코딩될 필요가 있다. 카메라 내부 파라미터(Camera Intrinsics), 외부 파라미터(Extrinsics), 시야각(Field of View), 장착 자세(Mounting Pose), 시간 특성(Timing Characteristic), 왜곡 계수(Distortion Parameter)는 관측을 올바르게 해석하는 데 직접적인 영향을 준다. 모든 이미지를 동일한 것으로 취급하면 모델은 시점 변화(Perspective Variation)를 객체 변화(Object Variation)로 오해할 수 있다.

좌표계(Coordinate Frame)는 또 다른 중요한 접지 문제를 만든다. 로봇 시스템은 센서 좌표계, 로봇 베이스 좌표계(Robot Base Frame), 엔드이펙터 좌표계, 지역 지도(Local Map), 전역 지도(Global Map), 월드 좌표계(World Frame) 등 여러 좌표계를 동시에 사용한다. 카메라 정면에 있는 점이 로봇 기준으로는 왼쪽에 있을 수도 있다. 지도 좌표계(Map Coordinate)로 표현된 목표 위치는 제어기가 사용하기 전에 적절히 변환되어야 한다.

파운데이션 모델은 기하학적으로 일관된 입력을 제공받거나 좌표계 간 관계(Frame Relationship)를 명시적으로 학습해야 한다. 심각한 보정 오류(Calibration Error)를 단순히 모델 규모가 커지면 자동으로 극복될 문제로 생각해서는 안 된다. 잘못된 좌표 변환은 체계적인 공간 오차(Systematic Spatial Error)를 발생시키며, 이는 직접적으로 위험한 행동으로 이어질 수 있다.

시간 정렬(Temporal Alignment)도 마찬가지로 중요하다. 관측, 상태 추정(State Estimate), 행동은 동일한 물리적 시점을 기준으로 해야 한다. 카메라는 초당 30프레임으로 동작할 수 있지만, 관절 엔코더는 수백 Hz, 힘 센서는 그보다 더 높은 속도로 동작하며, 언어 명령은 간헐적으로만 입력될 수 있다. 통신 지연, 버퍼링, 처리 시간 때문에 데이터 스트림이 서로 어긋날 수 있다.

영상과 수백 밀리초 뒤의 로봇 상태가 함께 학습되면 모델은 외형과 행동 사이의 잘못된 관계를 학습하게 된다. 이러한 문제는 빠른 이동, 접촉, 장애물 회피 상황에서 더욱 심각해진다. 따라서 하드웨어 타임스탬프(Hardware Timestamp), 시계 동기화(Clock Synchronization), 지연 시간 보정(Latency Calibration), 시간 인식 임베딩(Time-Aware Embedding)은 접지 학습의 필수 요소이다.

또 다른 주요 과제는 형태(Morphology)이다. 로봇은 관절 수(Number of Joints), 관절 배열(Joint Arrangement), 링크 길이(Link Length), 휠 구성(Wheel Configuration), 조향 메커니즘(Steering Mechanism), 다리 구조(Limb Structure), 적재 능력(Payload), 작업 공간(Workspace)이 모두 다르다. 6축 산업용 로봇에서 학습한 정책은 운동학 구조(Kinematic Structure)의 차이를 고려하지 않으면 7축 협동로봇(Collaborative Robot)에 직접 적용할 수 없다.

형태는 인식과 제어 모두에 영향을 준다. 로봇 자신의 몸체가 카메라 시야를 가릴 수 있다. 매니퓰레이터는 특정 자세에서만 물체에 도달할 수 있다. 대형 이동형 로봇은 회전과 정지에 더 넓은 공간이 필요하다. 사족보행 로봇은 발을 움직이는 동안 균형을 유지해야 한다. 비행 로봇은 자세와 추력을 지속적으로 제어해야 한다.

하나의 접근 방법은 형태를 구조화된 토큰(Structured Token)이나 그래프(Graph)로 표현하는 것이다. 각 관절, 링크, 센서, 액추에이터를 유형(Type), 위치(Position), 방향(Orientation), 한계(Limit), 연결 관계(Connectivity)와 함께 표현할 수 있다. 그러면 모델은 단순한 숫자 벡터가 아니라 구조화된 몸체를 대상으로 추론할 수 있다.

그래프 표현(Graph Representation)은 로봇의 몸체가 자연스럽게 운동학 트리(Kinematic Tree) 또는 일반적인 기계 그래프(Mechanical Graph)를 이루기 때문에 특히 유용하다. 메시지 전달(Message Passing)이나 어텐션(Attention)은 서로 연결된 구성 요소 사이의 관계를 강조할 수 있다. 이를 통해 하나의 공유 모델이 구조적 차이를 유지하면서 다양한 형태의 로봇을 처리할 수 있다.

그러나 몸체 설명만으로는 전이가 보장되지 않는다. 모델은 형태가 갖는 기능적 의미(Functional Significance)도 이해해야 한다. 수치적으로 비슷한 관절 한계를 가진 두 개의 관절이라도 위치에 따라 역할은 완전히 다를 수 있다. 어깨 관절(Shoulder Joint), 조향 관절(Steering Joint), 그리퍼 손가락(Gripper Finger), 카메라 짐벌(Camera Gimbal)은 모두 회전형 액추에이터이지만 서로 동일하게 취급해서는 안 된다.

액추에이터 특성(Actuator Characteristic)은 체화를 더욱 복잡하게 만든다. 모터는 토크(Torque), 속도(Speed), 백래시(Backlash), 순응성(Compliance), 마찰(Friction), 제어 대역폭(Control Bandwidth), 열 한계(Thermal Limit)가 모두 다르다. 유압 시스템(Hydraulic System)은 전기 모터와 다르게 동작한다. 직렬 탄성 액추에이터(Series Elastic Actuator)는 강체 산업용 관절과 다른 특성을 가진다. 한 로봇에서는 안전한 명령이 다른 로봇에서는 진동(Oscillation), 과열(Overheating), 과도한 힘을 유발할 수 있다.

피지컬 AI 파운데이션 모델은 원하는 행동이 항상 정확한 움직임으로 변환된다고 가정해서는 안 된다. 액추에이터 지연(Actuator Delay), 포화(Saturation), 데드존(Dead Zone), 백래시, 기계적 마모(Mechanical Wear)는 실제 실행에 영향을 준다. 이러한 요소들은 시스템 식별(System Identification), 학습 기반 동역학(Learned Dynamics), 또는 명시적 하드웨어 메타데이터(Hardware Metadata)를 통해 표현될 필요가 있다.

질량(Mass)과 관성(Inertia)도 매우 중요하다. 대형 적재물을 운반하는 중량 검사 로봇은 소형 교육용 로봇처럼 반응할 수 없다. 정지 거리, 전복 위험(Tipping Risk), 에너지 소비, 충돌 결과는 훨씬 크다. 모델은 동일한 속도 명령이라도 총질량(Total Mass), 무게 중심(Center of Gravity), 바닥 상태(Floor Condition), 적재물 분포(Payload Distribution)에 따라 물리적 의미가 달라진다는 사실을 이해해야 한다.

적재물 변화(Payload Change)는 동적인 체화(Dynamic Embodiment)를 만든다. 로봇의 몸체는 항상 동일하지 않다. 무거운 물체를 운반하는 이동형 매니퓰레이터는 빈 상태와 다른 가속도와 안정성 한계를 가진다. 긴 공구를 장착한 로봇 팔은 유효 작업 반경과 충돌 영역(Collision Envelope)이 달라진다. 적재물을 운반하는 UAV는 비행 시간이 감소하고 동역학도 변한다.

따라서 모델은 현재 적재 상태, 공구 구성(Tool Configuration), 무게 중심, 배터리 상태(Battery State) 등 변화하는 체화 정보를 표현해야 한다. 정적인 로봇 식별 정보만으로는 충분하지 않다. 체화는 설계에 의해 일부 결정되지만 현재 운용 상태에 따라서도 지속적으로 변화한다.

공구 사용(Tool Use)은 몸체를 더욱 확장한다. 로봇이 드라이버(Screwdriver), 검사 프로브(Inspection Probe), 흡착 장치(Suction Device), 카메라 붐(Camera Boom)을 장착하면 공구는 로봇의 유효한 몸체 일부가 된다. 모델은 공구의 형상(Geometry), 접촉 특성(Contact Property), 운용 제약, 로봇 움직임과의 관계를 이해해야 한다.

공구 접지(Tool Grounding)는 단순히 공구 종류를 인식하는 것 이상을 요구한다. 로봇은 실제 기능이 수행되는 끝단(Tool Tip)이 어디인지, 어느 방향으로 힘을 가해야 하는지, 어떤 결과가 성공적인 사용을 의미하는지를 알아야 한다. 따라서 공구 보정(Tool Calibration)과 공구 중심점(Tool Center Point) 정의는 매우 중요하다.

행동 공간(Action Space)의 차이는 범용 로봇 모델이 해결해야 할 가장 어려운 문제 가운데 하나이다. 언어 모델은 공통된 토큰 공간(Token Space)에서 동작하지만, 로봇은 매우 다양한 형식의 행동을 생성한다. 어떤 로봇은 관절 위치(Joint Position)를 사용하고, 다른 로봇은 관절 속도(Joint Velocity), 카르테시안 자세(Cartesian Pose), 또는 이산 내비게이션 명령을 사용할 수 있다.

공통 로봇 파운데이션 모델은 이러한 이질적인 행동을 표현하는 전략이 필요하다. 하나의 방법은 이동(Move), 회전(Rotate), 접근(Reach), 파지(Grasp), 놓기(Release), 힘 적용(Apply Force)과 같은 의도를 표현하는 공유 잠재 행동 공간(Shared Latent Action Space)으로 변환하는 것이다. 이후 로봇별 디코더가 이를 실제 플랫폼 명령으로 변환할 수 있다.

이 접근법은 전이를 향상시키지만 중요한 세부사항을 숨길 수도 있다. "앞으로 이동(move forward)"이라는 개념은 차동 구동(Differential Drive), 전방향 플랫폼(Omnidirectional Platform), 사족보행 로봇, 휴머노이드에서 모두 다른 의미를 가진다. 따라서 공유 잠재 표현은 플랫폼별 실행에 필요한 충분한 정보를 유지하면서도 지나치게 세분화된 표현이 되지 않아야 한다.

또 다른 방법은 명시적인 행동 스키마(Action Schema)를 모델 입력으로 제공하는 것이다. 모델은 사용 가능한 제어 방식(Control), 차원(Dimension), 단위(Unit), 제한값(Limit), 갱신 주기(Update Rate)를 입력받고, 그 스키마 안에서 행동을 생성한다. 이는 모델이 서로 다른 인터페이스를 정의에 따라 호출하는 도구 사용(Tool Use)과 유사하다.

행동 정규화(Action Normalization)도 도움이 될 수 있다. 관절 위치, 속도, 힘을 정규화된 범위로 변환하고 좌표계를 표준화할 수 있다. 그러나 정규화가 동역학이나 위험의 차이를 제거하는 것은 아니다. 동일한 정규화 값이라도 플랫폼에 따라 전혀 다른 물리적 결과를 만든다.

언어를 행동으로 접지하는 것은 특히 어렵다. 자연어 명령은 본질적으로 모호하기 때문이다. "조금 더 가까이"는 정확한 거리를 지정하지 않는다. "단단히 잡아라(grip firmly)"는 물체의 취약성에 따라 달라진다. "빨리"라는 표현은 안전이나 적재 안정성과 충돌할 수 있다. 인간의 언어는 공유된 문맥과 상식(Common Sense)에 크게 의존한다.

로봇은 현재 관측, 작업 문맥, 사용자 이력, 운영 규칙을 이용하여 이러한 모호성을 해결해야 한다. 불확실성이 높을 경우 임의로 해석하기보다 사용자에게 명확한 설명을 요청해야 한다. 따라서 접지된 언어 이해(Grounded Language Understanding)는 불확실성 추정(Uncertainty Estimation)과 상호작용적 의사소통(Interactive Communication)을 함께 요구한다.

공간 언어(Spatial Language)는 또 다른 어려움을 만든다. 왼쪽, 오른쪽, 뒤, 위, 가까움, 멀리와 같은 표현은 기준 좌표계(Reference Frame)에 따라 의미가 달라진다. "기계의 왼쪽으로 이동하라"는 명령은 작업자의 왼쪽일 수도 있고, 로봇의 왼쪽일 수도 있으며, 기계 자체의 왼쪽일 수도 있다. 시스템은 의도된 기준 좌표계를 식별하거나 사용자에게 확인해야 한다.

참조 해석(Reference Resolution)도 관련된 문제이다. "저것을 집어라(pick that one)"라는 명령은 화자의 시선(Gaze), 손가락 가리키기(Pointing), 대화 문맥(Dialogue Context), 이전 상호작용을 통해 어느 물체를 의미하는지를 판단해야 한다. 인간-로봇 협업(Human-Robot Collaboration)에서는 언어 접지가 비전(Vision), 제스처 인식(Gesture Recognition), 대화 메모리(Conversational Memory)를 함께 요구한다.

어포던스 접지(Affordance Grounding)는 객체가 어떤 행동을 지원하는지를 이해하는 문제이다. 손잡이는 당기기를, 버튼은 누르기를, 평평한 면은 물체를 올려놓기를 가능하게 한다. 이러한 관계는 외형만으로 결정되지 않는다. 겉모습이 비슷한 손잡이라도 단순 장식일 수 있으며, 평평해 보이는 면도 실제로는 무게를 지탱하지 못할 수 있다.

파운데이션 모델은 이미지, 비디오, 텍스트를 통해 광범위한 어포던스 사전지식(Affordance Prior)을 학습할 수 있지만, 신뢰할 수 있는 접지를 위해서는 실제 물리적 상호작용이 필요하다. 힘, 움직임, 소리, 촉각 피드백은 예상한 어포던스가 실제인지 여부를 알려준다. 능동적 인식(Active Perception)과 탐색 행동(Exploratory Action)은 이러한 불확실성을 줄일 수 있다.

접촉 접지(Contact Grounding)는 특히 어렵다. 접촉 전에는 비전과 기하학을 중심으로 추론하지만, 접촉 이후에는 힘, 토크(Torque), 촉각, 모터 전류, 변형이 핵심 정보가 된다. 모델은 자유 공간 이동과 실제 접촉 사이의 전환을 인식해야 한다.

접촉 이벤트(Contact Event)는 짧고 고주파이며 시간 동기화에 매우 민감하다. 작은 지연도 힘이 위험한 수준까지 증가하도록 만들 수 있다. 따라서 대형 범용 모델은 직접적인 고대역 접촉 제어에 적합하지 않다. 계층형 아키텍처(Hierarchical Architecture)는 접촉 전략은 파운데이션 모델이 담당하고, 힘 제어와 임피던스 제어(Impedance Control)는 하위 제어기가 수행하는 것이 바람직하다.

객체 영속성(Object Permanence)도 접지의 중요한 요소이다. 객체는 카메라 시야에서 가려지거나 사라질 수 있으며, 조명에 따라 외형이 변할 수도 있다. 로봇은 객체가 계속 존재하며 어디에 있을 가능성이 높은지를 기억해야 한다. 이를 위해서는 프레임 단위 인식(Frame-by-Frame Recognition)이 아니라 메모리와 월드 모델이 필요하다.

동적인 환경(Dynamic Environment)은 접지를 더욱 어렵게 만든다. 사람은 이동하고, 문은 열리고, 팔레트는 옮겨지며, 기계의 상태는 지속적으로 변한다. 한 시점에서 접지된 표현은 곧 오래된 정보가 될 수 있다. 시스템은 정보를 지속적으로 갱신하고 각 관측이 언제 수집되었는지를 알아야 한다.

최신성 표시(Freshness Indicator)와 타임스탬프는 이를 지원한다. 모델은 어제 생성된 지도와 현재 LiDAR 관측을 동일하게 취급해서는 안 된다. 정적인 지식과 실시간 센서 정보를 구분해야 하며, 둘이 충돌하면 어느 쪽이 더 신뢰할 수 있는지를 판단해야 한다.

부분 관측성(Partial Observability)은 피할 수 없다. 센서는 모든 것을 볼 수 없으며, 일부 중요한 변수는 직접 관측할 수 없다. 표면의 마찰계수, 물체 내부 손상, 사람의 의도는 직접 측정하기 어렵다. 로봇은 존재하지 않는 정확한 값을 만들어내기보다 불확실성을 유지해야 한다.

신념 상태(Belief State)는 알고 있는 것과 모르는 것을 함께 표현할 수 있다. 모델은 객체 자세, 접촉 상태, 인간의 의도에 대해 확률 분포를 유지할 수 있다. 이후 계획은 이러한 불확실성을 고려하여 필요하면 먼저 정보를 수집하는 행동을 선택해야 한다.

따라서 능동적 인식(Active Perception)은 매우 중요한 접지 능력이다. 로봇은 카메라를 움직이거나, 시점을 변경하거나, 물체를 만지거나, 사람에게 질문하여 불확실성을 줄일 수 있다. 인식은 더 이상 수동적인 입력 과정이 아니라 행동 선택(Action Selection)의 일부가 된다.

로봇은 장애물 뒤를 보기 위해 위치를 바꾸거나, 다른 면을 검사하기 위해 물체를 회전시키거나, 깊이 추정을 향상시키기 위해 천천히 접근할 수 있다. 파운데이션 모델은 즉시 행동을 실행하는 것보다 추가적인 센싱이 더 가치 있는 시점을 학습해야 한다.

시뮬레이션(Simulation)은 체화와 접지를 학습하기 위한 확장 가능한 환경을 제공한다. 로봇 모델, 센서, 동역학, 객체, 환경을 체계적으로 변화시킬 수 있으며, 하드웨어를 손상시키지 않고 수백만 번의 상호작용을 생성할 수 있다. 드문 실패나 위험한 상황도 안전하게 탐색할 수 있다.

그러나 시뮬레이션에는 현실 격차(Reality Gap)가 존재한다. 시뮬레이션 영상, 마찰, 접촉, 액추에이터 지연, 변형, 센서 노이즈는 실제 세계와 다르다. 정책은 실제에는 존재하지 않는 시뮬레이션 특성을 이용할 수도 있다. 시각적 사실감만으로는 정확한 동역학을 보장하지 않는다.

도메인 랜덤화(Domain Randomization)는 텍스처(Texture), 조명, 물체 특성, 동역학, 센서 노이즈, 지연 시간을 다양하게 변화시켜 이 문제를 완화하려 한다. 모델은 하나의 특정 시뮬레이션 조건에 의존하지 않도록 학습한다. 그러나 지나친 랜덤화는 비현실적인 데이터를 생성하고 학습을 불필요하게 어렵게 만들 수도 있다.

시스템 식별(System Identification)은 실제 로봇의 파라미터를 추정하여 시뮬레이션의 정확도를 높일 수 있다. 기록된 궤적을 이용하여 마찰, 모터 응답, 질량, 지연 모델을 보정할 수 있다. 그럼에도 실제 환경에서의 미세조정(Fine-Tuning)과 검증은 여전히 필요하다.

실제 로봇 데이터는 비용이 매우 크다. 하드웨어 사용 시간, 유지보수, 작업자, 안전 감독, 보정, 저장 공간이 필요하다. 물리적 상호작용은 텍스트나 이미지를 처리하는 것보다 훨씬 느리며, 실패가 발생하면 데이터 수집 자체가 중단될 수도 있다. 따라서 단순한 규모 확장이 어렵다.

원격 조작(Teleoperation)은 접지된 로봇 시연을 수집하는 대표적인 방법이다. 작업자가 로봇을 직접 조작하면서 관측과 행동을 기록한다. 이는 성공적인 행동을 제공하지만 작업자의 습관, 지연, 일관성 부족도 함께 포함된다. 동일한 작업이라도 작업자마다 다른 전략을 사용할 수 있다.

시연 데이터에는 체화 메타데이터(Embodiment Metadata)가 함께 기록되어야 한다. 로봇 구성(Robot Configuration), 공구, 적재 상태, 제어 모드(Control Mode), 보정 정보, 환경 정보가 모두 저장되어야 한다. 그렇지 않으면 서로 다른 시스템의 궤적을 올바르게 결합할 수 없다.

자율 데이터 수집(Autonomous Data Collection)은 규모를 확대할 수 있다. 로봇은 반복 작업을 수행하고 다양한 상황을 탐색하며 실패 사례를 스스로 수집할 수 있다. 그러나 자율 수집은 현재 정책의 편향을 강화할 수도 있다. 로봇이 어려운 상황을 회피하면 실제로 필요한 어려운 사례가 데이터셋에 포함되지 않게 된다.

실패와 복구 데이터(Failure and Recovery Data)는 접지에서 특히 중요하다. 성공적인 파지만 본 모델은 미끄러진 물체를 인식하지 못할 수 있다. 깨끗한 경로에서만 학습한 로봇은 장애물이 있는 환경에서 실패할 수 있다. 진정한 접지 지능은 예상이 어긋나는 상황과 수정 행동을 모두 경험해야 한다.

모델은 예상 결과와 실제 결과를 비교해야 한다. 물체를 이동시키려 했지만 움직이지 않았다면, 모델은 물체, 공구, 접촉, 행동에 대한 자신의 믿음을 수정해야 한다. 이러한 피드백 루프(Feedback Loop)는 월드 모델과 접지를 연결한다.

시간이 지나면서 체화 드리프트(Embodiment Drift)가 발생할 수도 있다. 바퀴가 마모되고, 관절에 백래시가 생기며, 카메라 위치가 변하고, 적재 장치가 느슨해지고, 배터리 성능이 저하된다. 초기 상태를 기준으로 보정된 모델은 점차 정확성을 잃게 된다. 따라서 지속적인 시스템 식별과 상태 모니터링이 필요하다.

접지는 센서와 액추에이터의 상태도 인식해야 한다. 카메라는 렌즈가 더러워져도 계속 영상을 출력할 수 있다. 엔코더는 드리프트할 수 있으며, 모터는 과열되어 토크가 감소할 수 있다. 모델은 이러한 진단 정보를 받아 행동을 적절히 조정해야 한다.

교차 체화 학습(Cross-Embodiment Learning)은 기회와 위험을 동시에 제공한다. 다양한 로봇에서 수집한 데이터는 작업 다양성과 일반화를 향상시킨다. 플랫폼을 초월하는 공통 개념도 학습될 수 있다. 그러나 충분한 메타데이터 없이 서로 다른 궤적을 결합하면 모델은 오히려 혼란스러워질 수 있다.

성공적인 교차 체화 데이터셋은 로봇 형태, 행동 형식, 제어 주기, 센서 구성, 적재 상태, 작업 의미를 명확히 기술해야 한다. 공통 작업 레이블(Common Task Label)과 정규화된 좌표계 규약은 하드웨어 특성을 유지하면서도 전이를 지원할 수 있다.

모델 아키텍처는 공유 구성 요소와 전용 구성 요소를 함께 사용할 수 있다. 공유 인코더(Shared Encoder)는 일반적인 인식과 작업 표현을 학습하고, 로봇별 어댑터(Adapter)와 디코더가 체화 특성을 처리한다. 또는 로봇별 인코더가 서로 다른 센서를 하나의 공유 잠재 공간으로 변환할 수도 있다.

파라미터 효율적인 적응(Parameter-Efficient Adaptation)은 새로운 플랫폼에 파운데이션 모델을 적용할 때 매우 유용하다. 작은 어댑터, 저랭크 업데이트(Low-Rank Update), 체화 임베딩(Embodiment Embedding)만 추가로 학습하고 대부분의 모델은 그대로 유지할 수 있다. 이를 통해 비용을 줄이고 기존의 일반 지식을 보존할 수 있다.

그러나 적응 데이터는 새로운 로봇의 고유한 물리적 특성을 반드시 포함해야 한다. 몇 장의 정적인 이미지로는 액추에이터 지연, 접촉 반응, 정지 거리를 학습할 수 없다. 체화 적응에는 실제 상호작용 궤적이 반드시 포함되어야 한다.

안전(Safety)은 체화와 매우 밀접한 관계를 가진다. 물리적 위험은 질량, 속도, 힘, 형상, 사람과의 거리 등에 의해 결정된다. 동일한 계획 오류라도 소형 탁상용 로봇과 2톤 이동형 플랫폼에서는 전혀 다른 결과를 초래한다.

파운데이션 모델은 모든 몸체에 동일한 안전 임계값(Safety Threshold)을 사용할 수 없다. 제한값은 로봇 종류, 적재 상태, 운용 모드, 환경, 현재 상태에 따라 달라져야 한다. 독립적인 안전 시스템이 이러한 제한을 결정론적으로 강제해야 한다.

충돌 형상(Collision Geometry)은 공구, 적재물, 펼쳐진 로봇 팔, 움직이는 부품을 모두 포함한 현재의 체화를 반영해야 한다. 긴 물체를 운반하는 로봇은 기본 차체보다 훨씬 큰 공간을 차지한다. 접지된 계획은 항상 최신 충돌 모델을 사용해야 한다.

인간-로봇 상호작용(Human-Robot Interaction)은 사회적 접지(Social Grounding)라는 새로운 과제를 만든다. 개인 공간(Personal Space), 접근 방향, 속도, 시선(Gaze), 의사소통 방식은 행동의 적절성에 영향을 준다. 이러한 기대는 환경, 문화, 작업장, 개인에 따라 달라질 수 있다.

로봇은 협업 중인 사람과 단순히 지나가는 사람을 구분해야 할 수도 있다. 양보하거나, 정지하거나, 자신의 의도를 설명하거나, 허가를 요청해야 할 수도 있다. 사회적 행동은 단순한 기하학적 충돌 회피만으로는 설명할 수 없다.

그러나 사회적 접지는 항상 보수적으로 이루어져야 한다. 인간의 의도를 예측하는 것은 본질적으로 불확실하며, 로봇은 근거 없는 확신에 의존해서는 안 된다. 인간의 움직임 예측은 안전 구역(Safety Zone)과 실패 안전(Fail-Safe) 동작과 함께 사용되어야 한다.

체화와 접지의 평가는 단순한 벤치마크 정확도를 넘어야 한다. 모델은 객체를 정확히 인식하면서도 실제로는 도달하지 못할 수 있다. 공간 관계를 정확히 설명하면서도 물리적으로 잘못된 행동을 생성할 수도 있다. 공구를 인식하지만 실제 사용은 실패할 수도 있다.

따라서 평가는 인식과 행동의 일관성(Perception-to-Action Consistency)을 포함해야 한다. 로봇은 목표를 식별하고, 실행 가능한 접근을 계획하며, 물리적 제약 안에서 실행하고, 결과를 검증해야 한다. 작업 성공률(Task Success), 기하학적 오차(Geometric Error), 힘 제어, 시간 특성, 복구 능력, 안전성을 함께 측정해야 한다.

교차 체화 평가는 지식이 위험한 가정을 하지 않으면서 새로운 로봇으로 전이되는지를 확인해야 한다. 형태, 적재물, 센서, 제어 주기, 작업 공간이 달라졌을 때도 안전하게 동작하는지를 시험해야 한다.

접지 평가는 모호한 명령(Ambiguous Instruction), 가림, 센서 성능 저하, 동적 환경, 미지의 객체, 상충되는 정보원을 포함해야 한다. 시스템은 성공적인 실행뿐 아니라 적절한 불확실성 표현과 명확한 설명 요청도 보여주어야 한다.

개방 루프(Open-Loop) 평가만으로는 충분하지 않다. 접지는 상호작용을 통해 형성되므로 폐루프 시뮬레이션(Closed-Loop Simulation)과 실제 환경 시험이 필요하다. 작은 행동 오차도 이후 관측을 변화시키며, 기록된 데이터에서는 보이지 않는 약점을 드러낼 수 있다.

해석 가능성(Interpretability)도 중요하다. 엔지니어는 모델이 어떤 객체를 명령과 연결했는지, 어떤 좌표계를 사용했는지, 어떤 체화 제약을 고려했는지, 왜 특정 행동을 선택했는지를 확인할 수 있어야 한다.

구조화된 중간 표현(Structured Intermediate Representation)은 관측 가능성을 높일 수 있다. 모델은 실행 전에 선택된 목표, 추정 자세(Estimated Pose), 기준 좌표계, 계획된 스킬, 예상 결과, 신뢰도, 감지된 제약조건을 출력할 수 있다.

이러한 출력은 완전한 설명은 아니지만 실패 분석을 훨씬 쉽게 만든다. 로봇이 잘못 움직였다면 문제의 원인이 언어 접지, 객체 연결, 기하학, 계획, 제어 중 어디에서 발생했는지를 확인할 수 있다.

계층형 아키텍처는 체화와 접지를 관리하는 가장 실용적인 방법이다. 파운데이션 모델은 의미 해석(Semantic Interpretation), 멀티모달 융합(Multimodal Fusion), 작업 분해(Task Decomposition), 적응을 담당한다. 기하학 모듈은 좌표 변환, 위치 추정, 충돌 검사, 운동학을 담당한다. 제어 시스템은 동역학, 힘 제어, 고주파 안정성을 담당한다.

월드 모델은 행동 결과를 예측하고, 결정론적 안전 구성요소는 물리적 제한을 강제한다. 인간 인터페이스는 모호성을 해결하고 감독을 제공한다. 이러한 역할 분담은 학습 기반 모델의 유연성을 유지하면서도 기존 엔지니어링 기법을 모두 대체하려 하지 않는다.

디지털 트윈(Digital Twin)은 로봇과 환경에 대한 구조화된 지식을 제공하여 접지를 강화할 수 있다. 디지털 트윈은 형상 정보, 장비 식별, 기준 위치, 유지보수 기록, 운용 제약을 포함할 수 있다. 파운데이션 모델은 실시간 센서 데이터와 이러한 구조화된 표현을 연결할 수 있다.

그러나 디지털 트윈도 시간이 지나면 오래될 수 있다. 모델은 설계 정보와 현재 현실을 구분해야 한다. 실시간 인식을 통해 실제 객체가 디지털 트윈이 예상하는 위치에 존재하는지를 확인해야 한다.

따라서 접지는 단순히 기호를 객체에 연결하는 일회성 과정이 아니다. 언어, 인식, 몸체 상태, 세계 지식, 행동, 피드백을 지속적으로 정렬하는 연속적인 과정이다. 로봇, 환경, 공구, 작업이 변할 때마다 이러한 정렬도 함께 갱신되어야 한다.

장기적인 목표는 의미적 지식을 폭넓게 전이하면서도 각각의 몸체에 정확하게 적응하는 로봇 파운데이션 모델을 만드는 것이다. 로봇은 파지, 내비게이션, 검사, 협업과 같은 개념을 공유하지만, 모든 체화는 이를 서로 다른 물리적 방식으로 구현한다는 사실을 이해해야 한다.

이 목표를 달성하려면 단순히 모델 규모를 키우는 것만으로는 충분하지 않다. 보정된 센서, 동기화된 데이터, 명시적인 체화 표현, 행동 기반 학습, 다양한 상호작용 경험, 신뢰성 있는 월드 모델, 불확실성 추정, 독립적인 안전 보장이 함께 필요하다.

로봇의 체화와 접지는 단순한 구현 세부사항이 아니다. 그것은 물리 세계를 설명하는 지능과 실제 물리 세계에서 행동할 수 있는 지능을 구분하는 경계이다. 모델의 내부 개념이 실제 객체, 상태, 힘, 공간, 행동, 결과와 일관되게 대응될 때 비로소 물리적으로 유용한 시스템이 된다.

따라서 피지컬 AI 파운데이션 모델의 핵심 과제는 넓은 일반화 능력을 유지하면서도 각각의 로봇 몸체가 가진 정확한 현실을 존중하는 것이다. 시스템은 범용적인 작업 이해와 플랫폼별 제어, 공유된 개념과 보정된 기하학, 유연한 추론과 물리적 제약을 동시에 결합해야 한다.

성공적인 아키텍처는 체화를 일반지능의 장애물로 보지 않는다. 오히려 몸체를 구조, 경험, 의미의 원천으로 활용한다. 로봇은 상호작용을 통해 개념이 실제로 무엇을 의미하는지를 학습한다. 접지는 추상적인 표현을 실제 물리적 능력으로 변환하며, 체화는 그러한 능력이 현실 세계에서 얼마나 안전하고 효과적으로 표현될 수 있는지를 결정한다.

##  

## 1.8 Foundation Model Evaluation for Physical AI

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

Evaluating a foundation model for Physical AI is fundamentally different from evaluating a language model, image classifier, or other digital system. A digital model can often be assessed by comparing its outputs with reference answers, labels, or benchmark scores. A Physical AI model must be evaluated through the complete chain from perception to reasoning, planning, control, and physical consequence. A response that appears correct in text may still produce an unsafe or ineffective robot action. Evaluation must therefore determine not only whether the model understands a task, but also whether it can execute the task reliably through a particular body in a dynamic environment.

The purpose of evaluation is broader than ranking models. It is used to identify capability boundaries, measure generalization, detect unsafe behavior, compare architectures, support engineering decisions, define deployment conditions, and verify that model updates do not degrade previously validated performance. For Physical AI, evaluation is part of the safety and lifecycle-management process rather than a final test conducted only after training.

A comprehensive evaluation framework must examine the model at several levels. It should measure individual perception and reasoning abilities, integrated task performance, closed-loop behavior, robustness to environmental variation, uncertainty awareness, recovery from failure, computational efficiency, and compliance with safety constraints. These levels are related but not interchangeable. Strong performance in one area does not guarantee strong performance in another.

The first evaluation level concerns representation and perception. A Physical AI foundation model must interpret information from cameras, depth sensors, LiDAR, force sensors, proprioception, maps, language, and other modalities. Evaluation should determine whether the model extracts meaningful and geometrically consistent representations from these inputs.

Visual evaluation may include object recognition, detection, segmentation, pose estimation, depth understanding, scene classification, anomaly detection, and tracking. For manipulation, the model may need to identify graspable regions, contact surfaces, handles, tools, and affordances. For navigation, it may need to recognize free space, obstacles, traversable terrain, doors, elevators, people, and traffic rules.

However, conventional visual accuracy is not enough. The representation must support action. A robot may classify an object correctly while estimating its position inaccurately. It may identify a handle but fail to determine whether it is reachable. Perception evaluation should therefore include metric geometry, target association, coordinate-frame consistency, and task relevance.

Cross-modal grounding must also be evaluated. The model should correctly connect language instructions to objects, regions, actions, and constraints in the current environment. When instructed to inspect the upper-right connector, it must select the correct connector rather than merely recognize the words. When told to move behind a machine, it must interpret the intended spatial reference frame.

Grounding tests should include ambiguous language, referring expressions, gestures, multiple similar objects, changing viewpoints, and incomplete information. The system should not only succeed when the instruction is clear but also request clarification when the available evidence is insufficient.

Temporal perception requires separate evaluation. The model should track objects through occlusion, recognize movement, detect contact transitions, estimate task progress, and distinguish old information from current observations. A static-image benchmark cannot measure whether a robot understands how a scene is changing over time.

The second evaluation level concerns embodiment awareness. The same foundation model may be expected to operate different robots, tools, payloads, and sensor configurations. Evaluation must determine whether the model understands the capabilities and limitations of the current body.

Embodiment tests may vary robot dimensions, joint limits, wheel configurations, payload, tool length, camera placement, actuator speed, and control frequency. The model should adapt its plan rather than applying an identical behavior to every platform. A trajectory suitable for a lightweight mobile robot may be inappropriate for a heavy autonomous platform. A grasp that is reachable for one manipulator may be impossible for another.

Cross-embodiment evaluation should separate task understanding from execution. The model may correctly understand the goal but fail to map it to the current robot. Tests should therefore examine whether high-level knowledge transfers while physical constraints remain respected.

Tool use should be evaluated as a change in embodiment. When a robot attaches a probe, gripper, camera mast, or other tool, the effective geometry and control interface change. The model should update its reach, collision envelope, tool center point, and task strategy accordingly.

The third evaluation level concerns instruction following and task decomposition. A Physical AI model must convert goals into actions while preserving constraints. Evaluation should include single-step commands, multi-condition instructions, long-horizon missions, and tasks that require changing behavior when environmental or internal conditions change.

A command such as "inspect stations one through five, avoid the restricted zone, recharge if the battery falls below thirty percent, and resume from the last completed station" requires memory, planning, conditional reasoning, and mission-state tracking. Success should be measured not only by final completion but also by whether every condition was respected.

Task decomposition evaluation should examine whether the model identifies meaningful subgoals, orders them correctly, and selects appropriate skills or tools. An effective model should break a complex mission into executable stages rather than generate a vague or physically impossible plan.

The quality of intermediate plans matters. A robot may eventually complete a task through inefficient, unstable, or risky behavior. Evaluation should therefore consider path length, number of unnecessary actions, energy consumption, execution time, and intervention frequency.

The fourth evaluation level concerns action quality. Robot actions may be expressed as discrete tokens, continuous commands, waypoints, trajectories, skill calls, or action chunks. Evaluation must match the output representation and the physical consequences of execution.

For manipulation, relevant measures include position error, orientation error, grasp success, contact force, slip, trajectory smoothness, and object damage. For mobile robots, measures may include path tracking, localization accuracy, stopping distance, clearance, velocity smoothness, and collision frequency. For inspection robots, image quality, target coverage, sensor viewpoint, measurement repeatability, and defect-detection performance may be important.

Open-loop action prediction can compare model outputs with expert demonstrations. Metrics such as action error or sequence likelihood are useful during development, but they are not sufficient for deployment decisions. Two different actions may both be valid, and small prediction errors can create large behavioral differences after repeated execution.

Closed-loop evaluation is therefore essential. The model must operate in simulation or on a real robot, where its actions influence future observations. This reveals error accumulation, exposure bias, unstable feedback, and recovery behavior that cannot be observed in static datasets.

The fifth evaluation level concerns world modeling and consequence prediction. A capable Physical AI foundation model should estimate how the environment will change after an action. Evaluation should measure the accuracy of predicted states, object motion, contact events, collisions, task progress, and uncertainty.

World-model evaluation should include both passive prediction and action-conditioned prediction. Predicting what usually happens next is different from predicting what will happen if the robot deliberately applies a particular action. The latter is more important for planning.

Prediction horizons should be evaluated separately. Short-term predictions may be accurate while long-term rollouts drift rapidly. Metrics should measure how error grows over time and whether the model recognizes increasing uncertainty.

A useful world model must preserve control-relevant information rather than merely reconstruct visually plausible scenes. It should accurately represent object location, free space, support relationships, contact, and other variables required for action selection. Photorealistic output with incorrect geometry is not sufficient.

Counterfactual evaluation can test whether the model distinguishes alternative actions. Given several possible trajectories, the model should predict different outcomes and identify which action is safer or more effective. This ability supports planning and risk assessment.

The sixth evaluation level concerns planning. Planning should be assessed for feasibility, completeness, efficiency, adaptability, and safety. A generated plan must satisfy robot kinematics, environmental geometry, resource limits, and mission rules.

Planning benchmarks should include tasks that require multiple steps, temporary goals, tool use, obstacle avoidance, and recovery. The environment should change during execution so that the model must revise the plan rather than follow a fixed script.

Long-horizon evaluation is particularly important. Many models perform well on short tasks but lose coherence over extended missions. They may forget completed steps, repeat actions, ignore later constraints, or fail to resume after interruption.

Mission-level measures can include completion rate, number of correctly completed subgoals, repeated or omitted actions, replanning quality, time to recovery, battery management, and human intervention. Evaluation should verify that the model maintains a consistent mission state.

The seventh evaluation level concerns generalization. A foundation model is expected to perform beyond the exact examples in its training data. Evaluation must therefore include variations in objects, scenes, viewpoints, tasks, language, robot bodies, and physical conditions.

Compositional generalization can be tested by combining familiar elements in unfamiliar ways. The robot may have seen red objects, storage bins, and upper shelves separately but not the exact instruction to place a red object in an upper bin. Successful execution indicates that the model can recombine learned concepts.

Domain generalization should test new facilities, floor materials, lighting conditions, layouts, and traffic patterns. The goal is not to create minor cosmetic changes but to determine whether the model has learned transferable structure rather than memorized one site.

Cross-embodiment generalization should test whether task knowledge transfers to robots with different kinematics or sensors. Sim-to-real evaluation should measure how well behavior learned in simulation transfers to physical hardware.

Generalization claims must be bounded. A model should not be described as general merely because it succeeds on a small collection of related tasks. The evaluated distribution, novelty level, and operational conditions should be documented clearly.

The eighth evaluation level concerns robustness. Robots operate under noise, sensor degradation, communication delay, calibration error, hardware wear, and unpredictable environmental change. A robust model should degrade gradually and predictably rather than fail suddenly.

Robustness tests may introduce image blur, glare, darkness, occlusion, missing depth, sparse LiDAR, delayed state estimates, inaccurate maps, force-sensor noise, packet loss, and localization drift. The model should maintain safe behavior even when task performance declines.

Sensor dropout is particularly important in multimodal systems. A model may become overly dependent on one modality and fail when that sensor becomes unavailable. Evaluation should remove or corrupt individual modalities to determine whether the system can use redundancy effectively.

Calibration robustness should also be measured, but calibration errors should not be normalized as acceptable operating conditions. Small realistic deviations can be tested to determine tolerance, while large errors should trigger fault detection rather than silent adaptation.

Adversarial robustness has a physical dimension. Unusual markings, reflections, confusing objects, malicious instructions, or manipulated environmental signals may cause incorrect behavior. Evaluation should examine both accidental and intentional sources of confusion.

The ninth evaluation level concerns failure detection and recovery. Physical tasks inevitably produce unexpected outcomes. Evaluation should determine whether the model recognizes failure, diagnoses likely causes, and selects an appropriate recovery strategy.

Failure scenarios may include missed grasps, dropped objects, blocked paths, wheel slip, localization loss, tool misalignment, sensor failure, unexpected human entry, low battery, or communication loss. The robot should not continue as if the task were proceeding normally.

Detection latency matters. Recognizing a failed grasp several seconds later may be too late. The system should compare predicted and observed outcomes continuously and respond within a safe time.

Recovery quality can be measured by success after retry, number of attempts, time lost, additional risk, and need for human assistance. Repeating the same failed action without modification should not be considered intelligent recovery.

The model should also know when not to recover autonomously. Some failures require stopping, escalation, maintenance, or human judgment. Evaluation should verify that the robot does not attempt unsafe self-repair or repeated aggressive retries.

The tenth evaluation level concerns uncertainty and calibration. A foundation model should express lower confidence when inputs are ambiguous, sensors are unreliable, or the situation differs from the training distribution. High confidence on incorrect actions is particularly dangerous.

Calibration evaluation compares predicted confidence with actual success frequency. A model that reports eighty percent confidence should succeed approximately eighty percent of the time under comparable conditions. Poorly calibrated confidence can cause inappropriate autonomy.

Uncertainty evaluation should distinguish known ambiguity from unknown situations. Multiple valid paths may create uncertainty without indicating danger, while unfamiliar hardware behavior may represent a serious out-of-distribution condition.

Selective prediction is useful for evaluation. The model is allowed to abstain or request help when confidence is low. Performance can then be measured as a relationship between coverage and risk. A practical system may achieve higher reliability by autonomously handling only the cases it understands well.

The operational response to uncertainty must also be tested. The robot may slow down, collect another observation, change viewpoint, invoke a specialized planner, ask a person, or enter a safe state. Accurate uncertainty estimates are not useful if they do not influence behavior.

The eleventh evaluation level concerns safety. Safety testing must examine both learned behavior and the independent systems that constrain it. The foundation model should respect human intent and operational rules, but deterministic safety mechanisms should enforce critical limits.

Tests should include collision avoidance, joint and velocity limits, force limits, stopping behavior, restricted zones, human proximity, unstable payloads, and emergency conditions. The model should not be evaluated only in normal operation.

Safety intervention metrics can reveal how often the runtime monitor rejects or modifies model outputs. Frequent intervention may indicate that the model is not safe enough even if accidents are prevented by the external layer.

Near-miss analysis is also important. A system may avoid actual collisions while repeatedly operating with dangerously small margins. Minimum clearance, time to collision, braking margin, force peaks, and trajectory deviation provide more informative measures than collision count alone.

Safety evaluation should include tail risk. Rare catastrophic failures may be hidden by strong average performance. Large-scale simulation, scenario generation, stress testing, and red teaming are necessary to discover low-probability but high-consequence events.

Human-robot interaction requires additional safety evaluation. The robot should respond conservatively to uncertain human intent, unexpected movement, and communication errors. Socially smooth behavior is useful, but physical safety must remain dominant.

The twelfth evaluation level concerns real-time and computational performance. A model cannot control a robot effectively if its decisions arrive too late or consume excessive resources. Evaluation should measure latency, throughput, memory, power, thermal behavior, and stability under load.

Average latency is not enough. Worst-case and high-percentile latency are critical for real-time systems. Occasional long pauses can be more dangerous than consistently moderate delay.

The evaluation should reflect the complete pipeline, including sensor preprocessing, model inference, planning, safety checking, communication, and command execution. Reporting neural-network inference time alone can hide important system delays.

Control frequency should be matched to the task. High-level planning may tolerate slower updates, while obstacle avoidance or force regulation requires faster response. Evaluation should verify that each component operates within its timing contract.

Energy efficiency matters for mobile platforms. Compute-intensive models may reduce mission duration or cause thermal throttling. Metrics may include energy per inference, energy per completed task, and mission time under realistic battery constraints.

The thirteenth evaluation level concerns system integration. A Physical AI foundation model rarely operates alone. It interacts with localization, mapping, databases, motion planners, digital twins, skill libraries, fleet systems, and human interfaces. Evaluation should verify the complete integrated behavior.

Interface errors may produce failures even when individual components perform well. Coordinate conventions, message formats, units, timestamps, version mismatches, and failure codes should be tested explicitly.

Tool use should be evaluated by measuring whether the model selects the correct tool, provides valid arguments, interprets the result correctly, and handles tool failure. Calling a motion planner is useful only if the returned path is integrated safely into execution.

Mission-level integration tests should include communication loss, stale data, service unavailability, and conflicting subsystem outputs. The model should not assume that every tool or service is always correct and available.

The fourteenth evaluation level concerns interpretability and observability. Complete explanation of a large neural model may be impossible, but engineers need sufficient information to diagnose behavior and support certification.

Evaluation should determine whether the system exposes useful intermediate states, such as selected targets, reference frames, subgoals, predicted outcomes, confidence, and safety status. These outputs can help distinguish perception errors from planning or control errors.

Logs should include synchronized model inputs, outputs, actions, tool calls, safety interventions, latency, resource use, and software versions. Replaying an incident should reproduce the decision context as accurately as possible.

Interpretability claims should be tested carefully. Attention maps or generated explanations may appear plausible without reflecting the actual cause of a decision. Counterfactual tests, input masking, feature perturbation, and trajectory replay provide stronger evidence.

The fifteenth evaluation level concerns human factors. Physical AI systems are often used by operators who may not be experts in machine learning. Evaluation should measure whether people can understand the robot's state, provide instructions, intervene, and recover from errors.

The robot should communicate what it intends to do, what it has completed, why it stopped, and when it requires assistance. Explanations should be accurate and operationally useful rather than merely fluent.

Usability evaluation may measure instruction success, clarification quality, operator workload, intervention time, trust calibration, and error recovery. Excessive trust is as dangerous as insufficient trust. Users should understand the validated capabilities and limitations of the system.

The sixteenth evaluation level concerns data and model contamination. Foundation models may have encountered benchmark examples, simulation environments, object models, or task descriptions during training. This can produce inflated results.

Evaluation datasets should be separated carefully from training data. Similarity analysis, version tracking, and hidden test sets can reduce contamination risk. For robot evaluation, physical environments and objects should include genuinely unseen combinations.

Memorization should be distinguished from generalization. A model that reproduces a familiar demonstration may appear competent without understanding the task. Novel layouts, altered object properties, and alternative valid strategies can expose this weakness.

The seventeenth evaluation level concerns continual learning and model updates. A model may improve on new tasks while losing previous capabilities. Every update should therefore undergo regression evaluation.

Regression suites should include core perception, grounding, planning, safety, latency, and embodiment tests. Changes in quantization, adapters, training data, hardware, or software libraries can all alter behavior.

Fleet deployment should use staged evaluation. A new model can first run offline, then in simulation, shadow mode, controlled hardware tests, limited pilot deployment, and finally broader rollout. Each stage should have clear acceptance criteria.

Rollback capability should also be evaluated. The system must be able to return to a validated model version quickly if unexpected behavior appears. Model files, adapters, calibration, action schemas, and safety policies should be versioned together.

The eighteenth evaluation level concerns benchmark design. Physical AI benchmarks should represent meaningful tasks rather than isolated academic subtasks. They should connect perception, reasoning, action, and consequence.

A benchmark should specify the robot embodiment, sensor configuration, action interface, environment, task distribution, safety constraints, and success conditions. Without these details, results cannot be compared meaningfully.

Benchmark diversity is important. A single score may hide major differences among manipulation, navigation, inspection, social interaction, and recovery. Reporting a structured capability profile is often more useful than one aggregate number.

Metrics should also reflect task quality. A robot that completes a task slowly, with repeated collisions prevented only by safety intervention, should not receive the same evaluation as a smooth and autonomous execution.

Standardization can support comparison, but benchmarks must evolve. Once systems optimize heavily for a fixed benchmark, the score may stop representing real-world competence. Hidden tests, changing scenarios, and periodic benchmark revision can reduce overfitting.

The nineteenth evaluation level concerns simulation and real-world transfer. Simulation enables safe, scalable, and reproducible testing. Thousands of scenarios can be generated with controlled variation in lighting, object placement, dynamics, sensor noise, and failures.

Simulation is especially valuable for rare and dangerous events. Emergency stops, unstable payloads, human intrusions, extreme weather, and sensor faults can be tested without physical harm.

However, simulation results should not be treated as sufficient evidence for real-world reliability. Differences in friction, contact, latency, visual appearance, hardware wear, and human behavior can change outcomes. Sim-to-real performance should be measured directly.

A progressive evaluation pathway is therefore necessary. Models should move from offline datasets to closed-loop simulation, hardware-in-the-loop testing, controlled laboratory experiments, pilot deployment, and operational monitoring.

The operational design domain should be defined from these results. It specifies the environments, tasks, speeds, payloads, sensor conditions, and robot configurations in which the system has been validated. Performance outside this domain should be considered uncertain.

The final evaluation level concerns lifecycle governance. Evaluation does not end when deployment begins. Real-world operation produces new data, failures, environmental changes, and model updates. Continuous monitoring is required to determine whether performance remains within validated limits.

Operational metrics should track task success, intervention, safety events, uncertainty, latency, hardware condition, and distribution shift. Significant changes should trigger investigation or revalidation.

Incident analysis should connect observed behavior to the exact model, data, calibration, hardware, and software versions involved. This traceability is necessary for maintenance, accountability, and certification.

Evaluation governance should define who creates tests, who approves deployment, who reviews failures, and what evidence is required for updates. Independent review may be necessary for high-risk applications.

Ultimately, Physical AI foundation model evaluation must measure more than intelligence in the abstract. It must determine whether broad learned capabilities can be transformed into reliable physical behavior through a particular embodiment and system architecture.

A strong model should perceive accurately, ground language and concepts, understand its body, plan feasible actions, predict consequences, generalize to new conditions, recognize uncertainty, recover from failure, and respect safety and timing constraints. These capabilities must be demonstrated in closed-loop interaction rather than inferred only from static benchmarks.

The most trustworthy evaluation framework combines offline metrics, simulation, hardware testing, real-world trials, stress testing, human assessment, and continuous operational monitoring. It examines both average performance and rare failures, both model behavior and surrounding safety systems, and both immediate capability and long-term stability.

For Physical AI, evaluation is not simply a method for assigning a score. It is the process through which the boundaries of autonomy are defined. A robot should be permitted to act independently only within the conditions where its perception, reasoning, control, and safety have been demonstrated to work together reliably.

피지컬 AI를 위한 파운데이션 모델을 평가하는 것은 언어 모델(Language Model), 이미지 분류기(Image Classifier), 또는 다른 디지털 시스템을 평가하는 것과 근본적으로 다르다. 디지털 모델은 일반적으로 출력 결과를 정답(Label), 기준 답안(Reference Answer), 벤치마크 점수(Benchmark Score)와 비교하여 평가할 수 있다. 반면 피지컬 AI 모델은 인식(Perception), 추론(Reasoning), 계획(Planning), 제어(Control), 그리고 실제 물리적 결과(Physical Consequence)에 이르는 전체 과정을 평가해야 한다. 텍스트로는 올바르게 보이는 응답이라도 실제 로봇에서는 위험하거나 비효율적인 행동을 생성할 수 있다. 따라서 평가는 모델이 작업을 이해하는지뿐 아니라, 특정한 몸체를 통해 동적인 환경에서 작업을 신뢰성 있게 수행할 수 있는지도 함께 검증해야 한다.

평가의 목적은 단순히 모델의 순위를 매기는 것이 아니다. 평가는 모델의 능력 한계를 확인하고, 일반화 성능을 측정하며, 위험한 행동을 탐지하고, 다양한 아키텍처를 비교하며, 엔지니어링 의사결정을 지원하고, 실제 배포 조건을 정의하며, 모델 업데이트가 기존에 검증된 성능을 저하시켰는지 확인하는 데 사용된다. 피지컬 AI에서는 평가는 학습이 끝난 뒤 수행하는 최종 시험이 아니라 안전(Safety)과 수명주기 관리(Lifecycle Management)의 일부이다.

포괄적인 평가 체계는 여러 수준(Level)에서 모델을 평가해야 한다. 개별 인식과 추론 능력, 통합 작업 수행 능력, 폐루프 행동(Closed-Loop Behavior), 환경 변화에 대한 강건성(Robustness), 불확실성 인식(Uncertainty Awareness), 실패 복구(Failure Recovery), 계산 효율성(Computational Efficiency), 안전 제약 준수(Safety Constraint Compliance)를 모두 측정해야 한다. 이러한 수준들은 서로 연관되어 있지만 동일한 것은 아니다. 한 영역에서 뛰어난 성능이 다른 영역의 우수한 성능을 보장하지는 않는다.

첫 번째 평가 수준은 표현(Representation)과 인식(Perception)이다. 피지컬 AI 파운데이션 모델은 카메라(Camera), 깊이 센서(Depth Sensor), LiDAR, 힘 센서(Force Sensor), 고유감각(Proprioception), 지도(Map), 언어(Language) 등 다양한 모달리티(Modality)의 정보를 해석해야 한다. 평가는 모델이 이러한 입력으로부터 의미 있고 기하학적으로 일관된 표현을 추출하는지를 확인해야 한다.

시각 평가(Visual Evaluation)는 객체 인식(Object Recognition), 객체 검출(Object Detection), 분할(Segmentation), 자세 추정(Pose Estimation), 깊이 이해(Depth Understanding), 장면 분류(Scene Classification), 이상 탐지(Anomaly Detection), 객체 추적(Tracking)을 포함할 수 있다. 조작(Manipulation)에서는 파지 가능한 영역(Graspable Region), 접촉면(Contact Surface), 손잡이(Handle), 공구(Tool), 어포던스(Affordance)를 식별해야 한다. 내비게이션(Navigation)에서는 자유 공간(Free Space), 장애물(Obstacle), 주행 가능 영역(Traversable Terrain), 문(Door), 엘리베이터(Elevator), 사람(People), 교통 규칙(Traffic Rule)을 인식해야 한다.

그러나 기존의 시각 정확도만으로는 충분하지 않다. 표현은 실제 행동을 지원해야 한다. 로봇은 객체를 정확하게 분류하면서도 위치를 잘못 추정할 수 있다. 손잡이를 인식했더라도 실제로 도달 가능한지는 판단하지 못할 수 있다. 따라서 인식 평가는 계량적 기하학(Metric Geometry), 목표 연결(Target Association), 좌표계 일관성(Coordinate Frame Consistency), 작업 관련성(Task Relevance)을 함께 포함해야 한다.

교차 모달 접지(Cross-Modal Grounding)도 평가 대상이다. 모델은 언어 명령을 현재 환경 속의 객체, 영역, 행동, 제약조건과 정확하게 연결해야 한다. "오른쪽 위 커넥터를 검사하라"는 명령을 받으면 단순히 단어를 이해하는 것이 아니라 올바른 커넥터를 선택해야 한다. "기계 뒤로 이동하라"는 지시는 의도된 공간 기준 좌표계를 정확하게 해석해야 한다.

접지 시험(Grounding Test)은 모호한 언어(Ambiguous Language), 지시 표현(Referring Expression), 제스처(Gesture), 유사한 객체가 여러 개인 환경, 변화하는 시점(Viewpoint), 불완전한 정보(Incomplete Information)를 포함해야 한다. 시스템은 명령이 명확할 때만 성공하는 것이 아니라 정보가 부족할 경우 사용자에게 명확한 설명을 요청할 수도 있어야 한다.

시간적 인식(Temporal Perception)은 별도로 평가되어야 한다. 모델은 가려짐(Occlusion) 속에서도 객체를 추적하고, 움직임을 인식하며, 접촉 상태의 변화를 감지하고, 작업 진행 상태(Task Progress)를 추정하며, 오래된 정보와 현재 관측을 구분해야 한다. 정적인 이미지 벤치마크만으로는 로봇이 시간이 흐르면서 장면이 어떻게 변하는지를 이해하는지 평가할 수 없다.

두 번째 평가 수준은 체화 인식(Embodiment Awareness)이다. 동일한 파운데이션 모델이 서로 다른 로봇, 공구, 적재물(Payload), 센서 구성을 처리해야 할 수도 있다. 평가는 모델이 현재 몸체의 능력과 한계를 이해하는지를 확인해야 한다.

체화 시험에서는 로봇의 크기, 관절 한계(Joint Limit), 휠 구성(Wheel Configuration), 적재 능력, 공구 길이, 카메라 위치, 액추에이터 속도, 제어 주기(Control Frequency)를 변화시킬 수 있다. 모델은 모든 플랫폼에 동일한 행동을 적용하는 것이 아니라 현재 몸체에 맞게 계획을 조정해야 한다. 경량 이동형 로봇에 적합한 궤적이 중량 자율주행 플랫폼에는 적합하지 않을 수 있으며, 한 매니퓰레이터에서 가능한 파지가 다른 로봇에서는 불가능할 수도 있다.

교차 체화 평가(Cross-Embodiment Evaluation)는 작업 이해(Task Understanding)와 실제 실행(Execution)을 구분해야 한다. 모델은 목표를 정확하게 이해하면서도 현재 로봇에 적절하게 행동을 변환하지 못할 수도 있다. 따라서 상위 수준의 지식은 전이되면서도 물리적 제약은 계속 유지되는지를 평가해야 한다.

공구 사용(Tool Use)은 체화의 변화로 평가해야 한다. 로봇이 프로브(Probe), 그리퍼(Gripper), 카메라 마스트(Camera Mast) 등을 장착하면 유효 형상(Effective Geometry)과 제어 인터페이스가 달라진다. 모델은 이에 맞추어 작업 반경(Reach), 충돌 영역(Collision Envelope), 공구 중심점(Tool Center Point), 작업 전략(Task Strategy)을 수정해야 한다.

세 번째 평가 수준은 명령 수행(Instruction Following)과 작업 분해(Task Decomposition)이다. 피지컬 AI 모델은 목표를 제약조건을 유지한 채 실제 행동으로 변환해야 한다. 평가는 단일 명령(Single-Step Command), 다중 조건 명령(Multi-Condition Instruction), 장기 임무(Long-Horizon Mission), 환경이나 내부 상태 변화에 따라 행동을 변경해야 하는 작업을 포함해야 한다.

예를 들어 "1번부터 5번 검사 지점을 검사하고, 제한 구역은 피하며, 배터리가 30% 이하가 되면 충전한 뒤 마지막 작업부터 다시 시작하라"는 명령은 메모리(Memory), 계획(Planning), 조건 추론(Conditional Reasoning), 임무 상태 관리(Mission-State Tracking)를 모두 요구한다. 평가는 최종 성공뿐 아니라 모든 조건을 제대로 준수했는지도 함께 측정해야 한다.

작업 분해 평가는 모델이 의미 있는 하위 목표(Subgoal)를 생성하고, 이를 올바른 순서로 배치하며, 적절한 스킬(Skill)이나 공구를 선택하는지를 확인해야 한다. 우수한 모델은 복잡한 임무를 실행 가능한 단계로 분해해야 하며, 모호하거나 물리적으로 불가능한 계획을 생성해서는 안 된다.

중간 계획(Intermediate Plan)의 품질도 중요하다. 로봇이 결국 작업을 완료하더라도 비효율적이거나 불안정하거나 위험한 방식으로 수행할 수 있다. 따라서 경로 길이(Path Length), 불필요한 행동 수, 에너지 소비(Energy Consumption), 수행 시간, 사람의 개입 빈도도 함께 평가해야 한다.

네 번째 평가 수준은 행동 품질(Action Quality)이다. 로봇의 행동은 이산 토큰(Discrete Token), 연속 명령(Continuous Command), 웨이포인트(Waypoint), 궤적(Trajectory), 스킬 호출(Skill Call), 행동 청크(Action Chunk) 등으로 표현될 수 있다. 평가는 출력 형식뿐 아니라 실제 물리적 결과까지 함께 고려해야 한다.

조작에서는 위치 오차(Position Error), 자세 오차(Orientation Error), 파지 성공률(Grasp Success), 접촉력(Contact Force), 미끄러짐(Slip), 궤적의 부드러움(Trajectory Smoothness), 물체 손상(Object Damage)을 측정할 수 있다. 이동형 로봇은 경로 추종(Path Tracking), 위치 추정 정확도(Localization Accuracy), 정지 거리(Stopping Distance), 장애물 간격(Clearance), 속도 변화의 부드러움, 충돌 빈도를 평가할 수 있다. 검사 로봇은 이미지 품질(Image Quality), 검사 영역 커버리지(Target Coverage), 센서 시점(Viewpoint), 반복 측정 정확도(Measurement Repeatability), 결함 검출 성능을 평가할 수 있다.

개방 루프(Open-Loop) 행동 예측은 전문가 시연(Expert Demonstration)과 모델 출력을 비교할 수 있다. 행동 오차(Action Error)나 시퀀스 우도(Sequence Likelihood)는 개발 단계에서는 유용하지만 실제 배포를 결정하기에는 충분하지 않다. 서로 다른 두 행동이 모두 올바를 수 있으며, 작은 예측 오차가 반복 실행되면서 큰 차이를 만들 수도 있기 때문이다.

따라서 폐루프 평가(Closed-Loop Evaluation)가 필수적이다. 모델은 시뮬레이션이나 실제 로봇에서 동작하며, 자신의 행동이 이후의 관측을 변화시키는 상황에서 평가되어야 한다. 이를 통해 오차 누적(Error Accumulation), 노출 편향(Exposure Bias), 불안정한 피드백(Unstable Feedback), 복구 행동(Recovery Behavior)을 확인할 수 있다.

다섯 번째 평가 수준은 월드 모델(World Model)과 결과 예측(Consequence Prediction)이다. 우수한 피지컬 AI 파운데이션 모델은 행동 이후 환경이 어떻게 변하는지를 예측할 수 있어야 한다. 평가는 미래 상태(Future State), 객체 이동(Object Motion), 접촉 이벤트(Contact Event), 충돌(Collision), 작업 진행(Task Progress), 불확실성(Uncertainty)의 예측 정확도를 측정해야 한다.

월드 모델 평가는 수동 예측(Passive Prediction)과 행동 조건부 예측(Action-Conditioned Prediction)을 모두 포함해야 한다. 일반적인 미래를 예측하는 것과 특정 행동을 수행했을 때 어떤 일이 일어나는지를 예측하는 것은 서로 다르며, 후자가 계획(Planning)에서는 훨씬 중요하다.

예측 시간 범위(Prediction Horizon)도 별도로 평가해야 한다. 단기 예측은 정확하지만 장기 예측은 빠르게 오차가 누적될 수 있다. 따라서 시간에 따른 오차 증가와 모델이 불확실성 증가를 적절히 인식하는지를 측정해야 한다.

유용한 월드 모델은 단순히 그럴듯한 영상을 생성하는 것이 아니라 제어에 필요한 정보를 유지해야 한다. 객체 위치, 자유 공간, 지지 관계(Support Relationship), 접촉 상태 등 행동 선택에 필요한 변수들을 정확하게 표현해야 한다. 시각적으로 사실적이지만 기하학이 잘못된 출력은 충분하지 않다.

반사실적 평가(Counterfactual Evaluation)는 모델이 서로 다른 행동의 결과를 구분하는지를 시험한다. 여러 후보 궤적이 주어졌을 때 모델은 각각 다른 결과를 예측하고 어떤 행동이 더 안전하거나 효율적인지를 판단해야 한다. 이는 계획과 위험 평가를 지원하는 중요한 능력이다.

여섯 번째 평가 수준은 계획(Planning)이다. 계획은 실행 가능성(Feasibility), 완전성(Completeness), 효율성(Efficiency), 적응성(Adaptability), 안전성(Safety)을 기준으로 평가해야 한다. 생성된 계획은 로봇 운동학(Kinematics), 환경 기하(Environmental Geometry), 자원 제한(Resource Limit), 임무 규칙(Mission Rule)을 모두 만족해야 한다.

계획 벤치마크는 다단계 작업, 임시 목표(Temporary Goal), 공구 사용, 장애물 회피, 복구 행동을 포함해야 한다. 또한 실행 중 환경이 변화하여 모델이 기존 계획을 수정해야 하는 상황도 포함해야 한다.

장기 임무(Long-Horizon Mission) 평가는 특히 중요하다. 많은 모델은 짧은 작업에서는 우수하지만 긴 임무에서는 일관성을 잃는다. 완료한 단계를 잊거나, 행동을 반복하거나, 후반부 제약을 무시하거나, 중단 이후 다시 이어서 수행하지 못할 수 있다.

임무 수준 평가는 완료율(Completion Rate), 올바르게 수행한 하위 목표 수, 반복되거나 누락된 행동, 재계획(Replanning) 품질, 복구 시간, 배터리 관리, 인간 개입 빈도를 포함할 수 있다. 모델이 일관된 임무 상태를 유지하는지도 함께 검증해야 한다.

일곱 번째 평가 수준은 일반화(Generalization)이다. 파운데이션 모델은 학습 데이터에 포함되지 않은 상황에서도 동작해야 한다. 따라서 새로운 객체, 장면, 시점, 작업, 언어, 로봇 몸체, 물리적 조건을 포함한 평가가 필요하다.

조합적 일반화(Compositional Generalization)는 익숙한 요소를 새로운 방식으로 결합하여 평가할 수 있다. 예를 들어 모델이 빨간 물체, 보관함, 상단 선반은 각각 학습했지만 "빨간 물체를 위쪽 보관함에 넣어라"는 명령은 처음 보는 경우에도 성공적으로 수행해야 한다.

도메인 일반화(Domain Generalization)는 새로운 공장, 바닥 재질, 조명 조건, 레이아웃(Layout), 교통 패턴을 시험해야 한다. 단순한 외형 변화가 아니라 실제로 전이 가능한 구조를 학습했는지를 평가하는 것이 목적이다.

교차 체화 일반화(Cross-Embodiment Generalization)는 서로 다른 운동학이나 센서를 가진 로봇으로 작업 지식이 전이되는지를 평가해야 한다. 시뮬레이션-현실 전이(Sim-to-Real Transfer)는 시뮬레이션에서 학습한 행동이 실제 하드웨어에서도 유지되는지를 측정해야 한다.

일반화에 대한 주장은 반드시 범위를 명확히 해야 한다. 몇 개의 유사한 작업에서 성공했다고 해서 모델을 범용이라고 표현해서는 안 된다. 평가 대상 분포(Evaluated Distribution), 새로운 정도(Novelty Level), 운영 조건을 명확히 문서화해야 한다.

여덟 번째 평가 수준은 강건성(Robustness)이다. 실제 로봇은 노이즈, 센서 성능 저하, 통신 지연, 보정 오차, 하드웨어 마모, 예측하기 어려운 환경 변화 속에서 동작한다. 강건한 모델은 갑작스럽게 실패하지 않고 성능이 점진적으로 저하되어야 한다.

강건성 시험은 이미지 블러(Image Blur), 눈부심(Glare), 어두운 환경, 가림, 깊이 정보 손실, 희소한 LiDAR 데이터, 지연된 상태 추정, 부정확한 지도, 힘 센서 노이즈, 패킷 손실(Packet Loss), 위치 추정 드리프트(Localization Drift)를 포함할 수 있다. 작업 성능이 저하되더라도 안전성은 유지되어야 한다.

센서 누락(Sensor Dropout)은 멀티모달 시스템에서 특히 중요하다. 모델이 하나의 센서에 지나치게 의존하면 해당 센서가 고장났을 때 전체 시스템이 실패할 수 있다. 따라서 특정 모달리티를 제거하거나 손상시켜도 다른 센서를 활용할 수 있는지를 평가해야 한다.

보정 강건성(Calibration Robustness)도 측정해야 한다. 다만 큰 보정 오류를 정상 운용 조건으로 간주해서는 안 된다. 현실적인 작은 오차에 대한 허용 범위는 측정할 수 있지만, 심각한 오류는 모델이 적응하는 것이 아니라 고장 탐지(Fault Detection)를 유발해야 한다.

적대적 강건성(Adversarial Robustness)은 물리적 측면도 포함한다. 특이한 표시, 반사, 혼동되는 객체, 악의적인 명령, 조작된 환경 신호는 잘못된 행동을 유발할 수 있다. 평가는 우발적인 오류와 의도적인 공격을 모두 포함해야 한다.

아홉 번째 평가 수준은 실패 탐지(Failure Detection)와 복구(Recovery)이다. 물리 작업에서는 예상하지 못한 결과가 반드시 발생한다. 평가는 모델이 실패를 인식하고 원인을 추정하며 적절한 복구 전략을 선택하는지를 확인해야 한다.

실패 시나리오는 파지 실패(Missed Grasp), 물체 낙하(Dropped Object), 막힌 경로(Blocked Path), 휠 슬립(Wheel Slip), 위치 추정 상실(Localization Loss), 공구 정렬 오류(Tool Misalignment), 센서 고장, 예상치 못한 사람의 진입, 배터리 부족, 통신 손실 등을 포함할 수 있다. 로봇은 작업이 정상적으로 진행되고 있는 것처럼 계속 행동해서는 안 된다.

실패 탐지 지연(Detection Latency)도 중요하다. 몇 초 뒤에야 파지 실패를 인식하는 것은 너무 늦을 수 있다. 시스템은 예상 결과와 실제 결과를 지속적으로 비교하고 안전한 시간 안에 대응해야 한다.

복구 품질은 재시도 성공률, 시도 횟수, 손실 시간, 추가 위험, 인간의 도움 필요 여부 등을 기준으로 평가할 수 있다. 동일한 실패 행동을 반복하는 것은 지능적인 복구라고 볼 수 없다.

모델은 스스로 복구해서는 안 되는 상황도 알아야 한다. 일부 실패는 정지, 유지보수, 인간의 판단이 필요하다. 평가는 로봇이 위험한 자가 복구나 무리한 반복 시도를 하지 않는지도 확인해야 한다.

열 번째 평가 수준은 불확실성(Uncertainty)과 신뢰도 보정(Calibration)이다. 파운데이션 모델은 입력이 모호하거나, 센서가 불안정하거나, 학습 분포와 다른 상황에서는 낮은 신뢰도를 표현해야 한다. 특히 잘못된 행동에 대해 높은 확신을 보이는 것은 매우 위험하다.

신뢰도 보정 평가는 모델이 예측한 신뢰도와 실제 성공률을 비교한다. 예를 들어 모델이 80%의 확신을 표현했다면 실제 성공률도 약 80% 수준이어야 한다. 잘못 보정된 신뢰도는 부적절한 자율성을 초래할 수 있다.

불확실성 평가는 알려진 모호성과 미지의 상황을 구분해야 한다. 여러 경로가 모두 가능한 상황은 위험하지 않은 불확실성이지만, 익숙하지 않은 하드웨어 동작은 심각한 분포 외 상황(Out-of-Distribution Condition)일 수 있다.

선택적 예측(Selective Prediction)은 유용한 평가 방법이다. 모델은 신뢰도가 낮을 경우 행동을 거부하거나 도움을 요청할 수 있다. 이후 처리 범위(Coverage)와 위험(Risk)의 관계를 평가할 수 있다. 실제 시스템은 자신이 잘 이해하는 상황만 자율적으로 처리함으로써 더 높은 신뢰성을 확보할 수 있다.

불확실성에 대한 실제 대응도 평가해야 한다. 로봇은 속도를 줄이고, 추가 관측을 수행하고, 시점을 변경하고, 전문 계획기를 호출하거나, 사람에게 질문하거나, 안전 상태(Safe State)로 진입할 수 있어야 한다. 정확한 불확실성 추정도 행동에 반영되지 않으면 의미가 없다.

열한 번째 평가 수준은 안전(Safety)이다. 안전 평가는 학습된 행동과 이를 제한하는 독립적인 안전 시스템을 모두 검증해야 한다. 파운데이션 모델은 인간의 의도와 운영 규칙을 따라야 하지만, 결정론적 안전 메커니즘(Deterministic Safety Mechanism)은 핵심 제한을 반드시 강제해야 한다.

시험은 충돌 회피(Collision Avoidance), 관절 및 속도 제한, 힘 제한, 정지 행동, 제한 구역, 사람과의 근접, 불안정한 적재물, 비상 상황(Emergency Condition)을 포함해야 한다. 정상적인 상황만 평가해서는 충분하지 않다.

안전 개입(Safety Intervention) 횟수는 중요한 지표이다. 런타임 모니터(Runtime Monitor)가 모델의 출력을 얼마나 자주 수정하거나 거부하는지를 측정할 수 있다. 사고는 발생하지 않더라도 외부 안전 계층이 지속적으로 개입한다면 모델 자체는 충분히 안전하지 않을 수 있다.

니어미스(Near-Miss) 분석도 중요하다. 실제 충돌은 없더라도 위험한 거리에서 반복적으로 움직인다면 이는 중요한 문제이다. 최소 간격(Minimum Clearance), 충돌까지 남은 시간(Time to Collision), 제동 여유(Braking Margin), 힘의 최대값, 궤적 편차(Trajectory Deviation)는 충돌 횟수보다 더 의미 있는 지표가 될 수 있다.

안전 평가는 꼬리 위험(Tail Risk)도 포함해야 한다. 평균 성능이 우수하더라도 드물지만 치명적인 실패가 존재할 수 있다. 대규모 시뮬레이션, 시나리오 생성, 스트레스 테스트, 레드팀 테스트(Red Teaming)를 통해 이러한 사건을 찾아야 한다.

인간-로봇 상호작용(Human-Robot Interaction)은 추가적인 안전 평가를 요구한다. 로봇은 인간의 의도가 불확실하거나 예상치 못한 움직임이 발생하거나 의사소통 오류가 발생했을 때 보수적으로 행동해야 한다. 사회적으로 자연스러운 행동도 중요하지만 물리적 안전이 항상 우선되어야 한다.

열두 번째 평가 수준은 실시간성과 계산 성능(Real-Time and Computational Performance)이다. 모델이 너무 늦게 판단하거나 과도한 계산 자원을 사용한다면 로봇을 효과적으로 제어할 수 없다. 평가는 지연 시간(Latency), 처리량(Throughput), 메모리 사용량, 전력 소비(Power), 열 특성(Thermal Behavior), 부하 상황에서의 안정성을 포함해야 한다.

평균 지연 시간만으로는 충분하지 않다. 최악의 경우(Worst Case)와 상위 백분위 지연 시간(High Percentile Latency)이 실시간 시스템에서는 더욱 중요하다. 드물게 발생하는 긴 지연은 항상 일정한 중간 수준의 지연보다 더 위험할 수 있다.

평가는 센서 전처리, 모델 추론, 계획, 안전 검사, 통신, 명령 실행을 포함한 전체 파이프라인을 대상으로 해야 한다. 신경망 추론 시간만 보고하면 중요한 시스템 지연이 숨겨질 수 있다.

제어 주기는 작업 특성에 맞추어야 한다. 상위 계획은 느린 갱신도 가능하지만 장애물 회피나 힘 제어는 훨씬 빠른 응답을 요구한다. 평가에서는 각 구성 요소가 요구된 시간 계약(Timing Contract)을 만족하는지를 검증해야 한다.

에너지 효율(Energy Efficiency)도 이동형 플랫폼에서는 중요하다. 계산량이 많은 모델은 임무 시간을 줄이거나 열로 인해 성능이 제한될 수 있다. 추론당 에너지, 작업 완료당 에너지, 실제 배터리 조건에서의 임무 지속 시간을 측정할 수 있다.

열세 번째 평가 수준은 시스템 통합(System Integration)이다. 피지컬 AI 파운데이션 모델은 단독으로 동작하지 않는다. 위치 추정(Localization), 지도 작성(Mapping), 데이터베이스, 모션 플래너(Motion Planner), 디지털 트윈(Digital Twin), 스킬 라이브러리(Skill Library), 플릿 시스템(Fleet System), 인간 인터페이스와 함께 동작한다. 평가는 전체 시스템이 통합되어 올바르게 동작하는지를 확인해야 한다.

인터페이스 오류는 개별 모듈이 모두 정상이어도 전체 시스템 실패를 만들 수 있다. 좌표계 규약(Coordinate Convention), 메시지 형식(Message Format), 단위(Unit), 타임스탬프, 버전 불일치(Version Mismatch), 실패 코드(Failure Code)를 명시적으로 시험해야 한다.

공구 사용(Tool Use)은 모델이 올바른 공구를 선택하고, 적절한 인자를 전달하며, 결과를 정확히 해석하고, 공구 실패를 처리하는지를 평가해야 한다. 모션 플래너를 호출하는 것만으로는 충분하지 않으며 반환된 경로를 안전하게 실제 실행에 통합해야 한다.

임무 수준 통합 시험은 통신 손실, 오래된 데이터(Stale Data), 서비스 장애(Service Unavailability), 서로 충돌하는 하위 시스템 출력도 포함해야 한다. 모델은 모든 도구와 서비스가 항상 올바르게 동작한다고 가정해서는 안 된다.

열네 번째 평가 수준은 해석 가능성(Interpretability)과 관측 가능성(Observability)이다. 대규모 신경망을 완전히 설명하는 것은 어려울 수 있지만, 엔지니어는 동작을 분석하고 인증을 지원할 수 있을 만큼 충분한 정보를 얻을 수 있어야 한다.

평가는 모델이 선택한 목표, 기준 좌표계, 하위 목표(Subgoal), 예상 결과, 신뢰도, 안전 상태(Safety Status)와 같은 유용한 중간 상태를 제공하는지를 확인해야 한다. 이러한 출력은 인식 오류인지 계획 오류인지 제어 오류인지를 구분하는 데 도움을 준다.

로그(Log)는 모델 입력, 출력, 행동, 도구 호출, 안전 개입, 지연 시간, 자원 사용량, 소프트웨어 버전을 시간 동기화하여 기록해야 한다. 사고가 발생하면 당시의 의사결정 상황을 최대한 정확하게 재현할 수 있어야 한다.

해석 가능성에 대한 주장도 신중히 검증해야 한다. 어텐션 맵(Attention Map)이나 생성된 설명은 그럴듯해 보일 수 있지만 실제 의사결정 원인을 반영하지 않을 수도 있다. 반사실적 시험, 입력 제거(Input Masking), 특징 교란(Feature Perturbation), 궤적 재생(Trajectory Replay)이 더 신뢰할 수 있는 증거를 제공한다.

열다섯 번째 평가 수준은 인간 요소(Human Factors)이다. 피지컬 AI 시스템은 기계학습 전문가가 아닌 작업자도 사용하는 경우가 많다. 평가는 사용자가 로봇 상태를 이해하고, 명령을 내리며, 개입하고, 오류를 복구할 수 있는지를 확인해야 한다.

로봇은 자신이 무엇을 하려 하는지, 무엇을 완료했는지, 왜 멈추었는지, 언제 도움이 필요한지를 명확히 전달해야 한다. 설명은 단순히 자연스러운 문장이 아니라 실제 운영에 도움이 되어야 한다.

사용성 평가(Usability Evaluation)는 명령 성공률, 설명 요청의 품질, 작업자 부담(Operator Workload), 개입 시간, 신뢰도 보정(Trust Calibration), 오류 복구를 측정할 수 있다. 지나친 신뢰도 위험하며, 사용자는 시스템의 검증된 능력과 한계를 올바르게 이해해야 한다.

열여섯 번째 평가 수준은 데이터 및 모델 오염(Data and Model Contamination)이다. 파운데이션 모델은 학습 과정에서 벤치마크 예제, 시뮬레이션 환경, 객체 모델, 작업 설명을 이미 접했을 가능성이 있다. 이는 실제보다 높은 성능을 보이게 만들 수 있다.

평가 데이터셋은 학습 데이터와 철저히 분리되어야 한다. 유사성 분석(Similarity Analysis), 버전 관리, 숨겨진 테스트셋(Hidden Test Set)을 통해 오염 위험을 줄일 수 있다. 로봇 평가는 실제 환경과 객체 조합도 완전히 새로운 상황을 포함해야 한다.

암기(Memorization)와 일반화는 구분되어야 한다. 모델은 익숙한 시연을 그대로 재현하면서도 실제 작업은 이해하지 못할 수 있다. 새로운 레이아웃, 변경된 객체 특성, 다른 해결 전략은 이러한 문제를 드러낼 수 있다.

열일곱 번째 평가 수준은 지속학습(Continual Learning)과 모델 업데이트(Model Update)이다. 새로운 작업 성능이 향상되는 동안 기존 능력이 사라질 수도 있다. 따라서 모든 업데이트는 회귀 평가(Regression Evaluation)를 수행해야 한다.

회귀 시험은 핵심 인식, 접지, 계획, 안전, 지연 시간, 체화 평가를 포함해야 한다. 양자화, 어댑터, 학습 데이터, 하드웨어, 소프트웨어 라이브러리 변경은 모두 행동을 변화시킬 수 있다.

플릿 배포(Fleet Deployment)는 단계적으로 평가해야 한다. 새로운 모델은 오프라인, 시뮬레이션, 섀도우 모드(Shadow Mode), 제한된 하드웨어 시험, 파일럿 배포(Pilot Deployment), 전체 배포 순으로 확대되어야 하며 각 단계마다 명확한 승인 기준(Acceptance Criteria)이 필요하다.

롤백(Rollback) 능력도 평가해야 한다. 예상하지 못한 문제가 발생하면 검증된 이전 모델로 빠르게 복귀할 수 있어야 한다. 모델 파일, 어댑터, 보정 데이터, 행동 스키마, 안전 정책은 함께 버전 관리되어야 한다.

열여덟 번째 평가 수준은 벤치마크 설계(Benchmark Design)이다. 피지컬 AI 벤치마크는 단순한 학술 문제보다 실제 의미 있는 작업을 포함해야 한다. 인식, 추론, 행동, 결과를 하나의 흐름으로 연결해야 한다.

벤치마크는 로봇 체화, 센서 구성, 행동 인터페이스, 환경, 작업 분포, 안전 제약, 성공 조건을 명확히 정의해야 한다. 이러한 정보 없이 결과를 비교하는 것은 의미가 없다.

벤치마크의 다양성도 중요하다. 하나의 점수는 조작, 내비게이션, 검사, 사회적 상호작용, 복구 능력의 차이를 숨길 수 있다. 따라서 하나의 종합 점수보다 구조화된 능력 프로파일(Capability Profile)을 제공하는 것이 더 유용할 수 있다.

평가 지표는 작업의 질(Quality)도 반영해야 한다. 반복적인 안전 개입으로 겨우 작업을 완료한 로봇과 부드럽고 완전 자율적으로 수행한 로봇은 동일한 평가를 받아서는 안 된다.

표준화(Standardization)는 비교를 쉽게 만들지만 벤치마크는 지속적으로 발전해야 한다. 시스템이 하나의 고정된 벤치마크만 최적화하면 점수가 실제 능력을 더 이상 반영하지 않을 수 있다. 숨겨진 시험, 변화하는 시나리오, 주기적인 벤치마크 개정이 과적합(Overfitting)을 줄이는 데 도움이 된다.

열아홉 번째 평가 수준은 시뮬레이션과 실제 환경 전이(Simulation and Real-World Transfer)이다. 시뮬레이션은 안전하고 확장 가능하며 재현 가능한 시험 환경을 제공한다. 조명, 객체 배치, 동역학, 센서 노이즈, 실패 상황을 다양하게 변화시키며 수천 개의 시나리오를 생성할 수 있다.

시뮬레이션은 드물고 위험한 사건을 평가하는 데 특히 유용하다. 비상 정지, 불안정한 적재물, 사람의 갑작스러운 접근, 극한 기상 조건, 센서 고장을 실제 위험 없이 시험할 수 있다.

그러나 시뮬레이션 결과만으로 실제 환경에서의 신뢰성을 입증해서는 안 된다. 마찰, 접촉, 지연, 시각적 외형, 하드웨어 마모, 인간 행동의 차이는 결과를 바꿀 수 있다. 따라서 시뮬레이션-현실 전이 성능은 반드시 실제 환경에서 직접 측정해야 한다.

따라서 단계적인 평가 절차가 필요하다. 모델은 오프라인 데이터셋, 폐루프 시뮬레이션, 하드웨어 인 더 루프(Hardware-in-the-Loop), 통제된 실험실, 파일럿 배포, 실제 운영 모니터링 순으로 검증되어야 한다.

이러한 결과를 바탕으로 운용 설계 영역(Operational Design Domain)을 정의해야 한다. 이는 검증된 환경, 작업, 속도, 적재물, 센서 조건, 로봇 구성의 범위를 의미한다. 이 범위를 벗어난 성능은 불확실한 것으로 간주해야 한다.

마지막 평가 수준은 수명주기 거버넌스(Lifecycle Governance)이다. 평가는 실제 배포 이후에도 끝나지 않는다. 실제 운영은 새로운 데이터, 실패 사례, 환경 변화, 모델 업데이트를 지속적으로 만들어낸다. 따라서 성능이 검증된 범위 안에 계속 머무르는지를 확인하기 위한 지속적인 모니터링이 필요하다.

운영 지표는 작업 성공률(Task Success), 인간 개입, 안전 이벤트, 불확실성, 지연 시간, 하드웨어 상태, 분포 변화(Distribution Shift)를 지속적으로 추적해야 한다. 의미 있는 변화가 발생하면 원인 분석이나 재검증이 이루어져야 한다.

사고 분석(Incident Analysis)은 당시 사용된 모델, 데이터, 보정 정보, 하드웨어, 소프트웨어 버전을 정확히 연결할 수 있어야 한다. 이러한 추적 가능성(Traceability)은 유지보수, 책임성(Accountability), 인증(Certification)에 필수적이다.

평가 거버넌스는 누가 시험을 설계하고, 누가 배포를 승인하며, 누가 실패를 검토하고, 업데이트에 어떤 증거가 필요한지를 정의해야 한다. 고위험 응용에서는 독립적인 검토(Independent Review)가 요구될 수도 있다.

궁극적으로 피지컬 AI 파운데이션 모델의 평가는 추상적인 지능만을 측정하는 것이 아니다. 학습된 범용 능력이 특정한 몸체와 시스템 아키텍처를 통해 신뢰할 수 있는 물리적 행동으로 변환되는지를 검증하는 과정이다.

우수한 모델은 정확하게 인식하고, 언어와 개념을 접지하며, 자신의 몸체를 이해하고, 실행 가능한 행동을 계획하며, 결과를 예측하고, 새로운 환경으로 일반화하며, 불확실성을 인식하고, 실패에서 복구하며, 안전성과 시간 제약을 준수해야 한다. 이러한 능력은 정적인 벤치마크가 아니라 폐루프 상호작용을 통해 입증되어야 한다.

가장 신뢰할 수 있는 평가 체계는 오프라인 지표, 시뮬레이션, 하드웨어 시험, 실제 환경 시험, 스트레스 테스트, 인간 평가, 지속적인 운영 모니터링을 함께 결합한다. 평균 성능뿐 아니라 드문 실패도 함께 분석하고, 모델 자체뿐 아니라 주변 안전 시스템까지 함께 평가하며, 현재의 능력뿐 아니라 장기적인 안정성까지 검증한다.

피지컬 AI에서 평가는 단순히 점수를 부여하는 절차가 아니다. 그것은 자율성(Autonomy)의 경계를 정의하는 과정이다. 로봇은 자신의 인식, 추론, 제어, 안전이 함께 신뢰성 있게 동작함이 검증된 조건 안에서만 독립적으로 행동하도록 허용되어야 한다.

##  

## 1.9 Open Source vs Proprietary Foundation Models

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

The choice between open-source and proprietary foundation models is one of the most important strategic decisions in modern artificial intelligence development. This decision affects not only model performance but also data governance, deployment architecture, security, customization, cost, intellectual property, regulatory compliance, long-term maintenance, and operational independence. For Physical AI, the distinction becomes even more significant because foundation models are connected to robots, sensors, industrial equipment, safety systems, digital twins, and real-time control infrastructure. A model is therefore not merely a software component that generates information. It may influence physical actions with direct consequences for people, machines, production, and the surrounding environment.

The terms open-source and proprietary are often used as if they represent two clearly separated categories, but the actual landscape is more complex. A fully open model may provide model architecture, source code, training procedures, datasets, preprocessing tools, model weights, evaluation results, and a license that permits modification and redistribution. Other models may release only their weights while keeping the training data or complete training process private. Some systems provide open inference code but impose restrictions on commercial use, redistribution, or certain application domains. Proprietary models may be accessible only through an application programming interface, a managed cloud platform, or a licensed on-premise package.

It is therefore important to distinguish open-source software, open-weight models, open-data projects, research-only releases, source-available systems, and fully proprietary services. The word "open" does not automatically mean that every part of the model lifecycle is transparent. Similarly, a proprietary model does not necessarily mean that all deployment options are cloud-based or that customization is impossible. Organizations must examine the actual license, technical documentation, access conditions, update policy, and deployment rights rather than relying only on marketing terminology.

Open-source foundation models provide access to the model implementation and, in many cases, the trained weights. This allows organizations to inspect the architecture, execute the model on their own infrastructure, modify components, fine-tune behavior, and integrate it deeply into existing systems. For research institutions, startups, robot manufacturers, and industrial developers, this level of control can significantly accelerate experimentation.

One of the greatest advantages of open models is deployment independence. A company can operate the model on local servers, private clouds, edge computers, or robot-mounted hardware without continuously sending information to an external provider. This is particularly important for Physical AI because robot data may contain proprietary manufacturing processes, facility layouts, product designs, human images, maintenance information, or safety-sensitive operational records.

Local deployment can reduce privacy and confidentiality risks. Sensor data can remain inside the factory, laboratory, hospital, warehouse, or vehicle. Organizations can define their own access controls, retention policies, encryption methods, and audit procedures. In industries where data sovereignty and cybersecurity are critical, this capability may be more important than obtaining the highest benchmark score.

Open models also support offline operation. A robot working in a remote facility, underground environment, outdoor plantation, shipyard, or network-restricted factory may not have reliable cloud access. If the model can run locally, core intelligence remains available during communication failure. This improves operational resilience and allows the system to maintain essential functions without depending on an external service.

Customization is another major advantage. Open models can be fine-tuned using organization-specific data, adapted to specialized terminology, compressed for edge deployment, connected to new sensor encoders, or modified to produce robot actions. Engineers can replace input layers, change tokenization, add cross-modal modules, integrate tool-use interfaces, or attach robot-specific action decoders.

For Physical AI, such architectural modification is often necessary. A general language or vision model does not automatically understand joint states, point clouds, tactile feedback, force signals, mission states, navigation maps, or robot-specific action schemas. Developers may need to add modality adapters, temporal encoders, world-model components, memory systems, or safety heads. Open access makes this possible.

Open models can also support parameter-efficient adaptation. Techniques such as low-rank adaptation, adapters, prompt tuning, and partial fine-tuning allow an organization to specialize a large foundation model without retraining every parameter. Different robots or customers can use separate adapters while sharing a common base model. This can reduce training cost and simplify product-line maintenance.

Transparency is another important benefit. Source code and model architecture can be inspected for hidden dependencies, unsafe operations, external communications, or implementation errors. Researchers can study how the model processes information, reproduce experiments, compare modifications, and build independent evaluation tools.

However, transparency has limits. Access to code and weights does not mean that the model's internal behavior is fully understandable. Large neural networks remain difficult to interpret, and training data may not be completely disclosed. An open model can still contain unknown biases, memorized information, security vulnerabilities, or unsafe behavior. Openness makes investigation easier but does not eliminate the need for validation.

Community development is one of the strongest forces behind open-source models. Researchers and engineers can contribute optimized inference engines, quantization methods, deployment tools, evaluation benchmarks, safety filters, training recipes, and hardware support. Problems found by one group can be documented and corrected by others. This distributed innovation can move faster than development inside a single organization.

Open ecosystems often support a wide range of hardware. Models may be optimized for GPUs, CPUs, embedded accelerators, edge devices, and specialized inference processors. This flexibility is valuable for robotics, where compute platforms vary widely from cloud clusters to compact onboard systems.

The ability to preserve a model version is also strategically important. An organization can archive the weights, code, dependencies, tokenizers, adapters, and configuration required for a validated deployment. The system can continue operating even if the original model provider changes direction, discontinues a service, modifies its pricing, or releases an incompatible update.

This supports reproducibility and certification. A robot deployed in a regulated or safety-sensitive environment may need to remain on a validated software version. Automatic changes to model behavior can invalidate previous testing. With an open model, the organization controls when and how updates occur.

Open-source deployment can also reduce vendor lock-in. If a model is deeply integrated through a proprietary interface, moving to another provider may require substantial redevelopment. An open architecture can make it easier to replace or compare models, although differences in tokenization, context format, multimodal interfaces, and output behavior can still create migration costs.

Despite these advantages, open models impose significant responsibilities. The organization must provide infrastructure, cybersecurity, deployment engineering, performance optimization, monitoring, model updates, and technical support. A model that is free to download is not free to operate.

Large models may require expensive accelerators, high-capacity memory, fast storage, and substantial energy. Engineers must optimize inference, manage model parallelism, support quantization, and maintain compatible software libraries. Total cost can exceed the apparent savings from avoiding API fees.

The organization must also evaluate safety independently. Open models may not include strong alignment, content filtering, uncertainty estimation, or physical safety mechanisms. Even when such features exist, they may not match the organization's robot, environment, or operational requirements.

For Physical AI, downloading a capable model and connecting it directly to a robot is not acceptable. The system requires deterministic safety limits, action validation, collision checking, runtime monitoring, emergency stopping, and clearly defined authority boundaries. These components are the responsibility of the system developer regardless of whether the base model is open or proprietary.

Security is another concern. Publicly available weights can be inspected by defenders, but they can also be analyzed by attackers. Vulnerabilities in tool-use behavior, prompt processing, model serialization, or deployment infrastructure may become widely known. Organizations must monitor security advisories, verify downloaded artifacts, isolate model execution, and protect the surrounding software stack.

Supply-chain security is particularly important. Open-source deployments often depend on many libraries, model repositories, container images, and community tools. A compromised dependency can affect the entire robot system. Signed artifacts, dependency scanning, reproducible builds, and controlled repositories are therefore necessary.

Licensing can be complex. Some models permit research use but restrict commercial deployment. Others prohibit use in specific industries or at certain organizational scales. Some licenses require attribution, disclosure of modifications, or redistribution under similar terms. Dataset licenses may differ from model licenses.

An organization must verify whether it can fine-tune, redistribute, embed, sell, or provide the model as part of a robot product. Legal review is especially important when the model will be deployed internationally or integrated into a commercial platform.

Intellectual-property risk also exists. The training data used by an open model may not be fully documented. The model may reproduce copyrighted, confidential, or personal information. Local deployment does not eliminate responsibility for model outputs or data handling.

Proprietary foundation models follow a different model of development and access. They are usually trained, hosted, maintained, and updated by a commercial provider. Users access them through an API, managed service, software platform, or enterprise agreement.

The most visible advantage of proprietary models is convenience. Organizations can use highly capable models without building the complete training and inference infrastructure. The provider handles model serving, scaling, hardware utilization, optimization, and many security operations.

This can accelerate prototyping. A team can test language understanding, multimodal reasoning, code generation, document analysis, or planning support with relatively little setup. The ability to access a frontier-scale model through a simple interface can reduce the initial barrier to experimentation.

Proprietary providers may also offer strong performance. They can invest in large datasets, extensive compute infrastructure, specialized training methods, safety tuning, evaluation, and inference optimization. Individual organizations may not be able to reproduce these capabilities independently.

Managed services may provide reliability features such as geographic redundancy, automatic scaling, usage monitoring, authentication, logging, enterprise support, and service-level agreements. These features can be valuable for customer-facing applications.

Proprietary models may also receive frequent improvements. The provider can update reasoning quality, multimodal capability, latency, tool use, context length, and security without requiring customers to retrain the model. For rapidly evolving applications, this can provide access to new capabilities quickly.

However, provider-controlled updates create risks for robotics. A model's behavior may change even when the customer's own software remains unchanged. Prompt interpretation, refusal behavior, output format, tool selection, latency, or model confidence may shift after an update.

For a digital assistant, such changes may be manageable. For a robot system that has undergone extensive validation, an unexpected behavioral change can be serious. Organizations may require fixed model versions, advance notice of updates, regression testing periods, and rollback mechanisms.

Data governance is a central issue. Cloud-based proprietary models may require robot data to be transmitted to external infrastructure. Even when providers offer strong privacy commitments, organizations must consider where the data is processed, how long it is retained, whether it is used for model improvement, who can access it, and which legal jurisdiction applies.

Industrial robot data may reveal factory layouts, production schedules, defects, maintenance problems, product designs, employee activities, or safety incidents. This information may be unsuitable for external transmission. Enterprise contracts, private networking, data isolation, or on-premise options may therefore be required.

Network dependence is another limitation. A cloud-based model cannot reliably support safety-critical or high-frequency robot control when communication latency is variable. Connectivity can fail, bandwidth can become limited, and remote service outages can interrupt operation.

Proprietary cloud models are therefore more appropriate for lower-frequency tasks such as mission interpretation, report generation, maintenance assistance, semantic planning, or offline analysis than for direct motor control. Local safety and control functions must remain independent.

Customization may be restricted. Providers may offer prompt engineering, retrieval augmentation, fine-tuning, or tool integration, but customers usually cannot modify the complete architecture. Adding a new sensor encoder, changing attention structure, integrating a specialized action tokenizer, or inspecting internal activations may be impossible.

This limits the suitability of some proprietary models as the central architecture of a robot foundation model. They may be powerful reasoning components but difficult to transform into fully embodied perception-action systems.

Vendor lock-in is another concern. Applications may become dependent on a provider's API format, function-calling protocol, model behavior, pricing, account policies, and regional availability. Switching providers may require new evaluation, prompt redesign, safety testing, and software changes.

Cost can also become unpredictable. API pricing may appear attractive during prototyping but grow rapidly when deployed across a large robot fleet. Robots generate continuous streams of images, sensor data, and state updates. Sending all of this information to a remote model can create substantial processing and communication expenses.

Lifecycle cost should include inference fees, data transfer, storage, support, redundancy, integration, and future migration. A proprietary model with high per-request capability may still be less economical than a smaller local model for frequent repetitive tasks.

Limited transparency can complicate safety assurance. Users may not know the model architecture, training data, optimization process, or detailed limitations. Providers may publish safety reports and evaluation results, but customers often cannot independently reproduce the full training process.

For regulated robotics, the inability to inspect or preserve a complete model may make certification more difficult. Auditors may require evidence about data handling, model changes, known failure modes, and version control. Contractual documentation may partially address this issue, but it cannot provide the same control as local ownership of the complete model artifact.

The difference between open and proprietary models should not be reduced to a simple comparison between freedom and performance. The best choice depends on the application, risk level, organization, data, infrastructure, and deployment scale.

Research groups may prefer open models because they need architectural access, reproducibility, and experimental flexibility. Startups may use proprietary services to develop prototypes quickly and later transition to open models as the product matures. Large enterprises may combine private deployments with managed proprietary models under strict contracts.

Physical AI often benefits from a hybrid approach. An open model can run locally for perception, robot-state processing, action generation, safety-related reasoning, and offline resilience. A proprietary model can provide advanced language understanding, complex planning assistance, technical document interpretation, or occasional high-compute reasoning.

In such an architecture, the proprietary model should not directly control physical actuators. It may propose plans, generate explanations, or assist an operator. Local systems should validate its outputs against robot state, geometry, mission rules, and safety constraints.

Another hybrid strategy uses a proprietary model as a teacher. The teacher can generate synthetic instructions, explanations, plans, annotations, or preference data. An open student model can then be trained or distilled for local deployment. This allows the system to benefit from a highly capable external model while reducing long-term dependence.

However, synthetic data generated by a proprietary teacher must be verified. Errors, biases, or unsafe assumptions can be transferred into the local model. The terms of service must also permit the intended use of generated outputs for training.

A proprietary model can also support development tools without becoming part of the runtime robot. It may help engineers write code, analyze logs, generate test cases, summarize incidents, or create documentation. The deployed robot can still rely on a validated open or internally developed model.

The appropriate model may also differ across the robot hierarchy. A large proprietary model may support cloud-based fleet analytics. A medium-sized open model may perform onboard semantic planning. Small specialized models may handle perception and control. Deterministic software may enforce safety.

This layered approach recognizes that no single model is ideal for every function. It allocates each component according to capability, latency, cost, privacy, and safety requirements.

Model performance must be evaluated in the intended robot environment. A proprietary model may outperform an open model on general benchmarks but perform poorly on specialized sensor data or action generation. An open model fine-tuned on high-quality robot data may be more useful despite having lower general language scores.

Evaluation should include task success, latency, memory, energy, robustness, uncertainty, safety interventions, tool-use reliability, and adaptation cost. The model should be compared as part of the complete system rather than in isolation.

Data availability strongly affects the decision. Organizations with large proprietary robot datasets may gain significant value from open models because they can fine-tune locally. Organizations with little training data may benefit more from the broad knowledge of a proprietary model.

The quality of internal engineering also matters. Operating open models requires expertise in machine learning, infrastructure, optimization, cybersecurity, and model evaluation. Without this capability, an open model may be deployed poorly. A proprietary service may provide more reliable results for a smaller team.

Conversely, an organization building a long-term robot platform may need internal capability regardless of the initial model choice. Dependence on external intelligence can limit product differentiation and strategic control. Developing an internal model stack may therefore be an investment in organizational competence.

Safety requirements should heavily influence the decision. A model used only to summarize inspection reports can tolerate more external dependence than a model involved in navigation or manipulation. The closer the model is to physical control, the greater the need for local execution, predictable latency, version stability, detailed validation, and independent safety enforcement.

Open models can support these requirements because they allow fixed versions and deep integration. However, the organization must provide the safety engineering. Proprietary models may include strong alignment and provider testing, but these safeguards are usually designed for general digital interaction rather than the exact physical risks of a specific robot.

Certification and regulation may favor systems with strong traceability. An organization should be able to identify which model version produced a decision, which data and calibration were used, what safety checks were applied, and whether the model changed after validation.

Open deployments can provide detailed traceability if managed properly. Proprietary deployments require contractual guarantees, version identifiers, audit logs, and change notifications. A provider that changes the model without preserving version control may be unsuitable for a validated robot system.

Cybersecurity must be considered in both cases. Open models create responsibility for securing the entire local stack. Proprietary models create external communication channels, credentials, service dependencies, and possible data exposure. Neither approach is automatically secure.

Threat modeling should examine model extraction, prompt injection, malicious tool calls, compromised updates, poisoned training data, unauthorized access, and service denial. For robots, attacks on the foundation model can affect physical behavior, so security must be integrated with safety.

Sustainability is another consideration. Large proprietary models may provide efficient shared infrastructure, while local open deployments may duplicate hardware across organizations. On the other hand, a smaller optimized open model may consume far less energy than repeatedly calling a much larger cloud model.

The most efficient solution depends on usage volume, model size, hardware utilization, network transfer, and task complexity. Lifecycle analysis should include both training and inference energy.

Long-term availability must also be examined. Open model weights can be archived and maintained internally, but community support may decline. Proprietary services may have professional support, but products can be discontinued or access conditions can change.

An organization should have an exit strategy. This may include maintaining alternative models, standardized interfaces, portable datasets, modular adapters, and benchmark suites that support migration.

The open-versus-proprietary decision may evolve over the product lifecycle. During early research, teams may use several proprietary and open models for comparison. During prototyping, a proprietary API may accelerate development. During productization, an open or licensed on-premise model may provide control. During large-scale operation, a hybrid fleet architecture may balance local autonomy and cloud intelligence.

The decision should be reviewed whenever the application changes. Adding human interaction, direct control, sensitive data, new regulatory requirements, or international deployment can alter the preferred architecture.

For Physical AI, strategic ownership of data may ultimately be more important than ownership of the base model. Robot data contains information about physical interaction, failures, recovery, embodiment, and operational context that general web-scale datasets do not provide. Organizations that systematically collect, synchronize, label, and govern this data can adapt many different base models.

A proprietary model may provide high initial capability, but a well-curated internal dataset can create long-term differentiation. Open models allow this data to be used more freely for specialized training, while proprietary providers may restrict how deeply the customer can modify the model.

The most valuable intellectual property may therefore lie in robot datasets, embodiment adapters, action representations, world models, safety systems, evaluation frameworks, and deployment infrastructure rather than in the base model alone.

Ultimately, open-source and proprietary foundation models represent different distributions of control, responsibility, capability, and risk. Open models provide architectural freedom, deployment independence, customization, reproducibility, and local data control. They also require significant technical expertise, infrastructure, security management, and independent safety validation.

Proprietary models provide convenient access to advanced capabilities, managed infrastructure, frequent improvements, and commercial support. They may also create dependence on external services, limited transparency, data-governance concerns, unpredictable updates, restricted customization, and long-term vendor lock-in.

For Physical AI, the strongest solution is often not an absolute commitment to one category. A carefully designed hybrid architecture can use open models for local perception, embodied reasoning, and controllable deployment while using proprietary systems selectively for advanced language processing, development assistance, or noncritical cloud reasoning.

Regardless of the model source, physical authority must remain bounded. Every model output that can affect a robot should be checked against current sensor data, embodiment constraints, geometric feasibility, mission rules, and deterministic safety limits. The choice of model changes how the system is built, but it does not remove the developer's responsibility for physical reliability.

The central question is therefore not simply whether a model is open or proprietary. It is whether the organization can understand, control, validate, maintain, secure, and govern the complete Physical AI system throughout its lifecycle. The most suitable foundation model is the one whose technical and legal properties support safe physical deployment, strategic independence, and sustainable long-term operation.

오픈소스(Open-Source) 파운데이션 모델과 독점형(Proprietary) 파운데이션 모델 가운데 무엇을 선택할지는 현대 인공지능 개발에서 가장 중요한 전략적 의사결정 중 하나이다. 이러한 선택은 모델 성능뿐 아니라 데이터 거버넌스(Data Governance), 배포 아키텍처(Deployment Architecture), 보안(Security), 맞춤화(Customization), 비용(Cost), 지식재산권(Intellectual Property), 규제 준수(Regulatory Compliance), 장기 유지보수(Long-Term Maintenance), 운영 독립성(Operational Independence)에 영향을 준다. 피지컬 AI(Physical AI)에서는 파운데이션 모델이 로봇, 센서, 산업 장비, 안전 시스템, 디지털 트윈(Digital Twin), 실시간 제어 인프라(Real-Time Control Infrastructure)와 연결되기 때문에 이러한 구분이 더욱 중요해진다. 모델은 단순히 정보를 생성하는 소프트웨어 구성요소가 아니다. 사람, 기계, 생산, 주변 환경에 직접적인 결과를 초래하는 물리적 행동에 영향을 줄 수 있다.

오픈소스와 독점형이라는 용어는 마치 명확하게 분리되는 두 범주처럼 사용되지만, 실제 환경은 훨씬 복잡하다. 완전히 개방된 모델은 모델 아키텍처(Model Architecture), 소스 코드(Source Code), 학습 절차(Training Procedure), 데이터셋(Dataset), 전처리 도구(Preprocessing Tool), 모델 가중치(Model Weight), 평가 결과(Evaluation Result), 수정과 재배포를 허용하는 라이선스(License)를 제공할 수 있다. 반면 일부 모델은 가중치만 공개하고 학습 데이터나 전체 학습 과정은 비공개로 유지한다. 어떤 시스템은 추론 코드(Inference Code)를 공개하지만 상업적 이용, 재배포, 특정 응용 분야에 제한을 둔다. 독점형 모델은 응용 프로그래밍 인터페이스(Application Programming Interface, API), 관리형 클라우드 플랫폼(Managed Cloud Platform), 또는 라이선스 기반 온프레미스 패키지(On-Premise Package)를 통해서만 접근할 수 있다.

따라서 오픈소스 소프트웨어(Open-Source Software), 오픈 웨이트 모델(Open-Weight Model), 오픈 데이터 프로젝트(Open-Data Project), 연구 전용 공개(Research-Only Release), 소스 공개형 시스템(Source-Available System), 완전한 독점형 서비스(Fully Proprietary Service)를 구분하는 것이 중요하다. "개방형(open)"이라는 단어가 모델 수명주기의 모든 부분이 자동으로 투명하다는 의미는 아니다. 마찬가지로 독점형 모델이라고 해서 반드시 모든 배포 방식이 클라우드 기반이거나 맞춤화가 불가능한 것도 아니다. 조직은 마케팅 용어만 믿기보다 실제 라이선스, 기술 문서, 접근 조건, 업데이트 정책(Update Policy), 배포 권한(Deployment Right)을 검토해야 한다.

오픈소스 파운데이션 모델은 모델 구현과 많은 경우 학습된 가중치에 접근할 수 있도록 한다. 이를 통해 조직은 아키텍처를 검토하고, 자체 인프라에서 모델을 실행하고, 구성요소를 수정하고, 행동을 미세조정(Fine-Tuning)하며, 기존 시스템과 깊이 있게 통합할 수 있다. 연구기관, 스타트업, 로봇 제조업체, 산업 개발자에게 이러한 수준의 통제권은 실험 속도를 크게 높일 수 있다.

오픈 모델(Open Model)의 가장 큰 장점 가운데 하나는 배포 독립성(Deployment Independence)이다. 기업은 데이터를 외부 제공업체로 지속적으로 전송하지 않고도 로컬 서버(Local Server), 프라이빗 클라우드(Private Cloud), 엣지 컴퓨터(Edge Computer), 로봇 탑재 하드웨어에서 모델을 운영할 수 있다. 이는 로봇 데이터에 독점 제조 공정, 시설 레이아웃, 제품 설계, 사람의 이미지, 유지보수 정보, 안전에 민감한 운영 기록이 포함될 수 있다는 점에서 피지컬 AI에 특히 중요하다.

로컬 배포(Local Deployment)는 개인정보 보호와 기밀성 위험을 줄일 수 있다. 센서 데이터는 공장, 연구소, 병원, 창고, 차량 내부에 그대로 유지될 수 있다. 조직은 자체 접근 제어(Access Control), 보유 정책(Retention Policy), 암호화 방식(Encryption Method), 감사 절차(Audit Procedure)를 정의할 수 있다. 데이터 주권(Data Sovereignty)과 사이버보안(Cybersecurity)이 중요한 산업에서는 이러한 능력이 최고의 벤치마크 점수보다 더 중요할 수 있다.

오픈 모델은 오프라인 운용(Offline Operation)도 지원한다. 원격 시설, 지하 환경, 야외 농장, 조선소, 네트워크가 제한된 공장에서 동작하는 로봇은 안정적인 클라우드 접속을 확보하지 못할 수 있다. 모델이 로컬에서 실행되면 통신 장애 중에도 핵심 지능을 사용할 수 있다. 이는 운영 복원력(Operational Resilience)을 높이고 외부 서비스에 의존하지 않으면서 필수 기능을 유지할 수 있도록 한다.

맞춤화(Customization)는 또 다른 주요 장점이다. 오픈 모델은 조직별 데이터로 미세조정할 수 있고, 전문 용어에 맞게 적응시키며, 엣지 배포를 위해 압축하고, 새로운 센서 인코더(Sensor Encoder)와 연결하거나, 로봇 행동을 생성하도록 수정할 수 있다. 엔지니어는 입력 계층(Input Layer)을 교체하고, 토큰화(Tokenization)를 변경하며, 교차 모달 모듈(Cross-Modal Module)을 추가하고, 도구 사용 인터페이스(Tool-Use Interface)를 통합하거나, 로봇별 행동 디코더(Robot-Specific Action Decoder)를 부착할 수 있다.

피지컬 AI에서는 이러한 아키텍처 수정이 자주 필요하다. 일반 언어 모델이나 비전 모델은 관절 상태(Joint State), 포인트 클라우드(Point Cloud), 촉각 피드백(Tactile Feedback), 힘 신호(Force Signal), 임무 상태(Mission State), 내비게이션 지도(Navigation Map), 로봇별 행동 스키마(Action Schema)를 자동으로 이해하지 않는다. 개발자는 모달리티 어댑터(Modality Adapter), 시간 인코더(Temporal Encoder), 월드 모델(World Model) 구성요소, 메모리 시스템(Memory System), 안전 헤드(Safety Head)를 추가해야 할 수 있다. 개방형 접근은 이러한 작업을 가능하게 한다.

오픈 모델은 파라미터 효율적 적응(Parameter-Efficient Adaptation)도 지원한다. 저랭크 적응(Low-Rank Adaptation), 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 부분 미세조정(Partial Fine-Tuning)을 이용하면 모든 파라미터를 다시 학습하지 않고도 대형 파운데이션 모델을 특화할 수 있다. 서로 다른 로봇이나 고객이 공통 기반 모델을 공유하면서 개별 어댑터를 사용할 수 있다. 이는 학습 비용을 줄이고 제품군 유지보수를 단순화할 수 있다.

투명성(Transparency)도 중요한 장점이다. 소스 코드와 모델 아키텍처를 검토하여 숨겨진 의존성(Hidden Dependency), 위험한 연산, 외부 통신, 구현 오류를 확인할 수 있다. 연구자는 모델이 정보를 처리하는 방식을 연구하고, 실험을 재현하며, 수정 사항을 비교하고, 독립적인 평가 도구를 구축할 수 있다.

그러나 투명성에도 한계가 있다. 코드와 가중치에 접근할 수 있다고 해서 모델의 내부 행동을 완전히 이해할 수 있는 것은 아니다. 대규모 신경망은 여전히 해석하기 어렵고, 학습 데이터가 완전히 공개되지 않을 수도 있다. 오픈 모델 역시 알려지지 않은 편향(Bias), 암기된 정보(Memorized Information), 보안 취약점(Security Vulnerability), 위험한 행동을 포함할 수 있다. 개방성은 조사를 쉽게 만들지만 검증의 필요성을 제거하지 않는다.

커뮤니티 개발(Community Development)은 오픈소스 모델의 가장 강력한 동력 가운데 하나이다. 연구자와 엔지니어는 최적화된 추론 엔진(Inference Engine), 양자화 방법(Quantization Method), 배포 도구(Deployment Tool), 평가 벤치마크(Evaluation Benchmark), 안전 필터(Safety Filter), 학습 레시피(Training Recipe), 하드웨어 지원을 제공할 수 있다. 한 집단이 발견한 문제를 다른 집단이 문서화하고 수정할 수 있다. 이러한 분산형 혁신(Distributed Innovation)은 하나의 조직 내부 개발보다 더 빠르게 진행될 수 있다.

오픈 생태계(Open Ecosystem)는 여러 종류의 하드웨어를 지원하는 경우가 많다. 모델은 GPU, CPU, 임베디드 가속기(Embedded Accelerator), 엣지 장치, 특수 추론 프로세서에 맞게 최적화될 수 있다. 이는 클라우드 클러스터부터 소형 온보드 시스템까지 계산 플랫폼이 매우 다양한 로보틱스에서 중요한 유연성을 제공한다.

모델 버전을 보존할 수 있다는 점도 전략적으로 중요하다. 조직은 검증된 배포에 필요한 가중치, 코드, 의존성, 토크나이저(Tokenizer), 어댑터, 설정을 보관할 수 있다. 원래 모델 제공업체가 방향을 변경하거나, 서비스를 중단하거나, 가격 정책을 바꾸거나, 호환되지 않는 업데이트를 제공하더라도 시스템을 계속 운용할 수 있다.

이는 재현성(Reproducibility)과 인증(Certification)을 지원한다. 규제 대상이나 안전에 민감한 환경에 배포된 로봇은 검증된 소프트웨어 버전을 유지해야 할 수 있다. 모델 행동이 자동으로 변경되면 이전 검증 결과가 무효화될 수 있다. 오픈 모델을 사용하면 조직이 업데이트 시점과 방법을 직접 통제할 수 있다.

오픈소스 배포는 공급업체 종속(Vendor Lock-In)도 줄일 수 있다. 애플리케이션이 독점형 인터페이스에 깊이 통합되면 다른 제공업체로 이동할 때 상당한 재개발이 필요할 수 있다. 오픈 아키텍처는 모델을 교체하거나 비교하기 쉽게 만들 수 있다. 다만 토큰화, 컨텍스트 형식(Context Format), 멀티모달 인터페이스, 출력 행동 차이로 인해 여전히 마이그레이션 비용(Migration Cost)이 발생할 수 있다.

이러한 장점에도 불구하고 오픈 모델은 조직에 상당한 책임을 부과한다. 조직은 인프라, 사이버보안, 배포 엔지니어링, 성능 최적화, 모니터링, 모델 업데이트, 기술 지원을 자체적으로 제공해야 한다. 무료로 다운로드할 수 있는 모델이라도 무료로 운영되는 것은 아니다.

대형 모델은 고가의 가속기, 대용량 메모리, 빠른 저장장치, 상당한 전력을 요구할 수 있다. 엔지니어는 추론을 최적화하고, 모델 병렬화(Model Parallelism)를 관리하며, 양자화를 지원하고, 호환 가능한 소프트웨어 라이브러리를 유지해야 한다. 총비용(Total Cost)은 API 요금을 지불하지 않아 얻는 표면적 절감액보다 커질 수도 있다.

조직은 안전성도 독립적으로 평가해야 한다. 오픈 모델에는 강력한 정렬(Alignment), 콘텐츠 필터링(Content Filtering), 불확실성 추정(Uncertainty Estimation), 물리적 안전 메커니즘이 포함되지 않을 수 있다. 이러한 기능이 존재하더라도 조직의 로봇, 환경, 운영 요구사항과 일치하지 않을 수 있다.

피지컬 AI에서 성능이 높은 모델을 다운로드하여 로봇에 직접 연결하는 것은 허용될 수 없다. 시스템에는 결정론적 안전 한계(Deterministic Safety Limit), 행동 검증(Action Validation), 충돌 검사(Collision Checking), 런타임 모니터링(Runtime Monitoring), 비상 정지(Emergency Stop), 명확하게 정의된 권한 경계(Authority Boundary)가 필요하다. 기반 모델이 개방형이든 독점형이든 이러한 구성요소는 시스템 개발자의 책임이다.

보안도 중요한 문제이다. 공개된 가중치는 방어자가 분석할 수 있지만 공격자도 분석할 수 있다. 도구 사용 행동, 프롬프트 처리(Prompt Processing), 모델 직렬화(Model Serialization), 배포 인프라의 취약점이 널리 알려질 수 있다. 조직은 보안 권고(Security Advisory)를 지속적으로 확인하고, 다운로드한 아티팩트(Artifact)를 검증하며, 모델 실행을 격리하고, 주변 소프트웨어 스택을 보호해야 한다.

공급망 보안(Supply-Chain Security)은 특히 중요하다. 오픈소스 배포는 많은 라이브러리, 모델 저장소(Model Repository), 컨테이너 이미지(Container Image), 커뮤니티 도구에 의존하는 경우가 많다. 하나의 손상된 의존성이 전체 로봇 시스템에 영향을 줄 수 있다. 따라서 서명된 아티팩트(Signed Artifact), 의존성 검사(Dependency Scanning), 재현 가능한 빌드(Reproducible Build), 통제된 저장소가 필요하다.

라이선스는 복잡할 수 있다. 일부 모델은 연구 목적으로만 사용할 수 있고 상업적 배포를 제한한다. 다른 모델은 특정 산업이나 일정 규모 이상의 조직에서 사용을 금지한다. 일부 라이선스는 출처 표시(Attribution), 수정 사항 공개, 동일 조건의 재배포를 요구한다. 데이터셋 라이선스는 모델 라이선스와 다를 수 있다.

조직은 모델을 미세조정하고, 재배포하고, 제품에 내장하고, 판매하거나, 로봇 제품의 일부로 제공할 수 있는지를 확인해야 한다. 특히 국제적으로 배포하거나 상용 플랫폼에 통합할 경우 법률 검토(Legal Review)가 중요하다.

지식재산 위험(Intellectual-Property Risk)도 존재한다. 오픈 모델의 학습 데이터가 완전히 문서화되지 않았을 수 있다. 모델이 저작권 자료, 기밀 정보, 개인정보를 재생산할 수 있다. 로컬 배포라고 해서 모델 출력이나 데이터 처리에 대한 책임이 사라지는 것은 아니다.

독점형 파운데이션 모델은 다른 개발 및 접근 방식을 따른다. 일반적으로 상업적 제공업체가 모델을 학습하고, 호스팅하고, 유지보수하고, 업데이트한다. 사용자는 API, 관리형 서비스(Managed Service), 소프트웨어 플랫폼, 기업 계약(Enterprise Agreement)을 통해 접근한다.

독점형 모델의 가장 눈에 띄는 장점은 편의성(Convenience)이다. 조직은 전체 학습 및 추론 인프라를 구축하지 않고도 높은 능력의 모델을 사용할 수 있다. 제공업체가 모델 서빙(Model Serving), 확장성(Scaling), 하드웨어 활용률, 최적화, 많은 보안 운영을 담당한다.

이는 프로토타이핑(Prototyping)을 가속할 수 있다. 팀은 비교적 적은 설정만으로 언어 이해, 멀티모달 추론, 코드 생성, 문서 분석, 계획 지원을 시험할 수 있다. 간단한 인터페이스를 통해 프런티어 규모 모델(Frontier-Scale Model)에 접근할 수 있다는 점은 초기 실험의 진입 장벽을 낮춘다.

독점형 제공업체는 높은 성능을 제공할 수도 있다. 대규모 데이터셋, 막대한 계산 인프라, 특수 학습 방법, 안전 튜닝(Safety Tuning), 평가, 추론 최적화에 투자할 수 있기 때문이다. 개별 조직이 이러한 역량을 독립적으로 재현하기는 어려울 수 있다.

관리형 서비스는 지역 중복성(Geographic Redundancy), 자동 확장(Auto-Scaling), 사용량 모니터링, 인증(Authentication), 로깅(Logging), 기업 지원(Enterprise Support), 서비스 수준 협약(Service-Level Agreement)과 같은 신뢰성 기능을 제공할 수 있다. 이러한 기능은 고객 대상 애플리케이션에 유용하다.

독점형 모델은 자주 개선될 수 있다. 제공업체는 고객이 모델을 재학습하지 않아도 추론 품질, 멀티모달 능력, 지연 시간, 도구 사용, 컨텍스트 길이, 보안을 개선할 수 있다. 빠르게 발전하는 응용 분야에서는 새로운 능력에 신속하게 접근할 수 있다는 장점이 있다.

그러나 제공업체가 통제하는 업데이트는 로보틱스에서 위험을 만들 수 있다. 고객의 소프트웨어가 변경되지 않았더라도 모델 행동이 달라질 수 있다. 프롬프트 해석, 거부 행동(Refusal Behavior), 출력 형식(Output Format), 도구 선택, 지연 시간, 모델 신뢰도가 업데이트 이후 변할 수 있다.

디지털 어시스턴트에서는 이러한 변화가 관리 가능할 수 있다. 그러나 광범위한 검증을 거친 로봇 시스템에서는 예상하지 못한 행동 변화가 심각한 문제가 될 수 있다. 조직은 고정된 모델 버전, 업데이트 사전 통지, 회귀 시험 기간(Regression Testing Period), 롤백 메커니즘을 요구할 수 있다.

데이터 거버넌스는 핵심 문제이다. 클라우드 기반 독점형 모델은 로봇 데이터를 외부 인프라로 전송해야 할 수 있다. 제공업체가 강력한 개인정보 보호 약속을 하더라도 조직은 데이터 처리 위치, 보유 기간, 모델 개선 활용 여부, 접근 가능 주체, 적용 법적 관할권(Legal Jurisdiction)을 검토해야 한다.

산업용 로봇 데이터는 공장 레이아웃, 생산 일정, 결함, 유지보수 문제, 제품 설계, 직원 활동, 안전 사고를 노출할 수 있다. 이러한 정보는 외부 전송에 부적합할 수 있다. 따라서 기업 계약, 프라이빗 네트워크(Private Networking), 데이터 격리(Data Isolation), 온프레미스 옵션이 필요할 수 있다.

네트워크 의존성도 한계이다. 클라우드 기반 모델은 통신 지연이 변동하는 상황에서 안전이 중요한 고주파 로봇 제어를 안정적으로 지원할 수 없다. 연결이 끊기고, 대역폭이 제한되며, 원격 서비스 장애가 운영을 중단시킬 수 있다.

따라서 독점형 클라우드 모델은 직접 모터 제어보다 임무 해석(Mission Interpretation), 보고서 생성(Report Generation), 유지보수 지원(Maintenance Assistance), 의미 계획(Semantic Planning), 오프라인 분석과 같은 저주파 작업에 더 적합하다. 로컬 안전 기능과 제어 기능은 독립적으로 유지되어야 한다.

맞춤화는 제한될 수 있다. 제공업체는 프롬프트 엔지니어링(Prompt Engineering), 검색 증강(Retrieval Augmentation), 미세조정, 도구 통합을 제공할 수 있지만, 고객이 전체 아키텍처를 수정하는 것은 일반적으로 불가능하다. 새로운 센서 인코더를 추가하거나, 어텐션 구조를 변경하거나, 특수 행동 토크나이저(Action Tokenizer)를 통합하거나, 내부 활성값(Internal Activation)을 분석하는 것은 불가능할 수 있다.

이는 일부 독점형 모델이 로봇 파운데이션 모델의 중심 아키텍처로 사용되기 어려운 이유이다. 강력한 추론 구성요소가 될 수는 있지만, 완전한 체화 인식-행동 시스템으로 변환하기는 어려울 수 있다.

공급업체 종속도 중요한 우려사항이다. 애플리케이션은 제공업체의 API 형식, 함수 호출 프로토콜(Function-Calling Protocol), 모델 행동, 가격 정책, 계정 정책, 지역별 가용성에 의존하게 될 수 있다. 다른 제공업체로 전환하려면 새로운 평가, 프롬프트 재설계, 안전 시험, 소프트웨어 수정이 필요할 수 있다.

비용도 예측하기 어려울 수 있다. API 가격은 프로토타입 단계에서는 매력적으로 보이지만 대규모 로봇 플릿에 배포하면 빠르게 증가할 수 있다. 로봇은 이미지, 센서 데이터, 상태 갱신을 지속적으로 생성한다. 이러한 모든 정보를 원격 모델로 전송하면 상당한 처리 비용과 통신 비용이 발생할 수 있다.

수명주기 비용(Lifecycle Cost)은 추론 요금, 데이터 전송, 저장, 지원, 중복성, 통합, 향후 마이그레이션을 포함해야 한다. 요청당 능력이 높은 독점형 모델이라도 반복적이고 빈번한 작업에서는 작은 로컬 모델보다 경제성이 낮을 수 있다.

제한된 투명성은 안전 보증(Safety Assurance)을 어렵게 만들 수 있다. 사용자는 모델 아키텍처, 학습 데이터, 최적화 과정, 세부 한계를 알지 못할 수 있다. 제공업체가 안전 보고서와 평가 결과를 공개하더라도 고객은 전체 학습 과정을 독립적으로 재현하기 어렵다.

규제 대상 로보틱스에서는 완전한 모델을 검토하거나 보존할 수 없다는 점이 인증을 어렵게 만들 수 있다. 감사자는 데이터 처리, 모델 변경, 알려진 실패 양상, 버전 관리에 대한 증거를 요구할 수 있다. 계약 문서가 일부 문제를 보완할 수 있지만 완전한 모델 아티팩트를 로컬에서 소유하는 것과 동일한 통제권을 제공하지는 않는다.

오픈 모델과 독점형 모델의 차이를 자유와 성능의 단순한 비교로 축소해서는 안 된다. 최적의 선택은 응용 분야, 위험 수준, 조직 역량, 데이터, 인프라, 배포 규모에 따라 달라진다.

연구 집단은 아키텍처 접근, 재현성, 실험적 유연성이 필요하기 때문에 오픈 모델을 선호할 수 있다. 스타트업은 빠르게 프로토타입을 개발하기 위해 독점형 서비스를 사용한 뒤 제품이 성숙하면 오픈 모델로 전환할 수 있다. 대기업은 엄격한 계약 아래 프라이빗 배포와 관리형 독점형 모델을 결합할 수 있다.

피지컬 AI는 하이브리드 접근(Hybrid Approach)에서 많은 이점을 얻을 수 있다. 오픈 모델은 인식, 로봇 상태 처리, 행동 생성, 안전 관련 추론, 오프라인 복원력을 위해 로컬에서 실행할 수 있다. 독점형 모델은 고급 언어 이해, 복잡한 계획 지원, 기술 문서 해석, 간헐적인 고연산 추론을 제공할 수 있다.

이러한 아키텍처에서 독점형 모델은 물리 액추에이터를 직접 제어해서는 안 된다. 계획을 제안하거나, 설명을 생성하거나, 작업자를 지원할 수 있다. 로컬 시스템은 그 출력을 현재 로봇 상태, 기하학, 임무 규칙, 안전 제약과 비교하여 검증해야 한다.

또 다른 하이브리드 전략은 독점형 모델을 교사 모델(Teacher Model)로 사용하는 것이다. 교사 모델은 합성 지시(Synthetic Instruction), 설명, 계획, 주석(Annotation), 선호 데이터(Preference Data)를 생성할 수 있다. 이후 오픈 학생 모델(Student Model)을 학습하거나 증류하여 로컬에 배포할 수 있다. 이를 통해 높은 능력의 외부 모델을 활용하면서 장기적인 의존성을 줄일 수 있다.

그러나 독점형 교사 모델이 생성한 합성 데이터는 검증되어야 한다. 오류, 편향, 위험한 가정이 로컬 모델로 전이될 수 있다. 또한 서비스 약관이 생성 결과를 학습에 사용하는 것을 허용하는지도 확인해야 한다.

독점형 모델은 런타임 로봇의 일부가 되지 않고 개발 도구로 사용할 수도 있다. 엔지니어가 코드를 작성하고, 로그를 분석하고, 테스트 사례(Test Case)를 생성하고, 사고를 요약하고, 문서를 작성하는 데 도움을 줄 수 있다. 실제 배포된 로봇은 검증된 오픈 모델이나 자체 개발 모델을 사용할 수 있다.

로봇 계층에 따라 적절한 모델도 달라질 수 있다. 대형 독점형 모델은 클라우드 기반 플릿 분석(Fleet Analytics)을 지원할 수 있다. 중간 크기의 오픈 모델은 온보드 의미 계획을 수행할 수 있다. 작은 전문 모델은 인식과 제어를 담당할 수 있다. 결정론적 소프트웨어는 안전을 강제할 수 있다.

이러한 계층형 접근은 하나의 모델이 모든 기능에 이상적이지 않다는 점을 인정한다. 각 구성요소를 능력, 지연 시간, 비용, 개인정보 보호, 안전 요구사항에 따라 배치한다.

모델 성능은 실제 로봇 환경에서 평가해야 한다. 독점형 모델이 일반 벤치마크에서 오픈 모델보다 우수하더라도 전문 센서 데이터나 행동 생성에서는 성능이 낮을 수 있다. 고품질 로봇 데이터로 미세조정된 오픈 모델은 일반 언어 점수가 낮더라도 실제로 더 유용할 수 있다.

평가는 작업 성공률(Task Success Rate), 지연 시간, 메모리, 에너지, 강건성, 불확실성, 안전 개입, 도구 사용 신뢰성, 적응 비용을 포함해야 한다. 모델 하나를 따로 비교하기보다 전체 시스템의 일부로 비교해야 한다.

데이터 가용성(Data Availability)은 선택에 큰 영향을 준다. 대규모 독점 로봇 데이터셋을 가진 조직은 로컬에서 자유롭게 미세조정할 수 있는 오픈 모델로부터 큰 가치를 얻을 수 있다. 학습 데이터가 거의 없는 조직은 독점형 모델의 폭넓은 사전지식에서 더 많은 이점을 얻을 수 있다.

내부 엔지니어링 역량의 품질도 중요하다. 오픈 모델을 운영하려면 머신러닝, 인프라, 최적화, 사이버보안, 모델 평가에 대한 전문성이 필요하다. 이러한 역량이 없다면 오픈 모델을 부적절하게 배포할 수 있다. 소규모 팀에는 독점형 서비스가 더 안정적인 결과를 제공할 수 있다.

반대로 장기적인 로봇 플랫폼을 구축하는 조직은 초기 모델 선택과 관계없이 내부 역량이 필요할 수 있다. 외부 지능에 대한 의존은 제품 차별화와 전략적 통제를 제한할 수 있다. 따라서 내부 모델 스택(Model Stack)을 구축하는 것은 조직 역량에 대한 투자일 수 있다.

안전 요구사항은 모델 선택에 큰 영향을 주어야 한다. 검사 보고서를 요약하는 모델은 내비게이션이나 조작에 관여하는 모델보다 외부 의존성을 더 많이 허용할 수 있다. 모델이 물리적 제어에 가까울수록 로컬 실행, 예측 가능한 지연 시간, 버전 안정성, 세부 검증, 독립적인 안전 강제의 중요성이 커진다.

오픈 모델은 고정 버전과 깊은 통합을 허용하므로 이러한 요구사항을 지원할 수 있다. 그러나 안전 엔지니어링은 조직이 직접 제공해야 한다. 독점형 모델은 강력한 정렬과 제공업체의 시험을 포함할 수 있지만, 이러한 보호 장치는 일반적인 디지털 상호작용을 위해 설계된 경우가 많고 특정 로봇의 물리적 위험과 정확히 일치하지 않을 수 있다.

인증과 규제는 높은 추적 가능성(Traceability)을 가진 시스템을 선호할 수 있다. 조직은 어떤 모델 버전이 결정을 생성했는지, 어떤 데이터와 보정이 사용되었는지, 어떤 안전 검사가 적용되었는지, 검증 이후 모델이 변경되었는지를 확인할 수 있어야 한다.

오픈 배포는 적절하게 관리하면 세부적인 추적 가능성을 제공할 수 있다. 독점형 배포는 계약상 보장, 버전 식별자, 감사 로그, 변경 통지를 요구한다. 모델을 변경하면서 버전 관리를 제공하지 않는 서비스는 검증된 로봇 시스템에 적합하지 않을 수 있다.

사이버보안은 두 방식 모두에서 고려해야 한다. 오픈 모델은 전체 로컬 스택을 보호할 책임을 조직에 부과한다. 독점형 모델은 외부 통신 채널, 인증 정보(Credential), 서비스 의존성, 데이터 노출 가능성을 만든다. 어느 방식도 자동으로 안전하지 않다.

위협 모델링(Threat Modeling)은 모델 추출(Model Extraction), 프롬프트 인젝션(Prompt Injection), 악의적인 도구 호출(Malicious Tool Call), 손상된 업데이트(Compromised Update), 오염된 학습 데이터(Poisoned Training Data), 무단 접근, 서비스 거부(Service Denial)를 포함해야 한다. 로봇에서는 파운데이션 모델 공격이 물리적 행동에 영향을 줄 수 있으므로 보안과 안전을 통합해야 한다.

지속가능성(Sustainability)도 고려해야 한다. 대형 독점형 모델은 효율적인 공유 인프라를 제공할 수 있지만, 로컬 오픈 배포는 여러 조직에서 하드웨어를 중복 구축할 수 있다. 반면 작은 최적화 오픈 모델은 훨씬 큰 클라우드 모델을 반복 호출하는 것보다 훨씬 적은 에너지를 사용할 수 있다.

가장 효율적인 해결책은 사용량, 모델 크기, 하드웨어 활용률, 네트워크 전송량, 작업 복잡도에 따라 달라진다. 수명주기 분석(Lifecycle Analysis)은 학습과 추론의 에너지를 모두 포함해야 한다.

장기적 가용성(Long-Term Availability)도 검토해야 한다. 오픈 모델 가중치는 보관하여 내부적으로 유지할 수 있지만 커뮤니티 지원이 줄어들 수 있다. 독점형 서비스는 전문적인 지원을 제공할 수 있지만 제품이 중단되거나 접근 조건이 변경될 수 있다.

조직은 종료 전략(Exit Strategy)을 준비해야 한다. 대체 모델, 표준화된 인터페이스, 이식 가능한 데이터셋(Portable Dataset), 모듈형 어댑터(Modular Adapter), 마이그레이션을 지원하는 벤치마크 모음을 유지할 수 있다.

오픈 모델과 독점형 모델의 선택은 제품 수명주기에 따라 변할 수 있다. 초기 연구에서는 여러 독점형 모델과 오픈 모델을 비교할 수 있다. 프로토타이핑 단계에서는 독점형 API가 개발을 가속할 수 있다. 제품화 단계에서는 오픈 모델이나 라이선스된 온프레미스 모델이 통제권을 제공할 수 있다. 대규모 운영 단계에서는 로컬 자율성과 클라우드 지능을 균형 있게 결합하는 하이브리드 플릿 아키텍처가 사용될 수 있다.

응용 분야가 변경될 때마다 선택을 다시 검토해야 한다. 인간 상호작용, 직접 제어, 민감한 데이터, 새로운 규제 요구사항, 국제 배포가 추가되면 선호되는 아키텍처가 바뀔 수 있다.

피지컬 AI에서는 기반 모델 자체의 소유권보다 데이터에 대한 전략적 소유권이 더 중요할 수 있다. 로봇 데이터에는 일반적인 웹 규모 데이터셋에 없는 물리적 상호작용, 실패, 복구, 체화, 운영 문맥에 대한 정보가 포함된다. 이러한 데이터를 체계적으로 수집하고, 동기화하고, 라벨링하고, 관리하는 조직은 다양한 기반 모델을 특화할 수 있다.

독점형 모델은 높은 초기 능력을 제공할 수 있지만, 잘 정제된 내부 데이터셋은 장기적인 차별화를 만든다. 오픈 모델은 이러한 데이터를 전문 학습에 더 자유롭게 활용할 수 있도록 하지만, 독점형 제공업체는 고객이 모델을 얼마나 깊이 수정할 수 있는지를 제한할 수 있다.

따라서 가장 가치 있는 지식재산은 기반 모델 자체보다 로봇 데이터셋, 체화 어댑터(Embodiment Adapter), 행동 표현(Action Representation), 월드 모델, 안전 시스템, 평가 프레임워크, 배포 인프라에 존재할 수 있다.

궁극적으로 오픈소스 파운데이션 모델과 독점형 파운데이션 모델은 통제권, 책임, 능력, 위험을 서로 다르게 분배한다. 오픈 모델은 아키텍처의 자유, 배포 독립성, 맞춤화, 재현성, 로컬 데이터 통제를 제공한다. 동시에 상당한 기술 전문성, 인프라, 보안 관리, 독립적인 안전 검증을 요구한다.

독점형 모델은 고급 능력에 대한 편리한 접근, 관리형 인프라, 빈번한 개선, 상업적 지원을 제공한다. 그러나 외부 서비스 의존, 제한된 투명성, 데이터 거버넌스 우려, 예측하기 어려운 업데이트, 제한된 맞춤화, 장기적인 공급업체 종속을 초래할 수 있다.

피지컬 AI에서는 하나의 범주만을 절대적으로 선택하는 것보다 신중하게 설계된 하이브리드 아키텍처가 가장 강력한 해결책이 되는 경우가 많다. 오픈 모델은 로컬 인식, 체화 추론(Embodied Reasoning), 제어 가능한 배포에 사용할 수 있고, 독점형 시스템은 고급 언어 처리, 개발 지원, 중요도가 낮은 클라우드 추론에 선택적으로 사용할 수 있다.

모델의 출처와 관계없이 물리적 권한(Physical Authority)은 제한되어야 한다. 로봇에 영향을 미칠 수 있는 모든 모델 출력은 현재 센서 데이터, 체화 제약, 기하학적 실행 가능성(Geometric Feasibility), 임무 규칙, 결정론적 안전 한계와 비교되어야 한다. 모델 선택은 시스템 구축 방식을 바꾸지만 물리적 신뢰성에 대한 개발자의 책임을 제거하지 않는다.

따라서 핵심 질문은 단순히 모델이 오픈소스인지 독점형인지가 아니다. 조직이 전체 피지컬 AI 시스템을 수명주기 전반에 걸쳐 이해하고, 통제하고, 검증하고, 유지보수하고, 보호하고, 관리할 수 있는지가 핵심이다. 가장 적합한 파운데이션 모델은 기술적·법적 특성이 안전한 물리적 배포, 전략적 독립성, 지속 가능한 장기 운영을 지원하는 모델이다.

##  

## 1.10 Physical AI Foundation Model Landscape Overview

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

The Physical AI foundation model landscape represents the convergence of large-scale artificial intelligence, robotics, multimodal perception, embodied learning, simulation, world modeling, and real-time control. Unlike conventional foundation models that operate primarily over text, images, audio, or software environments, Physical AI foundation models must connect perception and reasoning to actions that change the physical world. Their outputs may move a robot, manipulate an object, navigate through a facility, inspect industrial equipment, interact with people, or coordinate multiple autonomous systems. The landscape is therefore not defined by a single model family. It consists of interconnected model architectures, datasets, simulation platforms, robot-learning frameworks, deployment tools, and safety systems.

The field has evolved from task-specific robot policies toward generalist models that can perform multiple tasks, interpret natural-language instructions, transfer knowledge across environments, and adapt to different robot embodiments. Traditional robot learning generally required a separate policy for each task, platform, and environment. A manipulation policy trained to place one object using one robot arm often transferred poorly to another object, workspace, or robot. Physical AI foundation models attempt to overcome this fragmentation by learning reusable representations from large and diverse collections of robot trajectories, human demonstrations, web-scale visual-language data, simulation data, and synthetic experience.

Several overlapping model categories define the current landscape. These include visual foundation models, vision-language models, vision-language-action models, generalist robot policies, world foundation models, multimodal large language models, action-generative policies, and embodiment-specific control models. These categories should not be treated as completely separate. A practical robot foundation model may combine a pretrained vision-language backbone, a temporal state encoder, an action-generation module, a world model, external memory, and deterministic control software.

Visual foundation models provide a major part of the perceptual foundation. They learn general image representations from large datasets using supervised, contrastive, masked-reconstruction, or multimodal objectives. In robotics, these representations support object recognition, segmentation, pose estimation, anomaly detection, affordance prediction, scene understanding, and visual navigation. Models derived from Vision Transformer architectures are especially influential because image patches can be represented as tokens and integrated naturally with language, state, and action tokens.

However, a visual foundation model alone is not a Physical AI model. It may recognize an object without determining whether the object is reachable, graspable, stable, safe to move, or relevant to the current mission. Physical grounding requires geometry, temporal context, embodiment state, action consequence, and sensor feedback. The visual backbone must therefore be integrated with depth, point clouds, proprioception, force sensing, maps, and robot-specific constraints.

Vision-language models form another major layer of the landscape. These systems connect visual observations with language instructions, questions, semantic concepts, and symbolic knowledge. They can identify objects described in natural language, explain scenes, answer spatial questions, and support high-level task planning. Their broad knowledge is often acquired from large-scale image-text and web datasets rather than robot interaction alone.

The limitation of a vision-language model is that understanding and description do not automatically produce executable behavior. A model may correctly explain how a task should be performed while generating actions that violate geometry, timing, dynamics, or safety constraints. The development of vision-language-action models therefore represents a critical transition from semantic understanding to embodied control.

Vision-language-action models, commonly called VLA models, extend vision-language systems by representing robot actions as model outputs. Observations and language instructions are processed together, and the model generates action tokens, continuous action values, trajectories, or action chunks. RT-2 was an influential example of this direction. It incorporated web-scale vision-language knowledge into an end-to-end robot-control model and represented actions in a token-compatible form, demonstrating that semantic knowledge acquired from nonrobotic data could improve robotic generalization.

The importance of RT-2 was not that it solved general robotics, but that it demonstrated a scalable connection between internet-scale semantic pretraining and physical control. It showed that concepts learned from images and language could support novel object relationships and instruction following in robotic manipulation. This established the VLA model as one of the dominant architectural directions for Physical AI.

Open X-Embodiment and related multi-robot datasets changed the field by making cross-embodiment learning more practical. Instead of training only on one robot, researchers began combining demonstrations from different institutions, platforms, cameras, tasks, and action spaces. The central assumption is that diverse robot data contains transferable structure even when the hardware interfaces differ.

Octo is a prominent open-source generalist robot policy developed around this idea. It is a transformer-based diffusion policy pretrained on approximately 800,000 robot trajectories drawn from multiple datasets and embodiments. It supports language instructions, goal images, observation histories, different camera configurations, and adaptation to new sensory and action spaces. Its architecture emphasizes efficient fine-tuning rather than assuming that one fixed output interface can directly control every robot.

Octo represents an important branch of the landscape in which robot policies are designed as adaptable initializations. The goal is not necessarily perfect zero-shot control. Instead, a broadly pretrained policy provides a strong starting point that can be specialized with a relatively small amount of target-robot data. This approach recognizes that embodiment differences remain substantial and that adaptation is often more realistic than universal direct control.

OpenVLA represents another important open-model direction. It is a seven-billion-parameter vision-language-action model pretrained on approximately 970,000 robot episodes from the Open X-Embodiment ecosystem. Its architecture combines visual encoders, a projection layer, a language-model backbone, and tokenized robot actions. The project provides model weights and training tools and supports parameter-efficient adaptation to new robot configurations.

OpenVLA illustrates how language-model architectures can be repurposed for robot control. Images are converted into visual embeddings, instructions are represented as language tokens, and robot actions are generated in a tokenized form. These tokens are subsequently decoded into continuous commands. This creates a direct architectural connection between large language modeling and action-sequence generation.

The open VLA ecosystem is strategically important because it allows researchers and robot manufacturers to inspect models, modify input interfaces, add new sensors, change action representations, perform local fine-tuning, and deploy fixed versions. Open models such as Octo and OpenVLA also make it easier to study the effects of robot-data mixtures, visual backbones, action tokenization, diffusion decoding, and parameter-efficient adaptation.

Physical Intelligence's π0 family represents a related but distinct effort focused on generalist manipulation and continuous action generation. The original π0 combined a pretrained vision-language backbone with diverse cross-embodiment robot data and an action expert that generated continuous actions through flow matching. The training mixture included data from multiple robot platforms and dexterous manipulation tasks.

This approach highlights a major architectural question in the landscape: whether robot actions should be represented as discrete tokens or generated through continuous probabilistic models. Tokenized actions integrate naturally with autoregressive language-model training but may introduce quantization error. Diffusion and flow-based policies can model smooth, multimodal continuous action distributions but may require different inference and training mechanisms.

The π0 ecosystem also introduced FAST, an action-tokenization approach intended to make generalist policy training more efficient. The π0-FAST system was reported to train substantially faster than the original π0 while maintaining similar performance, illustrating that action representation is becoming as important as visual and language representation in scaling robot foundation models.

Later work in the same family focused more explicitly on open-world generalization and long-horizon household manipulation. π0.5 was designed to combine heterogeneous training signals and improve performance in previously unseen environments. The developers also acknowledged continuing errors in both semantic reasoning and motor execution, reflecting the broader reality that current VLA systems remain incomplete and require careful closed-loop validation.

NVIDIA's Isaac GR00T ecosystem represents another major branch of the Physical AI landscape, especially for humanoid and cross-embodiment robotics. GR00T is not only a model but part of a wider development platform that includes data collection, simulation, synthetic-data generation, training, post-training, and deployment tools. NVIDIA describes GR00T as a platform for robot foundation models and generalized humanoid reasoning and skills.

The original GR00T N1 model used a dual-system architecture separating slower reasoning from faster action generation. Its training data included egocentric human videos, real robot trajectories, simulated trajectories, and synthetic data. This reflects a broader trend toward combining human behavior data, robot demonstrations, and generated experience rather than relying on one data source.

The GR00T model line subsequently evolved through updated versions intended to improve generalization, language following, and post-training performance across humanoid systems. The broader significance of the GR00T ecosystem lies in its integration with simulation and synthetic-data infrastructure. Robot foundation models require far more than pretrained weights; they require mechanisms for creating data, adapting to hardware, evaluating behavior, and transferring policies from simulation into reality.

World foundation models constitute another major layer of the landscape. These models learn representations of how physical scenes change over time. Instead of producing only labels or actions, they can generate or predict future visual states, latent states, trajectories, or environmental outcomes. In robotics, a world model may be used to generate training data, evaluate candidate actions, simulate unusual events, or support model-based planning.

World models are particularly important because real robot data is expensive and slow to collect. Simulation and generative models can produce variations in objects, backgrounds, lighting, viewpoints, failure modes, sensor noise, and physical conditions. NVIDIA's use of Cosmos world foundation models together with GR00T development illustrates the growing connection between generative world modeling and robot-policy training.

A world foundation model should not be confused with a validated physics engine. Generative models may produce visually convincing but physically inaccurate futures. They are useful for data augmentation, representation learning, scenario generation, and approximate planning, but safety-critical decisions require geometric checks, physical constraints, and real-world validation.

Simulation platforms therefore remain a fundamental part of the landscape. Physics-based simulators such as those used in Isaac Sim, Isaac Lab, MuJoCo, Gazebo, and related environments allow developers to test policies before real deployment. They support domain randomization, sensor simulation, reinforcement learning, synthetic demonstrations, and hardware-in-the-loop evaluation.

The relationship between simulation and foundation models is becoming increasingly bidirectional. Simulators generate structured experience for model training, while foundation models generate scenes, tasks, trajectories, assets, and scenario variations for simulation. This combination can dramatically increase data diversity, although it also introduces the risk that models learn simulator-specific artifacts.

Data ecosystems are arguably the most important competitive layer in Physical AI. General text and image models benefit from internet-scale data, but physical interaction data is far more limited. Robot trajectories must include synchronized observations, actions, timestamps, coordinate frames, calibration, robot identity, task information, and outcome labels.

Cross-embodiment datasets attempt to aggregate data across multiple robots, but heterogeneity remains a major problem. Different datasets may use joint positions, joint velocities, Cartesian poses, gripper commands, or task-specific control formats. Camera placement, image resolution, control frequency, language labeling, and calibration quality may also differ.

This has led to growing interest in standardized robot-data formats and common action representations. The long-term objective is to create a robotics equivalent of web-scale pretraining, where data from many robots and sites can be combined without discarding embodiment-specific information. Achieving this objective requires detailed metadata, normalization procedures, converters, quality filters, and evaluation protocols.

Human demonstration data remains a critical resource. Teleoperation allows people to provide successful robot trajectories, while egocentric video provides large-scale information about human interaction. However, video does not directly reveal robot actions, forces, joint configurations, or executable trajectories. Models must learn a correspondence between human behavior and robot embodiment.

Synthetic data can fill gaps, but it must be validated. Generated trajectories may violate robot limits or represent physically implausible interactions. High-quality Physical AI datasets will therefore combine real robot demonstrations, autonomous experience, simulation, human video, web data, synthetic scenes, and failure-recovery examples.

Another important division in the landscape is between generalist models and specialized foundation models. Generalist VLA systems aim to support many manipulation tasks and environments. Specialized models may focus on autonomous driving, navigation, industrial inspection, dexterous hands, humanoid locomotion, drones, or medical robotics.

Specialized models may achieve higher reliability because their operational domain is narrower. An industrial inspection model can be trained around known equipment, defect types, viewpoints, and sensor conditions. A generalist household robot model must handle much greater variability but may be less predictable.

The likely future is not a single universal robot model replacing all specialized systems. A more practical architecture combines broad foundation models with domain-specific adapters, perception modules, action decoders, planners, and controllers. The foundation model provides semantic understanding and transfer, while specialized components provide accuracy and operational assurance.

Model architecture is also becoming increasingly hierarchical. A slow reasoning model may interpret goals, decompose tasks, query tools, and generate subgoals. A faster action model may convert these subgoals into short motion sequences. Conventional controllers execute high-frequency commands, and independent safety systems enforce hard limits.

This hierarchy addresses the mismatch between large-model inference and real-time control. Language and semantic planning may operate at relatively low frequencies, but motor stabilization, force control, emergency braking, and collision response require deterministic and much faster loops.

Memory systems are emerging as another important component. Long robot missions cannot store all raw observations inside a continuously expanding transformer context. Structured mission memory, semantic maps, object databases, digital twins, episodic summaries, and retrieval systems allow the robot to preserve relevant information without repeatedly processing its entire history.

Tool-using robot foundation models form another part of the landscape. A model may call a motion planner, grasp planner, database, inspection algorithm, mapping service, fleet-management system, or digital twin. This approach allows the foundation model to coordinate validated engineering tools instead of attempting to solve every problem internally.

For industrial Physical AI, tool orchestration may be more important than fully end-to-end control. A model can interpret a mission, select an inspection procedure, query equipment information, request a collision-free path, monitor completion, and generate a report while deterministic subsystems perform localization, navigation, and safety checking.

Open and proprietary ecosystems coexist throughout the landscape. Open models provide research access, customization, local deployment, and architectural control. Proprietary systems may offer stronger general reasoning, larger training resources, commercial support, or managed infrastructure. Many real deployments are likely to use hybrid architectures.

A proprietary cloud model may support complex document interpretation or occasional high-level reasoning, while an open model handles local perception and action. A proprietary model may also serve as a teacher that generates labels, plans, or synthetic instructions for an open student model. Physical authority should nevertheless remain constrained by local safety and control systems.

Evaluation remains one of the weakest and most important parts of the landscape. General language and vision benchmarks cannot establish robot reliability. Physical AI models require closed-loop testing that measures task completion, geometric accuracy, force, collision, timing, recovery, uncertainty, and human intervention.

Different projects often use different robots, tasks, scenes, and evaluation procedures, making direct comparison difficult. A model that performs well on tabletop manipulation may not generalize to mobile manipulation, industrial inspection, or humanoid operation. Future benchmarks must document embodiment, sensors, action interfaces, safety constraints, and operational conditions.

The field also faces a distinction between demonstration quality and deployment reliability. Impressive videos can show selected successful trials without revealing failure rate, intervention, latency, or environmental limitations. A mature landscape requires transparent reporting of unsuccessful trials, uncertainty, recovery behavior, and operational design domains.

Safety architectures remain external to most foundation models. Current VLA models can propose or generate actions, but they do not replace functional-safety systems, collision detection, emergency stopping, force limits, or certified control logic. Physical AI foundation models should therefore be viewed as intelligent components within a layered robot architecture rather than complete autonomous systems.

Cybersecurity is becoming increasingly important as models gain tool access and physical authority. Prompt injection, corrupted sensor inputs, malicious task instructions, compromised model updates, poisoned training data, or unsafe tool calls can affect robot behavior. Security evaluation must therefore be connected directly to physical safety.

Edge deployment is another defining challenge. Many frontier models are too large for practical onboard execution. Quantization, pruning, knowledge distillation, efficient attention, action chunking, caching, and smaller robot-specific policies are necessary to meet latency, memory, thermal, and energy constraints.

Cloud-edge architectures can distribute computation, but network dependence limits their use for safety-critical control. Core navigation, stopping, and fallback behavior must remain local. Cloud resources are more appropriate for fleet learning, model updates, difficult semantic reasoning, or offline analysis.

The landscape is gradually moving from isolated model releases toward complete Physical AI platforms. These platforms combine data engines, teleoperation, simulation, synthetic data, pretrained policies, post-training, evaluation, deployment, monitoring, and fleet feedback. The strategic advantage will increasingly come from the full learning loop rather than the base model alone.

Robot companies that can collect high-quality operational data, detect failures, produce recovery demonstrations, retrain models, validate updates, and deploy them safely across a fleet may gain more long-term value than organizations that only access a powerful pretrained model.

The next phase of the landscape will likely emphasize open-world generalization, cross-embodiment transfer, reasoning-action integration, autonomous data collection, and continuous improvement. Models will increasingly combine human knowledge from web-scale data with physical experience from robots and simulation.

They will also move toward more adaptive computation. Simple tasks may use small local policies, while unfamiliar situations invoke larger reasoning models, additional sensing, world-model simulation, or human assistance. Computation will be allocated according to task difficulty and uncertainty rather than applied uniformly.

The long-term objective is a foundation model that understands language and visual context, represents the robot body, predicts physical consequences, adapts to new tasks, and generates reliable action. Current systems demonstrate meaningful parts of this vision, but no model yet provides unrestricted general-purpose physical intelligence.

The present landscape should therefore be understood as an early but rapidly developing stack of capabilities. VLA models connect semantics to action. Generalist policies enable transfer across tasks and robots. World foundation models expand data and prediction. Simulation platforms provide scalable experience. Open ecosystems enable customization, while commercial platforms integrate development and deployment.

For practical Physical AI, the strongest architecture is likely to combine these elements rather than select one model as a complete solution. A multimodal foundation model can interpret the mission and environment. A robot policy can generate action chunks. A world model can estimate consequences. Specialized planners can enforce geometry. Low-level controllers can maintain stability. Independent safety systems can retain final authority.

The Physical AI foundation model landscape is therefore not simply a competition among model names. It is the development of a new computing stack in which data, simulation, perception, language, reasoning, world modeling, action generation, control, safety, and lifecycle learning are integrated. Its success will be determined not only by model scale, but by how reliably broad artificial intelligence can be grounded in specific robot bodies and transformed into safe, useful, and repeatable physical behavior.

피지컬 AI 파운데이션 모델 생태계는 대규모 인공지능(Large-Scale Artificial Intelligence), 로보틱스(Robotics), 멀티모달 인식(Multimodal Perception), 체화 학습(Embodied Learning), 시뮬레이션(Simulation), 월드 모델(World Model), 실시간 제어(Real-Time Control)가 융합되는 영역이다. 텍스트, 이미지, 음성, 소프트웨어 환경을 주로 다루는 기존의 파운데이션 모델과 달리, 피지컬 AI 파운데이션 모델은 인식과 추론을 실제 물리적 세계를 변화시키는 행동으로 연결해야 한다. 이러한 모델의 출력은 로봇을 이동시키고, 물체를 조작하며, 시설 내부를 자율 주행하고, 산업 설비를 검사하고, 사람과 상호작용하며, 여러 자율 시스템을 동시에 조정할 수 있다. 따라서 현재의 생태계는 하나의 모델 계열(Model Family)로 정의되지 않는다. 다양한 모델 아키텍처(Model Architecture), 데이터셋(Dataset), 시뮬레이션 플랫폼(Simulation Platform), 로봇 학습 프레임워크(Robot Learning Framework), 배포 도구(Deployment Tool), 안전 시스템(Safety System)이 서로 연결된 거대한 생태계를 형성하고 있다.

이 분야는 작업(Task)별로 개별 정책(Policy)을 학습하던 기존 로봇 학습 방식에서, 여러 작업을 수행하고, 자연어 명령을 이해하며, 서로 다른 환경으로 지식을 전이하고, 다양한 로봇 몸체(Embodiment)에 적응할 수 있는 범용 모델(Generalist Model)로 발전해 왔다. 전통적인 로봇 학습에서는 작업마다, 플랫폼마다, 환경마다 별도의 정책을 학습해야 했다. 특정 로봇 팔에서 하나의 물체를 집도록 학습된 정책은 다른 물체나 다른 작업 공간, 다른 로봇으로 거의 전이되지 않았다. 피지컬 AI 파운데이션 모델은 대규모 로봇 궤적(Robot Trajectory), 인간 시연(Human Demonstration), 웹 규모 비전-언어 데이터(Web-Scale Vision-Language Data), 시뮬레이션 데이터, 합성 경험(Synthetic Experience)을 함께 학습하여 재사용 가능한 표현(Reusable Representation)을 구축함으로써 이러한 단절을 극복하려고 한다.

현재 생태계는 여러 개의 서로 겹치는 모델 범주(Model Category)로 구성된다. 여기에는 비전 파운데이션 모델(Visual Foundation Model), 비전-언어 모델(Vision-Language Model), 비전-언어-행동 모델(Vision-Language-Action Model), 범용 로봇 정책(Generalist Robot Policy), 월드 파운데이션 모델(World Foundation Model), 멀티모달 대규모 언어 모델(Multimodal Large Language Model), 행동 생성 정책(Action-Generative Policy), 체화 특화 제어 모델(Embodiment-Specific Control Model)이 포함된다. 이러한 범주는 서로 완전히 독립적인 것이 아니다. 실제 로봇 파운데이션 모델은 사전학습된 비전-언어 백본(Vision-Language Backbone), 시간 상태 인코더(Temporal State Encoder), 행동 생성 모듈(Action Generation Module), 월드 모델, 외부 메모리(External Memory), 결정론적 제어 소프트웨어(Deterministic Control Software)를 함께 결합하는 경우가 많다.

비전 파운데이션 모델은 지각(Perception)의 핵심 기반을 제공한다. 이러한 모델은 지도학습(Supervised Learning), 대조학습(Contrastive Learning), 마스크 복원(Masked Reconstruction), 멀티모달 학습을 이용하여 대규모 데이터셋에서 일반적인 이미지 표현(Image Representation)을 학습한다. 로보틱스에서는 객체 인식(Object Recognition), 분할(Segmentation), 자세 추정(Pose Estimation), 이상 탐지(Anomaly Detection), 어포던스(Affordance) 예측, 장면 이해(Scene Understanding), 시각 기반 내비게이션(Visual Navigation)에 활용된다. 특히 비전 트랜스포머(Vision Transformer) 기반 모델은 이미지 패치를 토큰(Token)으로 표현하여 언어(Language), 상태(State), 행동(Action) 토큰과 자연스럽게 통합할 수 있기 때문에 매우 큰 영향을 미쳤다.

그러나 비전 파운데이션 모델만으로는 피지컬 AI 모델이라고 할 수 없다. 물체를 인식할 수는 있지만, 실제로 그 물체에 도달 가능한지, 집을 수 있는지, 안정적인지, 이동시켜도 안전한지, 현재 임무와 관련이 있는지는 판단하지 못할 수 있다. 물리적 접지(Physical Grounding)를 위해서는 기하학(Geometry), 시간적 문맥(Temporal Context), 체화 상태(Embodiment State), 행동 결과(Action Consequence), 센서 피드백(Sensor Feedback)이 필요하다. 따라서 비전 백본은 깊이 정보(Depth), 포인트 클라우드(Point Cloud), 고유감각(Proprioception), 힘 센서(Force Sensing), 지도(Map), 로봇 특유의 제약조건과 함께 통합되어야 한다.

비전-언어 모델은 현재 생태계의 또 다른 핵심 계층을 형성한다. 이러한 시스템은 시각 정보와 자연어 명령, 질문, 의미 개념, 상징적 지식을 연결한다. 자연어로 설명된 객체를 찾고, 장면을 설명하며, 공간적인 질문에 답하고, 상위 수준 작업 계획을 지원할 수 있다. 이러한 광범위한 지식은 대부분 로봇 데이터가 아니라 웹 규모의 이미지-텍스트 데이터셋으로부터 학습된다.

하지만 비전-언어 모델의 한계는 이해와 설명이 곧바로 실행 가능한 행동으로 이어지지 않는다는 점이다. 작업 수행 방법을 정확하게 설명하면서도 실제 행동은 기하학, 시간, 동역학, 안전 제약을 위반할 수 있다. 이러한 문제를 해결하기 위해 비전-언어-행동(Vision-Language-Action, VLA) 모델이 등장하였다. 이는 의미 이해(Semantic Understanding)에서 체화 제어(Embodied Control)로 이어지는 매우 중요한 전환이다.

비전-언어-행동 모델은 관측 정보와 언어 명령을 함께 처리하여 로봇 행동(Action Token), 연속 행동 값(Continuous Action Value), 궤적(Trajectory), 행동 청크(Action Chunk)를 생성한다. 대표적인 사례가 RT-2이다. RT-2는 웹 규모 비전-언어 지식을 로봇 제어 모델과 결합하고 행동을 토큰 형태로 표현하였다. 이를 통해 비로봇 데이터에서 학습한 의미 지식이 로봇의 일반화 성능을 향상시킬 수 있음을 보여주었다.

RT-2의 가장 큰 의미는 일반 로보틱스를 완성했다는 것이 아니라 인터넷 규모 의미 학습과 실제 물리적 제어를 연결할 수 있다는 가능성을 증명했다는 점이다. 이미지와 언어에서 학습한 개념이 새로운 객체 관계와 자연어 명령 수행에 활용될 수 있음을 보여주었으며, VLA 모델은 이후 피지컬 AI에서 가장 중요한 아키텍처 방향 중 하나가 되었다.

Open X-Embodiment와 같은 대규모 다중 로봇 데이터셋은 교차 체화(Cross-Embodiment) 학습을 현실적으로 가능하게 만들었다. 이제 연구자들은 하나의 로봇이 아니라 여러 기관, 다양한 플랫폼, 여러 카메라, 여러 작업, 서로 다른 행동 공간(Action Space)에서 수집한 데이터를 함께 학습하고 있다. 핵심 가정은 하드웨어가 다르더라도 로봇 데이터에는 공통적으로 전이 가능한 구조가 존재한다는 것이다.

Octo는 이러한 방향을 대표하는 오픈소스 범용 로봇 정책이다. 약 80만 개의 로봇 궤적을 기반으로 사전학습된 트랜스포머 기반 확산 정책(Diffusion Policy)이며, 자연어 명령, 목표 이미지, 관측 이력, 다양한 카메라 구성, 새로운 센서와 행동 공간에 적응할 수 있도록 설계되었다. Octo는 모든 로봇을 하나의 출력 인터페이스로 직접 제어하려 하기보다 효율적인 미세조정(Fine-Tuning)에 중점을 둔다.

Octo는 범용 사전학습 정책을 다양한 로봇에 적응 가능한 초기화 모델(Initialization Model)로 활용하는 방향을 보여준다. 목표는 완전한 제로샷 제어(Zero-Shot Control)가 아니라, 광범위하게 학습된 정책을 새로운 로봇 데이터로 소량만 추가 학습하여 빠르게 적응시키는 것이다. 이는 체화 차이가 여전히 크기 때문에 직접적인 범용 제어보다 적응 기반 접근이 현실적이라는 사실을 반영한다.

OpenVLA는 또 다른 중요한 오픈소스 VLA 모델이다. 약 97만 개의 로봇 에피소드(Robot Episode)를 기반으로 사전학습된 70억 개 파라미터의 비전-언어-행동 모델이다. 시각 인코더(Visual Encoder), 투영 계층(Projection Layer), 언어 모델 백본(Language Model Backbone), 토큰화된 로봇 행동(Tokenized Robot Action)을 결합하며, 모델 가중치와 학습 도구를 함께 공개하여 새로운 로봇 환경에 대한 파라미터 효율적 적응을 지원한다.

OpenVLA는 대규모 언어 모델 구조를 로봇 제어에 어떻게 적용할 수 있는지를 보여준다. 이미지는 시각 임베딩으로 변환되고, 명령은 언어 토큰으로 표현되며, 행동은 토큰으로 생성된 뒤 연속 제어 명령으로 복원된다. 이를 통해 언어 모델과 행동 생성이 하나의 구조 안에서 직접 연결된다.

Octo와 OpenVLA 같은 오픈 VLA 생태계는 연구자와 로봇 제조업체가 모델을 분석하고, 입력 인터페이스를 수정하며, 새로운 센서를 추가하고, 행동 표현을 변경하며, 로컬에서 미세조정하고, 검증된 버전을 배포할 수 있다는 점에서 전략적으로 매우 중요하다. 또한 로봇 데이터 혼합, 비전 백본, 행동 토큰화, 확산 기반 디코딩, 파라미터 효율적 적응이 성능에 미치는 영향을 자유롭게 연구할 수 있다.

Physical Intelligence의 π0 계열은 일반 조작(Generalist Manipulation)과 연속 행동 생성(Continuous Action Generation)에 초점을 맞춘 또 다른 중요한 방향이다. 초기 π0는 사전학습된 비전-언어 백본과 다양한 로봇 데이터를 결합하고, 플로우 매칭(Flow Matching)을 이용하여 연속 행동을 생성하는 행동 전문가(Action Expert)를 포함하였다. 다양한 로봇 플랫폼과 고난도 조작 데이터를 함께 학습하였다.

이 접근은 현재 생태계에서 중요한 질문을 제기한다. 로봇 행동을 이산 토큰(Discrete Token)으로 표현할 것인가, 아니면 연속 확률 모델(Continuous Probabilistic Model)로 생성할 것인가 하는 문제이다. 행동 토큰은 자기회귀 언어 모델과 자연스럽게 통합되지만 양자화 오차가 발생할 수 있다. 반면 확산(Diffusion) 및 플로우 기반 정책은 부드러운 연속 행동 분포를 표현할 수 있지만 다른 학습 및 추론 구조가 필요하다.

π0 생태계는 FAST라는 행동 토큰화 기법도 제안하였다. π0-FAST는 기존 π0보다 훨씬 빠르게 학습하면서도 비슷한 성능을 유지하는 것으로 보고되었다. 이는 행동 표현 방식 자체가 시각 표현과 언어 표현만큼 중요한 연구 주제가 되고 있음을 보여준다.

후속 모델인 π0.5는 개방형 환경(Open-World) 일반화와 장기 가정용 조작(Long-Horizon Household Manipulation)에 더 집중하였다. 또한 연구진은 의미 추론과 운동 실행 모두에서 여전히 오류가 존재함을 인정하였다. 이는 현재의 VLA 모델이 아직 완전하지 않으며 폐루프 검증(Closed-Loop Validation)이 필수적이라는 사실을 보여준다.

NVIDIA의 Isaac GR00T 생태계는 특히 휴머노이드(Humanoid)와 교차 체화 로보틱스를 위한 또 하나의 중요한 축이다. GR00T는 단순한 모델이 아니라 데이터 수집, 시뮬레이션, 합성 데이터 생성, 학습, 후처리(Post-Training), 배포 도구를 모두 포함하는 통합 플랫폼이다. NVIDIA는 이를 휴머노이드 로봇을 위한 파운데이션 모델 플랫폼으로 설명하고 있다.

초기 GR00T N1 모델은 느린 추론과 빠른 행동 생성을 분리하는 이중 시스템(Dual-System Architecture)을 사용하였다. 학습에는 인간의 1인칭 영상(Egocentric Video), 실제 로봇 궤적, 시뮬레이션 데이터, 합성 데이터를 함께 활용하였다. 이는 인간 행동 데이터, 로봇 시연, 생성 데이터를 동시에 활용하는 최근의 흐름을 잘 보여준다.

이후 GR00T 계열은 일반화, 자연어 명령 수행, 후학습(Post-Training) 성능을 향상시키는 방향으로 발전하였다. 특히 GR00T 생태계의 가장 큰 의미는 시뮬레이션 및 합성 데이터 생성 시스템과 완전히 통합되어 있다는 점이다. 로봇 파운데이션 모델에는 사전학습된 가중치뿐 아니라 데이터 생성, 하드웨어 적응, 행동 평가, 시뮬레이션에서 현실로의 전이까지 포함하는 전체 개발 체계가 필요하다.

월드 파운데이션 모델은 현재 생태계의 또 다른 핵심 계층이다. 이러한 모델은 물리적 환경이 시간에 따라 어떻게 변화하는지를 학습한다. 단순한 라벨이나 행동을 생성하는 것이 아니라 미래 장면, 잠재 상태(Latent State), 궤적, 환경 변화를 예측한다. 로보틱스에서는 행동 후보를 평가하고, 새로운 학습 데이터를 생성하며, 이상 상황을 시뮬레이션하고, 모델 기반 계획(Model-Based Planning)을 지원하는 데 활용된다.

월드 모델은 실제 로봇 데이터를 수집하는 비용이 매우 크기 때문에 특히 중요하다. 생성 모델은 객체, 배경, 조명, 시점, 실패 상황, 센서 노이즈, 다양한 물리 조건을 대량으로 생성할 수 있다. NVIDIA Cosmos와 GR00T를 함께 사용하는 접근은 생성형 월드 모델과 로봇 정책 학습이 점점 긴밀하게 연결되고 있음을 보여준다.

그러나 월드 파운데이션 모델은 물리 엔진(Physics Engine)과 동일하지 않다. 생성 모델은 시각적으로 매우 그럴듯한 미래를 생성하더라도 실제 물리적으로는 틀릴 수 있다. 따라서 데이터 증강, 표현 학습, 시나리오 생성, 근사 계획에는 유용하지만, 안전이 중요한 의사결정에는 기하학 검증, 물리 제약, 실제 환경 검증이 반드시 필요하다.

이러한 이유로 시뮬레이션 플랫폼은 여전히 피지컬 AI 생태계의 핵심 구성요소이다. Isaac Sim, Isaac Lab, MuJoCo, Gazebo와 같은 물리 기반 시뮬레이터는 실제 배포 이전에 정책을 시험할 수 있도록 하며, 도메인 랜덤화(Domain Randomization), 센서 시뮬레이션, 강화학습(Reinforcement Learning), 합성 시연(Synthetic Demonstration), Hardware-in-the-Loop 평가를 지원한다.

시뮬레이션과 파운데이션 모델의 관계도 점점 양방향이 되고 있다. 시뮬레이터는 학습 데이터를 생성하고, 반대로 파운데이션 모델은 시뮬레이션을 위한 장면(Scene), 작업(Task), 궤적(Trajectory), 자산(Asset), 시나리오를 생성한다. 이러한 결합은 데이터 다양성을 크게 높일 수 있지만, 동시에 시뮬레이터 특유의 오류를 모델이 학습하는 위험도 존재한다.

데이터 생태계(Data Ecosystem)는 아마도 피지컬 AI에서 가장 중요한 경쟁 요소일 것이다. 일반 언어 모델은 인터넷 규모 데이터를 활용하지만, 실제 물리적 상호작용 데이터는 매우 부족하다. 로봇 궤적에는 동기화된 관측, 행동, 시간 정보, 좌표계, 보정 정보, 로봇 ID, 작업 정보, 결과 라벨이 함께 포함되어야 한다.

교차 체화 데이터셋은 여러 로봇의 데이터를 통합하려고 하지만 여전히 데이터 이질성(Heterogeneity)이 큰 문제이다. 일부 데이터셋은 관절 위치를 사용하고, 다른 데이터셋은 관절 속도, Cartesian Pose, 그리퍼 명령, 작업별 제어 형식을 사용한다. 카메라 위치, 이미지 해상도, 제어 주기, 언어 라벨, 보정 품질도 모두 다르다.

이러한 문제 때문에 표준화된 로봇 데이터 형식(Standardized Robot Data Format)과 공통 행동 표현(Common Action Representation)에 대한 연구가 활발히 진행되고 있다. 장기적으로는 다양한 로봇과 다양한 환경에서 수집한 데이터를 하나의 거대한 사전학습 데이터셋으로 통합하는 것이 목표이다. 이를 위해서는 메타데이터(Metadata), 정규화(Normalization), 데이터 변환기(Converter), 품질 필터(Quality Filter), 평가 프로토콜이 필요하다.

인간 시연 데이터(Human Demonstration Data)는 여전히 매우 중요한 자원이다. 원격 조작(Teleoperation)은 성공적인 로봇 궤적을 제공하며, 1인칭 영상은 대규모 인간 행동 데이터를 제공한다. 그러나 영상만으로는 로봇 행동, 힘, 관절 상태, 실행 가능한 궤적을 직접 알 수 없다. 모델은 인간 행동과 로봇 체화 사이의 대응 관계를 학습해야 한다.

합성 데이터는 부족한 부분을 보완할 수 있지만 반드시 검증되어야 한다. 생성된 궤적이 로봇의 물리적 한계를 초과하거나 실제로 실행 불가능한 경우가 존재할 수 있기 때문이다. 따라서 고품질 피지컬 AI 데이터셋은 실제 로봇 시연, 자율 경험, 시뮬레이션, 인간 영상, 웹 데이터, 합성 장면, 실패 및 복구 사례를 함께 포함해야 한다.

현재 생태계는 범용 모델과 특화 모델로도 구분된다. 범용 VLA 시스템은 다양한 조작 작업과 환경을 처리하도록 설계된다. 반면 특화 모델은 자율주행, 내비게이션, 산업 검사, 다지 조작(Dexterous Manipulation), 휴머노이드 보행, 드론, 의료 로봇 등에 집중한다.

특화 모델은 운용 범위가 좁기 때문에 더 높은 신뢰성을 제공할 수 있다. 산업 검사 모델은 특정 설비, 결함 종류, 촬영 시점, 센서 환경에 최적화될 수 있다. 반면 범용 가정용 로봇은 훨씬 더 다양한 환경을 처리해야 하므로 예측 가능성이 낮아질 수 있다.

미래는 하나의 범용 모델이 모든 특화 모델을 대체하는 방향이 아닐 가능성이 높다. 보다 현실적인 구조는 범용 파운데이션 모델 위에 도메인 특화 어댑터(Domain-Specific Adapter), 인식 모듈, 행동 디코더, 계획기, 제어기를 결합하는 것이다. 파운데이션 모델은 의미 이해와 일반화를 제공하고, 특화 모듈은 정확성과 운영 신뢰성을 제공한다.

모델 구조도 점점 계층화(Hierarchical Architecture)되고 있다. 느린 추론 모델은 목표를 해석하고, 작업을 분해하며, 도구를 호출하고, 하위 목표를 생성한다. 빠른 행동 모델은 이를 짧은 행동 시퀀스로 변환한다. 기존 제어기는 고주파 명령을 실행하고, 독립적인 안전 시스템은 최종 한계를 강제한다.

이러한 계층 구조는 대형 모델 추론과 실시간 제어 사이의 속도 차이를 해결한다. 언어 추론과 의미 계획은 상대적으로 느려도 되지만, 모터 안정화, 힘 제어, 비상 제동, 충돌 회피는 훨씬 빠른 결정론적 제어 루프가 필요하다.

메모리 시스템도 중요한 요소가 되고 있다. 장시간 임무에서는 모든 관측 데이터를 트랜스포머 컨텍스트 안에 유지할 수 없다. 구조화된 임무 메모리(Mission Memory), 의미 지도(Semantic Map), 객체 데이터베이스, 디지털 트윈, 에피소드 요약(Episodic Summary), 검색 시스템(Retrieval System)이 필요한 정보만 유지하도록 돕는다.

도구 사용(Tool Use) 기반 로봇 파운데이션 모델도 중요한 방향이다. 모델은 모션 플래너(Motion Planner), 파지 계획기(Grasp Planner), 데이터베이스, 검사 알고리즘, 지도 작성 서비스, 플릿 관리 시스템, 디지털 트윈 등을 호출할 수 있다. 이를 통해 모든 문제를 내부에서 해결하려 하기보다 이미 검증된 엔지니어링 도구를 조정하는 역할을 수행한다.

산업용 피지컬 AI에서는 완전한 End-to-End 제어보다 이러한 도구 오케스트레이션(Tool Orchestration)이 더 중요할 수도 있다. 모델은 임무를 해석하고, 검사 절차를 선택하며, 설비 정보를 조회하고, 충돌 없는 경로를 요청하며, 완료 여부를 확인하고, 최종 보고서를 생성할 수 있다. 반면 위치 추정, 내비게이션, 안전 검사는 기존 결정론적 시스템이 담당한다.

현재 생태계에는 오픈 모델과 독점형 모델이 동시에 존재한다. 오픈 모델은 연구 접근성, 맞춤화, 로컬 배포, 아키텍처 제어를 제공한다. 독점형 모델은 더 강력한 일반 추론 능력, 대규모 학습 자원, 상업적 지원, 관리형 인프라를 제공할 수 있다. 실제 시스템은 두 가지를 함께 사용하는 하이브리드 구조가 될 가능성이 높다.

예를 들어 독점형 클라우드 모델은 복잡한 문서 해석이나 상위 추론을 담당하고, 오픈 모델은 로컬 인식과 행동 생성을 담당할 수 있다. 또는 독점형 모델이 합성 데이터와 계획을 생성하는 교사 모델(Teacher Model)이 되고, 오픈 모델이 학생 모델(Student Model)로 학습될 수도 있다. 그러나 최종 물리적 권한은 항상 로컬 안전 시스템과 제어 시스템이 유지해야 한다.

평가는 현재 생태계에서 가장 부족하면서도 가장 중요한 요소 중 하나이다. 일반 언어 및 비전 벤치마크는 로봇의 신뢰성을 입증할 수 없다. 피지컬 AI 모델은 작업 성공률, 기하학적 정확도, 힘, 충돌, 시간, 복구, 불확실성, 인간 개입을 포함하는 폐루프 평가가 필요하다.

또한 서로 다른 연구 프로젝트는 서로 다른 로봇, 작업, 환경, 평가 방식을 사용하기 때문에 직접 비교가 어렵다. 탁상 조작에서 우수한 모델이 이동 조작(Mobile Manipulation), 산업 검사, 휴머노이드 작업에서도 우수하다는 보장은 없다. 미래의 벤치마크는 체화, 센서, 행동 인터페이스, 안전 제약, 운영 조건을 명확히 기록해야 한다.

현재 분야는 시연 영상(Demonstration Video)과 실제 배포 신뢰성(Deployment Reliability)을 구분해야 하는 단계에 있다. 인상적인 영상은 성공 사례만 보여줄 수 있으며 실패율, 개입 횟수, 지연 시간, 환경 제약은 공개하지 않을 수 있다. 성숙한 생태계는 실패 사례, 불확실성, 복구 행동, 운용 설계 영역(Operational Design Domain)도 함께 공개해야 한다.

현재 대부분의 파운데이션 모델은 여전히 안전 시스템 외부에서 동작한다. VLA 모델은 행동을 제안하거나 생성할 수 있지만 기능 안전(Functional Safety), 충돌 감지, 비상 정지, 힘 제한, 인증된 제어기를 대체하지 않는다. 따라서 피지컬 AI 파운데이션 모델은 독립적인 안전 계층 위에서 동작하는 지능형 구성요소(Intelligent Component)로 이해해야 한다.

사이버보안도 점점 중요해지고 있다. 프롬프트 인젝션(Prompt Injection), 손상된 센서 입력, 악의적인 작업 명령, 변조된 모델 업데이트, 오염된 학습 데이터, 위험한 도구 호출은 모두 로봇 행동에 영향을 줄 수 있다. 따라서 보안 평가는 물리적 안전과 직접 연결되어야 한다.

엣지 배포(Edge Deployment)도 중요한 과제이다. 현재 최첨단 모델 대부분은 온보드 컴퓨터에서 직접 실행하기에는 너무 크다. 따라서 양자화(Quantization), 프루닝(Pruning), 지식 증류(Knowledge Distillation), 효율적 어텐션(Efficient Attention), 행동 청크(Action Chunking), 캐싱(Caching), 소형 로봇 정책이 필요하다.

클라우드와 엣지를 함께 사용하는 구조는 계산을 분산할 수 있지만, 네트워크 의존성 때문에 안전이 중요한 제어에는 적합하지 않다. 핵심 내비게이션, 비상 정지, 안전 동작은 반드시 로컬에서 수행되어야 한다. 클라우드는 플릿 학습(Fleet Learning), 모델 업데이트, 복잡한 의미 추론, 오프라인 분석에 적합하다.

현재 생태계는 단순한 모델 공개(Model Release)에서 완전한 피지컬 AI 플랫폼으로 발전하고 있다. 이러한 플랫폼은 데이터 엔진(Data Engine), 원격 조작(Teleoperation), 시뮬레이션, 합성 데이터, 사전학습 정책, 후학습(Post-Training), 평가, 배포, 모니터링, 플릿 피드백까지 포함한다. 앞으로의 경쟁력은 하나의 모델보다 전체 학습 순환 구조(Learning Loop)에서 결정될 가능성이 높다.

고품질 운영 데이터를 지속적으로 수집하고, 실패를 탐지하며, 복구 사례를 생성하고, 모델을 재학습하고, 업데이트를 검증하며, 이를 플릿 전체에 안전하게 배포할 수 있는 기업이 단순히 강력한 사전학습 모델을 사용하는 기업보다 더 큰 장기적 경쟁력을 확보할 가능성이 높다.

다음 단계의 피지컬 AI 생태계는 개방형 환경 일반화(Open-World Generalization), 교차 체화 전이(Cross-Embodiment Transfer), 추론과 행동의 통합(Reasoning-Action Integration), 자율 데이터 수집(Autonomous Data Collection), 지속적 개선(Continuous Improvement)에 더욱 집중할 것이다. 웹 규모의 인간 지식과 실제 로봇 경험 및 시뮬레이션 데이터를 하나의 모델 안에서 결합하는 방향으로 발전할 것이다.

또한 계산 자원도 더욱 적응적으로 활용될 것이다. 간단한 작업은 작은 로컬 정책이 처리하고, 새로운 상황에서는 대형 추론 모델, 추가 센서, 월드 모델 시뮬레이션, 인간 지원이 호출될 것이다. 계산은 항상 동일하게 수행되는 것이 아니라 작업의 난이도와 불확실성에 따라 동적으로 배분될 것이다.

장기적인 목표는 언어와 시각 문맥을 이해하고, 자신의 몸체를 표현하며, 물리적 결과를 예측하고, 새로운 작업에 적응하며, 신뢰할 수 있는 행동을 생성하는 파운데이션 모델이다. 현재의 시스템은 이러한 비전의 일부를 보여주고 있지만 아직 완전한 범용 물리 지능(General-Purpose Physical Intelligence)을 제공하는 모델은 존재하지 않는다.

따라서 현재의 생태계는 빠르게 발전하는 초기 기술 스택(Stack)으로 이해해야 한다. VLA 모델은 의미와 행동을 연결하고, 범용 정책은 작업과 로봇 간 전이를 지원하며, 월드 모델은 데이터 생성과 미래 예측을 수행하고, 시뮬레이션 플랫폼은 대규모 경험을 제공하며, 오픈 생태계는 맞춤화를 가능하게 하고, 상업 플랫폼은 개발과 배포를 통합한다.

실제 피지컬 AI에서는 이러한 요소를 하나의 모델로 대체하기보다 서로 결합하는 것이 가장 강력한 구조가 될 가능성이 높다. 멀티모달 파운데이션 모델은 임무와 환경을 이해하고, 로봇 정책은 행동 청크를 생성하며, 월드 모델은 결과를 예측하고, 특화 계획기는 기하학을 검증하며, 저수준 제어기는 안정성을 유지하고, 독립적인 안전 시스템이 최종 권한을 유지한다.

결국 피지컬 AI 파운데이션 모델 생태계는 단순한 모델 이름들의 경쟁이 아니다. 데이터, 시뮬레이션, 인식, 언어, 추론, 월드 모델, 행동 생성, 제어, 안전, 수명주기 학습이 하나의 컴퓨팅 스택으로 통합되는 과정이다. 그 성공 여부는 모델의 크기 자체보다도, 얼마나 신뢰성 있게 광범위한 인공지능을 특정한 로봇 몸체에 접지(Grounding)시키고 안전하며 유용하고 반복 가능한 물리적 행동으로 변환할 수 있는가에 의해 결정될 것이다.
