# Conceptualization Phase

## System Context Diagram
```mermaid
graph LR
    %% 스타일 설정
    classDef system fill:#1e3a8a,stroke:#1e40af,stroke-width:3px,color:#fff;
    classDef actor fill:#f3f4f6,stroke:#9ca3af,stroke-width:2px;
    classDef data fill:#fce7f3,stroke:#f472b6,stroke-width:2px;
    classDef tech fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px;

    %% 노드 정의
    USER(("User<br>(Operator)")):::actor
    RG["System<br>(Reasoning Guard)"]:::system
    
    subgraph AI Learning Pipeline
        DATA[("Training Datasets<br>- Fighter Jet Dataset<br>- Drone Dataset<br>- Rocket (Proxy) Dataset")]:::data
        TECH["Preprocessing & DL Tech<br>1. Zero-Padding<br>2. Data Augmentation<br>3. Normalization<br>4. PyTorch Engine"]:::tech
        DATA -->|"Load Images"| TECH
    end

    %% 연결 정의
    TECH -->|"Model Learning"| RG
    USER -->|"Input<br>(Unknown Image)"| RG
    RG -->|"Output<br>(Shoot/Hold Decision)"| USER
