# Portfolio Project Data — Adaptive Traffic Signal Control Using Deep Q-Learning

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/adaptive_traffic_signal_control/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image09.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "adaptive-traffic-signal-control-using-deep-q-learning",
  "code": "PRO-35 // ML-20",
  "title": "Adaptive Traffic Signal Control Using Deep Q-Learning",
  "category": "machine-learning",
  "categoryLabel": "Reinforcement Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Adaptive-Traffic-Signal-Control-Using-Deep-Q-Learning",
  "license": "MIT License",
  "banner": "/images/projects/adaptive_traffic_signal_control/banner.webp",
  "summary": "Deep Q-Network adaptive traffic signal controller cutting dynamic rush-hour vehicle waiting times by 83.7% versus fixed-time plans.",
  "projectDescription": "Physics-informed 4-way intersection Deep Q-Network controller optimizing phase timing under stochastic Poisson traffic arrivals. Enforces safety clearance, minimum and maximum green bounds, and outperforms Fixed-Time and Actuated baselines across 60 matched-seed benchmark runs.",
  "architectureSummary": "Poisson Traffic Simulator -> 12-D Normalized State -> LayerNorm DQN Agent -> Safety-Constrained Phase Control & Streamlit Studio",
  "techStack": [
    "Python",
    "PyTorch",
    "NumPy",
    "Pandas",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "PyYAML",
    "Pytest",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Reinforcement Learning Control System",
    "targetPlatform": "Python Simulation & Streamlit Studio",
    "primaryDomain": "Urban Traffic Signal Optimization",
    "architecturePipeline": "Poisson Sim -> 12D State -> MLP DQN",
    "testingAndAudit": "Verified (13/13 Pytest Passing)",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Adaptive Traffic Control RL System",
    "learningParadigm": "Reinforcement Learning",
    "modelDomain": "Urban Traffic Signal Control",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Deep Q-Learning",
      "Poisson Traffic Simulation",
      "Multi-Scenario Benchmarking"
    ]
  },
  "mlProblemContext": "Static pre-timed traffic signals allocate fixed green cycles that fail during asymmetric or rapidly shifting urban rush hours, causing wasted green intervals and severe queue spillback. Formulating 4-way signal control as a safety-constrained MDP enables real-time queue balancing without conflicting movements.",
  "mlDatasetOverview": {
    "profileBadge": "Stochastic Traffic Simulation",
    "source": "In-house Poisson TrafficGenerator & TrafficIntersectionEnv (config/default.yaml)",
    "modality": "Sequential State-Action Simulation & Time-Series Telemetry",
    "scale": "100 training episodes (72,000 decision steps) + 60 matched-seed 1-hr evaluation runs",
    "inputShape": "12-D Continuous Normalized State Vector per 5s Step",
    "dataFormat": "In-Memory NumPy Float32 Tensors, JSON Traces & CSV Metrics",
    "description": "Vehicle arrivals follow independent per-approach Poisson processes at 1-second simulation ticks across Low, Medium, High, and Dynamic Changing rush-hour demand regimes, with 5 independent random seeds [100, 200, 300, 400, 500] reserved strictly for testing.",
    "provenanceNote": "Fully synthetic physics-informed traffic generation grounded in Highway Capacity Manual saturation flow headways (0.55 veh/s per approach); zero external PII or proprietary sensor data."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "12-D Normalized MDP State Space",
    "spaceNote": "All state observations are bounded within [0.0, 1.0] to stabilize Q-network forward propagation across demand regimes.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Optimal Discrete Phase Action & Expected Q-Values",
      "type": "Binary Discrete Action (0: Keep Phase, 1: Switch Phase)",
      "distribution": "Greedy argmax policy evaluated across 10s-60s green bounds with mandatory 4s yellow clearance"
    },
    "inputSignals": [
      {
        "name": "Normalized Approach Queues (N, S, E, W)",
        "type": "Continuous Numerical [0, 1]",
        "importanceOrRole": "Primary Congestion Indicator",
        "description": "Queued vehicle counts per approach divided by max_queue_capacity (150 vehicles), directly driving green extension and phase switching."
      },
      {
        "name": "Normalized Mean Waiting Times (N, S, E, W)",
        "type": "Continuous Numerical [0, 1]",
        "importanceOrRole": "Delay Equity Signal",
        "description": "Average accumulated waiting time of queued vehicles per approach normalized against max_wait_norm (180s) to prevent corridor starvation."
      },
      {
        "name": "Active Green Phase & Elapsed Phase Ratio",
        "type": "Binary & Continuous [0, 1]",
        "importanceOrRole": "Temporal Signal Context",
        "description": "Indicates active corridor (0: North-South, 1: East-West) and elapsed duration scaled by max_green_time (60s)."
      },
      {
        "name": "Switch Eligibility Flag",
        "type": "Binary Indicator {0, 1}",
        "importanceOrRole": "Safety Constraint Feature",
        "description": "Signals 1.0 only when active green phase duration meets or exceeds min_green_time (10s) outside clearance intervals."
      },
      {
        "name": "Total Network Occupancy",
        "type": "Continuous Numerical [0, 1]",
        "importanceOrRole": "Global Saturation Feature",
        "description": "Sum of all queued vehicles normalized by 2 x max_queue_capacity (300 vehicles), tracking overall intersection load."
      }
    ],
    "dataCharacteristicsSummary": "Normalized queue and delay signals remain compact near zero under Low demand, oscillate moderately under Medium demand, escalate continuously when High demand exceeds the 3,960 veh/hr capacity, and exhibit sharp directional regime shifts at 15-minute marks in Changing demand."
  },
  "mlLearningApproach": {
    "paradigm": "Off-Policy Value-Based Deep Reinforcement Learning (MDP)",
    "methodologyDescription": "The agent approximates state-action Q-values using an online MLP paired with a synchronized target network and cyclic experience replay. A composite reward penalizes total queues, waiting times, corridor pressure imbalance, switching loss, and emergency spillback.",
    "justification": "Unlike rigid Fixed-Time cycles or myopic queue-ratio Actuated rules, DQN optimizes multi-step discounted future delay (gamma = 0.99), allowing it to anticipate directional surges while avoiding unnecessary 4-second yellow clearance penalties."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Poisson Vehicle Arrival Generation",
      "description": "Discrete vehicle entities are sampled per second from approach-specific Poisson rates covering steady and time-varying rush-hour scenarios.",
      "badge": "Simulation Input"
    },
    {
      "step": "02",
      "title": "FIFO Queue & Saturation Discharge Tracking",
      "description": "Vehicles enter per-approach FIFO deques, accumulate 1-second waiting increments, and discharge during valid green phases at 0.55 veh/s.",
      "badge": "Physics Engine"
    },
    {
      "step": "03",
      "title": "Bounded State Normalization",
      "description": "Raw queue lengths, average wait times, phase timers, and switch eligibility flags are scaled into a unified 12-D float32 vector bounded in [0, 1].",
      "badge": "Feature Scaling"
    },
    {
      "step": "04",
      "title": "Experience Replay Buffering",
      "description": "Sequential transitions (s, a, r, s', done) are stored in a 50,000-capacity cyclic NumPy buffer and sampled uniformly in batches of 64 to break temporal correlation.",
      "badge": "Decorrelation"
    }
  ],
  "mlDataPreparationNote": "Training uses seed 42 across alternating Medium, High, and Low demand episodes, while validation and final multi-seed evaluation strictly isolate unseen seeds [100, 200, 300, 400, 500] with epsilon = 0.0.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Deep Q-Network (DQN) with LayerNorm MLP",
    "architectureSummary": "QNetwork maps 12 state features through two 128-unit hidden layers with LayerNorm and ReLU activations to 2 action Q-values, paired with a hard-synced target network.",
    "configurationBadge": "100 Episodes | Seed 42",
    "hyperparameters": [
      {
        "name": "hidden_dims",
        "value": "[128, 128]",
        "description": "Two hidden linear layers regularized with LayerNorm and ReLU activations."
      },
      {
        "name": "learning_rate",
        "value": "0.0005",
        "description": "Adam optimizer step size for stable Bellman target updates."
      },
      {
        "name": "gamma",
        "value": "0.99",
        "description": "Discount factor prioritizing long-term intersection delay minimization."
      },
      {
        "name": "epsilon_schedule",
        "value": "1.0 -> 0.05 (10k steps)",
        "description": "Linear epsilon-greedy exploration decay over the first 10,000 steps."
      },
      {
        "name": "batch_size / buffer",
        "value": "64 / 50,000",
        "description": "Uniform mini-batch size sampled from cyclic FIFO experience replay."
      },
      {
        "name": "target_update_interval",
        "value": "400 steps (hard sync)",
        "description": "Gradient steps between full online-to-target weight copies."
      }
    ],
    "objectiveCriterion": "Smooth L1 Loss (Huber Loss) with Gradient Norm Clipping (max_norm = 5.0)",
    "optimizationSolver": "Adam Optimizer",
    "trainingProcess": "Trained for 100 one-hour episodes (72,000 steps, 191.1s runtime) rotating Medium, High, and Low demand. Validation every 10 episodes across unseen seeds selected Episode 40 (13.99s wait) as best_dqn_model.pt, avoiding overfitting to later training noise."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Matched-Seed Multi-Scenario Benchmarking (5 Seeds x 4 Regimes)",
    "description": "Deterministic greedy DQN policy (epsilon=0) was benchmarked against Fixed-Time and Actuated baselines over 60 one-hour runs using identical Poisson seeds [100, 200, 300, 400, 500], measuring waiting time, queue length, throughput, total delay, and peak spillback.",
    "metrics": [
      {
        "label": "CHANGING WAIT TIME",
        "value": "27.06s",
        "benchmark": "Baseline: 166.39s"
      },
      {
        "label": "CHANGING DELAY GAIN",
        "value": "-83.7%",
        "benchmark": "Baseline: 434.4k"
      },
      {
        "label": "MEDIUM WAIT TIME",
        "value": "14.51s",
        "benchmark": "Baseline: 25.40s"
      },
      {
        "label": "LOW WAIT TIME",
        "value": "9.21s",
        "benchmark": "Baseline: 13.54s"
      },
      {
        "label": "HIGH WAIT TIME",
        "value": "419.92s",
        "benchmark": "Baseline: 490.79s"
      },
      {
        "label": "VAL BEST WAIT (EP 40)",
        "value": "13.99s",
        "benchmark": "Target < 15.0s"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Interactive Live Traffic Intersection Simulation Dashboard",
      "badge": "Streamlit Live Sim",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image01.webp",
      "telemetryNote": "Medium Demand (1200s, Seed 42): 13.2s avg wait, 9.6 veh avg queue, 878 veh throughput, 59 phase switches."
    },
    {
      "title": "Multi-Seed Empirical Benchmark & Relative Improvement Tables",
      "badge": "Tabular Benchmark",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image02.webp",
      "telemetryNote": "N=5 seed comparison across all 4 demand scenarios confirming -83.74% wait time and -83.70% queue length in Changing demand."
    },
    {
      "title": "DQN Training Convergence & Loss Curves Dashboard",
      "badge": "Training Studio View",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image03.webp",
      "telemetryNote": "4-panel convergence view tracking smoothed episode reward, Huber loss, epsilon decay, and operational delay."
    },
    {
      "title": "MDP Formulation, Safety Constraints & Phase Timeline",
      "badge": "System Architecture",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image04.webp",
      "telemetryNote": "12-D state formulation, composite reward equation, 4s clearance constraints, and first-5-min phase Gantt chart."
    },
    {
      "title": "Dynamic Changing Demand Queue Length Time-Series",
      "badge": "Time-Series Trace",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image05.webp",
      "telemetryNote": "Fixed-Time queues spike past 250 vehicles during rush surges while DQN dynamically keeps queues below 45 vehicles."
    },
    {
      "title": "4-Panel DQN Training & Validation Diagnostics",
      "badge": "Training Convergence",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image06.webp",
      "telemetryNote": "Reward stabilizes near -650 after Episode 15; validation checkpoints select Episode 40 at 13.99s average wait."
    },
    {
      "title": "Scenario Improvement Summary & Signal Phase Gantt Chart",
      "badge": "Gain & Phase Analysis",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image07.webp",
      "telemetryNote": "Variable green phase durations adapt to live queue pressure, cutting delay by 32.0% (Low), 42.9% (Medium), and 83.7% (Changing)."
    },
    {
      "title": "Throughput, Queue Length & Waiting Time Comparison",
      "badge": "Multi-Seed Bar Plots",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image08.webp",
      "telemetryNote": "Matched-seed bar charts with std error bars showing near-100% demand clearance in non-saturated regimes."
    },
    {
      "title": "Total Accumulated Delay & Peak Queue Spillback Risk",
      "badge": "Delay & Spillback",
      "imageUrl": "/images/projects/adaptive_traffic_signal_control/image09.webp",
      "telemetryNote": "DQN cuts Changing-demand total delay from 434.4k to 71.0k veh-s and peak queue spillback from 272.2 to 58.0 vehicles."
    }
  ],
  "mlVisualizationTelemetryNote": "Validation waiting time (13.99s at Episode 40) aligns closely with held-out Medium Demand test performance (14.51 +/- 1.22s across 5 seeds), confirming stable generalization without overfitting.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "The DQN agent learns to hold green phases during active directional surges and trigger early switches once an approach clears past the 10s minimum green threshold, dynamically balancing corridor queue pressure while avoiding wasteful 4s yellow clearance cycles.",
    "keyDrivers": [
      {
        "factor": "Directional Queue Pressure Differential",
        "impact": "Weighted at 0.03 in reward; drives dynamic green extensions toward heavy NS or EW surges."
      },
      {
        "factor": "Phase Switch Penalty & 4s Clearance Loss",
        "impact": "Switch penalty of 1.5 suppresses rapid signal oscillation and preserves intersection discharge capacity."
      },
      {
        "factor": "10s Minimum & 60s Maximum Green Bounds",
        "impact": "Prevents starvation on low-volume cross-streets while guaranteeing safe clearance intervals."
      }
    ],
    "limitations": [
      {
        "title": "Oversaturated Throughput Trade-Off",
        "description": "Under extreme High demand (~4,896 veh/hr > 3,960 veh/hr capacity), higher switching (146.2 vs 102.0) cuts wait by 14.4% but lowers throughput by 5.0%."
      },
      {
        "title": "Synthetic Poisson & Isolated Intersection Scope",
        "description": "Assumes independent Poisson arrivals and exact queue sensing at a single 4-way intersection without upstream platoon coordination or left-turn yield phases."
      }
    ],
    "futureImprovements": [
      {
        "title": "Multi-Agent Corridor Coordination (MARL)",
        "target": "Extend single-intersection DQN to arterial networks with green-wave offset coordination."
      },
      {
        "title": "Saturation-Aware Switch Gating & Dueling Double DQN",
        "target": "Reduce clearance losses under oversaturated regimes and lower Q-value overestimation variance."
      }
    ],
    "discoveryNote": "All 13 Pytest unit and safety checks pass, verifying non-conflicting green phases, minimum green enforcement, queue non-negativity, and monotonic reward behavior."
  },
  "mlModelArtifacts": [
    {
      "name": "best_dqn_model.pt",
      "format": "PT"
    },
    {
      "name": "training_metadata.json",
      "format": "JSON"
    },
    {
      "name": "evaluation_summary.csv",
      "format": "CSV"
    },
    {
      "name": "evaluation_detailed.json",
      "format": "JSON"
    },
    {
      "name": "simulation_traces.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Reinforcement Learning",
    "modelArchitecture": "Deep Q-Network MLP",
    "coreFramework": "PyTorch",
    "validationMethod": "Multi-Seed Scenario Benchmark"
  }
}
```
