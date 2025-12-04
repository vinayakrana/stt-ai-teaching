#import "slides.typ": *

#show: course-theme(
  title: "Data Collection and Labeling",
  subtitle: "CS 203: Software Tools and Techniques for AI",
  author: "Prof. Nipun Batra",
)

#title-slide("Data Collection and Labeling", subtitle: "CS 203: Software Tools and Techniques for AI")

#section-slide[Module Overview]

= Four Core Components

#card[
  1. *Data Collection* — Tools and techniques for gathering data
  2. *Data Validation* — Ensuring data quality and reliability
  3. *Data Labeling* — Annotating datasets with ground truth
  4. *Data Augmentation* — Expanding datasets strategically
]

#v(1em)

#tip-box[
  Quality data is the foundation of successful AI systems. \
  *Garbage in, garbage out!*
]

#section-slide[Part 1: Data Collection]

= Why Data Collection Matters

- Real-world AI systems depend on *continuous data flow*
- User behavior changes over time → *models need fresh data*
- Production systems require *automated collection pipelines*
- Debugging often requires understanding *what data was seen*

= Common Data Sources

#columns-layout(
  [
    *Digital Sources*
    - Web applications
    - Mobile apps
    - IoT devices
    - APIs and databases
  ],
  [
    *Physical Sources*
    - Sensors
    - Cameras
    - Microphones
    - Manual entry
  ]
)

= Instrumentation

*Instrumentation* = Adding code to collect data about system behavior

#card[
  - *Minimal Performance Impact* — Don't slow down production
  - *Comprehensive Coverage* — Capture all relevant events
  - *Privacy-Aware* — Respect GDPR, CCPA
  - *Structured Logging* — Consistent formats for parsing
]

= Basic Instrumentation

```python
import logging
import json
from datetime import datetime

def log_user_action(user_id, action, metadata):
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "action": action,
        "metadata": metadata
    }
    logging.info(json.dumps(event))
```

= Web Analytics Tools

#columns-layout(
  [
    *Google Analytics 4*
    ```javascript
    gtag('event', 'search', {
      'search_term': query,
      'results_count': results.length
    });
    ```
    ✅ Free, Rich ecosystem \
    ⚠️ Privacy concerns
  ],
  [
    *Mixpanel*
    ```javascript
    mixpanel.track('Video Played', {
      'video_id': video.id,
      'duration': video.length
    });
    ```
    ✅ Detailed analytics \
    ⚠️ Costly at scale
  ]
)

#section-slide[Part 2: Data Validation]

= The Cost of Bad Data

#columns-layout(
  [
    *Problems*
    - Garbage In, Garbage Out
    - Silent Failures
    - Expensive Debugging
    - Lost Trust
  ],
  [
    *Real Impact*
    - E-commerce: 30% CTR drop
    - Medical: Misdiagnoses
    - Fraud: 45% false positives
  ]
)

= Data Quality Issues

#card[
  #columns-layout(
    [
      1. *Completeness* — Missing values
      2. *Accuracy* — Incorrect data
      3. *Consistency* — Contradictions
    ],
    [
      4. *Timeliness* — Stale data
      5. *Uniqueness* — Duplicates
      6. *Validity* — Rule violations
    ]
  )
]

= Validation with Pydantic

```python
from pydantic import BaseModel, Field, validator

class UserEvent(BaseModel):
    user_id: str = Field(..., min_length=1)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

    @validator('event_type')
    def validate_event_type(cls, v):
        allowed = ['click', 'view', 'purchase']
        if v not in allowed:
            raise ValueError('Invalid event type')
        return v
```

#section-slide[Part 3: Data Labeling]

= What is Data Labeling?

*Data Labeling* = Adding meaningful tags to raw data

#warning-box[
  - Supervised Learning *requires* labeled examples
  - Cost: Often *60-80%* of ML project time and budget
  - Quality labels → Better models
]

= Types of Labeling Tasks

#columns-layout(
  [
    *Classification*
    - Image: "cat" vs "dog"
    - Text: "spam" vs "not spam"
    
    *Object Detection*
    - Bounding boxes
    - Keypoint annotation
  ],
  [
    *Sequence Labeling*
    - Named Entity Recognition
    - Part-of-speech tagging
    
    *Structured Prediction*
    - Semantic segmentation
  ]
)

= Label Studio

#card[
  *Label Studio* — Open-source, multi-modal annotation platform
  
  - Multi-modal: Images, text, audio, video
  - Custom interfaces with XML config
  - ML-assisted labeling
  - Export: JSON, CSV, COCO, YOLO
]

= Inter-Annotator Agreement

*Problem:* Different annotators may label differently

#info-box[
  *Cohen's Kappa* measures agreement accounting for chance:
  $ kappa = (p_o - p_e) / (1 - p_e) $
]

- 0.0-0.20: Slight
- 0.61-0.80: *Substantial*
- 0.81-1.00: *Almost perfect*
