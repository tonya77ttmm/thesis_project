from dataclasses import dataclass
from typing import Optional
@dataclass
class Metrics:
    f1:float
    acc:float
    prec:float
    rec:float
    kappa:float
    auc:float
    epoch: Optional[int] = None