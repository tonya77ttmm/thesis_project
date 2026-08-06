from dataclasses import dataclass
@dataclass
class FrameSample:
    path:str
    label:int
    clip_id:str