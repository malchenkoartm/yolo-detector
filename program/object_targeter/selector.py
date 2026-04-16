import torch
from ultralytics.engine.results import Results, Boxes
from interfaces import IObjectSelector


class ObjectSelector(IObjectSelector):
        
    @staticmethod
    def get_target(boxes: Boxes, target: int) -> torch.Tensor | None:
        if boxes.id is None:
            return None
        matched = boxes.xyxy[boxes.id == target]
        return matched[0] if len(matched) else None

    @staticmethod
    def select_best(results: list[Results]) -> tuple[int, int] | None:
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[idx].to(torch.int).tolist()
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def select_first(results: list[Results]) -> tuple[int, int] | None:
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        x1, y1, x2, y2 = results[0].boxes.xyxy[0].to(torch.int).tolist()
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def select(results: list[Results], mode: str = 'first',
               target: int | None = None) -> tuple[int, int] | None:
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        if target is not None:
            box = ObjectSelector.get_target(results[0].boxes, target)
            if box is not None:
                x1, y1, x2, y2 = box.to(torch.int).tolist()
                return (x1 + x2) // 2, (y1 + y2) // 2

        return ObjectSelector.select_best(results) if mode == 'best' \
            else ObjectSelector.select_first(results)