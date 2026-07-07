from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    source_index: int
    target_index: int
    sampling_direction: int


DIRECTIONS = (
    DirectionSpec("0_to_1", 0, 1, 1),
    DirectionSpec("1_to_0", 1, 0, -1),
)

DIRECTION_BY_NAME = {direction.name: direction for direction in DIRECTIONS}
