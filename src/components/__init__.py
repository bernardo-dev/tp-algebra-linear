"""Componentes visuais reutilizáveis para slides e grades de pixels."""

from src.components.pixel_grid import (
    build_channel_grid,
    build_grayscale_grid,
    build_integer_matrix,
    build_rgb_grid,
)
from src.components.slide_layout import (
    apply_theme,
    create_info_card,
    create_slide_header,
    play_section_intro,
    transition_stage,
)

__all__ = [
    "apply_theme",
    "build_channel_grid",
    "build_grayscale_grid",
    "build_integer_matrix",
    "build_rgb_grid",
    "create_info_card",
    "create_slide_header",
    "play_section_intro",
    "transition_stage",
]
