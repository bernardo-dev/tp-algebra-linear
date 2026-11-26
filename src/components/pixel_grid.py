"""Builders reutilizáveis para grades de pixels (cinza, RGB, canais) e matrizes."""

from __future__ import annotations

import numpy as np
from manim import (
    BLACK,
    WHITE,
    IntegerMatrix,
    Square,
    Text,
    VGroup,
    rgb_to_color,
)

from src.config.theme import BLUE_CH, BORDER_MUTED, GREEN_CH, RED_CH, TEXT_COLOR


def build_grayscale_grid(
    matrix: np.ndarray,
    cell_size: float = 0.52,
    show_values: bool = True,
    font_size: int = 14,
) -> VGroup:
    """Constrói uma grade 2D de pixels em escala de cinza a partir de uma matriz NumPy."""
    rows, cols = matrix.shape
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            val = int(np.clip(matrix[i, j], 0, 255))
            norm = val / 255.0
            sq = Square(
                side_length=cell_size,
                fill_color=rgb_to_color([norm, norm, norm]),
                fill_opacity=1.0,
                stroke_color="#475569",
                stroke_width=1.5,
            )
            sq.move_to([
                (j - (cols - 1) / 2.0) * cell_size,
                ((rows - 1) / 2.0 - i) * cell_size,
                0.0,
            ])
            if show_values:
                txt_color = BLACK if val > 135 else WHITE
                lbl = Text(str(val), font_size=font_size, color=txt_color, weight="BOLD")
                if lbl.width > cell_size * 0.78:
                    lbl.scale_to_fit_width(cell_size * 0.78)
                lbl.move_to(sq.get_center())
                grid.add(VGroup(sq, lbl))
            else:
                grid.add(VGroup(sq))
    return grid


def build_rgb_grid(
    r_mat: np.ndarray,
    g_mat: np.ndarray,
    b_mat: np.ndarray,
    cell_size: float = 0.55,
) -> VGroup:
    """Constrói uma grade colorida RGB combinando três matrizes de canais."""
    rows, cols = r_mat.shape
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            r = float(np.clip(r_mat[i, j], 0, 255)) / 255.0
            g = float(np.clip(g_mat[i, j], 0, 255)) / 255.0
            b = float(np.clip(b_mat[i, j], 0, 255)) / 255.0
            sq = Square(
                side_length=cell_size,
                fill_color=rgb_to_color([r, g, b]),
                fill_opacity=1.0,
                stroke_color=BORDER_MUTED,
                stroke_width=1.5,
            )
            sq.move_to([
                (j - (cols - 1) / 2.0) * cell_size,
                ((rows - 1) / 2.0 - i) * cell_size,
                0.0,
            ])
            grid.add(sq)
    return grid


def build_channel_grid(
    matrix: np.ndarray,
    channel: str,
    cell_size: float = 0.48,
    show_values: bool = True,
    font_size: int = 13,
) -> VGroup:
    """Constrói uma fatia de canal único ('R', 'G' ou 'B') com borda temática."""
    ch = channel.upper()
    border_map = {"R": RED_CH, "G": GREEN_CH, "B": BLUE_CH}
    stroke = border_map.get(ch, BORDER_MUTED)

    rows, cols = matrix.shape
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            val = int(np.clip(matrix[i, j], 0, 255))
            norm = val / 255.0
            rgb = [
                norm if ch == "R" else 0.0,
                norm if ch == "G" else 0.0,
                norm if ch == "B" else 0.0,
            ]
            sq = Square(
                side_length=cell_size,
                fill_color=rgb_to_color(rgb),
                fill_opacity=1.0,
                stroke_color=stroke,
                stroke_width=1.4,
            )
            sq.move_to([
                (j - (cols - 1) / 2.0) * cell_size,
                ((rows - 1) / 2.0 - i) * cell_size,
                0.0,
            ])
            if show_values:
                txt_color = BLACK if (ch == "G" and val > 155) else WHITE
                lbl = Text(str(val), font_size=font_size, color=txt_color, weight="BOLD")
                if lbl.width > cell_size * 0.78:
                    lbl.scale_to_fit_width(cell_size * 0.78)
                lbl.move_to(sq.get_center())
                grid.add(VGroup(sq, lbl))
            else:
                grid.add(VGroup(sq))
    return grid


def build_integer_matrix(matrix: np.ndarray, scale: float = 0.55) -> IntegerMatrix:
    """Envolve IntegerMatrix do Manim com escala e cor consistentes."""
    mat = IntegerMatrix(
        matrix.astype(int).tolist(),
        h_buff=0.85,
        v_buff=0.65,
        element_to_mobject_config={"color": TEXT_COLOR},
    )
    mat.set_color(TEXT_COLOR)
    mat.scale(scale)
    return mat
