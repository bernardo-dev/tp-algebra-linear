"""Tópico 1: Vetores, Matrizes e Tensores no Processamento de Imagens."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    FadeIn,
    GrowArrow,
    MathTex,
    RoundedRectangle,
    Scene,
    Square,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
    rgb_to_color,
)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from src.config.theme import (
    ACCENT_COLOR,
    BLUE_CH,
    BODY_SIZE,
    GREEN_CH,
    MUTED_COLOR,
    PRIMARY_COLOR,
    RED_CH,
    SECONDARY_COLOR,
    SMALL_SIZE,
    SUCCESS_COLOR,
    TEXT_COLOR,
)


def _slide_scalar_to_rgb_vector(scene: Scene) -> None:
    """Slide 1.1: Do pixel escalar ao vetor RGB em R^3."""
    header = create_slide_header("TÓPICO 01", "Do Escalar ao Vetor RGB")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    # Lado esquerdo: Escalar (cinza) vs Vetor (RGB)
    gray_sq = Square(
        side_length=1.1,
        fill_color=rgb_to_color([180 / 255, 180 / 255, 180 / 255]),
        fill_opacity=1.0,
        stroke_color=TEXT_COLOR,
        stroke_width=2,
    )
    gray_val = Text("180", font_size=BODY_SIZE, color="#000000", weight="BOLD").move_to(gray_sq)
    gray_title = Text("Pixel Cinza (Escalar)", font_size=BODY_SIZE - 1, color=PRIMARY_COLOR, weight="BOLD")
    gray_sub = MathTex(r"p \in [0, 255] \subset \mathbb{R}", font_size=26, color=MUTED_COLOR)
    gray_box = VGroup(gray_title, VGroup(gray_sq, gray_val), gray_sub).arrange(DOWN, buff=0.2)

    ch_specs = [("R: 230", 230 / 255, 0, 0, RED_CH), ("G: 90", 0, 90 / 255, 0, GREEN_CH), ("B: 40", 0, 0, 40 / 255, BLUE_CH)]
    ch_tiles = VGroup()
    for lbl_txt, r, g, b, border in ch_specs:
        sq = Square(side_length=0.82, fill_color=rgb_to_color([r, g, b]), fill_opacity=1.0, stroke_color=border, stroke_width=2)
        lbl = Text(lbl_txt, font_size=SMALL_SIZE - 1, color=TEXT_COLOR, weight="BOLD")
        ch_tiles.add(VGroup(sq, lbl).arrange(DOWN, buff=0.1))
    ch_tiles.arrange(RIGHT, buff=0.25)

    rgb_sq = Square(
        side_length=1.1,
        fill_color=rgb_to_color([230 / 255, 90 / 255, 40 / 255]),
        fill_opacity=1.0,
        stroke_color=ACCENT_COLOR,
        stroke_width=2.5,
    )
    rgb_lbl = Text("Pixel RGB", font_size=SMALL_SIZE, color=TEXT_COLOR, weight="BOLD")
    rgb_res = VGroup(rgb_sq, rgb_lbl).arrange(DOWN, buff=0.1)
    comb_arrow = Arrow(LEFT * 0.45, RIGHT * 0.45, color=ACCENT_COLOR, stroke_width=3)
    rgb_row = VGroup(ch_tiles, comb_arrow, rgb_res).arrange(RIGHT, buff=0.28)

    left_col = VGroup(gray_box, rgb_row).arrange(DOWN, buff=0.65).move_to([-3.25, -0.45, 0])

    # Lado direito: Representação vetorial
    card = create_info_card(
        "Espaço Vetorial de Cores",
        [
            Text("Cada pixel colorido é um vetor coluna:", font_size=BODY_SIZE - 1, color=TEXT_COLOR),
            MathTex(
                r"\mathbf{p} = \begin{bmatrix} 230 \\ 90 \\ 40 \end{bmatrix} \in \mathbb{R}^3",
                font_size=30,
                color=ACCENT_COLOR,
            ),
            Text("Combinação linear da base canônica:", font_size=BODY_SIZE - 1, color=TEXT_COLOR),
            MathTex(
                r"\mathbf{p} = 230\,\mathbf{e}_1 + 90\,\mathbf{e}_2 + 40\,\mathbf{e}_3",
                font_size=26,
                color=PRIMARY_COLOR,
            ),
            Text("• Brilho total, mistura e saturação operam em R³.", font_size=SMALL_SIZE + 1, color=MUTED_COLOR),
        ],
        width=5.8,
        border_color=PRIMARY_COLOR,
    ).move_to([3.3, -0.45, 0])

    scene.play(FadeIn(gray_box, shift=UP * 0.2), run_time=0.6)
    scene.play(FadeIn(ch_tiles, shift=UP * 0.2), run_time=0.55)
    scene.play(GrowArrow(comb_arrow), FadeIn(rgb_res, scale=0.85), run_time=0.55)
    scene.play(FadeIn(card, shift=LEFT * 0.2), run_time=0.65)
    scene.wait(1.8)

    transition_stage(scene, [header, left_col, card])


def _slide_grayscale_matrix(scene: Scene) -> None:
    """Slide 1.2: Imagem em escala de cinza como matriz 2D."""
    header = create_slide_header("TÓPICO 01", "Imagem em Escala de Cinza como Matriz")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    img = np.array([
        [20, 20, 120, 20, 20],
        [20, 120, 240, 120, 20],
        [120, 240, 255, 240, 120],
        [20, 120, 240, 120, 20],
        [20, 20, 120, 20, 20],
    ])
    grid = build_grayscale_grid(img, cell_size=0.66, show_values=True, font_size=14)
    grid_lbl = Text("Grade de Pixels (5 × 5)", font_size=BODY_SIZE - 1, color=PRIMARY_COLOR, weight="BOLD")
    grid_group = VGroup(grid_lbl, grid).arrange(DOWN, buff=0.25).move_to([-3.55, -0.35, 0])

    mat = build_integer_matrix(img, scale=0.66)
    mat_eq = MathTex(r"I \in \mathbb{R}^{5 \times 5}", font_size=28, color=ACCENT_COLOR)
    mat_group = VGroup(mat, mat_eq).arrange(DOWN, buff=0.3).move_to([3.1, -0.35, 0])

    # Destacar célula (2, 3) => índice linear 2*5 + 3 = 13
    pixel_cell = grid[13]
    mat_entry = mat.get_rows()[2][3]
    hl_pixel = SurroundingRectangle(pixel_cell, color=ACCENT_COLOR, buff=0.04, stroke_width=3.5)
    hl_entry = SurroundingRectangle(mat_entry, color=ACCENT_COLOR, buff=0.08, stroke_width=3.5)
    link_arrow = Arrow(
        hl_pixel.get_right() + RIGHT * 0.08,
        hl_entry.get_left() + LEFT * 0.08,
        color=ACCENT_COLOR,
        stroke_width=3,
    )
    index_note = MathTex(r"I_{2,3} = 240", font_size=28, color=SUCCESS_COLOR).next_to(link_arrow, UP, buff=0.15)

    scene.play(FadeIn(grid_group, scale=0.95), run_time=0.65)
    scene.play(Write(mat_group), run_time=0.8)
    scene.play(Create(hl_pixel), Create(hl_entry), GrowArrow(link_arrow), FadeIn(index_note), run_time=0.7)
    scene.wait(1.9)

    transition_stage(scene, [header, grid_group, mat_group, hl_pixel, hl_entry, link_arrow, index_note])


def _slide_rgb_tensor(scene: Scene) -> None:
    """Slide 1.3: Imagem colorida como tensor de 3ª ordem."""
    header = create_slide_header("TÓPICO 01", "Imagem Colorida como Tensor (H × W × 3)")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    r_mat = np.array([
        [240, 230, 40, 30],
        [220, 200, 60, 40],
        [50, 70, 235, 210],
        [30, 50, 220, 190],
    ])
    g_mat = np.array([
        [70, 110, 210, 180],
        [90, 140, 230, 200],
        [190, 210, 80, 60],
        [150, 175, 65, 50],
    ])
    b_mat = np.array([
        [50, 60, 235, 220],
        [40, 80, 245, 210],
        [100, 120, 230, 200],
        [80, 95, 210, 185],
    ])

    rgb_grid = build_rgb_grid(r_mat, g_mat, b_mat, cell_size=0.62)
    rgb_cap = Text("Imagem Colorida", font_size=BODY_SIZE - 1, color=TEXT_COLOR, weight="BOLD")
    rgb_box = VGroup(rgb_cap, rgb_grid).arrange(DOWN, buff=0.22).move_to([-4.35, -0.35, 0])

    r_slice = build_channel_grid(r_mat, "R", cell_size=0.46, show_values=True, font_size=11)
    g_slice = build_channel_grid(g_mat, "G", cell_size=0.46, show_values=True, font_size=11)
    b_slice = build_channel_grid(b_mat, "B", cell_size=0.46, show_values=True, font_size=11)

    r_lbl = Text("Canal R", font_size=SMALL_SIZE, color=RED_CH, weight="BOLD")
    g_lbl = Text("Canal G", font_size=SMALL_SIZE, color=GREEN_CH, weight="BOLD")
    b_lbl = Text("Canal B", font_size=SMALL_SIZE, color=BLUE_CH, weight="BOLD")

    slices = VGroup(
        VGroup(r_lbl, r_slice).arrange(DOWN, buff=0.14),
        VGroup(g_lbl, g_slice).arrange(DOWN, buff=0.14),
        VGroup(b_lbl, b_slice).arrange(DOWN, buff=0.14),
    ).arrange(RIGHT, buff=0.42).move_to([2.1, 0.35, 0])

    decomp_arrow = Arrow(
        rgb_box.get_right() + RIGHT * 0.15,
        slices.get_left() + LEFT * 0.15,
        color=ACCENT_COLOR,
        stroke_width=3,
    )

    tensor_note = MathTex(
        r"\mathcal{T} \in \mathbb{R}^{H \times W \times 3} \quad\Longleftrightarrow\quad 3 \text{ matrizes } (I_R,\, I_G,\, I_B)",
        font_size=28,
        color=TEXT_COLOR,
    )
    tensor_card = RoundedRectangle(
        corner_radius=0.16,
        width=tensor_note.width + 0.8,
        height=tensor_note.height + 0.4,
        fill_color="#1E293B",
        fill_opacity=0.9,
        stroke_color=SECONDARY_COLOR,
        stroke_width=2,
    )
    tensor_banner = VGroup(tensor_card, tensor_note).move_to([2.1, -2.15, 0])

    scene.play(FadeIn(rgb_box, scale=0.95), run_time=0.6)
    scene.play(GrowArrow(decomp_arrow), FadeIn(slices, lag_ratio=0.18), run_time=0.85)
    scene.play(FadeIn(tensor_banner, shift=UP * 0.15), run_time=0.55)
    scene.wait(2.0)

    transition_stage(scene, [header, rgb_box, decomp_arrow, slices, tensor_banner])


def build_vectors_matrices_tensors(scene: Scene) -> None:
    """Executa todos os slides do Tópico 1 na cena informada."""
    apply_theme(scene)
    play_section_intro(
        scene,
        "TÓPICO 01",
        "Vetores, Matrizes e Tensores",
        "Como imagens digitais são estruturadas na Álgebra Linear",
        accent_color=PRIMARY_COLOR,
    )
    _slide_scalar_to_rgb_vector(scene)
    _slide_grayscale_matrix(scene)
    _slide_rgb_tensor(scene)


class VectorsMatricesTensorsScene(Scene):
    """Cena independente para o Tópico 1."""

    def construct(self) -> None:
        build_vectors_matrices_tensors(self)


if __name__ == "__main__":
    pass
