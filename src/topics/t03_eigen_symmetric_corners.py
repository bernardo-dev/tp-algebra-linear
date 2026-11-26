"""Tópico 3: Autovalores, Autovetores e Matrizes Simétricas (Detector de Harris)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Circle,
    Create,
    Dot,
    Ellipse,
    FadeIn,
    FadeOut,
    MathTex,
    Polygon,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    Write,
)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.components.pixel_grid import build_grayscale_grid
from src.components.slide_layout import (
    apply_theme,
    create_info_card,
    create_slide_header,
    play_section_intro,
    transition_stage,
)
from src.config.theme import (
    ACCENT_COLOR,
    BODY_SIZE,
    CARD_BG,
    DANGER_COLOR,
    MUTED_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SMALL_SIZE,
    SUCCESS_COLOR,
    TEXT_COLOR,
)


def _slide_structure_tensor(scene: Scene) -> None:
    """Slide 3.1: Tensor de Estrutura M e Teorema Espectral para matrizes simétricas."""
    header = create_slide_header("TÓPICO 03", "Tensor de Estrutura e Matrizes Simétricas")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    formula_card = create_info_card(
        "Matriz de Covariância dos Gradientes",
        [
            Text("Em uma janela local W, calculamos derivadas Ix e Iy:", font_size=SMALL_SIZE + 1, color=TEXT_COLOR),
            MathTex(
                r"M = \sum_{(x,y) \in W} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}",
                font_size=28,
                color=ACCENT_COLOR,
            ),
            Text("• M é real e simétrica: M = M^T.", font_size=SMALL_SIZE + 1, color=PRIMARY_COLOR),
            Text("• M é semidefinida positiva: v^T M v ≥ 0.", font_size=SMALL_SIZE + 1, color=MUTED_COLOR),
        ],
        width=5.8,
        border_color=PRIMARY_COLOR,
    ).move_to([-3.25, -0.45, 0])

    spectral_card = create_info_card(
        "Teorema Espectral (M = Q Λ Q^T)",
        [
            MathTex(
                r"M \mathbf{v}_i = \lambda_i \mathbf{v}_i \quad (\lambda_1 \ge \lambda_2 \ge 0)",
                font_size=26,
                color=SUCCESS_COLOR,
            ),
            Text("• Autovalores reais λ₁, λ₂ medem a intensidade da variação.", font_size=SMALL_SIZE + 1, color=TEXT_COLOR),
            Text("• Autovetores v₁ ⊥ v₂ são ortogonais e dão as direções principais.", font_size=SMALL_SIZE + 1, color=SECONDARY_COLOR),
            MathTex(
                r"Q = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 \end{bmatrix},\quad \Lambda = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}",
                font_size=24,
                color=ACCENT_COLOR,
            ),
        ],
        width=5.9,
        border_color=SUCCESS_COLOR,
    ).move_to([3.25, -0.45, 0])

    scene.play(FadeIn(formula_card, shift=RIGHT * 0.2), run_time=0.65)
    scene.play(FadeIn(spectral_card, shift=LEFT * 0.2), run_time=0.65)
    scene.wait(2.2)

    transition_stage(scene, [header, formula_card, spectral_card])


def _build_region_panel(
    title: str,
    accent: str,
    m_tex: str,
    eig_tex: str,
    desc: str,
    rx: float,
    ry: float,
    angle: float,
) -> VGroup:
    """Constrói o painel analítico lateral para uma janela (Plana, Borda ou Canto)."""
    title_mob = Text(title, font_size=BODY_SIZE, color=accent, weight="BOLD")
    mat_mob = MathTex(m_tex, font_size=24, color=TEXT_COLOR)
    eig_mob = MathTex(eig_tex, font_size=24, color=accent)
    desc_mob = Text(desc, font_size=SMALL_SIZE, color=MUTED_COLOR)

    ellipse = Ellipse(
        width=max(rx * 2, 0.22),
        height=max(ry * 2, 0.22),
        color=PRIMARY_COLOR,
        fill_color=PRIMARY_COLOR,
        fill_opacity=0.22,
        stroke_width=2.5,
    )
    v1_len = max(rx, 0.3)
    v2_len = max(ry, 0.25)
    v1 = Arrow(LEFT * v1_len, RIGHT * v1_len, buff=0, color=SUCCESS_COLOR, stroke_width=3.5, max_tip_length_to_length_ratio=0.22)
    v2 = Arrow(DOWN * v2_len, UP * v2_len, buff=0, color=SECONDARY_COLOR, stroke_width=3.5, max_tip_length_to_length_ratio=0.22)
    eig_plot = VGroup(ellipse, v1, v2).rotate(angle)
    plot_frame = RoundedRectangle(
        corner_radius=0.14,
        width=2.8,
        height=1.75,
        fill_color="#0F172A",
        fill_opacity=0.95,
        stroke_color="#334155",
        stroke_width=1.5,
    )
    eig_plot.move_to(plot_frame.get_center())
    visual = VGroup(plot_frame, eig_plot)

    content = VGroup(title_mob, mat_mob, visual, eig_mob, desc_mob).arrange(DOWN, buff=0.22)
    card_bg = RoundedRectangle(
        corner_radius=0.2,
        width=6.0,
        height=content.height + 0.55,
        fill_color=CARD_BG,
        fill_opacity=0.9,
        stroke_color=accent,
        stroke_width=2,
    )
    content.move_to(card_bg.get_center())
    return VGroup(card_bg, content)


def _slide_three_regions(scene: Scene) -> None:
    """Slide 3.2: Simulação da janela deslizante em região plana, borda e canto."""
    header = create_slide_header("TÓPICO 03", "Identificando Regiões pelos Autovalores")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    img = np.full((7, 7), 30, dtype=int)
    img[3:, 3:] = 230
    cell_size = 0.62
    grid = build_grayscale_grid(img, cell_size=cell_size, show_values=False)
    grid_lbl = Text("Imagem (7 × 7) + Janela W (3 × 3)", font_size=BODY_SIZE - 2, color=TEXT_COLOR, weight="BOLD")
    grid_group = VGroup(grid_lbl, grid).arrange(DOWN, buff=0.22).move_to([-3.6, -0.4, 0])

    # Índice linear para posição (row, col) na grade 7x7
    def cell_center(r: int, c: int) -> np.ndarray:
        return grid[r * 7 + c].get_center()

    window = Square(
        side_length=cell_size * 3.04,
        color=ACCENT_COLOR,
        stroke_width=4,
    ).move_to(cell_center(1, 1))

    states = [
        (
            1,
            1,
            _build_region_panel(
                "1. Região Plana",
                MUTED_COLOR,
                r"M \approx \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}",
                r"\lambda_1 \approx 0,\quad \lambda_2 \approx 0",
                "Sem variação de intensidade em nenhuma direção.",
                rx=0.22,
                ry=0.22,
                angle=0.0,
            ),
            False,
        ),
        (
            5,
            3,
            _build_region_panel(
                "2. Região de Borda",
                PRIMARY_COLOR,
                r"M \approx \begin{bmatrix} 900 & 0 \\ 0 & 0 \end{bmatrix}",
                r"\lambda_1 \gg \lambda_2 \approx 0",
                "Variação forte perpendicular à borda (v₁); nula ao longo dela.",
                rx=1.05,
                ry=0.2,
                angle=0.0,
            ),
            False,
        ),
        (
            3,
            3,
            _build_region_panel(
                "3. Região de Canto (Corner)",
                SUCCESS_COLOR,
                r"M = \begin{bmatrix} 620 & 280 \\ 280 & 620 \end{bmatrix}",
                r"\lambda_1 = 900,\quad \lambda_2 = 340 \;\; (\text{ambos grandes!})",
                "Alta variação em todas as direções: vértice detectado!",
                rx=1.0,
                ry=0.65,
                angle=math.pi / 4,
            ),
            True,
        ),
    ]

    scene.play(FadeIn(grid_group, scale=0.95), run_time=0.6)
    first_panel = states[0][2].move_to([3.1, -0.45, 0])
    scene.play(Create(window), FadeIn(first_panel), run_time=0.6)
    scene.wait(1.3)

    current_panel = first_panel
    corner_marker = None
    for row, col, next_panel, is_corner in states[1:]:
        next_panel.move_to([3.1, -0.45, 0])
        scene.play(
            window.animate.move_to(cell_center(row, col)),
            FadeOut(current_panel),
            FadeIn(next_panel),
            run_time=0.75,
        )
        current_panel = next_panel
        if is_corner:
            corner_pt = cell_center(3, 3) + np.array([-cell_size / 2, cell_size / 2, 0])
            corner_marker = VGroup(
                Dot(corner_pt, color=SUCCESS_COLOR, radius=0.11),
                Circle(radius=0.24, color=SUCCESS_COLOR, stroke_width=3).move_to(corner_pt),
            )
            scene.play(FadeIn(corner_marker, scale=1.4), run_time=0.5)
        scene.wait(1.6)

    transition_stage(scene, [header, grid_group, window, current_panel, corner_marker])


def _slide_harris_response(scene: Scene) -> None:
    """Slide 3.3: Critério de Harris (Traço e Determinante sem calcular raízes)."""
    header = create_slide_header("TÓPICO 03", "Critério de Resposta de Harris (R)")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    axes = Axes(
        x_range=[0, 10, 2],
        y_range=[0, 10, 2],
        x_length=5.0,
        y_length=4.7,
        axis_config={"color": MUTED_COLOR, "include_tip": True},
    ).move_to([-3.45, -0.5, 0])

    x_lbl = MathTex(r"\lambda_1", font_size=26, color=TEXT_COLOR).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
    y_lbl = MathTex(r"\lambda_2", font_size=26, color=TEXT_COLOR).next_to(axes.y_axis.get_end(), UP, buff=0.1)

    flat_zone = Polygon(
        axes.c2p(0, 0), axes.c2p(2.5, 0), axes.c2p(2.5, 2.5), axes.c2p(0, 2.5),
        fill_color=MUTED_COLOR, fill_opacity=0.28, stroke_color=MUTED_COLOR, stroke_width=1.5,
    )
    edge_x = Polygon(
        axes.c2p(2.5, 0), axes.c2p(9.5, 0), axes.c2p(9.5, 2.2), axes.c2p(2.5, 2.2),
        fill_color=DANGER_COLOR, fill_opacity=0.22, stroke_color=DANGER_COLOR, stroke_width=1.5,
    )
    edge_y = Polygon(
        axes.c2p(0, 2.5), axes.c2p(2.2, 2.5), axes.c2p(2.2, 9.5), axes.c2p(0, 9.5),
        fill_color=DANGER_COLOR, fill_opacity=0.22, stroke_color=DANGER_COLOR, stroke_width=1.5,
    )
    corner_zone = Polygon(
        axes.c2p(2.8, 2.8), axes.c2p(9.5, 2.8), axes.c2p(9.5, 9.5), axes.c2p(2.8, 9.5),
        fill_color=SUCCESS_COLOR, fill_opacity=0.26, stroke_color=SUCCESS_COLOR, stroke_width=2,
    )

    lbl_flat = Text("Plana", font_size=14, color=TEXT_COLOR, weight="BOLD").move_to(axes.c2p(1.25, 1.25))
    lbl_edge1 = Text("Borda (R < 0)", font_size=14, color=DANGER_COLOR, weight="BOLD").move_to(axes.c2p(6.0, 1.1))
    lbl_edge2 = Text("Borda", font_size=14, color=DANGER_COLOR, weight="BOLD").move_to(axes.c2p(1.1, 6.0))
    lbl_corner = Text("Canto (R >> 0)", font_size=16, color=SUCCESS_COLOR, weight="BOLD").move_to(axes.c2p(6.2, 6.2))

    plot_group = VGroup(axes, x_lbl, y_lbl, flat_zone, edge_x, edge_y, corner_zone, lbl_flat, lbl_edge1, lbl_edge2, lbl_corner)

    card = create_info_card(
        "Atalho Algébrico (Harris, 1988)",
        [
            Text("Evita calcular autovalores pixel a pixel usando:", font_size=SMALL_SIZE + 1, color=TEXT_COLOR),
            MathTex(
                r"\det(M) = \lambda_1 \lambda_2 \qquad \operatorname{tr}(M) = \lambda_1 + \lambda_2",
                font_size=24,
                color=PRIMARY_COLOR,
            ),
            MathTex(
                r"R = \det(M) - k\,(\operatorname{tr}(M))^2 \quad (k \approx 0{,}04)",
                font_size=24,
                color=ACCENT_COLOR,
            ),
            Text("• R >> 0: ambos autovalores grandes → Canto.", font_size=SMALL_SIZE + 1, color=SUCCESS_COLOR),
            Text("• R < 0: um autovalor domina → Borda.", font_size=SMALL_SIZE + 1, color=DANGER_COLOR),
            Text("• |R| pequeno: região homogênea → Plana.", font_size=SMALL_SIZE + 1, color=MUTED_COLOR),
        ],
        width=5.9,
        border_color=ACCENT_COLOR,
    ).move_to([3.25, -0.45, 0])

    scene.play(Create(axes), FadeIn(x_lbl, y_lbl), run_time=0.6)
    scene.play(FadeIn(flat_zone, edge_x, edge_y, corner_zone), FadeIn(lbl_flat, lbl_edge1, lbl_edge2, lbl_corner), run_time=0.75)
    scene.play(Write(card), run_time=0.75)
    scene.wait(2.2)

    transition_stage(scene, [header, plot_group, card])


def build_eigen_symmetric_corners(scene: Scene) -> None:
    """Executa todos os slides do Tópico 3 na cena informada."""
    apply_theme(scene)
    play_section_intro(
        scene,
        "TÓPICO 03",
        "Autovalores, Autovetores e Matrizes Simétricas",
        "O Tensor de Estrutura e a Detecção de Cantos de Harris",
        accent_color=SUCCESS_COLOR,
    )
    _slide_structure_tensor(scene)
    _slide_three_regions(scene)
    _slide_harris_response(scene)


class EigenSymmetricCornersScene(Scene):
    """Cena independente para o Tópico 3."""

    def construct(self) -> None:
        build_eigen_symmetric_corners(self)


if __name__ == "__main__":
    pass
