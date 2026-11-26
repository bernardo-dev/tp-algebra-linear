"""Tópico 2: Transformações Lineares aplicadas a Imagens."""

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
    ApplyMatrix,
    Arrow,
    Create,
    FadeIn,
    MathTex,
    NumberPlane,
    Scene,
    Square,
    Text,
    VGroup,
)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    MUTED_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SMALL_SIZE,
    SUCCESS_COLOR,
    TEXT_COLOR,
)


def _create_local_canvas() -> tuple[NumberPlane, VGroup, np.ndarray]:
    """Cria um plano cartesiano compacto à esquerda com uma figura de pixels em L."""
    plane = NumberPlane(
        x_range=[-3, 3, 1],
        y_range=[-3, 3, 1],
        x_length=5.2,
        y_length=5.2,
        background_line_style={
            "stroke_color": "#334155",
            "stroke_width": 1.2,
            "stroke_opacity": 0.85,
        },
        axis_config={"stroke_color": MUTED_COLOR, "stroke_width": 2},
    ).move_to([-3.35, -0.45, 0])

    unit = plane.get_x_unit_size()
    # Figura assimétrica em L (coordenadas locais em unidades da grade)
    cells = [
        (0, 0, PRIMARY_COLOR),
        (0, 1, PRIMARY_COLOR),
        (0, 2, ACCENT_COLOR),
        (1, 0, SECONDARY_COLOR),
    ]
    patch = VGroup()
    for cx, cy, color in cells:
        sq = Square(
            side_length=unit * 0.94,
            fill_color=color,
            fill_opacity=0.88,
            stroke_color=TEXT_COLOR,
            stroke_width=1.8,
        )
        sq.move_to(plane.c2p(cx + 0.5, cy + 0.5))
        patch.add(sq)

    origin = plane.c2p(0, 0)
    i_hat = Arrow(origin, plane.c2p(1, 0), buff=0, color=SUCCESS_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.22)
    j_hat = Arrow(origin, plane.c2p(0, 1), buff=0, color=SECONDARY_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.22)
    shape_group = VGroup(patch, i_hat, j_hat)
    return plane, shape_group, origin


def _slide_coordinate_mapping(scene: Scene) -> None:
    """Slide 2.1: Mapeamento matricial de coordenadas x' = Ax."""
    header = create_slide_header("TÓPICO 02", "Mapeamento de Coordenadas: x' = Ax")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    plane, shape_group, _ = _create_local_canvas()
    basis_legend = VGroup(
        MathTex(r"\hat{\imath} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}", font_size=22, color=SUCCESS_COLOR),
        MathTex(r"\hat{\jmath} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}", font_size=22, color=SECONDARY_COLOR),
    ).arrange(RIGHT, buff=0.5).next_to(plane, DOWN, buff=0.14)

    card = create_info_card(
        "Transformação Linear 2D",
        [
            Text("Cada pixel tem coordenada vetorial x = (x, y):", font_size=BODY_SIZE - 1, color=TEXT_COLOR),
            MathTex(
                r"\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = A\mathbf{x}",
                font_size=28,
                color=ACCENT_COLOR,
            ),
            Text("• A 1ª coluna de A indica para onde vai î.", font_size=SMALL_SIZE + 1, color=SUCCESS_COLOR),
            Text("• A 2ª coluna de A indica para onde vai ĵ.", font_size=SMALL_SIZE + 1, color=SECONDARY_COLOR),
            Text("• Origem preservada e retas permanecem retas.", font_size=SMALL_SIZE + 1, color=MUTED_COLOR),
        ],
        width=5.8,
        border_color=PRIMARY_COLOR,
    ).move_to([3.25, -0.45, 0])

    scene.play(Create(plane), run_time=0.6)
    scene.play(FadeIn(shape_group), FadeIn(basis_legend), run_time=0.6)
    scene.play(FadeIn(card, shift=LEFT * 0.2), run_time=0.6)
    scene.wait(1.8)

    transition_stage(scene, [header, plane, shape_group, basis_legend, card])


def _slide_scale_and_reflection(scene: Scene) -> None:
    """Slide 2.2: Escala (Scaling) e Reflexão Horizontal."""
    header = create_slide_header("TÓPICO 02", "Escala e Reflexão via Matrizes")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    plane, shape_group, origin = _create_local_canvas()
    shape_ref = shape_group.copy()

    scale_card = create_info_card(
        "1. Escala (Redimensionamento)",
        [
            MathTex(
                r"S = \begin{bmatrix} 1{,}5 & 0 \\ 0 & 0{,}8 \end{bmatrix}",
                font_size=30,
                color=ACCENT_COLOR,
            ),
            Text("Alonga horizontalmente (1,5×) e comprime verticalmente (0,8×).", font_size=BODY_SIZE - 2, color=TEXT_COLOR),
            MathTex(r"\det(S) = 1{,}5 \times 0{,}8 = 1{,}20", font_size=24, color=PRIMARY_COLOR),
        ],
        width=5.7,
        border_color=ACCENT_COLOR,
    ).move_to([3.25, -0.45, 0])

    scene.play(Create(plane), FadeIn(shape_group), FadeIn(scale_card), run_time=0.65)
    scene.play(ApplyMatrix([[1.5, 0.0], [0.0, 0.8]], shape_group, about_point=origin), run_time=1.3)
    scene.wait(1.0)

    # Restaurar e mostrar Reflexão
    refl_card = create_info_card(
        "2. Reflexão Horizontal (Espelho)",
        [
            MathTex(
                r"F = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}",
                font_size=30,
                color=SECONDARY_COLOR,
            ),
            Text("Inverte o eixo X mantendo Y inalterado.", font_size=BODY_SIZE - 2, color=TEXT_COLOR),
            MathTex(r"\det(F) = -1 \quad (\text{inverte orientação})", font_size=24, color=SECONDARY_COLOR),
        ],
        width=5.7,
        border_color=SECONDARY_COLOR,
    ).move_to([3.25, -0.45, 0])

    scene.play(
        shape_group.animate.become(shape_ref),
        FadeIn(refl_card),
        scale_card.animate.set_opacity(0.0),
        run_time=0.6,
    )
    scene.remove(scale_card)
    scene.play(ApplyMatrix([[-1.0, 0.0], [0.0, 1.0]], shape_group, about_point=origin), run_time=1.3)
    scene.wait(1.2)

    transition_stage(scene, [header, plane, shape_group, refl_card])


def _slide_rotation_and_shear(scene: Scene) -> None:
    """Slide 2.3: Rotação, Cisalhamento (Shear) e Composição."""
    header = create_slide_header("TÓPICO 02", "Rotação, Cisalhamento e Composição")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    plane, shape_group, origin = _create_local_canvas()

    c45 = math.cos(math.pi / 4)
    s45 = math.sin(math.pi / 4)
    rot_mat = [[c45, -s45], [s45, c45]]
    shear_mat = [[1.0, 0.6], [0.0, 1.0]]

    rot_card = create_info_card(
        "Rotação (θ = 45°)",
        [
            MathTex(
                r"R_{45^\circ} = \begin{bmatrix} \cos 45^\circ & -\sin 45^\circ \\ \sin 45^\circ & \cos 45^\circ \end{bmatrix}",
                font_size=24,
                color=PRIMARY_COLOR,
            ),
            Text("Matriz ortogonal: preserva distâncias e ângulos.", font_size=SMALL_SIZE + 1, color=TEXT_COLOR),
            MathTex(r"R^T R = I \quad\text{e}\quad \det(R) = 1", font_size=22, color=SUCCESS_COLOR),
        ],
        width=5.7,
        border_color=PRIMARY_COLOR,
    )

    shear_card = create_info_card(
        "Cisalhamento + Composição",
        [
            MathTex(
                r"H = \begin{bmatrix} 1 & 0{,}6 \\ 0 & 1 \end{bmatrix}",
                font_size=25,
                color=ACCENT_COLOR,
            ),
            Text("Múltiplos efeitos em um único produto matricial:", font_size=SMALL_SIZE + 1, color=TEXT_COLOR),
            MathTex(r"\mathbf{x}'' = (H \cdot R_{45^\circ})\,\mathbf{x}", font_size=24, color=SECONDARY_COLOR),
        ],
        width=5.7,
        border_color=ACCENT_COLOR,
    )
    cards_col = VGroup(rot_card, shear_card).arrange(DOWN, buff=0.25).move_to([3.25, -0.48, 0])

    scene.play(Create(plane), FadeIn(shape_group), FadeIn(rot_card), run_time=0.65)
    scene.play(ApplyMatrix(rot_mat, shape_group, about_point=origin), run_time=1.35)
    scene.wait(0.7)

    scene.play(FadeIn(shear_card, shift=UP * 0.15), run_time=0.55)
    scene.play(ApplyMatrix(shear_mat, shape_group, about_point=origin), run_time=1.35)
    scene.wait(1.6)

    transition_stage(scene, [header, plane, shape_group, rot_card, shear_card])


def build_linear_transformations(scene: Scene) -> None:
    """Executa todos os slides do Tópico 2 na cena informada."""
    apply_theme(scene)
    play_section_intro(
        scene,
        "TÓPICO 02",
        "Transformações Lineares",
        "Rotações, Escalas, Reflexões e Cisalhamento por Matrizes",
        accent_color=SECONDARY_COLOR,
    )
    _slide_coordinate_mapping(scene)
    _slide_scale_and_reflection(scene)
    _slide_rotation_and_shear(scene)


class LinearTransformationsScene(Scene):
    """Cena independente para o Tópico 2."""

    def construct(self) -> None:
        build_linear_transformations(self)


if __name__ == "__main__":
    pass
