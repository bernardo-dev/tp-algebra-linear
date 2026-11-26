"""Ponto de entrada principal da apresentação em Manim."""

from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Line,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

# Garante importações absolutas de `src.*` ao executar `manim src/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components.slide_layout import apply_theme, create_slide_header, transition_stage
from src.config.theme import (
    ACCENT_COLOR,
    BODY_SIZE,
    CARD_BG,
    MUTED_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SECTION_TITLE_SIZE,
    SMALL_SIZE,
    SUBTITLE_SIZE,
    SUCCESS_COLOR,
    TEXT_COLOR,
)
from src.topics import (
    EigenSymmetricCornersScene as _EigenSymmetricCornersScene,
    LinearTransformationsScene as _LinearTransformationsScene,
    ThresholdingScene as _ThresholdingScene,
    VectorsMatricesTensorsScene as _VectorsMatricesTensorsScene,
    build_eigen_symmetric_corners,
    build_linear_transformations,
    build_thresholding,
    build_vectors_matrices_tensors,
)


def build_intro(scene: Scene) -> None:
    """Constrói o slide de abertura e o roteiro dos 4 tópicos."""
    apply_theme(scene)

    main_title = Text(
        "Álgebra Linear no Processamento de Imagens",
        font_size=SECTION_TITLE_SIZE,
        color=TEXT_COLOR,
        weight="BOLD",
    )
    if main_title.width > 12.4:
        main_title.scale_to_fit_width(12.4)

    subtitle = Text(
        "Dos pixels em espaços vetoriais à detecção de cantos e segmentação",
        font_size=SUBTITLE_SIZE,
        color=PRIMARY_COLOR,
    )
    divider = Line(LEFT * 4.5, RIGHT * 4.5, color=ACCENT_COLOR, stroke_width=3)
    hero = VGroup(main_title, divider, subtitle).arrange(DOWN, buff=0.3).move_to([0, 1.45, 0])

    roadmap_specs = [
        ("01", "Vetores, Matrizes e Tensores", "Estrutura digital em R³, R^(H×W) e R^(H×W×3)", PRIMARY_COLOR),
        ("02", "Transformações Lineares", "Escala, reflexão, rotação e cisalhamento (Ax)", SECONDARY_COLOR),
        ("03", "Autovalores e Simétricas", "Tensor de estrutura e detector de Harris", SUCCESS_COLOR),
        ("04", "Thresholding", "Limiarização, histograma e produto de Hadamard", ACCENT_COLOR),
    ]

    cards = VGroup()
    for num, ttl, desc, col in roadmap_specs:
        badge_txt = Text(num, font_size=BODY_SIZE, color="#0B1120", weight="BOLD")
        badge_box = RoundedRectangle(
            corner_radius=0.1, width=0.55, height=0.45, fill_color=col, fill_opacity=1.0, stroke_width=0
        )
        badge = VGroup(badge_box, badge_txt)

        t_mob = Text(ttl, font_size=BODY_SIZE - 2, color=TEXT_COLOR, weight="BOLD")
        d_mob = Text(desc, font_size=SMALL_SIZE - 1, color=MUTED_COLOR)
        text_col = VGroup(t_mob, d_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        row = VGroup(badge, text_col).arrange(RIGHT, buff=0.24)
        if row.width > 5.6:
            row.scale_to_fit_width(5.6)

        bg = RoundedRectangle(
            corner_radius=0.16,
            width=6.0,
            height=1.25,
            fill_color=CARD_BG,
            fill_opacity=0.9,
            stroke_color=col,
            stroke_width=2,
        )
        row.move_to(bg.get_center()).align_to(bg, LEFT).shift(RIGHT * 0.22)
        cards.add(VGroup(bg, row))

    cards.arrange_in_grid(rows=2, cols=2, buff=0.32).move_to([0, -1.35, 0])

    scene.play(Write(main_title), Create(divider), run_time=0.85)
    scene.play(FadeIn(subtitle, shift=UP * 0.15), run_time=0.45)
    scene.play(FadeIn(cards, lag_ratio=0.15), run_time=0.85)
    scene.wait(2.2)

    transition_stage(scene, [hero, cards])


def build_conclusion(scene: Scene) -> None:
    """Constrói o slide de encerramento resumindo os conceitos conectados."""
    apply_theme(scene)
    header = create_slide_header("RESUMO FINAL", "Álgebra Linear como Linguagem da Visão Computacional")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    rows_data = [
        ("Vetores, Matrizes e Tensores", "Pixel RGB ∈ R³, Escala de Cinza ∈ R^(H×W), RGB ∈ R^(H×W×3)", PRIMARY_COLOR),
        ("Transformações Lineares", "Coordenadas mapeadas por x' = Ax (rotação, escala, shear)", SECONDARY_COLOR),
        ("Autovalores e Simétricas", "M = M^T revela cantos quando λ₁ e λ₂ são grandes (Harris)", SUCCESS_COLOR),
        ("Thresholding", "Binarização por limiar T e máscara I_seg = I ⊙ M_B", ACCENT_COLOR),
    ]

    summary_rows = VGroup()
    for topic, concept, color in rows_data:
        t_mob = Text(topic, font_size=BODY_SIZE - 1, color=color, weight="BOLD")
        c_mob = Text(concept, font_size=SMALL_SIZE + 1, color=TEXT_COLOR)
        line_content = VGroup(t_mob, c_mob).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        bg = RoundedRectangle(
            corner_radius=0.14,
            width=11.8,
            height=1.08,
            fill_color=CARD_BG,
            fill_opacity=0.9,
            stroke_color=color,
            stroke_width=2,
        )
        line_content.move_to(bg.get_center()).align_to(bg, LEFT).shift(RIGHT * 0.35)
        summary_rows.add(VGroup(bg, line_content))

    summary_rows.arrange(DOWN, buff=0.22).move_to([0, -0.35, 0])

    thanks = Text("Obrigado pela atenção!", font_size=SUBTITLE_SIZE, color=ACCENT_COLOR, weight="BOLD").move_to([0, -3.35, 0])

    scene.play(FadeIn(summary_rows, lag_ratio=0.16), run_time=0.95)
    scene.play(FadeIn(thanks, shift=UP * 0.15), run_time=0.45)
    scene.wait(2.5)
    scene.play(FadeOut(header), FadeOut(summary_rows), FadeOut(thanks), run_time=0.55)


class IntroScene(Scene):
    """Cena independente da introdução."""

    def construct(self) -> None:
        build_intro(self)


class ConclusionScene(Scene):
    """Cena independente de encerramento."""

    def construct(self) -> None:
        build_conclusion(self)


class VectorsMatricesTensorsScene(_VectorsMatricesTensorsScene):
    """Re-exportação da cena do Tópico 1 para execução direta via main.py."""


class LinearTransformationsScene(_LinearTransformationsScene):
    """Re-exportação da cena do Tópico 2 para execução direta via main.py."""


class EigenSymmetricCornersScene(_EigenSymmetricCornersScene):
    """Re-exportação da cena do Tópico 3 para execução direta via main.py."""


class ThresholdingScene(_ThresholdingScene):
    """Re-exportação da cena do Tópico 4 para execução direta via main.py."""


class FullPresentationScene(Scene):
    """Apresentação completa combinando introdução, 4 tópicos e conclusão."""

    def construct(self) -> None:
        build_intro(self)
        build_vectors_matrices_tensors(self)
        build_linear_transformations(self)
        build_eigen_symmetric_corners(self)
        build_thresholding(self)
        build_conclusion(self)


__all__ = [
    "ConclusionScene",
    "EigenSymmetricCornersScene",
    "FullPresentationScene",
    "IntroScene",
    "LinearTransformationsScene",
    "ThresholdingScene",
    "VectorsMatricesTensorsScene",
]
