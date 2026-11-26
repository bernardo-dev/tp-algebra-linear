"""Tópico 4: Thresholding (Limiarização) e Segmentação Matricial."""

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
    Axes,
    Create,
    FadeIn,
    FadeOut,
    GrowArrow,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
    rgb_to_color,
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

SAMPLE_IMG = np.array([
    [35, 50, 45, 70, 30, 40],
    [40, 210, 225, 200, 60, 35],
    [75, 235, 245, 230, 190, 50],
    [30, 195, 240, 220, 185, 65],
    [55, 60, 180, 215, 70, 45],
    [30, 40, 55, 45, 35, 25],
])


def _slide_threshold_definition(scene: Scene) -> None:
    """Slide 4.1: Definição algébrica da limiarização global (T = 128)."""
    header = create_slide_header("TÓPICO 04", "Limiarização Global: Binarização Matricial")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    t_val = 128
    binary_img = np.where(SAMPLE_IMG >= t_val, 255, 0)

    orig_grid = build_grayscale_grid(SAMPLE_IMG, cell_size=0.58, show_values=True, font_size=12)
    orig_lbl = Text("Matriz Original I", font_size=BODY_SIZE - 1, color=PRIMARY_COLOR, weight="BOLD")
    orig_box = VGroup(orig_lbl, orig_grid).arrange(DOWN, buff=0.2).move_to([-4.2, -0.5, 0])

    bin_grid = build_grayscale_grid(binary_img, cell_size=0.58, show_values=True, font_size=12)
    bin_lbl = Text("Matriz Binária B (T = 128)", font_size=BODY_SIZE - 1, color=SUCCESS_COLOR, weight="BOLD")
    bin_box = VGroup(bin_lbl, bin_grid).arrange(DOWN, buff=0.2).move_to([4.15, -0.5, 0])

    rule_card = create_info_card(
        "Operador por Limiar T",
        [
            MathTex(
                r"B_{i,j} = \begin{cases} 255, & I_{i,j} \ge T \\ 0, & I_{i,j} < T \end{cases}",
                font_size=26,
                color=ACCENT_COLOR,
            ),
            Text("Separa fundo escuro (< T) do objeto claro (≥ T).", font_size=SMALL_SIZE, color=TEXT_COLOR),
        ],
        width=4.1,
        border_color=ACCENT_COLOR,
    ).move_to([0.0, -0.45, 0])

    a1 = Arrow(orig_box.get_right() + RIGHT * 0.08, rule_card.get_left() + LEFT * 0.08, color=ACCENT_COLOR, stroke_width=3)
    a2 = Arrow(rule_card.get_right() + RIGHT * 0.08, bin_box.get_left() + LEFT * 0.08, color=SUCCESS_COLOR, stroke_width=3)

    scene.play(FadeIn(orig_box, scale=0.95), run_time=0.6)
    scene.play(GrowArrow(a1), FadeIn(rule_card), run_time=0.65)
    scene.play(GrowArrow(a2), FadeIn(bin_box, scale=0.95), run_time=0.7)
    scene.wait(2.0)

    transition_stage(scene, [header, orig_box, rule_card, bin_box, a1, a2])


def _slide_histogram_threshold_sweep(scene: Scene) -> None:
    """Slide 4.2: Histograma bimodal e o impacto da escolha do limiar T."""
    header = create_slide_header("TÓPICO 04", "Histograma de Intensidades e Escolha de T")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    axes = Axes(
        x_range=[0, 260, 64],
        y_range=[0, 10, 2],
        x_length=5.5,
        y_length=4.1,
        axis_config={"color": MUTED_COLOR, "include_tip": True},
    ).move_to([-3.35, -0.55, 0])

    x_lbl = Text("Intensidade (0 – 255)", font_size=SMALL_SIZE, color=MUTED_COLOR).next_to(axes.x_axis, DOWN, buff=0.18)
    y_lbl = Text("Pixels", font_size=SMALL_SIZE, color=MUTED_COLOR).next_to(axes.y_axis.get_end(), UP, buff=0.12)

    # Barras do histograma bimodal calculado a partir de SAMPLE_IMG
    bins = np.linspace(0, 256, 13)
    counts, _ = np.histogram(SAMPLE_IMG, bins=bins)
    bars = VGroup()
    bar_w = (axes.c2p(21, 0)[0] - axes.c2p(0, 0)[0]) * 0.82
    for idx, cnt in enumerate(counts):
        if cnt == 0:
            continue
        center_x = (bins[idx] + bins[idx + 1]) / 2.0
        bottom = axes.c2p(center_x, 0)
        top = axes.c2p(center_x, float(cnt))
        bar_h = max(top[1] - bottom[1], 0.05)
        color = PRIMARY_COLOR if center_x < 128 else SECONDARY_COLOR
        bar = Rectangle(
            width=bar_w,
            height=bar_h,
            fill_color=color,
            fill_opacity=0.82,
            stroke_color=TEXT_COLOR,
            stroke_width=1,
        ).move_to([bottom[0], bottom[1] + bar_h / 2.0, 0])
        bars.add(bar)

    hist_group = VGroup(axes, x_lbl, y_lbl, bars)

    sweep_configs = [
        (45, DANGER_COLOR, "T = 45 → Limiar baixo (ruído no fundo)"),
        (215, ACCENT_COLOR, "T = 215 → Limiar alto (perde partes do objeto)"),
        (130, SUCCESS_COLOR, "T = 130 → Limiar ótimo (vale entre os picos)"),
    ]

    def make_threshold_line(t_val: int, color: str) -> VGroup:
        p0 = axes.c2p(t_val, 0)
        p1 = axes.c2p(t_val, 9.5)
        line = Line(p0, p1, color=color, stroke_width=4)
        tag = Text(f"T={t_val}", font_size=SMALL_SIZE, color=color, weight="BOLD").next_to(p1, UP, buff=0.08)
        return VGroup(line, tag)

    def make_output_panel(t_val: int, color: str, msg: str) -> VGroup:
        b_mat = np.where(SAMPLE_IMG >= t_val, 255, 0)
        b_grid = build_grayscale_grid(b_mat, cell_size=0.56, show_values=True, font_size=11)
        status_lbl = Text(msg, font_size=SMALL_SIZE + 1, color=color, weight="BOLD")
        box = RoundedRectangle(
            corner_radius=0.14,
            width=max(status_lbl.width + 0.5, 4.8),
            height=status_lbl.height + 0.3,
            fill_color=CARD_BG,
            fill_opacity=0.9,
            stroke_color=color,
            stroke_width=2,
        )
        badge = VGroup(box, status_lbl)
        return VGroup(b_grid, badge).arrange(DOWN, buff=0.25).move_to([3.3, -0.45, 0])

    t0, c0, m0 = sweep_configs[0]
    t_line = make_threshold_line(t0, c0)
    out_panel = make_output_panel(t0, c0, m0)

    scene.play(Create(axes), FadeIn(x_lbl, y_lbl), FadeIn(bars, lag_ratio=0.08), run_time=0.8)
    scene.play(Create(t_line), FadeIn(out_panel), run_time=0.65)
    scene.wait(1.3)

    for t_val, color, msg in sweep_configs[1:]:
        next_line = make_threshold_line(t_val, color)
        next_panel = make_output_panel(t_val, color, msg)
        scene.play(
            FadeOut(t_line),
            FadeIn(next_line),
            FadeOut(out_panel),
            FadeIn(next_panel),
            run_time=0.75,
        )
        t_line = next_line
        out_panel = next_panel
        scene.wait(1.4)

    transition_stage(scene, [header, hist_group, t_line, out_panel])


def _slide_hadamard_masking(scene: Scene) -> None:
    """Slide 4.3: Máscara binária e Produto de Hadamard (I_seg = I ⊙ M_B)."""
    header = create_slide_header("TÓPICO 04", "Segmentação com Máscara e Produto de Hadamard")
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    mask = np.where(SAMPLE_IMG >= 130, 1, 0)
    segmented = SAMPLE_IMG * mask

    grid_i = build_grayscale_grid(SAMPLE_IMG, cell_size=0.46, show_values=True, font_size=10)
    lbl_i = Text("Original I", font_size=SMALL_SIZE + 1, color=PRIMARY_COLOR, weight="BOLD")
    box_i = VGroup(lbl_i, grid_i).arrange(DOWN, buff=0.18)

    op_dot = MathTex(r"\odot", font_size=42, color=ACCENT_COLOR)

    grid_m = build_grayscale_grid(mask * 255, cell_size=0.46, show_values=False)
    # Sobrepõe os valores 0 e 1 diretamente na máscara normalizada
    for idx, val in enumerate(mask.flatten()):
        txt_c = "#000000" if val == 1 else "#FFFFFF"
        lbl = Text(str(int(val)), font_size=12, color=txt_c, weight="BOLD").move_to(grid_m[idx].get_center())
        grid_m[idx].add(lbl)
    lbl_m = Text("Máscara M ∈ {0,1}", font_size=SMALL_SIZE + 1, color=SECONDARY_COLOR, weight="BOLD")
    box_m = VGroup(lbl_m, grid_m).arrange(DOWN, buff=0.18)

    op_eq = MathTex(r"=", font_size=42, color=TEXT_COLOR)

    grid_s = build_grayscale_grid(segmented, cell_size=0.46, show_values=True, font_size=10)
    lbl_s = Text("Segmentada I_seg", font_size=SMALL_SIZE + 1, color=SUCCESS_COLOR, weight="BOLD")
    box_s = VGroup(lbl_s, grid_s).arrange(DOWN, buff=0.18)

    row = VGroup(box_i, op_dot, box_m, op_eq, box_s).arrange(RIGHT, buff=0.35).move_to([0, 0.2, 0])

    footer_eq = MathTex(
        r"I_{\text{seg}} = I \odot M_B \quad\Longleftrightarrow\quad (I_{\text{seg}})_{i,j} = I_{i,j} \cdot (M_B)_{i,j}",
        font_size=28,
        color=TEXT_COLOR,
    )
    footer_note = Text(
        "O produto elemento a elemento (Hadamard) zera o fundo e preserva as intensidades do objeto.",
        font_size=SMALL_SIZE + 1,
        color=MUTED_COLOR,
    )
    footer_content = VGroup(footer_eq, footer_note).arrange(DOWN, buff=0.14)
    footer_bg = RoundedRectangle(
        corner_radius=0.16,
        width=11.8,
        height=footer_content.height + 0.4,
        fill_color=CARD_BG,
        fill_opacity=0.9,
        stroke_color=SUCCESS_COLOR,
        stroke_width=2,
    )
    footer = VGroup(footer_bg, footer_content).move_to([0, -2.35, 0])

    scene.play(FadeIn(box_i, shift=UP * 0.15), run_time=0.55)
    scene.play(Write(op_dot), FadeIn(box_m, shift=UP * 0.15), run_time=0.6)
    scene.play(Write(op_eq), FadeIn(box_s, scale=0.95), run_time=0.65)
    scene.play(FadeIn(footer, shift=UP * 0.15), run_time=0.55)
    scene.wait(2.1)

    transition_stage(scene, [header, row, footer])


def build_thresholding(scene: Scene) -> None:
    """Executa todos os slides do Tópico 4 na cena informada."""
    apply_theme(scene)
    play_section_intro(
        scene,
        "TÓPICO 04",
        "Thresholding (Limiarização)",
        "Segmentação Binária, Histogramas e Produto de Hadamard",
        accent_color=ACCENT_COLOR,
    )
    _slide_threshold_definition(scene)
    _slide_histogram_threshold_sweep(scene)
    _slide_hadamard_masking(scene)


class ThresholdingScene(Scene):
    """Cena independente para o Tópico 4."""

    def construct(self) -> None:
        build_thresholding(self)


if __name__ == "__main__":
    pass
