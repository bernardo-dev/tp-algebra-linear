"""Tópico 5: Convolução e Produto Escalar (Desfoque, Nitidez e Detecção de Contornos)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    GrowArrow,
    ImageMobject,
    Indicate,
    Line,
    MathTex,
    RESAMPLING_ALGORITHMS,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    UpdateFromAlphaFunc,
    VGroup,
    Write,
    linear,
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
    BORDER_MUTED,
    CARD_BG,
    DANGER_COLOR,
    MUTED_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SMALL_SIZE,
    SUCCESS_COLOR,
    TEXT_COLOR,
    VIOLET_COLOR,
)

PARROT_PATH = Path(__file__).resolve().parents[2] / "images" / "parrot_20x20.png"

K_BLUR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0
K_SHARP = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
K_EDGE = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)

EMPTY_SLATE_RGBA = np.array([15, 23, 42, 255], dtype=np.uint8)


def _load_parrot_rgb() -> np.ndarray:
    """Carrega images/parrot_20x20.png como matriz RGB uint8 (20, 20, 3)."""
    with Image.open(PARROT_PATH) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def _to_rgba(rgb: np.ndarray) -> np.ndarray:
    """Converte uma imagem RGB (H, W, 3) uint8 para RGBA (H, W, 4) com alpha=255."""
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=-1)


def _to_grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    """Converte imagem RGB para luminância em 3 canais iguais (H, W, 3)."""
    gray = np.round(
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _convolve_rgb(img_rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica convolução 3×3 via produto escalar por canal com padding de borda."""
    padded = np.pad(img_rgb.astype(float), ((1, 1), (1, 1), (0, 0)), mode="edge")
    rows, cols, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=float)
    for i in range(rows):
        for j in range(cols):
            patch = padded[i : i + 3, j : j + 3, :]
            out[i, j] = np.tensordot(kernel, patch, axes=([0, 1], [0, 1]))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _get_padded_patch(img_rgb: np.ndarray, r: int, c: int) -> np.ndarray:
    """Retorna a vizinhança 3×3 (uint8) centrada em (r, c) com padding de borda."""
    padded = np.pad(img_rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return padded[r : r + 3, c : c + 3, :].copy()


def _build_grid_overlay(rows: int, cols: int, size: float, center: np.ndarray, opacity: float = 0.32) -> VGroup:
    """Constrói linhas internas de grade sobre uma região quadrada."""
    lines = VGroup()
    cell = size / cols
    x0 = center[0] - size / 2.0
    x1 = center[0] + size / 2.0
    y0 = center[1] - size / 2.0
    y1 = center[1] + size / 2.0
    for c in range(1, cols):
        x = x0 + c * cell
        lines.add(Line([x, y0, 0], [x, y1, 0], stroke_color="#475569", stroke_width=0.6, stroke_opacity=opacity))
    for r in range(1, rows):
        y = y1 - r * cell
        lines.add(Line([x0, y, 0], [x1, y, 0], stroke_color="#475569", stroke_width=0.6, stroke_opacity=opacity))
    return lines


def _build_image_panel(
    rgba: np.ndarray,
    title: str,
    title_color: str,
    border_color: str,
    center: list[float] | np.ndarray,
    panel_size: float = 4.0,
):
    """Cria painel emoldurado 20×20 com ImageMobject (nearest), grade sutil e título."""
    center_arr = np.array(center, dtype=float)
    rows, cols = rgba.shape[:2]

    img_mob = ImageMobject(rgba.copy())
    img_mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img_mob.scale_to_fit_width(panel_size)
    img_mob.move_to(center_arr)

    grid_lines = _build_grid_overlay(rows, cols, panel_size, center_arr, opacity=0.32)
    border = Square(
        side_length=panel_size,
        stroke_color=border_color,
        stroke_width=2.2,
        fill_opacity=0.0,
    ).move_to(center_arr)

    title_mob = Text(title, font_size=BODY_SIZE - 2, color=title_color, weight="BOLD")
    if title_mob.width > panel_size + 0.4:
        title_mob.scale_to_fit_width(panel_size + 0.4)
    title_mob.next_to(border, UP, buff=0.16)

    def cell_center(r: int, c: int) -> np.ndarray:
        cell = panel_size / cols
        x = center_arr[0] - panel_size / 2.0 + (c + 0.5) * cell
        y = center_arr[1] + panel_size / 2.0 - (r + 0.5) * cell
        return np.array([x, y, 0.0])

    return img_mob, grid_lines, border, title_mob, cell_center


def _slide_convolution_dot_product_concept(scene: Scene) -> None:
    """Slide 5.1: Convolução 2D interpretada como produto escalar em R^9."""
    header = create_slide_header(
        "TÓPICO 05",
        "Convolução 2D como Produto Escalar",
        badge_color=VIOLET_COLOR,
    )
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    # Patch real 3×3 centrado no olho do papagaio em (4, 12) em escala de cinza
    eye_patch = np.array([
        [119, 227, 227],
        [227, 22, 227],
        [227, 227, 227],
    ])
    patch_grid = build_grayscale_grid(eye_patch, cell_size=0.58, show_values=True, font_size=13)
    patch_lbl = Text("Janela 3×3 (Olho do Papagaio)", font_size=SMALL_SIZE, color=PRIMARY_COLOR, weight="BOLD")
    patch_box = VGroup(patch_lbl, patch_grid).arrange(DOWN, buff=0.14)

    dot_sym = MathTex(r"\cdot", font_size=40, color=ACCENT_COLOR)
    kernel_tex = MathTex(
        r"\frac{1}{16}\begin{bmatrix}1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1\end{bmatrix}",
        font_size=26,
        color=VIOLET_COLOR,
    )
    eq_sym = MathTex(r"=", font_size=34, color=TEXT_COLOR)

    out_sq = Square(
        side_length=0.68,
        fill_color=rgb_to_color([169 / 255, 169 / 255, 169 / 255]),
        fill_opacity=1.0,
        stroke_color=ACCENT_COLOR,
        stroke_width=2.8,
    )
    out_val = Text("169", font_size=BODY_SIZE - 2, color="#000000", weight="BOLD").move_to(out_sq.get_center())
    out_lbl = Text("Pixel (4,12)", font_size=SMALL_SIZE - 2, color=SUCCESS_COLOR, weight="BOLD")
    out_box = VGroup(out_lbl, VGroup(out_sq, out_val)).arrange(DOWN, buff=0.12)

    visual_row = VGroup(patch_box, dot_sym, kernel_tex, eq_sym, out_box).arrange(RIGHT, buff=0.22)
    if visual_row.width > 6.35:
        visual_row.scale_to_fit_width(6.35)

    vec_p = MathTex(
        r"\mathbf{p} = [119,\, 227,\, 227,\, 227,\, 22,\, 227,\, 227,\, 227,\, 227]^T \in \mathbb{R}^9",
        font_size=21,
        color=PRIMARY_COLOR,
    )
    vec_k = MathTex(
        r"\mathbf{k} = \tfrac{1}{16}[1,\, 2,\, 1,\, 2,\, 4,\, 2,\, 1,\, 2,\, 1]^T \in \mathbb{R}^9",
        font_size=21,
        color=VIOLET_COLOR,
    )
    vec_dot = MathTex(
        r"\langle \mathbf{k}, \mathbf{p} \rangle = \frac{2704}{16} = 169",
        font_size=24,
        color=ACCENT_COLOR,
    )
    vec_group = VGroup(vec_p, vec_k, vec_dot).arrange(DOWN, buff=0.18)
    if vec_group.width > 6.2:
        vec_group.scale_to_fit_width(6.2)

    left_col = VGroup(visual_row, vec_group).arrange(DOWN, buff=0.45).move_to([-3.45, -0.42, 0])

    right_card = create_info_card(
        "Janela 3×3 → Produto Escalar em R⁹",
        [
            MathTex(
                r"O_{i,j} = (I * K)_{i,j} = \sum_{u=-1}^{1}\sum_{v=-1}^{1} K_{u,v}\, I_{i+u, j+v}",
                font_size=24,
                color=PRIMARY_COLOR,
            ),
            MathTex(
                r"O_{i,j} = \langle \mathbf{k}, \mathbf{p} \rangle = \mathbf{k}^T \mathbf{p} = \sum_{m=1}^{9} k_m\, p_m",
                font_size=25,
                color=ACCENT_COLOR,
            ),
            Text("• A janela 3×3 desliza pixel a pixel por toda a imagem.", font_size=SMALL_SIZE, color=TEXT_COLOR),
            Text("• Em imagens RGB, o produto escalar atua em cada canal.", font_size=SMALL_SIZE, color=TEXT_COLOR),
            Text("• Os 9 pesos de K definem desfoque, nitidez ou contornos.", font_size=SMALL_SIZE, color=SUCCESS_COLOR),
        ],
        width=5.8,
        border_color=VIOLET_COLOR,
    ).move_to([3.35, -0.42, 0])

    scene.play(FadeIn(patch_box, shift=UP * 0.15), run_time=0.55)
    scene.play(Write(dot_sym), Write(kernel_tex), Write(eq_sym), FadeIn(out_box, scale=0.9), run_time=0.75)
    scene.play(FadeIn(vec_group, shift=UP * 0.12), run_time=0.6)
    scene.play(FadeIn(right_card, shift=LEFT * 0.2), run_time=0.65)
    scene.wait(2.1)

    transition_stage(scene, [header, left_col, right_card])


def _slide_convolution_walkthrough(
    scene: Scene,
    *,
    slide_title: str,
    left_label: str,
    right_label: str,
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    kernel_tex: str,
    accent_color: str,
    footer_text: str,
    is_grayscale: bool = False,
    initial_rgb: np.ndarray | None = None,
) -> None:
    """Executa animação passo a passo de convolução 3×3 sobre imagem 20×20."""
    header = create_slide_header("TÓPICO 05", slide_title, badge_color=VIOLET_COLOR)
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.4)

    panel_size = 4.0
    cell_size = panel_size / 20.0
    left_center = [-4.45, -0.25, 0.0]
    right_center = [4.45, -0.25, 0.0]

    start_left_rgb = initial_rgb if initial_rgb is not None else input_rgb
    start_left_title = "Original RGB (20×20)" if initial_rgb is not None else left_label
    left_img, left_grid, left_border, left_title, left_cell_center = _build_image_panel(
        _to_rgba(start_left_rgb),
        start_left_title,
        PRIMARY_COLOR,
        PRIMARY_COLOR,
        left_center,
        panel_size=panel_size,
    )

    output_rgba = _to_rgba(output_rgb)
    output_flat = output_rgba.reshape(-1, 4)
    blank_rgba = np.empty((20, 20, 4), dtype=np.uint8)
    blank_rgba[:, :] = EMPTY_SLATE_RGBA

    right_img, right_grid, right_border, right_title, right_cell_center = _build_image_panel(
        blank_rgba,
        right_label,
        accent_color,
        accent_color,
        right_center,
        panel_size=panel_size,
    )

    # Janela deslizante 3×3 na imagem de entrada e cursor 1×1 no resultado
    win_outer = Square(
        side_length=cell_size * 3.0,
        stroke_color=accent_color,
        stroke_width=3.5,
        fill_color=accent_color,
        fill_opacity=0.12,
    )
    win_inner = Square(
        side_length=cell_size,
        stroke_color=ACCENT_COLOR,
        stroke_width=2.0,
        fill_opacity=0.0,
    )
    window = VGroup(win_outer, win_inner).move_to(left_cell_center(0, 0))

    cursor = Square(
        side_length=cell_size * 1.15,
        stroke_color=ACCENT_COLOR,
        stroke_width=3.0,
        fill_opacity=0.0,
    ).move_to(right_cell_center(0, 0))

    # Coluna central (x ∈ [-1.95, +1.95])
    k_title = Text("Kernel K (3×3)", font_size=SMALL_SIZE + 1, color=accent_color, weight="BOLD")
    k_math = MathTex(kernel_tex, font_size=24, color=TEXT_COLOR)
    if k_math.width > 3.45:
        k_math.scale_to_fit_width(3.45)
    k_content = VGroup(k_title, k_math).arrange(DOWN, buff=0.14)
    k_bg = RoundedRectangle(
        corner_radius=0.15,
        width=3.9,
        height=max(k_content.height + 0.36, 1.68),
        fill_color=CARD_BG,
        fill_opacity=0.92,
        stroke_color=accent_color,
        stroke_width=2,
    )
    k_content.move_to(k_bg.get_center())
    kernel_card = VGroup(k_bg, k_content).move_to([0.0, 0.98, 0.0])

    # Cartão ao vivo do produto escalar
    live_bg = RoundedRectangle(
        corner_radius=0.15,
        width=3.9,
        height=2.25,
        fill_color=CARD_BG,
        fill_opacity=0.92,
        stroke_color=BORDER_MUTED,
        stroke_width=2,
    ).move_to([0.0, -1.15, 0.0])
    live_title = Text("Passo a Passo: k · p", font_size=SMALL_SIZE, color=ACCENT_COLOR, weight="BOLD")
    live_title.move_to([0.0, -0.25, 0.0])

    patch_size = 0.88
    patch_center = np.array([-0.92, -0.96, 0.0])
    patch_img = ImageMobject(_to_rgba(_get_padded_patch(input_rgb, 0, 0)))
    patch_img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    patch_img.scale_to_fit_width(patch_size)
    patch_img.move_to(patch_center)
    patch_grid = _build_grid_overlay(3, 3, patch_size, patch_center, opacity=0.45)
    patch_border = Square(
        side_length=patch_size,
        stroke_color=PRIMARY_COLOR,
        stroke_width=1.8,
        fill_opacity=0.0,
    ).move_to(patch_center)

    op_dot_k = MathTex(r"\cdot K =", font_size=24, color=TEXT_COLOR).move_to([0.12, -0.96, 0.0])

    out_swatch = Square(
        side_length=0.56,
        fill_color=rgb_to_color(output_rgb[0, 0] / 255.0),
        fill_opacity=1.0,
        stroke_color=TEXT_COLOR,
        stroke_width=2.0,
    ).move_to([1.05, -0.96, 0.0])

    readout_pos = np.array([0.0, -1.66, 0.0])
    readout_slot = VGroup(
        Text("Pixel (0, 0) — 0 / 400", font_size=SMALL_SIZE - 1, color=TEXT_COLOR).move_to(readout_pos)
    )
    row_labels = VGroup(*[
        Text(f"Varrendo linha {r + 1:02d}/20...", font_size=SMALL_SIZE - 1, color=MUTED_COLOR)
        .move_to(readout_pos)
        .set_opacity(0.0)
        for r in range(20)
    ])

    bar_w = 3.2
    bar_track = RoundedRectangle(
        corner_radius=0.05,
        width=bar_w,
        height=0.12,
        fill_color="#0F172A",
        fill_opacity=1.0,
        stroke_color=BORDER_MUTED,
        stroke_width=1.0,
    ).move_to([0.0, -2.02, 0.0])
    bar_fill = Rectangle(
        width=0.02,
        height=0.10,
        fill_color=accent_color,
        fill_opacity=1.0,
        stroke_width=0.0,
    )
    bar_fill.move_to(bar_track.get_center()).align_to(bar_track, LEFT)

    # Setas curtas ligando Entrada -> Centro -> Resultado
    arrow_in = Arrow([-2.40, -0.25, 0], [-1.98, -0.25, 0], buff=0, color=PRIMARY_COLOR, stroke_width=3, max_tip_length_to_length_ratio=0.35)
    arrow_out = Arrow([1.98, -0.25, 0], [2.40, -0.25, 0], buff=0, color=accent_color, stroke_width=3, max_tip_length_to_length_ratio=0.35)

    # Rodapé explicativo
    footer_lbl = Text(footer_text, font_size=SMALL_SIZE + 1, color=TEXT_COLOR)
    if footer_lbl.width > 12.0:
        footer_lbl.scale_to_fit_width(12.0)
    footer_bg = RoundedRectangle(
        corner_radius=0.14,
        width=12.6,
        height=0.68,
        fill_color=CARD_BG,
        fill_opacity=0.9,
        stroke_color=accent_color,
        stroke_width=1.8,
    )
    footer_group = VGroup(footer_bg, footer_lbl).move_to([0.0, -3.22, 0.0])

    def set_readout(msg: str, color: str = TEXT_COLOR) -> None:
        new_mob = Text(msg, font_size=SMALL_SIZE - 1, color=color)
        if new_mob.width > 3.55:
            new_mob.scale_to_fit_width(3.55)
        new_mob.move_to(readout_pos)
        readout_slot.become(VGroup(new_mob))

    def show_row_label(row_idx: int | None) -> None:
        for idx, lbl in enumerate(row_labels):
            lbl.set_opacity(1.0 if row_idx == idx else 0.0)

    def format_pixel_msg(r: int, c: int) -> str:
        val = output_rgb[r, c]
        if is_grayscale:
            return f"({r:2d},{c:2d}) → I = {int(val[0])}"
        return f"({r:2d},{c:2d}) → RGB({int(val[0])},{int(val[1])},{int(val[2])})"

    def apply_step(step_count: int) -> tuple[int, int]:
        step = max(0, min(400, int(step_count)))
        new_rgba = np.empty((400, 4), dtype=np.uint8)
        new_rgba[:] = EMPTY_SLATE_RGBA
        if step > 0:
            new_rgba[:step] = output_flat[:step]
        right_img.pixel_array[:] = new_rgba.reshape(20, 20, 4)

        active = min(max(step - 1, 0), 399)
        r, c = divmod(active, 20)
        window.move_to(left_cell_center(r, c))
        cursor.move_to(right_cell_center(r, c))

        patch_img.pixel_array[:] = _to_rgba(_get_padded_patch(input_rgb, r, c))
        out_swatch.set_fill(rgb_to_color(output_rgb[r, c] / 255.0), opacity=1.0)

        frac = max(0.008, step / 400.0)
        bar_fill.stretch_to_fit_width(bar_w * frac)
        bar_fill.align_to(bar_track, LEFT)
        return r, c

    # Âncora invisível inserida antes dos elementos dinâmicos para evitar cache estático sobre o readout/cursor
    dummy_anchor = Dot(radius=0.0, fill_opacity=0.0, stroke_opacity=0.0)
    scene.add(dummy_anchor)

    # Entrada inicial dos painéis
    scene.play(
        FadeIn(left_img),
        FadeIn(left_grid, left_border, left_title),
        FadeIn(right_img),
        FadeIn(right_grid, right_border, right_title),
        FadeIn(kernel_card),
        FadeIn(live_bg, live_title, patch_img, patch_grid, patch_border, op_dot_k, out_swatch, readout_slot, row_labels, bar_track, bar_fill),
        GrowArrow(arrow_in),
        GrowArrow(arrow_out),
        FadeIn(footer_group, shift=UP * 0.12),
        run_time=0.85,
    )

    # No tópico de contornos, mostra o papagaio colorido por 0.5s antes da conversão para cinza
    if initial_rgb is not None:
        scene.wait(0.5)
        gray_img, _, _, gray_title, _ = _build_image_panel(
            _to_rgba(input_rgb),
            left_label,
            PRIMARY_COLOR,
            PRIMARY_COLOR,
            left_center,
            panel_size=panel_size,
        )
        scene.play(
            FadeOut(left_img),
            FadeIn(gray_img),
            FadeOut(left_title),
            FadeIn(gray_title),
            run_time=0.6,
        )
        left_img = gray_img
        left_title = gray_title

    scene.play(FadeIn(window), FadeIn(cursor), run_time=0.35)

    # Fase 1: Varredura contínua até o início da faixa do olho (passo 1 -> 89, (4, 8))
    readout_slot.set_opacity(0.0)
    last_row_p1 = -1

    def update_phase1(_mob, alpha: float) -> None:
        nonlocal last_row_p1
        step = max(1, int(round(1 + alpha * 88)))
        r, _ = apply_step(step)
        if r != last_row_p1:
            last_row_p1 = r
            show_row_label(r)

    scene.play(UpdateFromAlphaFunc(dummy_anchor, update_phase1), run_time=1.8, rate_func=linear)
    show_row_label(None)

    # Fase 2: Inspeção lenta pixel a pixel sobre a borda da cabeça e olho: (4, 9) .. (4, 13)
    for col in range(9, 14):
        step = 4 * 20 + col + 1
        target_l = left_cell_center(4, col)
        target_r = right_cell_center(4, col)
        scene.play(
            window.animate.move_to(target_l),
            cursor.animate.move_to(target_r),
            run_time=0.32,
        )
        apply_step(step)
        set_readout(format_pixel_msg(4, col), ACCENT_COLOR)
        scene.play(Indicate(out_swatch, scale_factor=1.16, color=ACCENT_COLOR), run_time=0.28)
        scene.wait(0.22)

    # Fase 3: Varredura contínua até completar os 400 pixels
    readout_slot.set_opacity(0.0)
    last_row_p3 = -1

    def update_phase3(_mob, alpha: float) -> None:
        nonlocal last_row_p3
        step = min(400, int(round(95 + alpha * (400 - 95))))
        r, _ = apply_step(step)
        if r != last_row_p3:
            last_row_p3 = r
            show_row_label(r)

    scene.play(UpdateFromAlphaFunc(dummy_anchor, update_phase3), run_time=5.4, rate_func=linear)
    show_row_label(None)
    apply_step(400)
    set_readout("Concluído: 400 / 400 pixels", SUCCESS_COLOR)

    # Fase 4: Comparação lado a lado limpa
    scene.play(FadeOut(window), FadeOut(cursor), run_time=0.35)
    scene.wait(1.8)

    scene.remove(dummy_anchor)
    transition_stage(
        scene,
        [
            header,
            left_img,
            left_grid,
            left_border,
            left_title,
            right_img,
            right_grid,
            right_border,
            right_title,
            kernel_card,
            live_bg,
            live_title,
            patch_img,
            patch_grid,
            patch_border,
            op_dot_k,
            out_swatch,
            readout_slot,
            row_labels,
            bar_track,
            bar_fill,
            arrow_in,
            arrow_out,
            footer_group,
        ],
    )


def _slide_filters_comparison(
    scene: Scene,
    parrot_rgb: np.ndarray,
    blurred_rgb: np.ndarray,
    sharpened_rgb: np.ndarray,
    edge_rgb: np.ndarray,
) -> None:
    """Slide 5.5: Comparação lado a lado dos 3 filtros gerados pelo mesmo produto escalar."""
    header = create_slide_header(
        "TÓPICO 05",
        "Resumo: 3 Filtros com o Mesmo Produto Escalar",
        badge_color=VIOLET_COLOR,
    )
    scene.play(FadeIn(header, shift=DOWN * 0.15), run_time=0.45)

    specs = [
        (parrot_rgb, "Original", "Imagem Base (20×20)", PRIMARY_COLOR, -4.65),
        (blurred_rgb, "Desfoque", "Gaussiano (∑K = 1)", SECONDARY_COLOR, -1.55),
        (sharpened_rgb, "Nitidez", "Sharpen (∑K = 1)", SUCCESS_COLOR, 1.55),
        (edge_rgb, "Contornos", "Laplaciano (∑K = 0)", ACCENT_COLOR, 4.65),
    ]

    cards = Group()
    stage_items = [header]
    for rgb, title, sub, col, x_pos in specs:
        img_m, grid_m, border_m, title_m, _ = _build_image_panel(
            _to_rgba(rgb),
            title,
            col,
            col,
            [x_pos, 0.3, 0.0],
            panel_size=2.65,
        )
        sub_m = Text(sub, font_size=SMALL_SIZE - 1, color=MUTED_COLOR).next_to(border_m, DOWN, buff=0.16)
        cards.add(Group(img_m, grid_m, border_m, title_m, sub_m))
        stage_items.extend([img_m, grid_m, border_m, title_m, sub_m])

    summary_eq = MathTex(
        r"O_{i,j} = \langle \mathbf{k}, \mathbf{p} \rangle",
        font_size=28,
        color=ACCENT_COLOR,
    )
    summary_txt = Text(
        "A mesma operação linear gera efeitos distintos alterando apenas os 9 pesos do kernel K.",
        font_size=SMALL_SIZE + 1,
        color=TEXT_COLOR,
    )
    summary_content = VGroup(summary_eq, summary_txt).arrange(DOWN, buff=0.14)
    summary_bg = RoundedRectangle(
        corner_radius=0.16,
        width=12.2,
        height=summary_content.height + 0.4,
        fill_color=CARD_BG,
        fill_opacity=0.92,
        stroke_color=VIOLET_COLOR,
        stroke_width=2,
    )
    summary_banner = VGroup(summary_bg, summary_content).move_to([0.0, -2.55, 0.0])
    stage_items.append(summary_banner)

    for card in cards:
        scene.play(FadeIn(card, shift=UP * 0.14), run_time=0.4)
    scene.play(FadeIn(summary_banner, shift=UP * 0.15), run_time=0.55)
    scene.wait(2.4)

    transition_stage(scene, stage_items)


def build_convolution_dot_product(scene: Scene) -> None:
    """Executa todos os slides do Tópico 5 na cena informada."""
    apply_theme(scene)
    play_section_intro(
        scene,
        "TÓPICO 05",
        "Convolução e Produto Escalar",
        "Desfoque, Nitidez e Detecção de Contornos Passo a Passo",
        accent_color=VIOLET_COLOR,
    )

    parrot_rgb = _load_parrot_rgb()
    blurred_rgb = _convolve_rgb(parrot_rgb, K_BLUR)
    sharpened_rgb = _convolve_rgb(blurred_rgb, K_SHARP)
    gray_rgb = _to_grayscale_rgb(parrot_rgb)
    edge_rgb = _convolve_rgb(gray_rgb, K_EDGE)

    _slide_convolution_dot_product_concept(scene)

    # Slide 5.2: Desfoque (Blur)
    _slide_convolution_walkthrough(
        scene,
        slide_title="Desfoque (Blur) Passo a Passo — Kernel Gaussiano",
        left_label="Entrada: Original (20×20)",
        right_label="Resultado: Desfoque",
        input_rgb=parrot_rgb,
        output_rgb=blurred_rgb,
        kernel_tex=r"\frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}",
        accent_color=SECONDARY_COLOR,
        footer_text="Média ponderada (∑K = 1): suaviza transições bruscas combinando cada pixel com seus 8 vizinhos.",
    )

    # Slide 5.3: Nitidez (Sharpen) aplicada sobre a imagem desfocada
    _slide_convolution_walkthrough(
        scene,
        slide_title="Nitidez (Sharpen) Passo a Passo — Realce de Detalhes",
        left_label="Entrada: Desfocada (20×20)",
        right_label="Resultado: Nitidez",
        input_rgb=blurred_rgb,
        output_rgb=sharpened_rgb,
        kernel_tex=r"\begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}",
        accent_color=SUCCESS_COLOR,
        footer_text="Centro positivo (5) menos vizinhos (∑K = 1): recupera bordas e amplifica diferenças locais.",
    )

    # Slide 5.4: Detecção de Contornos (Edge Detection)
    _slide_convolution_walkthrough(
        scene,
        slide_title="Detecção de Contornos Passo a Passo — Laplaciano",
        left_label="Entrada: Escala de Cinza",
        right_label="Resultado: Contornos",
        input_rgb=gray_rgb,
        output_rgb=edge_rgb,
        kernel_tex=r"\begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}",
        accent_color=ACCENT_COLOR,
        footer_text="Kernel de soma zero (∑K = 0): zera regiões uniformes e acende onde há salto de intensidade.",
        is_grayscale=True,
        initial_rgb=parrot_rgb,
    )

    # Slide 5.5: Comparação dos 3 filtros lado a lado
    _slide_filters_comparison(scene, parrot_rgb, blurred_rgb, sharpened_rgb, edge_rgb)


class ConvolutionDotProductScene(Scene):
    """Cena independente para o Tópico 5."""

    def construct(self) -> None:
        build_convolution_dot_product(self)


if __name__ == "__main__":
    pass
