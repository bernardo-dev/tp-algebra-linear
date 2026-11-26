"""Helpers padronizados para layout de slides, cabeçalhos e transições."""

from __future__ import annotations

from typing import Iterable

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    Mobject,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

from src.config.theme import (
    ACCENT_COLOR,
    BG_COLOR,
    BORDER_MUTED,
    CARD_BG,
    HEADER_TITLE_SIZE,
    MUTED_COLOR,
    PRIMARY_COLOR,
    SECTION_TITLE_SIZE,
    SMALL_SIZE,
    SUBTITLE_SIZE,
    TEXT_COLOR,
)


def apply_theme(scene: Scene) -> None:
    """Aplica a cor de fundo padrão da apresentação à cena."""
    scene.camera.background_color = BG_COLOR


def play_section_intro(
    scene: Scene,
    number: str,
    title: str,
    subtitle: str,
    accent_color: str = PRIMARY_COLOR,
    hold_time: float = 1.6,
) -> None:
    """Exibe um cartão animado de abertura de tópico e limpa o palco ao final."""
    badge_text = Text(number, font_size=SMALL_SIZE + 2, color=BG_COLOR, weight="BOLD")
    badge_box = RoundedRectangle(
        corner_radius=0.14,
        width=badge_text.width + 0.55,
        height=badge_text.height + 0.28,
        fill_color=accent_color,
        fill_opacity=1.0,
        stroke_width=0,
    )
    badge = VGroup(badge_box, badge_text)

    title_mob = Text(title, font_size=SECTION_TITLE_SIZE, color=TEXT_COLOR, weight="BOLD")
    if title_mob.width > 11.8:
        title_mob.scale_to_fit_width(11.8)

    divider = Line(LEFT * 3.5, RIGHT * 3.5, color=accent_color, stroke_width=3)
    subtitle_mob = Text(subtitle, font_size=SUBTITLE_SIZE, color=MUTED_COLOR)
    if subtitle_mob.width > 11.4:
        subtitle_mob.scale_to_fit_width(11.4)

    content = VGroup(badge, title_mob, divider, subtitle_mob).arrange(DOWN, buff=0.35)
    card = RoundedRectangle(
        corner_radius=0.25,
        width=max(content.width + 1.4, 9.0),
        height=content.height + 1.2,
        fill_color=CARD_BG,
        fill_opacity=0.85,
        stroke_color=accent_color,
        stroke_width=2,
    )
    banner = VGroup(card, content).move_to([0, 0, 0])

    scene.play(FadeIn(card, scale=0.95), GrowFromCenter(badge), run_time=0.55)
    scene.play(Write(title_mob), Create(divider), run_time=0.65)
    scene.play(FadeIn(subtitle_mob, shift=UP * 0.15), run_time=0.45)
    scene.wait(hold_time)
    scene.play(FadeOut(banner, shift=UP * 0.2), run_time=0.45)


def create_slide_header(
    badge_text: str,
    title_text: str,
    badge_color: str = PRIMARY_COLOR,
) -> VGroup:
    """Cria o cabeçalho fixo do slide com badge, título e linha divisória."""
    badge_lbl = Text(badge_text, font_size=SMALL_SIZE - 1, color=BG_COLOR, weight="BOLD")
    badge_bg = RoundedRectangle(
        corner_radius=0.1,
        width=badge_lbl.width + 0.36,
        height=badge_lbl.height + 0.2,
        fill_color=badge_color,
        fill_opacity=1.0,
        stroke_width=0,
    )
    badge = VGroup(badge_bg, badge_lbl)

    title = Text(title_text, font_size=HEADER_TITLE_SIZE, color=TEXT_COLOR, weight="BOLD")
    if title.width > 10.0:
        title.scale_to_fit_width(10.0)

    top_row = VGroup(badge, title).arrange(RIGHT, buff=0.3)
    top_row.to_edge(UP, buff=0.25).to_edge(LEFT, buff=0.55)

    divider = Line(LEFT * 6.5, RIGHT * 6.5, color=BORDER_MUTED, stroke_width=2)
    divider.next_to(top_row, DOWN, buff=0.16).set_x(0)

    return VGroup(top_row, divider)


def create_info_card(
    title: str,
    items: list[Mobject],
    width: float = 5.5,
    border_color: str = PRIMARY_COLOR,
    title_color: str = ACCENT_COLOR,
) -> VGroup:
    """Agrupa título e itens em um cartão estilizado de borda arredondada."""
    title_mob = Text(title, font_size=SUBTITLE_SIZE - 2, color=title_color, weight="BOLD")
    max_inner_w = width - 0.65
    if title_mob.width > max_inner_w:
        title_mob.scale_to_fit_width(max_inner_w)

    body_group = VGroup()
    for item in items:
        if item.width > max_inner_w:
            item.scale_to_fit_width(max_inner_w)
        body_group.add(item)
    body_group.arrange(DOWN, aligned_edge=LEFT, buff=0.22)

    content = VGroup(title_mob, body_group).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
    bg = RoundedRectangle(
        corner_radius=0.18,
        width=width,
        height=content.height + 0.56,
        fill_color=CARD_BG,
        fill_opacity=0.88,
        stroke_color=border_color,
        stroke_width=2,
    )
    content.move_to(bg.get_center()).align_to(bg, LEFT).shift(RIGHT * 0.3)
    return VGroup(bg, content)


def transition_stage(
    scene: Scene,
    to_remove: Iterable[Mobject],
    run_time: float = 0.45,
) -> None:
    """Remove os elementos antigos do palco com FadeOut suave."""
    mobjects = [m for m in to_remove if m is not None]
    if mobjects:
        scene.play(*[FadeOut(m) for m in mobjects], run_time=run_time)
