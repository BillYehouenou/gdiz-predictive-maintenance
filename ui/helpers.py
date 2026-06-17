def plot_layout(C: dict, **kw) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C["plot_bg"],
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", color=C["muted"], size=11),
        margin=dict(l=8, r=8, t=36, b=8),
        colorway=[C["blue"], C["orange"], C["green"], C["red"], C["teal"]],
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(size=10, color=C["muted"]),
        ),
    )
    base.update(kw)
    return base


def plot_axis(C: dict, **kw) -> dict:
    base = dict(
        gridcolor=C["grid"],
        linecolor=C["border"],
        tickcolor=C["border"],
        tickfont=dict(size=10, color=C["faint"]),
        title_font=dict(size=11, color=C["muted"]),
        zeroline=False,
    )
    base.update(kw)
    return base


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fmt_fcfa(amount: float) -> str:
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f} M FCFA"
    return f"{amount / 1_000:.0f} k FCFA"


def cost_delta_html(C: dict, delta: float, prev_label: str) -> str:
    if delta > 0:
        col, sym = C["red"], "▲"
    else:
        col, sym = C["green"], "▼"
    return (
        f"<div style='font-size:.72rem;color:{col};margin-top:-.15rem;line-height:0.9;'>"
        f"{sym} {fmt_fcfa(abs(delta))} vs {prev_label}</div>"
    )
