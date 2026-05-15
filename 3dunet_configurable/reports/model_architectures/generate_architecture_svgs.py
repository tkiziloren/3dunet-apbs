#!/usr/bin/env python3
"""Generate thesis-friendly SVG architecture diagrams for the 3D models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent
CONTENT_SHIFT_Y = 58


TEXT_TR = {
    "Input": "Girdi",
    "Output": "Çıktı",
    "Enc1": "Kod1",
    "Enc2": "Kod2",
    "Enc3": "Kod3",
    "Enc4": "Kod4",
    "Bottle": "Darboğaz",
    "Dec1": "Çöz1",
    "Dec2": "Çöz2",
    "Dec3": "Çöz3",
    "Dec4": "Çöz4",
    "Stage2": "Aşama2",
    "Stage4": "Aşama4",
    "Stage5": "Aşama5",
    "Stage6": "Aşama6",
    "Stage7": "Aşama7",
    "Up8": "Yükselt8",
    "Up9": "Yükselt9",
    "Up10": "Yükselt10",
    "Up11": "Yükselt11",
    "Feature grid": "Öznitelik gridi",
    "DoubleConv3D": "Çift Conv3D",
    "DoubleConv3D + GN": "Çift Conv3D + GN",
    "DoubleConv + dropout": "Çift Conv + dropout",
    "Bottleneck": "Darboğaz",
    "Residual block": "Artık blok",
    "Residual bottleneck": "Artık darboğaz",
    "Up + concat + block": "Üst + birl. + blok",
    "Up + concat + DoubleConv": "Üst + birl. + Çift Conv",
    "Up + concat + residual": "Üst + birl. + artık",
    "Attention gate + concat": "Dikkat + birl.",
    "Residual + CBAM": "Artık blok + CBAM",
    "Residual + CBAM + dropout": "Artık blok + CBAM + dropout",
    "Up + concat + CBAM": "Üst + birl. + CBAM",
    "Residual + GroupNorm": "Artık blok + GroupNorm",
    "Residual + GN + dropout": "Artık blok + GN + dropout",
    "Up + concat + residual GN": "Üst + birl. + GN",
    "Depthwise ConvNeXt": "Depthwise ConvNeXt",
    "ConvNeXt + dropout": "ConvNeXt + dropout",
    "Reduce + ConvNeXt": "Azaltma + ConvNeXt",
    "2x Conv3D + ReLU": "2x Conv3D + ReLU",
    "Deep 3D U-Net core": "Derin 3D U-Net çekirdeği",
    "Up + concat + conv": "Üst + birl. + conv",
    "Dense residual stage": "Yoğun artık aşama",
    "Dense V2 bottleneck": "Yoğun V2 darboğaz",
    "Dense residual decoder": "Yoğun artık çözücü",
    "Conv block": "Conv bloğu",
    "Nested skip": "İç içe atlama",
    "Final dense skip": "Son yoğun atlama",
    "nested path": "iç içe yol",
    "atom features": "atom öznitelikleri",
    "3x bottleneck": "3x darboğaz",
    "stride 2": "adım 2",
    "stride 3": "adım 3",
    "global bottleneck": "global darboğaz",
    "up bottleneck": "üst darboğaz",
    "concat + up": "birl. + üst",
    "Dense residual": "Yoğun artık blok",
    "Transformer bottleneck": "Transformer darboğazı",
    "Dense decoder": "Yoğun çözücü",
    "token attention": "token dikkati",
    "input": "girdi",
    "pool/down": "havuzlama",
    "pool": "havuzlama",
    "up": "üst örnekleme",
    "skip": "atlama",
    "nested skip": "iç içe atlama",
    "dense concat": "yoğun birleştirme",
    "Baseline 4-level 3D U-Net encoder-decoder for voxel-wise pocket segmentation": "Voksel düzeyinde cep segmentasyonu için temel 4 seviyeli 3B U-Net kodlayıcı-çözücü",
    "Attention-gated 4-level 3D U-Net used to filter skip features before decoding": "Çözümleme öncesinde atlama özniteliklerini süzen dikkat kapılı 4 seviyeli 3B U-Net",
    "Residual 3D encoder-decoder with BasicBlock3D modules and U-Net-like skips": "BasicBlock3D modülleri ve U-Net benzeri atlama bağlantıları kullanan artık 3B kodlayıcı-çözücü",
    "Nested 3D U-Net with dense skip pathways that reduce encoder-decoder semantic gap": "Kodlayıcı-çözücü anlam farkını azaltan yoğun atlama yollarına sahip iç içe 3B U-Net",
    "Residual 3D U-Net with CBAM channel and spatial attention inside each block": "Her blokta CBAM kanal ve uzamsal dikkat kullanan artık 3B U-Net",
    "ResNet3D4L-style architecture with GroupNorm blocks for small-batch stability": "Küçük batch koşullarında kararlılık için GroupNorm kullanan ResNet3D4L tarzı mimari",
    "Lightweight ConvNeXt-style 3D U-Net with depthwise convolution blocks": "Depthwise konvolüsyon blokları kullanan hafif ConvNeXt tarzı 3B U-Net",
    "Kalasanty-style 3D U-Net with pooling schedule 2, 2, 3, 3": "2, 2, 3, 3 havuzlama düzenine sahip Kalasanty tarzı 3B U-Net",
    "Dense PyTorch approximation of PUResNet v1 residual encoder-decoder topology": "PUResNet v1 artık kodlayıcı-çözücü topolojisinin dense PyTorch yaklaşımı",
    "Dense approximation of PUResNetV2 sparse residual encoder-decoder": "PUResNetV2 sparse artık kodlayıcı-çözücüsünün dense yaklaşımı",
    "Hybrid CNN encoder-decoder with transformer-style bottleneck for long-range context": "Uzak bağlamı yakalamak için transformer tarzı darboğaz kullanan hibrit CNN kodlayıcı-çözücü",
}


@dataclass
class Node:
    name: str
    x: int
    y: int
    channels: str
    shape: str
    note: str
    fill: str = "#e8f2ff"
    stroke: str = "#356fa8"
    width: int = 122
    height: int = 74


@dataclass
class Diagram:
    filename: str
    title: str
    subtitle: str
    nodes: list[Node]
    arrows: list[tuple[str, str, str, str]] = field(default_factory=list)
    skips: list[tuple[str, str, str]] = field(default_factory=list)
    extra_notes: list[str] = field(default_factory=list)
    width: int = 1320
    height: int = 720


def level_shape(input_size: int, level: int) -> str:
    size = input_size
    for _ in range(level):
        size = max(1, size // 2)
    return f"{size}^3"


def tr(text: str) -> str:
    return TEXT_TR.get(text, text)


def make_unet_nodes(
    *,
    in_channels: str = "C",
    out_channels: str = "1",
    base: int = 8,
    input_size: int = 36,
    block_note: str = "DoubleConv3D",
    bottleneck_note: str = "Bottleneck",
    decoder_note: str = "Up + concat + block",
    attention_note: str | None = None,
    node_fill: str = "#e8f2ff",
    node_stroke: str = "#356fa8",
) -> list[Node]:
    x_enc = [90, 245, 400, 555, 710]
    y_enc = [130, 230, 330, 430, 530]
    x_dec = [865, 1010, 1155, 1155]
    y_dec = [430, 330, 230, 130]
    nodes = [
        Node("Input", 30, 34, in_channels, f"{input_size}^3", "Feature grid", "#f3f4f6", "#6b7280", 116, 74),
        Node("Enc1", x_enc[0], y_enc[0], f"{base}", level_shape(input_size, 0), block_note, node_fill, node_stroke),
        Node("Enc2", x_enc[1], y_enc[1], f"{base*2}", level_shape(input_size, 1), block_note, node_fill, node_stroke),
        Node("Enc3", x_enc[2], y_enc[2], f"{base*4}", level_shape(input_size, 2), block_note, node_fill, node_stroke),
        Node("Enc4", x_enc[3], y_enc[3], f"{base*8}", level_shape(input_size, 3), block_note, node_fill, node_stroke),
        Node("Bottle", x_enc[4], y_enc[4], f"{base*16}", level_shape(input_size, 4), bottleneck_note, "#fef3c7", "#b45309"),
        Node("Dec4", x_dec[0], y_dec[0], f"{base*8}", level_shape(input_size, 3), decoder_note, "#e8fff3", "#18875f"),
        Node("Dec3", x_dec[1], y_dec[1], f"{base*4}", level_shape(input_size, 2), decoder_note, "#e8fff3", "#18875f"),
        Node("Dec2", x_dec[2], y_dec[2], f"{base*2}", level_shape(input_size, 1), decoder_note, "#e8fff3", "#18875f"),
        Node("Dec1", x_dec[3], y_dec[3], f"{base}", level_shape(input_size, 0), decoder_note, "#e8fff3", "#18875f"),
        Node("Output", 1196, 34, out_channels, f"{input_size}^3", "Conv3D 1x1", "#f3f4f6", "#6b7280", 116, 74),
    ]
    if attention_note:
        for node in nodes:
            if node.name.startswith("Dec"):
                node.note = attention_note
    return nodes


def node_map(nodes: list[Node]) -> dict[str, Node]:
    return {node.name: node for node in nodes}


def default_unet_arrows() -> list[tuple[str, str, str, str]]:
    return [
        ("Input", "Enc1", "arrow", "input"),
        ("Enc1", "Enc2", "down", "pool/down"),
        ("Enc2", "Enc3", "down", "pool/down"),
        ("Enc3", "Enc4", "down", "pool/down"),
        ("Enc4", "Bottle", "down", "pool/down"),
        ("Bottle", "Dec4", "up", "up"),
        ("Dec4", "Dec3", "up", "up"),
        ("Dec3", "Dec2", "up", "up"),
        ("Dec2", "Dec1", "up", "up"),
        ("Dec1", "Output", "arrow", "1x1"),
    ]


def default_skips() -> list[tuple[str, str, str]]:
    return [
        ("Enc4", "Dec4", "skip"),
        ("Enc3", "Dec3", "skip"),
        ("Enc2", "Dec2", "skip"),
        ("Enc1", "Dec1", "skip"),
    ]


def svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#1d4ed8"/>
    </marker>
    <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#b91c1c"/>
    </marker>
    <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#15803d"/>
    </marker>
    <marker id="arrow-gray" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#6b7280"/>
    </marker>
    <style>
      .title {{ font: 700 28px Arial, sans-serif; fill: #111827; }}
      .subtitle {{ font: 16px Arial, sans-serif; fill: #4b5563; }}
      .section {{ font: 700 18px Arial, sans-serif; fill: #374151; }}
      .node-title {{ font: 700 15px Arial, sans-serif; fill: #111827; }}
      .node-meta {{ font: 12px Arial, sans-serif; fill: #374151; }}
      .node-note {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
      .legend {{ font: 13px Arial, sans-serif; fill: #374151; }}
      .note {{ font: 13px Arial, sans-serif; fill: #374151; }}
    </style>
  </defs>
"""


def center(node: Node) -> tuple[int, int]:
    return node.x + node.width // 2, node.y + node.height // 2


def edge_point(src: Node, dst: Node) -> tuple[int, int, int, int]:
    sx, sy = center(src)
    dx, dy = center(dst)
    if abs(dx - sx) >= abs(dy - sy):
        if dx >= sx:
            x1 = src.x + src.width
            x2 = dst.x
        else:
            x1 = src.x
            x2 = dst.x + dst.width
        return x1, sy, x2, dy
    if dy >= sy:
        y1 = src.y + src.height
        y2 = dst.y
    else:
        y1 = src.y
        y2 = dst.y + dst.height
    return sx, y1, dx, y2


def draw_node(node: Node) -> str:
    lines = [
        f'  <rect x="{node.x}" y="{node.y}" width="{node.width}" height="{node.height}" rx="8" fill="{node.fill}" stroke="{node.stroke}" stroke-width="2"/>',
        f'  <text x="{node.x + node.width/2}" y="{node.y + 20}" text-anchor="middle" class="node-title">{escape(tr(node.name))}</text>',
        f'  <text x="{node.x + node.width/2}" y="{node.y + 39}" text-anchor="middle" class="node-meta">kanal: {escape(node.channels)}</text>',
        f'  <text x="{node.x + node.width/2}" y="{node.y + 56}" text-anchor="middle" class="node-meta">grid: {escape(node.shape)}</text>',
        f'  <text x="{node.x + node.width/2}" y="{node.y + 70}" text-anchor="middle" class="node-note">{escape(tr(node.note))}</text>',
    ]
    return "\n".join(lines)


def draw_arrow(src: Node, dst: Node, kind: str, label: str) -> str:
    color = {"down": "#b91c1c", "up": "#15803d", "arrow": "#1d4ed8", "gray": "#6b7280"}.get(kind, "#1d4ed8")
    marker = {"down": "arrow-red", "up": "arrow-green", "arrow": "arrow-blue", "gray": "arrow-gray"}.get(kind, "arrow-blue")
    x1, y1, x2, y2 = edge_point(src, dst)
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    return (
        f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" marker-end="url(#{marker})"/>\n'
        f'  <text x="{mx}" y="{my - 8}" text-anchor="middle" class="legend">{escape(tr(label))}</text>'
    )


def draw_skip(src: Node, dst: Node, label: str, index: int) -> str:
    sx, sy = center(src)
    dx, dy = center(dst)
    ctrl_y = max(170, min(sy, dy) - 44)
    path = f"M {sx} {sy - src.height//2} C {sx + 90} {ctrl_y}, {dx - 90} {ctrl_y}, {dx} {dy - dst.height//2}"
    return (
        f'  <path d="{path}" fill="none" stroke="#9ca3af" stroke-width="3" stroke-dasharray="8 7" marker-end="url(#arrow-gray)"/>\n'
        f'  <text x="{(sx+dx)/2}" y="{ctrl_y - 8}" text-anchor="middle" class="legend">{escape(tr(label))}</text>'
    )


def draw_legend(x: int, y: int) -> str:
    return f"""
  <g transform="translate({x},{y})">
    <text x="0" y="0" class="section">Açıklama</text>
    <line x1="0" y1="24" x2="42" y2="24" stroke="#1d4ed8" stroke-width="3" marker-end="url(#arrow-blue)"/>
    <text x="54" y="29" class="legend">Konvolüsyon/blok akışı</text>
    <line x1="0" y1="50" x2="42" y2="50" stroke="#b91c1c" stroke-width="3" marker-end="url(#arrow-red)"/>
    <text x="54" y="55" class="legend">Alt örnekleme</text>
    <line x1="0" y1="76" x2="42" y2="76" stroke="#15803d" stroke-width="3" marker-end="url(#arrow-green)"/>
    <text x="54" y="81" class="legend">Üst örnekleme</text>
    <line x1="0" y1="102" x2="42" y2="102" stroke="#9ca3af" stroke-width="3" stroke-dasharray="8 7" marker-end="url(#arrow-gray)"/>
    <text x="54" y="107" class="legend">Atlama bağlantısı</text>
  </g>
"""


def render_diagram(diagram: Diagram) -> str:
    shifted_nodes = [replace(node, y=node.y + CONTENT_SHIFT_Y) for node in diagram.nodes]
    nodes = node_map(shifted_nodes)
    parts = [svg_header(diagram.width, diagram.height)]
    parts.append(f'  <rect width="{diagram.width}" height="{diagram.height}" fill="#ffffff"/>')
    parts.append(f'  <text x="36" y="44" class="title">{escape(diagram.title)}</text>')
    parts.append(f'  <text x="36" y="72" class="subtitle">{escape(tr(diagram.subtitle))}</text>')
    parts.append('  <line x1="650" y1="118" x2="650" y2="690" stroke="#60a5fa" stroke-width="2" stroke-dasharray="6 6"/>')
    parts.append('  <text x="310" y="132" text-anchor="middle" class="section">Kodlayıcı</text>')
    parts.append('  <text x="968" y="132" text-anchor="middle" class="section">Çözücü / çıktı</text>')
    for src, dst, kind, label in diagram.arrows:
        parts.append(draw_arrow(nodes[src], nodes[dst], kind, label))
    for index, (src, dst, label) in enumerate(diagram.skips):
        parts.append(draw_skip(nodes[src], nodes[dst], label, index))
    for node in shifted_nodes:
        parts.append(draw_node(node))
    parts.append(draw_legend(36, 558))
    parts.append("</svg>\n")
    return "\n".join(parts)


def unet_diagram(filename: str, title: str, subtitle: str, *, block_note: str, bottleneck_note: str, decoder_note: str, notes: list[str]) -> Diagram:
    return Diagram(
        filename=filename,
        title=title,
        subtitle=subtitle,
        nodes=make_unet_nodes(block_note=block_note, bottleneck_note=bottleneck_note, decoder_note=decoder_note),
        arrows=default_unet_arrows(),
        skips=default_skips(),
        extra_notes=notes,
    )


def make_unetplusplus() -> Diagram:
    nodes = [
        Node("Input", 35, 45, "C", "36^3", "Feature grid", "#f3f4f6", "#6b7280", 110, 74),
        Node("X0,0", 150, 145, "8", "36^3", "Conv block"),
        Node("X1,0", 295, 245, "16", "18^3", "Conv block"),
        Node("X2,0", 440, 345, "32", "9^3", "Conv block"),
        Node("X3,0", 585, 445, "64", "4^3", "Conv block"),
        Node("X4,0", 730, 545, "128", "2^3", "Bottleneck", "#fef3c7", "#b45309"),
        Node("X0,1", 430, 130, "8", "36^3", "Nested skip"),
        Node("X0,2", 650, 130, "8", "36^3", "Nested skip"),
        Node("X0,3", 870, 130, "8", "36^3", "Nested skip"),
        Node("X0,4", 1090, 130, "8", "36^3", "Final dense skip", "#e8fff3", "#18875f"),
        Node("Output", 1220, 45, "1", "36^3", "Conv3D 1x1", "#f3f4f6", "#6b7280", 90, 74),
        Node("X1,*", 610, 250, "16", "18^3", "nested path", "#ecfeff", "#0891b2"),
        Node("X2,*", 760, 350, "32", "9^3", "nested path", "#ecfeff", "#0891b2"),
        Node("X3,1", 910, 450, "64", "4^3", "nested path", "#ecfeff", "#0891b2"),
    ]
    arrows = [
        ("Input", "X0,0", "arrow", "input"),
        ("X0,0", "X1,0", "down", "pool"),
        ("X1,0", "X2,0", "down", "pool"),
        ("X2,0", "X3,0", "down", "pool"),
        ("X3,0", "X4,0", "down", "pool"),
        ("X4,0", "X3,1", "up", "up"),
        ("X3,1", "X2,*", "up", "up"),
        ("X2,*", "X1,*", "up", "up"),
        ("X1,*", "X0,4", "up", "up"),
        ("X0,4", "Output", "arrow", "1x1"),
        ("X0,0", "X0,1", "gray", "dense concat"),
        ("X0,1", "X0,2", "gray", "dense concat"),
        ("X0,2", "X0,3", "gray", "dense concat"),
        ("X0,3", "X0,4", "gray", "dense concat"),
    ]
    return Diagram(
        filename="unetplusplus3d.svg",
        title="UNetPlusPlus3D",
        subtitle="Nested 3D U-Net with dense skip pathways that reduce encoder-decoder semantic gap",
        nodes=nodes,
        arrows=arrows,
        skips=[
            ("X1,0", "X1,*", "nested skip"),
            ("X2,0", "X2,*", "nested skip"),
            ("X3,0", "X3,1", "nested skip"),
        ],
        extra_notes=[
            "Best Work8 family so far: strong selection score and DCC/DVO balance.",
            "More expensive than ResNet3D4L because of nested decoder blocks.",
        ],
    )


def make_puresnet_v1() -> Diagram:
    nodes = [
        Node("Input", 35, 45, "18", "36^3", "atom features", "#f3f4f6", "#6b7280", 110, 74),
        Node("Stage2", 145, 145, "f", "36^3", "3x bottleneck"),
        Node("Stage4", 300, 245, "2f", "18^3", "stride 2"),
        Node("Stage5", 455, 345, "4f", "9^3", "stride 2"),
        Node("Stage6", 610, 445, "8f", "3^3", "stride 3"),
        Node("Stage7", 765, 545, "16f", "1^3", "global bottleneck", "#fef3c7", "#b45309"),
        Node("Up8", 910, 445, "16f", "3^3", "up bottleneck", "#e8fff3", "#18875f"),
        Node("Up9", 1040, 345, "8f", "9^3", "concat + up", "#e8fff3", "#18875f"),
        Node("Up10", 1160, 245, "4f", "18^3", "concat + up", "#e8fff3", "#18875f"),
        Node("Up11", 1160, 145, "2f", "36^3", "concat + up", "#e8fff3", "#18875f"),
        Node("Output", 1210, 45, "1", "36^3", "Conv3D 1x1", "#f3f4f6", "#6b7280", 100, 74),
    ]
    return Diagram(
        filename="puresnetv1like3d.svg",
        title="PUResNetV1Like3D",
        subtitle="Dense PyTorch approximation of PUResNet v1 residual encoder-decoder topology",
        nodes=nodes,
        arrows=[
            ("Input", "Stage2", "arrow", "input"),
            ("Stage2", "Stage4", "down", "stride 2"),
            ("Stage4", "Stage5", "down", "stride 2"),
            ("Stage5", "Stage6", "down", "stride 3"),
            ("Stage6", "Stage7", "down", "stride 3"),
            ("Stage7", "Up8", "up", "up"),
            ("Up8", "Up9", "up", "up"),
            ("Up9", "Up10", "up", "up"),
            ("Up10", "Up11", "up", "up"),
            ("Up11", "Output", "arrow", "1x1"),
        ],
        skips=[
            ("Stage6", "Up8", "skip"),
            ("Stage5", "Up9", "skip"),
            ("Stage4", "Up10", "skip"),
            ("Stage2", "Up11", "skip"),
        ],
        extra_notes=[
            "Uses bottleneck residual blocks: 1x1 -> 3x3 -> 1x1.",
            "Aggressive downsampling: 36 -> 18 -> 9 -> 3 -> 1.",
        ],
    )


def make_swin_like() -> Diagram:
    nodes = make_unet_nodes(
        in_channels="18/C",
        base=16,
        block_note="Dense residual",
        bottleneck_note="Transformer bottleneck",
        decoder_note="Dense decoder",
        node_fill="#f0f9ff",
        node_stroke="#0369a1",
    )
    nodes[5].fill = "#ede9fe"
    nodes[5].stroke = "#7c3aed"
    nodes[5].note = "token attention"
    return Diagram(
        filename="swinsitelike3d.svg",
        title="SwinSiteLike3D",
        subtitle="Hybrid CNN encoder-decoder with transformer-style bottleneck for long-range context",
        nodes=nodes,
        arrows=default_unet_arrows(),
        skips=default_skips(),
        extra_notes=[
            "Inspired by SwinSite; this implementation uses a dense transformer bottleneck.",
            "Useful as an attention/global-context candidate rather than exact SwinSite replication.",
        ],
    )


def make_diagrams() -> list[Diagram]:
    diagrams = [
        unet_diagram(
            "unet3d4l.svg",
            "UNet3D4L",
            "Baseline 4-level 3D U-Net encoder-decoder for voxel-wise pocket segmentation",
            block_note="DoubleConv3D",
            bottleneck_note="DoubleConv + dropout",
            decoder_note="Up + concat + DoubleConv",
            notes=[
                "Reference baseline for 3D volumetric segmentation.",
                "Skip connections preserve spatial detail for small pocket masks.",
            ],
        ),
        unet_diagram(
            "unet3d4la.svg",
            "UNet3D4LA",
            "Attention-gated 4-level 3D U-Net used to filter skip features before decoding",
            block_note="DoubleConv3D + GN",
            bottleneck_note="DoubleConv + dropout",
            decoder_note="Attention gate + concat",
            notes=[
                "Attention gates modulate encoder skip features using decoder context.",
                "Designed to suppress irrelevant protein regions in skip pathways.",
            ],
        ),
        unet_diagram(
            "resnet3d4l.svg",
            "ResNet3D4L",
            "Residual 3D encoder-decoder with BasicBlock3D modules and U-Net-like skips",
            block_note="Residual block",
            bottleneck_note="Residual bottleneck",
            decoder_note="Up + concat + residual",
            notes=[
                "Fastest Work8 family among completed models.",
                "Residual shortcuts help continuous APBS fields train stably.",
            ],
        ),
        make_unetplusplus(),
        unet_diagram(
            "cbamunet3d.svg",
            "CBAMUNet3D",
            "Residual 3D U-Net with CBAM channel and spatial attention inside each block",
            block_note="Residual + CBAM",
            bottleneck_note="Residual + CBAM + dropout",
            decoder_note="Up + concat + CBAM",
            notes=[
                "CBAM applies channel attention and spatial attention.",
                "Heavier than ResNet3D4L but useful for feature-importance behavior.",
            ],
        ),
        unet_diagram(
            "resnet3d4lgn.svg",
            "ResNet3D4LGN",
            "ResNet3D4L-style architecture with GroupNorm blocks for small-batch stability",
            block_note="Residual + GroupNorm",
            bottleneck_note="Residual + GN + dropout",
            decoder_note="Up + concat + residual GN",
            notes=[
                "Same high-level topology as ResNet3D4L.",
                "GroupNorm is less batch-size dependent than BatchNorm.",
            ],
        ),
        unet_diagram(
            "tinyconvnextunet3d.svg",
            "TinyConvNeXtUNet3D",
            "Lightweight ConvNeXt-style 3D U-Net with depthwise convolution blocks",
            block_note="Depthwise ConvNeXt",
            bottleneck_note="ConvNeXt + dropout",
            decoder_note="Reduce + ConvNeXt",
            notes=[
                "Modern convolution candidate for APBS-only representation.",
                "Depthwise blocks reduce parameters but 3D tensors can still be slow.",
            ],
        ),
        unet_diagram(
            "kalasantyunet3d.svg",
            "KalasantyUNet3D",
            "Kalasanty-style 3D U-Net with pooling schedule 2, 2, 3, 3",
            block_note="2x Conv3D + ReLU",
            bottleneck_note="Deep 3D U-Net core",
            decoder_note="Up + concat + conv",
            notes=[
                "Literature-style baseline for comparison with Kalasanty.",
                "Pooling schedule compresses 36^3 input toward compact latent maps.",
            ],
        ),
        make_puresnet_v1(),
        unet_diagram(
            "puresnetv2denselike3d.svg",
            "PUResNetV2DenseLike3D",
            "Dense approximation of PUResNetV2 sparse residual encoder-decoder",
            block_note="Dense residual stage",
            bottleneck_note="Dense V2 bottleneck",
            decoder_note="Dense residual decoder",
            notes=[
                "Approximation only; exact PUResNetV2 is sparse MinkowskiEngine-based.",
                "Included to test literature-inspired residual sparse-style behavior.",
            ],
        ),
        make_swin_like(),
    ]
    return diagrams


def write_readme(diagrams: list[Diagram]) -> None:
    rows = [
        "# Model Mimari Diyagramları",
        "",
        "Bu SVG dosyaları Deep-APBS deneylerinde kullanılan veya denenmesi planlanan ana 3B mimariler için tezde kullanılabilir kavramsal diyagramlardır.",
        "Amaç okunması zor PyTorch hesap grafiği dökmek değil, mimari ailelerinin çalışma mantığını anlaşılır şekilde özetlemektir.",
        "",
        "| Dosya | Model | Tezde kullanım |",
        "|---|---|---|",
    ]
    for diagram in diagrams:
        rows.append(f"| `{diagram.filename}` | {diagram.title} | {tr(diagram.subtitle)} |")
    rows.extend(
        [
            "",
            "Önerilen kullanım:",
            "",
            "- Bölüm 3: `unet3d4l.svg` genel 3B U-Net anlatımı için kullanılabilir.",
            "- Bölüm 4: gerçek deneylerde kullanılan Work8 modelleri için seçili diyagramlar kullanılabilir.",
            "- Literatür karşılaştırması: `kalasantyunet3d.svg`, `puresnetv1like3d.svg` ve `swinsitelike3d.svg` sadece bu modeller nihai yöntem karşılaştırmasında tartışılırsa kullanılmalı.",
            "",
            "Tüm diyagramlarda okunabilirlik için temsili `36^3` giriş gridi gösterilmiştir. Aynı topoloji bellek izin verdiğinde daha büyük gridlerle de kullanılabilir.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_index(diagrams: list[Diagram]) -> None:
    cards = []
    for diagram in diagrams:
        cards.append(
            f"""
      <section class="card">
        <h2>{escape(diagram.title)}</h2>
        <p>{escape(tr(diagram.subtitle))}</p>
        <img src="{escape(diagram.filename)}" alt="{escape(diagram.title)} mimari diyagramı" />
      </section>"""
        )
    html = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deep-APBS Model Mimari Diyagramları</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f7f7f8;
      color: #111827;
    }}
    header {{
      padding: 28px 36px 18px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    header p {{
      margin: 0;
      color: #4b5563;
      max-width: 900px;
      line-height: 1.45;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
      gap: 20px;
      padding: 24px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 20px;
    }}
    .card p {{
      margin: 0 0 14px;
      color: #4b5563;
      line-height: 1.35;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #eef0f4;
      border-radius: 6px;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Deep-APBS Model Mimari Diyagramları</h1>
    <p>Bağlanma bölgesi tahmini deneylerinde kullanılan veya denenmesi planlanan ana 3B mimariler için tezde kullanılabilir kavramsal SVG diyagramları.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    diagrams = make_diagrams()
    for diagram in diagrams:
        (OUT_DIR / diagram.filename).write_text(render_diagram(diagram), encoding="utf-8")
    write_readme(diagrams)
    write_index(diagrams)
    print(f"Wrote {len(diagrams)} SVG diagrams to {OUT_DIR}")


if __name__ == "__main__":
    main()
