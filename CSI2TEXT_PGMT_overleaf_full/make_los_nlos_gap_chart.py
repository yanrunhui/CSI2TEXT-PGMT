from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 2200, 1420
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

regular_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
bold_path = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(bold_path if bold else regular_path, size)


def centered_text(x, y, text, fnt, fill="#111111", anchor="ma"):
    d.multiline_text((x, y), text, font=fnt, fill=fill, anchor=anchor,
                     align="center", spacing=8)


axis_f = font(72)
tick_f = font(58)
label_f = font(64, True)
value_f = font(52)
ratio_f = font(66, True)

left, right = 235, 2110
top, bottom = 105, 930
ymax = 26.0


def ycoord(value):
    return bottom - (bottom - top) * value / ymax


for val in range(0, 26, 5):
    y = ycoord(val)
    d.line((left, y, right, y), fill="#d8d8d8", width=3)
    d.text((left - 28, y), str(val), font=tick_f, fill="#333333", anchor="rm")

d.line((left, top, left, bottom), fill="#222222", width=5)
d.line((left, bottom, right, bottom), fill="#222222", width=5)

ylabel = Image.new("RGBA", (760, 90), (255, 255, 255, 0))
yd = ImageDraw.Draw(ylabel)
yd.text((380, 45), "NLoS / LoS MAE ratio", font=axis_f,
        fill="#222222", anchor="mm")
ylabel = ylabel.rotate(90, expand=True)
img.paste(ylabel, (30, int((top + bottom - ylabel.height) / 2)), ylabel)

entries = [
    ("First delay", 4.38, "LoS: 8.174 ± 0.156 ns\nNLoS: 35.820 ± 0.599 ns", "#2878b5"),
    ("First angle", 21.0, "LoS: 1.680 ± 0.043°\nNLoS: 35.272 ± 0.748°", "#d94f45"),
    ("First power", 2.34, "LoS: 2.973 ± 0.638 dB\nNLoS: 6.966 ± 0.122 dB", "#4d9c70"),
]

centers = [560, 1170, 1780]
bar_w = 300
for x, (name, ratio, values, color) in zip(centers, entries):
    y = ycoord(ratio)
    d.rounded_rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom),
                        radius=12, fill=color)
    ratio_text = f"{ratio:.1f}×" if ratio >= 10 else f"{ratio:.2f}×"
    centered_text(x, y - 22, ratio_text, ratio_f, "#a7332d", anchor="ms")
    centered_text(x, bottom + 48, name, label_f, anchor="ma")
    centered_text(x, bottom + 125, values, value_f, "#222222", anchor="ma")

output = Path(__file__).resolve().parent / "figures" / "los_nlos_ratio.png"
img.save(output, dpi=(320, 320))
