"""Simple offline GIF module.

This module keeps the core idea of gif.js (build GIF animations from frames)
but intentionally strips all website/browser functionality.

No network requests are used; everything is local and pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class IndexedFrame:
    """Single indexed-color frame for GIF encoding.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        pixels: Flat sequence of palette indexes (len == width * height).
        delay_cs: Delay in centiseconds (1/100 sec).
    """

    width: int
    height: int
    pixels: Sequence[int]
    delay_cs: int = 10

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("pixels length must be width * height")
        if self.delay_cs < 0:
            raise ValueError("delay_cs must be >= 0")


class SimpleGIF:
    """Minimal animated GIF writer.

    Notes:
        - Input frames must already be indexed by the same palette.
        - Palette entries are RGB triples in range 0..255.
        - Uses an internal LZW encoder (no external dependencies).
    """

    def __init__(self, width: int, height: int, palette: Sequence[tuple[int, int, int]], loop: int = 0):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if not palette:
            raise ValueError("palette must not be empty")
        if len(palette) > 256:
            raise ValueError("palette cannot exceed 256 colors")

        self.width = width
        self.height = height
        self.palette = [self._validate_color(c) for c in palette]
        self.loop = loop
        self.frames: List[IndexedFrame] = []

    @staticmethod
    def _validate_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(color) != 3:
            raise ValueError("palette color must be RGB tuple")
        r, g, b = color
        if min(r, g, b) < 0 or max(r, g, b) > 255:
            raise ValueError("palette values must be in range 0..255")
        return color

    def add_frame(self, pixels: Sequence[int], delay_cs: int = 10) -> None:
        frame = IndexedFrame(self.width, self.height, pixels, delay_cs=delay_cs)
        for index in frame.pixels:
            if index < 0 or index >= len(self.palette):
                raise ValueError(f"palette index out of range: {index}")
        self.frames.append(frame)

    def save(self, output_path: str | Path) -> Path:
        if not self.frames:
            raise ValueError("at least one frame is required")

        out = Path(output_path)
        data = bytearray()
        data.extend(b"GIF89a")
        data.extend(self._u16(self.width))
        data.extend(self._u16(self.height))

        packed, gct = self._global_color_table()
        data.append(packed)
        data.append(0)  # background color index
        data.append(0)  # pixel aspect ratio
        data.extend(gct)

        # Netscape loop extension
        data.extend(b"!\xFF\x0BNETSCAPE2.0\x03\x01")
        data.extend(self._u16(self.loop if self.loop >= 0 else 0))
        data.append(0)

        for frame in self.frames:
            data.extend(self._graphics_control_extension(frame.delay_cs))
            data.extend(self._image_descriptor())
            data.extend(self._image_data(frame.pixels))

        data.append(0x3B)  # GIF trailer
        out.write_bytes(data)
        return out

    @staticmethod
    def _u16(value: int) -> bytes:
        return bytes((value & 0xFF, (value >> 8) & 0xFF))

    def _global_color_table(self) -> tuple[int, bytes]:
        table_size_pow = 1
        while (1 << table_size_pow) < len(self.palette):
            table_size_pow += 1
        table_len = 1 << table_size_pow

        gct = bytearray()
        for r, g, b in self.palette:
            gct.extend((r, g, b))
        while len(gct) < table_len * 3:
            gct.extend((0, 0, 0))

        size_field = table_size_pow - 1  # 2^(N+1)
        packed = 0b10000000 | 0b01110000 | size_field  # GCT present, 8-bit color resolution
        return packed, bytes(gct)

    @staticmethod
    def _graphics_control_extension(delay_cs: int) -> bytes:
        # No transparency; disposal method 0
        return b"!\xF9\x04\x00" + SimpleGIF._u16(delay_cs) + b"\x00\x00"

    def _image_descriptor(self) -> bytes:
        return b"," + self._u16(0) + self._u16(0) + self._u16(self.width) + self._u16(self.height) + b"\x00"

    def _image_data(self, pixels: Sequence[int]) -> bytes:
        min_code_size = max(2, (len(self.palette) - 1).bit_length())
        compressed = self._lzw_encode(pixels, min_code_size)

        chunks = bytearray()
        chunks.append(min_code_size)
        for i in range(0, len(compressed), 255):
            chunk = compressed[i : i + 255]
            chunks.append(len(chunk))
            chunks.extend(chunk)
        chunks.append(0)  # block terminator
        return bytes(chunks)

    @staticmethod
    def _lzw_encode(pixels: Sequence[int], min_code_size: int) -> bytes:
        clear_code = 1 << min_code_size
        end_code = clear_code + 1

        dictionary: dict[tuple[int, ...], int] = {(i,): i for i in range(clear_code)}
        next_code = end_code + 1
        code_size = min_code_size + 1

        codes: List[int] = [clear_code]
        w: tuple[int, ...] = ()

        for k in pixels:
            wk = w + (k,)
            if wk in dictionary:
                w = wk
            else:
                if w:
                    codes.append(dictionary[w])
                dictionary[wk] = next_code
                next_code += 1
                w = (k,)

                if next_code == (1 << code_size) and code_size < 12:
                    code_size += 1
                elif next_code >= 4096:
                    codes.append(clear_code)
                    dictionary = {(i,): i for i in range(clear_code)}
                    next_code = end_code + 1
                    code_size = min_code_size + 1
                    w = ()

        if w:
            codes.append(dictionary[w])
        codes.append(end_code)

        return SimpleGIF._pack_codes(codes, min_code_size)

    @staticmethod
    def _pack_codes(codes: Iterable[int], min_code_size: int) -> bytes:
        clear_code = 1 << min_code_size
        end_code = clear_code + 1

        code_size = min_code_size + 1
        next_code = end_code + 1

        out = bytearray()
        bit_buffer = 0
        bit_count = 0

        for code in codes:
            bit_buffer |= code << bit_count
            bit_count += code_size

            while bit_count >= 8:
                out.append(bit_buffer & 0xFF)
                bit_buffer >>= 8
                bit_count -= 8

            if code == clear_code:
                code_size = min_code_size + 1
                next_code = end_code + 1
            elif code != end_code:
                next_code += 1
                if next_code == (1 << code_size) and code_size < 12:
                    code_size += 1

        if bit_count:
            out.append(bit_buffer & 0xFF)
        return bytes(out)


def demo_build(output: str | Path = "demo.gif") -> Path:
    """Create a tiny local demo GIF with no external dependencies."""
    palette = [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]
    width, height = 64, 64
    gif = SimpleGIF(width, height, palette, loop=0)

    for frame_i in range(12):
        pixels = []
        for y in range(height):
            for x in range(width):
                if (x - frame_i * 3) % 16 < 8:
                    pixels.append(2 + (frame_i % 3))
                else:
                    pixels.append(1 if (x + y) % 2 == 0 else 0)
        gif.add_frame(pixels, delay_cs=7)

    return gif.save(output)


if __name__ == "__main__":
    path = demo_build()
    print(f"Created: {path}")
