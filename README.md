# Mine-RS

![img](./assets/captures/cap.png)

```
Frame: block-atlas (256×256)
Grid: 16×16 px

Col index = BlockType enum value:
  0=Air  1=Grass  2=Dirt  3=Stone  4=Sand  5=Water
  6=Wood 7=Leaves 8=Snow  9=Bedrock 10=Gravel
  11=Coal 12=Iron 13=Gold 14=Diamond 15=Torch

Row 0: top faces    (tile_index 0–15)
Row 1: side faces   (tile_index 16–31)
Row 2: bottom faces (tile_index 32–47)
Row 3, col 0: TallGrass cross (tile_index 48)
```