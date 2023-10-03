## Resize

```
mogrify -resize 50% -format r big/*_big.png
mmv 'big/*_big.r' '#1.png'
```

## Compress

```
pngquant *.png --quality=98 --speed 1
mmv '*-fs8.png' '../#1.png'
```