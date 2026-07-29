# 让 latexmk 默认使用 xelatex 引擎（ctex 中文文档必需）
$pdf_mode = 5;          # 5 = xelatex
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
