# Sonuç Dosyaları

- `benchmark/leaderboard.csv`: 20 modelin 1.060 klip üzerindeki toplu WER, CER,
  RTF ve AcoSemantic değerleri
- `benchmark/source_breakdown.csv`: Sonuçların Common Voice TR, ISSAI ve OpenSLR
  TR kaynaklarına göre kırılımı
- `benchmark/acosemantic_summary.csv`: ASCS dağılımı, WER korelasyonu ve duygu
  kayması özetleri
- `finetune/`: M0/M1/M2 değerlendirme ve eğitim üst veri özetleri
- `real-speech/`: Bağımsız V2 zor tıbbi terim testi ve bootstrap analizi

`finetune/eval_m*_summary.json` içindeki sentetik tıbbi bölüm eğitim verisiyle
örtüştüğü için bağımsız doğrulama olarak yorumlanmamalıdır. Ayrıntılı açıklama
[`docs/RESULTS.md`](../docs/RESULTS.md) belgesindedir.
