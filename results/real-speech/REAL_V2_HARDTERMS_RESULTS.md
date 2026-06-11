# Gerçek-Ses ZOR-TERİM Testi (200 zor tıbbi terim) — Sonuçlar

*2026-06-10 23:20*

Terimler medv3 sözlüğünden seçilen ZOR (ilaç/Latin) terimler; cümleler eğitimden farklı (sızıntısız). Eşleşme: strict-fair (diyakritik affedilir, fonetik harf hatası affedilmez). Eval: Colab L4.

| Konuşmacı | n | M0 WER | M1 WER | M2 WER | M0 CER | M1 CER | M2 CER | M0 recall | M1 recall | M2 recall |
|---|---|---|---|---|---|---|---|---|---|---|
| k1 (yağmur) | 200 | 0.1070 | 0.1064 | 0.0898 | 0.0204 | 0.0192 | 0.0177 | 0.670 (134) | 0.665 | 0.735 (147) |
| k2 (zehra) | 116 | 0.0691 | 0.0831 | 0.0395 | 0.0130 | 0.0151 | 0.0075 | 0.750 (87) | 0.724 | 0.836 (97) |
| k3 (yusuf) | 200 | 0.0911 | 0.0952 | 0.0730 | 0.0142 | 0.0153 | 0.0131 | 0.675 (135) | 0.690 | 0.750 (150) |

- M2 her konuşmacıda en iyi: zor-terim recall M0→M2 +6.5/+7.5/+8.6 puan; WER belirgin düşüş.
- Base zor terimlerde ~%67-75 (doygunluk YOK) → test ayırt edici. Hizalama doğrulandı.
- Figür: `figures/fig_real_v2_hardterms.png`

## Genel Türkçe (medikalsiz, gerçek klip) — ayrı, temiz eval

| Model | Genel WER | Genel CER | n |
|---|---|---|---|
| M0 base | 0.1213 | 0.0546 | 320 |
| M1 (genel LoRA) | 0.0792 | 0.0226 | 320 |
| M2 (genel+medikal) | 0.0795 | 0.0228 | 320 |

- **M1 genel WER'i %34.7, CER'yi %58.6 göreli olarak düşürdü**; M2 her iki kazanımı da korudu.
- Figür: `figures/fig_general_turkish_clean.png`
